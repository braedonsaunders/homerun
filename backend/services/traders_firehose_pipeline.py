"""Strategy-owned traders firehose filtering pipeline.

This module keeps trader firehose normalization, enrichment, filtering,
and opportunity conversion inside DB-loaded traders strategies.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import partial
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    DiscoveredWallet,
    Strategy,
    TrackedWallet,
    TraderGroup,
    TraderGroupMember,
)
from models.opportunity import ArbitrageOpportunity
from services.market_tradability import get_market_tradability_map
from services.opportunity_strategy_catalog import ensure_system_opportunity_strategies_seeded
from services.smart_wallet_pool import _looks_like_crypto_market, smart_wallet_pool
from services.strategy_loader import strategy_loader
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.strategy_sdk import StrategySDK
from utils.converters import normalize_market_id, safe_float
from utils.logger import get_logger
from utils.utcnow import utcnow

_safe_float = partial(safe_float, reject_nan_inf=True)

logger = get_logger("traders_firehose_pipeline")

_STRATEGY_SLUG = "traders_confluence"


def _normalize_wallet_address(value: object) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or not text.startswith("0x"):
        return None
    return text


def _parse_time(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None


def _strip_internal_schema(config: object) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    return {str(key): value for key, value in config.items() if str(key) != "_schema"}


def _apply_strategy_config(instance: Any, config: dict[str, Any]) -> None:
    if hasattr(instance, "configure") and callable(getattr(instance, "configure")):
        instance.configure(config)

    default_config = getattr(instance, "default_config", None)
    current_config = getattr(instance, "config", None)
    if isinstance(default_config, dict):
        merged = dict(default_config)
        merged.update(config)
        setattr(instance, "config", merged)
    elif isinstance(current_config, dict):
        merged = dict(current_config)
        merged.update(config)
        setattr(instance, "config", merged)

    internal_cfg = getattr(instance, "_config", None)
    if isinstance(internal_cfg, dict):
        merged = dict(internal_cfg)
        merged.update(config)
        setattr(instance, "_config", merged)


async def _load_strategy_row(session: AsyncSession) -> Optional[Strategy]:
    await ensure_system_opportunity_strategies_seeded(session)
    return (await session.execute(select(Strategy).where(Strategy.slug == _STRATEGY_SLUG))).scalars().first()


async def _resolve_traders_strategy(session: AsyncSession) -> Optional[Any]:
    row = await _load_strategy_row(session)
    if row is None:
        logger.error("Traders strategy row missing: %s", _STRATEGY_SLUG)
        return None
    if row.enabled is False:
        logger.info("Traders strategy disabled", slug=_STRATEGY_SLUG)
        return None

    await refresh_strategy_runtime_if_needed(session, source_keys=["traders"], force=False)
    loaded = strategy_loader._loaded.get(_STRATEGY_SLUG)
    if loaded is None:
        logger.error("Traders strategy runtime not loaded", slug=_STRATEGY_SLUG)
        return None

    config = StrategySDK.validate_trader_filter_config(_strip_internal_schema(row.config))
    instance = loaded.instance
    _apply_strategy_config(instance, config)
    return instance


async def _annotate_source_flags(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    all_addresses: set[str] = set()
    wallets_per_row: list[list[str]] = []
    for row in rows:
        row_wallets: list[str] = []
        for raw in row.get("wallets") or []:
            normalized = _normalize_wallet_address(raw)
            if not normalized:
                continue
            row_wallets.append(normalized)
            all_addresses.add(normalized)
        wallets_per_row.append(row_wallets)

    pool_addresses: set[str] = set()
    tracked_addresses: set[str] = set()
    group_ids_by_address: dict[str, set[str]] = {}

    if all_addresses:
        addr_list = list(all_addresses)

        discovered_rows = await session.execute(
            select(DiscoveredWallet.address, DiscoveredWallet.in_top_pool).where(
                func.lower(DiscoveredWallet.address).in_(addr_list)
            )
        )
        for address, in_top_pool in discovered_rows.all():
            normalized = _normalize_wallet_address(address)
            if normalized and bool(in_top_pool):
                pool_addresses.add(normalized)

        tracked_rows = await session.execute(
            select(TrackedWallet.address).where(func.lower(TrackedWallet.address).in_(addr_list))
        )
        for (address,) in tracked_rows.all():
            normalized = _normalize_wallet_address(address)
            if normalized:
                tracked_addresses.add(normalized)

        group_rows = await session.execute(
            select(TraderGroupMember.wallet_address, TraderGroupMember.group_id)
            .join(TraderGroup, TraderGroupMember.group_id == TraderGroup.id)
            .where(
                TraderGroup.is_active == True,  # noqa: E712
                func.lower(TraderGroupMember.wallet_address).in_(addr_list),
            )
        )
        for address, group_id in group_rows.all():
            normalized = _normalize_wallet_address(address)
            if not normalized:
                continue
            bucket = group_ids_by_address.setdefault(normalized, set())
            if group_id:
                bucket.add(str(group_id))

    for row, wallets in zip(rows, wallets_per_row):
        pool_wallets = sum(1 for addr in wallets if addr in pool_addresses)
        tracked_wallets = sum(1 for addr in wallets if addr in tracked_addresses)
        group_wallets = sum(1 for addr in wallets if group_ids_by_address.get(addr))
        matched_group_ids = sorted({gid for addr in wallets for gid in group_ids_by_address.get(addr, set())})

        row["source_flags"] = StrategySDK.normalize_trader_source_flags(
            {
                "from_pool": pool_wallets > 0,
                "from_tracked_traders": tracked_wallets > 0,
                "from_trader_groups": group_wallets > 0,
                "qualified": bool(pool_wallets or tracked_wallets or group_wallets),
            }
        )
        row["source_breakdown"] = {
            "wallets_considered": len(wallets),
            "pool_wallets": pool_wallets,
            "tracked_wallets": tracked_wallets,
            "group_wallets": group_wallets,
            "group_count": len(matched_group_ids),
            "group_ids": matched_group_ids,
        }


async def _annotate_firehose_context(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    market_ids = [
        normalize_market_id(row.get("market_id")) for row in rows if normalize_market_id(row.get("market_id"))
    ]
    tradability = await get_market_tradability_map(market_ids) if market_ids else {}
    now = utcnow()

    for row in rows:
        market_id = normalize_market_id(row.get("market_id"))
        detected = _parse_time(row.get("detected_at") or row.get("last_seen_at") or row.get("first_seen_at"))
        age_minutes = 0.0
        if detected is not None:
            age_minutes = max(0.0, (now - detected).total_seconds() / 60.0)

        row["firehose_market_tradable"] = bool(tradability.get(market_id, True))
        row["firehose_is_crypto"] = bool(
            _looks_like_crypto_market(
                row.get("market_question"),
                row.get("market_slug"),
                row.get("market_id"),
            )
        )
        row["firehose_age_minutes"] = age_minutes
        row["firehose_confidence"] = (
            _safe_float(row.get("strength"))
            if _safe_float(row.get("strength")) is not None
            else (
                (_safe_float(row.get("conviction_score")) or 0.0) / 100.0
                if (_safe_float(row.get("conviction_score")) or 0.0) > 1.0
                else (_safe_float(row.get("conviction_score")) or 0.0)
            )
        )
        row.update(StrategySDK.normalize_trader_signal(row))


def _apply_strategy_filter(
    strategy: Any,
    rows: list[dict[str, Any]],
    *,
    include_filtered: bool,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    method = getattr(strategy, "apply_firehose_filters", None)
    if not callable(method):
        logger.error("Traders strategy missing apply_firehose_filters", strategy=getattr(strategy, "strategy_type", ""))
        return []

    try:
        annotated = method(rows, include_filtered=True, limit=None)
    except Exception as exc:
        logger.error("Traders strategy firehose filter failed: %s", exc)
        return []

    if not isinstance(annotated, list):
        logger.error("Traders strategy returned non-list firehose payload")
        return []

    normalized_rows = [StrategySDK.normalize_trader_signal(dict(row)) for row in annotated if isinstance(row, dict)]
    passed_rows = [
        row
        for row in normalized_rows
        if bool(row.get("is_tradeable", row.get("is_actionable", row.get("is_valid", False))))
    ]
    filtered_out_rows = [
        row
        for row in normalized_rows
        if not bool(row.get("is_tradeable", row.get("is_actionable", row.get("is_valid", False))))
    ]

    output_rows = normalized_rows if include_filtered else passed_rows
    if limit and limit > 0:
        output_rows = output_rows[:limit]

    reason_counts: dict[str, int] = {}
    for row in filtered_out_rows:
        for reason in row.get("validation_reasons") or row.get("validation", {}).get("reasons", []):
            reason_key = str(reason)
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

    logger.info(
        "Traders firehose filtered",
        strategy=getattr(strategy, "strategy_type", _STRATEGY_SLUG),
        raw_rows=len(rows),
        filtered_rows=len(output_rows),
        passed_rows=len(passed_rows),
        filtered_out_rows=len(filtered_out_rows),
        include_filtered=bool(include_filtered),
        reason_counts=reason_counts,
    )
    return output_rows


async def apply_traders_firehose_strategy(
    rows: list[dict[str, Any]],
    *,
    include_filtered: bool = False,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    cloned_rows: list[dict[str, Any]] = [
        StrategySDK.normalize_trader_signal(dict(row)) for row in rows if isinstance(row, dict)
    ]
    if not cloned_rows:
        return []

    async with AsyncSessionLocal() as session:
        strategy = await _resolve_traders_strategy(session)
        await _annotate_source_flags(session, cloned_rows)

    if strategy is None:
        return []

    await _annotate_firehose_context(cloned_rows)
    return _apply_strategy_filter(
        strategy,
        cloned_rows,
        include_filtered=include_filtered,
        limit=limit,
    )


async def get_strategy_filtered_trader_opportunities(
    *,
    limit: int = 50,
    include_filtered: bool = False,
) -> list[dict[str, Any]]:
    safe_limit = max(1, int(limit))
    firehose_scan_limit = max(250, safe_limit * 6)
    firehose_rows = await smart_wallet_pool.get_tracked_trader_firehose_signals(
        limit=firehose_scan_limit,
        include_filtered=include_filtered,
    )
    return await apply_traders_firehose_strategy(
        firehose_rows,
        include_filtered=include_filtered,
        limit=safe_limit,
    )


async def get_strategy_trader_opportunities(
    *,
    limit: int = 50,
    include_filtered: bool = False,
) -> list[ArbitrageOpportunity]:
    safe_limit = max(1, int(limit))
    filtered_rows = await get_strategy_filtered_trader_opportunities(
        limit=max(250, safe_limit * 6),
        include_filtered=include_filtered,
    )
    return await build_strategy_trader_opportunities_from_rows(filtered_rows, limit=safe_limit)


async def build_strategy_trader_opportunities_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = 50,
) -> list[ArbitrageOpportunity]:
    safe_limit = max(1, int(limit))
    filtered_rows = [dict(row) for row in rows if isinstance(row, dict)]
    if not filtered_rows:
        return []

    async with AsyncSessionLocal() as session:
        strategy = await _resolve_traders_strategy(session)
    if strategy is None:
        return []

    builder = getattr(strategy, "build_opportunities_from_firehose", None)
    if not callable(builder):
        logger.error("Traders strategy missing build_opportunities_from_firehose", slug=_STRATEGY_SLUG)
        return []

    built = builder(filtered_rows, limit=safe_limit)
    if asyncio.iscoroutine(built):
        built = await built

    if not isinstance(built, list):
        logger.error("Traders strategy returned non-list opportunities", slug=_STRATEGY_SLUG)
        return []

    opportunities: list[ArbitrageOpportunity] = []
    for item in built:
        if isinstance(item, ArbitrageOpportunity):
            opportunities.append(item)
            continue
        if isinstance(item, dict):
            try:
                opportunities.append(ArbitrageOpportunity.model_validate(item))
            except Exception:
                continue

    logger.info(
        "Traders opportunities built",
        strategy=getattr(strategy, "strategy_type", _STRATEGY_SLUG),
        filtered_rows=len(filtered_rows),
        opportunities=len(opportunities),
    )
    return opportunities[:safe_limit]
