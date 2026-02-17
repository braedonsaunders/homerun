"""Strategy-owned traders firehose filtering pipeline.

This module keeps firehose gating for Traders opportunities in the
user-configurable ``traders_confluence`` opportunity strategy code.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
from services.market_tradability import get_market_tradability_map
from services.opportunity_strategy_catalog import ensure_system_opportunity_strategies_seeded
from services.strategy_loader import StrategyLoader as PluginLoader, StrategyValidationError as PluginValidationError
from services.smart_wallet_pool import _looks_like_crypto_market, smart_wallet_pool
from services.strategies.traders_confluence import TradersConfluenceStrategy
from utils.logger import get_logger
from utils.converters import normalize_market_id, safe_float
from utils.utcnow import utcnow

from functools import partial
_safe_float = partial(safe_float, reject_nan_inf=True)

logger = get_logger("traders_firehose_pipeline")

_STRATEGY_SLUG = "traders_confluence"
_strategy_cache_lock = asyncio.Lock()
_strategy_cache_key: Optional[tuple[str, int, str]] = None
_strategy_cache_instance: Optional[Any] = None


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
        try:
            instance.configure(config)
        except Exception as exc:
            logger.warning("Failed to apply configure() on traders strategy: %s", exc)

    default_config = getattr(instance, "default_config", None)
    current_config = getattr(instance, "config", None)
    if isinstance(default_config, dict):
        merged = dict(default_config)
        merged.update(config)
        try:
            setattr(instance, "config", merged)
        except Exception:
            pass
    elif isinstance(current_config, dict):
        merged = dict(current_config)
        merged.update(config)
        try:
            setattr(instance, "config", merged)
        except Exception:
            pass

    internal_cfg = getattr(instance, "_config", None)
    if isinstance(internal_cfg, dict):
        merged = dict(internal_cfg)
        merged.update(config)
        try:
            setattr(instance, "_config", merged)
        except Exception:
            pass


async def _load_strategy_row(session: AsyncSession) -> Optional[Strategy]:
    await ensure_system_opportunity_strategies_seeded(session)
    return (
        (
            await session.execute(
                select(Strategy).where(Strategy.slug == _STRATEGY_SLUG)
            )
        )
        .scalars()
        .first()
    )


async def _resolve_traders_strategy(session: AsyncSession) -> Optional[Any]:
    global _strategy_cache_key, _strategy_cache_instance

    row = await _load_strategy_row(session)
    if row is None:
        fallback = TradersConfluenceStrategy()
        _apply_strategy_config(fallback, {})
        return fallback

    if row.enabled is False:
        logger.info("Traders strategy is disabled; opportunities firehose suppressed")
        return None

    cache_key = (
        str(row.id),
        int(row.version or 0),
        (row.updated_at.isoformat() if row.updated_at else ""),
    )

    async with _strategy_cache_lock:
        if _strategy_cache_key == cache_key and _strategy_cache_instance is not None:
            return _strategy_cache_instance

        config = _strip_internal_schema(row.config)
        loader = PluginLoader()
        try:
            loaded = loader.load_plugin(
                slug=str(row.slug or _STRATEGY_SLUG),
                source_code=str(row.source_code or ""),
                config=config,
            )
            instance = loaded.instance
        except PluginValidationError as exc:
            logger.error("Failed to load traders strategy plugin source; using fallback: %s", exc)
            instance = TradersConfluenceStrategy()
        except Exception as exc:
            logger.error("Unexpected traders strategy load failure; using fallback: %s", exc)
            instance = TradersConfluenceStrategy()

        _apply_strategy_config(instance, config)
        _strategy_cache_key = cache_key
        _strategy_cache_instance = instance
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
            select(TrackedWallet.address).where(
                func.lower(TrackedWallet.address).in_(addr_list)
            )
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
        matched_group_ids = sorted(
            {gid for addr in wallets for gid in group_ids_by_address.get(addr, set())}
        )
        has_qualified_source = bool(pool_wallets or tracked_wallets or group_wallets)

        row["source_flags"] = {
            "from_pool": pool_wallets > 0,
            "from_tracked_traders": tracked_wallets > 0,
            "from_trader_groups": group_wallets > 0,
            "qualified": has_qualified_source,
        }
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
        normalize_market_id(row.get("market_id"))
        for row in rows
        if normalize_market_id(row.get("market_id"))
    ]
    tradability = await get_market_tradability_map(market_ids) if market_ids else {}
    now = utcnow()

    for row in rows:
        market_id = normalize_market_id(row.get("market_id"))
        detected = _parse_time(
            row.get("detected_at") or row.get("last_seen_at") or row.get("first_seen_at")
        )
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


def _fallback_filter(
    rows: list[dict[str, Any]],
    *,
    include_filtered: bool,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        tradable = bool(row.get("firehose_market_tradable", True))
        is_crypto = bool(row.get("firehose_is_crypto", False))
        passed = tradable and not is_crypto
        reasons: list[str] = []
        if not tradable:
            reasons.append("market_not_tradable")
        if is_crypto:
            reasons.append("crypto_market_excluded")

        row["is_valid"] = passed
        row["is_actionable"] = passed
        row["is_tradeable"] = passed
        row["validation_reasons"] = reasons
        row["validation"] = {
            "is_valid": passed,
            "is_actionable": passed,
            "is_tradeable": passed,
            "checks": {
                "has_market_id": bool(normalize_market_id(row.get("market_id"))),
                "upstream_tradable": tradable,
            },
            "reasons": reasons,
        }
        if passed or include_filtered:
            out.append(row)

    if limit and limit > 0:
        return out[:limit]
    return out


def _apply_strategy_filter(
    strategy: Any,
    rows: list[dict[str, Any]],
    *,
    include_filtered: bool,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    method = getattr(strategy, "apply_firehose_filters", None)
    if not callable(method):
        return _fallback_filter(rows, include_filtered=include_filtered, limit=limit)

    try:
        filtered = method(rows, include_filtered=include_filtered, limit=limit)
    except TypeError:
        filtered = method(rows)
    except Exception as exc:
        logger.error("Traders strategy firehose filter failed; using fallback: %s", exc)
        return _fallback_filter(rows, include_filtered=include_filtered, limit=limit)

    if not isinstance(filtered, list):
        logger.warning("Traders strategy returned non-list firehose payload; using fallback")
        return _fallback_filter(rows, include_filtered=include_filtered, limit=limit)

    cleaned = [row for row in filtered if isinstance(row, dict)]
    if limit and limit > 0:
        return cleaned[:limit]
    return cleaned


async def apply_traders_firehose_strategy(
    rows: list[dict[str, Any]],
    *,
    include_filtered: bool = False,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    cloned_rows: list[dict[str, Any]] = [dict(row) for row in rows if isinstance(row, dict)]
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
    min_tier: str = "WATCH",
    include_filtered: bool = False,
) -> list[dict[str, Any]]:
    safe_limit = max(1, int(limit))
    firehose_scan_limit = max(250, safe_limit * 6)
    firehose_rows = await smart_wallet_pool.get_tracked_trader_firehose_signals(
        limit=firehose_scan_limit,
        min_tier=min_tier,
        include_filtered=include_filtered,
    )
    return await apply_traders_firehose_strategy(
        firehose_rows,
        include_filtered=include_filtered,
        limit=safe_limit,
    )
