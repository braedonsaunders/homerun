"""Strategy-owned traders firehose filtering pipeline.

This module keeps trader firehose normalization, enrichment, filtering,
and opportunity conversion inside DB-loaded traders strategies.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    Strategy,
)
from models.opportunity import Opportunity
from services.opportunity_strategy_catalog import ensure_system_opportunity_strategies_seeded
from services.strategy_loader import strategy_loader
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.strategy_sdk import StrategySDK
from utils.logger import get_logger

logger = get_logger("traders_firehose_pipeline")

_STRATEGY_SLUG = "traders_confluence"


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

    if strategy is None:
        return []

    prepared_rows = cloned_rows
    prepare = getattr(strategy, "prepare_firehose_signals", None)
    if callable(prepare):
        prepared = prepare(cloned_rows)
        if asyncio.iscoroutine(prepared):
            prepared = await prepared
        if not isinstance(prepared, list):
            logger.error("Traders strategy returned non-list prepared firehose payload")
            return []
        prepared_rows = [StrategySDK.normalize_trader_signal(dict(row)) for row in prepared if isinstance(row, dict)]

    return _apply_strategy_filter(
        strategy,
        prepared_rows,
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
    firehose_rows = await StrategySDK.get_trader_firehose_signals(
        limit=firehose_scan_limit,
        include_filtered=include_filtered,
        include_source_context=False,
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
) -> list[Opportunity]:
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
) -> list[Opportunity]:
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

    opportunities: list[Opportunity] = []
    for item in built:
        if isinstance(item, Opportunity):
            opportunities.append(item)
            continue
        if isinstance(item, dict):
            try:
                opportunities.append(Opportunity.model_validate(item))
            except Exception:
                continue

    logger.info(
        "Traders opportunities built",
        strategy=getattr(strategy, "strategy_type", _STRATEGY_SLUG),
        filtered_rows=len(filtered_rows),
        opportunities=len(opportunities),
    )
    return opportunities[:safe_limit]
