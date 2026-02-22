"""Tracked-traders worker: smart pool + confluence lifecycle owner.

Moves smart wallet pool and confluence loops out of the API process.
Builds strategy-owned trader opportunities and writes them to shared snapshot state.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import timedelta
from typing import Any
from sqlalchemy import select

from utils.utcnow import utcnow
from models.database import AsyncSessionLocal, AppSettings
from services.data_events import DataEvent, EventType
from services.event_dispatcher import event_dispatcher
from services.insider_detector import insider_detector
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.market_cache import market_cache_service
from services.market_tradability import get_market_tradability_map
from services.scanner import scanner as market_scanner
from services.smart_wallet_pool import smart_wallet_pool
from services import shared_state
from services.strategy_sdk import StrategySDK
from services.wallet_intelligence import wallet_intelligence
from services.worker_state import (
    clear_worker_run_request,
    ensure_worker_control,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("tracked_traders_worker")


FULL_SWEEP_INTERVAL = timedelta(minutes=30)
INCREMENTAL_REFRESH_INTERVAL = timedelta(minutes=2)
ACTIVITY_RECONCILE_INTERVAL = timedelta(minutes=2)
POOL_RECOMPUTE_INTERVAL = timedelta(minutes=1)
FULL_INTELLIGENCE_INTERVAL = timedelta(minutes=20)
INSIDER_RESCORING_INTERVAL = timedelta(minutes=10)
POOL_RECOMPUTE_MODE_MAP = {"quality_only": "quality_only", "balanced": "balanced"}


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


async def _trader_opportunity_intent_settings() -> dict[str, Any]:
    config = {
        "confluence_limit": 50,
    }
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
            if not row:
                return config

            config["confluence_limit"] = _clamp_int(
                row.discovery_trader_opps_confluence_limit,
                default=50,
                minimum=1,
                maximum=400,
            )
    except Exception as exc:
        logger.warning("Failed to read trader opportunity intent settings: %s", exc)
    return config


async def _pool_runtime_settings() -> dict[str, Any]:
    _sdk = StrategySDK.POOL_ELIGIBILITY_DEFAULTS
    config = {"recompute_mode": "quality_only", **_sdk}
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
            if not row:
                return config
            stored_mode = str(row.discovery_pool_recompute_mode or "quality_only").strip().lower()
            config["recompute_mode"] = POOL_RECOMPUTE_MODE_MAP.get(stored_mode, "quality_only")
            # DB overrides — fallback to SDK defaults for each field.
            _db_overrides = {
                "target_pool_size": row.discovery_pool_target_size,
                "min_pool_size": row.discovery_pool_min_size,
                "max_pool_size": row.discovery_pool_max_size,
                "active_window_hours": row.discovery_pool_active_window_hours,
                "inactive_rising_retention_hours": row.discovery_pool_inactive_rising_retention_hours,
                "selection_score_quality_target_floor": row.discovery_pool_selection_score_floor,
                "max_hourly_replacement_rate": row.discovery_pool_max_hourly_replacement_rate,
                "replacement_score_cutoff": row.discovery_pool_replacement_score_cutoff,
                "max_cluster_share": row.discovery_pool_max_cluster_share,
                "high_conviction_threshold": row.discovery_pool_high_conviction_threshold,
                "insider_priority_threshold": row.discovery_pool_insider_priority_threshold,
                "min_eligible_trades": row.discovery_pool_min_eligible_trades,
                "max_eligible_anomaly": row.discovery_pool_max_eligible_anomaly,
                "core_min_win_rate": row.discovery_pool_core_min_win_rate,
                "core_min_sharpe": row.discovery_pool_core_min_sharpe,
                "core_min_profit_factor": row.discovery_pool_core_min_profit_factor,
                "rising_min_win_rate": row.discovery_pool_rising_min_win_rate,
                "slo_min_analyzed_pct": row.discovery_pool_slo_min_analyzed_pct,
                "slo_min_profitable_pct": row.discovery_pool_slo_min_profitable_pct,
                "leaderboard_wallet_trade_sample": row.discovery_pool_leaderboard_wallet_trade_sample,
                "incremental_wallet_trade_sample": row.discovery_pool_incremental_wallet_trade_sample,
                "full_sweep_interval_seconds": row.discovery_pool_full_sweep_interval_seconds,
                "incremental_refresh_interval_seconds": row.discovery_pool_incremental_refresh_interval_seconds,
                "activity_reconciliation_interval_seconds": row.discovery_pool_activity_reconciliation_interval_seconds,
                "pool_recompute_interval_seconds": row.discovery_pool_recompute_interval_seconds,
            }
            for key, db_value in _db_overrides.items():
                if db_value is not None:
                    config[key] = db_value
    except Exception as exc:
        logger.warning("Failed to read pool runtime settings: %s", exc)
    return config


async def _market_cache_hygiene_settings() -> dict:
    config = {
        "enabled": True,
        "interval_hours": 6,
        "retention_days": 120,
        "reference_lookback_days": 45,
        "weak_entry_grace_days": 7,
        "max_entries_per_slug": 3,
    }
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
            if not row:
                return config
            config["enabled"] = bool(
                row.market_cache_hygiene_enabled if row.market_cache_hygiene_enabled is not None else True
            )
            config["interval_hours"] = int(row.market_cache_hygiene_interval_hours or 6)
            config["retention_days"] = int(row.market_cache_retention_days or 120)
            config["reference_lookback_days"] = int(row.market_cache_reference_lookback_days or 45)
            config["weak_entry_grace_days"] = int(row.market_cache_weak_entry_grace_days or 7)
            config["max_entries_per_slug"] = int(row.market_cache_max_entries_per_slug or 3)
    except Exception as exc:
        logger.warning("Failed to read market cache hygiene settings: %s", exc)
    return config


async def _run_loop() -> None:
    worker_name = "tracked_traders"
    logger.info("Tracked-traders worker started")

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(
                session,
                source_keys=["traders"],
                force=True,
            )
    except Exception as exc:
        logger.warning("Tracked-traders strategy startup sync failed: %s", exc)

    now = utcnow()
    next_full_sweep = now
    next_incremental = now
    next_reconcile = now
    next_recompute = now
    next_full_intelligence = now
    next_insider_rescore = now
    last_insider_flagged = 0

    await wallet_intelligence.initialize()

    async with AsyncSessionLocal() as session:
        await ensure_worker_control(session, worker_name, default_interval=30)
        await write_worker_snapshot(
            session,
            worker_name,
            running=True,
            enabled=True,
            current_activity="Tracked-traders worker started; first cycle pending.",
            interval_seconds=30,
            last_run_at=None,
            stats={
                "pool_size": 0,
                "active_signals": 0,
                "signals_emitted_last_run": 0,
                "confluence_high_extreme": 0,
                "confluence_scanned": 0,
                "confluence_executable": 0,
                "insider_wallets_flagged": 0,
            },
        )

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_worker_control(session, worker_name, default_interval=30)
            try:
                await refresh_strategy_runtime_if_needed(
                    session,
                    source_keys=["traders"],
                )
            except Exception as exc:
                logger.warning("Tracked-traders strategy refresh check failed: %s", exc)

        interval = max(10, min(3600, int(control.get("interval_seconds") or 60)))
        paused = bool(control.get("is_paused", False))
        enabled = bool(control.get("is_enabled", True))
        requested = control.get("requested_run_at") is not None

        if (not enabled or paused) and not requested:
            async with AsyncSessionLocal() as session:
                await shared_state.write_traders_snapshot(
                    session,
                    [],
                    {
                        "running": True,
                        "enabled": enabled and not paused,
                        "interval_seconds": interval,
                        "last_scan": utcnow().isoformat(),
                        "current_activity": "Paused" if paused else "Disabled",
                        "strategies": [],
                    },
                )
                await write_worker_snapshot(
                    session,
                    worker_name,
                    running=True,
                    enabled=enabled and not paused,
                    current_activity="Paused" if paused else "Disabled",
                    interval_seconds=interval,
                    last_run_at=None,
                    stats={
                        "pool_size": 0,
                        "active_signals": 0,
                        "signals_emitted_last_run": 0,
                        "confluence_high_extreme": 0,
                        "confluence_scanned": 0,
                        "confluence_executable": 0,
                        "insider_wallets_flagged": 0,
                    },
                )
            await asyncio.sleep(min(10, interval))
            continue

        cycle_started = utcnow()
        emitted = 0
        confluence_count = 0
        confluence_scanned = 0
        insider_flagged = last_insider_flagged

        try:
            now = utcnow()
            activity_labels: list[str] = []
            pool_config = await _pool_runtime_settings()
            try:
                smart_wallet_pool.configure_runtime(pool_config)
                smart_wallet_pool.set_recompute_mode(pool_config["recompute_mode"])
            except Exception as exc:
                logger.warning("Failed to apply pool recompute mode '%s': %s", pool_config.get("recompute_mode"), exc)
            full_sweep_interval = timedelta(
                seconds=max(
                    10, int(pool_config.get("full_sweep_interval_seconds") or FULL_SWEEP_INTERVAL.total_seconds())
                )
            )
            incremental_refresh_interval = timedelta(
                seconds=max(
                    10,
                    int(
                        pool_config.get("incremental_refresh_interval_seconds")
                        or INCREMENTAL_REFRESH_INTERVAL.total_seconds()
                    ),
                )
            )
            activity_reconcile_interval = timedelta(
                seconds=max(
                    10,
                    int(
                        pool_config.get("activity_reconciliation_interval_seconds")
                        or ACTIVITY_RECONCILE_INTERVAL.total_seconds()
                    ),
                )
            )
            pool_recompute_interval = timedelta(
                seconds=max(
                    10,
                    int(pool_config.get("pool_recompute_interval_seconds") or POOL_RECOMPUTE_INTERVAL.total_seconds()),
                )
            )

            market_cache_cfg = await _market_cache_hygiene_settings()
            if market_cache_cfg["enabled"]:
                hygiene = await market_cache_service.run_hygiene_if_due(
                    force=requested,
                    interval_hours=market_cache_cfg["interval_hours"],
                    retention_days=market_cache_cfg["retention_days"],
                    reference_lookback_days=market_cache_cfg["reference_lookback_days"],
                    weak_entry_grace_days=market_cache_cfg["weak_entry_grace_days"],
                    max_entries_per_slug=market_cache_cfg["max_entries_per_slug"],
                )
                if hygiene.get("status") != "skipped":
                    activity_labels.append("market_cache_hygiene")
                    deleted = int(hygiene.get("markets_deleted", 0))
                    if deleted > 0:
                        activity_labels.append(f"market_cache_pruned:{deleted}")

            if requested or now >= next_full_sweep:
                activity_labels.append("full_sweep")
                try:
                    await asyncio.wait_for(smart_wallet_pool.run_full_sweep(), timeout=180)
                except asyncio.TimeoutError:
                    activity_labels.append("full_sweep_timeout")
                    logger.warning("Tracked-traders full_sweep timed out after 180s")
                next_full_sweep = now + full_sweep_interval

            if requested or now >= next_incremental:
                activity_labels.append("incremental_refresh")
                try:
                    await asyncio.wait_for(smart_wallet_pool.run_incremental_refresh(), timeout=90)
                except asyncio.TimeoutError:
                    activity_labels.append("incremental_refresh_timeout")
                    logger.warning("Tracked-traders incremental_refresh timed out after 90s")
                next_incremental = now + incremental_refresh_interval

            if requested or now >= next_reconcile:
                activity_labels.append("activity_reconcile")
                try:
                    await asyncio.wait_for(smart_wallet_pool.reconcile_activity(), timeout=45)
                except asyncio.TimeoutError:
                    activity_labels.append("activity_reconcile_timeout")
                    logger.warning("Tracked-traders activity_reconcile timed out after 45s")
                next_reconcile = now + activity_reconcile_interval

            if requested or now >= next_recompute:
                activity_labels.append("pool_recompute")
                try:
                    await asyncio.wait_for(smart_wallet_pool.recompute_pool(), timeout=90)
                except asyncio.TimeoutError:
                    activity_labels.append("pool_recompute_timeout")
                    logger.warning("Tracked-traders pool_recompute timed out after 90s")
                next_recompute = now + pool_recompute_interval

            if requested or now >= next_full_intelligence:
                activity_labels.append("full_intelligence")
                try:
                    await asyncio.wait_for(wallet_intelligence.run_full_analysis(), timeout=180)
                except asyncio.TimeoutError:
                    activity_labels.append("full_intelligence_timeout")
                    logger.warning("Tracked-traders full_intelligence timed out after 180s")
                next_full_intelligence = now + FULL_INTELLIGENCE_INTERVAL
            else:
                activity_labels.append("confluence_scan")
                try:
                    await asyncio.wait_for(wallet_intelligence.confluence.scan_for_confluence(), timeout=45)
                except asyncio.TimeoutError:
                    activity_labels.append("confluence_scan_timeout")
                    logger.warning("Tracked-traders confluence_scan timed out after 45s")

            if requested or now >= next_insider_rescore:
                activity_labels.append("insider_rescore")
                rescore = await insider_detector.rescore_wallets(stale_minutes=15)
                insider_flagged = int(rescore.get("flagged_insiders") or 0)
                last_insider_flagged = insider_flagged
                next_insider_rescore = now + INSIDER_RESCORING_INTERVAL

            trader_intent_settings = await _trader_opportunity_intent_settings()

            confluence_limit = int(trader_intent_settings["confluence_limit"])

            confluence_scan_limit = max(250, confluence_limit * 6)
            firehose_rows = await StrategySDK.get_trader_firehose_signals(
                limit=confluence_scan_limit,
                include_filtered=True,
                include_source_context=True,
            )
            if firehose_rows:
                market_ids = [
                    str(row.get("market_id") or "").strip().lower()
                    for row in firehose_rows
                    if isinstance(row, dict) and str(row.get("market_id") or "").strip()
                ]
                if market_ids:
                    tradability_map = await get_market_tradability_map(market_ids)
                    for row in firehose_rows:
                        if not isinstance(row, dict):
                            continue
                        market_id = str(row.get("market_id") or "").strip().lower()
                        if not market_id:
                            continue
                        if market_id in tradability_map:
                            tradable = bool(tradability_map[market_id])
                            row["is_tradeable"] = tradable
                            if not tradable:
                                row["is_active"] = False
            confluence_scanned = len(firehose_rows)
            deduped_by_stable_id: dict[str, Any] = {}
            deduped_opportunities: list[Any] = []
            strategies_used: list[str] = []
            emitted = 0

            # Dispatch trader_activity DataEvent so subscribed strategies
            # (e.g. TradersConfluenceStrategy) can react via on_event().
            if firehose_rows:
                try:
                    trader_event = DataEvent(
                        event_type=EventType.TRADER_ACTIVITY,
                        source="tracked_traders_worker",
                        timestamp=utcnow(),
                        payload={
                            "confluence_count": len(firehose_rows),
                            "signals": firehose_rows,
                        },
                    )
                    event_opps = await event_dispatcher.dispatch(trader_event)
                    if event_opps:
                        for opp in event_opps:
                            stable_id = str(getattr(opp, "stable_id", "") or getattr(opp, "id", "")).strip()
                            if stable_id and stable_id not in deduped_by_stable_id:
                                deduped_by_stable_id[stable_id] = opp
                        deduped_opportunities = list(deduped_by_stable_id.values())
                        if confluence_limit > 0:
                            deduped_opportunities = deduped_opportunities[:confluence_limit]
                        strategies_used = sorted(
                            {
                                str(getattr(opp, "strategy", "") or "").strip()
                                for opp in deduped_opportunities
                                if str(getattr(opp, "strategy", "") or "").strip()
                            }
                        )
                except Exception as exc:
                    logger.warning("trader_activity DataEvent dispatch failed: %s", exc)
            confluence_count = len(deduped_opportunities)

            # Attach shared sparkline price history from the scanner cache
            if deduped_opportunities:
                try:
                    await market_scanner.attach_price_history_to_opportunities(
                        deduped_opportunities,
                        timeout_seconds=0.0,
                    )
                except Exception as exc:
                    logger.warning("Sparkline backfill for trader opps failed: %s", exc)

            async with AsyncSessionLocal() as session:
                emitted = await bridge_opportunities_to_signals(
                    session,
                    deduped_opportunities,
                    source="traders",
                    sweep_missing=True,
                )
                await shared_state.write_traders_snapshot(
                    session,
                    deduped_opportunities,
                    {
                        "running": True,
                        "enabled": True,
                        "interval_seconds": interval,
                        "last_scan": cycle_started.isoformat(),
                        "current_activity": "Tracked traders strategy cycle complete.",
                        "strategies": strategies_used,
                    },
                )
                if requested:
                    await clear_worker_run_request(session, worker_name)

            pool_stats = await smart_wallet_pool.get_pool_stats()

            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    worker_name,
                    running=True,
                    enabled=True,
                    current_activity=(
                        "Idle - tracked-traders cycle complete."
                        if not activity_labels
                        else f"Ran: {', '.join(activity_labels)}"
                    ),
                    interval_seconds=interval,
                    last_run_at=cycle_started,
                    last_error=None,
                    stats={
                        "pool_size": int(pool_stats.get("pool_size") or 0),
                        "active_signals": confluence_count,
                        "signals_emitted_last_run": int(emitted),
                        "confluence_high_extreme": int(emitted),
                        "confluence_scanned": int(confluence_scanned),
                        "confluence_executable": int(confluence_count),
                        "insider_wallets_flagged": int(insider_flagged),
                        "intent_settings": trader_intent_settings,
                        "pool_stats": pool_stats,
                    },
                )

            logger.info(
                "Tracked-traders cycle complete: raw=%s filtered=%s opportunities=%s",
                confluence_scanned,
                confluence_count,
                emitted,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Tracked-traders worker cycle failed: %s", exc)
            async with AsyncSessionLocal() as session:
                if requested:
                    await clear_worker_run_request(session, worker_name)
                await write_worker_snapshot(
                    session,
                    worker_name,
                    running=True,
                    enabled=True,
                    current_activity=f"Last tracked-traders cycle error: {exc}",
                    interval_seconds=interval,
                    last_run_at=cycle_started,
                    last_error=str(exc),
                    stats={
                        "pool_size": 0,
                        "active_signals": confluence_count,
                        "signals_emitted_last_run": int(emitted),
                        "confluence_high_extreme": confluence_count,
                        "confluence_scanned": int(confluence_scanned),
                        "confluence_executable": int(confluence_count),
                        "insider_wallets_flagged": int(insider_flagged),
                    },
                )

        await asyncio.sleep(interval)


async def start_loop() -> None:
    """Run the tracked-traders worker loop (called from API process lifespan).

    market_cache_service.load_from_db() is already done in main.py lifespan.
    """
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Tracked-traders worker shutting down")
