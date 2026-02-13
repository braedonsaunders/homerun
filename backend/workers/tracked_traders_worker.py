"""Tracked-traders worker: smart pool + confluence lifecycle owner.

Moves smart wallet pool and confluence loops out of the API process.
Emits normalized signals from HIGH/EXTREME confluence only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import timedelta
from sqlalchemy import select

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from utils.utcnow import utcnow
from models.database import AsyncSessionLocal, AppSettings, init_database
from services.insider_detector import insider_detector
from services.signal_bus import emit_insider_intent_signals, emit_tracked_trader_signals
from services.market_cache import market_cache_service
from services.smart_wallet_pool import smart_wallet_pool
from services.wallet_intelligence import wallet_intelligence
from services.worker_state import (
    clear_worker_run_request,
    ensure_worker_control,
    read_worker_control,
    write_worker_snapshot,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tracked_traders_worker")


FULL_SWEEP_INTERVAL = timedelta(minutes=30)
INCREMENTAL_REFRESH_INTERVAL = timedelta(minutes=2)
ACTIVITY_RECONCILE_INTERVAL = timedelta(minutes=2)
POOL_RECOMPUTE_INTERVAL = timedelta(minutes=1)
FULL_INTELLIGENCE_INTERVAL = timedelta(minutes=20)
INSIDER_RESCORING_INTERVAL = timedelta(minutes=10)


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
            row = (
                await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            ).scalar_one_or_none()
            if not row:
                return config
            config["enabled"] = bool(
                row.market_cache_hygiene_enabled
                if row.market_cache_hygiene_enabled is not None
                else True
            )
            config["interval_hours"] = int(row.market_cache_hygiene_interval_hours or 6)
            config["retention_days"] = int(row.market_cache_retention_days or 120)
            config["reference_lookback_days"] = int(
                row.market_cache_reference_lookback_days or 45
            )
            config["weak_entry_grace_days"] = int(
                row.market_cache_weak_entry_grace_days or 7
            )
            config["max_entries_per_slug"] = int(
                row.market_cache_max_entries_per_slug or 3
            )
    except Exception as exc:
        logger.warning("Failed to read market cache hygiene settings: %s", exc)
    return config


async def _run_loop() -> None:
    worker_name = "tracked_traders"
    logger.info("Tracked-traders worker started")

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
                "insider_wallets_flagged": 0,
                "insider_intents_pending": 0,
                "insider_signals_emitted_last_run": 0,
            },
        )

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_worker_control(session, worker_name, default_interval=30)

        interval = max(10, min(3600, int(control.get("interval_seconds") or 60)))
        paused = bool(control.get("is_paused", False))
        enabled = bool(control.get("is_enabled", True))
        requested = control.get("requested_run_at") is not None

        if (not enabled or paused) and not requested:
            async with AsyncSessionLocal() as session:
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
                        "insider_wallets_flagged": 0,
                        "insider_intents_pending": 0,
                        "insider_signals_emitted_last_run": 0,
                    },
                )
            await asyncio.sleep(min(10, interval))
            continue

        cycle_started = utcnow()
        emitted = 0
        insider_emitted = 0
        confluence_count = 0
        insider_flagged = last_insider_flagged
        insider_pending = 0

        try:
            now = utcnow()
            activity_labels: list[str] = []

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
                await smart_wallet_pool.run_full_sweep()
                next_full_sweep = now + FULL_SWEEP_INTERVAL

            if requested or now >= next_incremental:
                activity_labels.append("incremental_refresh")
                await smart_wallet_pool.run_incremental_refresh()
                next_incremental = now + INCREMENTAL_REFRESH_INTERVAL

            if requested or now >= next_reconcile:
                activity_labels.append("activity_reconcile")
                await smart_wallet_pool.reconcile_activity()
                next_reconcile = now + ACTIVITY_RECONCILE_INTERVAL

            if requested or now >= next_recompute:
                activity_labels.append("pool_recompute")
                await smart_wallet_pool.recompute_pool()
                next_recompute = now + POOL_RECOMPUTE_INTERVAL

            if requested or now >= next_full_intelligence:
                activity_labels.append("full_intelligence")
                await wallet_intelligence.run_full_analysis()
                next_full_intelligence = now + FULL_INTELLIGENCE_INTERVAL
            else:
                activity_labels.append("confluence_scan")
                await wallet_intelligence.confluence.scan_for_confluence()

            if requested or now >= next_insider_rescore:
                activity_labels.append("insider_rescore")
                rescore = await insider_detector.rescore_wallets(stale_minutes=15)
                insider_flagged = int(rescore.get("flagged_insiders") or 0)
                last_insider_flagged = insider_flagged
                next_insider_rescore = now + INSIDER_RESCORING_INTERVAL

            activity_labels.append("insider_intents")
            await insider_detector.generate_intents()

            opportunities = await smart_wallet_pool.get_tracked_trader_opportunities(
                limit=250,
                min_tier="HIGH",
            )
            confluence_count = len(opportunities)
            pending_insider_intents = await insider_detector.list_intents(
                status_filter="pending",
                limit=1000,
            )
            insider_pending = len(pending_insider_intents)

            async with AsyncSessionLocal() as session:
                emitted = await emit_tracked_trader_signals(session, opportunities)
                insider_emitted = await emit_insider_intent_signals(
                    session,
                    pending_insider_intents,
                    max_age_minutes=180,
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
                        "confluence_high_extreme": confluence_count,
                        "insider_wallets_flagged": int(insider_flagged),
                        "insider_intents_pending": int(insider_pending),
                        "insider_signals_emitted_last_run": int(insider_emitted),
                        "pool_stats": pool_stats,
                    },
                )

            logger.info(
                "Tracked-traders cycle complete: signals=%s emitted=%s insider_emitted=%s",
                confluence_count,
                emitted,
                insider_emitted,
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
                        "insider_wallets_flagged": int(insider_flagged),
                        "insider_intents_pending": int(insider_pending),
                        "insider_signals_emitted_last_run": int(insider_emitted),
                    },
                )

        await asyncio.sleep(interval)


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
    await market_cache_service.load_from_db()
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Tracked-traders worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
