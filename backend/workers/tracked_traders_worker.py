"""Tracked-traders worker: smart pool + confluence lifecycle owner.

Moves smart wallet pool and confluence loops out of the API process.
Emits normalized executable intents based on trader opportunity settings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
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
TRADER_OPP_SIDE_MAP = {"all": "all", "buy": "buy", "sell": "sell"}
# Legacy values are kept for compatibility; tracked/pool map to confluence
# for worker emission purposes.
TRADER_OPP_SOURCE_MAP = {
    "all": "all",
    "confluence": "confluence",
    "insider": "insider",
    "tracked": "confluence",
    "pool": "confluence",
}
TRADER_OPP_MIN_TIER_MAP = {"WATCH": "WATCH", "HIGH": "HIGH", "EXTREME": "EXTREME"}
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


def _to_utc_naive(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return None


def _confluence_direction(signal: dict[str, Any]) -> str | None:
    outcome = str(signal.get("outcome") or "").strip().upper()
    if outcome == "YES":
        return "buy_yes"
    if outcome == "NO":
        return "buy_no"

    signal_type = str(signal.get("signal_type") or "").strip().lower()
    if "sell" in signal_type:
        return "buy_no"
    if "buy" in signal_type or "accumulation" in signal_type:
        return "buy_yes"
    return None


def _matches_side(side_filter: str, direction: str | None) -> bool:
    normalized = str(side_filter or "all").strip().lower()
    if normalized == "all":
        return True
    if normalized == "buy":
        return direction == "buy_yes"
    if normalized == "sell":
        return direction == "buy_no"
    return True


async def _trader_opportunity_intent_settings() -> dict[str, Any]:
    config = {
        "source_filter": "all",
        "min_tier": "WATCH",
        "side_filter": "all",
        "confluence_limit": 50,
        "insider_limit": 40,
        "insider_min_confidence": 0.62,
        "insider_max_age_minutes": 180,
    }
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
            if not row:
                return config

            source_filter = str(row.discovery_trader_opps_source_filter or "all").strip().lower()
            min_tier = str(row.discovery_trader_opps_min_tier or "WATCH").strip().upper()
            side_filter = str(row.discovery_trader_opps_side_filter or "all").strip().lower()

            config["source_filter"] = TRADER_OPP_SOURCE_MAP.get(source_filter, "all")
            config["min_tier"] = TRADER_OPP_MIN_TIER_MAP.get(min_tier, "WATCH")
            config["side_filter"] = TRADER_OPP_SIDE_MAP.get(side_filter, "all")
            config["confluence_limit"] = _clamp_int(
                row.discovery_trader_opps_confluence_limit,
                default=50,
                minimum=1,
                maximum=400,
            )
            config["insider_limit"] = _clamp_int(
                row.discovery_trader_opps_insider_limit,
                default=40,
                minimum=1,
                maximum=1000,
            )
            config["insider_min_confidence"] = _clamp_float(
                row.discovery_trader_opps_insider_min_confidence,
                default=0.62,
                minimum=0.0,
                maximum=1.0,
            )
            config["insider_max_age_minutes"] = _clamp_int(
                row.discovery_trader_opps_insider_max_age_minutes,
                default=180,
                minimum=1,
                maximum=1440,
            )
    except Exception as exc:
        logger.warning("Failed to read trader opportunity intent settings: %s", exc)
    return config


async def _pool_runtime_settings() -> dict[str, Any]:
    config = {
        "recompute_mode": "quality_only",
        "target_pool_size": 500,
        "min_pool_size": 400,
        "max_pool_size": 600,
        "active_window_hours": 72,
        "selection_score_quality_target_floor": 0.55,
        "max_hourly_replacement_rate": 0.15,
        "replacement_score_cutoff": 0.05,
        "max_cluster_share": 0.08,
        "high_conviction_threshold": 0.72,
        "insider_priority_threshold": 0.62,
        "min_eligible_trades": 50,
        "max_eligible_anomaly": 0.5,
        "core_min_win_rate": 0.60,
        "core_min_sharpe": 1.0,
        "core_min_profit_factor": 1.5,
        "rising_min_win_rate": 0.55,
        "slo_min_analyzed_pct": 95.0,
        "slo_min_profitable_pct": 80.0,
        "leaderboard_wallet_trade_sample": 160,
        "incremental_wallet_trade_sample": 80,
        "full_sweep_interval_seconds": 1800,
        "incremental_refresh_interval_seconds": 120,
        "activity_reconciliation_interval_seconds": 120,
        "pool_recompute_interval_seconds": 60,
    }
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
            if not row:
                return config
            stored_mode = str(row.discovery_pool_recompute_mode or "quality_only").strip().lower()
            config["recompute_mode"] = POOL_RECOMPUTE_MODE_MAP.get(stored_mode, "quality_only")
            config["target_pool_size"] = (
                row.discovery_pool_target_size if row.discovery_pool_target_size is not None else 500
            )
            config["min_pool_size"] = (
                row.discovery_pool_min_size if row.discovery_pool_min_size is not None else 400
            )
            config["max_pool_size"] = (
                row.discovery_pool_max_size if row.discovery_pool_max_size is not None else 600
            )
            config["active_window_hours"] = (
                row.discovery_pool_active_window_hours if row.discovery_pool_active_window_hours is not None else 72
            )
            config["selection_score_quality_target_floor"] = (
                row.discovery_pool_selection_score_floor
                if row.discovery_pool_selection_score_floor is not None
                else 0.55
            )
            config["max_hourly_replacement_rate"] = (
                row.discovery_pool_max_hourly_replacement_rate
                if row.discovery_pool_max_hourly_replacement_rate is not None
                else 0.15
            )
            config["replacement_score_cutoff"] = (
                row.discovery_pool_replacement_score_cutoff
                if row.discovery_pool_replacement_score_cutoff is not None
                else 0.05
            )
            config["max_cluster_share"] = (
                row.discovery_pool_max_cluster_share if row.discovery_pool_max_cluster_share is not None else 0.08
            )
            config["high_conviction_threshold"] = (
                row.discovery_pool_high_conviction_threshold
                if row.discovery_pool_high_conviction_threshold is not None
                else 0.72
            )
            config["insider_priority_threshold"] = (
                row.discovery_pool_insider_priority_threshold
                if row.discovery_pool_insider_priority_threshold is not None
                else 0.62
            )
            config["min_eligible_trades"] = (
                row.discovery_pool_min_eligible_trades if row.discovery_pool_min_eligible_trades is not None else 50
            )
            config["max_eligible_anomaly"] = (
                row.discovery_pool_max_eligible_anomaly if row.discovery_pool_max_eligible_anomaly is not None else 0.5
            )
            config["core_min_win_rate"] = (
                row.discovery_pool_core_min_win_rate if row.discovery_pool_core_min_win_rate is not None else 0.60
            )
            config["core_min_sharpe"] = (
                row.discovery_pool_core_min_sharpe if row.discovery_pool_core_min_sharpe is not None else 1.0
            )
            config["core_min_profit_factor"] = (
                row.discovery_pool_core_min_profit_factor
                if row.discovery_pool_core_min_profit_factor is not None
                else 1.5
            )
            config["rising_min_win_rate"] = (
                row.discovery_pool_rising_min_win_rate if row.discovery_pool_rising_min_win_rate is not None else 0.55
            )
            config["slo_min_analyzed_pct"] = (
                row.discovery_pool_slo_min_analyzed_pct if row.discovery_pool_slo_min_analyzed_pct is not None else 95.0
            )
            config["slo_min_profitable_pct"] = (
                row.discovery_pool_slo_min_profitable_pct
                if row.discovery_pool_slo_min_profitable_pct is not None
                else 80.0
            )
            config["leaderboard_wallet_trade_sample"] = (
                row.discovery_pool_leaderboard_wallet_trade_sample
                if row.discovery_pool_leaderboard_wallet_trade_sample is not None
                else 160
            )
            config["incremental_wallet_trade_sample"] = (
                row.discovery_pool_incremental_wallet_trade_sample
                if row.discovery_pool_incremental_wallet_trade_sample is not None
                else 80
            )
            config["full_sweep_interval_seconds"] = (
                row.discovery_pool_full_sweep_interval_seconds
                if row.discovery_pool_full_sweep_interval_seconds is not None
                else 1800
            )
            config["incremental_refresh_interval_seconds"] = (
                row.discovery_pool_incremental_refresh_interval_seconds
                if row.discovery_pool_incremental_refresh_interval_seconds is not None
                else 120
            )
            config["activity_reconciliation_interval_seconds"] = (
                row.discovery_pool_activity_reconciliation_interval_seconds
                if row.discovery_pool_activity_reconciliation_interval_seconds is not None
                else 120
            )
            config["pool_recompute_interval_seconds"] = (
                row.discovery_pool_recompute_interval_seconds
                if row.discovery_pool_recompute_interval_seconds is not None
                else 60
            )
    except Exception as exc:
        logger.warning("Failed to read pool runtime settings: %s", exc)
    return config


def _rank_confluence_signal(signal: dict[str, Any]) -> tuple[float, float, float]:
    conviction = float(signal.get("conviction_score") or 0.0)
    wallets = float(signal.get("cluster_adjusted_wallet_count") or signal.get("wallet_count") or 0.0)
    last_seen = _to_utc_naive(signal.get("last_seen_at") or signal.get("detected_at"))
    last_seen_ts = last_seen.timestamp() if last_seen is not None else 0.0
    return (conviction, wallets, last_seen_ts)


def _filter_executable_confluence(
    opportunities: list[dict[str, Any]],
    *,
    side_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for signal in opportunities:
        market_id = str(signal.get("market_id") or "").strip()
        if not market_id:
            continue
        if signal.get("is_tradeable") is False:
            continue
        direction = _confluence_direction(signal)
        if not _matches_side(side_filter, direction):
            continue

        wallets = int(signal.get("cluster_adjusted_wallet_count") or signal.get("wallet_count") or 0)
        if wallets <= 0:
            continue

        entry = signal.get("avg_entry_price")
        if entry is not None:
            try:
                price = float(entry)
            except Exception:
                continue
            if price < 0.0 or price > 1.0:
                continue

        eligible.append(signal)

    eligible.sort(key=_rank_confluence_signal, reverse=True)

    by_market: dict[str, dict[str, Any]] = {}
    for signal in eligible:
        market_key = str(signal.get("market_id") or "").strip().lower()
        if not market_key or market_key in by_market:
            continue
        by_market[market_key] = signal
        if len(by_market) >= limit:
            break

    return list(by_market.values())


def _rank_insider_intent(intent: Any) -> tuple[float, float, float]:
    confidence = float(getattr(intent, "confidence", 0.0) or 0.0)
    if confidence > 1.0:
        confidence = confidence / 100.0
    edge = float(getattr(intent, "edge_percent", 0.0) or 0.0)
    created_at = _to_utc_naive(getattr(intent, "created_at", None))
    created_ts = created_at.timestamp() if created_at is not None else 0.0
    return (confidence, edge, created_ts)


def _filter_executable_insider_intents(
    intents: list[Any],
    *,
    side_filter: str,
    min_confidence: float,
    max_age_minutes: int,
    limit: int,
    now: datetime,
) -> list[Any]:
    eligible: list[Any] = []
    max_age_seconds = float(max_age_minutes) * 60.0

    for intent in intents:
        market_id = str(getattr(intent, "market_id", "") or "").strip()
        if not market_id:
            continue

        direction = str(getattr(intent, "direction", "") or "").strip().lower()
        if direction not in {"buy_yes", "buy_no"}:
            continue
        if not _matches_side(side_filter, direction):
            continue

        confidence = float(getattr(intent, "confidence", 0.0) or 0.0)
        if confidence > 1.0:
            confidence = confidence / 100.0
        if confidence < min_confidence:
            continue

        created_at = _to_utc_naive(getattr(intent, "created_at", None))
        if created_at is None:
            continue
        age_seconds = (now - created_at).total_seconds()
        if age_seconds < 0:
            age_seconds = 0
        if age_seconds > max_age_seconds:
            continue

        eligible.append(intent)

    eligible.sort(key=_rank_insider_intent, reverse=True)

    by_market: dict[str, Any] = {}
    for intent in eligible:
        market_key = str(getattr(intent, "market_id", "") or "").strip().lower()
        if not market_key or market_key in by_market:
            continue
        by_market[market_key] = intent
        if len(by_market) >= limit:
            break

    return list(by_market.values())


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
                "insider_intents_pending": 0,
                "insider_intents_executable": 0,
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
                        "confluence_scanned": 0,
                        "confluence_executable": 0,
                        "insider_wallets_flagged": 0,
                        "insider_intents_pending": 0,
                        "insider_intents_executable": 0,
                        "insider_signals_emitted_last_run": 0,
                    },
                )
            await asyncio.sleep(min(10, interval))
            continue

        cycle_started = utcnow()
        emitted = 0
        insider_emitted = 0
        confluence_count = 0
        confluence_scanned = 0
        insider_flagged = last_insider_flagged
        insider_pending = 0
        insider_executable = 0

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
                seconds=max(10, int(pool_config.get("full_sweep_interval_seconds") or FULL_SWEEP_INTERVAL.total_seconds()))
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
                await smart_wallet_pool.run_full_sweep()
                next_full_sweep = now + full_sweep_interval

            if requested or now >= next_incremental:
                activity_labels.append("incremental_refresh")
                await smart_wallet_pool.run_incremental_refresh()
                next_incremental = now + incremental_refresh_interval

            if requested or now >= next_reconcile:
                activity_labels.append("activity_reconcile")
                await smart_wallet_pool.reconcile_activity()
                next_reconcile = now + activity_reconcile_interval

            if requested or now >= next_recompute:
                activity_labels.append("pool_recompute")
                await smart_wallet_pool.recompute_pool()
                next_recompute = now + pool_recompute_interval

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

            trader_intent_settings = await _trader_opportunity_intent_settings()

            source_filter = trader_intent_settings["source_filter"]
            side_filter = trader_intent_settings["side_filter"]
            confluence_limit = int(trader_intent_settings["confluence_limit"])
            insider_limit = int(trader_intent_settings["insider_limit"])
            insider_min_confidence = float(trader_intent_settings["insider_min_confidence"])
            insider_max_age_minutes = int(trader_intent_settings["insider_max_age_minutes"])

            confluence_scan_limit = max(250, confluence_limit * 4)
            opportunities = await smart_wallet_pool.get_tracked_trader_opportunities(
                limit=confluence_scan_limit,
                min_tier=str(trader_intent_settings["min_tier"]),
            )
            confluence_scanned = len(opportunities)
            pending_insider_intents = await insider_detector.list_intents(
                status_filter="pending",
                limit=max(1000, insider_limit * 8),
            )
            insider_pending = len(pending_insider_intents)

            executable_confluence = (
                _filter_executable_confluence(
                    opportunities,
                    side_filter=side_filter,
                    limit=confluence_limit,
                )
                if source_filter in {"all", "confluence"}
                else []
            )
            executable_insider = (
                _filter_executable_insider_intents(
                    pending_insider_intents,
                    side_filter=side_filter,
                    min_confidence=insider_min_confidence,
                    max_age_minutes=insider_max_age_minutes,
                    limit=insider_limit,
                    now=now,
                )
                if source_filter in {"all", "insider"}
                else []
            )

            confluence_count = len(executable_confluence)
            insider_executable = len(executable_insider)

            async with AsyncSessionLocal() as session:
                emitted = await emit_tracked_trader_signals(session, executable_confluence)
                insider_emitted = await emit_insider_intent_signals(
                    session,
                    executable_insider,
                    max_age_minutes=insider_max_age_minutes,
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
                        "confluence_scanned": int(confluence_scanned),
                        "confluence_executable": int(confluence_count),
                        "insider_wallets_flagged": int(insider_flagged),
                        "insider_intents_pending": int(insider_pending),
                        "insider_intents_executable": int(insider_executable),
                        "insider_signals_emitted_last_run": int(insider_emitted),
                        "intent_settings": trader_intent_settings,
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
                        "confluence_scanned": int(confluence_scanned),
                        "confluence_executable": int(confluence_count),
                        "insider_wallets_flagged": int(insider_flagged),
                        "insider_intents_pending": int(insider_pending),
                        "insider_intents_executable": int(insider_executable),
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
