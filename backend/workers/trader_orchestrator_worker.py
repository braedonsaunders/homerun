"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import asyncpg
import copy
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from sqlalchemy import and_, func, or_, select, text, update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from config import settings
from models.database import (
    AppSettings,
    AsyncSessionLocal,
    DiscoveredWallet,
    TraderDecision,
    TraderEvent,
    TraderOrder,
    TraderOrchestratorSnapshot,
    TrackedWallet,
    Trader,
    TraderGroupMember,
    init_database,
    release_conn,
)
from services.trader_orchestrator.live_market_context import (
    RuntimeTradeSignalView,
    build_cached_live_signal_contexts,
    build_live_signal_contexts,
)
from services.trader_orchestrator.session_engine import ExecutionSessionEngine
from services.trader_orchestrator.position_lifecycle import (
    load_market_info_for_orders,
    reconcile_shadow_positions,
)
from services.polymarket import polymarket_client
from services.simulation import simulation_service
from services.live_execution_service import live_execution_service
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.decision_gates import (
    apply_platform_decision_gates,
    is_within_trading_schedule_utc,
)
from services.worker_state import _commit_with_retry, _is_retryable_db_error
from services.trader_orchestrator.sources.registry import (
    normalize_source_key,
)
from services.trader_orchestrator.strategies.registry import (
    get_strategy as resolve_strategy_instance,
)
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_experiments import (
    get_active_strategy_experiment,
    resolve_experiment_assignment,
    upsert_strategy_experiment_assignment as _upsert_strategy_experiment_assignment,
)
from services.strategy_loader import StrategyValidationError, strategy_loader
from services.intent_runtime import get_intent_runtime
from services.execution_latency_metrics import execution_latency_metrics
from services.runtime_status import runtime_status
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.strategy_sdk import StrategySDK
from services.strategies.news_edge import validate_news_edge_config
from services.strategies.traders_copy_trade import validate_traders_copy_trade_config
from services.strategy_versioning import normalize_strategy_version, resolve_strategy_version
from services.runtime_signal_queue import get_queue_depth, publish_signal_batch, wait_for_signal_batch
from services.trader_orchestrator_state import (
    DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS,
    DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE,
    DEFAULT_LIVE_MARKET_CONTEXT,
    DEFAULT_LIVE_PROVIDER_HEALTH,
    DEFAULT_LIVE_RISK_CLAMPS,
    DEFAULT_PENDING_LIVE_EXIT_GUARD,
    ORCHESTRATOR_SNAPSHOT_ID,
    ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
    cleanup_trader_open_orders,
    create_trader_event as _create_trader_event,
    create_trader_order as _create_trader_order,
    get_pending_live_exit_summary_for_trader,
    list_live_wallet_positions_for_trader,
    list_traders,
    reconcile_live_provider_orders,
    read_orchestrator_control,
    read_orchestrator_snapshot,
    set_trader_paused,
    sync_trader_position_inventory,
    update_orchestrator_control,
    update_trader_decision,
    write_orchestrator_snapshot,
)
from services.signal_bus import expire_stale_signals
import services.trader_hot_state as hot_state
from services.ws_feeds import get_feed_manager
from utils.utcnow import utcnow
from utils.converters import coerce_bool as _coerce_bool, parse_iso_datetime, safe_float, safe_int
from utils.logger import get_logger
from utils.secrets import decrypt_secret

logger = get_logger("trader_orchestrator_worker")
strategy_db_loader = strategy_loader

_TRADER_TIMEOUT_CANCEL_GRACE_SECONDS = 5.0
_STRATEGY_EVALUATION_TIMEOUT_SECONDS = 15.0
_abandoned_trader_cycle_tasks: set[asyncio.Task] = set()
_inflight_trader_cycle_tasks: dict[str, asyncio.Task] = {}


def _discard_abandoned_trader_cycle(task: asyncio.Task) -> None:
    _abandoned_trader_cycle_tasks.discard(task)


def _clear_inflight_trader_cycle_task(trader_id: str, task: asyncio.Task) -> None:
    if trader_id and _inflight_trader_cycle_tasks.get(trader_id) is task:
        _inflight_trader_cycle_tasks.pop(trader_id, None)


# ── Hot-path shims ────────────────────────────────────────────────
# These replace the DB-backed originals with write-behind buffer calls.
# Signature matches the call sites so we don't rewrite 50+ callers.

class _DecisionStub:
    def __init__(self, row_id: str):
        self.id = row_id


async def record_signal_consumption(session, *, trader_id, signal_id, outcome, reason=None, decision_id=None, payload=None, commit=True):
    await hot_state.buffer_signal_consumption(
        trader_id=trader_id, signal_id=signal_id, outcome=outcome,
        reason=reason, decision_id=decision_id, payload=payload,
    )


async def upsert_trader_signal_cursor(session, *, trader_id, last_signal_created_at, last_signal_id, commit=True):
    last_runtime_sequence = get_intent_runtime().get_runtime_sequence(str(last_signal_id or ""))
    hot_state.update_signal_cursor(trader_id, "live", last_signal_created_at, last_signal_id, last_runtime_sequence)
    await hot_state.buffer_signal_cursor(
        trader_id=trader_id, last_signal_created_at=last_signal_created_at,
        last_signal_id=last_signal_id,
        last_runtime_sequence=last_runtime_sequence,
    )


async def create_trader_decision(session, *, trader_id, signal, strategy_key, strategy_version=None,
                                  decision, reason=None, score=None, checks_summary=None,
                                  risk_snapshot=None, payload=None, trace_id=None, commit=True):
    row_id = await hot_state.buffer_decision(
        trader_id=trader_id, signal_id=str(signal.id),
        signal_source=str(getattr(signal, "source", "")),
        strategy_key=strategy_key, strategy_version=strategy_version,
        decision=decision, reason=reason, score=score, trace_id=trace_id,
        checks_summary=checks_summary, risk_snapshot=risk_snapshot, payload=payload,
        publish=False,
    )
    return _DecisionStub(row_id)


async def create_trader_decision_checks(session, *, decision_id, checks, commit=True):
    await hot_state.buffer_decision_checks(decision_id=decision_id, checks=checks)


async def set_trade_signal_status(session, *, signal_id, status, commit=True):
    await hot_state.buffer_signal_status(signal_id=signal_id, status=status)
    await get_intent_runtime().update_signal_status(signal_id=str(signal_id or ""), status=str(status or ""))


async def create_trader_event(
    session,
    *,
    event_type,
    severity="info",
    trader_id=None,
    source=None,
    operator=None,
    message=None,
    trace_id=None,
    payload=None,
    commit=True,
):
    del session
    del commit
    return await hot_state.buffer_trader_event(
        event_type=str(event_type or ""),
        severity=str(severity or "info"),
        trader_id=str(trader_id or "") or None,
        source=str(source or "") or None,
        operator=str(operator or "") or None,
        message=message,
        trace_id=str(trace_id or "") or None,
        payload=dict(payload or {}),
    )


async def upsert_strategy_experiment_assignment(
    session,
    *,
    experiment_id,
    trader_id=None,
    signal_id=None,
    source_key,
    strategy_key,
    strategy_version,
    assignment_group,
    decision_id=None,
    order_id=None,
    payload=None,
    commit=True,
):
    del session
    del commit
    await hot_state.buffer_experiment_assignment(
        experiment_id=str(experiment_id or ""),
        trader_id=str(trader_id or "") or None,
        signal_id=str(signal_id or "") or None,
        source_key=str(source_key or ""),
        strategy_key=str(strategy_key or ""),
        strategy_version=int(strategy_version or 1),
        assignment_group=str(assignment_group or "control"),
        decision_id=str(decision_id or "") or None,
        order_id=str(order_id or "") or None,
        payload=dict(payload or {}),
    )


async def get_trader_signal_cursor(session, *, trader_id):
    return hot_state.get_signal_cursor(trader_id, "live")


async def get_trader_signal_sequence_cursor(session, *, trader_id):
    del session
    return hot_state.get_signal_sequence_cursor(trader_id, "live")


async def list_unconsumed_trade_signals(
    session,
    *,
    trader_id,
    sources=None,
    statuses=None,
    strategy_types_by_source=None,
    cursor_runtime_sequence=None,
    cursor_created_at=None,
    cursor_signal_id=None,
    limit=200,
):
    del session
    return await get_intent_runtime().list_unconsumed_signals(
        trader_id=str(trader_id or ""),
        sources=list(sources or []),
        statuses=list(statuses or []),
        strategy_types_by_source=dict(strategy_types_by_source or {}),
        cursor_runtime_sequence=int(cursor_runtime_sequence) if cursor_runtime_sequence is not None else None,
        cursor_created_at=cursor_created_at,
        cursor_signal_id=cursor_signal_id,
        limit=int(limit),
    )


async def get_open_position_count_for_trader(session, trader_id, mode=None, position_cap_scope=None):
    return hot_state.get_open_position_count(trader_id, mode or "live", position_cap_scope=position_cap_scope or "market_direction")


async def get_open_order_count_for_trader(session, trader_id, mode=None):
    return hot_state.get_open_order_count(trader_id, mode or "live")


async def get_open_market_ids_for_trader(session, trader_id, mode=None):
    return hot_state.get_open_market_ids(trader_id, mode or "live")


async def get_daily_realized_pnl(session, *, trader_id=None, mode=None):
    return hot_state.get_daily_realized_pnl(trader_id, mode or "live")


async def get_unrealized_pnl(session, *, trader_id=None, mode=None):
    return await hot_state.get_unrealized_pnl(trader_id, mode or "live")


async def get_consecutive_loss_count(session, *, trader_id, mode=None, limit=100, since=None):
    return hot_state.get_consecutive_loss_count(trader_id, mode or "live")


async def get_last_resolved_loss_at(session, *, trader_id, mode=None, since=None):
    return hot_state.get_last_resolved_loss_at(trader_id, mode or "live")


async def get_gross_exposure(session, mode=None):
    return hot_state.get_gross_exposure(mode or "live")


async def get_market_exposure(session, market_id, mode=None):
    return hot_state.get_market_exposure(market_id, mode or "live")


async def get_trader_source_exposure(session, *, trader_id, source, mode=None):
    return hot_state.get_trader_source_exposure(trader_id, source, mode or "live")


async def get_trader_copy_leader_exposure(session, *, trader_id, source_wallet, mode=None):
    return hot_state.get_copy_leader_exposure(trader_id, source_wallet, mode or "live")
create_trader_order = _create_trader_order
reconcile_paper_positions = reconcile_shadow_positions
_RESUME_POLICIES = {"resume_full", "manage_only", "flatten_then_start"}
_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE = 32
_ORCHESTRATOR_CYCLE_LOCK_KEY = 0x54524F5243485354  # "TRORCHST"
_ORCHESTRATOR_CRYPTO_CYCLE_LOCK_KEY = 0x54524F5243484352  # "TRORCHCR"
_orchestrator_lock_connection: asyncpg.Connection | None = None
_TRADER_IDLE_MAINTENANCE_INTERVAL_SECONDS = 60
_TRADER_LIVE_MAINTENANCE_INTERVAL_SECONDS = 15
_TRADER_SHADOW_MAINTENANCE_INTERVAL_SECONDS = 60
_HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS = 1.0
_STANDARD_MAX_SIGNALS_PER_CYCLE = 500
_STANDARD_DEFAULT_MAX_SIGNALS_PER_CYCLE = 200
_HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE = 5000
_HIGH_FREQUENCY_DEFAULT_MAX_SIGNALS_PER_CYCLE = 2000
_HIGH_FREQUENCY_DEFAULT_SCAN_BATCH_SIZE = 1000
_STANDARD_RUNTIME_TRIGGER_SIGNAL_LIMIT = 8
_HIGH_FREQUENCY_RUNTIME_TRIGGER_SIGNAL_LIMIT = 32
_STANDARD_RUNTIME_TRIGGER_SCAN_BATCH_SIZE = 4
_HIGH_FREQUENCY_RUNTIME_TRIGGER_SCAN_BATCH_SIZE = 16
_trader_idle_maintenance_last_run: dict[str, datetime] = {}
_TRADER_CYCLE_HEARTBEAT_EVENT_INTERVAL_SECONDS = 1
_trader_cycle_heartbeat_last_emitted: dict[str, datetime] = {}
_LANE_GENERAL = "general"
_LANE_CRYPTO = "crypto"
_TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS = 30
_TERMINAL_STALE_ORDER_MIN_AGE_MINUTES = 3
_TERMINAL_STALE_ORDER_ALERT_COOLDOWN_SECONDS = 300
_TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS = 20.0
_TRADER_PENDING_EXIT_SUMMARY_TIMEOUT_SECONDS = 5.0
_WS_FAILURE_PAUSE_THRESHOLD = 10
_ws_auto_paused = False
_LATENCY_SLA_STAGE_KEY = "ws_release_to_submit_start_ms"
_LATENCY_SLA_BREACH_LOG_COOLDOWN_SECONDS = 60.0
_OPEN_ORDER_TIMEOUT_CLEANUP_FAILURE_COOLDOWN_SECONDS = 30
_LIVE_PROVIDER_BLOCK_EVENT_COOLDOWN_SECONDS = 60
_LIVE_RISK_CLAMP_EVENT_COOLDOWN_SECONDS = 300
_CRYPTO_OPEN_ORDER_TIMEOUT_FLOOR_SECONDS = 20.0
_LIVE_PROVIDER_RECONCILE_MIN_INTERVAL_SECONDS = 5.0
_LIVE_PROVIDER_FAILURE_SNAPSHOT_CACHE_TTL_SECONDS = 2.0
_EDGE_CALIBRATION_LOOKBACK_DAYS_DEFAULT = 14
_EDGE_CALIBRATION_MAX_ROWS_DEFAULT = 500
_EDGE_CALIBRATION_MIN_SAMPLES = 24
_EDGE_CALIBRATION_BUCKET_MIN_SAMPLES = 8
_EDGE_CALIBRATION_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 4.0),
    (4.0, 7.0),
    (7.0, 10.0),
    (10.0, 1000.0),
)
_TERMINAL_ORDER_STATUSES = {
    "resolved_win",
    "closed_win",
    "win",
    "resolved_loss",
    "closed_loss",
    "loss",
    "cancelled",
    "failed",
}
_terminal_stale_order_last_checked_at: datetime | None = None
_terminal_stale_order_alert_last_emitted: dict[str, datetime] = {}
_open_order_timeout_cleanup_failure_cooldown_until: dict[str, datetime] = {}
_live_provider_entry_blocked_until: dict[str, datetime] = {}
_live_provider_block_event_cooldown_until: dict[str, datetime] = {}
_live_risk_clamp_event_cooldown_until: dict[str, datetime] = {}
_live_provider_reconcile_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
_live_provider_failure_snapshot_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
_trader_maintenance_last_run: dict[str, datetime] = {}
_latency_sla_breach_logged_at: dict[str, datetime] = {}
_TRADERS_SCOPE_CONTEXT_CACHE_TTL_SECONDS = 5.0
_traders_scope_context_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
_lane_snapshot_metrics: dict[str, dict[str, Any]] = {}
_lane_snapshot_seeded = False


def _empty_lane_snapshot_metrics() -> dict[str, Any]:
    return {
        "traders_total": 0,
        "traders_running": 0,
        "decisions_count": 0,
        "orders_count": 0,
        "execution_sessions_count": 0,
        "active_execution_sessions": 0,
        "open_orders": 0,
        "gross_exposure_usd": 0.0,
        "daily_pnl": 0.0,
    }


def _seed_lane_snapshot_metrics(snapshot: dict[str, Any] | None) -> None:
    global _lane_snapshot_seeded
    if _lane_snapshot_seeded:
        return
    base_snapshot = dict(snapshot or {})
    base_stats = dict(base_snapshot.get("stats") or {})
    seeded_general = _empty_lane_snapshot_metrics()
    seeded_general.update(
        {
            "traders_total": int(base_snapshot.get("traders_total", base_stats.get("traders_total", 0)) or 0),
            "traders_running": int(base_snapshot.get("traders_running", base_stats.get("traders_running", 0)) or 0),
            "decisions_count": int(base_snapshot.get("decisions_count", base_stats.get("decisions_count", 0)) or 0),
            "orders_count": int(base_snapshot.get("orders_count", base_stats.get("orders_count", 0)) or 0),
            "execution_sessions_count": int(
                base_stats.get("execution_sessions_count", base_snapshot.get("execution_sessions_count", 0)) or 0
            ),
            "active_execution_sessions": int(
                base_stats.get("active_execution_sessions", base_snapshot.get("active_execution_sessions", 0)) or 0
            ),
            "open_orders": int(base_snapshot.get("open_orders", base_stats.get("open_orders", 0)) or 0),
            "gross_exposure_usd": float(
                base_snapshot.get("gross_exposure_usd", base_stats.get("gross_exposure_usd", 0.0)) or 0.0
            ),
            "daily_pnl": float(base_snapshot.get("daily_pnl", base_stats.get("daily_pnl", 0.0)) or 0.0),
        }
    )
    _lane_snapshot_metrics[_LANE_GENERAL] = seeded_general
    _lane_snapshot_metrics[_LANE_CRYPTO] = _empty_lane_snapshot_metrics()
    _lane_snapshot_seeded = True


async def _build_orchestrator_snapshot_metrics(
    *,
    session: AsyncSession,
    lane: str,
    traders: list[dict[str, Any]] | None,
    decisions_delta: int = 0,
    orders_delta: int = 0,
) -> dict[str, Any]:
    if not _lane_snapshot_seeded:
        snapshot_seed_payload = await read_orchestrator_snapshot(session)
        snapshot_seed_stats = dict(snapshot_seed_payload.get("stats") or {}) if isinstance(snapshot_seed_payload, dict) else {}
        if not snapshot_seed_stats:
            persisted_row = await session.get(TraderOrchestratorSnapshot, ORCHESTRATOR_SNAPSHOT_ID)
            if persisted_row is not None:
                snapshot_seed_payload = {
                    "traders_total": int(getattr(persisted_row, "traders_total", 0) or 0),
                    "traders_running": int(getattr(persisted_row, "traders_running", 0) or 0),
                    "decisions_count": int(getattr(persisted_row, "decisions_count", 0) or 0),
                    "orders_count": int(getattr(persisted_row, "orders_count", 0) or 0),
                    "open_orders": int(getattr(persisted_row, "open_orders", 0) or 0),
                    "gross_exposure_usd": float(getattr(persisted_row, "gross_exposure_usd", 0.0) or 0.0),
                    "daily_pnl": float(getattr(persisted_row, "daily_pnl", 0.0) or 0.0),
                    "stats": dict(getattr(persisted_row, "stats_json", {}) or {}),
                }
        _seed_lane_snapshot_metrics(snapshot_seed_payload)

    previous_lane_metrics = dict(_lane_snapshot_metrics.get(lane) or _empty_lane_snapshot_metrics())
    lane_metrics = _empty_lane_snapshot_metrics()
    lane_metrics["decisions_count"] = int(previous_lane_metrics.get("decisions_count", 0) or 0) + max(0, decisions_delta)
    lane_metrics["orders_count"] = int(previous_lane_metrics.get("orders_count", 0) or 0) + max(0, orders_delta)
    lane_metrics["execution_sessions_count"] = int(previous_lane_metrics.get("execution_sessions_count", 0) or 0)
    lane_metrics["active_execution_sessions"] = int(previous_lane_metrics.get("active_execution_sessions", 0) or 0)

    normalized_traders = [trader for trader in list(traders or []) if isinstance(trader, dict)]
    lane_metrics["traders_total"] = len(normalized_traders)
    lane_metrics["traders_running"] = sum(
        1
        for trader in normalized_traders
        if bool(trader.get("is_enabled", True)) and not bool(trader.get("is_paused", False))
    )

    open_orders_total = 0
    gross_exposure_total = 0.0
    daily_pnl_total = 0.0
    for trader in normalized_traders:
        trader_id = str(trader.get("id") or "").strip()
        if not trader_id:
            continue
        trader_mode = _canonical_trader_mode(trader.get("mode"), default="shadow")
        open_orders_total += int(hot_state.get_open_order_count(trader_id, trader_mode) or 0)
        gross_exposure_total += float(hot_state.get_trader_gross_exposure(trader_id, trader_mode) or 0.0)
        daily_pnl_total += float(hot_state.get_daily_realized_pnl(trader_id, trader_mode) or 0.0)
    lane_metrics["open_orders"] = open_orders_total
    lane_metrics["gross_exposure_usd"] = gross_exposure_total
    lane_metrics["daily_pnl"] = daily_pnl_total
    _lane_snapshot_metrics[lane] = lane_metrics

    merged = _empty_lane_snapshot_metrics()
    for current_lane_metrics in _lane_snapshot_metrics.values():
        if not isinstance(current_lane_metrics, dict):
            continue
        merged["traders_total"] += int(current_lane_metrics.get("traders_total", 0) or 0)
        merged["traders_running"] += int(current_lane_metrics.get("traders_running", 0) or 0)
        merged["decisions_count"] += int(current_lane_metrics.get("decisions_count", 0) or 0)
        merged["orders_count"] += int(current_lane_metrics.get("orders_count", 0) or 0)
        merged["execution_sessions_count"] += int(current_lane_metrics.get("execution_sessions_count", 0) or 0)
        merged["active_execution_sessions"] += int(current_lane_metrics.get("active_execution_sessions", 0) or 0)
        merged["open_orders"] += int(current_lane_metrics.get("open_orders", 0) or 0)
        merged["gross_exposure_usd"] += float(current_lane_metrics.get("gross_exposure_usd", 0.0) or 0.0)
        merged["daily_pnl"] += float(current_lane_metrics.get("daily_pnl", 0.0) or 0.0)
    merged["execution_latency"] = await execution_latency_metrics.snapshot()
    return merged


def _traders_scope_cache_key(normalized_scope: dict[str, Any]) -> str:
    return json.dumps(normalized_scope, sort_keys=True, separators=(",", ":"))


def _prune_module_caches(active_trader_ids: set[str]) -> None:
    """Remove entries from module-level dicts for traders no longer active.

    Called once per orchestrator cycle after the trader list is loaded.
    Without this, the dicts grow per-trader/per-order across the lifetime
    of the process and never shrink — a slow memory leak.
    """
    for trader_id in list(_trader_idle_maintenance_last_run):
        if trader_id not in active_trader_ids:
            _trader_idle_maintenance_last_run.pop(trader_id, None)

    for trader_id in list(_trader_maintenance_last_run):
        if trader_id not in active_trader_ids:
            _trader_maintenance_last_run.pop(trader_id, None)

    for trader_id in list(_trader_cycle_heartbeat_last_emitted):
        if trader_id not in active_trader_ids:
            _trader_cycle_heartbeat_last_emitted.pop(trader_id, None)

    for trader_id in list(_live_provider_entry_blocked_until):
        if trader_id not in active_trader_ids:
            _live_provider_entry_blocked_until.pop(trader_id, None)

    for trader_id in list(_live_provider_block_event_cooldown_until):
        if trader_id not in active_trader_ids:
            _live_provider_block_event_cooldown_until.pop(trader_id, None)

    for trader_id in list(_live_risk_clamp_event_cooldown_until):
        if trader_id not in active_trader_ids:
            _live_risk_clamp_event_cooldown_until.pop(trader_id, None)

    for trader_id in list(_live_provider_reconcile_cache):
        if trader_id not in active_trader_ids:
            _live_provider_reconcile_cache.pop(trader_id, None)

    for key in list(_live_provider_failure_snapshot_cache):
        prefix = key.split(":", 1)[0]
        if prefix not in active_trader_ids:
            _live_provider_failure_snapshot_cache.pop(key, None)

    # _open_order_timeout_cleanup_failure_cooldown_until is keyed by
    # "trader_id:mode:source" — prune entries whose trader_id prefix is gone.
    for key in list(_open_order_timeout_cleanup_failure_cooldown_until):
        prefix = key.split(":", 1)[0]
        if prefix not in active_trader_ids:
            _open_order_timeout_cleanup_failure_cooldown_until.pop(key, None)

    # _terminal_stale_order_alert_last_emitted is keyed by order_id.
    # It already has a 24h expiry in the stale-order check path, but if the
    # check never runs (no stale orders), old entries accumulate.  Cap size.
    if len(_terminal_stale_order_alert_last_emitted) > 5000:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        for order_id, emitted_at in list(_terminal_stale_order_alert_last_emitted.items()):
            if emitted_at < cutoff:
                _terminal_stale_order_alert_last_emitted.pop(order_id, None)


_LIVE_PROVIDER_INFRA_ERROR_MARKERS = (
    "connection refused",
    "database system is not yet accepting connections",
    "dial tcp",
    "i/o timeout",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection closed",
    "transport error",
    "read: connection",
    "write: broken pipe",
    "upstream connect error",
    "network is unreachable",
)

_CANONICAL_TRADER_MODES = {"shadow", "live"}
_LEGACY_MODE_ALIASES = {
    "paper": "shadow",
}
_LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES: dict[str, str] = {
    "crypto_5m": "5m",
    "crypto_15m": "15m",
    "crypto_1h": "1h",
    "crypto_4h": "4h",
}
_LEGACY_STRATEGY_ALIASES: dict[str, str] = {
    **{legacy: "btc_eth_highfreq" for legacy in _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES.keys()},
    "news_reaction": "news_edge",
}


def _canonical_trader_mode(value: Any, *, default: str = "shadow") -> str:
    key = str(value or "").strip().lower()
    if not key:
        key = default
    key = _LEGACY_MODE_ALIASES.get(key, key)
    if key not in _CANONICAL_TRADER_MODES:
        return default
    return key


def _normalize_timeframe_scope(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if token in {"5m", "5min", "5"}:
            token = "5m"
        elif token in {"15m", "15min", "15"}:
            token = "15m"
        elif token in {"1h", "1hr", "60m", "60min"}:
            token = "1h"
        elif token in {"4h", "4hr", "240m", "240min"}:
            token = "4h"
        else:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalize_strategy_for_source(
    *,
    source_key: str,
    strategy_key: str,
    strategy_params: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    requested_key = str(strategy_key or "").strip().lower()
    params = dict(strategy_params or {})
    if not requested_key:
        return "", params, ""
    canonical_key = _LEGACY_STRATEGY_ALIASES.get(requested_key, requested_key)

    if source_key == "crypto" and requested_key in _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES:
        timeframe = _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES[requested_key]
        include_timeframes = _normalize_timeframe_scope(params.get("include_timeframes"))
        params["include_timeframes"] = [timeframe] if not include_timeframes else include_timeframes
        exclude_timeframes = [token for token in _normalize_timeframe_scope(params.get("exclude_timeframes")) if token != timeframe]
        if exclude_timeframes:
            params["exclude_timeframes"] = exclude_timeframes
        elif "exclude_timeframes" in params:
            params.pop("exclude_timeframes", None)
        params.setdefault("enable_live_market_context", False)
        params.setdefault("require_live_market_revalidation", False)
        params.setdefault("require_live_revalidation_for_sources", [])
        params.setdefault("enforce_market_data_freshness", False)
        params.setdefault("require_market_data_age_for_sources", [])

    return canonical_key, params, requested_key


async def _worker_sleep(interval_seconds: float) -> None:
    await asyncio.sleep(interval_seconds)


def _normalize_snapshot_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return _parse_iso(str(value or "").strip())


def _coerce_runtime_signal_snapshot(snapshot: dict[str, Any]) -> Any:
    payload_json = snapshot.get("payload_json")
    strategy_context_json = snapshot.get("strategy_context_json")
    expires_at = _normalize_snapshot_datetime(snapshot.get("expires_at"))
    created_at = _normalize_snapshot_datetime(snapshot.get("created_at"))
    updated_at = _normalize_snapshot_datetime(snapshot.get("updated_at"))
    return SimpleNamespace(
        id=str(snapshot.get("id") or "").strip(),
        source=str(snapshot.get("source") or "").strip(),
        source_item_id=str(snapshot.get("source_item_id") or "").strip(),
        signal_type=str(snapshot.get("signal_type") or "").strip(),
        strategy_type=str(snapshot.get("strategy_type") or "").strip(),
        market_id=str(snapshot.get("market_id") or "").strip(),
        market_question=str(snapshot.get("market_question") or "").strip(),
        direction=str(snapshot.get("direction") or "").strip(),
        entry_price=safe_float(snapshot.get("entry_price"), 0.0) or 0.0,
        effective_price=safe_float(snapshot.get("effective_price"), None),
        edge_percent=safe_float(snapshot.get("edge_percent"), 0.0) or 0.0,
        confidence=safe_float(snapshot.get("confidence"), 0.0) or 0.0,
        liquidity=safe_float(snapshot.get("liquidity"), 0.0) or 0.0,
        expires_at=expires_at,
        status=str(snapshot.get("status") or "pending").strip().lower(),
        payload_json=dict(payload_json or {}) if isinstance(payload_json, dict) else {},
        strategy_context_json=(
            dict(strategy_context_json or {}) if isinstance(strategy_context_json, dict) else {}
        ),
        quality_passed=snapshot.get("quality_passed"),
        dedupe_key=str(snapshot.get("dedupe_key") or "").strip(),
        runtime_sequence=safe_int(snapshot.get("runtime_sequence"), None),
        required_token_ids=[
            str(token_id).strip().lower()
            for token_id in (snapshot.get("required_token_ids") or [])
            if str(token_id).strip()
        ],
        deferred_until_ws=bool(snapshot.get("deferred_until_ws")),
        created_at=created_at,
        updated_at=updated_at,
    )


def _runtime_snapshot_matches_trader_source(
    snapshot: dict[str, Any],
    *,
    target_source: str,
    allowed_strategy_types: set[str],
) -> bool:
    snapshot_source = normalize_source_key(snapshot.get("source"))
    if snapshot_source and snapshot_source != target_source:
        return False
    strategy_type = str(snapshot.get("strategy_type") or "").strip().lower()
    if allowed_strategy_types and strategy_type and strategy_type not in allowed_strategy_types:
        return False
    return True


def _trigger_signal_snapshots_for_trader(
    trader: dict[str, Any],
    trigger_event: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]] | None:
    if not trigger_event:
        return None
    raw = trigger_event.get("source_signal_snapshots")
    if not isinstance(raw, dict) or not raw:
        return None

    source_configs = _normalize_source_configs(trader)
    trader_sources = set(_query_sources_for_configs(source_configs))
    if not trader_sources:
        return None
    strategy_types_by_source = _query_strategy_types_for_configs(source_configs)

    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_source_key, raw_snapshots in raw.items():
        source_key = normalize_source_key(raw_source_key)
        if source_key == "__all__":
            target_sources = sorted(trader_sources)
        elif source_key and source_key in trader_sources:
            target_sources = [source_key]
        else:
            continue
        if not isinstance(raw_snapshots, dict):
            continue
        normalized_snapshots = {
            str(signal_id).strip(): dict(snapshot)
            for signal_id, snapshot in raw_snapshots.items()
            if str(signal_id).strip() and isinstance(snapshot, dict)
        }
        if not normalized_snapshots:
            continue
        for target_source in target_sources:
            allowed_strategy_types = set(strategy_types_by_source.get(target_source) or [])
            for signal_id, snapshot in normalized_snapshots.items():
                if not _runtime_snapshot_matches_trader_source(
                    snapshot,
                    target_source=target_source,
                    allowed_strategy_types=allowed_strategy_types,
                ):
                    continue
                existing = filtered.setdefault(target_source, {})
                existing.setdefault(signal_id, snapshot)
    filtered = {source: snapshots for source, snapshots in filtered.items() if snapshots}
    return filtered or None


async def _build_triggered_trade_signals(
    session: Any,
    *,
    trader_id: str,
    signal_ids_by_source: dict[str, list[str]],
    signal_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] | None,
    sources: list[str],
    strategy_types_by_source: dict[str, list[str]],
    cursor_runtime_sequence: int | None,
    cursor_created_at: datetime | None,
    cursor_signal_id: str | None,
    statuses: list[str],
    limit: int,
) -> list[Any]:
    if not signal_ids_by_source or not sources:
        return []

    normalized_statuses = {str(status or "").strip().lower() for status in statuses if str(status or "").strip()}
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    ordered_sources = ["__all__", *sources]
    for source_key in ordered_sources:
        for raw_signal_id in signal_ids_by_source.get(source_key) or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen_ids:
                continue
            seen_ids.add(signal_id)
            ordered_ids.append(signal_id)
    if not ordered_ids:
        return []

    snapshots = signal_snapshots_by_source or {}
    rows: list[Any] = []
    source_order = [source for source in sources if source in signal_ids_by_source]
    if "__all__" in signal_ids_by_source:
        source_order = ["__all__", *source_order]
    seen_row_ids: set[str] = set()
    for source_key in source_order:
        for raw_signal_id in signal_ids_by_source.get(source_key) or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen_row_ids:
                continue
            snapshot = (snapshots.get(source_key) or {}).get(signal_id)
            if snapshot is None and source_key != "__all__":
                snapshot = (snapshots.get("__all__") or {}).get(signal_id)
            if not isinstance(snapshot, dict):
                continue
            row = _coerce_runtime_signal_snapshot(snapshot)
            if normalized_statuses and str(getattr(row, "status", "") or "").strip().lower() not in normalized_statuses:
                continue
            if cursor_runtime_sequence is not None:
                row_runtime_sequence = _signal_runtime_sequence(row)
                if row_runtime_sequence is None or row_runtime_sequence <= int(cursor_runtime_sequence):
                    continue
            seen_row_ids.add(signal_id)
            rows.append(row)

    rows.sort(key=_signal_sort_key)
    return rows[: max(1, min(limit, 5000))]


async def _wait_for_runtime_trigger(
    _trigger_queue: Any,
    timeout_seconds: float,
    *,
    consumer: str,
    group: str,
    claim_cursor: str,
    last_claim_run_at: datetime | None,
) -> tuple[dict[str, Any] | None, str, datetime]:
    trigger = await wait_for_signal_batch(
        lane=str(group or _LANE_GENERAL).rsplit(":", 1)[-1],
        timeout_seconds=timeout_seconds,
    )
    last_claim = utcnow()
    return trigger, claim_cursor, last_claim


def _session_dialect_name(session: Any) -> str:
    try:
        bind = session.get_bind()
        return str(getattr(getattr(bind, "dialect", None), "name", "") or "").lower()
    except (AttributeError, RuntimeError, TypeError) as exc:
        logger.debug("Unable to determine session dialect name", exc_info=exc)
        return ""


async def _write_orchestrator_snapshot_best_effort(session: Any, **snapshot_kwargs: Any) -> None:
    try:
        await write_orchestrator_snapshot(session, **snapshot_kwargs)
    except OperationalError as exc:
        if not _is_retryable_db_error(exc):
            raise
        if hasattr(session, "rollback"):
            await session.rollback()
        logger.warning("Skipped orchestrator snapshot write due to transient DB error")


async def submit_order(
    *,
    session_engine: ExecutionSessionEngine,
    trader_id: str,
    signal: RuntimeTradeSignalView,
    decision_id: str,
    strategy_key: str,
    strategy_version: int | None,
    strategy_params: dict[str, Any],
    risk_limits: dict[str, Any],
    mode: str,
    size_usd: float,
    reason: str,
):
    return await session_engine.execute_signal(
        trader_id=trader_id,
        signal=signal,
        decision_id=decision_id,
        strategy_key=strategy_key,
        strategy_version=strategy_version,
        strategy_params=strategy_params,
        risk_limits=risk_limits,
        mode=mode,
        size_usd=size_usd,
        reason=reason,
    )


def _enforce_strict_ws_strategy_params(
    strategy_params: dict[str, Any],
    *,
    strict_age_budget_ms: int,
    strict_ws_price_sources: list[str],
) -> dict[str, Any]:
    strategy_params["require_strict_ws_pricing"] = True
    strategy_params.setdefault("strict_ws_price_sources", list(strict_ws_price_sources))
    strategy_params["max_market_data_age_ms"] = max(
        int(strict_age_budget_ms),
        safe_int(strategy_params.get("max_market_data_age_ms"), int(strict_age_budget_ms)),
    )
    return strategy_params


def _parse_iso(ts: str | None) -> datetime | None:
    return parse_iso_datetime(ts, naive=False)


def _strategy_evaluation_timeout_seconds(
    control: dict[str, Any] | None,
    source_config: dict[str, Any] | None,
    strategy_params: dict[str, Any] | None,
) -> float:
    for raw_value in (
        (strategy_params or {}).get("strategy_evaluation_timeout_seconds"),
        (strategy_params or {}).get("evaluation_timeout_seconds"),
        (source_config or {}).get("strategy_evaluation_timeout_seconds"),
        (source_config or {}).get("evaluation_timeout_seconds"),
        (control or {}).get("strategy_evaluation_timeout_seconds"),
        getattr(settings, "TRADER_STRATEGY_EVALUATION_TIMEOUT_SECONDS", None),
    ):
        parsed = safe_float(raw_value, None, reject_nan_inf=True)
        if parsed is not None and parsed > 0.0:
            return max(0.1, float(parsed))
    return _STRATEGY_EVALUATION_TIMEOUT_SECONDS


def _compute_signal_latency_payload(
    signal: Any,
    *,
    live_context: dict[str, Any] | None = None,
    measured_at: datetime | None = None,
) -> dict[str, Any]:
    now = measured_at or utcnow()
    payload = getattr(signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    context = live_context if isinstance(live_context, dict) else {}
    execution_armed_at = _parse_iso(str(payload.get("execution_armed_at") or ""))
    source_observed_at = _parse_iso(
        str(
            context.get("source_observed_at")
            or payload.get("source_observed_at")
            or payload.get("live_market_fetched_at")
            or payload.get("last_priced_at")
            or ""
        )
    )
    emitted_at = _parse_iso(str(payload.get("signal_emitted_at") or payload.get("ingested_at") or ""))
    ingested_at = _parse_iso(str(payload.get("ingested_at") or ""))

    out: dict[str, Any] = {
        "measured_at": now.isoformat(),
        "execution_armed_at": execution_armed_at.isoformat() if execution_armed_at is not None else None,
        "source_observed_at": source_observed_at.isoformat() if source_observed_at is not None else None,
        "signal_emitted_at": emitted_at.isoformat() if emitted_at is not None else None,
        "ingested_at": ingested_at.isoformat() if ingested_at is not None else None,
        "market_data_age_ms": context.get("market_data_age_ms", payload.get("market_data_age_ms")),
    }
    if execution_armed_at is not None:
        out["armed_to_now_ms"] = max(0, int((now - execution_armed_at).total_seconds() * 1000))
    if source_observed_at is not None:
        out["source_to_now_ms"] = max(0, int((now - source_observed_at).total_seconds() * 1000))
    if emitted_at is not None:
        out["emit_to_now_ms"] = max(0, int((now - emitted_at).total_seconds() * 1000))
        out["ws_release_to_now_ms"] = max(0, int((now - emitted_at).total_seconds() * 1000))
    if ingested_at is not None:
        out["ingest_to_now_ms"] = max(0, int((now - ingested_at).total_seconds() * 1000))
    return out


def _build_execution_latency_sample(
    signal: Any,
    *,
    wake_started_at: datetime,
    context_ready_at: datetime,
    decision_ready_at: datetime,
    submit_started_at: datetime | None = None,
    submit_ack_at: datetime | None = None,
) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    armed_at = _parse_iso(str(payload.get("execution_armed_at") or payload.get("signal_emitted_at") or payload.get("ingested_at") or ""))
    emitted_at = _parse_iso(str(payload.get("signal_emitted_at") or payload.get("ingested_at") or ""))

    def _delta_ms(start: datetime | None, end: datetime | None) -> int | None:
        if start is None or end is None:
            return None
        return max(0, int((end - start).total_seconds() * 1000))

    sample = {
        "armed_to_ws_release_ms": _delta_ms(armed_at, emitted_at),
        "emit_to_queue_wake_ms": _delta_ms(emitted_at, wake_started_at),
        "ws_release_to_decision_ms": _delta_ms(emitted_at, decision_ready_at),
        "ws_release_to_submit_start_ms": _delta_ms(emitted_at, submit_started_at),
        "wake_to_context_ready_ms": _delta_ms(wake_started_at, context_ready_at),
        "context_ready_to_decision_ms": _delta_ms(context_ready_at, decision_ready_at),
        "decision_to_submit_start_ms": _delta_ms(decision_ready_at, submit_started_at),
        "submit_start_to_provider_ack_ms": _delta_ms(submit_started_at, submit_ack_at),
        "emit_to_submit_start_ms": _delta_ms(emitted_at, submit_started_at),
    }
    return sample


def _coerce_optional_bool(value: Any) -> bool | None:
    return _coerce_bool(value, None)


def _normalize_strategy_type_values(raw: Any) -> set[str]:
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = []
    normalized: set[str] = set()
    for item in values:
        value = str(item or "").strip().lower()
        if value:
            normalized.add(value)
    return normalized


def _strategy_instance_for_source_config(source_config: dict[str, Any] | None) -> Any | None:
    if not isinstance(source_config, dict):
        return None
    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
    if not strategy_key:
        return None
    loaded = strategy_loader.get_strategy(strategy_key)
    if loaded is None:
        try:
            return resolve_strategy_instance(strategy_key)
        except Exception as exc:
            logger.debug("Failed to resolve strategy instance for source config", strategy_key=strategy_key, exc_info=exc)
            return None
    instance = getattr(loaded, "instance", None)
    if instance is not None:
        return instance
    return loaded


def _merged_strategy_params_for_source_config(source_config: dict[str, Any]) -> dict[str, Any]:
    source_key = normalize_source_key(source_config.get("source_key"))
    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
    explicit_params = dict(source_config.get("strategy_params") or {})
    strategy_defaults: dict[str, Any] = {}
    strategy_version = normalize_strategy_version(source_config.get("strategy_version"))

    if strategy_version is None:
        strategy_instance = _strategy_instance_for_source_config(source_config)
        if strategy_instance is not None:
            configured_defaults = getattr(strategy_instance, "config", None)
            if isinstance(configured_defaults, dict):
                strategy_defaults = dict(configured_defaults)
            else:
                declared_defaults = getattr(strategy_instance, "default_config", None)
                if isinstance(declared_defaults, dict):
                    strategy_defaults = dict(declared_defaults)

    merged = {**strategy_defaults, **explicit_params}

    if source_key == "traders":
        if strategy_key == "traders_copy_trade":
            merged = validate_traders_copy_trade_config(merged)
        else:
            merged = StrategySDK.validate_trader_filter_config(merged)
    elif source_key == "news":
        merged = validate_news_edge_config(merged)

    return StrategySDK.normalize_strategy_retention_config(merged)


def _accepted_signal_strategy_types(source_config: dict[str, Any]) -> set[str]:
    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
    requested_strategy_key = str(source_config.get("requested_strategy_key") or "").strip().lower()
    if not strategy_key:
        return set()

    allowed = {strategy_key}
    if requested_strategy_key:
        allowed.add(requested_strategy_key)
    strategy_params = _merged_strategy_params_for_source_config(source_config)
    allowed.update(_normalize_strategy_type_values(strategy_params.get("accepted_signal_strategy_types")))

    strategy_instance = _strategy_instance_for_source_config(source_config)
    if strategy_instance is not None:
        allowed.update(
            _normalize_strategy_type_values(getattr(strategy_instance, "accepted_signal_strategy_types", None))
        )

    return allowed


def _supports_live_market_context(
    signal: Any,
    source_config: dict[str, Any] | None = None,
) -> bool:
    """Live-market enrichment is always enabled in the runtime."""
    del signal
    del source_config
    return True


def _is_exit_only_strategy_instance(strategy_instance: Any) -> bool:
    if strategy_instance is None:
        return False
    strategy_cls = strategy_instance.__class__
    class_dict = getattr(strategy_cls, "__dict__", {})
    has_custom_detect = any(name in class_dict for name in ("detect", "detect_async", "on_event"))
    has_custom_evaluate = "evaluate" in class_dict
    has_custom_should_exit = "should_exit" in class_dict
    return bool(has_custom_should_exit and not has_custom_detect and not has_custom_evaluate)


def _source_config_allows_new_entries(source_config: dict[str, Any] | None) -> bool:
    if not isinstance(source_config, dict):
        return True

    strategy_params = _merged_strategy_params_for_source_config(source_config)
    explicit_allow = _coerce_optional_bool(strategy_params.get("allow_new_entries"))
    if explicit_allow is not None:
        return bool(explicit_allow)
    explicit_disable = _coerce_optional_bool(strategy_params.get("disable_new_entries"))
    if explicit_disable is not None:
        return not bool(explicit_disable)

    strategy_instance = _strategy_instance_for_source_config(source_config)
    if strategy_instance is None:
        return True

    declared_allow = _coerce_optional_bool(getattr(strategy_instance, "allow_new_entries", None))
    if declared_allow is not None:
        return bool(declared_allow)
    declared_disable = _coerce_optional_bool(getattr(strategy_instance, "disable_new_entries", None))
    if declared_disable is not None:
        return not bool(declared_disable)

    if _is_exit_only_strategy_instance(strategy_instance):
        return False

    return True


def _normalize_resume_policy(value: Any) -> str:
    policy = str(value or "").strip().lower()
    if policy in _RESUME_POLICIES:
        return policy
    return "resume_full"


def _is_due(trader: dict[str, Any], now: datetime) -> bool:
    requested = _parse_iso(trader.get("requested_run_at"))
    if requested is not None:
        return True

    last_run = _parse_iso(trader.get("last_run_at"))
    interval = max(1, int(trader.get("interval_seconds") or 60))
    if last_run is None:
        return True
    return (now - last_run.astimezone(timezone.utc)).total_seconds() >= interval


def _is_high_frequency_maintenance_due(trader: dict[str, Any], now: datetime) -> bool:
    last_run = _parse_iso(trader.get("last_run_at"))
    if last_run is None:
        return True
    elapsed_seconds = (now - last_run.astimezone(timezone.utc)).total_seconds()
    return elapsed_seconds >= _HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS


def _trader_maintenance_interval_seconds(run_mode: str) -> int:
    if run_mode == "live":
        return _TRADER_LIVE_MAINTENANCE_INTERVAL_SECONDS
    return _TRADER_SHADOW_MAINTENANCE_INTERVAL_SECONDS


def _is_trader_maintenance_due(
    *,
    trader_id: str,
    run_mode: str,
    now: datetime,
) -> bool:
    if not trader_id:
        return False
    last_run = _trader_maintenance_last_run.get(trader_id)
    if last_run is None:
        return True
    interval_seconds = _trader_maintenance_interval_seconds(run_mode)
    return (now - last_run).total_seconds() >= float(interval_seconds)


def _is_high_frequency_crypto_trader(trader: dict[str, Any]) -> bool:
    source_configs = _normalize_source_configs(trader)
    if not source_configs:
        return False
    return "crypto" in _query_sources_for_configs(source_configs)


def _is_crypto_source_trader(trader: dict[str, Any]) -> bool:
    return _is_high_frequency_crypto_trader(trader)


def _trader_matches_lane(trader: dict[str, Any], lane: str) -> bool:
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    is_crypto = _is_crypto_source_trader(trader)
    if lane_key == _LANE_CRYPTO:
        return is_crypto
    return not is_crypto


def _stream_group_name_for_lane(lane: str) -> str:
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    return lane_key


def _cycle_lock_key_for_lane(lane: str) -> int:
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    if lane_key == _LANE_CRYPTO:
        return int(_ORCHESTRATOR_CRYPTO_CYCLE_LOCK_KEY)
    return int(_ORCHESTRATOR_CYCLE_LOCK_KEY)


def _runtime_trigger_matches_trader(
    trader: dict[str, Any],
    trigger_event: dict[str, Any] | None,
) -> bool:
    if not trigger_event:
        return False
    source_signal_ids = trigger_event.get("source_signal_ids")
    if isinstance(source_signal_ids, dict):
        return bool(_trigger_signal_ids_for_trader(trader, trigger_event))
    source = str(trigger_event.get("source") or "").strip().lower()
    if not source:
        return True
    source_configs = _normalize_source_configs(trader)
    if not source_configs:
        return False
    return source in _query_sources_for_configs(source_configs)


def _trigger_signal_ids_for_trader(
    trader: dict[str, Any],
    trigger_event: dict[str, Any] | None,
) -> dict[str, list[str]] | None:
    if not trigger_event:
        return None
    raw = trigger_event.get("source_signal_ids")
    if not isinstance(raw, dict) or not raw:
        return None

    source_configs = _normalize_source_configs(trader)
    trader_sources = set(_query_sources_for_configs(source_configs))
    if not trader_sources:
        return None
    strategy_types_by_source = _query_strategy_types_for_configs(source_configs)
    raw_snapshots = trigger_event.get("source_signal_snapshots")

    filtered: dict[str, list[str]] = {}
    seen_by_source: dict[str, set[str]] = {}
    global_ids: list[str] = []
    global_seen: set[str] = set()
    for source_key, signal_ids in raw.items():
        source_token = str(source_key or "").strip().lower()
        normalized_source = normalize_source_key(source_token)
        if not isinstance(signal_ids, list):
            continue
        if source_token == "__all__":
            if isinstance(raw_snapshots, dict):
                raw_source_snapshots = raw_snapshots.get(source_token)
                snapshot_map = raw_source_snapshots if isinstance(raw_source_snapshots, dict) else {}
                for target_source in sorted(trader_sources):
                    allowed_strategy_types = set(strategy_types_by_source.get(target_source) or [])
                    existing = filtered.setdefault(target_source, [])
                    seen = seen_by_source.setdefault(target_source, set())
                    for raw_signal_id in signal_ids:
                        signal_id = str(raw_signal_id or "").strip()
                        if not signal_id or signal_id in seen:
                            continue
                        snapshot = snapshot_map.get(signal_id)
                        if isinstance(snapshot, dict) and not _runtime_snapshot_matches_trader_source(
                            snapshot,
                            target_source=target_source,
                            allowed_strategy_types=allowed_strategy_types,
                        ):
                            continue
                        seen.add(signal_id)
                        existing.append(signal_id)
                continue
            for raw_signal_id in signal_ids:
                signal_id = str(raw_signal_id or "").strip()
                if not signal_id or signal_id in global_seen:
                    continue
                global_seen.add(signal_id)
                global_ids.append(signal_id)
            continue
        if not normalized_source or normalized_source not in trader_sources:
            continue
        raw_source_snapshots = raw_snapshots.get(source_token) if isinstance(raw_snapshots, dict) else None
        snapshot_map = raw_source_snapshots if isinstance(raw_source_snapshots, dict) else {}
        allowed_strategy_types = set(strategy_types_by_source.get(normalized_source) or [])
        seen = seen_by_source.setdefault(normalized_source, set())
        out = filtered.setdefault(normalized_source, [])
        for raw_signal_id in signal_ids:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen:
                continue
            snapshot = snapshot_map.get(signal_id)
            if isinstance(snapshot, dict) and not _runtime_snapshot_matches_trader_source(
                snapshot,
                target_source=normalized_source,
                allowed_strategy_types=allowed_strategy_types,
            ):
                continue
            seen.add(signal_id)
            out.append(signal_id)
    filtered = {source: ids for source, ids in filtered.items() if ids}
    if global_ids:
        filtered["__all__"] = global_ids
    if not filtered:
        return None
    return filtered


def _checks_to_payload(checks: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for check in checks:
        out.append(
            {
                "check_key": str(getattr(check, "key", "check")),
                "check_label": str(getattr(check, "label", "Check")),
                "passed": bool(getattr(check, "passed", False)),
                "score": getattr(check, "score", None),
                "detail": getattr(check, "detail", None),
                "payload": getattr(check, "payload", {}) or {},
            }
        )
    return out


def _is_live_credentials_configured(app_settings: AppSettings | None) -> bool:
    if app_settings is None:
        return False

    polymarket_ready = bool(
        decrypt_secret(app_settings.polymarket_api_key)
        and decrypt_secret(app_settings.polymarket_api_secret)
        and decrypt_secret(app_settings.polymarket_api_passphrase)
    )
    kalshi_ready = bool(
        (app_settings.kalshi_email or "").strip()
        and decrypt_secret(app_settings.kalshi_password)
        and decrypt_secret(app_settings.kalshi_api_key)
    )
    return polymarket_ready or kalshi_ready


def _normalize_wallet(value: Any) -> str:
    return StrategySDK.normalize_trader_wallet(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_source_configs(trader: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_configs_raw = trader.get("source_configs")
    if not isinstance(source_configs_raw, list):
        source_configs_raw = []

    normalized: dict[str, dict[str, Any]] = {}
    for raw in source_configs_raw:
        if not isinstance(raw, dict):
            continue
        source_key = normalize_source_key(raw.get("source_key"))
        raw_strategy_key = str(raw.get("strategy_key") or "").strip().lower()
        strategy_version = normalize_strategy_version(raw.get("strategy_version"))
        strategy_key, strategy_params, requested_strategy_key = _normalize_strategy_for_source(
            source_key=source_key,
            strategy_key=raw_strategy_key,
            strategy_params=dict(raw.get("strategy_params") or {}),
        )
        if not source_key or not strategy_key:
            continue
        source_config = {
            "source_key": source_key,
            "strategy_key": strategy_key,
            "requested_strategy_key": requested_strategy_key,
            "strategy_version": strategy_version,
            "strategy_params": strategy_params,
        }
        strategy_params = _merged_strategy_params_for_source_config(source_config)
        normalized[source_key] = {
            "source_key": source_key,
            "strategy_key": strategy_key,
            "requested_strategy_key": requested_strategy_key,
            "strategy_version": strategy_version,
            "strategy_params": strategy_params,
        }
    return normalized


def _source_open_order_timeout_seconds(source_config: dict[str, Any]) -> float | None:
    source_key = normalize_source_key(source_config.get("source_key"))
    strategy_params = dict(source_config.get("strategy_params") or {})
    explicit_timeout_seconds = StrategySDK.resolve_open_order_timeout_seconds(
        strategy_params,
        default_seconds=None,
    )
    timeout_explicit = explicit_timeout_seconds is not None
    strategy_instance = _strategy_instance_for_source_config(source_config)
    default_seconds = None
    if strategy_instance is not None:
        declared_default = safe_float(getattr(strategy_instance, "default_open_order_timeout_seconds", None), None)
        if declared_default is not None and declared_default > 0.0:
            default_seconds = float(declared_default)
        else:
            strategy_defaults = getattr(strategy_instance, "default_config", None)
            if isinstance(strategy_defaults, dict):
                default_seconds = StrategySDK.resolve_open_order_timeout_seconds(
                    strategy_defaults,
                    default_seconds=None,
                )
    timeout_seconds = StrategySDK.resolve_open_order_timeout_seconds(
        strategy_params,
        default_seconds=default_seconds,
    )
    if timeout_seconds is None:
        return None
    if source_key == "crypto":
        configured_floor = safe_float(strategy_params.get("min_open_order_timeout_seconds"), None)
        if configured_floor is not None:
            timeout_seconds = max(float(timeout_seconds), max(1.0, float(configured_floor)))
        elif not timeout_explicit:
            timeout_seconds = max(float(timeout_seconds), _CRYPTO_OPEN_ORDER_TIMEOUT_FLOOR_SECONDS)
    return float(timeout_seconds)


def _source_timeout_taker_rescue_policy(source_config: dict[str, Any]) -> dict[str, Any]:
    source_key = normalize_source_key(source_config.get("source_key"))
    strategy_params = dict(source_config.get("strategy_params") or {})
    enabled_default = source_key == "crypto"
    enabled = bool(
        strategy_params.get(
            "timeout_taker_rescue_enabled",
            strategy_params.get("attempt_live_taker_rescue", enabled_default),
        )
    )
    price_bps = safe_float(
        strategy_params.get(
            "timeout_taker_rescue_price_bps",
            strategy_params.get("live_taker_rescue_price_bps"),
        ),
        DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS,
    )
    if price_bps is None:
        price_bps = DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS
    time_in_force = str(
        strategy_params.get(
            "timeout_taker_rescue_time_in_force",
            strategy_params.get("live_taker_rescue_time_in_force", DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE),
        )
        or DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE
    ).strip().upper()
    if time_in_force not in {"IOC", "FOK", "GTC"}:
        time_in_force = DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE
    return {
        "enabled": enabled,
        "price_bps": max(0.0, float(price_bps)),
        "time_in_force": time_in_force,
    }


def _edge_bucket_key(min_edge: float, max_edge: float) -> str:
    if max_edge >= 999.0:
        return f"{int(min_edge)}+"
    return f"{int(min_edge)}-{int(max_edge)}"


def _resolve_shadow_account_id(control: dict[str, Any], trader: dict[str, Any]) -> str:
    settings_payload = control.get("settings")
    settings_payload = settings_payload if isinstance(settings_payload, dict) else {}
    metadata_payload = trader.get("metadata")
    metadata_payload = metadata_payload if isinstance(metadata_payload, dict) else {}
    candidates = (
        settings_payload.get("shadow_account_id"),
        settings_payload.get("paper_account_id"),
        metadata_payload.get("shadow_account_id"),
        metadata_payload.get("paper_account_id"),
    )
    for candidate in candidates:
        account_id = str(candidate or "").strip()
        if account_id:
            return account_id
    return ""


def _shadow_ledger_token_id(payload: dict[str, Any], direction: str) -> str:
    leg_payload = payload.get("leg")
    leg_payload = leg_payload if isinstance(leg_payload, dict) else {}
    direction_key = str(direction or "").strip().lower()
    if "yes" in direction_key:
        candidates = (
            payload.get("token_id"),
            payload.get("yes_token_id"),
            payload.get("selected_token_id"),
            leg_payload.get("token_id"),
        )
    elif "no" in direction_key:
        candidates = (
            payload.get("token_id"),
            payload.get("no_token_id"),
            payload.get("selected_token_id"),
            leg_payload.get("token_id"),
        )
    else:
        candidates = (
            payload.get("token_id"),
            payload.get("selected_token_id"),
            leg_payload.get("token_id"),
        )
    for raw in candidates:
        token = str(raw or "").strip()
        if token:
            return token
    return ""


async def _backfill_simulation_ledger_for_active_paper_orders(
    session: Any,
    *,
    trader_id: str,
    paper_account_id: str,
) -> dict[str, Any]:
    account_id = str(paper_account_id or "").strip()
    if not trader_id or not account_id:
        return {"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}

    mode_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    status_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    rows = list(
        (
            await session.execute(
                select(TraderOrder)
                .where(TraderOrder.trader_id == trader_id)
                .where(mode_expr.in_(("shadow", "paper")))
                .where(status_expr.in_(tuple(_ACTIVE_ORDER_STATUSES)))
                .order_by(TraderOrder.created_at.asc())
            )
        )
        .scalars()
        .all()
    )

    attempted = 0
    backfilled = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    now_utc = utcnow()
    for row in rows:
        payload = dict(row.payload_json or {})
        if isinstance(payload.get("simulation_ledger"), dict):
            skipped += 1
            continue
        attempted += 1
        notional = safe_float(payload.get("filled_notional_usd"), None)
        if notional is None or notional <= 0.0:
            notional = safe_float(payload.get("effective_notional_usd"), None)
        if notional is None or notional <= 0.0:
            notional = safe_float(row.notional_usd, 0.0) or 0.0
        entry_price = safe_float(row.effective_price, None)
        if entry_price is None or entry_price <= 0.0:
            entry_price = safe_float(payload.get("average_fill_price"), None)
        if entry_price is None or entry_price <= 0.0:
            entry_price = safe_float(row.entry_price, None)
        if entry_price is None or entry_price <= 0.0 or notional <= 0.0:
            skipped += 1
            errors.append(
                {
                    "order_id": str(row.id),
                    "reason": "invalid_fill_metrics",
                    "entry_price": entry_price,
                    "notional_usd": notional,
                }
            )
            continue

        direction = str(row.direction or "").strip().lower()
        paper_sim = payload.get("paper_simulation")
        paper_sim = paper_sim if isinstance(paper_sim, dict) else {}
        execution_fee_usd = safe_float(paper_sim.get("estimated_fee_usd"), None)
        execution_slippage_usd = safe_float(paper_sim.get("slippage_usd"), None)
        token_id = _shadow_ledger_token_id(payload, direction)
        signal_id = str(row.signal_id or "").strip() or str(row.id)
        strategy_type = str(row.strategy_key or payload.get("strategy_type") or "").strip() or "trader_orchestrator"
        try:
            ledger_row = await simulation_service.record_orchestrator_paper_fill(
                account_id=account_id,
                trader_id=str(trader_id),
                signal_id=signal_id,
                market_id=str(row.market_id or ""),
                market_question=str(row.market_question or payload.get("market_question") or ""),
                direction=direction,
                notional_usd=float(notional),
                entry_price=float(entry_price),
                strategy_type=strategy_type,
                token_id=token_id or None,
                payload=payload,
                execution_fee_usd=execution_fee_usd,
                execution_slippage_usd=execution_slippage_usd,
                session=session,
                commit=False,
            )
            payload["simulation_ledger"] = {
                **dict(ledger_row or {}),
                "backfilled_at": now_utc.isoformat(),
                "mode": _canonical_trader_mode(row.mode, default="shadow"),
            }
            row.payload_json = payload
            row.updated_at = now_utc
            backfilled += 1
        except Exception as exc:
            errors.append(
                {
                    "order_id": str(row.id),
                    "reason": "record_orchestrator_paper_fill_failed",
                    "error": str(exc),
                }
            )

    if backfilled > 0:
        await session.flush()
    return {
        "attempted": attempted,
        "backfilled": backfilled,
        "skipped": skipped,
        "errors": errors,
    }


async def _build_edge_calibration_profile(
    session: Any,
    *,
    trader_id: str,
    source_key: str,
    mode: str,
    lookback_days: int,
    max_rows: int,
) -> dict[str, Any]:
    if not hasattr(session, "execute"):
        return {
            "sample_size": 0,
            "threshold_edge_multiplier": 1.0,
            "size_multiplier": 1.0,
            "bucket_size_multipliers": {},
        }

    mode_key = _canonical_trader_mode(mode)
    source = normalize_source_key(source_key)
    if not trader_id or not source:
        return {
            "sample_size": 0,
            "threshold_edge_multiplier": 1.0,
            "size_multiplier": 1.0,
            "bucket_size_multipliers": {},
        }

    lookback_days_clamped = max(1, min(90, int(lookback_days)))
    max_rows_clamped = max(50, min(5000, int(max_rows)))
    now_utc = utcnow()
    lookback_start = now_utc - timedelta(days=lookback_days_clamped)

    status_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    source_expr = func.lower(func.coalesce(TraderOrder.source, ""))
    mode_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    query = (
        select(
            TraderOrder.edge_percent,
            TraderOrder.actual_profit,
        )
        .where(TraderOrder.trader_id == trader_id)
        .where(source_expr == source)
        .where(mode_expr == mode_key)
        .where(TraderOrder.created_at >= lookback_start)
        .where(TraderOrder.edge_percent.isnot(None))
        .where(TraderOrder.actual_profit.isnot(None))
        .where(status_expr.in_(tuple(_TERMINAL_ORDER_STATUSES)))
        .order_by(TraderOrder.created_at.desc())
        .limit(max_rows_clamped)
    )
    rows = (await session.execute(query)).all()
    if not rows:
        return {
            "sample_size": 0,
            "lookback_days": lookback_days_clamped,
            "threshold_edge_multiplier": 1.0,
            "size_multiplier": 1.0,
            "bucket_size_multipliers": {},
            "hit_ratio": 1.0,
        }

    sample_size = len(rows)
    actual_hits = 0.0
    expected_hits = 0.0
    bucket_stats: dict[str, dict[str, float]] = {
        _edge_bucket_key(bucket[0], bucket[1]): {"count": 0.0, "actual_hits": 0.0, "expected_hits": 0.0}
        for bucket in _EDGE_CALIBRATION_BUCKETS
    }

    for row in rows:
        edge_percent = max(0.0, safe_float(row.edge_percent, 0.0) or 0.0)
        actual_profit = safe_float(row.actual_profit, 0.0) or 0.0
        expected_win = _clamp(0.5 + (edge_percent / 200.0), 0.5, 0.9)
        realized_win = 1.0 if actual_profit > 0.0 else 0.0
        actual_hits += realized_win
        expected_hits += expected_win
        for bucket_min, bucket_max in _EDGE_CALIBRATION_BUCKETS:
            if edge_percent < bucket_min:
                continue
            if edge_percent >= bucket_max:
                continue
            bucket_key = _edge_bucket_key(bucket_min, bucket_max)
            bucket_row = bucket_stats[bucket_key]
            bucket_row["count"] += 1.0
            bucket_row["actual_hits"] += realized_win
            bucket_row["expected_hits"] += expected_win
            break

    expected_hits = max(1e-6, expected_hits)
    hit_ratio = actual_hits / expected_hits
    if sample_size < _EDGE_CALIBRATION_MIN_SAMPLES:
        threshold_edge_multiplier = 1.0
        size_multiplier = 1.0
    else:
        threshold_edge_multiplier = _clamp(1.0 + ((1.0 - hit_ratio) * 0.65), 0.80, 1.80)
        size_multiplier = _clamp(0.55 + (0.45 * hit_ratio), 0.35, 1.20)

    bucket_size_multipliers: dict[str, float] = {}
    for bucket_min, bucket_max in _EDGE_CALIBRATION_BUCKETS:
        bucket_key = _edge_bucket_key(bucket_min, bucket_max)
        bucket_row = bucket_stats[bucket_key]
        bucket_count = int(bucket_row["count"])
        if bucket_count < _EDGE_CALIBRATION_BUCKET_MIN_SAMPLES:
            continue
        bucket_expected = max(1e-6, bucket_row["expected_hits"])
        bucket_ratio = bucket_row["actual_hits"] / bucket_expected
        bucket_size_multipliers[bucket_key] = _clamp(0.50 + (0.50 * bucket_ratio), 0.30, 1.25)

    return {
        "sample_size": int(sample_size),
        "lookback_days": lookback_days_clamped,
        "hit_ratio": round(hit_ratio, 6),
        "realized_hit_rate": round(actual_hits / max(1.0, float(sample_size)), 6),
        "expected_hit_rate": round(expected_hits / max(1.0, float(sample_size)), 6),
        "threshold_edge_multiplier": round(float(threshold_edge_multiplier), 6),
        "size_multiplier": round(float(size_multiplier), 6),
        "bucket_size_multipliers": bucket_size_multipliers,
        "as_of": now_utc.isoformat(),
    }


async def _ensure_prefetched_source_runtime_state(
    session: Any,
    *,
    trader_id: str,
    run_mode: str,
    source_configs: dict[str, dict[str, Any]],
    source_keys: set[str],
    source_runtime_state: dict[str, dict[str, Any]],
) -> None:
    for source_key in sorted({normalize_source_key(source) for source in source_keys if normalize_source_key(source)}):
        if source_key in source_runtime_state:
            continue
        source_config = source_configs.get(source_key)
        if source_config is None:
            continue

        strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
        strategy_params = dict(source_config.get("strategy_params") or {})
        requested_strategy_version: int | None = None
        requested_strategy_version_error: str | None = None
        try:
            requested_strategy_version = normalize_strategy_version(source_config.get("strategy_version"))
        except ValueError as exc:
            requested_strategy_version_error = str(exc)
        experiment_row = await get_active_strategy_experiment(
            session,
            source_key=source_key,
            strategy_key=strategy_key,
        )

        version_requests: list[int | None] = []
        if requested_strategy_version_error is None:
            version_requests.append(requested_strategy_version)
        if experiment_row is not None:
            version_requests.extend(
                [
                    int(experiment_row.control_version or 1),
                    int(experiment_row.candidate_version or 1),
                ]
            )

        version_resolutions: dict[int | None, Any] = {}
        version_errors: dict[int | None, str] = {}
        seen_requests: set[int | None] = set()
        for version_request in version_requests:
            if version_request in seen_requests:
                continue
            seen_requests.add(version_request)
            try:
                version_resolutions[version_request] = await resolve_strategy_version(
                    session,
                    strategy_key=strategy_key,
                    requested_version=version_request,
                )
            except ValueError as exc:
                version_errors[version_request] = str(exc)

        edge_calibration_profile: dict[str, Any] | None = None
        if source_key == "crypto":
            lookback_days = max(
                1,
                min(
                    90,
                    safe_int(
                        strategy_params.get("edge_calibration_lookback_days"),
                        _EDGE_CALIBRATION_LOOKBACK_DAYS_DEFAULT,
                    ),
                ),
            )
            max_rows = max(
                50,
                min(
                    5000,
                    safe_int(
                        strategy_params.get("edge_calibration_max_rows"),
                        _EDGE_CALIBRATION_MAX_ROWS_DEFAULT,
                    ),
                ),
            )
            try:
                edge_calibration_profile = await _build_edge_calibration_profile(
                    session,
                    trader_id=trader_id,
                    source_key=source_key,
                    mode=run_mode,
                    lookback_days=lookback_days,
                    max_rows=max_rows,
                )
            except Exception as exc:
                logger.warning(
                    "Edge calibration profile failed for trader=%s source=%s strategy=%s",
                    trader_id,
                    source_key,
                    strategy_key,
                    exc_info=exc,
                )
                edge_calibration_profile = {
                    "sample_size": 0,
                    "threshold_edge_multiplier": 1.0,
                    "size_multiplier": 1.0,
                    "bucket_size_multipliers": {},
                }

        source_runtime_state[source_key] = {
            "experiment_row": experiment_row,
            "requested_strategy_version": requested_strategy_version,
            "requested_strategy_version_error": requested_strategy_version_error,
            "version_resolutions": version_resolutions,
            "version_errors": version_errors,
            "edge_calibration_profile": edge_calibration_profile,
        }


async def _enforce_source_open_order_timeouts(
    session: Any,
    *,
    trader_id: str,
    run_mode: str,
    source_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not source_configs:
        return {
            "configured": 0,
            "updated": 0,
            "suppressed": 0,
            "taker_rescue_attempted": 0,
            "taker_rescue_succeeded": 0,
            "taker_rescue_failed": 0,
            "sources": [],
            "errors": [],
        }

    scope = run_mode if run_mode in {"shadow", "live"} else "all"
    configured = 0
    updated = 0
    suppressed = 0
    taker_rescue_attempted = 0
    taker_rescue_succeeded = 0
    taker_rescue_failed = 0
    source_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    now_utc = utcnow()
    provider_reconcile: dict[str, Any] = {}

    expired_keys = [
        key for key, until in _open_order_timeout_cleanup_failure_cooldown_until.items() if until <= now_utc
    ]
    for key in expired_keys:
        _open_order_timeout_cleanup_failure_cooldown_until.pop(key, None)

    if scope == "live":
        cache_key = str(trader_id)
        cache_row = _live_provider_reconcile_cache.get(cache_key)
        cache_age_seconds: float | None = None
        if cache_row is not None:
            cache_ts, cached_payload = cache_row
            cache_age_seconds = max(0.0, (now_utc - cache_ts).total_seconds())
            if (
                cache_age_seconds < _LIVE_PROVIDER_RECONCILE_MIN_INTERVAL_SECONDS
                and isinstance(cached_payload, dict)
            ):
                provider_reconcile = dict(cached_payload)
                provider_reconcile["cache_hit"] = True
                provider_reconcile["cache_age_seconds"] = round(cache_age_seconds, 3)
        if not provider_reconcile:
            try:
                provider_reconcile = await reconcile_live_provider_orders(
                    session,
                    trader_id=trader_id,
                    commit=False,
                    broadcast=False,
                )
            except Exception as exc:
                logger.warning(
                    "Live provider reconcile before timeout cleanup failed for trader=%s",
                    trader_id,
                    exc_info=exc,
                )
                provider_reconcile = {
                    "provider_ready": False,
                    "error": str(exc),
                }
            _live_provider_reconcile_cache[cache_key] = (now_utc, dict(provider_reconcile))

    provider_ready_for_cleanup = bool(provider_reconcile.get("provider_ready", True))

    for source_key, source_config in source_configs.items():
        timeout_seconds = _source_open_order_timeout_seconds(source_config)
        if timeout_seconds is None:
            continue
        configured += 1
        rescue_policy = _source_timeout_taker_rescue_policy(source_config)
        rescue_enabled_for_source = bool(scope == "live" and rescue_policy["enabled"])
        if scope == "live" and not provider_ready_for_cleanup:
            suppressed += 1
            source_rows.append(
                {
                    "source": source_key,
                    "timeout_seconds": timeout_seconds,
                    "matched": 0,
                    "updated": 0,
                    "suppressed": True,
                    "suppressed_reason": "provider_unavailable",
                    "taker_rescue_enabled": rescue_enabled_for_source,
                }
            )
            continue
        cooldown_key = f"{trader_id}:{scope}:{source_key}"
        cooldown_until = _open_order_timeout_cleanup_failure_cooldown_until.get(cooldown_key)
        if cooldown_until is not None and now_utc < cooldown_until:
            suppressed += 1
            source_rows.append(
                {
                    "source": source_key,
                    "timeout_seconds": timeout_seconds,
                    "matched": 0,
                    "updated": 0,
                    "suppressed": True,
                    "suppressed_until": cooldown_until.isoformat(),
                    "taker_rescue_enabled": rescue_enabled_for_source,
                }
            )
            continue
        try:
            cleanup = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope=scope,
                max_age_seconds=timeout_seconds,
                source=source_key,
                require_unfilled=True,
                dry_run=False,
                target_status="cancelled",
                reason=f"max_open_order_timeout:{source_key}",
                attempt_live_taker_rescue=rescue_enabled_for_source,
                live_taker_rescue_price_bps=float(rescue_policy["price_bps"]),
                live_taker_rescue_time_in_force=str(rescue_policy["time_in_force"]),
            )
            _open_order_timeout_cleanup_failure_cooldown_until.pop(cooldown_key, None)
        except Exception as exc:
            logger.warning(
                "Source open-order timeout cleanup failed for trader=%s source=%s",
                trader_id,
                source_key,
                exc_info=exc,
            )
            cooldown_until = now_utc + timedelta(seconds=_OPEN_ORDER_TIMEOUT_CLEANUP_FAILURE_COOLDOWN_SECONDS)
            _open_order_timeout_cleanup_failure_cooldown_until[cooldown_key] = cooldown_until
            errors.append(
                {
                    "source": source_key,
                    "timeout_seconds": timeout_seconds,
                    "error": str(exc),
                    "suppressed_until": cooldown_until.isoformat(),
                    "cooldown_seconds": _OPEN_ORDER_TIMEOUT_CLEANUP_FAILURE_COOLDOWN_SECONDS,
                }
            )
            continue

        source_updated = int(cleanup.get("updated", 0))
        source_rescue_attempted = int(cleanup.get("taker_rescue_attempted", 0))
        source_rescue_succeeded = int(cleanup.get("taker_rescue_succeeded", 0))
        source_rescue_failed = int(cleanup.get("taker_rescue_failed", 0))
        taker_rescue_attempted += source_rescue_attempted
        taker_rescue_succeeded += source_rescue_succeeded
        taker_rescue_failed += source_rescue_failed
        updated += source_updated
        source_rows.append(
            {
                "source": source_key,
                "timeout_seconds": timeout_seconds,
                "matched": int(cleanup.get("matched", 0)),
                "updated": source_updated,
                "suppressed": False,
                "taker_rescue_enabled": rescue_enabled_for_source,
                "taker_rescue_price_bps": float(rescue_policy["price_bps"]),
                "taker_rescue_time_in_force": str(rescue_policy["time_in_force"]),
                "taker_rescue_attempted": source_rescue_attempted,
                "taker_rescue_succeeded": source_rescue_succeeded,
                "taker_rescue_failed": source_rescue_failed,
            }
        )

    return {
        "configured": configured,
        "updated": updated,
        "suppressed": suppressed,
        "taker_rescue_attempted": taker_rescue_attempted,
        "taker_rescue_succeeded": taker_rescue_succeeded,
        "taker_rescue_failed": taker_rescue_failed,
        "provider_reconcile": provider_reconcile,
        "sources": source_rows,
        "errors": errors,
    }


def _normalize_portfolio_config(risk_limits: dict[str, Any]) -> dict[str, Any]:
    raw = risk_limits.get("portfolio")
    config = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(config.get("enabled", False)),
        "target_utilization_pct": float(
            max(
                1.0,
                min(
                    100.0,
                    safe_float(config.get("target_utilization_pct"), 100.0),
                ),
            )
        ),
        "max_source_exposure_pct": float(
            max(
                1.0,
                min(
                    100.0,
                    safe_float(config.get("max_source_exposure_pct"), 100.0),
                ),
            )
        ),
        "min_order_notional_usd": float(
            max(
                1.0,
                safe_float(config.get("min_order_notional_usd"), 10.0),
            )
        ),
    }


def _allocate_portfolio_notional(
    *,
    requested_size_usd: float,
    gross_exposure_cap_usd: float,
    current_gross_exposure_usd: float,
    current_source_exposure_usd: float,
    source_key: str,
    portfolio_config: dict[str, Any],
) -> dict[str, Any]:
    requested_size = float(max(0.0, safe_float(requested_size_usd, 0.0)))
    gross_cap = float(max(0.0, safe_float(gross_exposure_cap_usd, 0.0)))
    target_utilization_pct = float(portfolio_config.get("target_utilization_pct", 100.0))
    max_source_exposure_pct = float(portfolio_config.get("max_source_exposure_pct", 100.0))
    min_order_notional = float(portfolio_config.get("min_order_notional_usd", 10.0))

    target_gross_cap = gross_cap * (target_utilization_pct / 100.0)
    current_gross = float(max(0.0, safe_float(current_gross_exposure_usd, 0.0)))
    current_source = float(max(0.0, safe_float(current_source_exposure_usd, 0.0)))
    remaining_gross = float(max(0.0, target_gross_cap - current_gross))
    source_cap = float(target_gross_cap * (max_source_exposure_pct / 100.0))
    source_remaining = float(max(0.0, source_cap - current_source))
    allocated_size = float(max(0.0, min(requested_size, remaining_gross, source_remaining)))

    reason = "Portfolio allocation accepted requested size"
    allowed = True

    if requested_size <= 0:
        allowed = False
        reason = "Portfolio blocked: non-positive requested notional"
    elif gross_cap <= 0:
        allowed = False
        reason = "Portfolio blocked: max_gross_exposure_usd must be positive"
    elif target_gross_cap <= 0:
        allowed = False
        reason = "Portfolio blocked: target utilization leaves no allocatable gross exposure"
    elif remaining_gross <= 0:
        allowed = False
        reason = "Portfolio blocked: no remaining gross exposure budget"
    elif source_remaining <= 0:
        allowed = False
        reason = f"Portfolio blocked: source exposure cap reached for '{source_key or 'unknown'}'"
    elif allocated_size < min_order_notional:
        allowed = False
        reason = (
            "Portfolio blocked: allocated notional below minimum order size "
            f"({allocated_size:.2f} < {min_order_notional:.2f})"
        )
    elif allocated_size < requested_size:
        reason = f"Portfolio capped: allocated {allocated_size:.2f} from requested {requested_size:.2f}"

    return {
        "allowed": bool(allowed),
        "reason": reason,
        "size_usd": allocated_size if allowed else 0.0,
        "requested_size_usd": requested_size,
        "target_gross_cap_usd": target_gross_cap,
        "remaining_gross_cap_usd": remaining_gross,
        "source_key": source_key,
        "source_cap_usd": source_cap,
        "source_exposure_usd": current_source,
        "source_remaining_usd": source_remaining,
        "min_order_notional_usd": min_order_notional,
        "target_utilization_pct": target_utilization_pct,
        "max_source_exposure_pct": max_source_exposure_pct,
    }


def _strategy_instance_from_loaded(candidate: Any) -> Any:
    if candidate is None:
        return None
    instance = getattr(candidate, "instance", None)
    if instance is not None:
        return instance
    return candidate


def _versioned_strategy_alias(strategy_key: str, strategy_version: int) -> str:
    return f"{str(strategy_key or '').strip().lower()}__v{int(strategy_version)}"


def _load_versioned_strategy_instance(
    *,
    strategy_key: str,
    strategy_version: int,
    source_code: str,
    config: dict[str, Any] | None,
) -> tuple[Any | None, str | None]:
    alias = _versioned_strategy_alias(strategy_key, strategy_version)
    loaded = strategy_loader.get_strategy(alias)
    if loaded is None:
        try:
            strategy_loader.load(alias, source_code, config or None)
        except StrategyValidationError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, str(exc)
        loaded = strategy_loader.get_strategy(alias)
    instance = _strategy_instance_from_loaded(loaded)
    if instance is None:
        return None, "Strategy cache miss"
    try:
        instance.key = str(strategy_key or "").strip().lower()
    except (AttributeError, TypeError) as exc:
        logger.debug("Unable to stamp versioned strategy key onto instance", strategy_key=strategy_key, exc_info=exc)
    return instance, None


async def _try_acquire_orchestrator_cycle_lock(session: Any) -> bool:
    """Acquire a cross-process lock when running on PostgreSQL.

    For non-PostgreSQL dialects, fall back to
    allowing the cycle to proceed.
    """
    dialect_name = _session_dialect_name(session)

    if dialect_name != "postgresql":
        return True

    try:
        result = await session.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": int(_ORCHESTRATOR_CYCLE_LOCK_KEY)},
        )
        return bool(result.scalar())
    except Exception as exc:
        logger.warning("Unable to acquire orchestrator cycle advisory lock: %s", exc)
        return False


async def _release_orchestrator_cycle_lock(session: Any) -> None:
    """Release the PostgreSQL advisory lock, if applicable."""
    dialect_name = _session_dialect_name(session)

    if dialect_name != "postgresql":
        return

    try:
        await session.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": int(_ORCHESTRATOR_CYCLE_LOCK_KEY)},
        )
    except Exception as exc:
        logger.warning("Unable to release orchestrator cycle advisory lock: %s", exc)


def _orchestrator_lock_dsn() -> str:
    dsn = str(settings.DATABASE_URL or "").strip()
    if dsn.startswith("postgresql+asyncpg://"):
        return "postgresql://" + dsn[len("postgresql+asyncpg://") :]
    if dsn.startswith("postgres://"):
        return "postgresql://" + dsn[len("postgres://") :]
    return dsn


async def _try_acquire_orchestrator_cycle_lock_connection(connection: asyncpg.Connection) -> bool:
    try:
        acquired = await connection.fetchval(
            "SELECT pg_try_advisory_lock($1)",
            int(_ORCHESTRATOR_CYCLE_LOCK_KEY),
        )
        return bool(acquired)
    except Exception as exc:
        logger.warning("Unable to acquire orchestrator cycle advisory lock: %s", exc)
        return False


async def _release_orchestrator_cycle_lock_connection(connection: asyncpg.Connection) -> None:
    try:
        await connection.execute(
            "SELECT pg_advisory_unlock($1)",
            int(_ORCHESTRATOR_CYCLE_LOCK_KEY),
        )
    except Exception as exc:
        logger.warning("Unable to release orchestrator cycle advisory lock: %s", exc)


async def _close_orchestrator_lock_connection(connection: asyncpg.Connection) -> None:
    try:
        close_result = connection.close()
        if asyncio.iscoroutine(close_result):
            await close_result
    except Exception as close_exc:
        logger.debug("Failed to close orchestrator lock connection cleanly", exc_info=close_exc)
        try:
            connection.terminate()
        except Exception as terminate_exc:
            logger.warning("Failed to terminate orchestrator lock connection", exc_info=terminate_exc)


async def _ensure_orchestrator_cycle_lock_owner() -> bool:
    """Own the cross-process orchestrator lock for this worker process."""
    return await _ensure_orchestrator_cycle_lock_owner_for_lane(_LANE_GENERAL)


async def _ensure_orchestrator_cycle_lock_owner_for_lane(lane: str) -> bool:
    """Own the cross-process orchestrator lock for a specific worker lane."""
    global _orchestrator_lock_connection

    if _orchestrator_lock_connection is not None:
        return True

    connection: asyncpg.Connection | None = None
    try:
        connection = await asyncpg.connect(
            dsn=_orchestrator_lock_dsn(),
            timeout=float(max(1.0, float(settings.DATABASE_CONNECT_TIMEOUT_SECONDS))),
            command_timeout=float(max(5.0, float(settings.DATABASE_POOL_TIMEOUT_SECONDS))),
            server_settings={"timezone": "UTC"},
        )
        acquired = await connection.fetchval(
            "SELECT pg_try_advisory_lock($1)",
            _cycle_lock_key_for_lane(lane),
        )
        if not acquired:
            return False

        _orchestrator_lock_connection = connection
        connection = None
        logger.info("Acquired orchestrator cross-process cycle lock")

        return True
    finally:
        if connection is not None:
            await _close_orchestrator_lock_connection(connection)


async def _release_orchestrator_cycle_lock_owner() -> None:
    """Release owned lock session on shutdown/error."""
    await _release_orchestrator_cycle_lock_owner_for_lane(_LANE_GENERAL)


async def _release_orchestrator_cycle_lock_owner_for_lane(lane: str) -> None:
    """Release owned lock session for a specific worker lane."""
    global _orchestrator_lock_connection
    if _orchestrator_lock_connection is None:
        return

    connection = _orchestrator_lock_connection
    _orchestrator_lock_connection = None
    try:
        await connection.execute(
            "SELECT pg_advisory_unlock($1)",
            _cycle_lock_key_for_lane(lane),
        )
        logger.info("Released orchestrator cross-process cycle lock")
    finally:
        await _close_orchestrator_lock_connection(connection)


def _query_sources_for_configs(source_configs: dict[str, dict[str, Any]]) -> list[str]:
    if not source_configs:
        return []
    return sorted(source_configs.keys())


def _query_strategy_types_for_configs(source_configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    if not source_configs:
        return {}
    out: dict[str, list[str]] = {}
    for source_key, source_config in source_configs.items():
        allowed = _accepted_signal_strategy_types(source_config)
        if not allowed:
            continue
        out[source_key] = sorted(allowed)
    return out


def _normalize_crypto_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_crypto_timeframe(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    compact = raw.replace("_", "").replace("-", "").replace(" ", "")
    mapping = {
        "5m": "5m",
        "5min": "5m",
        "5minute": "5m",
        "5minutes": "5m",
        "15m": "15m",
        "15min": "15m",
        "15minute": "15m",
        "15minutes": "15m",
        "1h": "1h",
        "1hr": "1h",
        "1hour": "1h",
        "60m": "1h",
        "60min": "1h",
        "4h": "4h",
        "4hr": "4h",
        "4hour": "4h",
        "240m": "4h",
        "240min": "4h",
    }
    return mapping.get(compact, compact)


def _normalize_crypto_scope_values(raw: Any, normalizer) -> set[str]:
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = []
    normalized: set[str] = set()
    for item in values:
        value = normalizer(item)
        if value:
            normalized.add(value)
    return normalized


def _crypto_signal_dimensions(signal: Any) -> tuple[str, str]:
    strategy_context = getattr(signal, "strategy_context_json", None)
    payload = getattr(signal, "payload_json", None)
    context = strategy_context if isinstance(strategy_context, dict) else {}
    body = payload if isinstance(payload, dict) else {}

    asset_raw = (
        context.get("asset")
        or context.get("coin")
        or context.get("symbol")
        or body.get("asset")
        or body.get("coin")
        or body.get("symbol")
    )
    timeframe_raw = (
        context.get("timeframe")
        or context.get("cadence")
        or context.get("interval")
        or body.get("timeframe")
        or body.get("cadence")
        or body.get("interval")
    )
    if not timeframe_raw:
        signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
        timeframe_raw = _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES.get(signal_strategy_type)
    return _normalize_crypto_asset(asset_raw), _normalize_crypto_timeframe(timeframe_raw)


def _crypto_signal_in_scope(signal: Any, source_config: dict[str, Any]) -> bool:
    strategy_params = dict(source_config.get("strategy_params") or {})
    include_assets = _normalize_crypto_scope_values(
        strategy_params.get("include_assets"),
        _normalize_crypto_asset,
    )
    exclude_assets = _normalize_crypto_scope_values(
        strategy_params.get("exclude_assets"),
        _normalize_crypto_asset,
    )
    include_timeframes = _normalize_crypto_scope_values(
        strategy_params.get("include_timeframes"),
        _normalize_crypto_timeframe,
    )
    exclude_timeframes = _normalize_crypto_scope_values(
        strategy_params.get("exclude_timeframes"),
        _normalize_crypto_timeframe,
    )

    asset, timeframe = _crypto_signal_dimensions(signal)

    if include_assets and asset and asset not in include_assets:
        return False
    if asset and asset in exclude_assets:
        return False
    if include_timeframes and timeframe and timeframe not in include_timeframes:
        return False
    if timeframe and timeframe in exclude_timeframes:
        return False
    return True


def _signal_wallets(signal: Any) -> set[str]:
    return StrategySDK.extract_trader_signal_wallets(signal)


def _is_live_provider_infra_error(error_text: Any) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return False
    if "not enough balance / allowance" in text:
        return False
    return any(marker in text for marker in _LIVE_PROVIDER_INFRA_ERROR_MARKERS)


def _live_provider_health_event_due(trader_id: str, now: datetime) -> bool:
    cooldown_until = _live_provider_block_event_cooldown_until.get(trader_id)
    if cooldown_until is not None and now < cooldown_until:
        return False
    _live_provider_block_event_cooldown_until[trader_id] = now + timedelta(
        seconds=_LIVE_PROVIDER_BLOCK_EVENT_COOLDOWN_SECONDS
    )
    return True


def _live_risk_clamp_event_due(trader_id: str, now: datetime) -> bool:
    cooldown_until = _live_risk_clamp_event_cooldown_until.get(trader_id)
    if cooldown_until is not None and now < cooldown_until:
        return False
    _live_risk_clamp_event_cooldown_until[trader_id] = now + timedelta(seconds=_LIVE_RISK_CLAMP_EVENT_COOLDOWN_SECONDS)
    return True


async def _live_risk_clamp_event_should_emit(
    session: Any,
    *,
    trader_id: str,
    changes: dict[str, Any],
    now: datetime,
) -> bool:
    if not changes:
        return False
    if not hasattr(session, "execute"):
        return _live_risk_clamp_event_due(trader_id, now)

    row = (
        await session.execute(
            select(TraderEvent.payload_json)
            .where(TraderEvent.trader_id == trader_id)
            .where(TraderEvent.event_type == "live_risk_clamped")
            .order_by(TraderEvent.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    payload = dict(row or {}) if isinstance(row, dict) else {}
    prior_changes = payload.get("changes") if isinstance(payload.get("changes"), dict) else {}
    if prior_changes == changes:
        return False
    return _live_risk_clamp_event_due(trader_id, now)


def _provider_reconcile_has_material_changes(payload: dict[str, Any]) -> bool:
    material_keys = (
        "status_changes",
        "session_status_changes",
        "notional_updates",
        "price_updates",
        "terminal_session_cancels",
    )
    return any(int(payload.get(key, 0) or 0) > 0 for key in material_keys)


async def _live_provider_failure_snapshot(
    session: Any,
    *,
    trader_id: str,
    window_seconds: int,
) -> dict[str, Any]:
    normalized_window_seconds = int(max(30, int(window_seconds)))
    if not hasattr(session, "execute"):
        return {"count": 0, "window_seconds": normalized_window_seconds, "errors": []}

    now = utcnow()
    cutoff = now - timedelta(seconds=normalized_window_seconds)
    cache_key = f"{trader_id}:{normalized_window_seconds}"
    for key, (cached_at, _cached_payload) in list(_live_provider_failure_snapshot_cache.items()):
        if (now - cached_at).total_seconds() > _LIVE_PROVIDER_FAILURE_SNAPSHOT_CACHE_TTL_SECONDS:
            _live_provider_failure_snapshot_cache.pop(key, None)

    cached = _live_provider_failure_snapshot_cache.get(cache_key)
    if cached is not None:
        cached_at, cached_payload = cached
        if (now - cached_at).total_seconds() <= _LIVE_PROVIDER_FAILURE_SNAPSHOT_CACHE_TTL_SECONDS:
            return copy.deepcopy(cached_payload)

    mode_key_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    try:
        rows = (
            await session.execute(
                select(
                    TraderOrder.id,
                    TraderOrder.status,
                    TraderOrder.error_message,
                    TraderOrder.payload_json,
                    TraderOrder.updated_at,
                )
                .where(TraderOrder.trader_id == trader_id)
                .where(mode_key_expr == "live")
                .where(TraderOrder.updated_at.is_not(None))
                .where(TraderOrder.updated_at >= cutoff)
                .order_by(TraderOrder.updated_at.desc())
                .limit(200)
            )
        ).all()
    except Exception as exc:
        if cached is not None:
            logger.warning(
                "Using cached live provider failure snapshot after refresh failure",
                trader_id=trader_id,
                exc_info=exc,
            )
            return copy.deepcopy(cached[1])
        if isinstance(exc, OperationalError) and _is_retryable_db_error(exc):
            logger.warning(
                "Live provider failure snapshot skipped due transient DB error",
                trader_id=trader_id,
            )
            return {"count": 0, "window_seconds": normalized_window_seconds, "errors": []}
        raise

    failures: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row.payload_json or {})
        pending_live_exit = payload.get("pending_live_exit")
        pending_live_exit = pending_live_exit if isinstance(pending_live_exit, dict) else {}
        provider_reconcile = payload.get("provider_reconciliation")
        provider_reconcile = provider_reconcile if isinstance(provider_reconcile, dict) else {}
        candidates = [
            row.error_message,
            payload.get("error_message"),
            payload.get("error"),
            pending_live_exit.get("last_error"),
            provider_reconcile.get("error"),
        ]
        matched_error = next((text for text in candidates if _is_live_provider_infra_error(text)), None)
        if matched_error is None:
            continue
        failures.append(
            {
                "order_id": str(row.id or "").strip() or None,
                "status": str(row.status or "").strip().lower() or None,
                "updated_at": row.updated_at.isoformat() if row.updated_at is not None else None,
                "error": str(matched_error),
            }
        )

    payload = {
        "count": len(failures),
        "window_seconds": normalized_window_seconds,
        "errors": failures[:8],
    }
    _live_provider_failure_snapshot_cache[cache_key] = (now, copy.deepcopy(payload))
    return payload

def _apply_live_risk_clamps(
    effective_risk_limits: dict[str, Any],
    live_risk_clamps: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}

    enforce_allow_averaging_off = bool(
        live_risk_clamps.get(
            "enforce_allow_averaging_off",
            DEFAULT_LIVE_RISK_CLAMPS["enforce_allow_averaging_off"],
        )
    )
    if enforce_allow_averaging_off:
        configured_allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))
        if configured_allow_averaging:
            changes["allow_averaging"] = {"configured": configured_allow_averaging, "effective": False}
        effective_risk_limits["allow_averaging"] = False

    configured_cooldown_seconds = max(0, safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
    min_cooldown_seconds = max(
        0,
        safe_int(
            live_risk_clamps.get("min_cooldown_seconds"),
            DEFAULT_LIVE_RISK_CLAMPS["min_cooldown_seconds"],
        ),
    )
    clamped_cooldown_seconds = max(configured_cooldown_seconds, min_cooldown_seconds)
    if clamped_cooldown_seconds != configured_cooldown_seconds:
        changes["cooldown_seconds"] = {
            "configured": configured_cooldown_seconds,
            "effective": clamped_cooldown_seconds,
        }
    effective_risk_limits["cooldown_seconds"] = clamped_cooldown_seconds

    configured_max_consecutive_losses = max(1, safe_int(effective_risk_limits.get("max_consecutive_losses"), 4))
    max_consecutive_losses_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_consecutive_losses_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_consecutive_losses_cap"],
        ),
    )
    clamped_max_consecutive_losses = min(configured_max_consecutive_losses, max_consecutive_losses_cap)
    if clamped_max_consecutive_losses != configured_max_consecutive_losses:
        changes["max_consecutive_losses"] = {
            "configured": configured_max_consecutive_losses,
            "effective": clamped_max_consecutive_losses,
        }
    effective_risk_limits["max_consecutive_losses"] = clamped_max_consecutive_losses

    configured_max_open_orders = max(1, safe_int(effective_risk_limits.get("max_open_orders"), 20))
    max_open_orders_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_open_orders_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_open_orders_cap"],
        ),
    )
    clamped_max_open_orders = min(configured_max_open_orders, max_open_orders_cap)
    if clamped_max_open_orders != configured_max_open_orders:
        changes["max_open_orders"] = {
            "configured": configured_max_open_orders,
            "effective": clamped_max_open_orders,
        }
    effective_risk_limits["max_open_orders"] = clamped_max_open_orders

    configured_max_open_positions = max(1, safe_int(effective_risk_limits.get("max_open_positions"), 12))
    max_open_positions_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_open_positions_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_open_positions_cap"],
        ),
    )
    clamped_max_open_positions = min(configured_max_open_positions, max_open_positions_cap)
    if clamped_max_open_positions != configured_max_open_positions:
        changes["max_open_positions"] = {
            "configured": configured_max_open_positions,
            "effective": clamped_max_open_positions,
        }
    effective_risk_limits["max_open_positions"] = clamped_max_open_positions

    configured_max_trade_notional = max(1.0, safe_float(effective_risk_limits.get("max_trade_notional_usd"), 350.0))
    max_trade_notional_usd_cap = max(
        1.0,
        safe_float(
            live_risk_clamps.get("max_trade_notional_usd_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_trade_notional_usd_cap"],
        ),
    )
    clamped_max_trade_notional = min(float(configured_max_trade_notional), float(max_trade_notional_usd_cap))
    if clamped_max_trade_notional != float(configured_max_trade_notional):
        changes["max_trade_notional_usd"] = {
            "configured": float(configured_max_trade_notional),
            "effective": clamped_max_trade_notional,
        }
    effective_risk_limits["max_trade_notional_usd"] = clamped_max_trade_notional

    configured_max_orders_per_cycle = max(1, safe_int(effective_risk_limits.get("max_orders_per_cycle"), 50))
    max_orders_per_cycle_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_orders_per_cycle_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_orders_per_cycle_cap"],
        ),
    )
    clamped_max_orders_per_cycle = min(configured_max_orders_per_cycle, max_orders_per_cycle_cap)
    if clamped_max_orders_per_cycle != configured_max_orders_per_cycle:
        changes["max_orders_per_cycle"] = {
            "configured": configured_max_orders_per_cycle,
            "effective": clamped_max_orders_per_cycle,
        }
    effective_risk_limits["max_orders_per_cycle"] = clamped_max_orders_per_cycle

    enforce_halt_on_consecutive_losses = bool(
        live_risk_clamps.get(
            "enforce_halt_on_consecutive_losses",
            DEFAULT_LIVE_RISK_CLAMPS["enforce_halt_on_consecutive_losses"],
        )
    )
    if enforce_halt_on_consecutive_losses:
        configured_halt = bool(effective_risk_limits.get("halt_on_consecutive_losses", False))
        if not configured_halt:
            changes["halt_on_consecutive_losses"] = {
                "configured": configured_halt,
                "effective": True,
            }
        effective_risk_limits["halt_on_consecutive_losses"] = True
    return changes


async def _build_traders_scope_context(session: Any, traders_scope: dict[str, Any]) -> dict[str, Any]:
    normalized_scope = StrategySDK.validate_trader_scope_config(traders_scope)
    now_utc = utcnow()

    for key, (cached_at, _cached_context) in list(_traders_scope_context_cache.items()):
        if (now_utc - cached_at).total_seconds() > _TRADERS_SCOPE_CONTEXT_CACHE_TTL_SECONDS:
            _traders_scope_context_cache.pop(key, None)

    scope_cache_key = _traders_scope_cache_key(normalized_scope)
    cached_row = _traders_scope_context_cache.get(scope_cache_key)
    if cached_row is not None:
        cached_at, cached_context = cached_row
        if (now_utc - cached_at).total_seconds() <= _TRADERS_SCOPE_CONTEXT_CACHE_TTL_SECONDS:
            return copy.deepcopy(cached_context)

    modes = {
        str(mode or "").strip().lower() for mode in (normalized_scope.get("modes") or []) if str(mode or "").strip()
    }

    async def _normalize_scope_wallets(values: list[Any]) -> set[str]:
        normalized: set[str] = set()
        for value in values:
            wallet = _normalize_wallet(value)
            if not wallet:
                continue
            if wallet.startswith("0x") and len(wallet) == 42:
                normalized.add(wallet)
                continue
            try:
                resolved = await polymarket_client.resolve_wallet_identifier(wallet)
            except ValueError:
                continue
            resolved_wallet = _normalize_wallet(resolved.get("address"))
            if resolved_wallet:
                normalized.add(resolved_wallet)
        return normalized

    tracked_wallets: set[str] = set()
    pool_wallets: set[str] = set()
    group_wallets: set[str] = set()
    tracked_rows: list[Any] = []
    pool_rows: list[Any] = []
    group_rows: list[Any] = []

    try:
        if "tracked" in modes:
            tracked_rows = list((await session.execute(select(TrackedWallet.address))).scalars().all())

        if "pool" in modes:
            pool_rows = list(
                (
                    await session.execute(
                        select(DiscoveredWallet.address).where(DiscoveredWallet.in_top_pool == True)  # noqa: E712
                    )
                )
                .scalars()
                .all()
            )

        if "group" in modes and list(normalized_scope.get("group_ids") or []):
            group_rows = list(
                (
                    await session.execute(
                        select(TraderGroupMember.wallet_address).where(
                            TraderGroupMember.group_id.in_(list(normalized_scope.get("group_ids") or []))
                        )
                    )
                )
                .scalars()
                .all()
            )

        async with release_conn(session):
            if tracked_rows:
                tracked_wallets = await _normalize_scope_wallets(tracked_rows)
            if pool_rows:
                pool_wallets = await _normalize_scope_wallets(pool_rows)
            if group_rows:
                group_wallets = await _normalize_scope_wallets(group_rows)

        context = StrategySDK.build_trader_scope_runtime_context(
            normalized_scope,
            tracked_wallets=tracked_wallets,
            pool_wallets=pool_wallets,
            group_wallets=group_wallets,
        )
        _traders_scope_context_cache[scope_cache_key] = (now_utc, dict(context))
        return context
    except Exception as exc:
        stale = _traders_scope_context_cache.get(scope_cache_key)
        if stale is not None:
            logger.warning(
                "Using cached traders scope context after refresh failure",
                exc_info=exc,
            )
            return copy.deepcopy(stale[1])
        raise


def _signal_matches_traders_scope(signal: Any, scope_context: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    return StrategySDK.match_trader_signal_scope(signal, scope_context)


async def _persist_trader_cycle_heartbeat(
    session: Any,
    trader_id: str,
    *,
    advance_run_clock: bool = True,
) -> None:
    now = utcnow()
    rowcount = 0
    if advance_run_clock:
        result = await session.execute(
            sa_update(Trader)
            .where(Trader.id == trader_id)
            .values(
                last_run_at=now,
                updated_at=now,
                requested_run_at=None,
            )
        )
        rowcount = int(result.rowcount or 0)
    else:
        result = await session.execute(
            sa_update(Trader)
            .where(
                and_(
                    Trader.id == trader_id,
                    Trader.requested_run_at.is_not(None),
                )
            )
            .values(
                requested_run_at=None,
                updated_at=now,
            )
        )
        rowcount = int(result.rowcount or 0)
    if rowcount <= 0:
        return
    await _commit_with_retry(
        session,
        retry_attempts=2,
        base_delay_seconds=0.05,
        max_delay_seconds=0.1,
    )


def _signal_cursor_timestamp(signal: Any) -> Any:
    updated_at = getattr(signal, "updated_at", None)
    if updated_at is not None:
        return updated_at
    return getattr(signal, "created_at", None)


def _signal_runtime_sequence(signal: Any) -> int | None:
    return safe_int(getattr(signal, "runtime_sequence", None), None)


def _signal_sort_key(signal: Any) -> tuple[int, str]:
    runtime_sequence = _signal_runtime_sequence(signal)
    signal_id = str(getattr(signal, "id", "") or "")
    return runtime_sequence or 0, signal_id


def _latency_stage_percentile(summary: dict[str, Any], *, stage_key: str, percentile_key: str) -> int | None:
    if not isinstance(summary, dict):
        return None
    stage_summary = summary.get(stage_key)
    if not isinstance(stage_summary, dict):
        return None
    return safe_int(stage_summary.get(percentile_key), None)


def _worst_latency_group(groups: dict[str, Any], *, stage_key: str) -> tuple[str | None, int | None]:
    if not isinstance(groups, dict):
        return None, None
    worst_label: str | None = None
    worst_p95: int | None = None
    for raw_label, summary in groups.items():
        p95 = _latency_stage_percentile(summary, stage_key=stage_key, percentile_key="p95")
        if p95 is None or (worst_p95 is not None and p95 <= worst_p95):
            continue
        worst_label = str(raw_label or "").strip() or None
        worst_p95 = p95
    return worst_label, worst_p95


def _maybe_log_execution_latency_sla_breach(*, lane: str, metrics: dict[str, Any]) -> None:
    execution_latency = metrics.get("execution_latency")
    if not isinstance(execution_latency, dict):
        return
    target_ms = safe_int(execution_latency.get("internal_sla_target_ms"), None)
    overall = execution_latency.get("overall")
    p95 = _latency_stage_percentile(overall, stage_key=_LATENCY_SLA_STAGE_KEY, percentile_key="p95")
    if target_ms is None or p95 is None or p95 <= target_ms:
        return

    lane_key = str(lane or _LANE_GENERAL).strip().lower() or _LANE_GENERAL
    now = utcnow()
    last_logged_at = _latency_sla_breach_logged_at.get(lane_key)
    if last_logged_at is not None and (now - last_logged_at).total_seconds() < _LATENCY_SLA_BREACH_LOG_COOLDOWN_SECONDS:
        return
    _latency_sla_breach_logged_at[lane_key] = now

    overall_p99 = _latency_stage_percentile(overall, stage_key=_LATENCY_SLA_STAGE_KEY, percentile_key="p99")
    worst_source, worst_source_p95 = _worst_latency_group(
        execution_latency.get("by_source") or {},
        stage_key=_LATENCY_SLA_STAGE_KEY,
    )
    worst_strategy, worst_strategy_p95 = _worst_latency_group(
        execution_latency.get("by_strategy") or {},
        stage_key=_LATENCY_SLA_STAGE_KEY,
    )
    worst_trader, worst_trader_p95 = _worst_latency_group(
        execution_latency.get("by_trader") or {},
        stage_key=_LATENCY_SLA_STAGE_KEY,
    )
    logger.warning(
        "Execution latency SLA breached",
        lane=lane_key,
        stage_key=_LATENCY_SLA_STAGE_KEY,
        target_ms=target_ms,
        p95=p95,
        p99=overall_p99,
        worst_source=worst_source,
        worst_source_p95=worst_source_p95,
        worst_strategy=worst_strategy,
        worst_strategy_p95=worst_strategy_p95,
        worst_trader=worst_trader,
        worst_trader_p95=worst_trader_p95,
    )


async def _list_triggered_trade_signals(
    session: Any,
    *,
    trader_id: str,
    signal_ids_by_source: dict[str, list[str]],
    sources: list[str],
    strategy_types_by_source: dict[str, list[str]],
    cursor_runtime_sequence: int | None,
    cursor_created_at: datetime | None,
    cursor_signal_id: str | None,
    statuses: list[str],
    limit: int,
) -> list[Any]:
    del session
    if not signal_ids_by_source or not sources:
        return []

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for source_key in ("__all__", *sources):
        for raw_signal_id in signal_ids_by_source.get(source_key) or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen_ids:
                continue
            seen_ids.add(signal_id)
            ordered_ids.append(signal_id)
    if not ordered_ids:
        return []

    runtime_rows = await get_intent_runtime().list_unconsumed_signals(
        trader_id=str(trader_id or ""),
        sources=list(sources or []),
        statuses=list(statuses or []),
        strategy_types_by_source=dict(strategy_types_by_source or {}),
        cursor_runtime_sequence=cursor_runtime_sequence,
        cursor_created_at=cursor_created_at,
        cursor_signal_id=cursor_signal_id,
        limit=max(len(ordered_ids), int(limit)),
    )
    rows_by_id = {
        str(getattr(row, "id", "") or "").strip(): row
        for row in runtime_rows
        if str(getattr(row, "id", "") or "").strip()
    }
    ordered_rows = [rows_by_id[signal_id] for signal_id in ordered_ids if signal_id in rows_by_id]
    ordered_rows.sort(key=_signal_sort_key)
    return ordered_rows[: max(1, min(int(limit), 5000))]


async def _emit_cycle_heartbeat_if_due(
    session: Any,
    *,
    trader_id: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    now = utcnow()
    last_emitted = _trader_cycle_heartbeat_last_emitted.get(trader_id)
    if last_emitted is not None:
        elapsed_seconds = (now - last_emitted).total_seconds()
        if elapsed_seconds < _TRADER_CYCLE_HEARTBEAT_EVENT_INTERVAL_SECONDS:
            return

    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="cycle_heartbeat",
        source="worker",
        message=message,
        payload=payload or {},
        commit=False,
    )
    _trader_cycle_heartbeat_last_emitted[trader_id] = now


def _market_winner_value(market_info: Any) -> str:
    if not isinstance(market_info, dict):
        return ""
    winner_raw = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") not in (None, "")
        else market_info.get("winner")
    )
    return str(winner_raw or "").strip()


def _market_outcome_prices(market_info: Any) -> list[float]:
    if not isinstance(market_info, dict):
        return []
    prices = market_info.get("outcome_prices")
    if not isinstance(prices, list):
        prices = market_info.get("outcomePrices")
    if not isinstance(prices, list):
        return []
    out: list[float] = []
    for raw in prices[:2]:
        parsed = safe_float(raw)
        if parsed is None:
            out.append(-1.0)
            continue
        out.append(min(1.0, max(0.0, parsed)))
    return out


def _is_terminal_market_state(
    market_info: Any,
    *,
    now: datetime,
    settle_floor: float = 0.98,
) -> bool:
    if not isinstance(market_info, dict):
        return False
    if bool(market_info.get("closed", False)):
        return True
    if _market_winner_value(market_info):
        return True

    is_tradable = polymarket_client.is_market_tradable(market_info, now=now)
    settle_floor = min(1.0, max(0.5, settle_floor))
    settle_ceiling = max(0.0, 1.0 - settle_floor)
    prices = _market_outcome_prices(market_info)
    if not is_tradable and len(prices) >= 2:
        yes_price = prices[0]
        no_price = prices[1]
        if yes_price >= settle_floor and no_price <= settle_ceiling:
            return True
        if no_price >= settle_floor and yes_price <= settle_ceiling:
            return True
    return False


async def _run_terminal_stale_order_watchdog(session: Any, *, now: datetime | None = None) -> dict[str, Any]:
    global _terminal_stale_order_last_checked_at

    now_utc = now or utcnow()
    if _terminal_stale_order_last_checked_at is not None:
        elapsed_seconds = (now_utc - _terminal_stale_order_last_checked_at).total_seconds()
        if elapsed_seconds < _TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS:
            return {"checked": False, "stale": 0, "alerted": 0}
    _terminal_stale_order_last_checked_at = now_utc

    status_key_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    age_anchor_expr = func.coalesce(TraderOrder.executed_at, TraderOrder.updated_at, TraderOrder.created_at)
    raw_result = await session.execute(
        select(
            TraderOrder.id,
            TraderOrder.trader_id,
            TraderOrder.mode,
            TraderOrder.status,
            TraderOrder.market_id,
            TraderOrder.executed_at,
            TraderOrder.updated_at,
            TraderOrder.created_at,
        )
        .where(status_key_expr.in_(tuple(_ACTIVE_ORDER_STATUSES)))
        .order_by(age_anchor_expr.asc())
        .limit(2000)
    )
    raw_rows = list(raw_result.all())
    candidates: list[Any] = []
    if raw_rows and hasattr(raw_rows[0], "_mapping"):
        candidates = [
            SimpleNamespace(
                id=row._mapping["id"],
                trader_id=row._mapping["trader_id"],
                mode=row._mapping["mode"],
                status=row._mapping["status"],
                market_id=row._mapping["market_id"],
                executed_at=row._mapping["executed_at"],
                updated_at=row._mapping["updated_at"],
                created_at=row._mapping["created_at"],
            )
            for row in raw_rows
        ]
    elif raw_rows:
        candidates = list(raw_rows)
    if not candidates:
        return {"checked": True, "stale": 0, "alerted": 0}

    market_info_by_id = await load_market_info_for_orders(list(candidates))
    stale_rows: list[dict[str, Any]] = []
    for row in candidates:
        market_id = str(row.market_id or "").strip()
        if not market_id:
            continue
        market_info = market_info_by_id.get(market_id)
        if not _is_terminal_market_state(market_info, now=now_utc):
            continue
        age_anchor = row.executed_at or row.updated_at or row.created_at
        if age_anchor is None:
            continue
        age_minutes = max(0.0, (now_utc - age_anchor).total_seconds() / 60.0)
        if age_minutes < float(_TERMINAL_STALE_ORDER_MIN_AGE_MINUTES):
            continue
        stale_rows.append(
            {
                "order_id": str(row.id),
                "trader_id": str(row.trader_id or ""),
                "mode": str(row.mode or ""),
                "status": str(row.status or ""),
                "market_id": market_id,
                "age_minutes": round(age_minutes, 2),
            }
        )

    if not stale_rows:
        return {"checked": True, "stale": 0, "alerted": 0}

    stale_order_ids = [str(row["order_id"]) for row in stale_rows if str(row["order_id"]).strip()]
    remediation = {
        "groups_attempted": 0,
        "groups_failed": 0,
        "provider_updates": 0,
        "closed": 0,
        "inventory_updates": 0,
    }
    stale_entities_result = await session.execute(select(TraderOrder).where(TraderOrder.id.in_(stale_order_ids)))
    stale_entities = list(stale_entities_result.scalars().all()) if hasattr(stale_entities_result, "scalars") else []
    stale_entities_by_group: dict[tuple[str, str], list[Any]] = {}
    for row in stale_entities:
        trader_id = str(getattr(row, "trader_id", "") or "").strip()
        mode_key = _canonical_trader_mode(getattr(row, "mode", ""), default="other")
        if not trader_id or mode_key not in {"shadow", "live"} or not hasattr(row, "payload_json"):
            continue
        stale_entities_by_group.setdefault((trader_id, mode_key), []).append(row)

    for (trader_id, mode_key), group_rows in stale_entities_by_group.items():
        remediation["groups_attempted"] = int(remediation["groups_attempted"]) + 1
        order_ids = [str(getattr(row, "id", "") or "").strip() for row in group_rows]
        order_ids = [order_id for order_id in order_ids if order_id]
        if not order_ids:
            continue
        try:
            if mode_key == "live":
                provider_result = await reconcile_live_provider_orders(
                    session,
                    trader_id=trader_id,
                    commit=False,
                    broadcast=False,
                )
                remediation["provider_updates"] = int(remediation["provider_updates"]) + int(
                    provider_result.get("updated_orders", 0) or 0
                )
                from services.trader_orchestrator.position_lifecycle import reconcile_live_positions

                lifecycle_result = await reconcile_live_positions(
                    session,
                    trader_id=trader_id,
                    trader_params={},
                    dry_run=False,
                    force_mark_to_market=True,
                    order_ids=order_ids,
                    reason="terminal_market_watchdog",
                )
            else:
                lifecycle_result = await reconcile_paper_positions(
                    session,
                    trader_id=trader_id,
                    trader_params={},
                    dry_run=False,
                    force_mark_to_market=True,
                    order_ids=order_ids,
                    reason="terminal_market_watchdog",
                )
            remediation["closed"] = int(remediation["closed"]) + int(lifecycle_result.get("closed", 0) or 0)
            inventory_result = await sync_trader_position_inventory(
                session,
                trader_id=trader_id,
                mode=mode_key,
            )
            remediation["inventory_updates"] = int(remediation["inventory_updates"]) + int(
                inventory_result.get("updates", 0) or 0
            )
        except Exception as exc:
            remediation["groups_failed"] = int(remediation["groups_failed"]) + 1
            logger.warning(
                "Terminal stale-order remediation failed for trader=%s mode=%s: %s",
                trader_id,
                mode_key,
                exc,
            )

    status_rows = await session.execute(
        select(TraderOrder.id, TraderOrder.status).where(TraderOrder.id.in_(stale_order_ids))
    )
    active_status_by_order_id: dict[str, str] = {}
    raw_status_rows = list(status_rows.all()) if hasattr(status_rows, "all") else list(status_rows)
    for row in raw_status_rows:
        if hasattr(row, "_mapping"):
            active_status_by_order_id[str(row._mapping["id"])] = str(row._mapping["status"] or "").strip().lower()
            continue
        order_id = str(getattr(row, "id", "") or "").strip()
        if not order_id:
            continue
        active_status_by_order_id[order_id] = str(getattr(row, "status", "") or "").strip().lower()
    stale_rows = [
        detail
        for detail in stale_rows
        if active_status_by_order_id.get(str(detail["order_id"])) in _ACTIVE_ORDER_STATUSES
    ]
    if not stale_rows:
        return {
            "checked": True,
            "stale": 0,
            "alerted": 0,
            "remediation": remediation,
        }

    stale_cutoff = now_utc - timedelta(hours=24)
    for order_id, emitted_at in list(_terminal_stale_order_alert_last_emitted.items()):
        if emitted_at < stale_cutoff:
            _terminal_stale_order_alert_last_emitted.pop(order_id, None)

    alert_rows: list[dict[str, Any]] = []
    for detail in stale_rows:
        order_id = detail["order_id"]
        last_emitted = _terminal_stale_order_alert_last_emitted.get(order_id)
        if last_emitted is not None:
            elapsed = (now_utc - last_emitted).total_seconds()
            if elapsed < _TERMINAL_STALE_ORDER_ALERT_COOLDOWN_SECONDS:
                continue
        _terminal_stale_order_alert_last_emitted[order_id] = now_utc
        alert_rows.append(detail)

    if not alert_rows:
        return {"checked": True, "stale": len(stale_rows), "alerted": 0}

    affected_traders = sorted({row["trader_id"] for row in alert_rows if row["trader_id"]})
    await create_trader_event(
        session,
        trader_id=None,
        event_type="terminal_stale_orders",
        severity="warn",
        source="worker",
        message=f"Detected {len(alert_rows)} stale active order(s) on terminal markets.",
        payload={
            "stale_count": len(stale_rows),
            "alerted_count": len(alert_rows),
            "affected_traders": affected_traders,
            "orders": alert_rows[:50],
            "remediation": remediation,
            "min_age_minutes": _TERMINAL_STALE_ORDER_MIN_AGE_MINUTES,
            "check_interval_seconds": _TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS,
        },
        commit=True,
    )

    return {
        "checked": True,
        "stale": len(stale_rows),
        "alerted": len(alert_rows),
        "affected_traders": len(affected_traders),
        "remediation": remediation,
    }


async def _run_trader_once(
    trader: dict[str, Any],
    control: dict[str, Any],
    *,
    process_signals: bool = True,
    trigger_signal_ids_by_source: dict[str, list[str]] | None = None,
    trigger_signal_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> tuple[int, int, int]:
    decisions_written = 0
    orders_written = 0
    processed_signals = 0
    prefiltered_signals = 0
    prefiltered_by_reason: dict[str, int] = {}
    crypto_scope_prefiltered_dimensions: dict[str, int] = {}

    async with AsyncSessionLocal() as session:
        trader_id = str(trader["id"])
        source_configs = _normalize_source_configs(trader)
        effective_process_signals = bool(process_signals)
        is_high_frequency_trader = _is_high_frequency_crypto_trader(trader)
        runtime_trigger_signal_limit = (
            _HIGH_FREQUENCY_RUNTIME_TRIGGER_SIGNAL_LIMIT
            if is_high_frequency_trader
            else _STANDARD_RUNTIME_TRIGGER_SIGNAL_LIMIT
        )
        default_strategy_params: dict[str, Any] = {}
        sources: list[str] = []
        strategy_types_by_source: dict[str, list[str]] = {}
        source_entry_modes: dict[str, bool] = {}
        if source_configs:
            default_source_config = next(iter(source_configs.values()))
            default_strategy_params = dict(default_source_config.get("strategy_params") or {})
            sources = _query_sources_for_configs(source_configs)
            strategy_types_by_source = _query_strategy_types_for_configs(source_configs)
            source_entry_modes = {
                source_key: _source_config_allows_new_entries(source_config)
                for source_key, source_config in source_configs.items()
            }
        elif effective_process_signals:
            await _emit_cycle_heartbeat_if_due(
                session,
                trader_id=trader_id,
                message="Idle cycle: no source configs configured.",
            )
            effective_process_signals = False
        risk_limits_raw = trader.get("risk_limits")
        risk_limits = dict(risk_limits_raw or {})
        if isinstance(risk_limits_raw, dict):
            normalized_limits = StrategySDK.validate_trader_risk_config(risk_limits_raw)
            for key in list(risk_limits.keys()):
                key_name = str(key)
                if not key_name:
                    continue
                if key_name in normalized_limits:
                    risk_limits[key_name] = normalized_limits[key_name]
        position_cap_scope = str(risk_limits.get("position_cap_scope") or "market_direction").strip().lower()
        if position_cap_scope not in {"market_direction", "market", "asset_timeframe"}:
            position_cap_scope = "market_direction"
        metadata = StrategySDK.validate_trader_runtime_metadata(trader.get("metadata"))
        run_mode = _canonical_trader_mode(control.get("mode"), default="shadow")
        resume_policy = _normalize_resume_policy(metadata.get("resume_policy"))
        cursor_runtime_sequence = None
        cursor_created_at = None
        cursor_signal_id = None
        prefetched_signals: list[Any] | None = None
        stream_trigger_mode = bool(trigger_signal_ids_by_source)
        if effective_process_signals:
            cursor_runtime_sequence = await get_trader_signal_sequence_cursor(
                session,
                trader_id=trader_id,
            )
            cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(
                session,
                trader_id=trader_id,
            )
            if sources:
                if stream_trigger_mode:
                    prefetched_signals = await _build_triggered_trade_signals(
                        session,
                        trader_id=trader_id,
                        signal_ids_by_source=trigger_signal_ids_by_source or {},
                        signal_snapshots_by_source=trigger_signal_snapshots_by_source,
                        sources=sources,
                        strategy_types_by_source=strategy_types_by_source,
                        cursor_runtime_sequence=cursor_runtime_sequence,
                        cursor_created_at=cursor_created_at,
                        cursor_signal_id=cursor_signal_id,
                        statuses=["pending", "selected"],
                        limit=max(
                            _STANDARD_MAX_SIGNALS_PER_CYCLE,
                            _HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE,
                        ),
                    )
                else:
                    pending_preview = await list_unconsumed_trade_signals(
                        session,
                        trader_id=trader_id,
                        sources=sources,
                        statuses=["pending", "selected"],
                        strategy_types_by_source=strategy_types_by_source,
                        cursor_runtime_sequence=cursor_runtime_sequence,
                        cursor_created_at=cursor_created_at,
                        cursor_signal_id=cursor_signal_id,
                        limit=1,
                    )
                    if pending_preview:
                        prefetched_signals = pending_preview

                if not prefetched_signals:
                    now = utcnow()
                    idle_maintenance_interval_seconds = int(
                        max(
                            15,
                            min(
                                900,
                                safe_int(
                                    default_strategy_params.get("idle_maintenance_interval_seconds"),
                                    _TRADER_IDLE_MAINTENANCE_INTERVAL_SECONDS,
                                ),
                            ),
                        )
                    )
                    last_idle_maintenance = _trader_idle_maintenance_last_run.get(trader_id)
                    if last_idle_maintenance is not None:
                        elapsed_seconds = (now - last_idle_maintenance).total_seconds()
                        if elapsed_seconds < idle_maintenance_interval_seconds:
                            if not stream_trigger_mode:
                                await _emit_cycle_heartbeat_if_due(
                                    session,
                                    trader_id=trader_id,
                                    message="Idle cycle: no pending signals.",
                                    payload={
                                        "idle_maintenance_interval_seconds": idle_maintenance_interval_seconds,
                                        "elapsed_seconds": elapsed_seconds,
                                        "triggered_cycle": stream_trigger_mode,
                                    },
                                )
                            effective_process_signals = False
                    if effective_process_signals:
                        _trader_idle_maintenance_last_run[trader_id] = now
                if stream_trigger_mode and prefetched_signals and len(prefetched_signals) > runtime_trigger_signal_limit:
                    overflow_signals = prefetched_signals[runtime_trigger_signal_limit:]
                    prefetched_signals = prefetched_signals[:runtime_trigger_signal_limit]
                    overflow_signal_ids_by_source: dict[str, list[str]] = {}
                    overflow_seen_ids_by_source: dict[str, set[str]] = {}
                    for overflow_signal in overflow_signals:
                        signal_source = normalize_source_key(getattr(overflow_signal, "source", "")) or "__all__"
                        signal_id = str(getattr(overflow_signal, "id", "") or "").strip()
                        if not signal_id:
                            continue
                        overflow_signal_ids_by_source.setdefault(signal_source, [])
                        overflow_seen_ids_by_source.setdefault(signal_source, set())
                        if signal_id in overflow_seen_ids_by_source[signal_source]:
                            continue
                        overflow_seen_ids_by_source[signal_source].add(signal_id)
                        overflow_signal_ids_by_source[signal_source].append(signal_id)
                    for signal_source, signal_ids in overflow_signal_ids_by_source.items():
                        await publish_signal_batch(
                            event_type="upsert_update",
                            source=None if signal_source == "__all__" else signal_source,
                            signal_ids=signal_ids,
                            trigger="runtime_signal_overflow",
                            reason="trader_orchestrator_trigger_batch_overflow",
                        )
        open_positions = 0
        open_market_ids: set[str] = set()
        timeout_cleanup = {
            "configured": 0,
            "updated": 0,
            "suppressed": 0,
            "taker_rescue_attempted": 0,
            "taker_rescue_succeeded": 0,
            "taker_rescue_failed": 0,
            "sources": [],
            "errors": [],
            "provider_reconcile": {},
        }
        maintenance_now = utcnow()
        run_trader_maintenance = _is_trader_maintenance_due(
            trader_id=trader_id,
            run_mode=run_mode,
            now=maintenance_now,
        )
        if stream_trigger_mode and effective_process_signals:
            run_trader_maintenance = False

        if run_trader_maintenance:
            if run_mode == "shadow":
                shadow_account_id = _resolve_shadow_account_id(control, trader)
                backfill_result = await _backfill_simulation_ledger_for_active_paper_orders(
                    session,
                    trader_id=trader_id,
                    paper_account_id=shadow_account_id,
                )
                if backfill_result.get("errors"):
                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="shadow_ledger_backfill_failed",
                        severity="warn",
                        source="worker",
                        message="Shadow ledger backfill encountered one or more errors.",
                        payload=backfill_result,
                    )
                force_flatten = resume_policy == "flatten_then_start"
                lifecycle_result = await reconcile_paper_positions(
                    session,
                    trader_id=trader_id,
                    trader_params=default_strategy_params,
                    dry_run=False,
                    force_mark_to_market=force_flatten,
                    reason="worker_flatten_then_start" if force_flatten else "worker_lifecycle",
                )
                closed_positions = int(lifecycle_result.get("closed", 0))
                if closed_positions > 0:
                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="shadow_positions_closed",
                        source="worker",
                        message=f"Closed {closed_positions} shadow position(s)",
                        payload={
                            "matched": lifecycle_result.get("matched"),
                            "closed": closed_positions,
                            "held": lifecycle_result.get("held"),
                            "skipped": lifecycle_result.get("skipped"),
                            "total_realized_pnl": lifecycle_result.get("total_realized_pnl"),
                            "by_status": lifecycle_result.get("by_status"),
                        },
                    )
            elif run_mode == "live":
                # Live order/position reconciliation is handled by
                # workers.trader_reconciliation_worker so this loop stays focused
                # on strategy selection and execution.
                pass
            if run_mode != "live":
                await sync_trader_position_inventory(
                    session,
                    trader_id=trader_id,
                    mode=run_mode,
                )
        session_engine = ExecutionSessionEngine(session)
        run_execution_maintenance = bool(run_trader_maintenance and run_mode != "live" and hasattr(session, "execute"))
        if run_execution_maintenance:
            try:
                reconcile_result = await asyncio.wait_for(
                    session_engine.reconcile_active_sessions(
                        mode=run_mode,
                        trader_id=trader_id,
                    ),
                    timeout=_TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                if session.in_transaction():
                    await session.rollback()
                reconcile_result = {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}
                logger.warning(
                    "Execution session reconcile timed out for trader=%s mode=%s timeout=%.1fs",
                    trader_id,
                    run_mode,
                    _TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS,
                )
            if reconcile_result.get("expired", 0) > 0:
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="execution_sessions_expired",
                    severity="warn",
                    source="worker",
                    message=f"Expired {int(reconcile_result['expired'])} execution session(s)",
                    payload=reconcile_result,
                )
        if run_execution_maintenance:
            try:
                timeout_cleanup = await asyncio.wait_for(
                    _enforce_source_open_order_timeouts(
                        session,
                        trader_id=trader_id,
                        run_mode=run_mode,
                        source_configs=source_configs,
                    ),
                    timeout=_TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                if session.in_transaction():
                    await session.rollback()
                timeout_cleanup = {
                    "configured": 0,
                    "updated": 0,
                    "suppressed": 0,
                    "taker_rescue_attempted": 0,
                    "taker_rescue_succeeded": 0,
                    "taker_rescue_failed": 0,
                    "sources": [],
                    "errors": [
                        (
                            f"open order timeout cleanup exceeded "
                            f"{_TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS:.1f}s"
                        )
                    ],
                    "provider_reconcile": {},
                }
                logger.warning(
                    "Open-order timeout cleanup timed out for trader=%s mode=%s timeout=%.1fs",
                    trader_id,
                    run_mode,
                    _TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS,
                )
        if run_trader_maintenance:
            _trader_maintenance_last_run[trader_id] = maintenance_now
        provider_reconcile_payload = timeout_cleanup.get("provider_reconcile")
        provider_reconcile_payload = (
            dict(provider_reconcile_payload) if isinstance(provider_reconcile_payload, dict) else {}
        )
        if run_mode == "live" and provider_reconcile_payload:
            provider_active_seen = int(provider_reconcile_payload.get("active_seen", 0) or 0)
            provider_status_changes = int(provider_reconcile_payload.get("status_changes", 0) or 0)
            provider_updates = int(provider_reconcile_payload.get("updated_orders", 0) or 0)
            provider_ready = bool(provider_reconcile_payload.get("provider_ready", True))
            provider_material_changes = _provider_reconcile_has_material_changes(provider_reconcile_payload)
            if (not provider_ready) or provider_material_changes:
                provider_severity = "warn" if not provider_ready else "info"
                provider_message = (
                    "Provider reconcile unavailable before timeout cleanup"
                    if not provider_ready
                    else (
                        f"Provider reconcile active_seen={provider_active_seen} "
                        f"status_changes={provider_status_changes} updated_orders={provider_updates}"
                    )
                )
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="provider_reconcile",
                    severity=provider_severity,
                    source="worker",
                    message=provider_message,
                    payload=provider_reconcile_payload,
                    commit=False,
                )
        if timeout_cleanup.get("updated", 0) > 0:
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="open_order_timeout_cleanup",
                source="worker",
                message=f"Cancelled {int(timeout_cleanup['updated'])} stale unfilled order(s)",
                payload=timeout_cleanup,
            )
        if timeout_cleanup.get("errors"):
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="open_order_timeout_cleanup_failed",
                severity="warn",
                source="worker",
                    message="One or more source open-order timeout cleanups failed",
                    payload=timeout_cleanup,
                )
        suppressed_sources = [
            dict(row)
            for row in (timeout_cleanup.get("sources") or [])
            if isinstance(row, dict) and bool(row.get("suppressed"))
        ]
        if suppressed_sources:
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="timeout_cleanup_suppressed",
                severity="warn",
                source="worker",
                message=f"Suppressed timeout cleanup for {len(suppressed_sources)} source(s)",
                payload={
                    "suppressed": int(timeout_cleanup.get("suppressed", len(suppressed_sources)) or 0),
                    "sources": suppressed_sources,
                    "provider_reconcile": provider_reconcile_payload,
                },
                commit=False,
            )

        if not effective_process_signals:
            await _persist_trader_cycle_heartbeat(
                session,
                trader_id,
                advance_run_clock=bool(process_signals),
            )
            return 0, 0, processed_signals

        open_positions = await get_open_position_count_for_trader(
            session,
            trader_id,
            mode=run_mode,
            position_cap_scope=position_cap_scope,
        )
        open_order_count = await get_open_order_count_for_trader(
            session,
            trader_id,
            mode=run_mode,
        )
        control_settings = dict(control.get("settings") or {})
        global_runtime_settings = dict(control_settings.get("global_runtime") or {})
        pending_live_exit_guard_settings = dict(global_runtime_settings.get("pending_live_exit_guard") or {})
        pending_live_exit_max_allowed = max(
            0,
            safe_int(
                pending_live_exit_guard_settings.get("max_pending_exits"),
                int(DEFAULT_PENDING_LIVE_EXIT_GUARD["max_pending_exits"]),
            ),
        )
        pending_live_exit_identity_guard_enabled = bool(
            pending_live_exit_guard_settings.get(
                "identity_guard_enabled",
                DEFAULT_PENDING_LIVE_EXIT_GUARD["identity_guard_enabled"],
            )
        )
        pending_live_exit_terminal_statuses = pending_live_exit_guard_settings.get("terminal_statuses")
        if not isinstance(pending_live_exit_terminal_statuses, list) or not pending_live_exit_terminal_statuses:
            pending_live_exit_terminal_statuses = list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
        live_provider_health_settings = dict(
            global_runtime_settings.get("live_provider_health") or DEFAULT_LIVE_PROVIDER_HEALTH
        )
        live_market_context_settings = dict(
            global_runtime_settings.get("live_market_context") or DEFAULT_LIVE_MARKET_CONTEXT
        )
        pending_live_exit_summary = {
            "count": 0,
            "order_ids": [],
            "market_ids": [],
            "signal_ids": [],
            "statuses": {},
            "identities": [],
            "identity_keys": [],
        }
        if run_mode == "live":
            if open_order_count > 0:
                try:
                    pending_live_exit_summary = await asyncio.wait_for(
                        get_pending_live_exit_summary_for_trader(
                            session,
                            trader_id,
                            mode=run_mode,
                            terminal_statuses=pending_live_exit_terminal_statuses,
                        ),
                        timeout=_TRADER_PENDING_EXIT_SUMMARY_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if session.in_transaction():
                        await session.rollback()
                    pending_live_exit_summary = {
                        "count": open_order_count,
                        "order_ids": [],
                        "market_ids": [],
                        "signal_ids": [],
                        "statuses": {},
                        "identities": [],
                        "identity_keys": [],
                        "timed_out": True,
                    }
                    logger.warning(
                        "Pending live-exit summary timed out for trader=%s timeout=%.1fs",
                        trader_id,
                        _TRADER_PENDING_EXIT_SUMMARY_TIMEOUT_SECONDS,
                    )
        pending_live_exit_count = int(pending_live_exit_summary.get("count", 0) or 0)
        effective_open_positions = max(open_positions, open_order_count)
        open_market_ids = await get_open_market_ids_for_trader(
            session,
            trader_id,
            mode=run_mode,
        )

        block_entries_reason = None
        block_entries_payload: dict[str, Any] = {
            "resume_policy": resume_policy,
            "mode": run_mode,
            "open_positions": open_positions,
        }
        block_entries_event_type = "trader_resume_policy"
        block_entries_event_severity = "warn"
        if resume_policy == "manage_only":
            block_entries_reason = "Resume policy manage_only blocks new entries"
        elif resume_policy == "flatten_then_start" and open_positions > 0:
            if run_mode == "shadow":
                block_entries_reason = (
                    f"Resume policy flatten_then_start waiting to flatten {open_positions} open shadow position(s)"
                )
            else:
                block_entries_reason = f"Resume policy flatten_then_start blocked: {open_positions} open live position(s) require manual flattening"
        elif source_configs and source_entry_modes and not any(source_entry_modes.values()):
            block_entries_reason = "Configured strategies disable new entries (manage-existing-only mode)"
            block_entries_event_type = "trader_strategy_entry_mode"
            block_entries_event_severity = "info"
            block_entries_payload["source_entry_modes"] = {
                source_key: ("entries_enabled" if allows_entries else "manage_existing_only")
                for source_key, allows_entries in source_entry_modes.items()
            }

        if run_mode == "live":
            provider_health_window_seconds = int(
                max(
                    30,
                    min(
                        900,
                        safe_int(
                            live_provider_health_settings.get("window_seconds"),
                            int(DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]),
                        ),
                    ),
                )
            )
            provider_health_min_errors = int(
                max(
                    1,
                    min(
                        20,
                        safe_int(
                            live_provider_health_settings.get("min_errors"),
                            int(DEFAULT_LIVE_PROVIDER_HEALTH["min_errors"]),
                        ),
                    ),
                )
            )
            provider_health_block_seconds = int(
                max(
                    15,
                    min(
                        3600,
                        safe_int(
                            live_provider_health_settings.get("block_seconds"),
                            int(DEFAULT_LIVE_PROVIDER_HEALTH["block_seconds"]),
                        ),
                    ),
                )
            )
            provider_health_now = utcnow()
            blocked_until = _live_provider_entry_blocked_until.get(trader_id)
            if blocked_until is not None and provider_health_now >= blocked_until:
                _live_provider_entry_blocked_until.pop(trader_id, None)
                blocked_until = None

            provider_health_snapshot: dict[str, Any]
            new_provider_block = False
            if blocked_until is None:
                provider_health_snapshot = await _live_provider_failure_snapshot(
                    session,
                    trader_id=trader_id,
                    window_seconds=provider_health_window_seconds,
                )
                if int(provider_health_snapshot.get("count", 0) or 0) >= provider_health_min_errors:
                    blocked_until = provider_health_now + timedelta(seconds=provider_health_block_seconds)
                    _live_provider_entry_blocked_until[trader_id] = blocked_until
                    new_provider_block = True
            else:
                provider_health_snapshot = {
                    "count": 0,
                    "window_seconds": provider_health_window_seconds,
                    "errors": [],
                }

            if blocked_until is not None and provider_health_now < blocked_until:
                if block_entries_reason is None:
                    recent_error_count = int(provider_health_snapshot.get("count", 0) or 0)
                    blocked_until_iso = blocked_until.isoformat().replace("+00:00", "Z")
                    block_entries_reason = (
                        "Live provider health guard: "
                        f"entries paused until {blocked_until_iso} after {recent_error_count} provider infra "
                        f"failure(s) in {provider_health_window_seconds}s (threshold={provider_health_min_errors})"
                    )
                    block_entries_event_type = "live_provider_health_block"
                    block_entries_payload = {
                        **block_entries_payload,
                        "provider_health": {
                            "window_seconds": provider_health_window_seconds,
                            "min_errors": provider_health_min_errors,
                            "block_seconds": provider_health_block_seconds,
                            "blocked_until": blocked_until_iso,
                            "recent_failure_count": recent_error_count,
                            "recent_failures": provider_health_snapshot.get("errors") or [],
                            "new_block": new_provider_block,
                        },
                    }

        if block_entries_reason is not None and effective_process_signals:
            effective_process_signals = False
            should_emit_block_event = True
            if block_entries_event_type == "live_provider_health_block":
                should_emit_block_event = _live_provider_health_event_due(trader_id, utcnow())
            if should_emit_block_event:
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type=block_entries_event_type,
                    severity=block_entries_event_severity,
                    source="worker",
                    message=block_entries_reason,
                    payload=block_entries_payload,
                )

        max_signals_cap = (
            _HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE if is_high_frequency_trader else _STANDARD_MAX_SIGNALS_PER_CYCLE
        )
        default_max_signals = (
            _HIGH_FREQUENCY_DEFAULT_MAX_SIGNALS_PER_CYCLE
            if is_high_frequency_trader
            else _STANDARD_DEFAULT_MAX_SIGNALS_PER_CYCLE
        )
        default_scan_batch_size = (
            _HIGH_FREQUENCY_DEFAULT_SCAN_BATCH_SIZE if is_high_frequency_trader else default_max_signals
        )
        max_signals_per_cycle = int(
            max(
                1,
                min(
                    max_signals_cap,
                    safe_int(default_strategy_params.get("max_signals_per_cycle"), default_max_signals),
                ),
            )
        )
        scan_batch_size = int(
            max(
                1,
                min(
                    max_signals_cap,
                    safe_int(
                        default_strategy_params.get("scan_batch_size"),
                        min(default_scan_batch_size, max_signals_per_cycle),
                    ),
                ),
            )
        )
        scan_batch_size = min(scan_batch_size, max_signals_per_cycle)
        if stream_trigger_mode:
            runtime_trigger_scan_batch_size = (
                _HIGH_FREQUENCY_RUNTIME_TRIGGER_SCAN_BATCH_SIZE
                if is_high_frequency_trader
                else _STANDARD_RUNTIME_TRIGGER_SCAN_BATCH_SIZE
            )
            max_signals_per_cycle = min(max_signals_per_cycle, runtime_trigger_signal_limit)
            scan_batch_size = min(scan_batch_size, runtime_trigger_scan_batch_size, max_signals_per_cycle)
        enable_live_market_context = bool(live_market_context_settings.get("enabled", True))
        strict_ws_pricing_only = _coerce_bool(
            live_market_context_settings.get("strict_ws_pricing_only"),
            bool(DEFAULT_LIVE_MARKET_CONTEXT.get("strict_ws_pricing_only", True)),
        )
        if strict_ws_pricing_only and not enable_live_market_context:
            enable_live_market_context = True
        strict_ws_pricing_enforced = bool(strict_ws_pricing_only and enable_live_market_context)
        strict_ws_price_sources = ["ws_strict"]
        strict_ws_price_source_set = set(strict_ws_price_sources)
        strict_market_data_age_ms = int(
            max(
                25,
                min(
                    10_000,
                    safe_int(
                        live_market_context_settings.get("max_market_data_age_ms"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT.get("max_market_data_age_ms", 100)),
                    ),
                ),
            )
        )
        scanner_strict_market_data_age_ms = int(
            max(
                strict_market_data_age_ms,
                min(
                    300_000,
                    max(
                        100,
                        safe_int(
                            getattr(settings, "SCANNER_STRICT_WS_MAX_AGE_MS", 30000),
                            30000,
                        ),
                    ),
                ),
            )
        )
        history_window_seconds = int(
            max(
                300,
                min(
                    21600,
                    safe_int(
                        live_market_context_settings.get("history_window_seconds"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT["history_window_seconds"]),
                    ),
                ),
            )
        )
        history_fidelity_seconds = int(
            max(
                30,
                min(
                    1800,
                    safe_int(
                        live_market_context_settings.get("history_fidelity_seconds"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT["history_fidelity_seconds"]),
                    ),
                ),
            )
        )
        max_history_points = int(
            max(
                20,
                min(
                    240,
                    safe_int(
                        live_market_context_settings.get("max_history_points"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT["max_history_points"]),
                    ),
                ),
            )
        )
        live_market_context_timeout_seconds = float(
            max(
                1.0,
                min(
                    12.0,
                    safe_float(
                        live_market_context_settings.get("timeout_seconds"),
                        float(DEFAULT_LIVE_MARKET_CONTEXT["timeout_seconds"]),
                    )
                    or float(DEFAULT_LIVE_MARKET_CONTEXT["timeout_seconds"]),
                ),
            )
        )
        live_market_context_request_timeout_seconds = float(
            max(
                0.5,
                min(
                    4.0,
                    live_market_context_timeout_seconds / 2.0,
                ),
            )
        )

        now_utc = datetime.now(timezone.utc)
        trading_schedule_ok = is_within_trading_schedule_utc(metadata, now_utc)
        global_limits = dict(control_settings.get("global_risk") or {})
        effective_risk_limits = dict(risk_limits)
        if "max_orders_per_cycle" not in effective_risk_limits:
            fallback_cycle_limit = safe_int(global_limits.get("max_orders_per_cycle"), 50)
            if fallback_cycle_limit > 0:
                effective_risk_limits["max_orders_per_cycle"] = fallback_cycle_limit
        if "max_daily_loss_usd" not in effective_risk_limits:
            fallback_daily_loss = safe_float(global_limits.get("max_daily_loss_usd"), 0.0) or 0.0
            if fallback_daily_loss > 0:
                effective_risk_limits["max_daily_loss_usd"] = fallback_daily_loss
        live_risk_clamp_changes: dict[str, dict[str, Any]] = {}
        if run_mode == "live":
            live_risk_clamp_settings = dict(global_runtime_settings.get("live_risk_clamps") or DEFAULT_LIVE_RISK_CLAMPS)
            live_risk_clamp_changes = _apply_live_risk_clamps(
                effective_risk_limits,
                live_risk_clamp_settings,
            )
            if await _live_risk_clamp_event_should_emit(
                session,
                trader_id=trader_id,
                changes=live_risk_clamp_changes,
                now=utcnow(),
            ):
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="live_risk_clamped",
                    severity="warn",
                    source="worker",
                    message="Applied live risk safety clamps for directional execution stability.",
                    payload={"changes": live_risk_clamp_changes},
                )
        allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))

        global_daily_pnl = await get_daily_realized_pnl(session, trader_id=None, mode=run_mode)
        trader_daily_pnl = await get_daily_realized_pnl(session, trader_id=trader_id, mode=run_mode)
        try:
            global_unrealized_pnl = await get_unrealized_pnl(session, trader_id=None, mode=run_mode)
            trader_unrealized_pnl = await get_unrealized_pnl(session, trader_id=trader_id, mode=run_mode)
        except Exception as exc:
            logger.warning(
                "Failed to load unrealized PnL for trader cycle; using zero fallback",
                trader_id=trader_id,
                mode=run_mode,
                exc_info=exc,
            )
            global_unrealized_pnl = 0.0
            trader_unrealized_pnl = 0.0
        # Track cumulative notional committed within this cycle so that
        # subsequent risk checks account for intra-cycle exposure even
        # before PnL is realized in the database.
        intra_cycle_committed_usd: float = 0.0
        trader_loss_streak = await get_consecutive_loss_count(session, trader_id=trader_id, mode=run_mode)
        last_loss_at = await get_last_resolved_loss_at(session, trader_id=trader_id, mode=run_mode)
        cooldown_seconds = max(0, safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
        cooldown_active = False
        cooldown_remaining_seconds = 0
        if cooldown_seconds > 0 and last_loss_at is not None and trader_loss_streak > 0:
            if last_loss_at.tzinfo is None:
                loss_anchor = last_loss_at.replace(tzinfo=timezone.utc)
            else:
                loss_anchor = last_loss_at.astimezone(timezone.utc)
            cooldown_until = loss_anchor + timedelta(seconds=cooldown_seconds)
            if cooldown_until > now_utc:
                cooldown_active = True
                cooldown_remaining_seconds = int((cooldown_until - now_utc).total_seconds())

        # Circuit breaker: auto-pause trader and trigger safe exit of open positions
        halt_on_losses = bool(effective_risk_limits.get("halt_on_consecutive_losses", False))
        max_consecutive_losses_limit = max(1, safe_int(effective_risk_limits.get("max_consecutive_losses"), 4))
        if halt_on_losses and trader_loss_streak >= max_consecutive_losses_limit:
            await set_trader_paused(session, trader_id, True)
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="circuit_breaker_pause",
                source="worker",
                message=(
                    f"Circuit breaker: auto-paused after {trader_loss_streak} consecutive losses "
                    f"(limit={max_consecutive_losses_limit})"
                ),
                payload={
                    "trader_loss_streak": trader_loss_streak,
                    "max_consecutive_losses": max_consecutive_losses_limit,
                    "action": "auto_pause",
                },
            )
            logger.warning(
                "Circuit breaker auto-paused trader %s after %d consecutive losses (limit=%d)",
                trader_id,
                trader_loss_streak,
                max_consecutive_losses_limit,
            )

            if run_mode == "live":
                from services.trader_orchestrator.position_lifecycle import reconcile_live_positions

                try:
                    safe_exit_result = await reconcile_live_positions(
                        session,
                        trader_id=trader_id,
                        trader_params=default_strategy_params,
                        dry_run=False,
                        force_mark_to_market=True,
                        reason="circuit_breaker_safe_exit",
                    )
                    safe_exit_closed = int(safe_exit_result.get("closed", 0))
                    if safe_exit_closed > 0:
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="circuit_breaker_safe_exit",
                            source="worker",
                            message=f"Circuit breaker safe exit: closed {safe_exit_closed} position(s)",
                            payload=safe_exit_result,
                        )
                except Exception as exc:
                    logger.warning(
                        "Circuit breaker safe exit failed for trader %s: %s",
                        trader_id,
                        exc,
                        exc_info=exc,
                    )
            elif run_mode == "shadow":
                try:
                    safe_exit_result = await reconcile_paper_positions(
                        session,
                        trader_id=trader_id,
                        trader_params=default_strategy_params,
                        dry_run=False,
                        force_mark_to_market=True,
                        reason="circuit_breaker_safe_exit",
                    )
                    safe_exit_closed = int(safe_exit_result.get("closed", 0))
                    if safe_exit_closed > 0:
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="circuit_breaker_safe_exit",
                            source="worker",
                            message=f"Circuit breaker safe exit: closed {safe_exit_closed} shadow position(s)",
                            payload=safe_exit_result,
                        )
                except Exception as exc:
                    logger.warning(
                        "Circuit breaker safe exit failed for trader %s: %s",
                        trader_id,
                        exc,
                        exc_info=exc,
                    )

            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals

        traders_scope_context: dict[str, Any] | None = None
        traders_source_config = source_configs.get("traders")
        if traders_source_config is not None:
            traders_params = dict(traders_source_config.get("strategy_params") or {})
            traders_scope_context = await _build_traders_scope_context(
                session,
                traders_params.get("traders_scope"),
            )
        edge_calibration_cache: dict[str, dict[str, Any]] = {}
        source_runtime_state_cache: dict[str, dict[str, Any]] = {}
        copy_inventory_context: dict[str, Any] = {}
        copy_inventory_loaded = False

        # Kill switch is the very first gate: short-circuit before any signal
        # fetching, live-market context building, or risk evaluation.
        if bool(control.get("kill_switch")):
            logger.info(
                "Kill switch active for trader %s — skipping all signal processing",
                trader_id,
            )
            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals

        while processed_signals < max_signals_per_cycle:
            batch_limit = min(scan_batch_size, max_signals_per_cycle - processed_signals)
            if prefetched_signals is not None:
                signals = prefetched_signals[:batch_limit]
                prefetched_signals = prefetched_signals[batch_limit:] if len(prefetched_signals) > batch_limit else None
            else:
                if stream_trigger_mode:
                    break
                signals = await list_unconsumed_trade_signals(
                    session,
                    trader_id=trader_id,
                    sources=sources,
                    statuses=["pending", "selected"],
                    strategy_types_by_source=strategy_types_by_source,
                    cursor_runtime_sequence=cursor_runtime_sequence,
                    cursor_created_at=cursor_created_at,
                    cursor_signal_id=cursor_signal_id,
                    limit=batch_limit,
                )
            if not signals:
                break

            scoped_signals: list[Any] = []
            prefiltered = 0
            for signal in signals:
                signal_source = normalize_source_key(getattr(signal, "source", ""))
                source_config = source_configs.get(signal_source)
                signal_id = str(signal.id)
                signal_ts = _signal_cursor_timestamp(signal)
                if source_config is not None:
                    if not source_entry_modes.get(signal_source, True):
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason="Signal excluded: strategy is manage-existing-only (new entries disabled)",
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal_ts,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        cursor_created_at = signal_ts
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        prefiltered_signals += 1
                        prefiltered_by_reason["strategy_entry_disabled"] = (
                            prefiltered_by_reason.get("strategy_entry_disabled", 0) + 1
                        )
                        prefiltered += 1
                        continue
                    expected_strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                    allowed_strategy_types = set(strategy_types_by_source.get(signal_source) or [])
                    strategy_mismatch = bool(
                        expected_strategy_key
                        and allowed_strategy_types
                        and signal_strategy_type not in allowed_strategy_types
                    )
                    if strategy_mismatch:
                        allowed_label = ", ".join(sorted(allowed_strategy_types)) or expected_strategy_key
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason=(
                                "Signal excluded by source strategy filter "
                                f"(signal={signal_strategy_type or 'unknown'}, allowed={allowed_label})"
                            ),
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal_ts,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        cursor_created_at = signal_ts
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        prefiltered_signals += 1
                        prefiltered_by_reason["source_strategy_filter"] = (
                            prefiltered_by_reason.get("source_strategy_filter", 0) + 1
                        )
                        prefiltered += 1
                        continue
                if signal_source != "crypto" or source_config is None or _crypto_signal_in_scope(signal, source_config):
                    scoped_signals.append(signal)
                    continue
                await record_signal_consumption(
                    session,
                    trader_id=trader_id,
                    signal_id=signal_id,
                    outcome="skipped",
                    reason="Signal excluded by crypto scope filters",
                    commit=False,
                )
                await upsert_trader_signal_cursor(
                    session,
                    trader_id=trader_id,
                    last_signal_created_at=signal_ts,
                    last_signal_id=signal_id,
                    commit=False,
                )
                cursor_created_at = signal_ts
                cursor_signal_id = signal_id
                cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                processed_signals += 1
                prefiltered_signals += 1
                prefiltered_by_reason["crypto_scope_filter"] = prefiltered_by_reason.get("crypto_scope_filter", 0) + 1
                signal_asset, signal_timeframe = _crypto_signal_dimensions(signal)
                dimension_key = f"{signal_asset or 'unknown'}:{signal_timeframe or 'unknown'}"
                crypto_scope_prefiltered_dimensions[dimension_key] = (
                    crypto_scope_prefiltered_dimensions.get(dimension_key, 0) + 1
                )
                prefiltered += 1
            if prefiltered > 0:
                await _commit_with_retry(session)
            if not scoped_signals:
                continue
            signals = scoped_signals
            await _ensure_prefetched_source_runtime_state(
                session,
                trader_id=trader_id,
                run_mode=run_mode,
                source_configs=source_configs,
                source_keys={
                    normalize_source_key(getattr(signal, "source", ""))
                    for signal in signals
                    if source_configs.get(normalize_source_key(getattr(signal, "source", ""))) is not None
                },
                source_runtime_state=source_runtime_state_cache,
            )

            live_contexts: dict[str, dict[str, Any]] = {}
            if enable_live_market_context:
                context_candidates: list[Any] = []
                for sig in signals:
                    if strict_ws_pricing_enforced:
                        context_candidates.append(sig)
                        continue
                    source_key = normalize_source_key(getattr(sig, "source", ""))
                    source_config = source_configs.get(source_key)
                    if _supports_live_market_context(sig, source_config):
                        context_candidates.append(sig)
                try:
                    if strict_ws_pricing_enforced:
                        live_contexts = await build_cached_live_signal_contexts(
                            context_candidates,
                            max_history_points=max_history_points,
                            strict_ws_only=strict_ws_pricing_enforced,
                        )
                    elif stream_trigger_mode:
                        live_contexts = await build_cached_live_signal_contexts(
                            context_candidates,
                            max_history_points=max_history_points,
                            strict_ws_only=False,
                        )
                        if len(live_contexts) < len(context_candidates):
                            async with release_conn(session):
                                live_contexts = await asyncio.wait_for(
                                    build_live_signal_contexts(
                                        context_candidates,
                                        history_window_seconds=history_window_seconds,
                                        history_fidelity_seconds=history_fidelity_seconds,
                                        max_history_points=max_history_points,
                                        market_fetch_timeout_seconds=live_market_context_request_timeout_seconds,
                                        prices_batch_timeout_seconds=live_market_context_request_timeout_seconds,
                                        history_fetch_timeout_seconds=live_market_context_request_timeout_seconds,
                                        strict_ws_only=strict_ws_pricing_enforced,
                                    ),
                                    timeout=live_market_context_timeout_seconds,
                                )
                    else:
                        async with release_conn(session):
                            live_contexts = await asyncio.wait_for(
                                build_live_signal_contexts(
                                    context_candidates,
                                    history_window_seconds=history_window_seconds,
                                    history_fidelity_seconds=history_fidelity_seconds,
                                    max_history_points=max_history_points,
                                    market_fetch_timeout_seconds=live_market_context_request_timeout_seconds,
                                    prices_batch_timeout_seconds=live_market_context_request_timeout_seconds,
                                    history_fetch_timeout_seconds=live_market_context_request_timeout_seconds,
                                    strict_ws_only=strict_ws_pricing_enforced,
                                ),
                                timeout=live_market_context_timeout_seconds,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    try:
                        live_contexts = await build_cached_live_signal_contexts(
                            context_candidates,
                            max_history_points=max_history_points,
                            strict_ws_only=strict_ws_pricing_enforced,
                        )
                    except Exception as cache_exc:
                        logger.debug("Failed to load cached live market contexts after refresh failure", exc_info=cache_exc)
                        live_contexts = {}
                    if not live_contexts:
                        exc_name = type(exc).__name__
                        exc_message = str(exc).strip()
                        if exc_message:
                            logger.warning("Live market context refresh failed (%s): %s", exc_name, exc_message)
                        else:
                            logger.warning("Live market context refresh failed (%s)", exc_name)

            deferred_signals = 0
            deferred_by_reason: dict[str, int] = {}
            defer_signal_processing = False
            for signal in signals:
                signal_id = str(signal.id)
                signal_wake_started_at = utcnow()
                context_ready_at = signal_wake_started_at
                signal_source = normalize_source_key(getattr(signal, "source", ""))
                source_config = source_configs.get(signal_source)
                strategy_key = ""
                resolved_strategy_key = ""
                resolved_strategy_version: int | None = None
                requested_strategy_version: int | None = None
                experiment_row = None
                assignment_group: str | None = None
                assignment_sample_pct: float | None = None
                assignment_source = "pinned_config"

                try:
                    if source_config is None:
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason="No source configuration for signal source",
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await _commit_with_retry(session)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        continue

                    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    requested_strategy_key = str(source_config.get("requested_strategy_key") or "").strip().lower()
                    strategy_key_for_output = requested_strategy_key or strategy_key
                    strategy_params = dict(source_config.get("strategy_params") or {})
                    prefetched_source_runtime = source_runtime_state_cache.get(signal_source) or {}
                    prefetched_version_resolutions = (
                        dict(prefetched_source_runtime.get("version_resolutions") or {})
                        if isinstance(prefetched_source_runtime.get("version_resolutions"), dict)
                        else {}
                    )
                    prefetched_version_errors = (
                        dict(prefetched_source_runtime.get("version_errors") or {})
                        if isinstance(prefetched_source_runtime.get("version_errors"), dict)
                        else {}
                    )
                    strict_age_budget_ms = strict_market_data_age_ms
                    if signal_source == "scanner":
                        strict_age_budget_ms = scanner_strict_market_data_age_ms
                    if strict_ws_pricing_enforced:
                        strategy_params = _enforce_strict_ws_strategy_params(
                            strategy_params,
                            strict_age_budget_ms=strict_age_budget_ms,
                            strict_ws_price_sources=list(strict_ws_price_sources),
                        )
                    requested_strategy_version = prefetched_source_runtime.get("requested_strategy_version")
                    requested_strategy_version_error = (
                        str(prefetched_source_runtime.get("requested_strategy_version_error") or "").strip() or None
                    )
                    live_context = live_contexts.get(signal_id, {})
                    if strict_ws_pricing_enforced:
                        live_source = str(
                            live_context.get("market_data_source")
                            or live_context.get("live_selected_price_source")
                            or ""
                        ).strip().lower()
                        live_price = safe_float(live_context.get("live_selected_price"), None)
                        live_age_ms = safe_float(live_context.get("market_data_age_ms"), None)
                        strict_context_ok = (
                            live_source in strict_ws_price_source_set
                            and live_price is not None
                            and live_price > 0.0
                            and live_age_ms is not None
                            and live_age_ms <= strict_age_budget_ms
                        )
                        if not strict_context_ok:
                            live_reason_parts: list[str] = []
                            live_reason_parts.append(f"source={live_source or 'unknown'}")
                            live_reason_parts.append(
                                f"age_ms={live_age_ms:.0f}" if live_age_ms is not None else "age_ms=unknown"
                            )
                            live_reason_parts.append(f"max_age_ms={strict_age_budget_ms}")
                            await get_intent_runtime().defer_signal(
                                signal_id=signal_id,
                                required_token_ids=list(getattr(signal, "required_token_ids", None) or []),
                                reason="strict_ws_pricing_live_context_unavailable",
                            )
                            deferred_signals += 1
                            deferred_by_reason["strict_ws_pricing_live_context_unavailable"] = (
                                deferred_by_reason.get("strict_ws_pricing_live_context_unavailable", 0) + 1
                            )
                            defer_signal_processing = True
                            continue
                    context_ready_at = utcnow()
                    runtime_signal = RuntimeTradeSignalView(signal, live_context=live_context)
                    runtime_signal.source = signal_source
                    traders_scope_payload: dict[str, Any] | None = None
                    edge_calibration_profile: dict[str, Any] | None = None
                    if signal_source == "crypto":
                        edge_calibration_key = f"{signal_source}:{run_mode}:{strategy_key}"
                        edge_calibration_profile = edge_calibration_cache.get(edge_calibration_key)
                        if edge_calibration_profile is None:
                            edge_calibration_profile = (
                                dict(prefetched_source_runtime.get("edge_calibration_profile") or {})
                                if isinstance(prefetched_source_runtime.get("edge_calibration_profile"), dict)
                                else {}
                            )
                            edge_calibration_cache[edge_calibration_key] = dict(edge_calibration_profile)
                    experiment_row = prefetched_source_runtime.get("experiment_row")
                    assignment_source = "pinned_config"
                    resolved_version_request = requested_strategy_version
                    if experiment_row is not None:
                        assignment_group, assigned_version, assignment_sample_pct = resolve_experiment_assignment(
                            experiment=experiment_row,
                            trader_id=trader_id,
                            signal_id=signal_id,
                        )
                        resolved_version_request = int(assigned_version)
                        assignment_source = "experiment"

                    # ── Strategy/version resolution (unified loader + immutable versions) ─────────────
                    version_resolution = prefetched_version_resolutions.get(resolved_version_request)
                    strategy_detail = prefetched_version_errors.get(resolved_version_request)
                    if (
                        strategy_detail is None
                        and requested_strategy_version_error is not None
                        and resolved_version_request == requested_strategy_version
                    ):
                        strategy_detail = requested_strategy_version_error
                    if version_resolution is None:
                        blocked_reason = "strategy_version_unavailable"
                        strategy_detail = str(strategy_detail or "Strategy version resolution was unavailable")
                        checks_payload = [
                            {
                                "check_key": "strategy_version_available",
                                "check_label": "Strategy version available",
                                "passed": False,
                                "score": None,
                                "detail": strategy_detail,
                                "payload": {
                                    "requested_strategy_key": strategy_key_for_output,
                                    "requested_strategy_version": requested_strategy_version,
                                    "resolved_strategy_version": resolved_version_request,
                                },
                            }
                        ]
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            strategy_key=strategy_key_for_output,
                            strategy_version=resolved_version_request,
                            decision="blocked",
                            reason=blocked_reason,
                            score=0.0,
                            checks_summary={"count": len(checks_payload)},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config,
                                "strategy_runtime_error": strategy_detail,
                                "experiment": {
                                    "id": str(experiment_row.id) if experiment_row is not None else None,
                                    "assignment_group": assignment_group,
                                    "assignment_source": assignment_source,
                                    "assignment_sample_pct": assignment_sample_pct,
                                },
                            },
                            commit=False,
                        )
                        decisions_written += 1
                        await create_trader_decision_checks(
                            session,
                            decision_id=decision_row.id,
                            checks=checks_payload,
                            commit=False,
                        )
                        if experiment_row is not None:
                            await upsert_strategy_experiment_assignment(
                                session,
                                experiment_id=str(experiment_row.id),
                                trader_id=trader_id,
                                signal_id=signal_id,
                                source_key=signal_source,
                                strategy_key=strategy_key_for_output,
                                strategy_version=int(resolved_version_request or 1),
                                assignment_group=str(assignment_group or "control"),
                                decision_id=decision_row.id,
                                payload={
                                    "sample_pct": assignment_sample_pct,
                                    "assignment_source": assignment_source,
                                },
                                commit=False,
                            )
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status="skipped",
                            commit=False,
                        )
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=decision_row.id,
                            outcome="blocked",
                            reason=blocked_reason,
                            commit=False,
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="strategy_unavailable",
                            severity="warn",
                            source=signal_source,
                            message=blocked_reason,
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal.id,
                                "requested_strategy_key": strategy_key_for_output,
                                "requested_strategy_version": requested_strategy_version,
                                "resolved_strategy_version": resolved_version_request,
                                "error": strategy_detail,
                            },
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await _commit_with_retry(session)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        continue

                    resolved_strategy_key = (
                        str(version_resolution.strategy.slug or strategy_key).strip().lower() or strategy_key
                    )
                    resolved_strategy_version = int(
                        version_resolution.version_row.version or version_resolution.latest_version or 1
                    )
                    strategy = None
                    strategy_detail: str | None = None

                    if int(resolved_strategy_version) == int(version_resolution.latest_version):
                        strategy_status = strategy_loader.get_availability(resolved_strategy_key)
                        if strategy_status.available:
                            loaded = strategy_loader.get_strategy(strategy_status.resolved_key or resolved_strategy_key)
                            strategy = _strategy_instance_from_loaded(loaded)

                        if strategy is None and strategy_status.available:
                            signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                            if signal_strategy_type:
                                loaded = strategy_loader.get_strategy(signal_strategy_type)
                                candidate = _strategy_instance_from_loaded(loaded)
                                if candidate is not None and hasattr(candidate, "evaluate"):
                                    strategy = candidate

                        if strategy is None and strategy_status.available:
                            loaded = strategy_loader.get_strategy(signal_source)
                            candidate = _strategy_instance_from_loaded(loaded)
                            if candidate is not None and hasattr(candidate, "evaluate"):
                                strategy = candidate

                        if strategy is None:
                            strategy_detail = str(strategy_status.reason or "Strategy cache miss")
                    else:
                        strategy, strategy_detail = _load_versioned_strategy_instance(
                            strategy_key=resolved_strategy_key,
                            strategy_version=resolved_strategy_version,
                            source_code=str(version_resolution.version_row.source_code or ""),
                            config=dict(version_resolution.version_row.config or {}),
                        )

                    if strategy is None:
                        blocked_reason = f"strategy_unavailable:{strategy_key_for_output}:v{resolved_strategy_version}"
                        strategy_detail = str(strategy_detail or blocked_reason)
                        checks_payload = [
                            {
                                "check_key": "strategy_available",
                                "check_label": "Strategy available",
                                "passed": False,
                                "score": None,
                                "detail": strategy_detail,
                                "payload": {
                                    "requested_strategy_key": strategy_key_for_output,
                                    "resolved_strategy_key": resolved_strategy_key,
                                    "requested_strategy_version": requested_strategy_version,
                                    "resolved_strategy_version": resolved_strategy_version,
                                },
                            }
                        ]
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            strategy_key=strategy_key_for_output,
                            strategy_version=resolved_strategy_version,
                            decision="blocked",
                            reason=blocked_reason,
                            score=0.0,
                            checks_summary={"count": len(checks_payload)},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config,
                                "strategy_runtime_error": strategy_detail,
                                "experiment": {
                                    "id": str(experiment_row.id) if experiment_row is not None else None,
                                    "assignment_group": assignment_group,
                                    "assignment_source": assignment_source,
                                    "assignment_sample_pct": assignment_sample_pct,
                                },
                            },
                            commit=False,
                        )
                        decisions_written += 1
                        if experiment_row is not None:
                            await upsert_strategy_experiment_assignment(
                                session,
                                experiment_id=str(experiment_row.id),
                                trader_id=trader_id,
                                signal_id=signal_id,
                                source_key=signal_source,
                                strategy_key=strategy_key_for_output,
                                strategy_version=resolved_strategy_version,
                                assignment_group=str(assignment_group or "control"),
                                decision_id=decision_row.id,
                                payload={
                                    "sample_pct": assignment_sample_pct,
                                    "assignment_source": assignment_source,
                                },
                                commit=False,
                            )
                        await create_trader_decision_checks(
                            session,
                            decision_id=decision_row.id,
                            checks=checks_payload,
                            commit=False,
                        )
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status="skipped",
                            commit=False,
                        )
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=decision_row.id,
                            outcome="blocked",
                            reason=blocked_reason,
                            commit=False,
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="strategy_unavailable",
                            severity="warn",
                            source=signal_source,
                            message=blocked_reason,
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal.id,
                                "requested_strategy_key": strategy_key_for_output,
                                "resolved_strategy_key": resolved_strategy_key,
                                "error": strategy_detail,
                            },
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await _commit_with_retry(session)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        continue

                    if signal_source == "traders":
                        if traders_scope_context is None:
                            traders_params = dict(source_config.get("strategy_params") or {})
                            traders_scope_context = await _build_traders_scope_context(
                                session,
                                traders_params.get("traders_scope"),
                            )
                        scope_ok, scope_payload = _signal_matches_traders_scope(runtime_signal, traders_scope_context)
                        traders_scope_payload = scope_payload
                        if not scope_ok:
                            checks_payload = [
                                {
                                    "check_key": "traders_scope",
                                    "check_label": "Traders scope",
                                    "passed": False,
                                    "score": None,
                                    "detail": "Signal wallets did not match selected traders_scope modes.",
                                    "payload": scope_payload,
                                }
                            ]
                            decision_row = await create_trader_decision(
                                session,
                                trader_id=trader_id,
                                signal=runtime_signal,
                                strategy_key=resolved_strategy_key,
                                strategy_version=resolved_strategy_version,
                                decision="skipped",
                                reason="Signal excluded by traders_scope",
                                score=0.0,
                                checks_summary={"count": len(checks_payload)},
                                risk_snapshot={},
                                payload={
                                    "source_key": signal_source,
                                    "source_config": source_config,
                                    "traders_scope": scope_payload,
                                    "experiment": {
                                        "id": str(experiment_row.id) if experiment_row is not None else None,
                                        "assignment_group": assignment_group,
                                        "assignment_source": assignment_source,
                                        "assignment_sample_pct": assignment_sample_pct,
                                    },
                                },
                                commit=False,
                            )
                            decisions_written += 1
                            if experiment_row is not None:
                                await upsert_strategy_experiment_assignment(
                                    session,
                                    experiment_id=str(experiment_row.id),
                                    trader_id=trader_id,
                                    signal_id=signal_id,
                                    source_key=signal_source,
                                    strategy_key=resolved_strategy_key,
                                    strategy_version=int(resolved_strategy_version or 1),
                                    assignment_group=str(assignment_group or "control"),
                                    decision_id=decision_row.id,
                                    payload={
                                        "sample_pct": assignment_sample_pct,
                                        "assignment_source": assignment_source,
                                    },
                                    commit=False,
                                )
                            await create_trader_decision_checks(
                                session,
                                decision_id=decision_row.id,
                                checks=checks_payload,
                                commit=False,
                            )
                            await set_trade_signal_status(
                                session,
                                signal_id=signal_id,
                                status="skipped",
                                commit=False,
                            )
                            await record_signal_consumption(
                                session,
                                trader_id=trader_id,
                                signal_id=signal_id,
                                decision_id=decision_row.id,
                                outcome="skipped",
                                reason="Signal excluded by traders_scope",
                                commit=False,
                            )
                            await create_trader_event(
                                session,
                                trader_id=trader_id,
                                event_type="decision",
                                source=signal_source,
                                message="Signal excluded by traders_scope",
                                payload={
                                    "decision_id": decision_row.id,
                                    "signal_id": signal.id,
                                    "decision": "skipped",
                                    "order_status": None,
                                    "traders_scope": scope_payload,
                                },
                                commit=False,
                            )
                            await upsert_trader_signal_cursor(
                                session,
                                trader_id=trader_id,
                                last_signal_created_at=_signal_cursor_timestamp(signal),
                                last_signal_id=signal_id,
                                commit=False,
                            )
                            await _commit_with_retry(session)
                            cursor_created_at = _signal_cursor_timestamp(signal)
                            cursor_signal_id = signal_id
                            cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                            processed_signals += 1
                            continue

                    copy_risk_context: dict[str, Any] = {}
                    copy_allocation_context: dict[str, Any] = {}
                    copy_inventory_context_for_signal: dict[str, Any] = {}
                    if signal_source == "traders" and resolved_strategy_key == "traders_copy_trade":
                        runtime_signal_payload = (
                            runtime_signal.payload_json
                            if isinstance(getattr(runtime_signal, "payload_json", None), dict)
                            else {}
                        )
                        runtime_strategy_context = (
                            runtime_signal_payload.get("strategy_context")
                            if isinstance(runtime_signal_payload.get("strategy_context"), dict)
                            else {}
                        )
                        runtime_copy_event = (
                            runtime_strategy_context.get("copy_event")
                            if isinstance(runtime_strategy_context.get("copy_event"), dict)
                            else {}
                        )
                        runtime_source_trade = (
                            runtime_signal_payload.get("source_trade")
                            if isinstance(runtime_signal_payload.get("source_trade"), dict)
                            else {}
                        )
                        copy_signal_side = str(
                            runtime_copy_event.get("side")
                            or runtime_source_trade.get("side")
                            or ""
                        ).strip().upper()
                        source_wallet = StrategySDK.extract_primary_trader_signal_wallet(runtime_signal)
                        trader_total_daily_pnl = float(trader_daily_pnl + trader_unrealized_pnl)
                        configured_daily_loss_cap_usd = abs(
                            safe_float(
                                effective_risk_limits.get("max_daily_loss_usd"),
                                safe_float(global_limits.get("max_daily_loss_usd"), 0.0),
                            )
                            or 0.0
                        )
                        trader_drawdown_pct = None
                        trader_metadata = trader.get("metadata") if isinstance(trader.get("metadata"), dict) else {}
                        for drawdown_key in (
                            "copy_drawdown_pct",
                            "drawdown_pct",
                            "account_drawdown_pct",
                        ):
                            if drawdown_key not in trader_metadata:
                                continue
                            trader_drawdown_pct = safe_float(trader_metadata.get(drawdown_key), None)
                            if trader_drawdown_pct is not None:
                                trader_drawdown_pct = max(0.0, trader_drawdown_pct)
                                break
                        if trader_drawdown_pct is None and configured_daily_loss_cap_usd > 0.0:
                            trader_drawdown_pct = max(
                                0.0,
                                (-trader_total_daily_pnl / configured_daily_loss_cap_usd) * 100.0,
                            )
                        trader_daily_loss_usd = max(0.0, -float(trader_total_daily_pnl))
                        copy_risk_context = {
                            "trader_drawdown_pct": trader_drawdown_pct,
                            "trader_daily_loss_usd": trader_daily_loss_usd,
                            "trader_total_daily_pnl_usd": trader_total_daily_pnl,
                            "configured_daily_loss_cap_usd": configured_daily_loss_cap_usd,
                        }

                        current_source_exposure_usd = await get_trader_source_exposure(
                            session,
                            trader_id=trader_id,
                            source=signal_source,
                            mode=run_mode,
                        )
                        current_leader_exposure_usd = (
                            await get_trader_copy_leader_exposure(
                                session,
                                trader_id=trader_id,
                                source_wallet=source_wallet,
                                mode=run_mode,
                            )
                            if source_wallet
                            else 0.0
                        )
                        copy_allocation_context = {
                            "source_wallet": source_wallet,
                            "current_source_exposure_usd": float(current_source_exposure_usd),
                            "current_leader_exposure_usd": float(current_leader_exposure_usd),
                        }

                        if run_mode == "live":
                            if not copy_inventory_loaded or copy_signal_side == "SELL":
                                copy_inventory_loaded = True
                                try:
                                    wallet_snapshot = await list_live_wallet_positions_for_trader(
                                        session,
                                        trader_id=trader_id,
                                        include_managed=True,
                                    )
                                except Exception as exc:
                                    copy_inventory_context = {
                                        "available": False,
                                        "wallet_address": None,
                                        "token_inventory": {},
                                        "error": str(exc),
                                        "fetched_at": utcnow().isoformat(),
                                    }
                                else:
                                    token_inventory: dict[str, dict[str, Any]] = {}
                                    for raw_position in wallet_snapshot.get("positions") or []:
                                        if not isinstance(raw_position, dict):
                                            continue
                                        token_key = str(raw_position.get("token_id") or "").strip().lower()
                                        if not token_key:
                                            continue
                                        size = max(0.0, safe_float(raw_position.get("size"), 0.0))
                                        existing_token = token_inventory.get(token_key)
                                        if existing_token is None:
                                            token_inventory[token_key] = {
                                                "size": size,
                                                "market_id": str(raw_position.get("market_id") or "").strip(),
                                                "outcome": str(raw_position.get("outcome") or "").strip().upper() or None,
                                                "direction": str(raw_position.get("direction") or "").strip().lower() or None,
                                                "current_price": safe_float(raw_position.get("current_price"), None),
                                            }
                                        else:
                                            existing_token["size"] = float(existing_token.get("size", 0.0) or 0.0) + size
                                    copy_inventory_context = {
                                        "available": bool(token_inventory),
                                        "wallet_address": wallet_snapshot.get("wallet_address"),
                                        "token_inventory": token_inventory,
                                        "error": None,
                                        "fetched_at": utcnow().isoformat(),
                                    }
                            copy_inventory_context_for_signal = copy_inventory_context

                    loop = asyncio.get_running_loop()
                    evaluation_timeout_seconds = _strategy_evaluation_timeout_seconds(
                        control,
                        source_config,
                        strategy_params,
                    )
                    decision_future = loop.run_in_executor(
                        None,
                        strategy.evaluate,
                        runtime_signal,
                        {
                            "params": strategy_params,
                            "trader": trader,
                            "mode": control.get("mode", "shadow"),
                            "live_market": live_context,
                            "source_config": source_config,
                            "traders_scope_context": (
                                traders_scope_context if signal_source == "traders" else None
                            ),
                            "edge_calibration": edge_calibration_profile,
                            "copy_risk_context": copy_risk_context,
                            "copy_allocation_context": copy_allocation_context,
                            "copy_inventory_context": copy_inventory_context_for_signal,
                        },
                    )
                    async with release_conn(session):
                        try:
                            decision_obj = await asyncio.wait_for(
                                decision_future,
                                timeout=evaluation_timeout_seconds,
                            )
                        except asyncio.TimeoutError as exc:
                            if not decision_future.done():
                                decision_future.cancel()
                            raise RuntimeError(
                                f"Strategy evaluation timed out after {evaluation_timeout_seconds:.1f}s"
                            ) from exc
                        except asyncio.CancelledError:
                            if not decision_future.done():
                                decision_future.cancel()
                            raise
                    checks_payload = _checks_to_payload(decision_obj.checks)

                    if live_context:
                        live_price = live_context.get("live_selected_price")
                        checks_payload.append(
                            {
                                "check_key": "live_market_price",
                                "check_label": "Live market price",
                                "passed": live_price is not None,
                                "score": live_price,
                                "detail": (
                                    "Using live selected-outcome midpoint"
                                    if live_price is not None
                                    else "Live selected-outcome midpoint unavailable"
                                ),
                                "payload": {
                                    "selected_outcome": live_context.get("selected_outcome"),
                                    "fetched_at": live_context.get("fetched_at"),
                                },
                            }
                        )
                        drift_pct = live_context.get("entry_price_delta_pct")
                        drift_score = safe_float(drift_pct)
                        max_drift = safe_float(
                            effective_risk_limits.get("max_entry_drift_pct"),
                            10.0,
                        )
                        checks_payload.append(
                            {
                                "check_key": "live_entry_drift",
                                "check_label": "Entry drift from signal",
                                "passed": drift_score is None or abs(drift_score) <= max_drift,
                                "score": drift_score,
                                "detail": (
                                    f"drift={drift_score:.2f}%"
                                    if drift_score is not None
                                    else "Signal entry unavailable; drift skipped"
                                ),
                                "payload": {
                                    "signal_entry_price": live_context.get("signal_entry_price"),
                                    "live_selected_price": live_context.get("live_selected_price"),
                                    "adverse_price_move": live_context.get("adverse_price_move"),
                                },
                            }
                        )

                    risk_runtime_payload = {
                        "global_daily_realized_pnl_usd": global_daily_pnl,
                        "trader_daily_realized_pnl_usd": trader_daily_pnl,
                        "intra_cycle_committed_usd": intra_cycle_committed_usd,
                        "trader_consecutive_losses": trader_loss_streak,
                        "cooldown_seconds": cooldown_seconds,
                        "cooldown_active": cooldown_active,
                        "cooldown_remaining_seconds": cooldown_remaining_seconds,
                        "trader_open_positions": effective_open_positions,
                        "trader_open_orders": open_order_count,
                        "pending_live_exit_count": pending_live_exit_count,
                        "pending_live_exit_statuses": pending_live_exit_summary.get("statuses") or {},
                        "live_risk_clamps": live_risk_clamp_changes,
                        "trading_schedule_ok": trading_schedule_ok,
                    }

                    risk_evaluator = None
                    portfolio_allocator = None
                    portfolio_runtime_payload: dict[str, Any] = {}
                    if decision_obj.decision == "selected" and trading_schedule_ok:
                        gross_exposure = await get_gross_exposure(session, mode=run_mode)
                        market_exposure = await get_market_exposure(
                            session,
                            str(signal.market_id),
                            mode=run_mode,
                        )
                        adjusted_global_daily_pnl = global_daily_pnl - intra_cycle_committed_usd
                        adjusted_trader_daily_pnl = trader_daily_pnl - intra_cycle_committed_usd
                        risk_snapshot_base = {
                            "global_daily_realized_pnl_usd": global_daily_pnl,
                            "trader_daily_realized_pnl_usd": trader_daily_pnl,
                            "global_unrealized_pnl_usd": global_unrealized_pnl,
                            "trader_unrealized_pnl_usd": trader_unrealized_pnl,
                            "intra_cycle_committed_usd": intra_cycle_committed_usd,
                            "adjusted_global_daily_pnl_usd": adjusted_global_daily_pnl,
                            "adjusted_trader_daily_pnl_usd": adjusted_trader_daily_pnl,
                            "trader_consecutive_losses": trader_loss_streak,
                            "cooldown_seconds": cooldown_seconds,
                            "cooldown_active": cooldown_active,
                            "cooldown_remaining_seconds": cooldown_remaining_seconds,
                            "trader_open_positions": effective_open_positions,
                            "trader_open_orders": open_order_count,
                            "pending_live_exit_count": pending_live_exit_count,
                            "pending_live_exit_statuses": pending_live_exit_summary.get("statuses") or {},
                        }

                        def _evaluate_runtime_risk(size_for_eval: float):
                            risk_result = evaluate_risk(
                                size_usd=size_for_eval,
                                gross_exposure_usd=gross_exposure,
                                trader_open_positions=effective_open_positions,
                                trader_open_orders=open_order_count,
                                market_exposure_usd=market_exposure,
                                global_limits=global_limits,
                                trader_limits=effective_risk_limits,
                                global_daily_realized_pnl_usd=adjusted_global_daily_pnl,
                                trader_daily_realized_pnl_usd=adjusted_trader_daily_pnl,
                                global_unrealized_pnl_usd=global_unrealized_pnl,
                                trader_unrealized_pnl_usd=trader_unrealized_pnl,
                                trader_consecutive_losses=trader_loss_streak,
                                cycle_orders_placed=orders_written,
                                cooldown_active=cooldown_active,
                                mode=run_mode,
                            )
                            return risk_result, dict(risk_snapshot_base)

                        risk_evaluator = _evaluate_runtime_risk

                        portfolio_config = _normalize_portfolio_config(effective_risk_limits)
                        if bool(portfolio_config.get("enabled", False)):
                            source_exposure = await get_trader_source_exposure(
                                session,
                                trader_id=trader_id,
                                source=signal_source,
                                mode=run_mode,
                            )
                            gross_exposure_cap = safe_float(
                                effective_risk_limits.get("max_gross_exposure_usd"),
                                safe_float(global_limits.get("max_gross_exposure_usd"), 5000.0),
                            )
                            portfolio_runtime_payload = {
                                "source_key": signal_source,
                                "gross_exposure_cap_usd": gross_exposure_cap,
                                "current_gross_exposure_usd": gross_exposure,
                                "current_source_exposure_usd": source_exposure,
                                "target_utilization_pct": portfolio_config["target_utilization_pct"],
                                "max_source_exposure_pct": portfolio_config["max_source_exposure_pct"],
                                "min_order_notional_usd": portfolio_config["min_order_notional_usd"],
                            }

                            def _evaluate_portfolio(size_for_eval: float):
                                return _allocate_portfolio_notional(
                                    requested_size_usd=size_for_eval,
                                    gross_exposure_cap_usd=gross_exposure_cap,
                                    current_gross_exposure_usd=gross_exposure,
                                    current_source_exposure_usd=source_exposure,
                                    source_key=signal_source,
                                    portfolio_config=portfolio_config,
                                )

                            portfolio_allocator = _evaluate_portfolio

                    gate_result = apply_platform_decision_gates(
                        decision_obj=decision_obj,
                        runtime_signal=runtime_signal,
                        strategy=strategy,
                        checks_payload=checks_payload,
                        trading_schedule_ok=trading_schedule_ok,
                        trading_schedule_config=metadata.get("trading_schedule_utc"),
                        global_limits=global_limits,
                        effective_risk_limits=effective_risk_limits,
                        allow_averaging=allow_averaging,
                        open_market_ids=open_market_ids,
                        pending_live_exit_count=pending_live_exit_count,
                        pending_live_exit_summary=pending_live_exit_summary,
                        pending_live_exit_max_allowed=pending_live_exit_max_allowed,
                        pending_live_exit_identity_guard_enabled=pending_live_exit_identity_guard_enabled,
                        portfolio_allocator=portfolio_allocator,
                        risk_evaluator=risk_evaluator,
                        invoke_hooks=True,
                        strategy_params=strategy_params,
                    )
                    final_decision = gate_result["final_decision"]
                    final_reason = gate_result["final_reason"]
                    score = gate_result["score"]
                    size_usd = gate_result["size_usd"]
                    checks_payload = gate_result["checks_payload"]
                    risk_snapshot = gate_result["risk_snapshot"]
                    strategy_payload = (
                        dict(decision_obj.payload) if isinstance(decision_obj.payload, dict) else {}
                    )
                    execution_plan_override = strategy_payload.get("execution_plan_override")
                    execution_plan_override_applied = False
                    if final_decision == "selected" and isinstance(execution_plan_override, dict):
                        override_legs = execution_plan_override.get("legs")
                        if isinstance(override_legs, list) and override_legs:
                            runtime_payload = (
                                dict(runtime_signal.payload_json)
                                if isinstance(getattr(runtime_signal, "payload_json", None), dict)
                                else {}
                            )
                            runtime_payload["execution_plan"] = dict(execution_plan_override)
                            runtime_signal.payload_json = runtime_payload
                            execution_plan_override_applied = True

                    decision_ready_at = utcnow()
                    decision_latency_payload = _compute_signal_latency_payload(
                        runtime_signal,
                        live_context=live_context,
                        measured_at=decision_ready_at,
                    )

                    decision_row = await create_trader_decision(
                        session,
                        trader_id=trader_id,
                        signal=runtime_signal,
                        strategy_key=resolved_strategy_key,
                        strategy_version=resolved_strategy_version,
                        decision=final_decision,
                        reason=final_reason,
                        score=score,
                        checks_summary={"count": len(checks_payload)},
                        risk_snapshot=risk_snapshot,
                        payload={
                            "source_key": signal_source,
                            "source_config": source_config,
                            "strategy_payload": strategy_payload,
                            "strategy_decision": {
                                "decision": gate_result["strategy_decision"],
                                "reason": gate_result["strategy_reason"],
                            },
                            "execution_plan_override_applied": execution_plan_override_applied,
                            "latency": decision_latency_payload,
                            "platform_gates": gate_result["platform_gates"],
                            "size_usd": size_usd,
                            "traders_scope": traders_scope_payload,
                            "edge_calibration": edge_calibration_profile,
                            "live_market": {
                                "available": bool(live_context.get("available")),
                                "fetched_at": live_context.get("fetched_at"),
                                "live_market_fetched_at": live_context.get("live_market_fetched_at"),
                                "selected_outcome": live_context.get("selected_outcome"),
                                "live_selected_price": live_context.get("live_selected_price"),
                                "market_data_source": live_context.get("market_data_source"),
                                "source_observed_at": live_context.get("source_observed_at"),
                                "market_data_age_ms": live_context.get("market_data_age_ms"),
                                "signal_entry_price": live_context.get("signal_entry_price"),
                                "entry_price_delta": live_context.get("entry_price_delta"),
                                "entry_price_delta_pct": live_context.get("entry_price_delta_pct"),
                                "live_edge_percent": live_context.get("live_edge_percent"),
                                "history_summary": live_context.get("history_summary") or {},
                                "history_tail": live_context.get("history_tail") or [],
                            },
                            "risk_runtime": {
                                "global_daily_realized_pnl_usd": risk_runtime_payload["global_daily_realized_pnl_usd"],
                                "trader_daily_realized_pnl_usd": risk_runtime_payload["trader_daily_realized_pnl_usd"],
                                "intra_cycle_committed_usd": risk_runtime_payload["intra_cycle_committed_usd"],
                                "trader_consecutive_losses": risk_runtime_payload["trader_consecutive_losses"],
                                "cooldown_seconds": risk_runtime_payload["cooldown_seconds"],
                                "cooldown_active": risk_runtime_payload["cooldown_active"],
                                "cooldown_remaining_seconds": risk_runtime_payload["cooldown_remaining_seconds"],
                                "trader_open_positions": risk_runtime_payload["trader_open_positions"],
                                "trader_open_orders": risk_runtime_payload["trader_open_orders"],
                                "pending_live_exit_count": risk_runtime_payload["pending_live_exit_count"],
                                "pending_live_exit_statuses": risk_runtime_payload["pending_live_exit_statuses"],
                                "live_risk_clamps": risk_runtime_payload["live_risk_clamps"],
                                "trading_schedule_ok": risk_runtime_payload["trading_schedule_ok"],
                            },
                            "copy_runtime": {
                                "risk": copy_risk_context,
                                "allocation": copy_allocation_context,
                                "inventory": {
                                    "available": bool(copy_inventory_context_for_signal.get("available")),
                                    "wallet_address": copy_inventory_context_for_signal.get("wallet_address"),
                                    "token_count": len(copy_inventory_context_for_signal.get("token_inventory") or {}),
                                    "error": copy_inventory_context_for_signal.get("error"),
                                    "fetched_at": copy_inventory_context_for_signal.get("fetched_at"),
                                },
                            },
                            "portfolio_runtime": portfolio_runtime_payload,
                            "experiment": {
                                "id": str(experiment_row.id) if experiment_row is not None else None,
                                "assignment_group": assignment_group,
                                "assignment_source": assignment_source,
                                "assignment_sample_pct": assignment_sample_pct,
                                "resolved_strategy_version": resolved_strategy_version,
                            },
                        },
                        commit=False,
                    )
                    decisions_written += 1
                    if experiment_row is not None:
                        await upsert_strategy_experiment_assignment(
                            session,
                            experiment_id=str(experiment_row.id),
                            trader_id=trader_id,
                            signal_id=signal_id,
                            source_key=signal_source,
                            strategy_key=resolved_strategy_key,
                            strategy_version=int(resolved_strategy_version or 1),
                            assignment_group=str(assignment_group or "control"),
                            decision_id=decision_row.id,
                            payload={
                                "sample_pct": assignment_sample_pct,
                                "assignment_source": assignment_source,
                                "decision": final_decision,
                            },
                            commit=False,
                        )

                    await create_trader_decision_checks(
                        session,
                        decision_id=decision_row.id,
                        checks=checks_payload,
                        commit=False,
                    )
                    freshness_gate = next(
                        (
                            gate
                            for gate in gate_result["platform_gates"]
                            if isinstance(gate, dict) and str(gate.get("gate") or "") == "market_data_freshness"
                        ),
                        None,
                    )
                    live_revalidation_gate = next(
                        (
                            gate
                            for gate in gate_result["platform_gates"]
                            if isinstance(gate, dict) and str(gate.get("gate") or "") == "live_market_revalidation"
                        ),
                        None,
                    )
                    freshness_payload = (
                        dict(freshness_gate.get("payload") or {})
                        if isinstance(freshness_gate, dict) and isinstance(freshness_gate.get("payload"), dict)
                        else {}
                    )
                    freshness_status = str((freshness_gate or {}).get("status") or "").strip().lower()
                    if freshness_status not in {"passed", "blocked"}:
                        live_revalidation_status = str((live_revalidation_gate or {}).get("status") or "").strip().lower()
                        if live_revalidation_status == "blocked":
                            freshness_status = "blocked"
                            if not freshness_payload:
                                freshness_payload = (
                                    dict(live_revalidation_gate.get("payload") or {})
                                    if isinstance(live_revalidation_gate, dict)
                                    and isinstance(live_revalidation_gate.get("payload"), dict)
                                    else {}
                                )
                    freshness_source = str(freshness_payload.get("source") or signal_source or "").strip().lower()
                    if freshness_status in {"passed", "blocked"} and freshness_source in {"scanner", "crypto"}:
                        freshness_age_ms = safe_float(freshness_payload.get("age_ms"), None)
                        freshness_max_age_ms = safe_float(freshness_payload.get("max_age_ms"), None)
                        freshness_severity = "warn" if freshness_status == "blocked" else "info"
                        freshness_message = (
                            f"Market data freshness {freshness_status}: source={freshness_source or 'unknown'} "
                            f"age_ms={freshness_age_ms if freshness_age_ms is not None else 'unknown'} "
                            f"max={freshness_max_age_ms if freshness_max_age_ms is not None else 'unknown'}"
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="market_data_freshness_source",
                            severity=freshness_severity,
                            source=freshness_source or signal_source,
                            message=freshness_message,
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal_id,
                                "status": freshness_status,
                                "source": freshness_source or None,
                                "timeframe": freshness_payload.get("timeframe"),
                                "age_ms": freshness_age_ms,
                                "max_age_ms": freshness_max_age_ms,
                                "market_data_source": freshness_payload.get("market_data_source"),
                                "required_sources": freshness_payload.get("required_sources"),
                                "freshness_enforced": freshness_payload.get("freshness_enforced"),
                                "final_decision": final_decision,
                                "final_reason": final_reason,
                            },
                            commit=False,
                        )

                    order_status = None
                    if final_decision == "selected":
                        submit_started_at = utcnow()
                        async with release_conn(session):
                            submit_result = await submit_order(
                                session_engine=session_engine,
                                trader_id=trader_id,
                                signal=runtime_signal,
                                decision_id=decision_row.id,
                                strategy_key=resolved_strategy_key,
                                strategy_version=resolved_strategy_version,
                                strategy_params=strategy_params,
                                risk_limits=effective_risk_limits,
                                mode=str(control.get("mode", "shadow")),
                                size_usd=size_usd,
                                reason=final_reason,
                            )
                        submit_completed_at = utcnow()
                        submit_latency_payload = _compute_signal_latency_payload(
                            runtime_signal,
                            live_context=live_context,
                            measured_at=submit_completed_at,
                        )
                        submit_latency_payload["decision_to_submit_ms"] = max(
                            0,
                            int((submit_completed_at - submit_started_at).total_seconds() * 1000),
                        )
                        latency_sample = _build_execution_latency_sample(
                            runtime_signal,
                            wake_started_at=signal_wake_started_at,
                            context_ready_at=context_ready_at,
                            decision_ready_at=decision_ready_at,
                            submit_started_at=submit_started_at,
                            submit_ack_at=submit_completed_at,
                        )
                        await execution_latency_metrics.record(
                            trader_id=trader_id,
                            source=signal_source,
                            strategy_key=resolved_strategy_key,
                            payload=latency_sample,
                        )
                        submit_latency_payload.update(
                            {
                                "armed_to_ws_release_ms": latency_sample.get("armed_to_ws_release_ms"),
                                "emit_to_queue_wake_ms": latency_sample.get("emit_to_queue_wake_ms"),
                                "ws_release_to_decision_ms": latency_sample.get("ws_release_to_decision_ms"),
                                "ws_release_to_submit_start_ms": latency_sample.get("ws_release_to_submit_start_ms"),
                                "wake_to_context_ready_ms": latency_sample.get("wake_to_context_ready_ms"),
                                "context_ready_to_decision_ms": latency_sample.get("context_ready_to_decision_ms"),
                                "decision_to_submit_start_ms": latency_sample.get("decision_to_submit_start_ms"),
                                "submit_start_to_provider_ack_ms": latency_sample.get("submit_start_to_provider_ack_ms"),
                                "emit_to_submit_start_ms": latency_sample.get("emit_to_submit_start_ms"),
                            }
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="execution_latency",
                            severity="info",
                            source=signal_source,
                            message=(
                                "Hot-path execution latency "
                                f"armed_to_release_ms={submit_latency_payload.get('armed_to_ws_release_ms', 'unknown')} "
                                f"release_to_submit_ms={submit_latency_payload.get('ws_release_to_submit_start_ms', 'unknown')} "
                                f"provider_ack_ms={submit_latency_payload.get('submit_start_to_provider_ack_ms', 'unknown')}"
                            ),
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal_id,
                                "strategy_key": resolved_strategy_key,
                                "latency": submit_latency_payload,
                            },
                            commit=False,
                        )
                        execution_session_id = None
                        execution_error_message = None
                        if isinstance(submit_result, tuple):
                            normalized_order_status = str(submit_result[0] or "").strip().lower()
                            order_status = normalized_order_status
                            execution_error_message = (
                                str(submit_result[2] or "").strip() if len(submit_result) > 2 else ""
                            ) or None
                            if normalized_order_status == "skipped":
                                final_decision = "skipped"
                                skip_reason = str(submit_result[2] or "").strip() if len(submit_result) > 2 else ""
                                if skip_reason:
                                    final_reason = skip_reason
                                await update_trader_decision(
                                    session,
                                    decision_id=decision_row.id,
                                    decision="skipped",
                                    reason=final_reason,
                                    payload_patch={
                                        "execution_status": normalized_order_status,
                                        "execution_skip_reason": final_reason,
                                    },
                                    commit=False,
                                )
                            elif normalized_order_status in {"failed", "cancelled", "expired"}:
                                final_decision = "failed"
                                if execution_error_message:
                                    final_reason = execution_error_message
                                await update_trader_decision(
                                    session,
                                    decision_id=decision_row.id,
                                    decision="failed",
                                    reason=final_reason,
                                    payload_patch={
                                        "execution_status": normalized_order_status,
                                        "execution_error": execution_error_message,
                                    },
                                    commit=False,
                                )
                            if normalized_order_status in {
                                "executed",
                                "completed",
                                "working",
                                "hedging",
                                "partial",
                                "open",
                            }:
                                orders_written += 1
                            if normalized_order_status in {
                                "executed",
                                "completed",
                                "working",
                                "hedging",
                                "partial",
                                "open",
                                "submitted",
                            }:
                                open_positions = await get_open_position_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                    position_cap_scope=position_cap_scope,
                                )
                                open_order_count = await get_open_order_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                )
                                effective_open_positions = max(open_positions, open_order_count)
                                opened_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                                if opened_market_id:
                                    open_market_ids.add(opened_market_id)
                                intra_cycle_committed_usd += size_usd
                        else:
                            normalized_order_status = str(submit_result.status or "").strip().lower()
                            order_status = normalized_order_status
                            execution_session_id = str(submit_result.session_id or "").strip() or None
                            execution_error_message = str(submit_result.error_message or "").strip() or None
                            orders_written += int(submit_result.orders_written or 0)
                            if normalized_order_status == "skipped":
                                skip_reason = str(submit_result.error_message or "").strip()
                                final_decision = "skipped"
                                if skip_reason:
                                    final_reason = skip_reason
                                await update_trader_decision(
                                    session,
                                    decision_id=decision_row.id,
                                    decision="skipped",
                                    reason=final_reason,
                                    payload_patch={
                                        "execution_status": normalized_order_status,
                                        "execution_skip_reason": final_reason,
                                        "execution_session_id": str(submit_result.session_id or ""),
                                    },
                                    commit=False,
                                )
                            elif normalized_order_status in {"failed", "cancelled", "expired"}:
                                final_decision = "failed"
                                if execution_error_message:
                                    final_reason = execution_error_message
                                await update_trader_decision(
                                    session,
                                    decision_id=decision_row.id,
                                    decision="failed",
                                    reason=final_reason,
                                    payload_patch={
                                        "execution_status": normalized_order_status,
                                        "execution_error": execution_error_message,
                                        "execution_session_id": str(submit_result.session_id or ""),
                                    },
                                    commit=False,
                                )

                            if normalized_order_status in {
                                "executed",
                                "completed",
                                "working",
                                "hedging",
                                "partial",
                                "open",
                                "submitted",
                            }:
                                for _co in getattr(submit_result, "created_orders", None) or []:
                                    hot_state.record_order_created(
                                        trader_id=trader_id,
                                        mode=run_mode,
                                        order_id=str(_co.get("order_id", "")),
                                        status=normalized_order_status,
                                        market_id=str(_co.get("market_id", "")),
                                        direction=str(_co.get("direction", "")),
                                        source=str(_co.get("source", "")),
                                        notional_usd=float(_co.get("notional_usd", 0.0)),
                                        entry_price=float(_co.get("entry_price", 0.0)),
                                        token_id=str(_co.get("token_id", "")),
                                        filled_shares=float(_co.get("filled_shares", 0.0)),
                                        payload=dict(_co.get("payload") or {}),
                                    )
                                open_positions = await get_open_position_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                    position_cap_scope=position_cap_scope,
                                )
                                open_order_count = await get_open_order_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                )
                                effective_open_positions = max(open_positions, open_order_count)
                                opened_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                                if opened_market_id:
                                    open_market_ids.add(opened_market_id)
                                intra_cycle_committed_usd += size_usd
                        if final_decision == "failed" and order_status in {"failed", "cancelled", "expired"}:
                            await create_trader_event(
                                session,
                                trader_id=trader_id,
                                event_type="execution_failed",
                                severity="error",
                                source=signal_source,
                                message=final_reason,
                                payload={
                                    "decision_id": decision_row.id,
                                    "signal_id": signal_id,
                                    "strategy_key": resolved_strategy_key,
                                    "order_status": order_status,
                                    "execution_session_id": execution_session_id,
                                    "error": execution_error_message,
                                    "market_id": getattr(runtime_signal, "market_id", None),
                                    "market_question": getattr(runtime_signal, "market_question", None),
                                    "direction": getattr(runtime_signal, "direction", None),
                                },
                                commit=False,
                            )
                    else:
                        signal_status = "failed" if final_decision == "failed" else "skipped"
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status=signal_status,
                            commit=False,
                        )

                    await record_signal_consumption(
                        session,
                        trader_id=trader_id,
                        signal_id=signal_id,
                        decision_id=decision_row.id,
                        outcome=order_status or final_decision,
                        reason=final_reason,
                        commit=False,
                    )

                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="decision",
                        severity="error" if final_decision == "failed" else "info",
                        source=signal_source,
                        message=final_reason,
                        payload={
                            "decision_id": decision_row.id,
                            "signal_id": signal.id,
                            "decision": final_decision,
                            "order_status": order_status,
                            "market_id": getattr(runtime_signal, "market_id", None),
                            "market_question": getattr(runtime_signal, "market_question", None),
                            "direction": getattr(runtime_signal, "direction", None),
                        },
                        commit=False,
                    )
                    await upsert_trader_signal_cursor(
                        session,
                        trader_id=trader_id,
                        last_signal_created_at=_signal_cursor_timestamp(signal),
                        last_signal_id=signal_id,
                        commit=False,
                    )
                    await _commit_with_retry(session)
                    cursor_created_at = _signal_cursor_timestamp(signal)
                    cursor_signal_id = signal_id
                    cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                    processed_signals += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    try:
                        await session.rollback()
                    except Exception:
                        logger.warning(
                            "Trader %s session rollback failed after signal %s error; session may be invalidated",
                            trader_id, signal_id,
                        )
                    logger.exception(
                        "Trader %s failed to process signal %s",
                        trader_id,
                        signal_id,
                    )
                    error_type = exc.__class__.__name__
                    error_message = str(exc or "").strip() or error_type
                    strategy_for_error = resolved_strategy_key or strategy_key or "unknown_strategy"
                    strategy_version_for_error = resolved_strategy_version
                    failure_reason = f"Signal processing failed ({error_type})"
                    try:
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=signal,
                            strategy_key=strategy_for_error,
                            strategy_version=strategy_version_for_error,
                            decision="failed",
                            reason=failure_reason,
                            score=0.0,
                            checks_summary={"count": 0},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config or {},
                                "error_type": error_type,
                                "error_message": error_message,
                            },
                            commit=False,
                        )
                        decisions_written += 1
                        if experiment_row is not None:
                            await upsert_strategy_experiment_assignment(
                                session,
                                experiment_id=str(experiment_row.id),
                                trader_id=trader_id,
                                signal_id=signal_id,
                                source_key=signal_source,
                                strategy_key=strategy_for_error,
                                strategy_version=int(strategy_version_for_error or experiment_row.control_version or 1),
                                assignment_group=str(assignment_group or "control"),
                                decision_id=decision_row.id,
                                payload={
                                    "sample_pct": assignment_sample_pct,
                                    "assignment_source": assignment_source,
                                    "error_type": error_type,
                                },
                                commit=False,
                            )
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status="failed",
                            commit=False,
                        )
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=decision_row.id,
                            outcome="failed",
                            reason=failure_reason,
                            payload={
                                "error_type": error_type,
                                "error_message": error_message,
                            },
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="decision_error",
                            severity="warn",
                            source=signal_source,
                            message="Signal processing failed; advanced cursor to avoid hard loop.",
                            payload={
                                "signal_id": signal_id,
                                "decision_id": decision_row.id,
                                "error_type": error_type,
                                "error_message": error_message,
                            },
                            commit=False,
                        )
                        await _commit_with_retry(session)
                    except Exception as recovery_exc:
                        logger.warning(
                            "Trader %s failed to record error decision for signal %s: %s",
                            trader_id, signal_id, recovery_exc,
                        )
                        # Buffer the failure via hot state as a last resort
                        await hot_state.buffer_signal_consumption(
                            trader_id=trader_id, signal_id=signal_id,
                            outcome="failed", reason=failure_reason,
                        )
                    cursor_created_at = _signal_cursor_timestamp(signal)
                    cursor_signal_id = signal_id
                    cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                    processed_signals += 1

            if process_signals:
                if decisions_written == 0 and orders_written == 0:
                    if deferred_signals > 0:
                        await _emit_cycle_heartbeat_if_due(
                            session,
                            trader_id=trader_id,
                            message="Idle cycle: pending signals deferred awaiting live market context.",
                            payload={
                                "processed_signals": processed_signals,
                                "decisions_written": decisions_written,
                                "orders_written": orders_written,
                                "deferred_signals": deferred_signals,
                                "deferred_by_reason": deferred_by_reason,
                            },
                        )
                    elif prefiltered_signals > 0:
                        payload = {
                            "processed_signals": processed_signals,
                            "decisions_written": decisions_written,
                            "orders_written": orders_written,
                            "prefiltered_signals": prefiltered_signals,
                            "prefiltered_by_reason": prefiltered_by_reason,
                        }
                        if crypto_scope_prefiltered_dimensions:
                            payload["crypto_scope_prefiltered_dimensions"] = crypto_scope_prefiltered_dimensions
                        await _emit_cycle_heartbeat_if_due(
                            session,
                            trader_id=trader_id,
                            message="Idle cycle: pending signals filtered before strategy evaluation.",
                            payload=payload,
                        )
                elif processed_signals == 0:
                    if not stream_trigger_mode:
                        await _emit_cycle_heartbeat_if_due(
                            session,
                            trader_id=trader_id,
                            message="Idle cycle: no pending signals.",
                            payload={
                                "processed_signals": processed_signals,
                                "decisions_written": decisions_written,
                                "orders_written": orders_written,
                            },
                        )
            if stream_trigger_mode and prefetched_signals:
                deferred_signal_ids_by_source: dict[str, list[str]] = {}
                deferred_seen_ids_by_source: dict[str, set[str]] = {}
                for deferred_signal in prefetched_signals:
                    deferred_source = normalize_source_key(getattr(deferred_signal, "source", "")) or "__all__"
                    deferred_signal_id = str(getattr(deferred_signal, "id", "") or "").strip()
                    if not deferred_signal_id:
                        continue
                    deferred_signal_ids_by_source.setdefault(deferred_source, [])
                    deferred_seen_ids_by_source.setdefault(deferred_source, set())
                    if deferred_signal_id in deferred_seen_ids_by_source[deferred_source]:
                        continue
                    deferred_seen_ids_by_source[deferred_source].add(deferred_signal_id)
                    deferred_signal_ids_by_source[deferred_source].append(deferred_signal_id)
                for deferred_source, deferred_signal_ids in deferred_signal_ids_by_source.items():
                    await publish_signal_batch(
                        event_type="upsert_update",
                        source=None if deferred_source == "__all__" else deferred_source,
                        signal_ids=deferred_signal_ids,
                        trigger="runtime_signal_overflow",
                        reason="trader_orchestrator_runtime_trigger_batch_deferred",
                    )
                prefetched_signals = None
            await _persist_trader_cycle_heartbeat(session, trader_id)
            if defer_signal_processing or stream_trigger_mode:
                break

    return decisions_written, orders_written, processed_signals


async def _reconcile_orphan_open_orders(session: Any) -> dict[str, int]:
    """Close/cancel open orders for trader_ids that no longer exist."""
    mode_key_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    status_key_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    rows = (
        await session.execute(
            select(
                TraderOrder.trader_id,
                mode_key_expr.label("mode_key"),
                func.count(TraderOrder.id).label("count"),
            )
            .select_from(TraderOrder)
            .outerjoin(Trader, Trader.id == TraderOrder.trader_id)
            .where(Trader.id.is_(None))
            .where(status_key_expr.in_(tuple(_ACTIVE_ORDER_STATUSES)))
            .group_by(TraderOrder.trader_id, mode_key_expr)
            .limit(_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE)
        )
    ).all()

    if not rows:
        return {
            "traders_seen": 0,
            "rows_seen": 0,
            "shadow_closed": 0,
            "paper_closed": 0,
            "non_shadow_cancelled": 0,
            "non_paper_cancelled": 0,
        }

    per_trader_mode: dict[tuple[str, str], int] = {}
    for row in rows:
        trader_id = str(row.trader_id or "").strip()
        if not trader_id:
            continue
        mode_key = str(row.mode_key or "").strip().lower()
        if mode_key == "":
            mode_key = "other"
        else:
            mode_key = _LEGACY_MODE_ALIASES.get(mode_key, mode_key)
            if mode_key not in _CANONICAL_TRADER_MODES:
                mode_key = "other"
        per_trader_mode[(trader_id, mode_key)] = int(row.count or 0)

    shadow_closed = 0
    non_shadow_cancelled = 0
    for trader_id, mode_key in per_trader_mode:
        if mode_key == "shadow":
            result = await reconcile_paper_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=False,
                force_mark_to_market=False,
                reason="orphan_trader_lifecycle",
            )
            shadow_closed += int(result.get("closed", 0))
            await sync_trader_position_inventory(session, trader_id=trader_id, mode="shadow")
            continue

        cleanup = await cleanup_trader_open_orders(
            session,
            trader_id=trader_id,
            scope="all",
            dry_run=False,
            target_status="cancelled",
            reason="orphan_trader_cleanup",
        )
        non_shadow_cancelled += int(cleanup.get("updated", 0))

    await create_trader_event(
        session,
        trader_id=None,
        event_type="orphan_orders_reconciled",
        severity="warn",
        source="worker",
        message=(
            f"Reconciled orphan trader open orders "
            f"(shadow_closed={shadow_closed}, non_shadow_cancelled={non_shadow_cancelled})"
        ),
        payload={
            "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
            "rows_seen": len(per_trader_mode),
            "shadow_closed": shadow_closed,
            "paper_closed": shadow_closed,
            "non_shadow_cancelled": non_shadow_cancelled,
            "non_paper_cancelled": non_shadow_cancelled,
        },
        commit=True,
    )

    return {
        "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
        "rows_seen": len(per_trader_mode),
        "shadow_closed": shadow_closed,
        "paper_closed": shadow_closed,
        "non_shadow_cancelled": non_shadow_cancelled,
        "non_paper_cancelled": non_shadow_cancelled,
    }


async def _run_trader_once_with_timeout(
    trader: dict[str, Any],
    control: dict[str, Any],
    *,
    process_signals: bool,
    trigger_signal_ids_by_source: dict[str, list[str]] | None,
    trigger_signal_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] | None,
    timeout_seconds: float,
) -> tuple[int, int, int]:
    timeout = max(1.0, float(timeout_seconds))
    trader_id = str(trader.get("id") or "")
    existing = _inflight_trader_cycle_tasks.get(trader_id) if trader_id else None
    if existing is not None and not existing.done():
        logger.warning(
            "Trader cycle skipped because prior run is still finishing for trader=%s process_signals=%s",
            trader_id,
            process_signals,
        )
        return 0, 0, 0

    task = asyncio.create_task(
        _run_trader_once(
            trader,
            control,
            process_signals=process_signals,
            trigger_signal_ids_by_source=trigger_signal_ids_by_source,
            trigger_signal_snapshots_by_source=trigger_signal_snapshots_by_source,
        ),
        name=f"trader-cycle-{trader_id or 'unknown'}",
    )
    if trader_id:
        _inflight_trader_cycle_tasks[trader_id] = task
        task.add_done_callback(
            lambda done_task, current_trader_id=trader_id: _clear_inflight_trader_cycle_task(
                current_trader_id,
                done_task,
            )
        )

    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done:
            return task.result()

        task.cancel()
        done_after, _ = await asyncio.wait({task}, timeout=_TRADER_TIMEOUT_CANCEL_GRACE_SECONDS)
        if done_after:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "Trader cycle raised after timeout cancellation",
                    trader_id=trader_id,
                    process_signals=process_signals,
                    exc_info=exc,
                )
        else:
            _abandoned_trader_cycle_tasks.add(task)
            task.add_done_callback(_discard_abandoned_trader_cycle)
            logger.warning(
                "Trader cycle did not finish within %ss cancel-grace; holding reference until completion trader=%s process_signals=%s",
                _TRADER_TIMEOUT_CANCEL_GRACE_SECONDS,
                trader_id,
                process_signals,
            )
        raise asyncio.TimeoutError()
    except asyncio.TimeoutError:
        logger.warning(
            "Trader cycle timed out for trader=%s process_signals=%s timeout=%.1fs",
            trader_id,
            process_signals,
            timeout,
        )
        if trader_id:
            try:
                async with AsyncSessionLocal() as session:
                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="cycle_timeout",
                        severity="warn",
                        source="worker",
                        message=f"Trader cycle timed out after {timeout:.1f}s",
                        payload={
                            "process_signals": bool(process_signals),
                            "timeout_seconds": timeout,
                        },
                        commit=True,
                    )
            except Exception as exc:
                logger.warning("Failed to persist trader timeout event: %s", exc)
        return 0, 0, 0
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            try:
                await asyncio.shield(asyncio.wait({task}, timeout=_TRADER_TIMEOUT_CANCEL_GRACE_SECONDS))
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "Trader cycle cleanup wait failed after parent cancellation",
                    trader_id=trader_id,
                    exc_info=exc,
                )
            if not task.done():
                _abandoned_trader_cycle_tasks.add(task)
                task.add_done_callback(_discard_abandoned_trader_cycle)
                logger.warning(
                    "Trader cycle parent cancelled during cleanup; holding reference until completion trader=%s",
                    trader_id,
                )
        raise
    except OperationalError as exc:
        if not _is_retryable_db_error(exc):
            raise
        logger.warning(
            "Trader cycle skipped due to transient DB error for trader=%s process_signals=%s",
            trader_id,
            process_signals,
        )
        return 0, 0, 0


async def run_worker_loop(*, lane: str = _LANE_GENERAL, write_snapshot: bool = True) -> None:
    global _ws_auto_paused
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    stream_group = _stream_group_name_for_lane(lane_key)
    logger.info("Starting trader orchestrator worker loop")

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
        await refresh_strategy_runtime_if_needed(source_keys=None, force=True)
    except Exception as exc:
        logger.warning("Orchestrator strategy startup sync failed: %s", exc)

    from services.trader_hot_state import seed as _seed_hot_state
    try:
        await _seed_hot_state()
    except Exception as exc:
        logger.warning("Hot state seed failed at startup: %s", exc)

    runtime_trigger_event: dict[str, Any] | None = None
    stream_consumer_name = f"{lane_key}-{os.getpid()}-{utcnow().strftime('%Y%m%d%H%M%S%f')}"
    stream_claim_cursor = "0-0"
    stream_last_claim_run_at: datetime | None = None
    maintenance_interval_seconds = max(
        0.5,
        float(getattr(settings, "ORCHESTRATOR_MAINTENANCE_INTERVAL_SECONDS", 5.0) or 5.0),
    )
    last_maintenance_at: datetime | None = None

    try:
        while True:
            cycle_interval = ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
            sleep_seconds = 2
            skip_cycle = False
            manual_force_cycle = False
            cycle_trigger = runtime_trigger_event
            runtime_trigger_event = None
            runtime_triggered_cycle = bool(
                isinstance(cycle_trigger, dict)
                and str(cycle_trigger.get("event_type") or "").strip().lower() == "runtime_signal_batch"
            )
            if not runtime_triggered_cycle:
                try:
                    queued_signal_depth = int(get_queue_depth(lane=lane_key) or 0)
                except Exception:
                    queued_signal_depth = 0
                if queued_signal_depth > 0:
                    (
                        cycle_trigger,
                        stream_claim_cursor,
                        stream_last_claim_run_at,
                    ) = await _wait_for_runtime_trigger(
                        None,
                        0.05,
                        consumer=stream_consumer_name,
                        group=stream_group,
                        claim_cursor=stream_claim_cursor,
                        last_claim_run_at=stream_last_claim_run_at,
                    )
                    runtime_triggered_cycle = bool(
                        isinstance(cycle_trigger, dict)
                        and str(cycle_trigger.get("event_type") or "").strip().lower() == "runtime_signal_batch"
                    )
            now_for_maintenance = utcnow()
            maintenance_due = (
                last_maintenance_at is None
                or (now_for_maintenance - last_maintenance_at).total_seconds() >= maintenance_interval_seconds
            )
            run_maintenance = bool(not runtime_triggered_cycle or maintenance_due)
            manage_only_cycle = False
            manage_only_reasons: list[str] = []
            needs_live_execution_init = False
            mode = "shadow"
            control: dict[str, Any] = {}
            traders: list[dict[str, Any]] = []
            try:
                acquired_lock = (
                    await _ensure_orchestrator_cycle_lock_owner()
                    if lane_key == _LANE_GENERAL
                    else await _ensure_orchestrator_cycle_lock_owner_for_lane(lane_key)
                )
                if not acquired_lock:
                    logger.debug(
                        "Orchestrator cycle lock held by another worker instance; waiting to retry",
                    )
                    await _worker_sleep(2)
                    continue

                async with AsyncSessionLocal() as session:
                    control = await read_orchestrator_control(session)
                    manual_force_cycle = _parse_iso(control.get("requested_run_at")) is not None
                    if manual_force_cycle:
                        run_maintenance = False

                    if run_maintenance:
                        try:
                            await refresh_strategy_runtime_if_needed(source_keys=None)
                        except Exception as exc:
                            logger.warning("Failed strategy runtime refresh pass: %s", exc)

                    if run_maintenance:
                        try:
                            await expire_stale_signals(session)
                        except OperationalError as exc:
                            if _is_retryable_db_error(exc):
                                if hasattr(session, "rollback"):
                                    await session.rollback()
                                logger.warning("Skipped stale-signal expiry due to transient DB error")
                            else:
                                raise
                        except Exception as exc:
                            if hasattr(session, "rollback"):
                                await session.rollback()
                            logger.warning("Failed stale-signal expiry pass: %s", exc)

                        try:
                            orphan_cleanup = await _reconcile_orphan_open_orders(session)
                            if orphan_cleanup.get("rows_seen", 0) > 0:
                                logger.warning(
                                    "Reconciled orphan trader orders",
                                    extra={"orphan_cleanup": orphan_cleanup},
                                )
                        except Exception as exc:
                            if hasattr(session, "rollback"):
                                await session.rollback()
                            logger.warning("Failed orphan-order reconciliation pass: %s", exc)

                    if run_maintenance:
                        control = await read_orchestrator_control(session)
                    control_active = bool(control.get("is_enabled", False)) and not bool(control.get("is_paused", True))
                    if control_active and run_maintenance:
                        try:
                            stale_watchdog = await _run_terminal_stale_order_watchdog(session)
                            if stale_watchdog.get("alerted", 0) > 0:
                                logger.warning(
                                    "Detected stale active orders on terminal markets",
                                    extra={"stale_watchdog": stale_watchdog},
                                )
                        except Exception as exc:
                            if hasattr(session, "rollback"):
                                await session.rollback()
                            logger.warning("Failed stale-terminal-order watchdog pass: %s", exc)
                    if run_maintenance:
                        try:
                            async with release_conn(session):
                                await hot_state.flush_audit_buffer()
                                await hot_state.reseed_if_stale()
                        except Exception as exc:
                            logger.warning("Hot state maintenance failed: %s", exc)
                        last_maintenance_at = utcnow()
                    mode = _canonical_trader_mode(control.get("mode"), default="shadow")
                    cycle_interval = max(
                        1,
                        int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
                    )
                    sleep_seconds = cycle_interval
                    default_trader_cycle_timeout = float(max(12, cycle_interval * 3))
                    control_settings = dict(control.get("settings") or {})
                    global_runtime_settings = dict(control_settings.get("global_runtime") or {})
                    configured_trader_cycle_timeout = safe_float(
                        global_runtime_settings.get("trader_cycle_timeout_seconds"),
                        0.0,
                    )
                    effective_trader_cycle_timeout = (
                        configured_trader_cycle_timeout
                        if configured_trader_cycle_timeout > 0
                        else default_trader_cycle_timeout
                    )
                    trader_cycle_timeout_seconds = float(
                        max(
                            3.0,
                            min(
                                180.0,
                                effective_trader_cycle_timeout,
                            ),
                        )
                    )

                    if not control.get("is_enabled"):
                        traders = [
                            trader
                            for trader in await list_traders(
                                session,
                                mode=mode if mode in {"shadow", "live"} else None,
                            )
                            if _trader_matches_lane(trader, lane_key)
                        ]
                        has_active_traders = any(
                            bool(trader.get("is_enabled", True)) and not bool(trader.get("is_paused", False))
                            for trader in traders
                        )
                        if has_active_traders:
                            manage_only_cycle = True
                            manage_only_reasons.append("global_disabled")
                        else:
                            await _write_orchestrator_snapshot_best_effort(
                                session,
                                running=False,
                                enabled=False,
                                current_activity="Disabled",
                                interval_seconds=cycle_interval,
                                stats=await _build_orchestrator_snapshot_metrics(
                                    session=session,
                                    lane=lane_key,
                                    traders=traders,
                                ),
                            )
                            skip_cycle = True

                    if not skip_cycle and bool(control.get("is_paused")):
                        manage_only_cycle = True
                        manage_only_reasons.append("global_pause")

                    if not skip_cycle and bool(control.get("kill_switch")):
                        manage_only_cycle = True
                        manage_only_reasons.append("kill_switch")

                    # WS feed health gate — auto-pause when Polymarket WS is persistently down
                    # Must run even when manage_only_cycle (e.g. global_pause from WS auto-pause)
                    # so the recovery branch can clear _ws_auto_paused and unpause.
                    if not skip_cycle and mode == "live":
                        try:
                            ws_health = get_feed_manager().health_check()
                            poly_failures = ws_health.get("polymarket", {}).get("stats", {}).get("consecutive_failures", 0)
                            if poly_failures >= _WS_FAILURE_PAUSE_THRESHOLD:
                                if not _ws_auto_paused:
                                    _ws_auto_paused = True
                                    await update_orchestrator_control(session, is_paused=True)
                                    await create_trader_event(
                                        session,
                                        trader_id=None,
                                        event_type="ws_feed_failure",
                                        source="worker",
                                        severity="error",
                                        message=(
                                            f"Polymarket WS feed down — {poly_failures} consecutive failures. "
                                            "Auto-trader paused."
                                        ),
                                    )
                                    logger.error(
                                        "Polymarket WS persistently down, auto-pausing orchestrator",
                                        consecutive_failures=poly_failures,
                                    )
                                manage_only_cycle = True
                                manage_only_reasons.append("ws_feed_down")
                            elif _ws_auto_paused and poly_failures == 0:
                                _ws_auto_paused = False
                                await create_trader_event(
                                    session,
                                    trader_id=None,
                                    event_type="ws_feed_recovery",
                                    source="worker",
                                    severity="warning",
                                    message=(
                                        "Polymarket WS feed recovered. "
                                        "Orchestrator remains paused — resume manually when ready."
                                    ),
                                )
                                logger.warning(
                                    "Polymarket WS recovered; orchestrator remains paused (manual resume required)"
                                )
                        except Exception as exc:
                            logger.debug("WS health check skipped: %s", exc)

                    if not skip_cycle:
                        if not skip_cycle and mode == "live":
                            app_settings = await session.get(AppSettings, "default")
                            if not _is_live_credentials_configured(app_settings):
                                await _write_orchestrator_snapshot_best_effort(
                                    session,
                                    running=False,
                                    enabled=True,
                                    current_activity="Blocked: live credentials missing",
                                    interval_seconds=cycle_interval,
                                    last_error=None,
                                    stats=await _build_orchestrator_snapshot_metrics(
                                        session=session,
                                        lane=lane_key,
                                        traders=traders,
                                    ),
                                )
                                skip_cycle = True
                            else:
                                needs_live_execution_init = True

                    if not skip_cycle and not traders:
                        traders = [
                            trader
                            for trader in await list_traders(
                                session,
                                mode=mode if mode in {"shadow", "live"} else None,
                            )
                            if _trader_matches_lane(trader, lane_key)
                        ]

                if not skip_cycle and needs_live_execution_init and not await live_execution_service.ensure_initialized():
                    async with AsyncSessionLocal() as session:
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=False,
                            enabled=True,
                            current_activity="Blocked: live trading service failed to initialize",
                            interval_seconds=cycle_interval,
                            last_error=None,
                            stats=await _build_orchestrator_snapshot_metrics(
                                session=session,
                                lane=lane_key,
                                traders=traders,
                            ),
                        )
                    skip_cycle = True

                if skip_cycle:
                    (
                        runtime_trigger_event,
                        stream_claim_cursor,
                        stream_last_claim_run_at,
                    ) = await _wait_for_runtime_trigger(
                        None,
                        sleep_seconds,
                        consumer=stream_consumer_name,
                        group=stream_group,
                        claim_cursor=stream_claim_cursor,
                        last_claim_run_at=stream_last_claim_run_at,
                    )
                    continue

                cycle_trigger_label = (
                    str(cycle_trigger.get("event_type") or "").strip().lower()
                    if isinstance(cycle_trigger, dict)
                    else ""
                ) or ("requested_run" if manual_force_cycle else "scheduled")

                if write_snapshot:
                    async with AsyncSessionLocal() as session:
                        if manual_force_cycle:
                            await update_orchestrator_control(session, requested_run_at=None)
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=True,
                            enabled=bool(control.get("is_enabled", False)) and not bool(control.get("is_paused", True)),
                            current_activity=f"Starting cycle[{cycle_trigger_label}:{lane_key}]",
                            interval_seconds=cycle_interval,
                            last_error=None,
                            stats=await _build_orchestrator_snapshot_metrics(
                                session=session,
                                lane=lane_key,
                                traders=traders,
                            ),
                        )

                total_decisions = 0
                total_orders = 0
                total_processed_signals = 0
                now = datetime.now(timezone.utc)
                high_frequency_trader_ids: set[str] = set()
                high_frequency_crypto_active = False
                cycle_traders = list(traders)
                if runtime_triggered_cycle and isinstance(cycle_trigger, dict):
                    cycle_traders = [
                        trader for trader in traders if _runtime_trigger_matches_trader(trader, cycle_trigger)
                    ]

                for trader in cycle_traders:
                    trader_id = str(trader.get("id") or "")
                    if not trader_id:
                        continue
                    if not _is_crypto_source_trader(trader):
                        continue
                    high_frequency_trader_ids.add(trader_id)
                    if bool(trader.get("is_enabled", True)) and not bool(trader.get("is_paused", False)):
                        high_frequency_crypto_active = True

                if high_frequency_crypto_active and not manage_only_cycle:
                    sleep_seconds = min(
                        float(sleep_seconds),
                        _HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS,
                    )
                crypto_trader_specs: list[dict[str, Any]] = []
                non_crypto_trader_specs: list[dict[str, Any]] = []
                for trader in cycle_traders:
                    trader_id = str(trader.get("id") or "")
                    if not trader.get("is_enabled", True):
                        continue
                    trader_mode = _canonical_trader_mode(trader.get("mode"), default="shadow")
                    if trader_mode != mode:
                        continue

                    is_paused = bool(trader.get("is_paused", False))
                    runtime_trigger_for_trader = _runtime_trigger_matches_trader(trader, cycle_trigger)
                    due = manual_force_cycle or runtime_trigger_for_trader or _is_due(trader, now)
                    if trader_id in high_frequency_trader_ids and not manage_only_cycle and not is_paused:
                        due = manual_force_cycle or runtime_trigger_for_trader or _is_high_frequency_maintenance_due(
                            trader,
                            now,
                        )
                    process_signals_for_trader = True
                    if manage_only_cycle or is_paused or not due:
                        process_signals_for_trader = False
                    trigger_signal_ids_by_source = (
                        _trigger_signal_ids_for_trader(trader, cycle_trigger) if runtime_trigger_for_trader else None
                    )
                    trigger_signal_snapshots_by_source = (
                        _trigger_signal_snapshots_for_trader(trader, cycle_trigger) if runtime_trigger_for_trader else None
                    )
                    if (
                        process_signals_for_trader
                        and runtime_trigger_for_trader
                        and isinstance(cycle_trigger, dict)
                        and str(cycle_trigger.get("event_type") or "").strip().lower() == "runtime_signal_batch"
                        and trigger_signal_ids_by_source is None
                    ):
                        process_signals_for_trader = False

                    spec = {
                        "trader": trader,
                        "process_signals": process_signals_for_trader,
                        "trigger_signal_ids_by_source": trigger_signal_ids_by_source,
                        "trigger_signal_snapshots_by_source": trigger_signal_snapshots_by_source,
                    }
                    if _is_crypto_source_trader(trader):
                        crypto_trader_specs.append(spec)
                    else:
                        non_crypto_trader_specs.append(spec)

                crypto_tasks = [
                    asyncio.create_task(
                        _run_trader_once_with_timeout(
                            spec["trader"],
                            control,
                            process_signals=bool(spec["process_signals"]),
                            trigger_signal_ids_by_source=spec["trigger_signal_ids_by_source"],
                            trigger_signal_snapshots_by_source=spec["trigger_signal_snapshots_by_source"],
                            timeout_seconds=trader_cycle_timeout_seconds,
                        )
                    )
                    for spec in crypto_trader_specs
                ]
                if crypto_tasks:
                    await asyncio.sleep(0)

                for spec in non_crypto_trader_specs:
                    decisions, orders, processed_signals = await _run_trader_once_with_timeout(
                        spec["trader"],
                        control,
                        process_signals=bool(spec["process_signals"]),
                        trigger_signal_ids_by_source=spec["trigger_signal_ids_by_source"],
                        trigger_signal_snapshots_by_source=spec["trigger_signal_snapshots_by_source"],
                        timeout_seconds=trader_cycle_timeout_seconds,
                    )
                    total_decisions += decisions
                    total_orders += orders
                    total_processed_signals += processed_signals

                if crypto_tasks:
                    for decisions, orders, processed_signals in await asyncio.gather(*crypto_tasks):
                        total_decisions += decisions
                        total_orders += orders
                        total_processed_signals += processed_signals

                _prune_module_caches({str(t.get("id") or "") for t in traders})

                if write_snapshot:
                    async with AsyncSessionLocal() as session:
                        if manual_force_cycle and not manage_only_cycle:
                            await update_orchestrator_control(session, requested_run_at=None)
                        metrics = await _build_orchestrator_snapshot_metrics(
                            session=session,
                            lane=lane_key,
                            traders=traders,
                            decisions_delta=total_decisions,
                            orders_delta=total_orders,
                        )
                        metrics["decisions_last_cycle"] = total_decisions
                        metrics["orders_last_cycle"] = total_orders
                        metrics["signals_processed_last_cycle"] = total_processed_signals
                        metrics["cycle_lane"] = lane_key
                        metrics["cycle_trigger"] = cycle_trigger_label
                        metrics["maintenance_run"] = bool(run_maintenance)
                        metrics["maintenance_interval_seconds"] = float(maintenance_interval_seconds)
                        _maybe_log_execution_latency_sla_breach(
                            lane=lane_key,
                            metrics=metrics,
                        )
                        open_orders = int(metrics.get("open_orders", 0) or 0)
                        activity = (
                            f"Cycle[{cycle_trigger_label}:{lane_key}] signals={total_processed_signals} "
                            f"decisions={total_decisions} orders={total_orders}"
                        )
                        if manage_only_cycle:
                            reasons = ",".join(manage_only_reasons) if manage_only_reasons else "gated"
                            activity = (
                                f"Manage-only[{cycle_trigger_label}:{lane_key}] ({reasons}) signals={total_processed_signals} "
                                f"decisions={total_decisions} orders={total_orders}"
                            )
                        elif (
                            total_processed_signals == 0 and total_decisions == 0 and total_orders == 0 and open_orders > 0
                        ):
                            activity = (
                                f"Cycle[{trigger_label}:{lane_key}] monitoring open orders={open_orders} (no new pending signals)"
                            )
                        elif high_frequency_crypto_active and total_processed_signals == 0:
                            activity = (
                                f"Cycle[{trigger_label}:{lane_key}] high-frequency crypto monitor active (no new pending signals)"
                            )
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=True,
                            enabled=True,
                            current_activity=activity,
                            interval_seconds=cycle_interval,
                            last_run_at=utcnow(),
                            stats=metrics,
                        )

                (
                    runtime_trigger_event,
                    stream_claim_cursor,
                    stream_last_claim_run_at,
                ) = await _wait_for_runtime_trigger(
                    None,
                    sleep_seconds,
                    consumer=stream_consumer_name,
                    group=stream_group,
                    claim_cursor=stream_claim_cursor,
                    last_claim_run_at=stream_last_claim_run_at,
                )
            except Exception as exc:
                logger.exception("Trader orchestrator worker cycle failed: %s", exc)
                try:
                    async with AsyncSessionLocal() as session:
                        control = await read_orchestrator_control(session)
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=False,
                            enabled=bool(control.get("is_enabled", False)),
                            current_activity="Worker error",
                            interval_seconds=max(
                                1, int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS)
                            ),
                            last_error=str(exc),
                            stats=await _build_orchestrator_snapshot_metrics(
                                session=session,
                                lane=lane_key,
                                traders=traders,
                            ),
                        )
                except Exception as snapshot_exc:
                    logger.warning(
                        "Failed to persist orchestrator worker error state",
                        error=str(snapshot_exc),
                    )
                (
                    runtime_trigger_event,
                    stream_claim_cursor,
                    stream_last_claim_run_at,
                ) = await _wait_for_runtime_trigger(
                    None,
                    2,
                    consumer=stream_consumer_name,
                    group=stream_group,
                    claim_cursor=stream_claim_cursor,
                    last_claim_run_at=stream_last_claim_run_at,
                )
    finally:
        if lane_key == _LANE_GENERAL:
            await _release_orchestrator_cycle_lock_owner()
        else:
            await _release_orchestrator_cycle_lock_owner_for_lane(lane_key)


async def start_loop(*, lane: str = _LANE_GENERAL, notifier_enabled: bool = True, write_snapshot: bool = True) -> None:
    """Run the trader orchestrator worker loop (called from API process lifespan)."""
    notifier = None
    notifier_start_task: asyncio.Task | None = None
    if notifier_enabled:
        try:
            from services.notifier import notifier as notifier_service

            notifier = notifier_service
            notifier_start_task = asyncio.create_task(notifier.start(), name="autotrader-notifier-start")

            def _handle_notifier_start_done(task: asyncio.Task) -> None:
                try:
                    task.result()
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.warning("Autotrader notifier start failed (non-critical): %s", exc)
                else:
                    logger.info("Autotrader notifier started")

            notifier_start_task.add_done_callback(_handle_notifier_start_done)
            logger.info("Autotrader notifier start scheduled")
        except Exception as exc:
            logger.warning("Autotrader notifier start failed (non-critical): %s", exc)

    try:
        try:
            await run_worker_loop(lane=lane, write_snapshot=write_snapshot)
        except TypeError:
            await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader orchestrator worker shutting down")
    finally:
        if notifier_start_task is not None and not notifier_start_task.done():
            notifier_start_task.cancel()
            try:
                await notifier_start_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Notifier start task cleanup skipped: %s", exc)
        if notifier is not None:
            try:
                await notifier.shutdown()
            except Exception as exc:
                logger.debug("Notifier shutdown skipped: %s", exc)


async def main(*, lane: str = _LANE_GENERAL, notifier_enabled: bool = True, write_snapshot: bool = True) -> None:
    await init_database()
    try:
        await start_loop(lane=lane, notifier_enabled=notifier_enabled, write_snapshot=write_snapshot)
    except asyncio.CancelledError:
        logger.info("Trader orchestrator worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())

