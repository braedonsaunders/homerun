"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import asyncpg
import copy
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional

from sqlalchemy import and_, func, select, text, update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import (
    AppSettings,
    AsyncSessionLocal,
    apply_telemetry_async_commit,
    DiscoveredWallet,
    TradeSignal,
    TraderEvent,
    TraderOrder,
    TraderOrchestratorSnapshot,
    TraderPosition,
    TraderSignalCursor,
    TrackedWallet,
    Trader,
    TraderGroupMember,
    init_database,
    release_conn,
)
from services.trader_orchestrator.gate_pipeline import get_gate_metrics
from services.trader_orchestrator.hot_path import allow_polymarket_rest_call, hot_path_no_rest
from services.wallet_state_cache import get_wallet_state_cache
from services.trader_orchestrator.live_market_context import (
    RuntimeTradeSignalView,
    build_cached_live_signal_contexts,
)
from services.trader_orchestrator.session_engine import ExecutionSessionEngine
from services.trader_orchestrator.position_lifecycle import (
    load_market_info_for_orders,
    reconcile_shadow_positions,
)
from services.polymarket import polymarket_client
from services.simulation import simulation_service
from services.live_execution_service import live_execution_service
from services.live_pressure import current_backpressure_level, is_db_pressure_active, maybe_mark_db_pressure
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.strategies.base import StrategyDecision
from services.trader_orchestrator.decision_gates import (
    apply_platform_decision_gates,
    is_within_trading_schedule_utc,
)
from services.worker_state import _commit_with_retry, _is_retryable_db_error
from services.trader_orchestrator.sources.registry import (
    normalize_source_key,
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
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.strategy_sdk import StrategySDK
from services.strategies.news_edge import validate_news_edge_config
from services.strategy_versioning import normalize_strategy_version, resolve_strategy_version
from services.runtime_signal_queue import get_queue_depth, publish_signal_batch, wait_for_signal_batch
from services.runtime_status import runtime_status
from services.trader_orchestrator_state import (
    DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS,
    DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE,
    DEFAULT_LIVE_MARKET_CONTEXT,
    DEFAULT_LIVE_PROVIDER_HEALTH,
    DEFAULT_PENDING_LIVE_EXIT_GUARD,
    ORCHESTRATOR_SNAPSHOT_ID,
    ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
    cleanup_trader_open_orders,
    create_trader_decision as _create_trader_decision,
    create_trader_decision_checks as _create_trader_decision_checks,
    create_trader_event as _create_trader_event,
    create_trader_order as _create_trader_order,
    get_gross_exposure as _db_get_gross_exposure,
    # Re-exported into this module's namespace so
    # ``services.trader_cycle_context`` (and the orchestrator's test
    # fixtures) can monkeypatch it via the worker module path.  The
    # cycle context's projection refresh delegates to this name —
    # see ``trader_cycle_context._query_pending_live_exit``.
    get_pending_live_exit_summary_for_trader,  # noqa: F401
    list_unconsumed_trade_signals as _list_unconsumed_trade_signals_authoritative,
    list_traders,
    record_signal_consumption as _record_signal_consumption,
    reconcile_live_provider_orders,
    read_orchestrator_control,
    read_orchestrator_snapshot,
    set_trader_paused,
    sync_trader_position_inventory,
    upsert_trader_signal_cursor as _persist_trader_signal_cursor,
    update_orchestrator_control,
    update_trader_decision as _persist_trader_decision_update,
    write_orchestrator_snapshot,
)
from services.signal_bus import (
    expire_stale_signals,
)
import services.trader_hot_state as hot_state
from services.trader_cycle_context import trader_cycle_context
from services.ws_feeds import get_feed_manager
from utils.utcnow import utcnow
from utils.converters import coerce_bool as _coerce_bool, parse_iso_datetime, safe_float, safe_int, to_iso
from utils.logger import get_logger
from utils.secrets import decrypt_secret

logger = get_logger("trader_orchestrator_worker")
strategy_db_loader = strategy_loader

_TRADER_TIMEOUT_CANCEL_GRACE_SECONDS = 5.0
_MIN_LIVE_PROCESS_SIGNAL_CYCLE_TIMEOUT_SECONDS = 60.0  # do not cancel live order submission mid-flight
_STRATEGY_EVALUATION_TIMEOUT_SECONDS = 15.0

# Fast-tier traders are owned by ``fast_trader_runtime`` — this shared
# orchestrator loop skips them entirely so their per-tick budget is not
# consumed by the 20-30s cycles that "normal" / "slow" traders can incur.
_FAST_LATENCY_CLASS = "fast"
LIVE_ENTRY_BLOCKING_ORDER_STATUSES = (
    "pending",
    "placing",
    "submitted",
    "open",
    "working",
    "partial",
    "partially_filled",
    "hedging",
    "executed",
    "completed",
)
LIVE_AUTHORITY_PENDING_ENTRY_STATUSES = ("placing", "failed", "cancelled", "error")
LIVE_AUTHORITY_PENDING_STATES = {"pre_submit_persisted", "submit_timeout"}
LIVE_AUTHORITY_PENDING_MARKET_BLOCK_SECONDS = 1800.0


def _is_fast_tier_trader(trader: dict) -> bool:
    """Return True if the trader is owned by the fast-tier runtime."""
    if not isinstance(trader, dict):
        return False
    cls = str(trader.get("latency_class") or "normal").strip().lower()
    return cls == _FAST_LATENCY_CLASS


def _submission_intent_state(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    submission_intent = payload.get("submission_intent")
    if not isinstance(submission_intent, dict):
        return ""
    return str(submission_intent.get("state") or "").strip().lower()


def _live_authority_pending_entry_blocks_market(status: Any, payload: Any, updated_at: Any) -> bool:
    status_key = str(status or "").strip().lower()
    if status_key not in LIVE_AUTHORITY_PENDING_ENTRY_STATUSES:
        return False
    if _submission_intent_state(payload) not in LIVE_AUTHORITY_PENDING_STATES:
        return False
    if not isinstance(updated_at, datetime):
        return True
    updated = updated_at if updated_at.tzinfo is not None else updated_at.replace(tzinfo=timezone.utc)
    return (utcnow() - updated.astimezone(timezone.utc)).total_seconds() <= LIVE_AUTHORITY_PENDING_MARKET_BLOCK_SECONDS


_SIGNAL_DEADLOCK_MAX_RETRIES = 3
_SIGNAL_DEADLOCK_BASE_DELAY = 0.1
_SIGNAL_TRANSIENT_ERROR_MARKERS = (
    "deadlock detected",
    "serialization failure",
    "could not serialize access",
    "lock timeout",
    "statement timeout",
    "canceling statement due to statement timeout",
    "server disconnected",
    "connection is closed",
    "connection was closed",
    "connection has been invalidated",
    "connection invalidated",
    "another operation is in progress",
    "cannot switch to state",
    "got result for unknown protocol state",
)


def _is_transient_db_error(exc: Exception) -> bool:
    """Return True for DB/session errors that are safe to retry without consuming the signal."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, InterfaceError, OperationalError)):
        return True
    if _is_retryable_db_error(exc):
        return True
    msg = str(getattr(exc, "orig", exc)).lower()
    return any(marker in msg for marker in _SIGNAL_TRANSIENT_ERROR_MARKERS)

_STRATEGY_EVAL_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="strategy-eval")
_abandoned_trader_cycle_tasks: set[asyncio.Task] = set()
_inflight_trader_cycle_tasks: dict[str, set[asyncio.Task]] = {}
# Per-task metadata (start_time + process_signals flag).  Keyed by Task
# rather than trader_id so multiple concurrent cycles for the same
# trader can each carry their own info.
_inflight_trader_cycle_start: dict[asyncio.Task, float] = {}
_inflight_trader_cycle_process_signals: dict[asyncio.Task, bool] = {}
_pending_runtime_cycle_specs: dict[str, dict[str, Any]] = {}
_skip_log_last_at: dict[str, float] = {}  # trader_id -> last log monotonic time
_skip_log_suppressed: dict[str, int] = {}  # trader_id -> suppressed count
_SKIP_LOG_INTERVAL_SECONDS = 60.0  # only log once per minute per trader
_OVERLAP_WARNING_MIN_STUCK_SECONDS = 15.0
# Force-kill threshold for trader-cycle tasks that keep running after
# cancellation. Trader cycles are serialized per trader, so one stuck
# provider or DB await can block every queued runtime trigger for that
# trader until this guard clears it.
_STUCK_TASK_HARD_KILL_SECONDS = 90.0


def _discard_abandoned_trader_cycle(task: asyncio.Task) -> None:
    _abandoned_trader_cycle_tasks.discard(task)
    # Consume any pending exception so abandoned tasks don't surface as
    # "Task exception was never retrieved" in the global asyncio handler.
    # We deliberately abandoned these (soft-timeout / cancel-grace path),
    # so the TimeoutError/CancelledError is expected and already logged.
    if not task.cancelled():
        try:
            task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass


def _clear_inflight_trader_cycle_task(trader_id: str, task: asyncio.Task) -> None:
    if trader_id:
        bucket = _inflight_trader_cycle_tasks.get(trader_id)
        if bucket is not None:
            bucket.discard(task)
            if not bucket:
                _inflight_trader_cycle_tasks.pop(trader_id, None)
    _inflight_trader_cycle_process_signals.pop(task, None)
    _inflight_trader_cycle_start.pop(task, None)


def _running_inflight_tasks(trader_id: str) -> list[asyncio.Task]:
    """Return live in-flight cycle tasks for ``trader_id`` (skipping done)."""
    if not trader_id:
        return []
    bucket = _inflight_trader_cycle_tasks.get(trader_id)
    if not bucket:
        return []
    return [t for t in bucket if not t.done()]


def _count_running_signal_cycles(trader_id: str) -> int:
    """Count running cycles flagged as signal-processing for ``trader_id``."""
    return sum(
        1
        for task in _running_inflight_tasks(trader_id)
        if _inflight_trader_cycle_process_signals.get(task, False)
    )


def _oldest_running_task(trader_id: str) -> asyncio.Task | None:
    running = _running_inflight_tasks(trader_id)
    if not running:
        return None
    return min(running, key=lambda t: _inflight_trader_cycle_start.get(t, 0.0))


def _merge_trigger_signal_ids_by_source(
    current: dict[str, list[str]] | None,
    incoming: dict[str, list[str]] | None,
) -> dict[str, list[str]] | None:
    merged: dict[str, list[str]] = {}
    seen_ids: dict[str, set[str]] = {}
    for payload in (current, incoming):
        if not isinstance(payload, dict):
            continue
        for raw_source, raw_signal_ids in payload.items():
            source_key = str(raw_source or "").strip().lower()
            if not source_key:
                continue
            merged.setdefault(source_key, [])
            seen_ids.setdefault(source_key, set())
            if not isinstance(raw_signal_ids, list):
                continue
            for raw_signal_id in raw_signal_ids:
                signal_id = str(raw_signal_id or "").strip()
                if not signal_id or signal_id in seen_ids[source_key]:
                    continue
                seen_ids[source_key].add(signal_id)
                merged[source_key].append(signal_id)
    return merged or None


def _merge_trigger_signal_snapshots_by_source(
    current: dict[str, dict[str, dict[str, Any]]] | None,
    incoming: dict[str, dict[str, dict[str, Any]]] | None,
) -> dict[str, dict[str, dict[str, Any]]] | None:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for payload in (current, incoming):
        if not isinstance(payload, dict):
            continue
        for raw_source, raw_snapshots in payload.items():
            source_key = str(raw_source or "").strip().lower()
            if not source_key or not isinstance(raw_snapshots, dict):
                continue
            merged.setdefault(source_key, {})
            for raw_signal_id, snapshot in raw_snapshots.items():
                signal_id = str(raw_signal_id or "").strip()
                if not signal_id or not isinstance(snapshot, dict):
                    continue
                merged[source_key][signal_id] = copy.deepcopy(snapshot)
    return merged or None


def _restamp_trigger_signal_snapshots_by_source(
    snapshots_by_source: dict[str, dict[str, dict[str, Any]]] | None,
    *,
    emitted_at_iso: str,
) -> dict[str, dict[str, dict[str, Any]]] | None:
    if not isinstance(snapshots_by_source, dict):
        return None
    restamped: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_source, snapshots in snapshots_by_source.items():
        source_key = str(raw_source or "").strip().lower()
        if not source_key or not isinstance(snapshots, dict):
            continue
        for raw_signal_id, snapshot in snapshots.items():
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or not isinstance(snapshot, dict):
                continue
            snapshot_copy = copy.deepcopy(snapshot)
            payload_json = snapshot_copy.get("payload_json")
            if not isinstance(payload_json, dict):
                payload_json = {}
            else:
                payload_json = dict(payload_json)
            payload_json["signal_emitted_at"] = emitted_at_iso
            snapshot_copy["payload_json"] = payload_json
            restamped.setdefault(source_key, {})[signal_id] = snapshot_copy
    return restamped or None


def _queue_pending_runtime_cycle(
    *,
    trader: dict[str, Any],
    control: dict[str, Any],
    process_signals: bool,
    trigger_signal_ids_by_source: dict[str, list[str]] | None,
    trigger_signal_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] | None,
    timeout_seconds: float,
) -> None:
    trader_id = str(trader.get("id") or "").strip()
    if not trader_id:
        return
    existing = _pending_runtime_cycle_specs.get(trader_id)
    if existing is None:
        _pending_runtime_cycle_specs[trader_id] = {
            "trader": copy.deepcopy(trader),
            "control": copy.deepcopy(control),
            "process_signals": bool(process_signals),
            "trigger_signal_ids_by_source": _merge_trigger_signal_ids_by_source(
                None,
                trigger_signal_ids_by_source,
            ),
            "trigger_signal_snapshots_by_source": _merge_trigger_signal_snapshots_by_source(
                None,
                trigger_signal_snapshots_by_source,
            ),
            "timeout_seconds": float(max(1.0, timeout_seconds)),
        }
        return
    existing["trader"] = copy.deepcopy(trader)
    existing["control"] = copy.deepcopy(control)
    existing["process_signals"] = bool(existing.get("process_signals") or process_signals)
    existing["trigger_signal_ids_by_source"] = _merge_trigger_signal_ids_by_source(
        existing.get("trigger_signal_ids_by_source"),
        trigger_signal_ids_by_source,
    )
    existing["trigger_signal_snapshots_by_source"] = _merge_trigger_signal_snapshots_by_source(
        existing.get("trigger_signal_snapshots_by_source"),
        trigger_signal_snapshots_by_source,
    )
    existing["timeout_seconds"] = float(
        max(
            safe_float(existing.get("timeout_seconds"), 1.0),
            max(1.0, float(timeout_seconds)),
        )
    )


async def _launch_pending_runtime_cycle_if_any(trader_id: str) -> None:
    pending = _pending_runtime_cycle_specs.pop(str(trader_id or "").strip(), None)
    if not isinstance(pending, dict):
        return
    trigger_signal_snapshots_by_source = _restamp_trigger_signal_snapshots_by_source(
        pending.get("trigger_signal_snapshots_by_source"),
        emitted_at_iso=utcnow().isoformat(),
    )
    await _run_trader_once_with_timeout(
        pending["trader"],
        pending["control"],
        process_signals=bool(pending.get("process_signals")),
        trigger_signal_ids_by_source=pending.get("trigger_signal_ids_by_source"),
        trigger_signal_snapshots_by_source=trigger_signal_snapshots_by_source,
        timeout_seconds=float(max(1.0, safe_float(pending.get("timeout_seconds"), 1.0))),
    )


def _handle_trader_cycle_task_done(trader_id: str, task: asyncio.Task) -> None:
    _clear_inflight_trader_cycle_task(trader_id, task)
    # Skip-log counters are per-trader, not per-task: clear them only
    # when the trader has no remaining inflight cycles, otherwise we'd
    # silently drop the suppressed-warning bookkeeping for siblings
    # that are still running.
    if trader_id and not _running_inflight_tasks(trader_id):
        _skip_log_last_at.pop(trader_id, None)
        _skip_log_suppressed.pop(trader_id, None)
    if not trader_id or trader_id not in _pending_runtime_cycle_specs:
        return
    try:
        task.get_loop().create_task(
            _launch_pending_runtime_cycle_if_any(trader_id),
            name=f"trader-runtime-pending-{trader_id}",
        )
    except Exception:
        return


def _handle_runtime_trigger_dispatch_task_done(trader_id: str, task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        logger.warning(
            "Runtime trigger trader cycle raised uncaught exception for trader=%s",
            trader_id,
            exc_info=exc,
        )


_ORCH_LAST_ITER_MONO: dict[str, float] = {}


async def _pause_until_next_cycle(
    *,
    process_runtime_triggers: bool,
    sleep_seconds: float,
    stream_consumer_name: str,
    stream_group: str,
    stream_claim_cursor: str,
    stream_last_claim_run_at: datetime | None,
) -> tuple[dict[str, Any] | None, str, datetime | None]:
    if not process_runtime_triggers:
        await _worker_sleep(max(0.05, float(sleep_seconds)))
        return None, stream_claim_cursor, stream_last_claim_run_at
    # Wait for a signal batch up to the full cycle interval, BUT in short
    # increments so a manual Start (operator sets requested_run_at via the
    # trader panel) wakes the loop within ~_POLL_SECONDS instead of a whole
    # interval. The "30s to start" was this pause blocking the entire
    # run_interval_seconds inside wait_for_signal_batch, oblivious to the
    # control flip — only noticed on the next iteration. A real signal batch
    # still returns instantly (it ends the current increment early).
    remaining = max(0.05, float(sleep_seconds))
    cursor = stream_claim_cursor
    last_claim = stream_last_claim_run_at
    _POLL_SECONDS = 3.0
    while remaining > 0:
        wait = min(_POLL_SECONDS, remaining)
        trigger, cursor, last_claim = await _wait_for_runtime_trigger(
            None,
            wait,
            consumer=stream_consumer_name,
            group=stream_group,
            claim_cursor=cursor,
            last_claim_run_at=last_claim,
        )
        if trigger is not None:
            return trigger, cursor, last_claim
        remaining -= wait
        if remaining <= 0:
            break
        # Cheap read-only control check: if an operator just hit Start, break the
        # pause now so the loop's manual_force_cycle path runs immediately
        # instead of after a full interval.
        try:
            async with AsyncSessionLocal() as _ctl_session:
                _ctl = await read_orchestrator_control(_ctl_session, read_only=True)
            if _parse_iso(_ctl.get("requested_run_at")) is not None:
                return None, cursor, last_claim
        except Exception:
            pass
    return None, cursor, last_claim


# ── Hot-path shims ────────────────────────────────────────────────
# These keep the call sites stable while restoring synchronous DB-backed
# writes for authoritative execution and audit state.


async def record_signal_consumption(session, *, trader_id, signal_id, outcome, reason=None, decision_id=None, payload=None, signal_updated_at=None, commit=True):
    await _record_signal_consumption(
        session,
        trader_id=trader_id,
        signal_id=signal_id,
        outcome=outcome,
        reason=reason,
        decision_id=decision_id,
        payload=payload,
        signal_updated_at=signal_updated_at,
        commit=commit,
    )


async def upsert_trader_signal_cursor(session, *, trader_id, last_signal_created_at, last_signal_id, commit=True):
    last_runtime_sequence = None
    normalized_signal_id = str(last_signal_id or "").strip()
    if normalized_signal_id:
        signal_row = await session.get(TradeSignal, normalized_signal_id)
        if signal_row is not None:
            last_runtime_sequence = safe_int(getattr(signal_row, "runtime_sequence", None), None)
        if last_runtime_sequence is None:
            last_runtime_sequence = get_intent_runtime().get_runtime_sequence(normalized_signal_id)
    hot_state.update_signal_cursor(trader_id, "live", last_signal_created_at, last_signal_id, last_runtime_sequence)
    await _persist_trader_signal_cursor(
        session,
        trader_id=trader_id,
        last_signal_created_at=last_signal_created_at,
        last_signal_id=last_signal_id,
        commit=False,
    )
    row = await session.get(TraderSignalCursor, trader_id)
    if row is not None:
        row.last_runtime_sequence = last_runtime_sequence
    if commit:
        await _commit_with_retry(session)
    else:
        await session.flush()


async def create_trader_decision(session, *, trader_id, signal, strategy_key, strategy_version=None,
                                  decision, reason=None, score=None, checks_summary=None,
                                  risk_snapshot=None, payload=None, trace_id=None, commit=True,
                                  flush_on_add=True):
    return await _create_trader_decision(
        session,
        trader_id=trader_id,
        signal=signal,
        strategy_key=strategy_key,
        strategy_version=strategy_version,
        decision=decision,
        reason=reason,
        score=score,
        checks_summary=checks_summary,
        risk_snapshot=risk_snapshot,
        payload=payload,
        trace_id=trace_id,
        commit=commit,
        flush_on_add=flush_on_add,
    )


async def create_trader_decision_checks(session, *, decision_id, checks, commit=True):
    await _create_trader_decision_checks(
        session,
        decision_id=decision_id,
        checks=checks,
        commit=commit,
    )


async def update_trader_decision(
    session,
    *,
    decision_id,
    decision=None,
    reason=None,
    payload_patch=None,
    checks_summary_patch=None,
    commit=True,
):
    return await _persist_trader_decision_update(
        session,
        decision_id=str(decision_id or ""),
        decision=decision,
        reason=reason,
        payload_patch=payload_patch,
        checks_summary_patch=checks_summary_patch,
        commit=commit,
    )


async def set_trade_signal_status(session, *, signal_id, status, commit=True):
    """Buffered ``trade_signals`` status writer for the orchestrator
    hot path.

    Routes the DB UPDATE + ``TradeSignalEmission`` audit-row INSERT
    through ``hot_state.buffer_signal_status`` (audit-flush task,
    short transaction) instead of executing them inline on the
    orchestrator's session.  This eliminates the per-row lock
    contention against ``intent_runtime._project_upsert_batch``
    that the 5/2026/05 10h soak surfaced as the dominant bottleneck
    after the opportunity_state batched-UPSERT fix landed (commit
    e6c78e9): 6+ ``UPDATE trade_signals SET ...`` lock-contention
    incidents per soak hour, with intent_runtime projection_queue
    backing up to 782 entries.

    The in-memory ``intent_runtime`` status is updated synchronously
    so any read-after-write within the same trader cycle observes
    the new value.  The ``session`` and ``commit`` parameters are
    accepted for backward-compatibility with the call-site signature
    but are deliberately ignored — the audit-flush task owns its own
    transaction lifecycle and the orchestrator session must not
    block waiting for that commit.
    """
    sid = str(signal_id or "").strip()
    st = str(status or "").strip()
    if not sid:
        return
    normalized_status = st.lower()
    await hot_state.buffer_signal_status(
        signal_id=sid,
        status=st,
        emission_event_type="status_update",
        emission_reason=f"status:{normalized_status}",
    )
    await get_intent_runtime().update_signal_status(signal_id=sid, status=st)


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
    flush_on_add=True,
):
    return await _create_trader_event(
        session,
        event_type=str(event_type or ""),
        severity=str(severity or "info"),
        trader_id=str(trader_id or "") or None,
        source=str(source or "") or None,
        operator=str(operator or "") or None,
        message=message,
        trace_id=str(trace_id or "") or None,
        payload=dict(payload or {}),
        commit=commit,
        flush_on_add=flush_on_add,
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
    await _upsert_strategy_experiment_assignment(
        session,
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
        commit=commit,
    )


async def get_trader_signal_cursor(session, *, trader_id):
    return hot_state.get_signal_cursor(trader_id, "live")


async def get_trader_signal_sequence_cursor(session, *, trader_id):
    del session
    return hot_state.get_signal_sequence_cursor(trader_id, "live")


def _session_has_pending_writes(session: Any) -> bool:
    for attr in ("new", "dirty", "deleted"):
        try:
            if getattr(session, attr, None):
                return True
        except Exception:
            return True
    return False


def _session_in_transaction(session: Any) -> bool:
    in_transaction = getattr(session, "in_transaction", None)
    if callable(in_transaction):
        return bool(in_transaction())
    return bool(in_transaction)


async def _release_clean_signal_read_transaction(session: Any, rows: list[Any]) -> None:
    if _session_has_pending_writes(session):
        return

    expunge = getattr(session, "expunge", None)
    if callable(expunge):
        for row in rows:
            try:
                expunge(row)
            except Exception:
                continue

    if _session_in_transaction(session):
        await session.rollback()


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
    exclude_market_ids=None,
    limit=200,
):
    normalized_excluded_market_ids = {
        str(market_id or "").strip()
        for market_id in (exclude_market_ids or [])
        if str(market_id or "").strip()
    }
    try:
        runtime_rows = await get_intent_runtime().list_unconsumed_signals(
            trader_id=str(trader_id or ""),
            sources=list(sources or []),
            statuses=list(statuses or []),
            strategy_types_by_source=dict(strategy_types_by_source or {}),
            cursor_runtime_sequence=int(cursor_runtime_sequence) if cursor_runtime_sequence is not None else None,
            cursor_created_at=cursor_created_at,
            cursor_signal_id=cursor_signal_id,
            limit=int(limit),
        )
    except Exception as exc:
        runtime_rows = []
        logger.debug("Intent runtime signal read failed; falling back to DB", exc_info=exc)
    if runtime_rows:
        if normalized_excluded_market_ids:
            runtime_rows = [
                row
                for row in runtime_rows
                if str(getattr(row, "market_id", "") or "").strip() not in normalized_excluded_market_ids
            ]
        return runtime_rows[: max(1, min(int(limit), 5000))]
    # 2026-05-10: when the intent_runtime cache has nothing for this
    # trader, the authoritative DB fallback was guarded only by the
    # REACTIVE ``is_db_pressure_active()`` flag — which only trips
    # after a query has actually thrown.  During slow-soak the
    # fallback still runs, and the soak 2026-05-10 20:42-20:51
    # captured ``mnt_prefetch_signals`` at 968-2812ms per cycle as a
    # result.  Honouring proactive backpressure ≥0.5 (same threshold
    # as the trader_cycle_context fix) skips the DB SELECT once the
    # intent_runtime queue starts saturating; the orchestrator then
    # treats it as "no pending signals" and the cycle becomes idle,
    # which is correct under load.  The cache will catch up once
    # pressure clears.
    if is_db_pressure_active() or current_backpressure_level() >= 0.5:
        return []
    try:
        rows = await _list_unconsumed_trade_signals_authoritative(
            session,
            trader_id=str(trader_id or ""),
            sources=list(sources or []),
            statuses=list(statuses or []),
            strategy_types_by_source=dict(strategy_types_by_source or {}),
            cursor_runtime_sequence=int(cursor_runtime_sequence) if cursor_runtime_sequence is not None else None,
            cursor_created_at=cursor_created_at,
            cursor_signal_id=cursor_signal_id,
            exclude_market_ids=list(normalized_excluded_market_ids),
            limit=int(limit),
        )
        await _release_clean_signal_read_transaction(session, rows)
        return rows
    except (OperationalError, DBAPIError, asyncpg.PostgresError, asyncio.TimeoutError, TimeoutError) as exc:
        maybe_mark_db_pressure(exc, component="trader_orchestrator_signals", ttl_seconds=60.0)
        raise


def _json_safe_runtime_signal_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return to_iso(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_runtime_signal_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_runtime_signal_value(item) for item in value]
    return value


def _runtime_signal_trigger_snapshot(signal: Any, *, emitted_at_iso: str) -> dict[str, Any]:
    payload_json = dict(getattr(signal, "payload_json", None) or {})
    payload_json["signal_emitted_at"] = emitted_at_iso
    snapshot = {
        "id": str(getattr(signal, "id", "") or "").strip(),
        "source": str(getattr(signal, "source", "") or "").strip(),
        "source_item_id": str(getattr(signal, "source_item_id", "") or "").strip(),
        "signal_type": str(getattr(signal, "signal_type", "") or "").strip(),
        "strategy_type": str(getattr(signal, "strategy_type", "") or "").strip(),
        "market_id": str(getattr(signal, "market_id", "") or "").strip(),
        "market_question": str(getattr(signal, "market_question", "") or "").strip(),
        "direction": str(getattr(signal, "direction", "") or "").strip(),
        "entry_price": safe_float(getattr(signal, "entry_price", None), None),
        "effective_price": safe_float(getattr(signal, "effective_price", None), None),
        "edge_percent": safe_float(getattr(signal, "edge_percent", None), None),
        "confidence": safe_float(getattr(signal, "confidence", None), None),
        "liquidity": safe_float(getattr(signal, "liquidity", None), None),
        "expires_at": getattr(signal, "expires_at", None),
        "status": str(getattr(signal, "status", "") or "").strip().lower() or "pending",
        "payload_json": payload_json,
        "strategy_context_json": dict(getattr(signal, "strategy_context_json", None) or {}),
        "quality_passed": getattr(signal, "quality_passed", None),
        "dedupe_key": str(getattr(signal, "dedupe_key", "") or "").strip(),
        "runtime_sequence": safe_int(getattr(signal, "runtime_sequence", None), None),
        "required_token_ids": list(getattr(signal, "required_token_ids", None) or []),
        "deferred_until_ws": bool(getattr(signal, "deferred_until_ws", False)),
        "created_at": getattr(signal, "created_at", None),
        "updated_at": getattr(signal, "updated_at", None),
    }
    safe_snapshot = _json_safe_runtime_signal_value(snapshot)
    return safe_snapshot if isinstance(safe_snapshot, dict) else {}


# Round 9-A: in-process LRU of signal_ids we've already attempted to
# persist in *this* worker process.  The actual INSERT is
# ``on_conflict_do_nothing``, so on the second+ sighting of a signal_id
# it is a guaranteed no-op on Postgres — but the no-op still costs us
# the full asyncpg round-trip through the pool, and pool contention is
# the current stall driver (ps_sp_execute = 0.6-4.7s/cycle in the
# post-R8 soak, with ps_sp_prep ~0ms).  Skipping the repeat INSERT in
# Python shaves that entire bucket.
#
# Safety: worst case on orchestrator restart is we re-issue the same
# idempotent INSERT once per signal the first time we see it after
# boot — exactly the pre-R9 behavior.  No data loss, no divergence.
_RUNTIME_SIGNAL_PERSIST_LRU_CAP = 4096
_runtime_signal_persist_lru: "OrderedDict[str, None]" = OrderedDict()


async def _ensure_runtime_signal_persisted(
    session,
    signal: Any,
    *,
    accumulate: Optional[Callable[[str, float], None]] = None,
) -> None:
    """Idempotently upsert a runtime signal row.

    Round 8-B instrumentation: when ``accumulate`` is supplied, timing
    buckets for (a) JSON-safe payload prep and (b) the actual DB
    ``execute`` are recorded separately.  ps_signal_persist was the
    single worst bucket in the post-R7 soak (2.3-6.8s per cycle) but
    there was zero sub-breakdown, so we couldn't tell whether the time
    was in Python-side payload serialization or Postgres round-trip
    latency.  Now we can.

    Round 9-A: short-circuit when this ``signal_id`` has already been
    persisted by this process.  The INSERT is on_conflict_do_nothing so
    the repeat is a Postgres-side no-op — but under the current pool
    pressure the round-trip alone was owning multi-second stalls.
    """
    signal_id = str(getattr(signal, "id", "") or "").strip()
    if not signal_id:
        return
    _prep_started = time.monotonic() if accumulate is not None else 0.0
    # Round 9-A fast path: already persisted in this process.  Account
    # for both buckets with a zero duration so the per-signal
    # ps_signal_persist total still reconciles with its sub-buckets
    # (ps_unaccounted stays honest).
    if signal_id in _runtime_signal_persist_lru:
        # Refresh LRU order — this id is hot.
        _runtime_signal_persist_lru.move_to_end(signal_id)
        if accumulate is not None:
            accumulate("ps_sp_prep", _prep_started)
            accumulate("ps_sp_execute", time.monotonic())
        return
    row_values = dict(
        id=signal_id,
        source=str(getattr(signal, "source", "") or "").strip(),
        source_item_id=str(getattr(signal, "source_item_id", "") or "").strip() or None,
        signal_type=str(getattr(signal, "signal_type", "") or "").strip(),
        strategy_type=str(getattr(signal, "strategy_type", "") or "").strip() or None,
        market_id=str(getattr(signal, "market_id", "") or "").strip(),
        market_question=str(getattr(signal, "market_question", "") or "").strip() or None,
        direction=str(getattr(signal, "direction", "") or "").strip() or None,
        entry_price=safe_float(getattr(signal, "entry_price", None), None),
        edge_percent=safe_float(getattr(signal, "edge_percent", None), None),
        confidence=safe_float(getattr(signal, "confidence", None), None),
        liquidity=safe_float(getattr(signal, "liquidity", None), None),
        expires_at=getattr(signal, "expires_at", None),
        payload_json=_json_safe_runtime_signal_value(dict(getattr(signal, "payload_json", None) or {})),
        strategy_context_json=_json_safe_runtime_signal_value(
            dict(getattr(signal, "strategy_context_json", None) or {})
        ),
        quality_passed=getattr(signal, "quality_passed", None),
        quality_rejection_reasons=None,
        dedupe_key=str(getattr(signal, "dedupe_key", "") or "").strip(),
        runtime_sequence=safe_int(getattr(signal, "runtime_sequence", None), None),
        status=str(getattr(signal, "status", "") or "").strip().lower() or "pending",
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    if accumulate is not None:
        accumulate("ps_sp_prep", _prep_started)
        _exec_started = time.monotonic()
    await session.execute(
        pg_insert(TradeSignal)
        .values(**row_values)
        .on_conflict_do_nothing()
    )
    if accumulate is not None:
        accumulate("ps_sp_execute", _exec_started)
    # Round 9-A: remember this signal_id so the next sighting is a
    # pure in-process no-op.  Bounded eviction — FIFO via OrderedDict.
    _runtime_signal_persist_lru[signal_id] = None
    if len(_runtime_signal_persist_lru) > _RUNTIME_SIGNAL_PERSIST_LRU_CAP:
        _runtime_signal_persist_lru.popitem(last=False)


async def get_open_position_count_for_trader(session, trader_id, mode=None, position_cap_scope=None):
    return hot_state.get_open_position_count(trader_id, mode or "live", position_cap_scope=position_cap_scope or "market_direction")


async def get_open_order_count_for_trader(session, trader_id, mode=None):
    return hot_state.get_open_order_count(trader_id, mode or "live")


async def get_occupied_market_ids_for_trader(session, trader_id, mode=None):
    mode_key = str(mode or "live").strip().lower()
    occupied_market_ids: set[str] = set(hot_state.get_occupied_market_ids(trader_id, mode_key))

    position_query = (
        select(TraderPosition.market_id)
        .where(TraderPosition.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderPosition.status, "")) == "open")
    )
    if mode_key:
        position_query = position_query.where(func.lower(func.coalesce(TraderPosition.mode, "")) == mode_key)
    position_rows = (await session.execute(position_query)).all()
    for row in position_rows:
        market_id = str(row.market_id or "").strip()
        if market_id:
            occupied_market_ids.add(market_id)

    order_query = (
        select(TraderOrder.market_id)
        .where(TraderOrder.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderOrder.status, "")).in_(LIVE_ENTRY_BLOCKING_ORDER_STATUSES))
    )
    if mode_key:
        order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    order_rows = (await session.execute(order_query)).all()
    for row in order_rows:
        market_id = str(row.market_id or "").strip()
        if market_id:
            occupied_market_ids.add(market_id)

    authority_pending_cutoff = utcnow() - timedelta(seconds=LIVE_AUTHORITY_PENDING_MARKET_BLOCK_SECONDS)
    authority_pending_query = (
        select(TraderOrder.market_id, TraderOrder.status, TraderOrder.payload_json, TraderOrder.updated_at)
        .where(TraderOrder.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderOrder.status, "")).in_(LIVE_AUTHORITY_PENDING_ENTRY_STATUSES))
        .where(func.coalesce(TraderOrder.updated_at, TraderOrder.created_at) >= authority_pending_cutoff)
    )
    if mode_key:
        authority_pending_query = authority_pending_query.where(
            func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key
        )
    authority_pending_rows = (await session.execute(authority_pending_query)).all()
    for row in authority_pending_rows:
        market_id = str(row.market_id or "").strip()
        if market_id and _live_authority_pending_entry_blocks_market(row.status, row.payload_json, row.updated_at):
            occupied_market_ids.add(market_id)

    return occupied_market_ids


async def get_reentry_cooldown_market_ids_for_trader(session, trader_id, mode=None):
    return hot_state.get_reentry_cooldown_market_ids(trader_id, mode or "live")


async def get_daily_realized_pnl(session, *, trader_id=None, mode=None):
    return hot_state.get_daily_realized_pnl(trader_id, mode or "live")


async def get_unrealized_pnl(session, *, trader_id=None, mode=None):
    return await hot_state.get_unrealized_pnl(trader_id, mode or "live", ws_only=True)


async def get_consecutive_loss_count(session, *, trader_id, mode=None, limit=100, since=None):
    return hot_state.get_consecutive_loss_count(trader_id, mode or "live")


async def get_last_resolved_loss_at(session, *, trader_id, mode=None, since=None):
    return hot_state.get_last_resolved_loss_at(trader_id, mode or "live")


async def get_gross_exposure(session, mode=None):
    mode_key = mode or "live"
    hot_exposure = hot_state.get_gross_exposure(mode_key)
    if hot_exposure > 0.0:
        return hot_exposure
    return await _db_get_gross_exposure(session, mode=mode_key)


async def get_market_exposure(session, market_id, mode=None):
    return hot_state.get_market_exposure(market_id, mode or "live")


async def get_trader_source_exposure(session, *, trader_id, source, mode=None):
    return hot_state.get_trader_source_exposure(trader_id, source, mode or "live")


async def get_trader_copy_leader_exposure(session, *, trader_id, source_wallet, mode=None):
    return hot_state.get_copy_leader_exposure(trader_id, source_wallet, mode or "live")
create_trader_order = _create_trader_order
_RESUME_POLICIES = {"resume_full", "manage_only", "flatten_then_start"}
_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "completed", "open"}
_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE = 32
_ORCHESTRATOR_CYCLE_LOCK_KEY = 0x54524F5243485354  # "TRORCHST"
_ORCHESTRATOR_CRYPTO_CYCLE_LOCK_KEY = 0x54524F5243484352  # "TRORCHCR"
_orchestrator_lock_connection: asyncpg.Connection | None = None
_TRADER_IDLE_MAINTENANCE_INTERVAL_SECONDS = 60
_TRADER_LIVE_MAINTENANCE_INTERVAL_SECONDS = 30
_TRADER_SHADOW_MAINTENANCE_INTERVAL_SECONDS = 60
_HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS = 1.0
_STANDARD_MAX_SIGNALS_PER_CYCLE = 500
_STANDARD_DEFAULT_MAX_SIGNALS_PER_CYCLE = 200
_HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE = 5000
_HIGH_FREQUENCY_DEFAULT_MAX_SIGNALS_PER_CYCLE = 2000
_HIGH_FREQUENCY_DEFAULT_SCAN_BATCH_SIZE = 1000
_STANDARD_RUNTIME_TRIGGER_SIGNAL_LIMIT = 1
# Runtime-trigger releases are single-signal by design. The trader is
# serialized per trader to keep risk/order state coherent; releasing a
# multi-signal batch into a serialized executor turns later signals into
# queue backlog and breaches the 200 ms release-to-submit SLA. Overflow
# is re-published after the current signal cycle with a fresh release
# timestamp and snapshot, so market-data freshness gates still use the
# original observed-at age while execution latency measures the moment a
# signal is actually eligible to run.
_HIGH_FREQUENCY_RUNTIME_TRIGGER_SIGNAL_LIMIT = 1
_STANDARD_RUNTIME_TRIGGER_SCAN_BATCH_SIZE = 1
_HIGH_FREQUENCY_RUNTIME_TRIGGER_SCAN_BATCH_SIZE = 1
# Max concurrent runtime-trigger cycles per trader.  Allowing N parallel
# cycles trades a small race window on shared trader state (open-position
# count, occupied markets) for queue throughput.  Pre-2026-04-29 this
# was 4: in practice each cycle holds a DB connection for 10-20s of
# HTTP-bound enrichment, so 4 × ~6 traders = 24 concurrent connections
# into the live-trading codepath, each fighting the others for the
# regular pool's 80 slots and creating the lock-contention cascade
# observed as ``db_list_signals: 3-9s`` in the fast-trader stats and
# ``running=4 stuck=29s`` in the orchestrator skip telemetry.
#
# Runtime-trigger cycles are serialized per trader.  The worker already
# coalesces pending runtime specs, so parallel cycles only multiply DB
# writers for the same trader and widen the duplicate-submit window.
_RUNTIME_TRIGGER_MAX_CONCURRENT_PER_TRADER = 1
_trader_idle_maintenance_last_run: dict[str, datetime] = {}
_TRADER_CYCLE_HEARTBEAT_EVENT_INTERVAL_SECONDS = 1
_trader_cycle_heartbeat_last_emitted: dict[str, datetime] = {}
# Throttle the "Trader cycle skipped: wallet state stale" warning.  We
# want to know exactly when the gate first refused, but not spam the
# log every cycle while it stays refusing — orchestrator cycles fire
# on every signal arrival so the same trader can hit this gate dozens
# of times per second under runtime triggers.  Map keys to (trader_id,
# fresh_reason) so a reason flip (e.g. ``no_rest_seed_yet`` →
# ``rest_seed_stale:120s``) re-logs.
_trader_wallet_stale_log_state: dict[str, tuple[str, float]] = {}
_TRADER_WALLET_STALE_LOG_INTERVAL_SECONDS = 30.0
_LANE_GENERAL = "general"
_LANE_CRYPTO = "crypto"
_RUNTIME_TRIGGER_DEFAULT_CYCLE_TIMEOUT_SECONDS = 10.0
_ORCHESTRATOR_SNAPSHOT_MIN_PERSIST_INTERVAL_SECONDS = 10.0
_orchestrator_snapshot_last_persist_mono: dict[str, float] = {}
_TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS = 30
_TERMINAL_STALE_ORDER_MIN_AGE_MINUTES = 3
_TERMINAL_STALE_ORDER_ALERT_COOLDOWN_SECONDS = 300
# Lowered from 20.0s on 2026-05-07 — maintenance runs INLINE inside the
# trader cycle (which itself is bounded at 20s). With two maintenance
# steps each at 20s, a single contended one could blow the entire cycle
# budget — observed in soak as stage_timings_ms={'maintenance': 11922.0}
# eating the whole 12.89s slow-cycle window. Tighter per-step caps mean
# slow steps re-run on the next maintenance interval rather than
# starving signal processing.
_TRADER_MAINTENANCE_STEP_TIMEOUT_SECONDS = 8.0
_TRADER_PENDING_EXIT_SUMMARY_TIMEOUT_SECONDS = 5.0
_WS_FAILURE_PAUSE_THRESHOLD = 10
_ws_auto_paused = False
_LATENCY_SLA_STAGE_KEY = "ws_release_to_submit_start_ms"
_LATENCY_SLA_BREACH_LOG_COOLDOWN_SECONDS = 60.0
# Sub-stages that together make up ws_release_to_submit_start_ms. Surfaced
# in the SLA breach log so the warning tells us WHERE inside the span the
# time burned (queue wake, live-context build, strategy/risk eval, or the
# decision->submit DB commit block) rather than just the overall p95.
_LATENCY_SLA_SUBSTAGE_KEYS: tuple[str, ...] = (
    "emit_to_queue_wake_ms",
    "wake_to_context_ready_ms",
    "context_ready_to_decision_ms",
    "ws_release_to_decision_ms",
    "decision_to_submit_start_ms",
    "submit_round_trip_ms",
)
# Per-sample warning threshold: any individual latency sample whose
# ws_release_to_submit_start_ms exceeds this fires a structured warning
# with the sub-stage breakdown for that specific signal. Throttled per
# trader so a single slow trader can't flood the soak log.
_LATENCY_SLA_PER_SAMPLE_WARN_MS = 5000
_LATENCY_SLA_PER_SAMPLE_WARN_COOLDOWN_SECONDS = 30.0
_latency_per_sample_warn_logged_at: dict[str, datetime] = {}
_ORCHESTRATOR_SNAPSHOT_PRESSURE_SKIP_LEVEL = 0.55
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
_execution_session_maintenance_last_run: dict[str, datetime] = {}
_latency_sla_breach_logged_at: dict[str, datetime] = {}
_latency_sla_breach_last_signature: dict[str, tuple[int | None, int | None, int | None]] = {}
_TRADERS_SCOPE_CONTEXT_CACHE_TTL_SECONDS = 5.0
_traders_scope_context_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
_lane_snapshot_metrics: dict[str, dict[str, Any]] = {}
_lane_snapshot_seeded = False

# ── Per-market execution no-fill cooldown ─────────────────────────
# After a failed/cancelled live entry, cool down this specific market for a
# trader so we do not hammer the same bad venue state or illiquid book.
_NOFILL_COOLDOWN_SECONDS = 300.0  # 5 minutes between retries
_NOFILL_MAX_CONSECUTIVE = 3       # after 3 consecutive no-fills, extend cooldown
_NOFILL_EXTENDED_COOLDOWN_SECONDS = 900.0  # 15 minutes after 3 fails
_NOFILL_CANCELLED_COOLDOWN_SECONDS = 1800.0  # 30 minutes after venue-cancelled/expired entries
_nofill_cooldowns: dict[tuple[str, str], float] = {}  # (trader_id, market_id) -> monotonic expiry
_nofill_counts: dict[tuple[str, str], int] = {}        # consecutive no-fill count


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
    session: Any,
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
    # Phase 4: compact per-gate stats for the operator UI.  Top 10 by
    # run count — these are the gates the orchestrator is exercising
    # most, which is what the dashboard wants to surface "what's
    # blocking trades right now".  Stats are in-memory and reset on
    # restart; the snapshot is a point-in-time view, not a log.
    try:
        gate_snaps = get_gate_metrics().snapshot()
        gate_snaps.sort(key=lambda s: s.runs, reverse=True)
        merged["gate_stats_top10"] = [
            {
                "name": s.name,
                "cost_class": s.cost_class,
                "runs": s.runs,
                "rejects": s.rejects,
                "reject_rate": s.reject_rate,
                "p50_ms": s.p50_ms,
                "p95_ms": s.p95_ms,
                "p99_ms": s.p99_ms,
                "budget_ms": s.budget_ms,
                "budget_violations": s.budget_violations,
            }
            for s in gate_snaps[:10]
        ]
    except Exception:
        # Metrics is a side channel — never let a bug here break
        # snapshot persistence.  Drop the field and continue.
        merged["gate_stats_top10"] = []
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
_LEGACY_STRATEGY_ALIASES: dict[str, str] = {
    "news_reaction": "news_edge",
}


def _canonical_trader_mode(value: Any, *, default: str = "shadow") -> str:
    key = str(value or "").strip().lower()
    if not key:
        key = default
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
    exclude_market_ids: list[str] | None = None,
    limit: int,
) -> list[Any]:
    if not signal_ids_by_source or not sources:
        return []

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
    snapshot_ids: set[str] = set()
    for source_key in ordered_sources:
        for raw_signal_id in signal_ids_by_source.get(source_key) or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id:
                continue
            snapshot = (snapshots.get(source_key) or {}).get(signal_id)
            if snapshot is None and source_key != "__all__":
                snapshot = (snapshots.get("__all__") or {}).get(signal_id)
            if isinstance(snapshot, dict):
                snapshot_ids.add(signal_id)
    # Snapshot fast-path: a snapshot-backed runtime trigger already carries the
    # authoritative TradeSignal data in its payload, so only the ids WITHOUT an
    # in-payload snapshot need an authoritative DB read.  When every triggered
    # id has a snapshot this skips the trade_signals query entirely -- the redundant
    # authoritative read per triggered cycle was a measurable contributor to the
    # trader-cycle slowness / mnt_prefetch_signals cost in the 20h soak.  When some
    # ids lack a snapshot the read still runs (and still drives db-pressure detection)
    # for exactly those ids.
    missing_snapshot_ids = [signal_id for signal_id in ordered_ids if signal_id not in snapshot_ids]
    try:
        eligible_rows = []
        if missing_snapshot_ids:
            eligible_rows = await _list_unconsumed_trade_signals_authoritative(
                session,
                trader_id=trader_id,
                sources=list(sources or []),
                statuses=list(statuses or []),
                strategy_types_by_source=dict(strategy_types_by_source or {}),
                cursor_runtime_sequence=cursor_runtime_sequence,
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                signal_ids=missing_snapshot_ids,
                exclude_market_ids=list(exclude_market_ids or []),
                limit=max(len(missing_snapshot_ids), int(limit)),
            )
            await _release_clean_signal_read_transaction(session, eligible_rows)
    except (OperationalError, DBAPIError, asyncpg.PostgresError, asyncio.TimeoutError, TimeoutError) as exc:
        maybe_mark_db_pressure(exc, component="trader_orchestrator_trigger_signals", ttl_seconds=60.0)
        eligible_rows = []
    eligible_rows_by_id = {
        str(getattr(row, "id", "") or "").strip(): row
        for row in eligible_rows
        if str(getattr(row, "id", "") or "").strip()
    }

    rows: list[Any] = []
    source_order = [source for source in sources if source in signal_ids_by_source]
    if "__all__" in signal_ids_by_source:
        source_order = ["__all__", *source_order]
    seen_row_ids: set[str] = set()
    normalized_statuses = {
        str(status or "").strip().lower()
        for status in (statuses or [])
        if str(status or "").strip()
    }
    excluded_market_ids = {
        str(market_id or "").strip()
        for market_id in (exclude_market_ids or [])
        if str(market_id or "").strip()
    }
    for source_key in source_order:
        allowed_strategy_types = set(strategy_types_by_source.get(source_key) or [])
        for raw_signal_id in signal_ids_by_source.get(source_key) or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen_row_ids:
                continue
            authoritative_row = eligible_rows_by_id.get(signal_id)
            snapshot = (snapshots.get(source_key) or {}).get(signal_id)
            if snapshot is None and source_key != "__all__":
                snapshot = (snapshots.get("__all__") or {}).get(signal_id)
            if authoritative_row is None and not isinstance(snapshot, dict):
                continue
            if isinstance(snapshot, dict):
                row = _coerce_runtime_signal_snapshot(snapshot)
                if authoritative_row is not None:
                    row.status = str(getattr(authoritative_row, "status", "") or getattr(row, "status", "")).strip().lower()
                    row.runtime_sequence = getattr(authoritative_row, "runtime_sequence", getattr(row, "runtime_sequence", None))
                    if not str(getattr(row, "market_question", "") or "").strip():
                        row.market_question = str(getattr(authoritative_row, "market_question", "") or "").strip()
                    if getattr(row, "expires_at", None) is None:
                        row.expires_at = getattr(authoritative_row, "expires_at", None)
                    if getattr(row, "created_at", None) is None:
                        row.created_at = getattr(authoritative_row, "created_at", None)
                    if getattr(row, "updated_at", None) is None:
                        row.updated_at = getattr(authoritative_row, "updated_at", None)
                else:
                    row_strategy_type = str(getattr(row, "strategy_type", "") or "").strip().lower()
                    if allowed_strategy_types and row_strategy_type not in allowed_strategy_types:
                        continue
                    if normalized_statuses and str(getattr(row, "status", "") or "").strip().lower() not in normalized_statuses:
                        continue
                row_runtime_sequence = _signal_runtime_sequence(row)
                if cursor_runtime_sequence is not None and row_runtime_sequence is not None and row_runtime_sequence <= int(cursor_runtime_sequence):
                    continue
                row_timestamp = _signal_cursor_timestamp(row)
                if (
                    cursor_created_at is not None
                    and row_timestamp is not None
                    and row_timestamp < cursor_created_at
                ):
                    continue
                if (
                    cursor_created_at is not None
                    and row_timestamp is not None
                    and row_timestamp == cursor_created_at
                    and cursor_signal_id
                    and signal_id <= str(cursor_signal_id)
                ):
                    continue
            else:
                row = authoritative_row
            row_market_id = str(getattr(row, "market_id", "") or "").strip()
            if row_market_id and row_market_id in excluded_market_ids:
                continue
            # Honor signal expiry even on the snapshot-only path.  When
            # list_unconsumed filters out an expired signal, authoritative_row
            # is None but the snapshot still contains its data; without this
            # check the orchestrator re-evaluates the same expired signal on
            # every scanner re-publish, wasting cycle budget indefinitely.
            row_expires_at = getattr(row, "expires_at", None)
            if isinstance(row_expires_at, datetime):
                _exp = row_expires_at
                if _exp.tzinfo is not None:
                    _exp = _exp.astimezone(timezone.utc).replace(tzinfo=None)
                if _exp <= utcnow().replace(tzinfo=None):
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


def _session_has_pending_writes(session: Any) -> bool:
    try:
        return bool(getattr(session, "new", None) or getattr(session, "dirty", None) or getattr(session, "deleted", None))
    except Exception:
        return True


# Single-flight background persist for the orchestrator TELEMETRY snapshot.
# The 16h soak's await-stack dumps showed the main loop parked 90s+ inside
# write_orchestrator_snapshot -> COMMIT ("Starting cycle[scheduled:general]"
# stalls, and the operator-visible 30s "Start takes forever"): under commit
# contention (reconcile / market-runtime / catalog flushes all committing
# chunky JSON), the hot loop was blocked on a STATUS-ROW commit. The in-memory
# runtime_status is already updated before the persist, and the DB row only
# feeds the UI + SLO monitor — so the trading loop must never await it.
# Latest-wins per lane: while a persist is in flight, newer snapshots replace
# the pending slot instead of queueing, so telemetry lags (acceptable) rather
# than piling up tasks.
_orch_snapshot_bg_pending: dict[str, dict[str, Any]] = {}
_orch_snapshot_bg_tasks: dict[str, asyncio.Task] = {}
# Last PERSISTED (running, enabled, has_error) per lane. The UI's Start button
# reads the DB snapshot via GET /status (the API process cannot see this
# worker's in-memory runtime_status), so a disabled->running transition must
# reach the DB row immediately — the 10s min-persist throttle below only
# exists to damp steady-state heartbeat spam. Without this, the first
# running=True snapshot after an operator Start was silently dropped whenever
# any snapshot (e.g. "Disabled") had persisted in the prior 10s, leaving the
# button on "starting..." while the loop was already trading.
_orch_snapshot_last_persisted_state: dict[str, tuple[bool, bool, bool]] = {}


def _orch_snapshot_state_sig(snapshot_kwargs: dict[str, Any]) -> tuple[bool, bool, bool]:
    return (
        bool(snapshot_kwargs.get("running")),
        bool(snapshot_kwargs.get("enabled")),
        bool(snapshot_kwargs.get("last_error")),
    )


# Single-flight background hot-state maintenance (audit-buffer flush + stale
# reseed). See the call site in run_worker_loop for why this must never be
# awaited inline on the cycle loop.
_hot_state_maintenance_task: asyncio.Task | None = None


def _kick_hot_state_maintenance() -> None:
    global _hot_state_maintenance_task
    existing = _hot_state_maintenance_task
    if existing is not None and not existing.done():
        return

    async def _run() -> None:
        try:
            await hot_state.flush_audit_buffer()
            await hot_state.reseed_if_stale()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Hot state maintenance failed: %s", exc)

    _hot_state_maintenance_task = asyncio.create_task(
        _run(), name="hot-state-maintenance"
    )


async def _drain_orchestrator_snapshot_persists(lane_key: str) -> None:
    while True:
        snapshot_kwargs = _orch_snapshot_bg_pending.pop(lane_key, None)
        if snapshot_kwargs is None:
            return
        try:
            async with AsyncSessionLocal() as fresh_session:
                # Telemetry durability class: async commit so this snapshot
                # never waits in (or lengthens) the WAL group-commit queue
                # (see models.database.apply_telemetry_async_commit). The
                # transactional caller-session path keeps full durability.
                await apply_telemetry_async_commit(fresh_session)
                await write_orchestrator_snapshot(fresh_session, **snapshot_kwargs)
            _orchestrator_snapshot_last_persist_mono[lane_key] = time.monotonic()
            _orch_snapshot_last_persisted_state[lane_key] = _orch_snapshot_state_sig(snapshot_kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Telemetry only — never let a snapshot failure propagate anywhere
            # near the trading loop. The next snapshot attempt retries.
            logger.warning(
                "Orchestrator snapshot background persist failed (telemetry only)",
                lane=lane_key,
                exc_info=exc,
            )


async def _write_orchestrator_snapshot_best_effort(
    session: Any,
    *,
    lane: str = _LANE_GENERAL,
    persist: bool = True,
    **snapshot_kwargs: Any,
) -> None:
    runtime_status.update_orchestrator(
        lane=lane,
        running=snapshot_kwargs.get("running"),
        enabled=snapshot_kwargs.get("enabled"),
        current_activity=snapshot_kwargs.get("current_activity"),
        interval_seconds=snapshot_kwargs.get("interval_seconds"),
        last_run_at=to_iso(snapshot_kwargs.get("last_run_at")),
        last_error=snapshot_kwargs.get("last_error"),
        stats=dict(snapshot_kwargs.get("stats") or {}),
    )
    if not persist:
        return
    lane_key = str(lane or _LANE_GENERAL).strip().lower() or _LANE_GENERAL
    has_pending_writes = _session_has_pending_writes(session)
    use_fresh_session = not has_pending_writes and isinstance(session, AsyncSession)
    # State TRANSITIONS (disabled<->running, enabled flip, error appearing/
    # clearing) bypass both the min-interval throttle and the pressure skip:
    # the UI's Start/Stop button state hangs off this DB row (see the note at
    # _orch_snapshot_last_persisted_state). Steady-state snapshots (same sig)
    # stay throttled exactly as before — and so does the FIRST snapshot of a
    # fresh process (no prior sig): that's boot housekeeping, not an operator
    # action, and it must still defer under backpressure.
    _prev_state_sig = _orch_snapshot_last_persisted_state.get(lane_key)
    state_changed = (
        _prev_state_sig is not None
        and _prev_state_sig != _orch_snapshot_state_sig(snapshot_kwargs)
    )
    completed_cycle_snapshot = snapshot_kwargs.get("last_run_at") is not None
    if not has_pending_writes and not state_changed and not completed_cycle_snapshot:
        now_mono = time.monotonic()
        last_persisted_at = _orchestrator_snapshot_last_persist_mono.get(lane_key)
        if (
            last_persisted_at is not None
            and now_mono - last_persisted_at < _ORCHESTRATOR_SNAPSHOT_MIN_PERSIST_INTERVAL_SECONDS
        ):
            return
        heartbeat_required = lane_key == _LANE_GENERAL and (
            bool(snapshot_kwargs.get("enabled")) or bool(snapshot_kwargs.get("running"))
        )
        pressure_level = current_backpressure_level()
        if (
            not heartbeat_required
            and (is_db_pressure_active() or pressure_level >= _ORCHESTRATOR_SNAPSHOT_PRESSURE_SKIP_LEVEL)
        ):
            return
    if use_fresh_session:
        # Telemetry-only persist on a fresh session: hand it to the background
        # drainer and return WITHOUT awaiting the commit (see note above).
        _orch_snapshot_bg_pending[lane_key] = dict(snapshot_kwargs)
        existing = _orch_snapshot_bg_tasks.get(lane_key)
        if existing is None or existing.done():
            _orch_snapshot_bg_tasks[lane_key] = asyncio.create_task(
                _drain_orchestrator_snapshot_persists(lane_key),
                name=f"orch-snapshot-persist-{lane_key}",
            )
        return
    # Caller session has pending writes: the snapshot rides the caller's
    # transaction (committing here also commits those writes), so this path
    # MUST stay synchronous and keep its original error semantics.
    try:
        await write_orchestrator_snapshot(session, **snapshot_kwargs)
        _orchestrator_snapshot_last_persist_mono[lane_key] = time.monotonic()
        _orch_snapshot_last_persisted_state[lane_key] = _orch_snapshot_state_sig(snapshot_kwargs)
    except (OperationalError, DBAPIError, asyncpg.PostgresError, asyncio.TimeoutError, TimeoutError) as exc:
        if hasattr(session, "rollback"):
            await session.rollback()
        if isinstance(exc, OperationalError) and not _is_retryable_db_error(exc):
            raise
        logger.warning("Skipped orchestrator snapshot write due to transient DB error", exc_info=exc)


async def submit_order(
    *,
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
    explicit_strategy_params: dict[str, Any] | None = None,
    session_engine: Any | None = None,
):
    if session_engine is not None:
        return await session_engine.execute_signal(
            trader_id=trader_id,
            signal=signal,
            decision_id=decision_id,
            strategy_key=strategy_key,
            strategy_version=strategy_version,
            strategy_params=strategy_params,
            explicit_strategy_params=explicit_strategy_params,
            risk_limits=risk_limits,
            mode=mode,
            size_usd=size_usd,
            reason=reason,
        )
    async with AsyncSessionLocal() as submit_session:
        _engine = ExecutionSessionEngine(submit_session)
        return await _engine.execute_signal(
            trader_id=trader_id,
            signal=signal,
            decision_id=decision_id,
            strategy_key=strategy_key,
            strategy_version=strategy_version,
            strategy_params=strategy_params,
            explicit_strategy_params=explicit_strategy_params,
            risk_limits=risk_limits,
            mode=mode,
            size_usd=size_usd,
            reason=reason,
        )


def _execution_outcome_decision_check(
    *,
    status: str,
    reason: str | None,
    execution_session_id: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any] | None:
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"skipped", "failed", "cancelled", "expired"}:
        return None

    detail = str(reason or error_message or "").strip()
    normalized_detail = detail.lower()
    check_key = "execution_submission"
    check_label = "Execution submission"
    if "buy pre-submit gate failed" in normalized_detail:
        check_key = "buy_pre_submit_gate"
        check_label = "Buy collateral balance/allowance"

    if not detail:
        detail = (
            "Execution skipped before order submission."
            if normalized_status == "skipped"
            else "Execution failed during order submission."
        )

    return {
        "check_key": check_key,
        "check_label": check_label,
        "passed": False,
        "score": None,
        "detail": detail,
        "payload": {
            "execution_status": normalized_status,
            "execution_session_id": str(execution_session_id or "") or None,
            "error": str(error_message or "") or None,
        },
    }


def _enforce_strict_ws_strategy_params(
    strategy_params: dict[str, Any],
    *,
    strict_age_budget_ms: int,
    strict_ws_price_sources: list[str],
) -> dict[str, Any]:
    strategy_params["require_strict_ws_pricing"] = True
    strategy_params.setdefault("strict_ws_price_sources", list(strict_ws_price_sources))
    existing_age_budget_ms = safe_int(strategy_params.get("max_market_data_age_ms"), None)
    if existing_age_budget_ms is None:
        strategy_params["max_market_data_age_ms"] = int(strict_age_budget_ms)
    else:
        strategy_params["max_market_data_age_ms"] = min(
            int(strict_age_budget_ms),
            int(existing_age_budget_ms),
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


def _remaining_cycle_budget_seconds(
    *,
    cycle_started_mono: float,
    cycle_timeout_seconds: float,
    reserve_seconds: float = 0.0,
) -> float | None:
    if cycle_timeout_seconds <= 0:
        return None
    remaining = float(cycle_timeout_seconds) - (time.monotonic() - cycle_started_mono) - float(reserve_seconds)
    return max(0.0, remaining)


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
    released_at = execution_armed_at or emitted_at

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
    if released_at is not None:
        out["ws_release_to_now_ms"] = max(0, int((now - released_at).total_seconds() * 1000))
    if ingested_at is not None:
        out["ingest_to_now_ms"] = max(0, int((now - ingested_at).total_seconds() * 1000))
    return out


def _payload_live_market_context(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    live_market = payload.get("live_market")
    return dict(live_market) if isinstance(live_market, dict) else {}


def _build_execution_latency_sample(
    signal: Any,
    *,
    wake_started_at: datetime,
    context_ready_at: datetime,
    decision_ready_at: datetime,
    submit_started_at: datetime | None = None,
    submit_completed_at: datetime | None = None,
) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    armed_at = _parse_iso(str(payload.get("execution_armed_at") or ""))
    emitted_at = _parse_iso(str(payload.get("signal_emitted_at") or payload.get("ingested_at") or ""))
    released_at = armed_at or emitted_at
    if emitted_at is not None and (armed_at is None or emitted_at > armed_at):
        released_at = emitted_at

    def _delta_ms(start: datetime | None, end: datetime | None) -> int | None:
        if start is None or end is None:
            return None
        return max(0, int((end - start).total_seconds() * 1000))

    sample = {
        "armed_to_ws_release_ms": _delta_ms(armed_at, released_at),
        "emit_to_queue_wake_ms": _delta_ms(emitted_at, wake_started_at),
        "ws_release_to_decision_ms": _delta_ms(released_at, decision_ready_at),
        "ws_release_to_submit_start_ms": _delta_ms(released_at, submit_started_at),
        "wake_to_context_ready_ms": _delta_ms(wake_started_at, context_ready_at),
        "context_ready_to_decision_ms": _delta_ms(context_ready_at, decision_ready_at),
        "decision_to_submit_start_ms": _delta_ms(decision_ready_at, submit_started_at),
        "submit_round_trip_ms": _delta_ms(submit_started_at, submit_completed_at),
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
    return strategy_loader.get_instance(strategy_key)


def _format_filter_diagnostics_message(cfd: dict[str, Any]) -> str:
    """Format a heartbeat message from crypto filter diagnostics.

    Strategies provide their own ``message`` key.  Falls back to a generic
    summary using ``markets_scanned`` / ``signals_emitted`` if absent.
    """
    msg = cfd.get("message")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()
    scanned = cfd.get("markets_scanned", 0)
    emitted = cfd.get("signals_emitted", 0)
    return f"Scanned {scanned} markets, {emitted} signals"


def _get_strategy_filter_diagnostics(trader: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Get filter diagnostics from the trader's configured strategy.

    Looks up the strategy instance for the trader and calls
    ``get_filter_diagnostics()`` on it.  Returns None when the strategy
    has no diagnostics — the caller falls through to generic stats.
    """
    if trader is None:
        return None
    source_configs = _normalize_source_configs(trader)
    for source_config in source_configs.values():
        strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
        if not strategy_key:
            continue
        instance = strategy_loader.get_instance(strategy_key)
        if instance is None:
            continue
        fn = getattr(instance, "get_filter_diagnostics", None)
        if callable(fn):
            result = fn()
            if result:
                return result
    return None


def _merged_strategy_params_for_source_config(
    source_config: dict[str, Any],
    *,
    include_strategy_defaults: bool = True,
) -> dict[str, Any]:
    source_key = normalize_source_key(source_config.get("source_key"))
    explicit_params = dict(source_config.get("strategy_params") or {})
    strategy_defaults: dict[str, Any] = {}
    strategy_version = normalize_strategy_version(source_config.get("strategy_version"))

    if include_strategy_defaults and strategy_version is None:
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


def _is_execution_session_maintenance_due(
    *,
    trader_id: str,
    run_mode: str,
    now: datetime,
) -> bool:
    if not trader_id:
        return False
    last_run = _execution_session_maintenance_last_run.get(trader_id)
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
        explicit_strategy_params = _merged_strategy_params_for_source_config(
            source_config,
            include_strategy_defaults=False,
        )
        strategy_params = _merged_strategy_params_for_source_config(source_config)
        normalized[source_key] = {
            "source_key": source_key,
            "strategy_key": strategy_key,
            "requested_strategy_key": requested_strategy_key,
            "strategy_version": strategy_version,
            "explicit_strategy_params": explicit_strategy_params,
            "strategy_params": strategy_params,
        }
    return normalized


def _source_open_order_timeout_seconds(
    source_config: dict[str, Any],
    *,
    trader_risk_limits: dict[str, Any] | None = None,
) -> float | None:
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
    # Fallback to per-trader risk_limits.order_ttl_seconds when strategy_params + strategy
    # defaults are both silent. This makes the per-trader Risk Limits flyout's
    # order_ttl_seconds knob actually drive the reaper.
    if not timeout_explicit and default_seconds is None and isinstance(trader_risk_limits, dict):
        risk_limits_default = StrategySDK.resolve_open_order_timeout_seconds(
            trader_risk_limits,
            default_seconds=None,
        )
        if risk_limits_default is not None:
            default_seconds = float(risk_limits_default)
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
        metadata_payload.get("shadow_account_id"),
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


async def _backfill_simulation_ledger_for_active_shadow_orders(
    session: Any,
    *,
    trader_id: str,
    shadow_account_id: str,
) -> dict[str, Any]:
    account_id = str(shadow_account_id or "").strip()
    if not trader_id or not account_id:
        return {"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}

    mode_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    status_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    rows = list(
        (
            await session.execute(
                select(TraderOrder)
                .where(TraderOrder.trader_id == trader_id)
                .where(mode_expr == "shadow")
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
        shadow_sim = payload.get("shadow_simulation") or payload.get("paper_simulation")
        shadow_sim = shadow_sim if isinstance(shadow_sim, dict) else {}
        execution_fee_usd = safe_float(shadow_sim.get("estimated_fee_usd"), None)
        execution_slippage_usd = safe_float(shadow_sim.get("slippage_usd"), None)
        token_id = _shadow_ledger_token_id(payload, direction)
        signal_id = str(row.signal_id or "").strip() or str(row.id)
        strategy_type = str(row.strategy_key or payload.get("strategy_type") or "").strip() or "trader_orchestrator"
        try:
            ledger_row = await simulation_service.record_orchestrator_shadow_fill(
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
                    "reason": "record_orchestrator_shadow_fill_failed",
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
    accumulate: Callable[[str, float], None] | None = None,
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
        _prs_get_experiment_started = time.monotonic()
        experiment_row = await get_active_strategy_experiment(
            session,
            source_key=source_key,
            strategy_key=strategy_key,
        )
        if accumulate is not None:
            accumulate("prs_get_experiment", _prs_get_experiment_started)

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
        _prs_resolve_version_started = time.monotonic()
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
        if accumulate is not None:
            accumulate("prs_resolve_version", _prs_resolve_version_started)

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
            _prs_edge_calibration_started = time.monotonic()
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
            if accumulate is not None:
                accumulate("prs_edge_calibration", _prs_edge_calibration_started)

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
    trader_risk_limits: dict[str, Any] | None = None,
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
        timeout_seconds = _source_open_order_timeout_seconds(
            source_config,
            trader_risk_limits=trader_risk_limits,
        )
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
        candidates_deferred = int(cleanup.get("candidates_deferred", 0))
        if candidates_deferred > 0:
            # Surface persistent backlog so we know when a trader has more
            # stuck orders than the per-call cap can drain in one cycle.
            logger.warning(
                "Open-order cleanup deferred %d/%d candidates over per-call cap "
                "trader=%s source=%s — backlog will drain on subsequent cycles",
                candidates_deferred,
                int(cleanup.get("candidates_total", 0)),
                trader_id,
                source_key,
            )
        source_rows.append(
            {
                "source": source_key,
                "timeout_seconds": timeout_seconds,
                "matched": int(cleanup.get("matched", 0)),
                "candidates_total": int(cleanup.get("candidates_total", 0)),
                "candidates_deferred": candidates_deferred,
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


_orchestrator_lock_consecutive_failures: int = 0
_LOCK_STALE_EVICTION_THRESHOLD: int = 10  # evict after this many consecutive failures (~20s at 2s sleep)


async def _ensure_orchestrator_cycle_lock_owner_for_lane(lane: str) -> bool:
    """Own the cross-process orchestrator lock for a specific worker lane."""
    global _orchestrator_lock_connection, _orchestrator_lock_consecutive_failures

    if _orchestrator_lock_connection is not None:
        return True

    lock_key = _cycle_lock_key_for_lane(lane)
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
            lock_key,
        )
        if not acquired:
            _orchestrator_lock_consecutive_failures += 1
            if _orchestrator_lock_consecutive_failures >= _LOCK_STALE_EVICTION_THRESHOLD:
                await _evict_stale_advisory_lock_holder(connection, lock_key)
                _orchestrator_lock_consecutive_failures = 0
            return False

        _orchestrator_lock_consecutive_failures = 0
        _orchestrator_lock_connection = connection
        connection = None
        logger.info("Acquired orchestrator cross-process cycle lock")

        return True
    finally:
        if connection is not None:
            await _close_orchestrator_lock_connection(connection)


async def _evict_stale_advisory_lock_holder(conn: asyncpg.Connection, lock_key: int) -> None:
    """Terminate stale PG sessions holding the orchestrator advisory lock.

    Advisory locks are session-scoped — if the worker process that acquired the
    lock crashes without cleanly closing its PG connection, the lock persists
    until the connection times out (which may be never with keep-alive).  This
    helper finds the PID holding the lock, checks whether it is idle (no active
    query), and terminates it so the current worker can acquire the lock.
    """
    try:
        high = (lock_key >> 32) & 0xFFFFFFFF
        low = lock_key & 0xFFFFFFFF
        # Convert to signed int32 to match pg_locks classid/objid
        import ctypes
        classid = ctypes.c_int32(high).value
        objid = ctypes.c_int32(low).value

        rows = await conn.fetch(
            """
            SELECT l.pid
            FROM pg_locks l
            JOIN pg_stat_activity a ON a.pid = l.pid
            WHERE l.locktype = 'advisory'
              AND l.classid = $1
              AND l.objid = $2
              AND l.granted = true
              AND a.pid != pg_backend_pid()
              AND a.state = 'idle'
            """,
            classid,
            objid,
        )
        for row in rows:
            stale_pid = row["pid"]
            terminated = await conn.fetchval(
                "SELECT pg_terminate_backend($1)",
                stale_pid,
            )
            logger.warning(
                "Evicted stale advisory lock holder",
                stale_pid=stale_pid,
                terminated=terminated,
                lock_key=lock_key,
            )
    except Exception as exc:
        logger.debug("Stale advisory lock eviction failed: %s", exc)


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
    cached = _live_provider_failure_snapshot_cache.get(cache_key)
    for key, (cached_at, _cached_payload) in list(_live_provider_failure_snapshot_cache.items()):
        if (now - cached_at).total_seconds() > _LIVE_PROVIDER_FAILURE_SNAPSHOT_CACHE_TTL_SECONDS:
            _live_provider_failure_snapshot_cache.pop(key, None)
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
        # Deliberately NOT including pending_live_exit.last_error here.
        # Exit-side timeouts are market-specific (thin book, FAK no-match,
        # stop-loss chasing a moving price) - they are NOT provider
        # infrastructure failures and should not pause new entries for
        # every market.  Entry-side timeouts still count via
        # row.error_message and provider_reconciliation.error.
        candidates = [
            row.error_message,
            payload.get("error_message"),
            payload.get("error"),
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
    """Apply explicitly configured live-risk clamps to trader effective limits."""
    changes: dict[str, dict[str, Any]] = {}
    live_risk_clamps = live_risk_clamps if isinstance(live_risk_clamps, dict) else {}

    if live_risk_clamps.get("enforce_allow_averaging_off"):
        configured_allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))
        if configured_allow_averaging:
            changes["allow_averaging"] = {"configured": configured_allow_averaging, "effective": False}
        effective_risk_limits["allow_averaging"] = False

    if "min_cooldown_seconds" in live_risk_clamps:
        configured_cooldown_seconds = max(0, safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
        min_cooldown_seconds = max(0, safe_int(live_risk_clamps["min_cooldown_seconds"], 0))
        clamped_cooldown_seconds = max(configured_cooldown_seconds, min_cooldown_seconds)
        if clamped_cooldown_seconds != configured_cooldown_seconds:
            changes["cooldown_seconds"] = {
                "configured": configured_cooldown_seconds,
                "effective": clamped_cooldown_seconds,
            }
        effective_risk_limits["cooldown_seconds"] = clamped_cooldown_seconds

    if "max_consecutive_losses_cap" in live_risk_clamps:
        configured = max(1, safe_int(effective_risk_limits.get("max_consecutive_losses"), 4))
        cap = max(1, safe_int(live_risk_clamps["max_consecutive_losses_cap"], 1000))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["max_consecutive_losses"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["max_consecutive_losses"] = clamped

    if "max_open_orders_cap" in live_risk_clamps:
        configured = max(1, safe_int(effective_risk_limits.get("max_open_orders"), 20))
        cap = max(1, safe_int(live_risk_clamps["max_open_orders_cap"], 1000))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["max_open_orders"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["max_open_orders"] = clamped

    if "max_open_positions_cap" in live_risk_clamps:
        configured = max(1, safe_int(effective_risk_limits.get("max_open_positions"), 12))
        cap = max(1, safe_int(live_risk_clamps["max_open_positions_cap"], 1000))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["max_open_positions"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["max_open_positions"] = clamped

    if "max_trade_notional_usd_cap" in live_risk_clamps:
        configured = max(1.0, safe_float(effective_risk_limits.get("max_trade_notional_usd"), 350.0))
        cap = max(1.0, safe_float(live_risk_clamps["max_trade_notional_usd_cap"], 1_000_000.0))
        clamped = min(float(configured), float(cap))
        if clamped != float(configured):
            changes["max_trade_notional_usd"] = {"configured": float(configured), "effective": clamped}
        effective_risk_limits["max_trade_notional_usd"] = clamped

    if "max_per_market_exposure_usd_cap" in live_risk_clamps:
        configured = max(1.0, safe_float(effective_risk_limits.get("max_per_market_exposure_usd"), 350.0))
        cap = max(1.0, safe_float(live_risk_clamps["max_per_market_exposure_usd_cap"], 1_000_000.0))
        clamped = min(float(configured), float(cap))
        if clamped != float(configured):
            changes["max_per_market_exposure_usd"] = {"configured": float(configured), "effective": clamped}
        effective_risk_limits["max_per_market_exposure_usd"] = clamped

    if "max_orders_per_cycle_cap" in live_risk_clamps:
        configured = max(1, safe_int(effective_risk_limits.get("max_orders_per_cycle"), 50))
        cap = max(1, safe_int(live_risk_clamps["max_orders_per_cycle_cap"], 1000))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["max_orders_per_cycle"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["max_orders_per_cycle"] = clamped

    if live_risk_clamps.get("enforce_halt_on_consecutive_losses"):
        configured_halt = bool(effective_risk_limits.get("halt_on_consecutive_losses", False))
        if not configured_halt:
            changes["halt_on_consecutive_losses"] = {"configured": configured_halt, "effective": True}
        effective_risk_limits["halt_on_consecutive_losses"] = True

    # Clamps for the freshly-wired risk knobs. Same ceiling semantics as the
    # existing clamps: if the trader's configured value is above the global
    # cap, the cap wins.
    if "max_daily_spend_usd_cap" in live_risk_clamps:
        configured = max(1.0, safe_float(effective_risk_limits.get("max_daily_spend_usd"), 2000.0))
        cap = max(1.0, safe_float(live_risk_clamps["max_daily_spend_usd_cap"], 100_000_000.0))
        clamped = min(float(configured), float(cap))
        if clamped != float(configured):
            changes["max_daily_spend_usd"] = {"configured": float(configured), "effective": clamped}
        effective_risk_limits["max_daily_spend_usd"] = clamped

    if "max_spread_bps_cap" in live_risk_clamps:
        configured = max(0.0, safe_float(effective_risk_limits.get("max_spread_bps"), 75.0))
        cap = max(0.0, safe_float(live_risk_clamps["max_spread_bps_cap"], 10_000.0))
        clamped = min(float(configured), float(cap)) if configured > 0.0 else float(cap)
        # When the configured knob is 0 (knob off), the cap still applies
        # so the global ceiling enforces a real spread limit even for
        # traders who left their knob untouched.
        if clamped != float(configured):
            changes["max_spread_bps"] = {"configured": float(configured), "effective": clamped}
        effective_risk_limits["max_spread_bps"] = clamped

    if "slippage_bps_cap" in live_risk_clamps:
        configured = max(0.0, safe_float(effective_risk_limits.get("slippage_bps"), 35.0))
        cap = max(0.0, safe_float(live_risk_clamps["slippage_bps_cap"], 10_000.0))
        clamped = min(float(configured), float(cap)) if configured > 0.0 else float(cap)
        if clamped != float(configured):
            changes["slippage_bps"] = {"configured": float(configured), "effective": clamped}
        effective_risk_limits["slippage_bps"] = clamped

    if "retry_limit_cap" in live_risk_clamps:
        configured = max(0, safe_int(effective_risk_limits.get("retry_limit"), 2))
        cap = max(0, safe_int(live_risk_clamps["retry_limit_cap"], 50))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["retry_limit"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["retry_limit"] = clamped

    if "retry_backoff_ms_cap" in live_risk_clamps:
        configured = max(0, safe_int(effective_risk_limits.get("retry_backoff_ms"), 250))
        cap = max(0, safe_int(live_risk_clamps["retry_backoff_ms_cap"], 60_000))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["retry_backoff_ms"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["retry_backoff_ms"] = clamped

    if "order_ttl_seconds_cap" in live_risk_clamps:
        configured = max(1, safe_int(effective_risk_limits.get("order_ttl_seconds"), 1200))
        cap = max(1, safe_int(live_risk_clamps["order_ttl_seconds_cap"], 86_400))
        clamped = min(configured, cap)
        if clamped != configured:
            changes["order_ttl_seconds"] = {"configured": configured, "effective": clamped}
        effective_risk_limits["order_ttl_seconds"] = clamped

    return changes


async def _build_traders_scope_context(session: Any, traders_scope: dict[str, Any]) -> dict[str, Any]:
    normalized_scope = StrategySDK.validate_trader_scope_config(traders_scope)
    now_utc = utcnow()

    scope_cache_key = _traders_scope_cache_key(normalized_scope)
    cached_row = _traders_scope_context_cache.get(scope_cache_key)
    for key, (cached_at, _cached_context) in list(_traders_scope_context_cache.items()):
        if (now_utc - cached_at).total_seconds() > _TRADERS_SCOPE_CONTEXT_CACHE_TTL_SECONDS:
            _traders_scope_context_cache.pop(key, None)
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
            # Hot-path gate: usernames in scope config must be resolved
            # outside the trader cycle.  Skip silently rather than block.
            if not allow_polymarket_rest_call("resolve_wallet_identifier"):
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
        if cached_row is not None:
            logger.warning(
                "Using cached traders scope context after refresh failure",
                exc_info=exc,
            )
            return copy.deepcopy(cached_row[1])
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


# R10-E: fire-and-forget heartbeat for the end-of-cycle hot path.
#
# ``_persist_trader_cycle_heartbeat`` does an UPDATE on Trader then a
# blocking commit.  Under pool pressure R10-A instrumentation captured
# ``slp_heartbeat`` at 3797 ms — budget that directly delays the next
# trader's cycle start.  The heartbeat is a liveness signal (the
# orchestrator's own scheduler doesn't read it in the hot path; it is
# consumed by the GUI + external monitors that can tolerate seconds of
# lag), so we detach the commit to a background task whose own short
# session cannot steal budget from the cycle.
#
# We keep references to in-flight heartbeat tasks so they aren't
# garbage-collected mid-flight (asyncio only weakly holds task refs).
_heartbeat_bg_tasks: set[asyncio.Task] = set()


async def _persist_trader_cycle_heartbeat_bg(
    trader_id: str,
    *,
    advance_run_clock: bool = True,
) -> None:
    """Detached body: open a fresh session, run the UPDATE + commit."""
    try:
        async with AsyncSessionLocal() as session:
            await _persist_trader_cycle_heartbeat(
                session,
                trader_id,
                advance_run_clock=advance_run_clock,
            )
    except Exception as exc:
        logger.debug("Fire-and-forget heartbeat failed: %s", exc)


def schedule_trader_cycle_heartbeat(
    trader_id: str,
    *,
    advance_run_clock: bool = True,
) -> None:
    """Fire-and-forget heartbeat: caller returns immediately.  Safe to
    call from the trader-cycle hot path; the actual UPDATE/commit runs
    on a background task with its own session, decoupled from the
    caller's connection.
    """
    try:
        task = asyncio.create_task(
            _persist_trader_cycle_heartbeat_bg(
                trader_id,
                advance_run_clock=advance_run_clock,
            ),
            name=f"heartbeat-{trader_id}",
        )
    except Exception as exc:
        logger.debug("Failed to schedule fire-and-forget heartbeat: %s", exc)
        return
    _heartbeat_bg_tasks.add(task)
    task.add_done_callback(_heartbeat_bg_tasks.discard)


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


def _maybe_log_per_sample_latency_breach(
    *,
    trader_id: str,
    source: str,
    strategy_key: str,
    signal_id: Any,
    sample: dict[str, Any],
) -> None:
    """Emit a warning-level log for any single hot-path sample whose
    ``ws_release_to_submit_start_ms`` exceeds the per-sample threshold,
    with the full sub-stage breakdown so the soak log immediately reveals
    which stage burned the budget on THIS specific signal.

    Throttled per-trader so one chronically slow trader cannot flood the
    log; the SLA aggregate log still shows the full percentile picture.
    """
    total_ms = safe_int(sample.get(_LATENCY_SLA_STAGE_KEY), None)
    if total_ms is None or total_ms < _LATENCY_SLA_PER_SAMPLE_WARN_MS:
        return

    trader_key = str(trader_id or "").strip() or "unknown"
    now = utcnow()
    last_logged_at = _latency_per_sample_warn_logged_at.get(trader_key)
    if (
        last_logged_at is not None
        and (now - last_logged_at).total_seconds() < _LATENCY_SLA_PER_SAMPLE_WARN_COOLDOWN_SECONDS
    ):
        return
    _latency_per_sample_warn_logged_at[trader_key] = now

    logger.warning(
        "Execution latency per-sample breach",
        trader_id=trader_key,
        source=str(source or "unknown"),
        strategy_key=str(strategy_key or "unknown"),
        signal_id=str(signal_id or "unknown"),
        stage_key=_LATENCY_SLA_STAGE_KEY,
        threshold_ms=_LATENCY_SLA_PER_SAMPLE_WARN_MS,
        total_ms=total_ms,
        armed_to_ws_release_ms=safe_int(sample.get("armed_to_ws_release_ms"), None),
        emit_to_queue_wake_ms=safe_int(sample.get("emit_to_queue_wake_ms"), None),
        wake_to_context_ready_ms=safe_int(sample.get("wake_to_context_ready_ms"), None),
        context_ready_to_decision_ms=safe_int(sample.get("context_ready_to_decision_ms"), None),
        ws_release_to_decision_ms=safe_int(sample.get("ws_release_to_decision_ms"), None),
        decision_to_submit_start_ms=safe_int(sample.get("decision_to_submit_start_ms"), None),
        # Decision->submit-start internal spans (populated alongside the
        # outer stages when the submit path attaches the commit-block
        # sub-timings). Reveals whether the budget burned in the pre-commit
        # DB writes, the COMMIT itself, or the post-commit bookkeeping.
        decision_ready_to_commit_start_ms=safe_int(sample.get("decision_ready_to_commit_start_ms"), None),
        commit_wait_ms=safe_int(sample.get("commit_wait_ms"), None),
        commit_done_to_submit_start_ms=safe_int(sample.get("commit_done_to_submit_start_ms"), None),
        submit_round_trip_ms=safe_int(sample.get("submit_round_trip_ms"), None),
    )


def _maybe_log_execution_latency_sla_breach(*, lane: str, metrics: dict[str, Any]) -> None:
    lane_key = str(lane or _LANE_GENERAL).strip().lower() or _LANE_GENERAL
    execution_latency = metrics.get("execution_latency")
    if not isinstance(execution_latency, dict):
        _latency_sla_breach_last_signature.pop(lane_key, None)
        return
    target_ms = safe_int(execution_latency.get("internal_sla_target_ms"), None)
    overall = execution_latency.get("overall")
    p95 = _latency_stage_percentile(overall, stage_key=_LATENCY_SLA_STAGE_KEY, percentile_key="p95")
    if target_ms is None or p95 is None or p95 <= target_ms:
        _latency_sla_breach_last_signature.pop(lane_key, None)
        return
    alert_threshold_ms = max(int(target_ms * 5), 1000)
    if p95 < alert_threshold_ms:
        _latency_sla_breach_last_signature.pop(lane_key, None)
        return

    overall_p99 = _latency_stage_percentile(overall, stage_key=_LATENCY_SLA_STAGE_KEY, percentile_key="p99")
    sample_count = safe_int(execution_latency.get("sample_count"), None)
    signature = (sample_count, p95, overall_p99)
    if _latency_sla_breach_last_signature.get(lane_key) == signature:
        return
    now = utcnow()
    last_logged_at = _latency_sla_breach_logged_at.get(lane_key)
    if last_logged_at is not None and (now - last_logged_at).total_seconds() < _LATENCY_SLA_BREACH_LOG_COOLDOWN_SECONDS:
        return
    _latency_sla_breach_logged_at[lane_key] = now
    _latency_sla_breach_last_signature[lane_key] = signature

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
    # Sub-stage breakdown so the warning reveals WHERE in the span the
    # time is going. Overall p95 across all traders, plus the worst
    # trader's per-stage p95 so a single misbehaving trader surfaces its
    # specific bottleneck (queue wake vs. live-context vs. DB commit).
    substage_overall_p95 = {
        key: _latency_stage_percentile(overall, stage_key=key, percentile_key="p95")
        for key in _LATENCY_SLA_SUBSTAGE_KEYS
    }
    worst_trader_substage_p95: dict[str, int | None] = {}
    if worst_trader:
        worst_trader_summary = (execution_latency.get("by_trader") or {}).get(worst_trader)
        if isinstance(worst_trader_summary, dict):
            worst_trader_substage_p95 = {
                key: _latency_stage_percentile(worst_trader_summary, stage_key=key, percentile_key="p95")
                for key in _LATENCY_SLA_SUBSTAGE_KEYS
            }
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
        substage_overall_p95=substage_overall_p95,
        worst_trader_substage_p95=worst_trader_substage_p95,
    )


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

    # 2026-05-10: ``load_market_info_for_orders`` does an 8-way
    # concurrent Polymarket REST fan-out.  Holding the DB session
    # across that fan-out caused 11s+ checkout invalidations
    # (asyncpg ``connection is closed``) that broke the SELECT at
    # line 4317 below — observed twice in the same soak with the
    # same order_id 30 min apart, and reliably failing both watchdog
    # passes.  ``session.commit()`` returns the underlying connection
    # to the pool while keeping the session reusable
    # (``expire_on_commit=False`` on AsyncSessionLocal); the next
    # ``session.execute()`` acquires a fresh connection.
    await session.commit()

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
                    force_mark_to_market=False,
                    order_ids=order_ids,
                    reason="terminal_market_watchdog",
                )
            else:
                lifecycle_result = await reconcile_shadow_positions(
                    session,
                    trader_id=trader_id,
                    trader_params={},
                    dry_run=False,
                    force_mark_to_market=True,
                    order_ids=order_ids,
                    reason="terminal_market_watchdog",
                )
            remediation["closed"] = int(remediation["closed"]) + int(lifecycle_result.get("closed", 0) or 0)
            # SOAK-2026-05-18: surface skipped_reasons + would_close from
            # reconcile_live_positions so the watchdog log explains WHY
            # closed=0 when remediation runs but doesn't close.  Soak
            # observed the same stale order flagged every 5 min with
            # closed=0, inventory_updates=1 — no signal as to whether
            # reconcile saw a candidate it refused to close or whether
            # the order didn't match the would-close criteria at all.
            remediation["would_close"] = int(remediation.get("would_close", 0)) + int(
                lifecycle_result.get("would_close", 0) or 0
            )
            lifecycle_skipped = lifecycle_result.get("skipped_reasons") or {}
            if isinstance(lifecycle_skipped, dict) and lifecycle_skipped:
                merged = dict(remediation.get("skipped_reasons") or {})
                for reason, count in lifecycle_skipped.items():
                    merged[str(reason)] = int(merged.get(str(reason), 0)) + int(count or 0)
                remediation["skipped_reasons"] = merged
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
    cycle_timeout_seconds: float = 0.0,
) -> tuple[int, int, int]:
    cycle_started_mono = time.monotonic()
    # Per-stage timings populated at major checkpoints below. Logged at
    # exit if the cycle exceeded the soft threshold so we can tell which
    # stage burned the budget (analogous to fast_trader_runtime's
    # stage_timings_ms field).
    stage_timings_ms: dict[str, float] = {}
    _stage_last_mono = cycle_started_mono
    _accumulators: dict[str, float] = {}

    def _checkpoint(name: str) -> None:
        nonlocal _stage_last_mono
        now = time.monotonic()
        stage_timings_ms[name] = round((now - _stage_last_mono) * 1000.0, 1)
        _stage_last_mono = now

    def _accumulate(name: str, started_mono: float) -> None:
        """Add to a cumulative bucket — used inside per-signal loops."""
        delta_ms = (time.monotonic() - started_mono) * 1000.0
        _accumulators[name] = _accumulators.get(name, 0.0) + delta_ms

    def _accumulate_value(name: str, value: float) -> None:
        """R12-C: push a pre-computed millisecond/count value into a bucket.

        Identical to ``_accumulate`` except it takes the value directly
        instead of timing from a monotonic start.  Used to surface
        sub-metrics (commit_call_ms, retry_count, dirty_rows) that
        ``_commit_with_retry`` computes internally.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        _accumulators[name] = _accumulators.get(name, 0.0) + numeric

    # ARCHITECTURAL CONTRACT: the orchestrator hot path MUST NOT call
    # Polymarket REST APIs.  All wallet state is push-driven via the
    # CLOB user-channel WS into ``WalletStateCache``; market data is
    # push-driven via the market-channel WS into ``feed_manager.cache``.
    # The reconciliation worker (separate worker, slow cadence) is the
    # only writer that calls REST.  ``hot_path_no_rest()`` activates
    # the contextvar that gates every ``polymarket_client.X`` call site
    # in the orchestrator's call chain — accidental future regressions
    # short-circuit to cached/empty data and surface in the
    # ``hot_path.get_blocked_call_counters()`` diagnostic.
    with hot_path_no_rest():
        return await _run_trader_once_inner(
            trader=trader,
            control=control,
            process_signals=process_signals,
            trigger_signal_ids_by_source=trigger_signal_ids_by_source,
            trigger_signal_snapshots_by_source=trigger_signal_snapshots_by_source,
            cycle_timeout_seconds=cycle_timeout_seconds,
            cycle_started_mono=cycle_started_mono,
            stage_timings_ms=stage_timings_ms,
            _accumulators=_accumulators,
            _checkpoint=_checkpoint,
            _accumulate=_accumulate,
            _accumulate_value=_accumulate_value,
        )


async def _run_trader_once_inner(
    *,
    trader: dict[str, Any],
    control: dict[str, Any],
    process_signals: bool,
    trigger_signal_ids_by_source: dict[str, list[str]] | None,
    trigger_signal_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] | None,
    cycle_timeout_seconds: float,
    cycle_started_mono: float,
    stage_timings_ms: dict[str, float],
    _accumulators: dict[str, float],
    _checkpoint,
    _accumulate,
    _accumulate_value,
) -> tuple[int, int, int]:
    decisions_written = 0
    orders_written = 0
    processed_signals = 0
    prefiltered_signals = 0
    prefiltered_by_reason: dict[str, int] = {}
    crypto_scope_prefiltered_dimensions: dict[str, int] = {}
    # Hoisted to function scope so the post-loop cleanup is safe on any
    # early-return path that bypasses the inner declaration block.  The
    # nested live-context setup at the start of the per-signal loop
    # overwrites these with the real task / state when it runs.
    live_contexts_task: asyncio.Task[dict[str, dict[str, Any]]] | None = None
    live_contexts_loaded: bool = True

    # Freshness gate: refuse to trade on stale wallet state.  When the
    # WS disconnects or the reconciliation worker's REST seed lapses,
    # we'd rather skip the cycle and emit an audit alert than make
    # decisions on data that may be minutes out of date.  This is the
    # institutional defensive-degradation contract.
    run_mode_for_gate = _canonical_trader_mode(control.get("mode"), default="shadow")
    if run_mode_for_gate == "live" and process_signals:
        ws_cache = get_wallet_state_cache()
        fresh, fresh_reason = ws_cache.is_fresh()
        if not fresh:
            # Should never fire after bootstrap — see init contract.
            # Throttle to once per 30s per trader (re-logs on reason
            # transition) so a stuck reseed doesn't fill the log with
            # identical messages.
            trader_id_for_log = str(trader.get("id") or "")
            now_mono = time.monotonic()
            prior = _trader_wallet_stale_log_state.get(trader_id_for_log)
            should_log = (
                prior is None
                or prior[0] != fresh_reason
                or now_mono - prior[1] >= _TRADER_WALLET_STALE_LOG_INTERVAL_SECONDS
            )
            if should_log:
                logger.warning(
                    "Trader cycle skipped: wallet state stale",
                    trader_id=trader_id_for_log,
                    reason=fresh_reason,
                    cache_stats=ws_cache.stats_snapshot(),
                )
                _trader_wallet_stale_log_state[trader_id_for_log] = (fresh_reason, now_mono)
            return 0, 0, 0
        # Recovery path: clear the throttle state so the next stale
        # transition re-logs immediately rather than waiting 30s.
        _trader_wallet_stale_log_state.pop(str(trader.get("id") or ""), None)

    _mnt_session_open_started = time.monotonic()
    async with AsyncSessionLocal() as session:
        # R12-A: the per-cycle ``SET LOCAL statement_timeout`` used to
        # cost 0.5-1.3s per cycle (R11-D measurements:
        # ``mnt_session_stmt_timeout = 531/704/1281 ms``) — a full DB
        # round trip on a busy event loop that queued behind other
        # tasks.  The pool already configures ``statement_timeout`` as
        # an asyncpg ``server_settings`` entry (see database.py, 30s
        # default) so every checked-out connection already has a
        # server-side cap.  ``asyncio.wait_for`` wraps the cycle at the
        # asyncio layer, so the belt-and-suspenders per-cycle SET LOCAL
        # was buying us almost nothing in return for ~1s of steady-state
        # overhead.  Drop it entirely.  ``mnt_session_checkout`` stays
        # as instrumentation to prove the checkout itself is free.
        _accumulate("mnt_session_checkout", _mnt_session_open_started)
        _accumulate("mnt_session_open", _mnt_session_open_started)
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
            _mnt_cursor_fetch_started = time.monotonic()
            cursor_runtime_sequence = await get_trader_signal_sequence_cursor(
                session,
                trader_id=trader_id,
            )
            cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(
                session,
                trader_id=trader_id,
            )
            _accumulate("mnt_cursor_fetch", _mnt_cursor_fetch_started)
            if sources:
                _mnt_prefetch_signals_started = time.monotonic()
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
                        exclude_market_ids=None,
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
                _accumulate("mnt_prefetch_signals", _mnt_prefetch_signals_started)

                _mnt_idle_path_started = time.monotonic()
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
                                _idle_payload: dict[str, Any] = {
                                    "idle_maintenance_interval_seconds": idle_maintenance_interval_seconds,
                                    "elapsed_seconds": elapsed_seconds,
                                    "triggered_cycle": stream_trigger_mode,
                                }
                                _idle_message = "Idle cycle: no pending signals."
                                if _is_crypto_source_trader(trader):
                                    try:
                                        from services.market_runtime import get_market_runtime as _get_mrt
                                        _mrt = _get_mrt()
                                        if _mrt:
                                            _idle_payload["dispatch"] = {
                                                "total": getattr(_mrt, "_dispatch_count", 0),
                                                "last_at": getattr(_mrt, "_dispatch_last_at", None),
                                                "handlers": getattr(_mrt, "_dispatch_last_handlers", 0),
                                                "opportunities": getattr(_mrt, "_dispatch_last_opportunities", 0),
                                                "signals_published": getattr(_mrt, "_dispatch_last_signals_published", 0),
                                                "last_error": getattr(_mrt, "_dispatch_last_error", None),
                                            }
                                    except Exception:
                                        pass
                                try:
                                    _cfd = _get_strategy_filter_diagnostics(trader=trader)
                                    if _cfd:
                                        _idle_payload["filter_diagnostics"] = _cfd
                                        _idle_message = _format_filter_diagnostics_message(_cfd)
                                except Exception:
                                    pass
                                await _emit_cycle_heartbeat_if_due(
                                    session,
                                    trader_id=trader_id,
                                    message=_idle_message,
                                    payload=_idle_payload,
                                )
                            effective_process_signals = False
                    if effective_process_signals:
                        _trader_idle_maintenance_last_run[trader_id] = now
                _accumulate("mnt_idle_path", _mnt_idle_path_started)
        open_positions = 0
        occupied_market_ids: set[str] = set()
        reentry_cooldown_market_ids: set[str] = set()
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
        run_execution_session_maintenance = _is_execution_session_maintenance_due(
            trader_id=trader_id,
            run_mode=run_mode,
            now=maintenance_now,
        )
        pre_maintenance_open_order_count = 0
        if stream_trigger_mode and effective_process_signals:
            run_trader_maintenance = False
        elif effective_process_signals and prefetched_signals:
            run_trader_maintenance = False
        if run_trader_maintenance:
            _mnt_open_order_count_started = time.monotonic()
            pre_maintenance_open_order_count = int(
                await get_open_order_count_for_trader(
                    session,
                    trader_id,
                    mode=run_mode,
                )
                or 0
            )
            _accumulate("mnt_open_order_count", _mnt_open_order_count_started)

        if run_trader_maintenance:
            if run_mode == "shadow":
                _mnt_shadow_backfill_started = time.monotonic()
                shadow_account_id = _resolve_shadow_account_id(control, trader)
                backfill_result = await _backfill_simulation_ledger_for_active_shadow_orders(
                    session,
                    trader_id=trader_id,
                    shadow_account_id=shadow_account_id,
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
                _accumulate("mnt_shadow_backfill", _mnt_shadow_backfill_started)
                _mnt_shadow_reconcile_started = time.monotonic()
                force_flatten = resume_policy == "flatten_then_start"
                lifecycle_result = await reconcile_shadow_positions(
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
                _accumulate("mnt_shadow_reconcile", _mnt_shadow_reconcile_started)
            elif run_mode == "live":
                # Live order/position reconciliation is handled by
                # workers.trader_reconciliation_worker so this loop stays focused
                # on strategy selection and execution.
                pass
            if run_mode != "live":
                _mnt_inventory_sync_started = time.monotonic()
                await sync_trader_position_inventory(
                    session,
                    trader_id=trader_id,
                    mode=run_mode,
                )
                _accumulate("mnt_inventory_sync", _mnt_inventory_sync_started)
        session_engine = ExecutionSessionEngine(session)
        run_execution_maintenance = bool(run_execution_session_maintenance and hasattr(session, "execute"))
        if run_execution_maintenance:
            _mnt_session_reconcile_started = time.monotonic()
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
            if reconcile_result.get("failed", 0) > 0:
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="execution_sessions_failed",
                    severity="warn",
                    source="worker",
                    message=f"Failed {int(reconcile_result['failed'])} stale execution session(s)",
                    payload=reconcile_result,
                )
            _accumulate("mnt_session_reconcile", _mnt_session_reconcile_started)
        if run_trader_maintenance and pre_maintenance_open_order_count > 0:
            _mnt_enforce_timeouts_started = time.monotonic()
            try:
                timeout_cleanup = await asyncio.wait_for(
                    _enforce_source_open_order_timeouts(
                        session,
                        trader_id=trader_id,
                        run_mode=run_mode,
                        source_configs=source_configs,
                        trader_risk_limits=risk_limits,
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
            _accumulate("mnt_enforce_timeouts", _mnt_enforce_timeouts_started)
        if run_trader_maintenance:
            _trader_maintenance_last_run[trader_id] = maintenance_now
        if run_execution_maintenance:
            _execution_session_maintenance_last_run[trader_id] = maintenance_now
        _checkpoint("maintenance")
        _setup_maint_events_started = time.monotonic()
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
        _accumulate("setup_maint_events", _setup_maint_events_started)

        if not effective_process_signals:
            await _persist_trader_cycle_heartbeat(
                session,
                trader_id,
                advance_run_clock=bool(process_signals),
            )
            return 0, 0, processed_signals

        # ── Setup-stage state: pre-fetched via TraderCycleContext ──
        #
        # The previous implementation issued ~12 sequential DB queries
        # against the shared session here (open_position_count,
        # open_order_count, demoted_strategy_types, pending_live_exit,
        # occupied_market_ids, reentry_cooldown, provider_failure,
        # daily_realized/unrealized PnL, consecutive_loss_count,
        # last_resolved_loss_at).  Each query was sub-100ms in
        # isolation but the shared session forced strict serialization,
        # and under DB pressure the cumulative pool wait blew the
        # setup stage to p50≈5s / p99≈15s (10h soak 5/2026/05).  This
        # was the dominant contributor to orchestrator backlog and the
        # 15000-cap signal eviction in intent_runtime.
        #
        # Replaced with ``trader_cycle_context.acquire(...)`` which
        # serves all per-trader hot reads from the lock-free hot_state
        # projection, all cross-trader-shared values from a 1s-refreshed
        # global snapshot, and the two remaining DB-bound projections
        # (pending_live_exit, provider_failure) from event-driven
        # caches with a 30s reconciler safety net.  Net DB cost on the
        # hot path: zero.  See ``services/trader_cycle_context.py``.
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
        # provider_health_window_seconds is also re-clamped further
        # below for the live-mode block-decision path; we compute it
        # here so the cycle context's provider-failure projection is
        # keyed off the same window the operator configured.
        _ctx_provider_window_seconds = int(
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
        _setup_ctx_acquire_started = time.monotonic()
        trader_ctx = await trader_cycle_context.acquire(
            trader_id=trader_id,
            mode=run_mode,
            terminal_statuses=pending_live_exit_terminal_statuses,
            provider_window_seconds=_ctx_provider_window_seconds,
        )
        _accumulate("setup_ctx_acquire", _setup_ctx_acquire_started)
        # R11-A: surface per-step timings inside acquire() so slow cycles
        # log which of the 10 awaits ate the setup_ctx_acquire budget.
        # ``acquire_timings_ms`` is already in milliseconds — convert to
        # the seconds-since-start baseline _accumulate expects.
        _ca_timings = getattr(trader_ctx, "acquire_timings_ms", None)
        if _ca_timings:
            _now = time.monotonic()
            for _label, _ms in _ca_timings.items():
                # Fake a "started" timestamp so _accumulate records the
                # same millisecond delta we already measured inside
                # acquire().  No additional clock call per label.
                _accumulate(f"setup_ca_{_label}", _now - (float(_ms) / 1000.0))
        open_positions = trader_ctx.open_position_count
        open_order_count = trader_ctx.open_order_count
        demoted_strategy_types = set(trader_ctx.global_snapshot.demoted_strategy_types)
        # Materialise the pending-live-exit summary as a mutable dict
        # with mutable nested collections.  The cycle context returns
        # immutable mappings/tuples for safety; downstream code in this
        # module appends/mutates without copy, so we restore the
        # legacy mutable shape here.
        pending_live_exit_summary = {
            "count": int(trader_ctx.pending_live_exit_summary.get("count", 0) or 0),
            "order_ids": list(trader_ctx.pending_live_exit_summary.get("order_ids") or ()),
            "market_ids": list(trader_ctx.pending_live_exit_summary.get("market_ids") or ()),
            "signal_ids": list(trader_ctx.pending_live_exit_summary.get("signal_ids") or ()),
            "statuses": dict(trader_ctx.pending_live_exit_summary.get("statuses") or {}),
            "terminal_statuses": list(
                trader_ctx.pending_live_exit_summary.get("terminal_statuses")
                or pending_live_exit_terminal_statuses
            ),
            "identities": [
                dict(item) for item in (trader_ctx.pending_live_exit_summary.get("identities") or ())
            ],
            "identity_keys": list(trader_ctx.pending_live_exit_summary.get("identity_keys") or ()),
        }
        pending_live_exit_count = int(pending_live_exit_summary.get("count", 0) or 0)
        effective_open_positions = max(open_positions, open_order_count)
        occupied_market_ids = set(trader_ctx.occupied_market_ids)
        reentry_cooldown_market_ids = set(trader_ctx.reentry_cooldown_market_ids)

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
                # Pre-fetched via ``trader_cycle_context``: 30 s
                # event-driven cache, 30 s reconciler safety net,
                # zero DB cost on the hot path.  When the runtime
                # config window differs from the cycle-context
                # window, we materialise a mutable dict so downstream
                # code can extend it without aliasing the projection.
                provider_health_snapshot = {
                    "count": int(trader_ctx.live_provider_failure_snapshot.get("count", 0) or 0),
                    "window_seconds": int(
                        trader_ctx.live_provider_failure_snapshot.get(
                            "window_seconds", provider_health_window_seconds
                        )
                        or provider_health_window_seconds
                    ),
                    "errors": [
                        dict(err)
                        for err in (trader_ctx.live_provider_failure_snapshot.get("errors") or ())
                    ],
                }
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
            _setup_block_entries_emit_started = time.monotonic()
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
            _accumulate("setup_block_entries_emit", _setup_block_entries_emit_started)

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
        strict_market_data_age_ms = int(
            max(
                25,
                min(
                    30_000,
                    safe_int(
                        live_market_context_settings.get("max_market_data_age_ms"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT.get("max_market_data_age_ms", 10000)),
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
            _setup_risk_clamps_started = time.monotonic()
            live_risk_clamp_settings = dict(global_runtime_settings.get("live_risk_clamps") or {})
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
            _accumulate("setup_risk_clamps", _setup_risk_clamps_started)
        allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))
        # PnL + loss-streak come from the pre-fetched cycle context.
        # Global values resolve via ``trader_ctx.global_snapshot`` which
        # is refreshed once per second by a long-lived background task —
        # not duplicated per trader.  Per-trader values come from the
        # lock-free hot_state reads materialised when the context was
        # acquired.  Net cost: zero DB reads, zero pool checkouts.
        global_daily_pnl = float(
            trader_ctx.global_snapshot.global_daily_realized_pnl_by_mode.get(run_mode, 0.0)
        )
        trader_daily_pnl = float(trader_ctx.trader_daily_realized_pnl)
        global_unrealized_pnl = float(
            trader_ctx.global_snapshot.global_unrealized_pnl_by_mode.get(run_mode, 0.0)
        )
        trader_unrealized_pnl = float(trader_ctx.trader_unrealized_pnl)
        # Track cumulative notional committed within this cycle so that
        # subsequent risk checks account for intra-cycle exposure even
        # before PnL is realized in the database.
        intra_cycle_committed_usd: float = 0.0
        intra_cycle_seen_market_ids: set[str] = set()
        trader_loss_streak = int(trader_ctx.consecutive_loss_count)
        last_loss_at = trader_ctx.last_resolved_loss_at
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
            _setup_circuit_breaker_started = time.monotonic()
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
                        force_mark_to_market=False,
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
                    safe_exit_result = await reconcile_shadow_positions(
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

            _accumulate("setup_circuit_breaker", _setup_circuit_breaker_started)
            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals

        traders_scope_context: dict[str, Any] | None = None
        traders_source_config = source_configs.get("traders")
        if traders_source_config is not None:
            _setup_traders_scope_started = time.monotonic()
            traders_params = dict(traders_source_config.get("strategy_params") or {})
            traders_scope_context = await _build_traders_scope_context(
                session,
                traders_params.get("traders_scope"),
            )
            _accumulate("setup_traders_scope", _setup_traders_scope_started)
        edge_calibration_cache: dict[str, dict[str, Any]] = {}
        source_runtime_state_cache: dict[str, dict[str, Any]] = {}
        copy_inventory_context: dict[str, Any] = {}
        copy_inventory_loaded = False

        _checkpoint("setup")
        # Kill switch is the very first gate: short-circuit before any signal
        # fetching, live-market context building, or risk evaluation. The
        # per-trader block_new_orders flag has the same semantics scoped to
        # this trader — manage-only mode where reconciliation/exits keep
        # running but no new entry signals get processed.
        if bool(control.get("kill_switch")):
            logger.info(
                "Kill switch active for trader %s — skipping all signal processing",
                trader_id,
            )
            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals
        if bool(trader.get("block_new_orders")):
            logger.info(
                "block_new_orders active for trader %s — skipping all signal processing",
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
                _sl_fetch_started = time.monotonic()
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
                _accumulate("sl_fetch_signals", _sl_fetch_started)
            if not signals:
                if process_signals and not stream_trigger_mode and processed_signals == 0:
                    no_signal_payload: dict[str, Any] = {
                        "processed_signals": processed_signals,
                        "decisions_written": decisions_written,
                        "orders_written": orders_written,
                    }
                    no_signal_message = "Idle cycle: no pending signals."
                    try:
                        filter_diagnostics = _get_strategy_filter_diagnostics(trader=trader)
                        if filter_diagnostics:
                            no_signal_payload["filter_diagnostics"] = filter_diagnostics
                            no_signal_message = _format_filter_diagnostics_message(filter_diagnostics)
                    except Exception:
                        pass
                    await _emit_cycle_heartbeat_if_due(
                        session,
                        trader_id=trader_id,
                        message=no_signal_message,
                        payload=no_signal_payload,
                    )
                    await _persist_trader_cycle_heartbeat(session, trader_id)
                break

            scoped_signals: list[Any] = []
            prefiltered = 0
            _sl_prefilter_started = time.monotonic()
            for signal in signals:
                signal_source = normalize_source_key(getattr(signal, "source", ""))
                source_config = source_configs.get(signal_source)
                signal_id = str(signal.id)
                signal_ts = _signal_cursor_timestamp(signal)
                if source_config is not None:
                    if not source_entry_modes.get(signal_source, True):
                        # Buffered audit writes — was 2 sync DB writes
                        # per filtered signal followed by a single
                        # ``_commit_with_retry`` at the bottom of this
                        # prefilter loop.  Production soak (5/2026/05)
                        # showed 4-7s of ``signal_loop`` time hidden in
                        # this prefilter when 20+ signals failed the
                        # strategy filter.  Hot-state audit_flush_loop
                        # drains these in its own short tx, off the hot
                        # path, with retry.
                        await hot_state.buffer_signal_consumption(
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason="Signal excluded: strategy is manage-existing-only (new entries disabled)",
                        )
                        await hot_state.buffer_signal_cursor(
                            trader_id=trader_id,
                            last_signal_created_at=signal_ts,
                            last_signal_id=signal_id,
                            last_runtime_sequence=safe_int(_signal_runtime_sequence(signal), None),
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
                        await hot_state.buffer_signal_consumption(
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason=(
                                "Signal excluded by source strategy filter "
                                f"(signal={signal_strategy_type or 'unknown'}, allowed={allowed_label})"
                            ),
                        )
                        await hot_state.buffer_signal_cursor(
                            trader_id=trader_id,
                            last_signal_created_at=signal_ts,
                            last_signal_id=signal_id,
                            last_runtime_sequence=safe_int(_signal_runtime_sequence(signal), None),
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
                await hot_state.buffer_signal_consumption(
                    trader_id=trader_id,
                    signal_id=signal_id,
                    outcome="skipped",
                    reason="Signal excluded by crypto scope filters",
                )
                await hot_state.buffer_signal_cursor(
                    trader_id=trader_id,
                    last_signal_created_at=signal_ts,
                    last_signal_id=signal_id,
                    last_runtime_sequence=safe_int(_signal_runtime_sequence(signal), None),
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
            # No commit / reset needed: the prefilter writes above are
            # all routed through ``hot_state.buffer_*`` which drains
            # via the audit_flush_loop in its own short transactions.
            # Was ``await _commit_with_retry(session); session.reset()``
            # which under DB lock contention added 4-7s of dead time
            # to the ``signal_loop`` bucket per cycle.
            _accumulate("sl_prefilter_loop", _sl_prefilter_started)
            if not scoped_signals:
                if prefiltered_signals > 0 and decisions_written == 0 and orders_written == 0 and process_signals:
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
                continue
            signals = scoped_signals
            occupied_signal_ids: set[str] = set()
            cooldown_signal_ids: set[str] = set()
            if run_mode == "live" and occupied_market_ids:
                occupied_signal_ids = {
                    str(signal.id)
                    for signal in signals
                    if str(getattr(signal, "market_id", "") or "").strip() in occupied_market_ids
                }
            if run_mode == "live" and reentry_cooldown_market_ids:
                cooldown_signal_ids = {
                    str(signal.id)
                    for signal in signals
                    if str(getattr(signal, "market_id", "") or "").strip() in reentry_cooldown_market_ids
                }
            _sl_runtime_state_started = time.monotonic()
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
                accumulate=_accumulate,
            )
            _accumulate("sl_prefetched_runtime_state", _sl_runtime_state_started)

            live_contexts: dict[str, dict[str, Any]] = {}
            live_contexts_task: asyncio.Task[dict[str, dict[str, Any]]] | None = None
            live_contexts_loaded = not enable_live_market_context
            live_context_timeout_seconds: Optional[float] = None
            context_candidates: list[Any] = []
            fallback_candidates: list[Any] = []
            if enable_live_market_context:
                for sig in signals:
                    if str(sig.id) in occupied_signal_ids or str(sig.id) in cooldown_signal_ids:
                        continue
                    sig_source = normalize_source_key(getattr(sig, "source", ""))
                    if strict_ws_pricing_enforced and sig_source in {"crypto", "scanner"}:
                        context_candidates.append(sig)
                        continue
                    if strict_ws_pricing_enforced:
                        fallback_candidates.append(sig)
                        continue
                    source_config = source_configs.get(sig_source)
                    if _supports_live_market_context(sig, source_config):
                        context_candidates.append(sig)

                async def _load_live_contexts() -> dict[str, dict[str, Any]]:
                    loaded_contexts: dict[str, dict[str, Any]] = {}
                    if strict_ws_pricing_enforced and context_candidates:
                        loaded_contexts = await build_cached_live_signal_contexts(
                            context_candidates,
                            max_history_points=max_history_points,
                            strict_ws_only=True,
                        )
                    elif context_candidates:
                        loaded_contexts = await build_cached_live_signal_contexts(
                            context_candidates,
                            max_history_points=max_history_points,
                            strict_ws_only=False,
                        )
                    if fallback_candidates:
                        try:
                            loaded_contexts.update(
                                await build_cached_live_signal_contexts(
                                    fallback_candidates,
                                    max_history_points=max_history_points,
                                    strict_ws_only=False,
                                )
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as fb_exc:
                            logger.warning("Fallback live market context failed (%s)", type(fb_exc).__name__)
                    return loaded_contexts

                live_context_timeout_seconds = _remaining_cycle_budget_seconds(
                    cycle_started_mono=cycle_started_mono,
                    cycle_timeout_seconds=cycle_timeout_seconds,
                    reserve_seconds=2.0,
                )
                # Run the batch live-context fetch concurrent with the
                # per-signal loop's prefilter + setup work.  The previous
                # implementation awaited this sequentially BEFORE the
                # loop, so every cycle paid the full fetch cost as
                # outside-per-signal overhead.  ``build_cached_live_signal_contexts``
                # is pure-memory (no DB / HTTP) so we don't need
                # ``release_conn``; the task simply runs while the loop
                # walks occupied/cooldown signals, resolves strategy
                # versions, runs validation, etc.  The first iteration
                # that needs ``live_context`` awaits the task via
                # ``_ensure_live_contexts_resolved``.
                live_contexts_task = asyncio.create_task(
                    _load_live_contexts(),
                    name=f"live-contexts-{trader_id}",
                )

            async def _ensure_live_contexts_resolved() -> dict[str, dict[str, Any]]:
                nonlocal live_contexts, live_contexts_loaded, live_contexts_task
                if live_contexts_loaded:
                    return live_contexts
                task = live_contexts_task
                if task is None:
                    live_contexts_loaded = True
                    return live_contexts
                try:
                    if live_context_timeout_seconds is not None:
                        if live_context_timeout_seconds < 0.25:
                            task.cancel()
                            live_contexts = {}
                        else:
                            live_contexts = await asyncio.wait_for(
                                asyncio.shield(task),
                                timeout=live_context_timeout_seconds,
                            )
                    else:
                        live_contexts = await task
                except asyncio.TimeoutError:
                    live_contexts = {}
                    logger.warning(
                        "Live market context refresh timed out for trader=%s timeout=%.2fs",
                        trader_id,
                        max(0.25, float(live_context_timeout_seconds or 0.0)),
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
                        logger.debug(
                            "Failed to load cached live market contexts after refresh failure",
                            exc_info=cache_exc,
                        )
                        live_contexts = {}
                    if not live_contexts:
                        exc_name = type(exc).__name__
                        exc_message = str(exc).strip()
                        if exc_message:
                            logger.warning(
                                "Live market context refresh failed (%s): %s", exc_name, exc_message
                            )
                        else:
                            logger.warning("Live market context refresh failed (%s)", exc_name)
                finally:
                    live_contexts_loaded = True
                    live_contexts_task = None
                return live_contexts

            deferred_signals = 0
            deferred_by_reason: dict[str, int] = {}
            expired_signals = 0
            expired_by_reason: dict[str, int] = {}
            defer_signal_processing = False
            _last_per_signal_started: float | None = None
            # Tracks which sub-stage the current iteration is inside.  When
            # an iteration burns >= ``_per_signal_slow_log_seconds`` we log
            # the offender — strategy / source / market_id / stage — so the
            # outlier trader (production: c8851ef7…) is identifiable from
            # the log alone instead of requiring a fresh soak with
            # speculative instrumentation.
            _current_signal_stage: str = "init"
            _current_signal_stage_started: float = time.monotonic()
            _per_signal_slow_log_seconds: float = 8.0

            def _enter_stage(stage_name: str) -> None:
                nonlocal _current_signal_stage, _current_signal_stage_started
                _current_signal_stage = stage_name
                _current_signal_stage_started = time.monotonic()

            for signal in signals:
                _now = time.monotonic()
                if _last_per_signal_started is not None:
                    # Cumulative per-signal cost — captured regardless of
                    # which `continue` branch the prior iteration took.
                    _per_signal_elapsed = (_now - _last_per_signal_started) * 1000.0
                    _accumulators["per_signal_total"] = (
                        _accumulators.get("per_signal_total", 0.0)
                        + _per_signal_elapsed
                    )
                    if _per_signal_elapsed >= _per_signal_slow_log_seconds * 1000.0:
                        # Capture the exact pre-strategy stage that ate
                        # the budget — nothing else in the orchestrator
                        # tells us which stage was running when the slow
                        # iteration finished.
                        try:
                            logger.warning(
                                "Per-signal iteration slow",
                                trader_id=trader_id,
                                mode=run_mode,
                                signal_id=str(getattr(signal, "id", "")),
                                signal_source=normalize_source_key(
                                    getattr(signal, "source", "")
                                ),
                                strategy_key=str(getattr(signal, "strategy_key", "") or ""),
                                market_id=str(getattr(signal, "market_id", "") or ""),
                                runtime_sequence=safe_int(
                                    getattr(signal, "runtime_sequence", None), None
                                ),
                                elapsed_ms=round(_per_signal_elapsed, 1),
                                last_stage=_current_signal_stage,
                                last_stage_age_ms=round(
                                    (_now - _current_signal_stage_started) * 1000.0, 1
                                ),
                            )
                        except Exception:
                            # Diagnostic log must never break the loop.
                            pass
                _last_per_signal_started = _now
                _enter_stage("loop_top")
                # Budget-based bail: if the cycle has consumed enough of
                # its timeout that we likely cannot process another
                # signal cleanly, defer the rest to the next cycle.
                # Pre-fix the loop ran to completion regardless of
                # budget; a 23-signal batch (≈3.5s/signal) consistently
                # blew the 30s soft timeout, leaving the cycle stuck
                # in the inflight slot and starving sibling traders.
                # Reserve ~3s for the trailing commit + book-keeping;
                # if less than that remains, defer the remaining
                # signals to the next reconcile cycle.
                _budget_remaining = _remaining_cycle_budget_seconds(
                    cycle_started_mono=cycle_started_mono,
                    cycle_timeout_seconds=cycle_timeout_seconds,
                    reserve_seconds=3.0,
                )
                if _budget_remaining is not None and _budget_remaining <= 0:
                    deferred_signals += int(len(signals) - signals.index(signal))
                    deferred_by_reason["cycle_budget_exhausted"] = deferred_by_reason.get(
                        "cycle_budget_exhausted", 0
                    ) + (len(signals) - signals.index(signal))
                    defer_signal_processing = True
                    break
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
                    _enter_stage("signal_persist")
                    _ps_persist_started = time.monotonic()
                    # Round 8-B: pass `accumulate` so the helper records
                    # ps_sp_prep (row_values + JSON-safe serialization)
                    # and ps_sp_execute (the DB round-trip) separately.
                    # Post-R7 ps_signal_persist was 2.3-6.8s per cycle
                    # with zero sub-breakdown — we need to know whether
                    # Python or Postgres owns the time before we can
                    # optimize further.
                    await _ensure_runtime_signal_persisted(
                        session, signal, accumulate=_accumulate
                    )
                    _accumulate("ps_signal_persist", _ps_persist_started)
                    if source_config is None:
                        _enter_stage("no_source_config_skip")
                        _ps_skipwrite_started = time.monotonic()
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
                        _accumulate("ps_skip_writes", _ps_skipwrite_started)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        continue

                    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    requested_strategy_key = str(source_config.get("requested_strategy_key") or "").strip().lower()
                    strategy_key_for_output = requested_strategy_key or strategy_key
                    strategy_params = dict(source_config.get("strategy_params") or {})
                    explicit_strategy_params = dict(source_config.get("explicit_strategy_params") or {})
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
                    if strict_ws_pricing_enforced and signal_source in ("crypto", "scanner"):
                        strategy_params = _enforce_strict_ws_strategy_params(
                            strategy_params,
                            strict_age_budget_ms=strict_age_budget_ms,
                            strict_ws_price_sources=list(strict_ws_price_sources),
                        )
                        signal_payload = getattr(signal, "payload_json", None)
                        signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
                        signal_emitted_at = _parse_iso(
                            str(signal_payload.get("signal_emitted_at") or signal_payload.get("ingested_at") or "")
                        )
                        signal_release_anchor_at = signal_emitted_at
                        if signal_source == "scanner":
                            scanner_row_timestamp = _signal_cursor_timestamp(signal)
                            if isinstance(scanner_row_timestamp, datetime):
                                if scanner_row_timestamp.tzinfo is None:
                                    scanner_row_timestamp = scanner_row_timestamp.replace(tzinfo=timezone.utc)
                                if signal_release_anchor_at is None or scanner_row_timestamp > signal_release_anchor_at:
                                    signal_release_anchor_at = scanner_row_timestamp
                        strict_release_age_budget_ms = max(
                            50,
                            safe_int(strategy_params.get("max_market_data_age_ms"), int(strict_age_budget_ms)),
                        )
                        if signal_source in ("crypto", "scanner") and signal_release_anchor_at is not None:
                            signal_release_age_ms = max(
                                0,
                                int((signal_wake_started_at - signal_release_anchor_at).total_seconds() * 1000),
                            )
                            if signal_release_age_ms > strict_release_age_budget_ms:
                                _enter_stage("strict_ws_release_stale")
                                _ps_strict_started = time.monotonic()
                                runtime = get_intent_runtime()
                                if run_mode == "live" and hasattr(runtime, "update_signal_status"):
                                    await runtime.update_signal_status(
                                        signal_id=signal_id,
                                        status="expired",
                                    )
                                    await set_trade_signal_status(
                                        session,
                                        signal_id=signal_id,
                                        status="expired",
                                        commit=False,
                                    )
                                    expired_signals += 1
                                    expired_by_reason["strict_ws_pricing_signal_release_stale"] = (
                                        expired_by_reason.get("strict_ws_pricing_signal_release_stale", 0) + 1
                                    )
                                elif hasattr(runtime, "defer_signal"):
                                    await runtime.defer_signal(
                                        signal_id=signal_id,
                                        required_token_ids=list(getattr(signal, "required_token_ids", None) or []),
                                        reason="strict_ws_pricing_signal_release_stale",
                                        min_observed_at=signal_release_anchor_at,
                                        required_max_age_ms=strict_release_age_budget_ms,
                                    )
                                    deferred_signals += 1
                                    deferred_by_reason["strict_ws_pricing_signal_release_stale"] = (
                                        deferred_by_reason.get("strict_ws_pricing_signal_release_stale", 0) + 1
                                    )
                                else:
                                    await set_trade_signal_status(
                                        session,
                                        signal_id=signal_id,
                                        status="expired",
                                        commit=False,
                                    )
                                    expired_signals += 1
                                    expired_by_reason["strict_ws_pricing_signal_release_stale"] = (
                                        expired_by_reason.get("strict_ws_pricing_signal_release_stale", 0) + 1
                                    )
                                _accumulate("ps_strict_ws_writes", _ps_strict_started)
                                defer_signal_processing = True
                                continue
                    requested_strategy_version = prefetched_source_runtime.get("requested_strategy_version")
                    requested_strategy_version_error = (
                        str(prefetched_source_runtime.get("requested_strategy_version_error") or "").strip() or None
                    )
                    signal_market_id = str(getattr(signal, "market_id", "") or "").strip()
                    _enter_stage("prefilter_check")
                    if signal_id in occupied_signal_ids:
                        _enter_stage("occupied_writes")
                        _ps_pre_started = time.monotonic()
                        # Stacking-guard short-circuit: the heavy write path
                        # (trader_decision + decision_checks + trader_event) is
                        # pure noise for occupied markets — the scanner publishes
                        # the same markets repeatedly and we block them repeatedly.
                        # All three audit writes routed through hot_state buffers
                        # (was 3 sync session writes per occupied signal — under
                        # DB lock contention this hit ps_prefilter_writes=9.4s
                        # for trader c8851ef7 in production).  audit_flush_loop
                        # drains in its own short tx with retry.
                        blocked_reason = "Stacking guard: market already occupied"
                        await hot_state.buffer_signal_status(
                            signal_id=signal_id,
                            status="skipped",
                        )
                        await hot_state.buffer_signal_consumption(
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=None,
                            outcome="blocked",
                            reason=blocked_reason,
                            payload={"prefilter": "occupied_market", "market_id": signal_market_id},
                        )
                        await hot_state.buffer_signal_cursor(
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            last_runtime_sequence=safe_int(_signal_runtime_sequence(signal), None),
                        )
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        prefiltered_signals += 1
                        prefiltered_by_reason["occupied_market"] = prefiltered_by_reason.get("occupied_market", 0) + 1
                        _accumulate("ps_prefilter_writes", _ps_pre_started)
                        continue
                    if signal_id in cooldown_signal_ids:
                        _enter_stage("cooldown_writes")
                        _ps_pre_started = time.monotonic()
                        # Same short-circuit as the stacking-guard branch above:
                        # all audit writes buffered through hot_state.
                        blocked_reason = "Reentry cooldown: market recently closed"
                        await hot_state.buffer_signal_status(
                            signal_id=signal_id,
                            status="skipped",
                        )
                        await hot_state.buffer_signal_consumption(
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=None,
                            outcome="blocked",
                            reason=blocked_reason,
                            payload={"prefilter": "reentry_cooldown", "market_id": signal_market_id},
                        )
                        await hot_state.buffer_signal_cursor(
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            last_runtime_sequence=safe_int(_signal_runtime_sequence(signal), None),
                        )
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        prefiltered_signals += 1
                        prefiltered_by_reason["reentry_cooldown"] = (
                            prefiltered_by_reason.get("reentry_cooldown", 0) + 1
                        )
                        _accumulate("ps_prefilter_writes", _ps_pre_started)
                        continue
                    if not live_contexts_loaded:
                        _enter_stage("live_context_await")
                        _ps_lc_started = time.monotonic()
                        await _ensure_live_contexts_resolved()
                        _accumulate("ps_live_context_await", _ps_lc_started)
                    live_context = live_contexts.get(signal_id, {})
                    payload_live_context = _payload_live_market_context(signal)
                    if not live_context and payload_live_context:
                        live_context = payload_live_context
                    runtime_signal = RuntimeTradeSignalView(signal, live_context=live_context)
                    runtime_signal.source = signal_source
                    if strict_ws_pricing_enforced and signal_source == "crypto":
                        effective_strict_sources: list[str] = []
                        for raw_source in (strategy_params.get("strict_ws_price_sources") or strict_ws_price_sources):
                            normalized_source = str(raw_source or "").strip().lower()
                            if normalized_source and normalized_source not in effective_strict_sources:
                                effective_strict_sources.append(normalized_source)
                        effective_source_set = set(effective_strict_sources)
                        effective_max_age_ms = max(
                            25,
                            safe_int(strategy_params.get("max_market_data_age_ms"), int(strict_age_budget_ms)),
                        )
                        live_source = str(
                            live_context.get("market_data_source")
                            or live_context.get("live_selected_price_source")
                            or ""
                        ).strip().lower()
                        live_price = safe_float(live_context.get("live_selected_price"), None)
                        live_age_ms = safe_float(live_context.get("market_data_age_ms"), None)
                        strict_context_ok = (
                            live_source in effective_source_set
                            and live_price is not None
                            and live_price > 0.0
                            and live_age_ms is not None
                            and live_age_ms <= effective_max_age_ms
                        )
                        if not strict_context_ok:
                            _enter_stage("strict_ws_live_context_unavailable")
                            _ps_strict_ctx_started = time.monotonic()
                            live_reason_parts: list[str] = []
                            live_reason_parts.append(f"source={live_source or 'unknown'}")
                            live_reason_parts.append(
                                f"age_ms={live_age_ms:.0f}" if live_age_ms is not None else "age_ms=unknown"
                            )
                            live_reason_parts.append(f"max_age_ms={effective_max_age_ms}")
                            live_reason_parts.append(f"required_sources={effective_strict_sources}")
                            runtime = get_intent_runtime()
                            if signal_source == "crypto" and run_mode == "live" and hasattr(runtime, "update_signal_status"):
                                await runtime.update_signal_status(
                                    signal_id=signal_id,
                                    status="expired",
                                )
                                await set_trade_signal_status(
                                    session,
                                    signal_id=signal_id,
                                    status="expired",
                                    commit=False,
                                )
                                expired_signals += 1
                                expired_by_reason["strict_ws_pricing_live_context_unavailable"] = (
                                    expired_by_reason.get("strict_ws_pricing_live_context_unavailable", 0) + 1
                                )
                            elif hasattr(runtime, "defer_signal"):
                                await runtime.defer_signal(
                                    signal_id=signal_id,
                                    required_token_ids=list(getattr(signal, "required_token_ids", None) or []),
                                    reason="strict_ws_pricing_live_context_unavailable",
                                    min_observed_at=signal_wake_started_at,
                                    required_max_age_ms=effective_max_age_ms,
                                )
                                deferred_signals += 1
                                deferred_by_reason["strict_ws_pricing_live_context_unavailable"] = (
                                    deferred_by_reason.get("strict_ws_pricing_live_context_unavailable", 0) + 1
                                )
                            else:
                                await set_trade_signal_status(
                                    session,
                                    signal_id=signal_id,
                                    status="expired",
                                    commit=False,
                                )
                                expired_signals += 1
                                expired_by_reason["strict_ws_pricing_live_context_unavailable"] = (
                                    expired_by_reason.get("strict_ws_pricing_live_context_unavailable", 0) + 1
                                )
                            _accumulate("ps_strict_ws_writes", _ps_strict_ctx_started)
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
                        _enter_stage("decision_skip_strategy_version")
                        _ps_skip_decision_started = time.monotonic()
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
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        _accumulate("ps_decision_skip_writes", _ps_skip_decision_started)
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
                        _enter_stage("decision_skip_strategy_unavailable")
                        _ps_skip_decision_started = time.monotonic()
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
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        processed_signals += 1
                        _accumulate("ps_decision_skip_writes", _ps_skip_decision_started)
                        continue

                    if signal_source == "traders":
                        if traders_scope_context is None:
                            _enter_stage("traders_scope_context_build")
                            _ps_scope_started = time.monotonic()
                            traders_params = dict(source_config.get("strategy_params") or {})
                            async with AsyncSessionLocal() as _scope_session:
                                traders_scope_context = await _build_traders_scope_context(
                                    _scope_session,
                                    traders_params.get("traders_scope"),
                                )
                            _accumulate("ps_traders_scope_build", _ps_scope_started)
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
                            _enter_stage("decision_skip_traders_scope")
                            _ps_skip_decision_started = time.monotonic()
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
                            cursor_created_at = _signal_cursor_timestamp(signal)
                            cursor_signal_id = signal_id
                            cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                            processed_signals += 1
                            _accumulate("ps_decision_skip_writes", _ps_skip_decision_started)
                            continue

                    copy_risk_context: dict[str, Any] = {}
                    copy_allocation_context: dict[str, Any] = {}
                    copy_inventory_context_for_signal: dict[str, Any] = {}
                    if signal_source == "traders":
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

                        _enter_stage("copy_exposure_lookup")
                        _ps_copy_exposure_started = time.monotonic()
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
                        _accumulate("ps_copy_exposure_lookup", _ps_copy_exposure_started)
                        copy_allocation_context = {
                            "source_wallet": source_wallet,
                            "current_source_exposure_usd": float(current_source_exposure_usd),
                            "current_leader_exposure_usd": float(current_leader_exposure_usd),
                        }

                        if run_mode == "live":
                            # Lazy-load inventory once per cycle. Previously
                            # this re-fetched on every SELL signal, which on
                            # backlogged copy-trade traders meant 20-50s of
                            # extra DB+Polymarket work per cycle and was the
                            # main reason 78865ce8 / f88cc061 / c8851ef7
                            # cycles were hitting the 30s soft timeout.
                            # Strategy code validates inventory downstream
                            # at order submission, so per-signal staleness
                            # within a cycle (max 30s) is bounded and safe.
                            if not copy_inventory_loaded:
                                copy_inventory_loaded = True
                                _wallet_fetch_started = time.monotonic()
                                # Inventory now sourced from the in-memory
                                # WS-driven wallet_state_cache instead of
                                # ``list_live_wallet_positions_for_trader``.
                                # Production saw 1-5s on the DB path; cache
                                # read is a sub-microsecond dict snapshot.
                                # Freshness is enforced by the same gate the
                                # rest of the hot path uses (REST seed +
                                # WS connectivity); a stale cache yields
                                # ``available=False`` so strategy code falls
                                # through to its existing safe-default branch
                                # rather than acting on lies.
                                try:
                                    ws_cache = get_wallet_state_cache()
                                    fresh, fresh_reason = ws_cache.is_fresh()
                                    if not fresh:
                                        copy_inventory_context = {
                                            "available": False,
                                            "wallet_address": ws_cache.wallet_address(),
                                            "token_inventory": {},
                                            "error": f"wallet_state_cache_stale:{fresh_reason}",
                                            "fetched_at": utcnow().isoformat(),
                                        }
                                    else:
                                        token_inventory = ws_cache.to_token_inventory()
                                        copy_inventory_context = {
                                            "available": bool(token_inventory),
                                            "wallet_address": ws_cache.wallet_address(),
                                            "token_inventory": token_inventory,
                                            "error": None,
                                            "fetched_at": utcnow().isoformat(),
                                        }
                                except Exception as exc:
                                    copy_inventory_context = {
                                        "available": False,
                                        "wallet_address": None,
                                        "token_inventory": {},
                                        "error": str(exc),
                                        "fetched_at": utcnow().isoformat(),
                                    }
                                _accumulate("ps_copy_inventory_fetch", _wallet_fetch_started)
                            copy_inventory_context_for_signal = copy_inventory_context

                    _pre_gate_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                    _pre_gate_occupied = bool(
                        _pre_gate_market_id
                        and (run_mode == "live" or not allow_averaging)
                        and (
                            _pre_gate_market_id in occupied_market_ids
                            or _pre_gate_market_id in intra_cycle_seen_market_ids
                        )
                    )
                    _pre_gate_reentry_cooldown = bool(
                        _pre_gate_market_id
                        and _pre_gate_market_id in reentry_cooldown_market_ids
                    )
                    import time as _time_mod
                    _nofill_key = (trader_id, _pre_gate_market_id)
                    _nofill_expiry = _nofill_cooldowns.get(_nofill_key, 0.0)
                    _nofill_blocked = bool(
                        _pre_gate_market_id
                        and _nofill_expiry > _time_mod.monotonic()
                    )
                    _skip_strategy_evaluation = bool(
                        _pre_gate_occupied
                        or _pre_gate_reentry_cooldown
                        or _nofill_blocked
                    )

                    if _skip_strategy_evaluation:
                        decision_obj = StrategyDecision(
                            decision="selected",
                            reason="Platform pre-gate short-circuited strategy evaluation",
                            score=None,
                            size_usd=None,
                            checks=[],
                            payload={},
                        )
                        checks_payload = []
                    else:
                        loop = asyncio.get_running_loop()
                        evaluation_timeout_seconds = _strategy_evaluation_timeout_seconds(
                            control,
                            source_config,
                            strategy_params,
                        )
                        remaining_cycle_budget = _remaining_cycle_budget_seconds(
                            cycle_started_mono=cycle_started_mono,
                            cycle_timeout_seconds=cycle_timeout_seconds,
                            reserve_seconds=1.0,
                        )
                        if remaining_cycle_budget is not None:
                            evaluation_timeout_seconds = max(
                                0.1,
                                min(float(evaluation_timeout_seconds), float(remaining_cycle_budget)),
                            )
                        _enter_stage("strategy_evaluate")
                        _evaluate_started = time.monotonic()
                        # Layer the strategy's loaded DB config UNDER the per-
                        # trader params so evaluate() sees the same gate values
                        # as on_event().  Trader params win (last layer) — any
                        # per-trader override behaves as before, but a gate
                        # tuned via the strategy-manager UI now applies at
                        # decision time too.
                        _merged_eval_params: dict[str, Any] = {}
                        _strategy_db_config = getattr(strategy, "config", None)
                        if isinstance(_strategy_db_config, dict):
                            _merged_eval_params.update(_strategy_db_config)
                        if isinstance(strategy_params, dict):
                            _merged_eval_params.update(strategy_params)
                        decision_future = loop.run_in_executor(
                            _STRATEGY_EVAL_POOL,
                            strategy.evaluate,
                            runtime_signal,
                            {
                                "params": _merged_eval_params,
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
                                _accumulate("ps_strategy_evaluate", _evaluate_started)
                                raise RuntimeError(
                                    f"Strategy evaluation timed out after {evaluation_timeout_seconds:.1f}s"
                                ) from exc
                            except asyncio.CancelledError:
                                if not decision_future.done():
                                    decision_future.cancel()
                                _accumulate("ps_strategy_evaluate", _evaluate_started)
                                raise
                        _accumulate("ps_strategy_evaluate", _evaluate_started)
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
                    if not _skip_strategy_evaluation and decision_obj.decision == "selected" and trading_schedule_ok:
                        _enter_stage("risk_eval_setup")
                        _ps_risk_started = time.monotonic()
                        gross_exposure = await get_gross_exposure(session, mode=run_mode)
                        market_exposure = await get_market_exposure(
                            session,
                            str(signal.market_id),
                            mode=run_mode,
                        )
                        _accumulate("ps_risk_eval_setup", _ps_risk_started)
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

                    _pre_gate_occupied = _pre_gate_occupied and str(getattr(decision_obj, "decision", "") or "").strip() == "selected"
                    _pre_gate_reentry_cooldown = (
                        _pre_gate_reentry_cooldown
                        and str(getattr(decision_obj, "decision", "") or "").strip() == "selected"
                    )
                    _nofill_blocked = _nofill_blocked and str(getattr(decision_obj, "decision", "") or "").strip() == "selected"
                    _pre_gate_blocked = bool(_pre_gate_occupied or _pre_gate_reentry_cooldown or _nofill_blocked)

                    if _pre_gate_blocked and _nofill_blocked:
                        gate_result = {
                            "final_decision": "blocked",
                            "final_reason": (
                                f"Failed-entry cooldown: market in "
                                f"{int(_nofill_expiry - _time_mod.monotonic())}s cooldown after recent failed/cancelled entry"
                            ),
                            "score": getattr(decision_obj, "score", None),
                            "size_usd": getattr(decision_obj, "size_usd", None),
                            "checks_payload": checks_payload,
                            "risk_snapshot": None,
                            "strategy_decision": str(getattr(decision_obj, "decision", "blocked") or "blocked"),
                            "strategy_reason": str(getattr(decision_obj, "reason", "") or ""),
                            "platform_gates": [
                                {
                                    "gate": "nofill_cooldown",
                                    "passed": False,
                                    "reason": "Market in failed-entry cooldown",
                                }
                            ],
                        }
                    elif _pre_gate_reentry_cooldown:
                        gate_result = {
                            "final_decision": "blocked",
                            "final_reason": "Reentry cooldown: market recently closed (pre-gate)",
                            "score": getattr(decision_obj, "score", None),
                            "size_usd": getattr(decision_obj, "size_usd", None),
                            "checks_payload": checks_payload,
                            "risk_snapshot": None,
                            "strategy_decision": str(getattr(decision_obj, "decision", "blocked") or "blocked"),
                            "strategy_reason": str(getattr(decision_obj, "reason", "") or ""),
                            "platform_gates": [
                                {
                                    "gate": "reentry_cooldown_pre_gate",
                                    "passed": False,
                                    "reason": "Market is in reentry cooldown (pre-gate)",
                                }
                            ],
                        }
                    elif _pre_gate_blocked:
                        gate_result = {
                            "final_decision": "blocked",
                            "final_reason": "Stacking guard: market already occupied (pre-gate)",
                            "score": getattr(decision_obj, "score", None),
                            "size_usd": getattr(decision_obj, "size_usd", None),
                            "checks_payload": checks_payload,
                            "risk_snapshot": None,
                            "strategy_decision": str(getattr(decision_obj, "decision", "blocked") or "blocked"),
                            "strategy_reason": str(getattr(decision_obj, "reason", "") or ""),
                            "platform_gates": [
                                {
                                    "gate": "stacking_guard_pre_gate",
                                    "passed": False,
                                    "reason": "Market already occupied (pre-gate)",
                                }
                            ],
                        }
                    else:
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
                            occupied_market_ids=occupied_market_ids | intra_cycle_seen_market_ids,
                            pending_live_exit_count=pending_live_exit_count,
                            pending_live_exit_summary=pending_live_exit_summary,
                            pending_live_exit_max_allowed=pending_live_exit_max_allowed,
                            pending_live_exit_identity_guard_enabled=pending_live_exit_identity_guard_enabled,
                            portfolio_allocator=portfolio_allocator,
                            risk_evaluator=risk_evaluator,
                            invoke_hooks=True,
                            strategy_params=strategy_params,
                            global_runtime=global_runtime_settings,
                            execution_mode=run_mode,
                            demoted_strategy_types=demoted_strategy_types,
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
                    order_status = None
                    if final_decision == "selected":
                        if _pre_gate_market_id:
                            intra_cycle_seen_market_ids.add(_pre_gate_market_id)

                        # ── DB-level stacking verification before execution ──
                        # Final authoritative check against the database before
                        # spending real money.  The hot-state pre-gate is fast but
                        # can miss entries during reseed or after wallet_absent_close.
                        if _pre_gate_market_id and run_mode == "live":
                            position_occupied = bool(
                                (
                                    await session.execute(
                                        select(func.count())
                                        .select_from(TraderPosition)
                                        .where(TraderPosition.trader_id == trader_id)
                                        .where(func.lower(func.coalesce(TraderPosition.mode, "")) == run_mode)
                                        .where(func.lower(func.coalesce(TraderPosition.status, "")) == "open")
                                        .where(TraderPosition.market_id == _pre_gate_market_id)
                                    )
                                ).scalar_one()
                                or 0
                            ) > 0
                            authority_pending_cutoff = utcnow() - timedelta(
                                seconds=LIVE_AUTHORITY_PENDING_MARKET_BLOCK_SECONDS
                            )
                            order_rows = (
                                await session.execute(
                                    select(
                                        TraderOrder.status,
                                        TraderOrder.payload_json,
                                        TraderOrder.updated_at,
                                        TraderOrder.created_at,
                                    )
                                    .where(TraderOrder.trader_id == trader_id)
                                    .where(func.lower(func.coalesce(TraderOrder.mode, "")) == run_mode)
                                    .where(TraderOrder.market_id == _pre_gate_market_id)
                                    .where(
                                        func.lower(func.coalesce(TraderOrder.status, "")).in_(
                                            LIVE_ENTRY_BLOCKING_ORDER_STATUSES
                                            + LIVE_AUTHORITY_PENDING_ENTRY_STATUSES
                                        )
                                    )
                                    .where(
                                        func.lower(func.coalesce(TraderOrder.status, "")).in_(
                                            LIVE_ENTRY_BLOCKING_ORDER_STATUSES
                                        )
                                        | (
                                            func.coalesce(TraderOrder.updated_at, TraderOrder.created_at)
                                            >= authority_pending_cutoff
                                        )
                                    )
                                )
                            ).all()
                            order_occupied = any(
                                str(row.status or "").strip().lower() in LIVE_ENTRY_BLOCKING_ORDER_STATUSES
                                or _live_authority_pending_entry_blocks_market(
                                    row.status,
                                    row.payload_json,
                                    row.updated_at or row.created_at,
                                )
                                for row in order_rows
                            )
                            if position_occupied or order_occupied:
                                _db_occupied_market_ids = [_pre_gate_market_id]
                                final_decision = "blocked"
                                final_reason = "Stacking guard: market already occupied (DB verification)"
                                checks_payload = list(checks_payload)
                                checks_payload.append(
                                    {
                                        "check_key": "stacking_guard_db_verification",
                                        "check_label": "Stacking guard",
                                        "passed": False,
                                        "score": None,
                                        "detail": "Market already occupied by an open live order in DB verification.",
                                        "payload": {
                                            "market_id": _pre_gate_market_id,
                                            "occupied_market_ids": sorted(_db_occupied_market_ids),
                                            "allow_averaging": allow_averaging,
                                            "live_single_market_guard": True,
                                        },
                                    }
                                )
                                gate_result["platform_gates"] = list(gate_result["platform_gates"])
                                gate_result["platform_gates"].append(
                                    {
                                        "gate": "stacking_guard_db_verification",
                                        "passed": False,
                                        "reason": "Market already occupied (DB verification)",
                                        "payload": {
                                            "market_id": _pre_gate_market_id,
                                            "occupied_market_ids": sorted(_db_occupied_market_ids),
                                        },
                                    }
                                )
                                logger.warning(
                                    "DB stacking guard blocked order that passed pre-gates "
                                    "trader=%s market=%s occupied_markets=%d",
                                    trader_id,
                                    _pre_gate_market_id,
                                    len(_db_occupied_market_ids),
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
                    # Monotonic marker used to slice decision_to_submit_start_ms
                    # into three sub-spans: decision-ready -> commit-start
                    # (pre-commit DB writes), commit-wait (the actual
                    # _commit_with_retry duration), and commit-done ->
                    # submit-start (session.reset + cursor bookkeeping).
                    _decision_ready_mono = time.monotonic()
                    decision_latency_payload = _compute_signal_latency_payload(
                        runtime_signal,
                        live_context=live_context,
                        measured_at=decision_ready_at,
                    )
                    decision_check_count = len(checks_payload)

                    _enter_stage("decision_write_main")
                    _decision_writes_mono = time.monotonic()
                    # Round 7: isolate create_trader_decision timing so
                    # we can tell whether ps_decision_writes' 3-9s is
                    # dominated by this INSERT vs. the surrounding
                    # experiment-assignment / checks / signal-status
                    # work that also lands inside the same bucket.
                    _ps_dw_create_decision_mono = time.monotonic()
                    decision_row = await create_trader_decision(
                        session,
                        trader_id=trader_id,
                        signal=runtime_signal,
                        strategy_key=resolved_strategy_key,
                        strategy_version=resolved_strategy_version,
                        decision=final_decision,
                        reason=final_reason,
                        score=score,
                        checks_summary={"count": decision_check_count},
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
                        # Skip the per-INSERT flush — all per-signal
                        # writes share one transaction that flushes
                        # at the final commit.  Saves a Postgres
                        # round-trip per signal under contention.
                        flush_on_add=False,
                    )
                    _accumulate("ps_dw_create_decision", _ps_dw_create_decision_mono)
                    decisions_written += 1
                    _ps_dw_experiment_mono = time.monotonic()
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
                    _accumulate("ps_dw_experiment", _ps_dw_experiment_mono)

                    _ps_dw_checks_mono = time.monotonic()
                    # Round 8-A: route decision_checks through the audit
                    # buffer so the N-row INSERT no longer sits inside
                    # the per-signal transaction.  Post-R7 soak showed
                    # ps_dw_checks owning 1.4-4.4s of every cycle — it
                    # was the ONE unbuffered per-check write left, and
                    # `buffer_decision_checks` + `_flush_decision_checks_bulk`
                    # already exist in trader_hot_state for exactly this
                    # purpose.  Deduplicated bulk flush runs in the
                    # background audit task, so the per-signal critical
                    # path drops to a buffer append under `_audit_lock`.
                    await hot_state.buffer_decision_checks(
                        decision_id=decision_row.id,
                        checks=checks_payload,
                    )
                    _accumulate("ps_dw_checks", _ps_dw_checks_mono)
                    # Round 9-B: route freshness trader_event through the
                    # hot_state audit buffer instead of create_trader_event.
                    # Post-R8 soak showed ps_decision_writes = 2-10s/cycle
                    # with EVERY ps_dw_* sub-bucket at 0 — the only
                    # unaccounted-for call inside the decision_writes
                    # window was this `create_trader_event`, and its
                    # per-signal row.add() path sits inside the same
                    # per-signal transaction that everything else is
                    # fighting.  buffer_trader_event appends under the
                    # in-memory `_audit_lock` and the bulk flush runs on
                    # the background audit task — same pattern that
                    # dissolved ps_dw_checks in Round 8-A.  The
                    # `ps_dw_freshness_event` bucket lets us confirm
                    # the gap closes (or surfaces whatever else is
                    # hiding in the window if it doesn't).
                    _ps_dw_freshness_event_mono = time.monotonic()
                    if freshness_status in {"passed", "blocked"} and freshness_source in {"scanner", "crypto"}:
                        freshness_age_ms = safe_float(freshness_payload.get("age_ms"), None)
                        freshness_max_age_ms = safe_float(freshness_payload.get("max_age_ms"), None)
                        freshness_severity = "warn" if freshness_status == "blocked" else "info"
                        freshness_message = (
                            f"Market data freshness {freshness_status}: source={freshness_source or 'unknown'} "
                            f"age_ms={freshness_age_ms if freshness_age_ms is not None else 'unknown'} "
                            f"max={freshness_max_age_ms if freshness_max_age_ms is not None else 'unknown'}"
                        )
                        await hot_state.buffer_trader_event(
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
                        )
                    _accumulate("ps_dw_freshness_event", _ps_dw_freshness_event_mono)

                    if final_decision == "selected":
                        if run_mode == "live":
                            await set_trade_signal_status(
                                session,
                                signal_id=signal_id,
                                status="selected",
                                commit=False,
                            )
                            await record_signal_consumption(
                                session,
                                trader_id=trader_id,
                                signal_id=signal_id,
                                decision_id=decision_row.id,
                                outcome="claiming",
                                reason="Claimed for live submission",
                                payload={
                                    "phase": "pre_submit_claim",
                                    "source_key": signal_source,
                                    "strategy_key": resolved_strategy_key,
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
                            cursor_created_at = _signal_cursor_timestamp(signal)
                            cursor_signal_id = signal_id
                            cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                        _commit_mono = time.monotonic()
                        # R12-C: capture CommitMetrics so we can split
                        # the 3s ``ps_db_commit`` budget into the actual
                        # commit-call time vs retry/rollback overhead,
                        # and tag each commit with its pending row count
                        # so "big transaction" and "lock-wait" cases are
                        # distinguishable in the next soak.
                        _commit_metrics = await _commit_with_retry(session)
                        _commit_done_mono = time.monotonic()
                        _accumulate("ps_db_commit", _commit_mono)
                        if _commit_metrics is not None:
                            _accumulate_value("ps_db_commit_call", _commit_metrics.commit_call_ms)
                            _accumulate_value("ps_db_commit_retries", _commit_metrics.retry_count)
                            _accumulate_value("ps_db_commit_dirty_rows", _commit_metrics.dirty_rows)
                        try:
                            await session.reset()
                        except Exception:
                            pass
                        submit_started_at = utcnow()
                        _submit_start_mono = time.monotonic()
                        # Slice decision_to_submit_start_ms into its three
                        # internal spans so the per-sample breach log can
                        # pinpoint which burned the budget.
                        _decision_to_commit_start_ms = max(
                            0, int((_commit_mono - _decision_ready_mono) * 1000.0)
                        )
                        _commit_wait_ms = max(
                            0, int((_commit_done_mono - _commit_mono) * 1000.0)
                        )
                        _commit_done_to_submit_start_ms = max(
                            0, int((_submit_start_mono - _commit_done_mono) * 1000.0)
                        )
                        _enter_stage("submit_order")
                        _submit_mono = time.monotonic()
                        # NOTE: do NOT wrap this in release_conn(session).
                        # Line 7706 above already calls
                        # ``await session.reset()`` between commit and
                        # submit — the existing pattern documented at
                        # live_market_context.py:1798 ("The trader cycle
                        # calls session.reset() between commit and submit
                        # to release the DB connection during the provider
                        # round trip").  A Phase 1 attempt to add
                        # release_conn here doubled the reset/flush
                        # sequence and produced SQLAlchemy
                        # IllegalStateChangeError (gkpj) on every signal:
                        # 234 trader cycle failures in 26 min in one soak,
                        # 97 in 21 min in another.  submit_order opens its
                        # own AsyncSessionLocal so the orchestrator's
                        # released connection here is sufficient.
                        submit_result = await submit_order(
                            trader_id=trader_id,
                            signal=runtime_signal,
                            decision_id=decision_row.id,
                            strategy_key=resolved_strategy_key,
                            strategy_version=resolved_strategy_version,
                            strategy_params=strategy_params,
                            explicit_strategy_params=explicit_strategy_params,
                            risk_limits=effective_risk_limits,
                            mode=str(control.get("mode", "shadow")),
                            size_usd=size_usd,
                            reason=final_reason,
                        )
                        _accumulate("ps_submit_order", _submit_mono)
                        _submit_session_timing_ms: dict[str, float] = {}
                        # Surface the place_order sub-stage breakdown so
                        # the cycle-slow log names which step (lock_wait
                        # vs create_order vs post_order vs buy_gate) ate
                        # the time.  Without this ps_submit_order is
                        # opaque — production saw 39.6 s with no path
                        # to the offender.
                        try:
                            # ``submit_result`` is either a SessionExecutionResult
                            # (live path) or a tuple-shaped fallback for shadow.
                            # Live path stashes the per-leg place_order
                            # breakdown under payload["submit_breakdowns"].
                            _result_payload = getattr(submit_result, "payload", None)
                            if not isinstance(_result_payload, dict):
                                _result_payload = {}
                            _execution_timing = _result_payload.get("execution_timing_ms") or {}
                            if isinstance(_execution_timing, dict):
                                for _stage_name, _stage_ms in _execution_timing.items():
                                    try:
                                        _ms_val = float(_stage_ms or 0.0)
                                    except (TypeError, ValueError):
                                        continue
                                    if _ms_val <= 0.0:
                                        continue
                                    _timing_key = f"submit_session_{_stage_name}"
                                    stage_timings_ms[_timing_key] = round(_ms_val, 1)
                                    _submit_session_timing_ms[_stage_name] = round(_ms_val, 1)
                            _wave_breakdowns = _result_payload.get("submit_breakdowns") or []
                            if not isinstance(_wave_breakdowns, list):
                                _wave_breakdowns = []
                            # Take the slowest leg (by total_ms) — that's
                            # the one that owned the cycle's submit time.
                            if _wave_breakdowns:
                                _slowest = max(
                                    (b for b in _wave_breakdowns if isinstance(b, dict)),
                                    key=lambda b: float(b.get("total_ms", 0.0) or 0.0),
                                    default=None,
                                )
                                if _slowest:
                                    for _stage_name, _stage_ms in _slowest.items():
                                        if _stage_name == "total_ms" or _stage_name == "attempts":
                                            continue
                                        try:
                                            _ms_val = float(_stage_ms or 0.0)
                                        except (TypeError, ValueError):
                                            continue
                                        if _ms_val <= 0.0:
                                            continue
                                        stage_timings_ms[f"submit_{_stage_name}"] = round(_ms_val, 1)
                                    # Also record attempt count + total
                                    # so we can spot retry storms.
                                    if "attempts" in _slowest:
                                        try:
                                            stage_timings_ms["submit_attempts"] = float(_slowest["attempts"])
                                        except (TypeError, ValueError):
                                            pass
                        except Exception:
                            # Diagnostic surfacing must never break the loop.
                            pass
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
                        if _submit_session_timing_ms:
                            submit_latency_payload["submit_session_timing_ms"] = dict(_submit_session_timing_ms)
                        latency_sample = _build_execution_latency_sample(
                            runtime_signal,
                            wake_started_at=signal_wake_started_at,
                            context_ready_at=context_ready_at,
                            decision_ready_at=decision_ready_at,
                            submit_started_at=submit_started_at,
                            submit_completed_at=submit_completed_at,
                        )
                        # Attach the decision->submit-start internal sub-spans
                        # (pre-commit DB writes, commit wait, session reset +
                        # cursor bookkeeping). These are NOT aggregated in
                        # execution_latency_metrics._STAGE_KEYS but travel
                        # with the sample so the per-sample breach warning
                        # can log them alongside the outer stages.
                        latency_sample["decision_ready_to_commit_start_ms"] = _decision_to_commit_start_ms
                        latency_sample["commit_wait_ms"] = _commit_wait_ms
                        latency_sample["commit_done_to_submit_start_ms"] = _commit_done_to_submit_start_ms
                        await execution_latency_metrics.record(
                            trader_id=trader_id,
                            source=signal_source,
                            strategy_key=resolved_strategy_key,
                            payload=latency_sample,
                        )
                        # Surface per-sample SLA breaches with the full
                        # sub-stage breakdown so the soak log pinpoints
                        # which stage (queue wake, live-context build,
                        # strategy/risk eval, DB commit block, or the
                        # CLOB submit round-trip) spent the budget.
                        _maybe_log_per_sample_latency_breach(
                            trader_id=trader_id,
                            source=signal_source,
                            strategy_key=resolved_strategy_key,
                            signal_id=signal_id,
                            sample=latency_sample,
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
                                "submit_round_trip_ms": latency_sample.get("submit_round_trip_ms"),
                                "emit_to_submit_start_ms": latency_sample.get("emit_to_submit_start_ms"),
                            }
                        )
                        await hot_state.buffer_trader_event(
                            trader_id=trader_id,
                            event_type="execution_latency",
                            severity="info",
                            source=signal_source,
                            message=(
                                "Hot-path execution latency "
                                f"armed_to_release_ms={submit_latency_payload.get('armed_to_ws_release_ms', 'unknown')} "
                                f"release_to_submit_ms={submit_latency_payload.get('ws_release_to_submit_start_ms', 'unknown')} "
                                f"submit_round_trip_ms={submit_latency_payload.get('submit_round_trip_ms', 'unknown')}"
                            ),
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal_id,
                                "strategy_key": resolved_strategy_key,
                                "latency": submit_latency_payload,
                            },
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
                                execution_check = _execution_outcome_decision_check(
                                    status=normalized_order_status,
                                    reason=final_reason,
                                    error_message=execution_error_message,
                                )
                                if execution_check is not None:
                                    await create_trader_decision_checks(
                                        session,
                                        decision_id=decision_row.id,
                                        checks=[execution_check],
                                        commit=False,
                                    )
                                await update_trader_decision(
                                    session,
                                    decision_id=decision_row.id,
                                    decision="skipped",
                                    reason=final_reason,
                                    payload_patch={
                                        "execution_status": normalized_order_status,
                                        "execution_skip_reason": final_reason,
                                    },
                                    checks_summary_patch={"count": decision_check_count + 1},
                                    commit=False,
                                )
                            elif normalized_order_status in {"failed", "cancelled", "expired"}:
                                final_decision = "failed"
                                if execution_error_message:
                                    final_reason = execution_error_message
                                execution_check = _execution_outcome_decision_check(
                                    status=normalized_order_status,
                                    reason=final_reason,
                                    error_message=execution_error_message,
                                )
                                if execution_check is not None:
                                    await create_trader_decision_checks(
                                        session,
                                        decision_id=decision_row.id,
                                        checks=[execution_check],
                                        commit=False,
                                    )
                                await update_trader_decision(
                                    session,
                                    decision_id=decision_row.id,
                                    decision="failed",
                                    reason=final_reason,
                                    payload_patch={
                                        "execution_status": normalized_order_status,
                                        "execution_error": execution_error_message,
                                    },
                                    checks_summary_patch={"count": decision_check_count + 1},
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
                                    occupied_market_ids.add(opened_market_id)
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
                                execution_check = _execution_outcome_decision_check(
                                    status=normalized_order_status,
                                    reason=final_reason,
                                    execution_session_id=execution_session_id,
                                    error_message=execution_error_message,
                                )
                                if execution_check is not None:
                                    await create_trader_decision_checks(
                                        session,
                                        decision_id=decision_row.id,
                                        checks=[execution_check],
                                        commit=False,
                                    )
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
                                    checks_summary_patch={"count": decision_check_count + 1},
                                    commit=False,
                                )
                            elif normalized_order_status in {"failed", "cancelled", "expired"}:
                                final_decision = "failed"
                                if execution_error_message:
                                    final_reason = execution_error_message
                                execution_check = _execution_outcome_decision_check(
                                    status=normalized_order_status,
                                    reason=final_reason,
                                    execution_session_id=execution_session_id,
                                    error_message=execution_error_message,
                                )
                                if execution_check is not None:
                                    await create_trader_decision_checks(
                                        session,
                                        decision_id=decision_row.id,
                                        checks=[execution_check],
                                        commit=False,
                                    )
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
                                    checks_summary_patch={"count": decision_check_count + 1},
                                    commit=False,
                                )

                            # Record per-market failed-entry cooldown after
                            # execution failure so we don't hammer the same
                            # market after venue cancels / no-fills.
                            if final_decision == "failed" and _pre_gate_market_id:
                                _nf_key = (trader_id, _pre_gate_market_id)
                                _nf_count = _nofill_counts.get(_nf_key, 0) + 1
                                _nofill_counts[_nf_key] = _nf_count
                                if normalized_order_status in {"cancelled", "expired"}:
                                    _cd = _NOFILL_CANCELLED_COOLDOWN_SECONDS
                                else:
                                    _cd = (
                                        _NOFILL_EXTENDED_COOLDOWN_SECONDS
                                        if _nf_count >= _NOFILL_MAX_CONSECUTIVE
                                        else _NOFILL_COOLDOWN_SECONDS
                                    )
                                _nofill_cooldowns[_nf_key] = _time_mod.monotonic() + _cd
                                logger.info(
                                    "Failed-entry cooldown set for trader=%s market=%s status=%s count=%d cooldown=%ds",
                                    trader_id,
                                    _pre_gate_market_id,
                                    normalized_order_status,
                                    _nf_count,
                                    int(_cd),
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
                                # Clear no-fill cooldown on successful fill
                                _nf_success_key = (trader_id, _pre_gate_market_id)
                                _nofill_cooldowns.pop(_nf_success_key, None)
                                _nofill_counts.pop(_nf_success_key, None)

                                for _co in getattr(submit_result, "created_orders", None) or []:
                                    hot_state.upsert_active_order(
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
                                    occupied_market_ids.add(opened_market_id)
                                intra_cycle_committed_usd += size_usd
                        if final_decision == "failed" and order_status in {"failed", "cancelled", "expired"}:
                            await hot_state.buffer_trader_event(
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
                            )
                            # Mark the signal as failed so it won't be retried
                            # endlessly.  Without this, FAK no-fill errors loop
                            # on the same signal every cycle, blocking all other
                            # signals from being evaluated.
                            await set_trade_signal_status(
                                session,
                                signal_id=signal_id,
                                status="failed",
                                commit=False,
                            )
                    else:
                        status_update: str | None = None
                        if final_decision == "failed":
                            status_update = "failed"
                        elif final_decision == "skipped":
                            status_update = "skipped"
                        elif isinstance(final_reason, str) and final_reason.startswith("Signal stale:"):
                            status_update = "skipped"
                        if status_update is not None:
                            # Off-path: buffered into trader_hot_state's
                            # audit-flush loop instead of await-on-DB.
                            # Hot-path drops from ~1s/UPDATE under
                            # contention to microseconds (lock-protected
                            # list append).
                            _ps_dw_set_status_mono = time.monotonic()
                            await hot_state.buffer_signal_status(
                                signal_id=signal_id,
                                status=status_update,
                            )
                            _accumulate("ps_dw_set_status", _ps_dw_set_status_mono)
                        # If the block reason is signal staleness, also mark
                        # the signal as expired so the scanner's continuous
                        # re-publishes cannot reactivate it through
                        # list_unconsumed_trade_signals' updated_at>consumed_at
                        # clause.  Without this the orchestrator re-evaluates
                        # the same stale signal every scanner tick and spams
                        # the decision log with "Signal stale" blocks.
                        if isinstance(final_reason, str) and final_reason.startswith("Signal stale:"):
                            try:
                                _ps_dw_expire_mono = time.monotonic()
                                await session.execute(
                                    sa_update(TradeSignal)
                                    .where(TradeSignal.id == signal_id)
                                    .values(expires_at=utcnow().replace(tzinfo=None))
                                )
                                _accumulate("ps_dw_expire_stale", _ps_dw_expire_mono)
                            except Exception as _exp_exc:
                                logger.debug(
                                    "Failed to expire stale signal",
                                    signal_id=signal_id,
                                    exc_info=_exp_exc,
                                )

                    # Buffered: append-only consumption row.  Was
                    # 250-2641ms per call under contention; now sub-ms.
                    _ps_dw_record_mono = time.monotonic()
                    await hot_state.buffer_signal_consumption(
                        trader_id=trader_id,
                        signal_id=signal_id,
                        decision_id=decision_row.id,
                        outcome=order_status or final_decision,
                        reason=final_reason,
                    )
                    _accumulate("ps_dw_record_consumption", _ps_dw_record_mono)

                    # Buffered: append-only trader_event row.
                    _ps_dw_event_mono = time.monotonic()
                    await hot_state.buffer_trader_event(
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
                    )
                    _accumulate("ps_dw_create_event", _ps_dw_event_mono)
                    # Buffered: cursor UPSERT.  Was 516-4782ms per call
                    # under pool contention; now sub-ms.  The flush
                    # loop deduplicates runs with the same key so a
                    # burst of cursor advances collapses to a single
                    # UPSERT — strict net win.
                    _ps_dw_cursor_mono = time.monotonic()
                    await hot_state.buffer_signal_cursor(
                        trader_id=trader_id,
                        last_signal_created_at=_signal_cursor_timestamp(signal),
                        last_signal_id=signal_id,
                        last_runtime_sequence=safe_int(
                            getattr(signal, "runtime_sequence", None), None
                        ),
                    )
                    _accumulate("ps_dw_upsert_cursor", _ps_dw_cursor_mono)
                    # Close the decision-writes bucket: from the
                    # create_trader_decision call through the cursor
                    # upsert above.  Captures every per-signal INSERT
                    # except the commit itself (which has its own
                    # bucket below).
                    try:
                        _accumulate("ps_decision_writes", _decision_writes_mono)
                    except NameError:
                        # Iteration took an early branch that bypassed
                        # the create_trader_decision marker — no bucket
                        # to close.  Safe to ignore.
                        pass
                    _commit_mono = time.monotonic()
                    # R12-C: same metrics split as the submit-path commit
                    # above — see comment there for rationale.
                    _commit_metrics = await _commit_with_retry(session)
                    _accumulate("ps_db_commit", _commit_mono)
                    if _commit_metrics is not None:
                        _accumulate_value("ps_db_commit_call", _commit_metrics.commit_call_ms)
                        _accumulate_value("ps_db_commit_retries", _commit_metrics.retry_count)
                        _accumulate_value("ps_db_commit_dirty_rows", _commit_metrics.dirty_rows)
                    try:
                        await session.reset()
                    except Exception:
                        pass
                    cursor_created_at = _signal_cursor_timestamp(signal)
                    cursor_signal_id = signal_id
                    cursor_runtime_sequence = _signal_runtime_sequence(signal) or cursor_runtime_sequence
                    processed_signals += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # SOAK-2026-05-18: accumulate ps_decision_writes on
                    # the failure path too.  Without this, any exception
                    # between line 7482 (the _decision_writes_mono start)
                    # and the success-path accumulate at 8268 leaves the
                    # bucket at zero — the time spent appears as
                    # ps_unaccounted instead, which made the recent soak's
                    # 11s "unaccounted" look like a mystery latency sink
                    # when it was just gkpj-failed cycles' decision-write
                    # work not being attributed.  NameError guard handles
                    # iterations that took an early branch before setting
                    # _decision_writes_mono.
                    try:
                        _accumulate("ps_decision_writes", _decision_writes_mono)
                    except NameError:
                        pass
                    try:
                        await session.rollback()
                    except Exception:
                        logger.warning(
                            "Trader %s session rollback failed after signal %s error; session may be invalidated",
                            trader_id, signal_id,
                        )
                    # Transient DB errors (deadlock, serialization failure):
                    # skip this signal without recording a permanent failure
                    # or advancing the cursor.  The signal stays pending and
                    # will be retried on the next orchestrator cycle once the
                    # conflicting transaction has finished.
                    if _is_transient_db_error(exc):
                        logger.warning(
                            "Trader %s transient DB error on signal %s, will retry next cycle: %s",
                            trader_id,
                            signal_id,
                            exc,
                        )
                        continue
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
                        _commit_mono = time.monotonic()
                        # R12-C: error-recovery path commit — instrument
                        # the same way so we can tell whether the
                        # decision_error write itself is slow (e.g. if
                        # it's contending for a lock on trade_signals
                        # alongside the main path).
                        _commit_metrics = await _commit_with_retry(session)
                        _accumulate("ps_db_commit", _commit_mono)
                        if _commit_metrics is not None:
                            _accumulate_value("ps_db_commit_call", _commit_metrics.commit_call_ms)
                            _accumulate_value("ps_db_commit_retries", _commit_metrics.retry_count)
                            _accumulate_value("ps_db_commit_dirty_rows", _commit_metrics.dirty_rows)
                        try:
                            await session.reset()
                        except Exception:
                            pass
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

            # Capture the final per-signal duration once the for-loop exits.
            if _last_per_signal_started is not None:
                _accumulators["per_signal_total"] = (
                    _accumulators.get("per_signal_total", 0.0)
                    + (time.monotonic() - _last_per_signal_started) * 1000.0
                )
                _last_per_signal_started = None
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
                    elif expired_signals > 0:
                        await _emit_cycle_heartbeat_if_due(
                            session,
                            trader_id=trader_id,
                            message="Idle cycle: stale crypto signals expired before evaluation.",
                            payload={
                                "processed_signals": processed_signals,
                                "decisions_written": decisions_written,
                                "orders_written": orders_written,
                                "expired_signals": expired_signals,
                                "expired_by_reason": expired_by_reason,
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
                        _no_signal_payload: dict[str, Any] = {
                            "processed_signals": processed_signals,
                            "decisions_written": decisions_written,
                            "orders_written": orders_written,
                        }
                        _no_signal_message = "Idle cycle: no pending signals."
                        try:
                            _cfd = _get_strategy_filter_diagnostics(trader=trader)
                            if _cfd:
                                _no_signal_payload["filter_diagnostics"] = _cfd
                                _no_signal_message = _format_filter_diagnostics_message(_cfd)
                        except Exception:
                            pass
                        await _emit_cycle_heartbeat_if_due(
                            session,
                            trader_id=trader_id,
                            message=_no_signal_message,
                            payload=_no_signal_payload,
                        )
            _sl_post_loop_started = time.monotonic()
            if stream_trigger_mode and prefetched_signals:
                _slp_deferred_republish_started = time.monotonic()
                deferred_signal_ids_by_source: dict[str, list[str]] = {}
                deferred_signal_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] = {}
                deferred_seen_ids_by_source: dict[str, set[str]] = {}
                deferred_emitted_at_iso = utcnow().isoformat()
                for deferred_signal in prefetched_signals:
                    deferred_source = normalize_source_key(getattr(deferred_signal, "source", "")) or "__all__"
                    deferred_signal_id = str(getattr(deferred_signal, "id", "") or "").strip()
                    if not deferred_signal_id:
                        continue
                    deferred_signal_ids_by_source.setdefault(deferred_source, [])
                    deferred_signal_snapshots_by_source.setdefault(deferred_source, {})
                    deferred_seen_ids_by_source.setdefault(deferred_source, set())
                    if deferred_signal_id in deferred_seen_ids_by_source[deferred_source]:
                        continue
                    deferred_seen_ids_by_source[deferred_source].add(deferred_signal_id)
                    deferred_signal_ids_by_source[deferred_source].append(deferred_signal_id)
                    deferred_signal_snapshots_by_source[deferred_source][deferred_signal_id] = (
                        _runtime_signal_trigger_snapshot(
                            deferred_signal,
                            emitted_at_iso=deferred_emitted_at_iso,
                        )
                    )
                for deferred_source, deferred_signal_ids in deferred_signal_ids_by_source.items():
                    await publish_signal_batch(
                        event_type="upsert_update",
                        source=None if deferred_source == "__all__" else deferred_source,
                        signal_ids=deferred_signal_ids,
                        trigger="runtime_signal_overflow",
                        reason="trader_orchestrator_runtime_trigger_batch_deferred",
                        emitted_at=deferred_emitted_at_iso,
                        signal_snapshots=deferred_signal_snapshots_by_source.get(deferred_source) or None,
                    )
                prefetched_signals = None
                _accumulate("slp_deferred_republish", _slp_deferred_republish_started)
            _slp_heartbeat_started = time.monotonic()
            # R10-E: fire-and-forget to keep the cycle's hot path off
            # the heartbeat commit's pool wait.  The outer session is
            # still used for in-cycle writes above this line; the
            # heartbeat UPDATE moves to a detached task with its own
            # short session.
            schedule_trader_cycle_heartbeat(trader_id)
            _accumulate("slp_heartbeat", _slp_heartbeat_started)
            _accumulate("sl_post_loop", _sl_post_loop_started)
            if defer_signal_processing or stream_trigger_mode:
                break

    # Clean up the prefetched live-context task if no signal ever
    # consumed it (all prefiltered, empty signals batch, etc.).  The
    # task is cheap (pure-memory build) so cancel + don't await — we
    # just don't want it lingering as a leaked task.
    if live_contexts_task is not None and not live_contexts_loaded:
        live_contexts_task.cancel()
        live_contexts_task = None
        live_contexts_loaded = True

    _checkpoint("signal_loop")
    # Surface cumulative buckets (e.g. live_context) alongside checkpoints
    # so the warning shows where inside signal_loop time was spent.
    for _name, _ms in _accumulators.items():
        stage_timings_ms[_name] = round(_ms, 1)
    # Compute the residual gap inside per_signal_total that isn't
    # covered by any of the ``ps_*`` buckets.  When this is large,
    # there's a slow op in the per-signal pipeline that hasn't been
    # instrumented yet — the operator should add a new ``ps_*`` bucket
    # around the suspect call (decision row INSERT, signal-consumption
    # ledger write, audit-log write, etc.).  Negative values indicate
    # double-counting between buckets and the iteration timer.
    _per_signal_total = float(_accumulators.get("per_signal_total", 0.0) or 0.0)
    if _per_signal_total > 0:
        _instrumented = sum(
            float(_accumulators.get(name, 0.0) or 0.0)
            for name in (
                "ps_strategy_evaluate",
                "ps_copy_inventory_fetch",
                "ps_submit_order",
                "ps_db_commit",
                "ps_decision_writes",
                # Pre-strategy buckets (added when ps_unaccounted ate
                # 55-66s with zero ps_* coverage in production soak).
                "ps_signal_persist",
                "ps_skip_writes",
                "ps_strict_ws_writes",
                "ps_prefilter_writes",
                "ps_live_context_await",
                "ps_decision_skip_writes",
                "ps_traders_scope_build",
                "ps_copy_exposure_lookup",
                "ps_risk_eval_setup",
            )
        )
        stage_timings_ms["ps_unaccounted"] = round(
            max(0.0, _per_signal_total - _instrumented), 1
        )
    cycle_duration_s = time.monotonic() - cycle_started_mono
    if cycle_duration_s > 10.0:
        logger.warning(
            "Trader cycle slow",
            trader_id=trader_id,
            mode=run_mode,
            duration_s=round(cycle_duration_s, 2),
            processed_signals=processed_signals,
            decisions_written=decisions_written,
            orders_written=orders_written,
            stage_timings_ms=stage_timings_ms,
        )
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
            "non_shadow_cancelled": 0,
        }

    per_trader_mode: dict[tuple[str, str], int] = {}
    for row in rows:
        trader_id = str(row.trader_id or "").strip()
        if not trader_id:
            continue
        mode_key = str(row.mode_key or "").strip().lower()
        if mode_key == "" or mode_key not in _CANONICAL_TRADER_MODES:
            mode_key = "other"
        per_trader_mode[(trader_id, mode_key)] = int(row.count or 0)

    shadow_closed = 0
    non_shadow_cancelled = 0
    for trader_id, mode_key in per_trader_mode:
        if mode_key == "shadow":
            result = await reconcile_shadow_positions(
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
            "non_shadow_cancelled": non_shadow_cancelled,
        },
        commit=True,
    )

    return {
        "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
        "rows_seen": len(per_trader_mode),
        "shadow_closed": shadow_closed,
        "non_shadow_cancelled": non_shadow_cancelled,
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
    # Per-trader cycle timeout override: check risk_limits then source strategy_params.
    per_trader_timeout = safe_float(
        (dict(trader.get("risk_limits") or {})).get("trader_cycle_timeout_seconds"),
        0.0,
    )
    if per_trader_timeout <= 0:
        source_configs = trader.get("source_configs")
        if isinstance(source_configs, dict):
            for _sc in source_configs.values():
                _sp = (dict(_sc) if isinstance(_sc, dict) else {}).get("strategy_params")
                if isinstance(_sp, dict):
                    per_trader_timeout = safe_float(_sp.get("trader_cycle_timeout_seconds"), 0.0)
                    if per_trader_timeout > 0:
                        break
    effective_timeout = per_trader_timeout if per_trader_timeout > 0 else timeout_seconds
    requested_timeout = max(1.0, min(180.0, float(effective_timeout)))
    timeout = requested_timeout
    run_mode = _canonical_trader_mode(control.get("mode"), default="shadow")
    if process_signals and run_mode == "live":
        timeout = max(_MIN_LIVE_PROCESS_SIGNAL_CYCLE_TIMEOUT_SECONDS, timeout)
    trader_id = str(trader.get("id") or "")
    has_runtime_trigger = bool(trigger_signal_ids_by_source) or bool(trigger_signal_snapshots_by_source)
    is_signal_processing = bool(process_signals and has_runtime_trigger)
    # Hard-kill any task stuck so long it's leaking DB connections.  We
    # only force-kill the OLDEST task — concurrent siblings stay alive.
    oldest = _oldest_running_task(trader_id)
    if oldest is not None:
        oldest_started_at = _inflight_trader_cycle_start.get(oldest, 0.0)
        stuck_seconds = time.monotonic() - oldest_started_at if oldest_started_at else 0.0
        if oldest_started_at and stuck_seconds > _STUCK_TASK_HARD_KILL_SECONDS:
            logger.error(
                "Force-killing stuck trader cycle task trader=%s stuck_seconds=%.0f",
                trader_id,
                stuck_seconds,
            )
            oldest.cancel()
            _clear_inflight_trader_cycle_task(trader_id, oldest)
            _skip_log_last_at.pop(trader_id, None)
            _skip_log_suppressed.pop(trader_id, None)

    # Preemption: if a maintenance-only cycle is running and a runtime
    # signal trigger arrives, preempt the maintenance task so the signal
    # gets prompt service. Signal cycles for the same trader do not
    # preempt each other; the concurrency gate below coalesces them.
    if is_signal_processing:
        for sibling in list(_running_inflight_tasks(trader_id)):
            if _inflight_trader_cycle_process_signals.get(sibling, False):
                continue
            sibling.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(sibling),
                    timeout=min(1.0, _TRADER_TIMEOUT_CANCEL_GRACE_SECONDS),
                )
            except asyncio.TimeoutError:
                _abandoned_trader_cycle_tasks.add(sibling)
                sibling.add_done_callback(_discard_abandoned_trader_cycle)
                _clear_inflight_trader_cycle_task(trader_id, sibling)
                logger.warning(
                    "Preempted maintenance-only trader cycle after cancel-grace expiry trader=%s",
                    trader_id,
                )
            except asyncio.CancelledError:
                _clear_inflight_trader_cycle_task(trader_id, sibling)
            except Exception as exc:
                _clear_inflight_trader_cycle_task(trader_id, sibling)
                logger.warning(
                    "Maintenance-only trader cycle raised during runtime-trigger preemption trader=%s",
                    trader_id,
                    exc_info=exc,
                )
            else:
                _clear_inflight_trader_cycle_task(trader_id, sibling)

    # Concurrency gate.  Runtime triggers are coalesced while any cycle
    # is running for the trader, then launched from the done callback.
    # Maintenance-only cycles stay strictly serial — they're not
    # latency-sensitive and might do book-keeping that races with itself.
    running_signal_cycles = _count_running_signal_cycles(trader_id)
    if is_signal_processing:
        at_capacity = running_signal_cycles >= _RUNTIME_TRIGGER_MAX_CONCURRENT_PER_TRADER
    else:
        at_capacity = bool(_running_inflight_tasks(trader_id))

    if at_capacity:
        # Same queueing/skip behaviour as before — emit one rate-limited
        # warning per trader and let the next cycle's done-callback
        # drain anything we deferred.
        oldest_started_at = _inflight_trader_cycle_start.get(oldest, 0.0) if oldest is not None else 0.0
        stuck_seconds = time.monotonic() - oldest_started_at if oldest_started_at else 0.0
        now_mono = time.monotonic()
        last_log = _skip_log_last_at.get(trader_id, 0.0)
        warning_eligible = stuck_seconds >= _OVERLAP_WARNING_MIN_STUCK_SECONDS
        if warning_eligible and now_mono - last_log >= _SKIP_LOG_INTERVAL_SECONDS:
            suppressed = _skip_log_suppressed.get(trader_id, 0)
            suffix = f" (suppressed {suppressed} duplicates)" if suppressed else ""
            if is_signal_processing:
                _queue_pending_runtime_cycle(
                    trader=trader,
                    control=control,
                    process_signals=process_signals,
                    trigger_signal_ids_by_source=trigger_signal_ids_by_source,
                    trigger_signal_snapshots_by_source=trigger_signal_snapshots_by_source,
                    timeout_seconds=requested_timeout,
                )
                logger.warning(
                    "Queued pending runtime trigger because trader is at signal-cycle concurrency cap trader=%s running=%d stuck=%.0fs%s",
                    trader_id,
                    running_signal_cycles,
                    stuck_seconds,
                    suffix,
                )
            else:
                logger.warning(
                    "Trader cycle skipped because prior run is still finishing for trader=%s process_signals=%s stuck=%.0fs%s",
                    trader_id,
                    process_signals,
                    stuck_seconds,
                    suffix,
                )
            _skip_log_last_at[trader_id] = now_mono
            _skip_log_suppressed[trader_id] = 0
        else:
            if is_signal_processing:
                _queue_pending_runtime_cycle(
                    trader=trader,
                    control=control,
                    process_signals=process_signals,
                    trigger_signal_ids_by_source=trigger_signal_ids_by_source,
                    trigger_signal_snapshots_by_source=trigger_signal_snapshots_by_source,
                    timeout_seconds=requested_timeout,
                )
            if warning_eligible:
                _skip_log_suppressed[trader_id] = _skip_log_suppressed.get(trader_id, 0) + 1
        return 0, 0, 0

    task = asyncio.create_task(
        _run_trader_once(
            trader,
            control,
            process_signals=process_signals,
            trigger_signal_ids_by_source=trigger_signal_ids_by_source,
            trigger_signal_snapshots_by_source=trigger_signal_snapshots_by_source,
            cycle_timeout_seconds=timeout,
        ),
        name=f"trader-cycle-{trader_id or 'unknown'}",
    )
    if trader_id:
        _inflight_trader_cycle_tasks.setdefault(trader_id, set()).add(task)
        _inflight_trader_cycle_start[task] = time.monotonic()
        _inflight_trader_cycle_process_signals[task] = bool(process_signals)
        task.add_done_callback(
            lambda done_task, current_trader_id=trader_id: _handle_trader_cycle_task_done(
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
            raise asyncio.TimeoutError()
        logger.warning(
            "Trader cycle did not finish within %ss cancel-grace; leaving inflight task running trader=%s process_signals=%s",
            _TRADER_TIMEOUT_CANCEL_GRACE_SECONDS,
            trader_id,
            process_signals,
        )
        _abandoned_trader_cycle_tasks.add(task)
        task.add_done_callback(_discard_abandoned_trader_cycle)
        return 0, 0, 0
    except asyncio.TimeoutError:
        logger.warning(
            "Trader cycle timed out for trader=%s process_signals=%s timeout=%.1fs",
            trader_id,
            process_signals,
            timeout,
        )
        if trader_id:
            try:
                await hot_state.buffer_trader_event(
                    trader_id=trader_id,
                    event_type="cycle_timeout",
                    severity="warn",
                    source="worker",
                    message=f"Trader cycle timed out after {timeout:.1f}s",
                    payload={
                        "process_signals": bool(process_signals),
                        "timeout_seconds": timeout,
                    },
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
                _clear_inflight_trader_cycle_task(trader_id, task)
                logger.warning(
                    "Trader cycle parent cancelled during cleanup; abandoned and cleared inflight lock trader=%s",
                    trader_id,
                )
        raise
    except (OperationalError, InterfaceError, DBAPIError) as exc:
        # InterfaceError ("connection is closed") fires when postgres kills
        # an idle-in-transaction session via idle_in_transaction_session_timeout.
        # Pre-2026-05-20 these fell through to the generic Exception clause and
        # logged as ERROR — they're transient, the next cycle just retries.
        # DBAPIError is the common base so we catch the asyncpg + psycopg2
        # variants of "connection closed" uniformly.
        if not _is_retryable_db_error(exc):
            raise
        logger.warning(
            "Trader cycle skipped due to transient DB error for trader=%s process_signals=%s exc_type=%s",
            trader_id,
            process_signals,
            type(exc).__name__,
        )
        return 0, 0, 0
    except Exception as exc:
        logger.exception(
            "Trader cycle raised unexpected exception for trader=%s process_signals=%s: %s",
            trader_id,
            process_signals,
            exc,
        )
        return 0, 0, 0


async def _build_runtime_trigger_specs(
    *,
    lane_key: str,
    cycle_trigger: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
    control: dict[str, Any] | None = None
    traders: list[dict[str, Any]] = []
    mode = "shadow"
    async with AsyncSessionLocal() as session:
        control = await read_orchestrator_control(session, read_only=True)
        if (
            not bool(control.get("is_enabled", False))
            or bool(control.get("is_paused", True))
            or bool(control.get("kill_switch", False))
        ):
            return control, [], 3.0
        mode = _canonical_trader_mode(control.get("mode"), default="shadow")
        if mode == "live":
            app_settings = await session.get(AppSettings, "default")
            if not _is_live_credentials_configured(app_settings):
                return control, [], 3.0
        traders = [
            trader
            for trader in await list_traders(
                session,
                mode=mode if mode in {"shadow", "live"} else None,
            )
            if _trader_matches_lane(trader, lane_key) and not _is_fast_tier_trader(trader)
        ]
    if mode == "live" and not live_execution_service.is_ready():
        # Non-blocking readiness check — NEVER await ensure_initialized() on the
        # orchestrator hot path. ensure_initialized() runs the full
        # live_execution.initialize() under init_lock; when a provider flakes
        # (CLOB read timeout / Polygon RPC evicted) that init is slow or fails,
        # so awaiting it here held the lock and stalled the cycle for minutes —
        # the ingest-lag / stale-fill regression. The scheduled-cycle gate was
        # already switched to is_ready() (4c43e5a4); this is its runtime-trigger
        # twin, which the A.2 cross-plane signal path made hot. Background init
        # (host._initialize_live_execution_background) keeps retrying; the cycle
        # resumes the instant live execution is ready.
        return control, [], 3.0
    control_settings = dict((control or {}).get("settings") or {})
    global_runtime_settings = dict(control_settings.get("global_runtime") or {})
    # Runtime triggers use a dedicated (shorter) timeout — these cycles are
    # lightweight (no maintenance, no signal scanning) and must be fast.
    # Falls back to _RUNTIME_TRIGGER_DEFAULT_CYCLE_TIMEOUT_SECONDS (10s).
    configured_rt_timeout = safe_float(
        global_runtime_settings.get("runtime_trigger_cycle_timeout_seconds"),
        0.0,
    )
    if configured_rt_timeout <= 0:
        configured_rt_timeout = _RUNTIME_TRIGGER_DEFAULT_CYCLE_TIMEOUT_SECONDS
    trader_cycle_timeout_seconds = float(max(3.0, min(60.0, configured_rt_timeout)))
    specs: list[dict[str, Any]] = []
    for trader in traders:
        if not bool(trader.get("is_enabled", True)) or bool(trader.get("is_paused", False)):
            continue
        trader_mode = _canonical_trader_mode(trader.get("mode"), default="shadow")
        if trader_mode != mode:
            continue
        if not _runtime_trigger_matches_trader(trader, cycle_trigger):
            continue
        trigger_signal_ids_by_source = _trigger_signal_ids_for_trader(trader, cycle_trigger)
        if (
            isinstance(cycle_trigger, dict)
            and str(cycle_trigger.get("event_type") or "").strip().lower() == "runtime_signal_batch"
            and trigger_signal_ids_by_source is None
        ):
            continue
        specs.append(
            {
                "trader": trader,
                "process_signals": True,
                "trigger_signal_ids_by_source": trigger_signal_ids_by_source,
                "trigger_signal_snapshots_by_source": _trigger_signal_snapshots_for_trader(trader, cycle_trigger),
            }
        )
    return control, specs, trader_cycle_timeout_seconds


async def run_runtime_trigger_loop(*, lane: str = _LANE_GENERAL) -> None:
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    stream_group = _stream_group_name_for_lane(lane_key)
    stream_consumer_name = f"{lane_key}-runtime-{os.getpid()}-{utcnow().strftime('%Y%m%d%H%M%S%f')}"
    stream_claim_cursor = "0-0"
    stream_last_claim_run_at: datetime | None = None
    logger.info("Starting trader orchestrator runtime trigger loop", lane=lane_key)
    while True:
        cycle_trigger, stream_claim_cursor, stream_last_claim_run_at = await _wait_for_runtime_trigger(
            None,
            60.0,
            consumer=stream_consumer_name,
            group=stream_group,
            claim_cursor=stream_claim_cursor,
            last_claim_run_at=stream_last_claim_run_at,
        )
        runtime_triggered_cycle = bool(
            isinstance(cycle_trigger, dict)
            and str(cycle_trigger.get("event_type") or "").strip().lower() == "runtime_signal_batch"
        )
        if not runtime_triggered_cycle:
            continue
        try:
            control, specs, trader_cycle_timeout_seconds = await _build_runtime_trigger_specs(
                lane_key=lane_key,
                cycle_trigger=cycle_trigger,
            )
            if not control or not specs:
                continue
            for spec in specs:
                trader_id = str(spec["trader"].get("id") or "")
                task = asyncio.create_task(
                    _run_trader_once_with_timeout(
                        spec["trader"],
                        control,
                        process_signals=bool(spec["process_signals"]),
                        trigger_signal_ids_by_source=spec["trigger_signal_ids_by_source"],
                        trigger_signal_snapshots_by_source=spec["trigger_signal_snapshots_by_source"],
                        timeout_seconds=trader_cycle_timeout_seconds,
                    ),
                    name=f"trader-runtime-dispatch-{trader_id or 'unknown'}",
                )
                task.add_done_callback(
                    lambda done_task, current_trader_id=trader_id: _handle_runtime_trigger_dispatch_task_done(
                        current_trader_id,
                        done_task,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Runtime trigger dispatch failed",
                lane=lane_key,
                exc_info=exc,
            )


async def run_worker_loop(
    *,
    lane: str = _LANE_GENERAL,
    write_snapshot: bool = True,
    process_runtime_triggers: bool = True,
    process_scheduled_signals: bool = True,
) -> None:
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
    _stale_signal_expiry_consecutive_failures = 0
    _stale_signal_expiry_backoff_until: datetime | None = None

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
            if process_runtime_triggers and not runtime_triggered_cycle:
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
                    try:
                        async with AsyncSessionLocal() as _lock_session:
                            await _write_orchestrator_snapshot_best_effort(
                                _lock_session,
                                lane=lane_key,
                                persist=write_snapshot,
                                running=False,
                                enabled=True,
                                current_activity="Waiting for orchestrator cycle lock",
                                interval_seconds=cycle_interval,
                            )
                    except Exception:
                        pass
                    await _worker_sleep(2)
                    continue

                # 2026-05-20: this prep phase was previously a single
                # ``async with AsyncSessionLocal() as session:`` spanning
                # ~150 lines.  The session was held across read_orchestrator_control,
                # list_traders, an in-memory WS health check, optional
                # update_orchestrator_control + create_trader_event writes,
                # session.get(AppSettings), and a final list_traders fallback —
                # plus all the pure-Python computation between them.  Production
                # logs showed this single session held for **up to 62 seconds**
                # with ``uow_dirty=0 uow_new=0 tables=-`` (i.e. zero DB work)
                # because the event loop was busy elsewhere between awaits.
                # Postgres responds in <1ms locally; the latency is Python-side
                # scheduling, but the held session still parks an MVCC snapshot
                # and a pool connection.
                #
                # Split into six discrete phases.  Each phase opens its own
                # session ONLY for its DB work; in-memory computation runs
                # between sessions with no transaction held.  Atomicity is
                # preserved where required: the WS auto-pause control+event
                # writes stay in one session because they're a single
                # decision; the disabled/credentials snapshot writes use the
                # snapshot helper's fresh-session fast-path so the outer
                # session never needs to be passed in.

                # Phase 1: read orchestrator control (single short session).
                async with AsyncSessionLocal() as session:
                    control = await read_orchestrator_control(session, read_only=True)

                # Phase 2: pure-Python derivation from control.  No session.
                manual_force_cycle = _parse_iso(control.get("requested_run_at")) is not None
                if manual_force_cycle:
                    run_maintenance = False
                mode = _canonical_trader_mode(control.get("mode"), default="shadow")
                cycle_interval = max(
                    1,
                    int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
                )
                sleep_seconds = cycle_interval
                default_trader_cycle_timeout = float(max(30, cycle_interval * 3))
                control_settings = dict(control.get("settings") or {})
                global_runtime_settings = dict(control_settings.get("global_runtime") or {})
                configured_trader_cycle_timeout = safe_float(
                    global_runtime_settings.get("trader_cycle_timeout_seconds"),
                    0.0,
                )
                # Scheduled cycles do maintenance + full signal scans —
                # they need substantially more headroom than runtime
                # triggers.  Floor at 30s to avoid the cycle timing out
                # before individual DB queries can complete.
                effective_trader_cycle_timeout = (
                    configured_trader_cycle_timeout
                    if configured_trader_cycle_timeout > 0
                    else default_trader_cycle_timeout
                )
                trader_cycle_timeout_seconds = float(
                    max(
                        30.0,
                        min(
                            180.0,
                            effective_trader_cycle_timeout,
                        ),
                    )
                )

                # Phase 3: disabled-cycle branch — needs traders list to
                # decide between manage-only vs hard-skip.  Re-opens the
                # session for the list_traders read; closes before any
                # decision so the snapshot write below uses its own session.
                if not control.get("is_enabled"):
                    async with AsyncSessionLocal() as session:
                        traders = [
                            trader
                            for trader in await list_traders(
                                session,
                                mode=mode if mode in {"shadow", "live"} else None,
                            )
                            if _trader_matches_lane(trader, lane_key) and not _is_fast_tier_trader(trader)
                        ]
                    has_active_traders = any(
                        bool(trader.get("is_enabled", True)) and not bool(trader.get("is_paused", False))
                        for trader in traders
                    )
                    if has_active_traders:
                        manage_only_cycle = True
                        manage_only_reasons.append("global_disabled")
                    else:
                        async with AsyncSessionLocal() as session:
                            disabled_stats = await _build_orchestrator_snapshot_metrics(
                                session=session,
                                lane=lane_key,
                                traders=traders,
                            )
                            await _write_orchestrator_snapshot_best_effort(
                                session,
                                lane=lane_key,
                                persist=write_snapshot,
                                running=False,
                                enabled=False,
                                current_activity="Disabled",
                                interval_seconds=cycle_interval,
                                stats=disabled_stats,
                            )
                        skip_cycle = True

                if not skip_cycle and bool(control.get("is_paused")):
                    manage_only_cycle = True
                    manage_only_reasons.append("global_pause")

                if not skip_cycle and bool(control.get("kill_switch")):
                    manage_only_cycle = True
                    manage_only_reasons.append("kill_switch")

                # Phase 4: WS feed health gate — check is in-memory; the
                # pause/resume DB action (when state changes) runs in a
                # short atomic session so update_orchestrator_control +
                # create_trader_event remain in one transaction.
                if not skip_cycle and mode == "live":
                    try:
                        ws_health = get_feed_manager().health_check()
                        poly_failures = ws_health.get("polymarket", {}).get("stats", {}).get("consecutive_failures", 0)
                        if poly_failures >= _WS_FAILURE_PAUSE_THRESHOLD:
                            if not _ws_auto_paused:
                                _ws_auto_paused = True
                                async with AsyncSessionLocal() as session:
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
                            async with AsyncSessionLocal() as session:
                                await update_orchestrator_control(session, is_paused=False)
                                await create_trader_event(
                                    session,
                                    trader_id=None,
                                    event_type="ws_feed_recovery",
                                    source="worker",
                                    severity="info",
                                    message="Polymarket WS feed recovered. Auto-trader resumed.",
                                )
                            logger.info("Polymarket WS recovered; auto-resuming orchestrator")
                    except Exception as exc:
                        logger.debug("WS health check skipped: %s", exc)

                # Phase 5: live-credentials check.  Read AppSettings + verify
                # inside a single short session, then close BEFORE the
                # branch decision.  ``_is_live_credentials_configured``
                # is sync (just decrypts secrets in-memory) — calling it
                # inside the session keeps the ORM object live for attribute
                # access without holding the session for long.
                if not skip_cycle and mode == "live":
                    async with AsyncSessionLocal() as session:
                        app_settings = await session.get(AppSettings, "default")
                        creds_ok = _is_live_credentials_configured(app_settings)
                    if not creds_ok:
                        async with AsyncSessionLocal() as session:
                            creds_stats = await _build_orchestrator_snapshot_metrics(
                                session=session,
                                lane=lane_key,
                                traders=traders,
                            )
                            await _write_orchestrator_snapshot_best_effort(
                                session,
                                lane=lane_key,
                                persist=write_snapshot,
                                running=False,
                                enabled=True,
                                current_activity="Blocked: live credentials missing",
                                interval_seconds=cycle_interval,
                                last_error=None,
                                stats=creds_stats,
                            )
                        skip_cycle = True
                    else:
                        needs_live_execution_init = True

                # Phase 6: trader list fallback (only if Phase 3 didn't run).
                if not skip_cycle and not traders:
                    async with AsyncSessionLocal() as session:
                        traders = [
                            trader
                            for trader in await list_traders(
                                session,
                                mode=mode if mode in {"shadow", "live"} else None,
                            )
                            if _trader_matches_lane(trader, lane_key) and not _is_fast_tier_trader(trader)
                        ]

                # Live-trading readiness gate.  Check is_ready() — a cheap
                # synchronous flag — rather than awaiting ensure_initialized().
                # ensure_initialized() runs the full live_execution.initialize()
                # inline (or blocks on its init lock) whenever the service is
                # not yet ready, and that heavy credential/network/transport
                # setup can wedge this cycle loop indefinitely: if a user armed
                # live before the background initializer finished, the
                # orchestrator would sit at "Live start command queued" forever
                # and never flip to running.  Initialization is owned by
                # _initialize_live_execution_background (which retries until
                # ready); here we only defer the live cycle until it is.
                if not skip_cycle and needs_live_execution_init and not live_execution_service.is_ready():
                    async with AsyncSessionLocal() as session:
                        init_stats = await _build_orchestrator_snapshot_metrics(
                            session=session,
                            lane=lane_key,
                            traders=traders,
                        )
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            lane=lane_key,
                            persist=write_snapshot,
                            running=False,
                            enabled=True,
                            current_activity="Blocked: live trading service initializing",
                            interval_seconds=cycle_interval,
                            last_error=None,
                            stats=init_stats,
                        )
                    skip_cycle = True

                if skip_cycle:
                    (
                        runtime_trigger_event,
                        stream_claim_cursor,
                        stream_last_claim_run_at,
                    ) = await _pause_until_next_cycle(
                        process_runtime_triggers=process_runtime_triggers,
                        sleep_seconds=sleep_seconds,
                        stream_consumer_name=stream_consumer_name,
                        stream_group=stream_group,
                        stream_claim_cursor=stream_claim_cursor,
                        stream_last_claim_run_at=stream_last_claim_run_at,
                    )
                    continue

                cycle_trigger_label = (
                    str(cycle_trigger.get("event_type") or "").strip().lower()
                    if isinstance(cycle_trigger, dict)
                    else ""
                ) or ("requested_run" if manual_force_cycle else "scheduled")

                async with AsyncSessionLocal() as session:
                    if manual_force_cycle:
                        await update_orchestrator_control(session, requested_run_at=None)
                    start_metrics = await _build_orchestrator_snapshot_metrics(
                        session=session,
                        lane=lane_key,
                        traders=traders,
                    )
                    await _write_orchestrator_snapshot_best_effort(
                        session,
                        lane=lane_key,
                        persist=write_snapshot,
                        running=True,
                        enabled=bool(control.get("is_enabled", False)) and not bool(control.get("is_paused", True)),
                        current_activity=f"Starting cycle[{cycle_trigger_label}:{lane_key}]",
                        interval_seconds=cycle_interval,
                        last_error=None,
                        stats=start_metrics,
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
                    due = bool(manual_force_cycle or runtime_trigger_for_trader)
                    if process_scheduled_signals:
                        due = bool(due or _is_due(trader, now))
                    if trader_id in high_frequency_trader_ids and not manage_only_cycle and not is_paused:
                        due = bool(manual_force_cycle or runtime_trigger_for_trader)
                        if process_scheduled_signals:
                            due = bool(
                                due
                                or _is_high_frequency_maintenance_due(
                                    trader,
                                    now,
                                )
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

                all_trader_tasks = [
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
                    for spec in crypto_trader_specs + non_crypto_trader_specs
                ]
                if all_trader_tasks:
                    results = await asyncio.gather(*all_trader_tasks, return_exceptions=True)
                    for idx, result in enumerate(results):
                        if isinstance(result, asyncio.CancelledError):
                            continue
                        if isinstance(result, BaseException):
                            spec = (crypto_trader_specs + non_crypto_trader_specs)[idx]
                            failed_trader_id = str(spec["trader"].get("id") or "")
                            logger.warning(
                                "Trader cycle raised uncaught exception for trader=%s: %s",
                                failed_trader_id,
                                result,
                                exc_info=result,
                            )
                            continue
                        decisions, orders, processed_signals = result
                        total_decisions += decisions
                        total_orders += orders
                        total_processed_signals += processed_signals

                _prune_module_caches({str(t.get("id") or "") for t in traders})

                # ── Deferred maintenance (runs after trader processing) ──
                if run_maintenance:
                    try:
                        await refresh_strategy_runtime_if_needed(source_keys=None)
                    except Exception as exc:
                        logger.warning("Failed strategy runtime refresh pass: %s", exc)

                if run_maintenance:
                    _skip_expiry = (
                        _stale_signal_expiry_backoff_until is not None
                        and utcnow() < _stale_signal_expiry_backoff_until
                    )
                    async with AsyncSessionLocal() as maint_session:
                        if _skip_expiry:
                            pass  # backoff active — skip expiry this cycle
                        else:
                            try:
                                await expire_stale_signals(maint_session)
                                _stale_signal_expiry_consecutive_failures = 0
                                _stale_signal_expiry_backoff_until = None
                            except OperationalError as exc:
                                if _is_retryable_db_error(exc):
                                    if hasattr(maint_session, "rollback"):
                                        await maint_session.rollback()
                                    _stale_signal_expiry_consecutive_failures += 1
                                    _backoff_secs = min(300.0, 5.0 * (2 ** min(_stale_signal_expiry_consecutive_failures - 1, 6)))
                                    _stale_signal_expiry_backoff_until = utcnow() + timedelta(seconds=_backoff_secs)
                                    logger.warning(
                                        "Skipped stale-signal expiry due to transient DB error (backoff %.0fs, failures=%d)",
                                        _backoff_secs, _stale_signal_expiry_consecutive_failures,
                                    )
                                else:
                                    raise
                            except Exception as exc:
                                if hasattr(maint_session, "rollback"):
                                    await maint_session.rollback()
                                _stale_signal_expiry_consecutive_failures += 1
                                _backoff_secs = min(300.0, 5.0 * (2 ** min(_stale_signal_expiry_consecutive_failures - 1, 6)))
                                _stale_signal_expiry_backoff_until = utcnow() + timedelta(seconds=_backoff_secs)
                                logger.warning(
                                    "Failed stale-signal expiry pass (backoff %.0fs, failures=%d): %s",
                                    _backoff_secs, _stale_signal_expiry_consecutive_failures, exc,
                                )

                        try:
                            orphan_cleanup = await _reconcile_orphan_open_orders(maint_session)
                            if orphan_cleanup.get("rows_seen", 0) > 0:
                                logger.warning(
                                    "Reconciled orphan trader orders",
                                    extra={"orphan_cleanup": orphan_cleanup},
                                )
                        except Exception as exc:
                            if hasattr(maint_session, "rollback"):
                                await maint_session.rollback()
                            logger.warning("Failed orphan-order reconciliation pass: %s", exc)

                    control_active = bool(control.get("is_enabled", False)) and not bool(control.get("is_paused", True))
                    if control_active:
                        async with AsyncSessionLocal() as maint_session:
                            try:
                                stale_watchdog = await _run_terminal_stale_order_watchdog(maint_session)
                                if stale_watchdog.get("alerted", 0) > 0:
                                    logger.warning(
                                        "Detected stale active orders on terminal markets",
                                        extra={"stale_watchdog": stale_watchdog},
                                    )
                            except Exception as exc:
                                if hasattr(maint_session, "rollback"):
                                    await maint_session.rollback()
                                logger.warning("Failed stale-terminal-order watchdog pass: %s", exc)

                    # Hot-state maintenance runs in the BACKGROUND (single-flight
                    # across both lanes). The 24h-soak stall dumps caught the
                    # 489s "Starting cycle" freezes here: reseed_if_stale ran
                    # _seed_from_db INLINE under _seed_lock on this loop — the
                    # crypto lane held the lock in a minutes-long DB read while
                    # the general lane queued on the same lock, freezing BOTH
                    # lanes. _seed_from_db builds into shadow dicts and swaps
                    # atomically precisely so concurrent readers stay correct,
                    # so trading on the current in-memory state while the reseed
                    # runs in the background is exactly as safe as the steady
                    # state between reseeds.
                    _kick_hot_state_maintenance()
                    last_maintenance_at = utcnow()

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
                            f"Cycle[{cycle_trigger_label}:{lane_key}] monitoring open orders={open_orders} (no new pending signals)"
                        )
                    elif high_frequency_crypto_active and total_processed_signals == 0:
                        activity = (
                            f"Cycle[{cycle_trigger_label}:{lane_key}] high-frequency crypto monitor active (no new pending signals)"
                        )
                    await _write_orchestrator_snapshot_best_effort(
                        session,
                        lane=lane_key,
                        persist=write_snapshot,
                        running=True,
                        enabled=True,
                        current_activity=activity,
                        interval_seconds=cycle_interval,
                        last_run_at=utcnow(),
                        stats=metrics,
                    )

                # DIAGNOSTIC (cold-start latency): one line per orchestrator
                # iteration so a Start's 30s can be pinned from logs — gap shows
                # how long the loop was parked (pause), the flags show what it
                # decided. Remove once the cold-start cause is fixed.
                try:
                    _now_mono = time.monotonic()
                    _gap = _now_mono - _ORCH_LAST_ITER_MONO.get(lane_key, _now_mono)
                    _ORCH_LAST_ITER_MONO[lane_key] = _now_mono
                    logger.info(
                        "orch-iter lane=%s enabled=%s force=%s skip=%s manage_only=%s req_run_at=%s gap=%.1fs",
                        lane_key,
                        bool(control.get("is_enabled")),
                        manual_force_cycle,
                        skip_cycle,
                        manage_only_cycle,
                        bool(control.get("requested_run_at")),
                        _gap,
                    )
                except Exception:
                    pass

                # Cortex fleet commander — check if a cycle is due
                if lane_key == _LANE_GENERAL and not skip_cycle:
                    try:
                        from workers.cortex_worker import maybe_run_cortex_tick

                        # Hard timeout: cortex's settings read runs INLINE on the
                        # orchestrator hot path, so a transient DB-connection hang
                        # there can freeze the entire scheduled-cycle loop (observed
                        # a 1920s stall: run_worker_loop -> maybe_run_cortex_tick ->
                        # _load_cortex_settings -> DB execute, awaiting forever).
                        # Cortex is COLD and optional on this path — cap it and skip
                        # on timeout so the loop can NEVER block on it. The heavy
                        # cortex cycle itself already runs as a detached task.
                        await asyncio.wait_for(maybe_run_cortex_tick(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Cortex tick exceeded 5s; skipped to protect the orchestrator loop"
                        )
                    except Exception as cortex_exc:
                        logger.debug("Cortex tick check failed (non-critical): %s", cortex_exc)

                (
                    runtime_trigger_event,
                    stream_claim_cursor,
                    stream_last_claim_run_at,
                ) = await _pause_until_next_cycle(
                    process_runtime_triggers=process_runtime_triggers,
                    sleep_seconds=sleep_seconds,
                    stream_consumer_name=stream_consumer_name,
                    stream_group=stream_group,
                    stream_claim_cursor=stream_claim_cursor,
                    stream_last_claim_run_at=stream_last_claim_run_at,
                )
            except Exception as exc:
                logger.exception("Trader orchestrator worker cycle failed: %s", exc)
                try:
                    async with AsyncSessionLocal() as session:
                        control = await read_orchestrator_control(session, read_only=True)
                        error_stats = await _build_orchestrator_snapshot_metrics(
                            session=session,
                            lane=lane_key,
                            traders=traders,
                        )
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            lane=lane_key,
                            persist=write_snapshot,
                            running=False,
                            enabled=bool(control.get("is_enabled", False)),
                            current_activity="Worker error",
                            interval_seconds=max(
                                1, int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS)
                            ),
                            last_error=str(exc),
                            stats=error_stats,
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
                ) = await _pause_until_next_cycle(
                    process_runtime_triggers=process_runtime_triggers,
                    sleep_seconds=2,
                    stream_consumer_name=stream_consumer_name,
                    stream_group=stream_group,
                    stream_claim_cursor=stream_claim_cursor,
                    stream_last_claim_run_at=stream_last_claim_run_at,
                )
    finally:
        if lane_key == _LANE_GENERAL:
            await _release_orchestrator_cycle_lock_owner()
        else:
            await _release_orchestrator_cycle_lock_owner_for_lane(lane_key)


async def start_loop(*, lane: str = _LANE_GENERAL, notifier_enabled: bool = True, write_snapshot: bool = True) -> None:
    """Run the trader orchestrator worker loop (called from API process lifespan).

    R10-B.1d: The TelegramNotifier used to start here on the trading
    plane.  Its monitor loop + queue worker held asyncpg connections
    for 30+ seconds under load, starving the trader hot path.  The
    notifier now lives on the discovery plane (see
    ``workers/host.py::_initialize_services``); callers on this plane
    publish via ``services.notifier_bridge.publish_alert`` which round-
    trips through Redis to the host plane's subscriber.  The
    ``notifier_enabled`` parameter is retained for API-surface stability
    but is a no-op here.
    """
    runtime_trigger_task: asyncio.Task | None = None
    lane_key = str(lane or _LANE_GENERAL).strip().lower() or _LANE_GENERAL
    # notifier_enabled is intentionally ignored on this plane — kept in
    # the signature for callers that still pass it.
    _ = notifier_enabled

    # Start the cycle-context manager BEFORE the worker loop so the
    # global snapshot has time to warm up (~1 s) before the first
    # trader cycle reads from it.  Failure here is not fatal — the
    # context falls back to lock-free hot_state reads + empty global
    # snapshot, which is still a strict superset of correctness.
    try:
        await trader_cycle_context.start()
        await trader_cycle_context.wait_warm(timeout=2.0)
    except Exception as exc:
        logger.warning("Trader cycle context start failed (non-fatal): %s", exc)

    try:
        try:
            runtime_trigger_task = asyncio.create_task(
                run_runtime_trigger_loop(lane=lane_key),
                name=f"trader-orchestrator-runtime-{lane_key}",
            )
            if lane_key == _LANE_GENERAL:
                await run_worker_loop(
                    lane=lane_key,
                    write_snapshot=write_snapshot,
                    process_runtime_triggers=False,
                    process_scheduled_signals=False,
                )
            else:
                await run_worker_loop(
                    lane=lane_key,
                    write_snapshot=write_snapshot,
                    process_runtime_triggers=False,
                )
        except TypeError:
            await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader orchestrator worker shutting down")
    finally:
        try:
            await trader_cycle_context.stop()
        except Exception as exc:
            logger.debug("Trader cycle context stop skipped: %s", exc)
        if runtime_trigger_task is not None and not runtime_trigger_task.done():
            runtime_trigger_task.cancel()
            try:
                await runtime_trigger_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Runtime trigger loop cleanup skipped: %s", exc)
        # R10-B.1d: notifier lifecycle moved off this plane.


async def main(*, lane: str = _LANE_GENERAL, notifier_enabled: bool = True, write_snapshot: bool = True) -> None:
    await init_database()
    try:
        await start_loop(lane=lane, notifier_enabled=notifier_enabled, write_snapshot=write_snapshot)
    except asyncio.CancelledError:
        logger.info("Trader orchestrator worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())

