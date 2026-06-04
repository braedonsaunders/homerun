"""Dedicated live reconciliation worker for trader order/position state."""

from __future__ import annotations

import asyncio
import json as _json
import time
import traceback
from datetime import timedelta
from typing import Any

import asyncpg

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm.exc import StaleDataError

from models.database import AsyncSessionLocal, init_database, recover_pool
from services.session_gate import session_gate
from services.event_bus import event_bus
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.live_execution_service import live_execution_service
from services.live_pressure import db_pressure_snapshot, is_db_pressure_active
from services.trader_orchestrator.position_lifecycle import (
    reconcile_live_positions,
    register_open_orders as register_exit_orders,
    unregister_token as unregister_exit_token,
    get_registered_token_ids as get_exit_registered_tokens,
)
from services.trader_orchestrator_state import (
    _dedupe_live_authority_rows,
    _normalize_live_risk_clamps,
    create_trader_event,
    get_open_order_count_for_trader,
    list_traders,
    read_orchestrator_control,
    recover_missing_live_trader_orders,
    reconcile_live_provider_orders,
    sync_trader_position_inventory,
    OPEN_ORDER_STATUSES,
)
from services.wallet_ws_monitor import wallet_ws_monitor
from services.worker_state import (
    _is_retryable_db_error,
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.converters import safe_float
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("trader_reconciliation_worker")

WORKER_NAME = "trader_reconciliation"
# REST reconciliation is the periodic drift-correction safety net; the
# Polymarket CLOB user-channel WS (polymarket_user_feed -> WalletStateCache)
# is the real-time path for order/fill state.  30s matches the documented
# safety-net cadence — the prior 1s scheduled cadence REST-polled Polymarket
# every second even when fully idle (no positions, no trading).
DEFAULT_INTERVAL_SECONDS = 30
_IDLE_SLEEP_SECONDS = 1
_POSITION_MARK_SYNC_INTERVAL_SECONDS = 10.0
_last_position_mark_sync_at = 0.0
# Refresh the LiveExecutionService in-memory position snapshot from the
# Polymarket data API on this cadence.  Without this, the in-memory
# ledger is only updated at service init; off-app activity (manual sells
# on the UI, transfers, redemptions) silently drifts the ledger and
# causes "balance is not enough" rejections at exit time.
_LIVE_WALLET_POSITIONS_SYNC_INTERVAL_SECONDS = 30.0
_last_live_wallet_positions_sync_at = 0.0
_LIVE_WALLET_POSITIONS_SYNC_TIMEOUT_SECONDS = 20.0
# Sweep open CLOB orders that are clearly stuck — old enough to be
# unintentional and far enough off-market that they will not fill
# during their remaining lifetime.  Strategies still re-quote on their
# normal cadence; the sweeper only handles abandoned residue.
# Cycle cadence is process-local; the actual thresholds (age, drift,
# residual) are sourced from ``config.Settings`` so operators can tune
# them via env vars without an alembic migration.
# Pre-2026-04-28 sweep ran every 5 minutes — too lenient when our
# own unfilled limit orders accumulate and reserve venue-side
# collateral.  The 02:42 cascade showed $122 of a $237 wallet was
# held captive by stale BUYs while every SELL retry kept rejecting
# for "not enough balance / allowance" on a $61 order needing $61
# of fee buffer.  60 seconds is aggressive enough that a BUY which
# fails to fill by its first reconcile cycle gets cancelled before
# the next reconcile; combined with the tighter age (1h) and
# drift (1.5x) thresholds in ``config.Settings``, the open-order
# float should never accumulate to more than a single trader's
# in-flight working set.
_STALE_OPEN_ORDER_SWEEP_INTERVAL_SECONDS = 60.0
_last_stale_open_order_sweep_at = 0.0
# Stuck-position surveillance: scan every 5 minutes for orders whose
# exit retry has been circuit-broken (blocked_persistent_timeout etc.)
# and where the underlying position can be verified directly on-chain.
# Pure observation; never writes.  Tunable via env if 5m proves too
# noisy or too coarse in practice.
_STUCK_POSITION_SCAN_INTERVAL_SECONDS = 300.0
_last_stuck_position_scan_at = 0.0
_STALE_OPEN_ORDER_SWEEP_TIMEOUT_SECONDS = 25.0
# Polymarket-truth realized P&L verification: every 5 minutes, walk all
# orders closed today, fetch the wallet's actual on-chain SELL trades,
# match them to our orders, and overwrite actual_profit with verified
# values. This is the only way our reported P&L stays 100% in sync with
# the user's actual Polymarket account (covers bot-initiated SELLs,
# manual user sells via the Polymarket UI, and any other on-chain exit).
_REALIZED_PNL_VERIFY_INTERVAL_SECONDS = 300.0
_last_realized_pnl_verify_at = 0.0
_REALIZED_PNL_VERIFY_TIMEOUT_SECONDS = 30.0
_MAX_CONSECUTIVE_DB_FAILURES = 3
_CONTROL_REFRESH_SECONDS = 5.0
_ACTIVE_POSITION_TICK_SECONDS = 60.0
_EVENT_QUEUE_MAXSIZE = 4096
_previous_active_order_ids: set[str] = set()
_pg_notify_conn: asyncpg.Connection | None = None
_WALLET_MONITOR_REFRESH_SECONDS = 15.0
_TRADER_RECONCILE_ATTEMPTS = 3
_TRADER_RECONCILE_RETRY_BASE_DELAY_SECONDS = 0.05
_TRADER_RECONCILE_RETRY_MAX_DELAY_SECONDS = 0.3
_TRADER_RECONCILE_TIMEOUT_SECONDS = 30.0
# Cycle budget covers all traders processed sequentially.  With per-
# trader budget at 30s + 5s cancel grace, four traders worst-case
# need 4 × 35 = 140s, plus inter-trader sleep + the cooldown + the
# wallet positions sync.  Pre-2026-04-28 this was 120s — exactly at
# 4 × 30s with zero headroom for the cancel grace, so any single
# slow trader blew the cycle and the abandoned-task pile-up
# triggered the "Reconciliation cycle skipped because the prior
# timed-out cycle is still finishing" cascade.  180s gives 40s of
# margin for the trailing book-keeping.
_RECONCILIATION_CYCLE_TIMEOUT_SECONDS = 180.0
_MAX_TRIGGER_DRAIN_PER_CYCLE = 128
_TIMEOUT_CANCEL_GRACE_SECONDS = 5.0
_STARTUP_INTER_TRADER_SLEEP_SECONDS = 0.0
_DEFAULT_INTER_TRADER_SLEEP_SECONDS = 0.1
_SCHEDULED_CYCLE_COOLDOWN_SECONDS = 1.0
# Coalesce event-triggered REST reconciliation. Every trader_order /
# execution_* event used to fire a full Polymarket REST pass with only 0.5s
# between, hammering the CLOB API ~once/second during active trading. The WS
# user feed already pushes order/fill state in real time, so debounce
# event-driven REST passes to one per 30s (the drift-correction safety net)
# and leave the loop free for the orchestrator to process signals in between.
_EVENT_CYCLE_COOLDOWN_SECONDS = 30.0
_POSITION_TICK_CYCLE_COOLDOWN_SECONDS = 0.25
_TERMINAL_AUDIT_INTERVAL_SECONDS = 300.0
_TERMINAL_AUDIT_TIMEOUT_SECONDS = 8.0
_terminal_audit_next_due_by_trader: dict[str, float] = {}
_RECONCILE_TRIGGER_EVENTS = frozenset(
    {
        "trader_order",
        "execution_order",
        "execution_session",
        "execution_session_event",
        "wallet_trade",
        "provider_status",
    }
)
_abandoned_timed_tasks: set[asyncio.Task] = set()
_inflight_timed_tasks: dict[str, asyncio.Task] = {}

# 2026-05-08: per-trader dedup for the "Live reconciliation timed out"
# warning. During the 2026-05-07 soak a single slow trader produced the
# same warning every 30s for 10 minutes straight (same attempt=3/3,
# same trader_id, same reason). Collapse into one warning per
# _TRADER_RECONCILE_TIMEOUT_LOG_INTERVAL_SECONDS window; carry a
# suppressed-count so the next emitted warning tells the operator how
# many were skipped.
_TRADER_RECONCILE_TIMEOUT_LOG_INTERVAL_SECONDS = 300.0
_reconcile_timeout_last_log_mono: dict[str, float] = {}
_reconcile_timeout_suppressed_count: dict[str, int] = {}


def _terminal_audit_due(trader_id: str) -> bool:
    now_mono = time.monotonic()
    if trader_id not in _terminal_audit_next_due_by_trader:
        _terminal_audit_next_due_by_trader[trader_id] = now_mono + _TERMINAL_AUDIT_INTERVAL_SECONDS
        return False
    next_due = _terminal_audit_next_due_by_trader.get(trader_id, 0.0)
    if now_mono < next_due:
        return False
    _terminal_audit_next_due_by_trader[trader_id] = now_mono + _TERMINAL_AUDIT_INTERVAL_SECONDS
    return True


def _should_emit_reconcile_timeout_warning(trader_id: str) -> tuple[bool, int]:
    """Return (emit, suppressed_count_since_last_emit).

    The caller emits the warning when ``emit`` is True, and attaches
    ``suppressed_count`` to the log so the operator can tell that a
    single warning now represents N underlying failures.
    """
    now_mono = time.monotonic()
    last = _reconcile_timeout_last_log_mono.get(trader_id, 0.0)
    if (now_mono - last) >= _TRADER_RECONCILE_TIMEOUT_LOG_INTERVAL_SECONDS:
        suppressed = _reconcile_timeout_suppressed_count.pop(trader_id, 0)
        _reconcile_timeout_last_log_mono[trader_id] = now_mono
        return True, suppressed
    _reconcile_timeout_suppressed_count[trader_id] = (
        _reconcile_timeout_suppressed_count.get(trader_id, 0) + 1
    )
    return False, 0


class _TimedTaskStillRunningError(RuntimeError):
    pass


def _inter_trader_sleep_seconds(*, reason: str) -> float:
    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason == "startup":
        return _STARTUP_INTER_TRADER_SLEEP_SECONDS
    return _DEFAULT_INTER_TRADER_SLEEP_SECONDS


def _post_cycle_cooldown_seconds(*, reason: str, provider_pass: bool) -> float:
    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason == "position_tick" or not provider_pass:
        return _POSITION_TICK_CYCLE_COOLDOWN_SECONDS
    if normalized_reason.startswith("event:"):
        return _EVENT_CYCLE_COOLDOWN_SECONDS
    return _SCHEDULED_CYCLE_COOLDOWN_SECONDS


def _reconcile_retry_delay_seconds(attempt: int) -> float:
    return min(
        _TRADER_RECONCILE_RETRY_BASE_DELAY_SECONDS * (2**attempt),
        _TRADER_RECONCILE_RETRY_MAX_DELAY_SECONDS,
    )


def _discard_abandoned_task(task: asyncio.Task) -> None:
    _abandoned_timed_tasks.discard(task)
    # Consume any pending exception so abandoned tasks don't surface as
    # "Task exception was never retrieved" in the global asyncio handler.
    # We deliberately abandoned these (soft-timeout path), so the
    # TimeoutError/CancelledError is expected and already logged.
    if not task.cancelled():
        try:
            task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass


# Strong-ref bag for fire-and-forget background tasks spawned out of
# the reconciliation loop.  asyncio only weakly references tasks; if
# we don't hold a ref the task can be GC'd mid-flight.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro, *, label: str) -> asyncio.Task:
    """Schedule *coro* as a fire-and-forget background task.

    Use this for non-critical work that must NOT block the reconciliation
    cycle (e.g., HTTP refreshes that have been observed to time out at
    20s+).  The task's exception is logged at warning level.
    """
    task = asyncio.create_task(coro, name=f"trader-reconciliation-bg-{label}")
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if t.cancelled():
            return
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning("background task %s failed", label, exc_info=exc)

    task.add_done_callback(_on_done)
    return task


def _clear_inflight_timed_task(label: str, task: asyncio.Task) -> None:
    if _inflight_timed_tasks.get(label) is task:
        _inflight_timed_tasks.pop(label, None)


async def _graceful_timeout(coro, *, timeout: float, label: str):
    """Run *coro* with a wall-time bound; cancel it on timeout.

    Pre-2026-04-28 this function only abandoned the task on timeout —
    it kept running in the background, accumulating to dozens of
    in-flight reconciles whenever the venue or DB was slow.  The
    "stuck=210s" telemetry from the cascade was that pile-up: the
    next cycle's pre-check saw the old task still running and skipped,
    while the leaked task continued to consume DB connections, RPC
    bandwidth, and event-loop time.

    The fix: actually call ``task.cancel()`` on timeout, then wait
    up to ``_TIMEOUT_CANCEL_GRACE_SECONDS`` for the task to honor the
    cancel.  Tasks that don't honor cancel within the grace window
    are added to the abandoned set as before — but the common case
    of a stuck ``await polymarket_http(...)`` will release its
    DB-connection / asyncio resources within a tick.

    Cancelling work that's mid-DB-write is acceptable here: the
    lifecycle's ``release_conn(session)`` pattern means SDK calls
    don't span open transactions, and the next reconcile cycle
    re-reads truth from the chain regardless.
    """
    existing = _inflight_timed_tasks.get(label)
    if existing is not None and not existing.done():
        close = getattr(coro, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise _TimedTaskStillRunningError(label)

    task = asyncio.create_task(coro, name=f"trader-reconciliation-{label}")
    _inflight_timed_tasks[label] = task
    task.add_done_callback(lambda done_task, step_label=label: _clear_inflight_timed_task(step_label, done_task))

    async def _cancel_and_drain() -> None:
        if task.done():
            return
        task.cancel()
        try:
            await asyncio.shield(asyncio.wait({task}, timeout=_TIMEOUT_CANCEL_GRACE_SECONDS))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        if not task.done():
            # Task didn't honor cancel within grace window — fall
            # back to abandonment so we don't block the caller.
            _abandoned_timed_tasks.add(task)
            task.add_done_callback(_discard_abandoned_task)

    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done:
            return task.result()
        await _cancel_and_drain()
        raise asyncio.TimeoutError()
    except asyncio.CancelledError:
        await _cancel_and_drain()
        raise


def _default_strategy_params(trader: dict[str, Any]) -> dict[str, Any]:
    source_configs = trader.get("source_configs")
    if not isinstance(source_configs, list):
        return {}
    for row in source_configs:
        if not isinstance(row, dict):
            continue
        params = row.get("strategy_params")
        if isinstance(params, dict):
            return dict(params)
    return {}


def _clamped_live_lifecycle_params(trader: dict[str, Any], control_settings: dict[str, Any]) -> dict[str, Any]:
    params = _default_strategy_params(trader)
    risk_limits = trader.get("risk_limits")
    if isinstance(risk_limits, dict):
        params.update(dict(risk_limits))

    global_runtime = control_settings.get("global_runtime")
    global_runtime = global_runtime if isinstance(global_runtime, dict) else {}
    live_risk_clamps = _normalize_live_risk_clamps(
        global_runtime.get("live_risk_clamps"),
        explicit=bool(global_runtime.get("live_risk_clamps_explicit")),
    )

    trade_cap = safe_float(live_risk_clamps.get("max_trade_notional_usd_cap"))
    if trade_cap is not None and trade_cap > 0.0:
        configured = safe_float(params.get("max_trade_notional_usd"))
        params["max_trade_notional_usd"] = min(configured, trade_cap) if configured and configured > 0.0 else trade_cap

    market_cap = safe_float(live_risk_clamps.get("max_per_market_exposure_usd_cap"))
    if market_cap is not None and market_cap > 0.0:
        configured_market = safe_float(params.get("max_per_market_exposure_usd"))
        params["max_per_market_exposure_usd"] = (
            min(configured_market, market_cap) if configured_market and configured_market > 0.0 else market_cap
        )
        configured_position = safe_float(params.get("max_position_notional_usd"))
        params["max_position_notional_usd"] = (
            min(configured_position, market_cap) if configured_position and configured_position > 0.0 else market_cap
        )

    return params


def _empty_cycle_summary() -> dict[str, Any]:
    return {
        "traders_seen": 0,
        "traders_processed": 0,
        "provider_unavailable_traders": 0,
        "provider_active_seen": 0,
        "provider_status_changes": 0,
        "provider_updates": 0,
        "provider_notional_updates": 0,
        "provider_price_updates": 0,
        "positions_would_close": 0,
        "positions_closed": 0,
        "positions_held": 0,
        "positions_skipped": 0,
        "inventory_open_positions": 0,
        "inventory_updates": 0,
        "inventory_inserts": 0,
        "inventory_closures": 0,
        "failures": 0,
    }


def _wallet_monitor_snapshot_stats() -> dict[str, Any]:
    status = wallet_ws_monitor.get_status()
    raw_stats = status.get("stats")
    stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
    return {
        "wallet_monitor_running": bool(status.get("running")),
        "wallet_monitor_ws_connected": bool(status.get("ws_connected")),
        "wallet_monitor_fallback_polling": bool(status.get("fallback_polling")),
        "wallet_monitor_tracked_wallets": int(status.get("tracked_wallets") or 0),
        "wallet_monitor_last_event_detected_at": str(stats.get("last_event_detected_at") or ""),
        "wallet_monitor_last_block_seen_at": str(stats.get("last_block_seen_at") or ""),
        "wallet_monitor_last_block_processed_at": str(stats.get("last_block_processed_at") or ""),
        "wallet_monitor_last_fallback_poll_at": str(stats.get("last_fallback_poll_at") or ""),
        "wallet_monitor_events_detected_total": int(stats.get("events_detected") or 0),
        "wallet_monitor_blocks_processed_total": int(stats.get("blocks_processed") or 0),
        "wallet_monitor_errors_total": int(stats.get("errors") or 0),
    }


async def _sync_live_wallet_monitor_source(current_wallet: str) -> str:
    execution_wallet = ""
    try:
        if await live_execution_service.ensure_initialized():
            execution_wallet = str(live_execution_service.get_execution_wallet_address() or "").strip().lower()
    except Exception as exc:
        logger.warning("Failed to initialize trading service for wallet WS monitor sync", exc_info=exc)

    if execution_wallet != current_wallet:
        wallet_ws_monitor.set_wallets_for_source(WORKER_NAME, [execution_wallet] if execution_wallet else [])
        current_wallet = execution_wallet

    if execution_wallet:
        try:
            await wallet_ws_monitor.start()
        except Exception as exc:
            logger.warning("Failed to start wallet WS monitor for reconciliation worker", exc_info=exc)

    return current_wallet


async def _reconcile_live_state_for_trader(
    trader: dict[str, Any],
    *,
    provider_pass: bool,
    control_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Per-task connection budget. The reconcile path opens up to 4
    # sessions per trader (provider, open-order count, lifecycle,
    # inventory). With N concurrent reconciliations across the bot
    # fleet this could in principle hold 4N pool slots. A limit of 8
    # gives generous headroom for the current 6-bot workload while
    # bounding the worst case under future scale or accidental
    # over-parallelism (e.g. a future change adds an extra concurrent
    # session per trader). Tasks beyond the limit queue at the gate
    # rather than starving the main pool.
    async with session_gate("trader_reconciliation", limit=8):
        return await _reconcile_live_state_for_trader_inner(
            trader,
            provider_pass=provider_pass,
            control_settings=control_settings,
        )


async def _reconcile_live_state_for_trader_inner(
    trader: dict[str, Any],
    *,
    provider_pass: bool,
    control_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trader_id = str(trader.get("id") or "").strip()
    if not trader_id:
        return {
            "provider": {"active_seen": 0, "status_changes": 0, "updated_orders": 0},
            "lifecycle": {"would_close": 0, "closed": 0},
            "inventory": {"open_positions": 0, "updates": 0, "inserts": 0, "closures": 0},
        }

    provider_result: dict[str, Any] = {
        "provider_ready": True,
        "active_seen": 0,
        "status_changes": 0,
        "updated_orders": 0,
        "notional_updates": 0,
        "price_updates": 0,
    }

    # Provider reconcile + open order count are independent — run in parallel.
    async def _provider_pass() -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            return await reconcile_live_provider_orders(
                session,
                trader_id=trader_id,
                commit=True,
                broadcast=True,
            )

    async def _open_order_count() -> int:
        async with AsyncSessionLocal() as session:
            return await get_open_order_count_for_trader(session, trader_id, mode="live", statuses=OPEN_ORDER_STATUSES)

    if provider_pass:
        provider_result, active_open_orders = await asyncio.gather(
            _provider_pass(), _open_order_count()
        )
    else:
        active_open_orders = await _open_order_count()

    active_seen = int(provider_result.get("active_seen", 0) or 0)
    lifecycle_result: dict[str, Any] = {"would_close": 0, "closed": 0}
    trader_params = _clamped_live_lifecycle_params(trader, control_settings or {})
    if (not provider_pass) or active_seen > 0 or active_open_orders > 0:
        async with AsyncSessionLocal() as session:
            lifecycle_result = await reconcile_live_positions(
                session,
                trader_id=trader_id,
                trader_params=trader_params,
                dry_run=False,
                include_terminal_audit=False,
                reason="reconciliation_worker",
            )
    if _terminal_audit_due(trader_id):
        try:
            async with AsyncSessionLocal() as session:
                terminal_audit_result = await asyncio.wait_for(
                    reconcile_live_positions(
                        session,
                        trader_id=trader_id,
                        trader_params=trader_params,
                        dry_run=False,
                        include_active_candidates=False,
                        include_terminal_audit=True,
                        reason="reconciliation_terminal_audit",
                    ),
                    timeout=_TERMINAL_AUDIT_TIMEOUT_SECONDS,
                )
            lifecycle_result["terminal_audit"] = terminal_audit_result
        except asyncio.TimeoutError:
            logger.warning(
                "Terminal live reconciliation audit timed out for trader=%s timeout=%.1fs",
                trader_id,
                _TERMINAL_AUDIT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Terminal live reconciliation audit failed for trader=%s", trader_id, exc_info=exc)
    async with AsyncSessionLocal() as session:
        inventory_result = await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode="live",
            commit=True,
        )

    return {
        "provider": provider_result,
        "lifecycle": lifecycle_result,
        "inventory": inventory_result,
    }


async def _get_pg_notify_conn() -> asyncpg.Connection | None:
    """Return (or create) a raw asyncpg connection for NOTIFY.

    Re-uses a module-level connection so we don't open a new one per cycle.
    """
    global _pg_notify_conn
    if _pg_notify_conn is not None:
        try:
            if not _pg_notify_conn.is_closed():
                return _pg_notify_conn
        except Exception:
            pass
        _pg_notify_conn = None

    try:
        from config import settings as _settings
        dsn = str(_settings.DATABASE_URL).replace("+asyncpg", "")
        _pg_notify_conn = await asyncpg.connect(dsn=dsn)
        return _pg_notify_conn
    except Exception as exc:
        logger.debug("Failed to create asyncpg NOTIFY connection: %s", exc)
        return None


async def _pg_notify_position_changes(
    new_active: set[str],
    closed_ids: set[str],
    opened_ids: set[str],
    order_token_map: dict[str, str],
) -> None:
    """Fire PG NOTIFY for each detected position change (fill or closure)."""
    conn = await _get_pg_notify_conn()
    if conn is None:
        return

    payloads: list[str] = []
    for oid in opened_ids:
        payloads.append(_json.dumps({
            "action": "fill",
            "order_id": oid,
            "token_id": order_token_map.get(oid, ""),
        }))
    for oid in closed_ids:
        payloads.append(_json.dumps({
            "action": "close",
            "order_id": oid,
            "token_id": order_token_map.get(oid, ""),
        }))

    for payload in payloads:
        try:
            await conn.execute("SELECT pg_notify('position_change', $1)", payload)
        except Exception as exc:
            logger.debug("pg_notify failed: %s", exc)
            # Reset connection on failure so it reconnects next cycle
            global _pg_notify_conn
            try:
                await conn.close()
            except Exception:
                pass
            _pg_notify_conn = None
            break


async def _sync_live_wallet_positions() -> None:
    """Periodically refresh ``live_execution_service`` in-memory positions.

    Throttled to ``_LIVE_WALLET_POSITIONS_SYNC_INTERVAL_SECONDS``.
    Independent of the WalletStateCache reseeder (see
    ``_run_wallet_cache_reseeder_loop`` for that).
    """
    global _last_live_wallet_positions_sync_at
    now = time.monotonic()
    if (now - _last_live_wallet_positions_sync_at) < _LIVE_WALLET_POSITIONS_SYNC_INTERVAL_SECONDS:
        return
    _last_live_wallet_positions_sync_at = now
    if not live_execution_service.is_ready():
        return
    try:
        await asyncio.wait_for(
            live_execution_service.sync_positions(),
            timeout=_LIVE_WALLET_POSITIONS_SYNC_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "live wallet positions sync timed out after %.1fs",
            _LIVE_WALLET_POSITIONS_SYNC_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("live wallet positions sync failed", exc_info=exc)


# ---------------------------------------------------------------------------
# WalletStateCache reseeder
# ---------------------------------------------------------------------------
# Independent loop with its own cadence: aggressive (5s) until first
# successful seed, then settles to ``_WALLET_CACHE_RESEED_INTERVAL_SECONDS``.
# Decoupling from ``_sync_live_wallet_positions`` matters because the
# freshness gate in the orchestrator's hot path blocks live trading until
# the first successful seed lands — getting that seed out as fast as
# possible is the difference between "trader idle for 30s on boot" and
# "trader idle for 5s on boot".
_WALLET_CACHE_RESEED_INTERVAL_SECONDS = 30.0
_WALLET_CACHE_RESEED_BOOTSTRAP_INTERVAL_SECONDS = 5.0


async def _run_wallet_cache_reseeder_loop(stop_event: asyncio.Event) -> None:
    """Continuously seed ``WalletStateCache`` from REST.

    Started by the trading-plane worker host alongside
    ``trader_reconciliation_worker``.  Aggressive 5s cadence until the
    first successful seed lands so the freshness gate clears as quickly
    as possible after boot; 30s cadence afterwards for drift-correction.

    Robustness contract — production saw ``rest_seed_count`` stuck at 1
    for 400+ seconds despite the 30s cadence, meaning a cycle hung
    inside an await that didn't honor its inner ``wait_for`` timeout.
    Defenses:
      * OUTER ``wait_for(70s)`` around the whole reseed so ANY hang
        bails out, even if internals trap cancellation.
      * Cycle-level logging at WARNING when seed is stale, so a stuck
        loop surfaces in production logs without spam during healthy
        operation.
      * Catches ``BaseException`` (not just ``Exception``) so silent
        cancels are visible.  Re-raises ``CancelledError`` so the
        worker host's task-restart can reschedule us.
    """
    from services.wallet_state_cache import get_wallet_state_cache

    cache = get_wallet_state_cache()
    logger.warning(
        "WalletStateCache reseeder loop started",
        bootstrap_interval_s=_WALLET_CACHE_RESEED_BOOTSTRAP_INTERVAL_SECONDS,
        steady_interval_s=_WALLET_CACHE_RESEED_INTERVAL_SECONDS,
    )
    cycle = 0
    # Hard upper bound on a single reseed call.  All internal awaits
    # already have their own timeouts that sum to ~55s, so 70s gives
    # them a few seconds of slack.  If we exceed this it means an
    # await ignored its own timeout — usually because the awaited
    # coroutine was running CPU-bound C code that didn't yield to
    # the cancel signal.
    _RESEED_HARD_BUDGET_SECONDS = 70.0
    while not stop_event.is_set():
        cycle += 1
        if cache._last_rest_seed_succeeded and cache._last_rest_seed_mono is not None:
            interval = _WALLET_CACHE_RESEED_INTERVAL_SECONDS
        else:
            interval = _WALLET_CACHE_RESEED_BOOTSTRAP_INTERVAL_SECONDS

        # Surface degraded state at WARNING so operators see it under
        # any log filter.  Healthy steady state stays quiet.
        seed_age_s = (
            (time.monotonic() - cache._last_rest_seed_mono)
            if cache._last_rest_seed_mono is not None
            else None
        )
        if cache._rest_seed_count == 0 or (seed_age_s is not None and seed_age_s > 90.0):
            logger.warning(
                "WalletStateCache reseed cycle starting (degraded state)",
                cycle=cycle,
                seed_count=cache._rest_seed_count,
                last_succeeded=cache._last_rest_seed_succeeded,
                seed_age_seconds=(round(seed_age_s, 1) if seed_age_s is not None else None),
                next_interval_s=interval,
            )

        cycle_start_mono = time.monotonic()
        try:
            await asyncio.wait_for(
                _reseed_wallet_state_cache_from_rest(),
                timeout=_RESEED_HARD_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WalletStateCache reseed cycle exceeded hard budget — "
                "an inner await did not honor its timeout",
                cycle=cycle,
                budget_s=_RESEED_HARD_BUDGET_SECONDS,
                elapsed_s=round(time.monotonic() - cycle_start_mono, 1),
            )
        except asyncio.CancelledError:
            # A child cancellation should NOT kill the reseeder loop —
            # production showed the loop dying after one cycle because
            # the host's worker-freshness monitor restart-cancelled the
            # parent task while we were mid-cycle, propagating cancel
            # into our wait_for and bubbling out as CancelledError.
            # Re-raising killed the loop permanently; instead, check
            # whether we were genuinely asked to stop and continue if
            # not.  Worker-shutdown sets stop_event BEFORE cancelling
            # the task (see ``run_worker_loop``'s finally block), so
            # a stop_event check after a small yield disambiguates
            # genuine shutdown from spurious cancellation.
            elapsed = round(time.monotonic() - cycle_start_mono, 1)
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            if stop_event.is_set():
                logger.warning(
                    "WalletStateCache reseeder cycle cancelled (shutdown)",
                    cycle=cycle,
                    elapsed_s=elapsed,
                )
                raise
            logger.warning(
                "WalletStateCache reseeder cycle cancelled (recovering — "
                "stop_event not set, will retry next interval)",
                cycle=cycle,
                elapsed_s=elapsed,
            )
        except BaseException as exc:
            logger.warning(
                "WalletStateCache REST reseed cycle failed",
                cycle=cycle,
                elapsed_s=round(time.monotonic() - cycle_start_mono, 1),
                exc_info=exc,
            )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return  # stop signaled
        except asyncio.TimeoutError:
            pass


async def _reseed_wallet_state_cache_from_rest() -> None:
    """Pull a fresh open-positions + closed-positions snapshot from
    Polymarket and feed it to ``WalletStateCache.seed_from_rest``.

    The user-channel WS handles real-time deltas; this function exists
    purely to:
      * Bootstrap the cache at startup (before any WS event has fired)
      * Audit and correct drift periodically (~30s)

    Call sites: ``_sync_live_wallet_positions`` (every 30s) and
    bootstrap path.  No direct exposure to the orchestrator hot path.

    Robust bootstrap contract: if ``live_execution_service`` is not yet
    initialized (e.g. credentials still loading, or a prior init failed
    silently), this function calls ``ensure_initialized()`` to drive the
    retry — silent no-ops here would leave the freshness gate refusing
    every cycle indefinitely.  As a final fallback, if init still
    refuses but the WalletStateCache already has a wallet pinned (from
    ``polymarket_user_feed.configure_credentials``), seed against that
    wallet directly so the freshness gate can clear.
    """
    from services.wallet_state_cache import get_wallet_state_cache

    cache = get_wallet_state_cache()

    # Try to recover from a stuck init by driving a retry.  This is the
    # difference between "trading silently blocked for hours" and
    # "trading recovers within one reseeder tick after creds load".
    if not live_execution_service.is_ready():
        try:
            recovered = await asyncio.wait_for(
                live_execution_service.ensure_initialized(),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            recovered = False
        except Exception as exc:
            logger.warning(
                "WalletStateCache reseeder: ensure_initialized() raised; "
                "falling back to cache-pinned wallet if available",
                exc_info=exc,
            )
            recovered = False
        if not recovered:
            # Last-resort: use the wallet the cache was pinned to via the
            # user-channel WS credential wiring.  If that's also empty,
            # there is genuinely nothing to seed against.
            cache_wallet = cache.wallet_address()
            if not cache_wallet:
                logger.warning(
                    "WalletStateCache reseeder skipped: live_execution_service "
                    "not ready and cache has no pinned wallet; the freshness "
                    "gate will keep refusing trades.  Last init error: %s",
                    live_execution_service.get_last_init_error(),
                )
                return
            logger.warning(
                "WalletStateCache reseeder using cache-pinned wallet=%s "
                "(live_execution_service not ready, last_init_error=%s)",
                cache_wallet,
                live_execution_service.get_last_init_error(),
            )

    cache_wallet = cache.wallet_address()
    wallet_address = (
        # Once the user-channel feed pins the cache wallet, reseed that
        # wallet consistently.  The live client can temporarily expose
        # the EOA while the proxy funder is still resolving; reseeding
        # against that transient address repeatedly fights the cache pin
        # and leaves the hot path reading stale wallet state.
        cache_wallet
        or getattr(live_execution_service, "_proxy_funder_address", None)
        or getattr(live_execution_service, "_wallet_address", None)
        or getattr(live_execution_service, "_eoa_address", None)
    )
    if not wallet_address:
        logger.warning(
            "WalletStateCache reseeder skipped: no wallet address resolved "
            "from live_execution_service or cache.  Trading remains blocked.",
        )
        return
    wallet_lower = str(wallet_address).strip().lower()
    if not wallet_lower:
        logger.warning(
            "WalletStateCache reseeder skipped: wallet address empty after "
            "normalization.  Trading remains blocked.",
        )
        return

    from services.polymarket import polymarket_client
    from services.wallet_state_cache import get_wallet_state_cache

    cache = get_wallet_state_cache()

    # Open positions (real-time) — small list (~tens to a few hundred).
    try:
        open_positions = await asyncio.wait_for(
            polymarket_client.get_wallet_positions(wallet_lower),
            timeout=_LIVE_WALLET_POSITIONS_SYNC_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(
            "WalletStateCache open-positions REST seed failed",
            wallet=wallet_lower,
            exc_info=exc,
        )
        cache.seed_from_rest(
            wallet_address=wallet_lower,
            positions=[],
            closed_positions=[],
            succeeded=False,
        )
        return

    if not isinstance(open_positions, list):
        open_positions = []

    # Closed (held-to-resolution) positions are larger and slower; we
    # don't need them every cycle.  Skip them most of the time and
    # only refresh on a slower cadence.  When we skip, pass an empty
    # list — seed_from_rest preserves existing closed-position state
    # because the merge logic only adds/updates, never removes resolved.
    closed_positions: list[dict[str, Any]] = []
    if _should_refresh_closed_positions():
        try:
            closed_positions_raw = await asyncio.wait_for(
                polymarket_client.get_closed_positions_paginated(
                    wallet_lower,
                    max_positions=300,
                ),
                timeout=12.0,
            )
            if isinstance(closed_positions_raw, list):
                closed_positions = closed_positions_raw
        except Exception as exc:
            logger.debug(
                "WalletStateCache closed-positions REST seed failed (non-fatal)",
                wallet=wallet_lower,
                exc_info=exc,
            )

    result = cache.seed_from_rest(
        wallet_address=wallet_lower,
        positions=open_positions,
        closed_positions=closed_positions,
        succeeded=True,
    )
    logger.debug(
        "WalletStateCache REST seed complete",
        wallet=wallet_lower,
        open_seeded=result.get("open_seeded", 0),
        closed_seeded=result.get("closed_seeded", 0),
        removed_stale=result.get("removed_stale", 0),
    )

    # Replay the full subscription set on the user-channel WS so it
    # covers every market we've ever held a position in (per the
    # architectural mandate, option a).  Idempotent — already-subscribed
    # condition_ids are no-ops.
    try:
        from services.ws_feeds import get_feed_manager

        condition_ids = cache.iter_tracked_condition_ids()
        if condition_ids:
            await get_feed_manager().ensure_user_subscribed(condition_ids)
    except Exception as sub_exc:
        logger.debug(
            "User-channel WS subscription refresh failed (non-fatal)",
            exc_info=sub_exc,
        )


_last_closed_positions_refresh_at_mono: float = 0.0
_CLOSED_POSITIONS_REFRESH_INTERVAL_SECONDS = 300.0  # 5 minutes


def _should_refresh_closed_positions() -> bool:
    global _last_closed_positions_refresh_at_mono
    now_mono = time.monotonic()
    if (now_mono - _last_closed_positions_refresh_at_mono) < _CLOSED_POSITIONS_REFRESH_INTERVAL_SECONDS:
        return False
    _last_closed_positions_refresh_at_mono = now_mono
    return True


async def _scan_stuck_positions_throttled() -> None:
    """Run the stuck-position surveillance pass once per
    ``_STUCK_POSITION_SCAN_INTERVAL_SECONDS``.  No-ops if not yet due.
    """
    global _last_stuck_position_scan_at
    now_mono = time.monotonic()
    if (now_mono - _last_stuck_position_scan_at) < _STUCK_POSITION_SCAN_INTERVAL_SECONDS:
        return
    _last_stuck_position_scan_at = now_mono
    try:
        from services.stuck_position_monitor import run_stuck_position_scan_once

        await run_stuck_position_scan_once()
    except Exception as exc:
        logger.warning("stuck_position_monitor pass failed", exc_info=exc)


async def _sweep_stale_open_orders() -> dict[str, Any]:
    # Cancel CLOB orders that are old enough and off-market enough that they
    # are functionally abandoned.  Three reasons we sweep:
    #   1. Take-profit limits whose target price is far above the current
    #      mid (e.g. SELL @ 0.44 vs mid 0.14) — strategy left them and
    #      moved on; they sit consuming inventory reservation forever.
    #   2. BUY residuals whose remaining size is below the Polymarket
    #      minimum (~5 shares) — they cannot fill regardless of price.
    #   3. Anything older than the configured age that no strategy has
    #      touched in that time.
    global _last_stale_open_order_sweep_at
    now_mono = time.monotonic()
    if (now_mono - _last_stale_open_order_sweep_at) < _STALE_OPEN_ORDER_SWEEP_INTERVAL_SECONDS:
        return {"checked": False, "cancelled": 0, "scanned": 0}
    _last_stale_open_order_sweep_at = now_mono

    summary: dict[str, Any] = {
        "checked": True,
        "scanned": 0,
        "cancelled": 0,
        "errors": [],
    }
    try:
        if not live_execution_service.is_ready():
            return summary
        open_orders = await asyncio.wait_for(
            live_execution_service.get_open_orders(),
            timeout=_STALE_OPEN_ORDER_SWEEP_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        summary["errors"].append("get_open_orders_timeout")
        return summary
    except Exception as exc:
        logger.warning("stale-order sweep failed to fetch open orders", exc_info=exc)
        summary["errors"].append(str(exc))
        return summary

    if not open_orders:
        return summary

    # Resolve thresholds at call time so env-var changes pick up without
    # a restart and the test suite can monkey-patch values per run.
    from config import settings as _live_settings

    age_hours = float(getattr(_live_settings, "STALE_ORDER_AGE_HOURS", 6.0) or 6.0)
    residual_shares = float(
        getattr(_live_settings, "STALE_ORDER_RESIDUAL_SHARES", 1.0) or 0.0
    )
    drift_multiple = float(
        getattr(_live_settings, "STALE_ORDER_PRICE_DRIFT_MULTIPLE", 2.5) or 0.0
    )
    age_hours_no_mid = float(
        getattr(_live_settings, "STALE_ORDER_AGE_HOURS_NO_MID", age_hours * 4) or (age_hours * 4)
    )

    summary["scanned"] = len(open_orders)
    age_cutoff = utcnow() - timedelta(hours=age_hours)
    candidates: list[tuple[Any, str]] = []
    token_ids: set[str] = set()
    for order in open_orders:
        try:
            created = order.created_at
            if created is None or created > age_cutoff:
                continue
            remaining = float(order.size or 0.0) - float(order.filled_size or 0.0)
            if remaining <= 0:
                continue
            target = str(order.clob_order_id or order.id or "").strip()
            if not target:
                continue
            if residual_shares > 0 and remaining < residual_shares:
                candidates.append((order, "residual_below_min"))
                continue
            tid = str(getattr(order, "token_id", "") or "").strip()
            if tid:
                token_ids.add(tid)
            candidates.append((order, "pending_price_check"))
        except Exception:
            continue

    if not candidates:
        return summary

    mids: dict[str, float] = {}
    if token_ids:
        try:
            from services.polymarket import polymarket_client

            price_payload = await asyncio.wait_for(
                polymarket_client.get_prices_batch(list(token_ids)),
                timeout=10.0,
            )
            for tid, data in (price_payload or {}).items():
                if isinstance(data, dict):
                    mid = data.get("mid") or data.get("last") or data.get("price")
                else:
                    mid = data
                try:
                    mid_f = float(mid) if mid is not None else 0.0
                except (TypeError, ValueError):
                    mid_f = 0.0
                if mid_f > 0:
                    mids[str(tid)] = mid_f
        except Exception as exc:
            logger.warning("stale-order sweep mid-price lookup failed", exc_info=exc)

    cancellable: list[tuple[Any, str]] = []
    for order, reason in candidates:
        if reason == "residual_below_min":
            cancellable.append((order, reason))
            continue
        tid = str(getattr(order, "token_id", "") or "").strip()
        mid = mids.get(tid, 0.0)
        try:
            limit = float(order.price or 0.0)
        except (TypeError, ValueError):
            limit = 0.0
        side = getattr(order, "side", None)
        side_text = str(getattr(side, "value", side) or "").strip().upper()
        if mid > 0 and limit > 0 and drift_multiple > 0:
            if side_text == "SELL" and limit >= mid * drift_multiple:
                cancellable.append((order, f"sell_far_above_mid mid={mid:.4f}"))
                continue
            if side_text == "BUY" and mid >= limit * drift_multiple:
                cancellable.append((order, f"buy_far_below_mid mid={mid:.4f}"))
                continue
        # No mid available — fall back to age-only cancellation if the
        # order is significantly older than the configured threshold.
        try:
            order_age_hours = (utcnow() - order.created_at).total_seconds() / 3600.0
        except Exception:
            order_age_hours = 0.0
        if order_age_hours >= age_hours_no_mid:
            cancellable.append((order, f"age_only hours={order_age_hours:.1f}"))

    for order, reason in cancellable:
        target = str(order.clob_order_id or order.id or "").strip()
        if not target:
            continue
        try:
            ok = await asyncio.wait_for(
                live_execution_service.cancel_order(target),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            ok = False
        if ok:
            summary["cancelled"] = int(summary["cancelled"]) + 1
            logger.info(
                "stale-order sweeper cancelled order",
                order_id=target,
                reason=reason,
                side=str(getattr(order, "side", "")),
                price=float(getattr(order, "price", 0) or 0),
                size=float(getattr(order, "size", 0) or 0),
                filled=float(getattr(order, "filled_size", 0) or 0),
            )
        else:
            summary["errors"].append(target)
    return summary


async def _verify_realized_pnl_against_wallet_trades() -> None:
    """Periodically reconcile reported realized P&L against Polymarket truth.

    Walks today's closed orders, fetches the wallet's actual on-chain
    SELL trades from the Polymarket data API, FIFO-matches sells to
    orders by token_id + timestamp, and writes the verified P&L back to
    each row (with verification_status=wallet_activity and the trade's
    transactionHash on verification_tx_hash).

    Catches both bot-initiated SELLs and any manual user sells made via
    the Polymarket UI — both end up as on-chain trade records on the
    same proxy wallet, indistinguishable to the data API. That is the
    correct behavior: we want our DB to match Polymarket regardless of
    who pulled the trigger.

    Orders with no matching SELL trade are NOT touched; they remain
    summary_only with actual_profit=NULL until either a trade record
    appears or the resolution-payout path (separate) verifies them.
    """
    global _last_realized_pnl_verify_at
    now_mono = time.monotonic()
    if (now_mono - _last_realized_pnl_verify_at) < _REALIZED_PNL_VERIFY_INTERVAL_SECONDS:
        return
    _last_realized_pnl_verify_at = now_mono
    try:
        if not live_execution_service.is_ready():
            return
        wallet_address = live_execution_service.get_execution_wallet_address()
        if not wallet_address:
            return
        from services.polymarket_trade_verifier import (
            verify_orders_against_closed_positions,
            verify_orders_from_bot_lineage,
        )
        # PASS 0: bot-lineage FIRST and HIGHEST authority.
        #
        # Uses ONLY data the bot itself recorded — its own SELL fills
        # (matched by the bot's specific clob_order_id, immune to manual
        # user trades on the same wallet) plus deterministic resolution
        # payouts from the bot's recorded BUY size/outcome. This is the
        # ONLY way to guarantee per-bot-order P&L on a wallet that mixes
        # bot trades with occasional manual user trades.
        async with AsyncSessionLocal() as bot_session:
            try:
                bot_result = await asyncio.wait_for(
                    verify_orders_from_bot_lineage(
                        bot_session,
                        commit=True,
                    ),
                    timeout=_REALIZED_PNL_VERIFY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "polymarket_trade_verifier (bot_lineage) timed out after %.1fs",
                    _REALIZED_PNL_VERIFY_TIMEOUT_SECONDS,
                )
                return
        if (bot_result.get("verified_sell_fill", 0) > 0
                or bot_result.get("verified_resolution", 0) > 0):
            logger.info(
                "polymarket_trade_verifier sweep (bot_lineage)",
                examined=bot_result.get("examined"),
                verified_sell_fill=bot_result.get("verified_sell_fill"),
                verified_resolution=bot_result.get("verified_resolution"),
                unmatched=bot_result.get("unmatched"),
                pnl_delta=round(bot_result.get("pnl_delta", 0.0), 4),
            )
        # PASS 1: closed_positions FIRST.
        #
        # For resolved markets, the on-chain settlement price (curPrice
        # on /data/closed-positions) is deterministic truth and cannot
        # be misattributed by manual user trades on the same wallet/
        # token (unlike the trade matcher, which can pull in unrelated
        # SELL trades and conflate them with the bot's exit). We run
        # this first so trade-matcher results don't override the
        # higher-authority settlement.
        async with AsyncSessionLocal() as cp_session:
            try:
                cp_result = await asyncio.wait_for(
                    verify_orders_against_closed_positions(
                        cp_session,
                        wallet_address=wallet_address,
                        commit=True,
                    ),
                    timeout=_REALIZED_PNL_VERIFY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "polymarket_trade_verifier (closed_positions) timed out after %.1fs",
                    _REALIZED_PNL_VERIFY_TIMEOUT_SECONDS,
                )
                return
        if cp_result.get("verified", 0) > 0:
            logger.info(
                "polymarket_trade_verifier sweep (closed_positions)",
                examined=cp_result.get("examined"),
                verified=cp_result.get("verified"),
                unmatched=cp_result.get("unmatched"),
                skipped_already_verified=cp_result.get("skipped_already_verified"),
                closed_positions_fetched=cp_result.get("closed_positions_fetched"),
                pnl_delta=round(cp_result.get("pnl_delta", 0.0), 4),
            )
        # NOTE: verify_orders_against_wallet_trades (the FIFO trade
        # matcher) was deliberately removed from the pipeline. On a
        # wallet that occasionally has manual user trades alongside
        # the bot's, FIFO matching by (asset_id, side, timestamp)
        # systematically misattributes manual SELL trades to bot
        # orders that happen to share the token — there is no
        # per-trade order_id linkage from Polymarket's data API to
        # disambiguate. Sample violation: bot's small SPY entry got
        # matched to the user's manual sell, attributed +$219.51
        # phantom win to the bot. Bot-lineage (uses bot's OWN
        # clob_order_id from pending_live_exit) and closed_positions
        # (uses bot's OWN cost basis × wallet's deterministic
        # settlement curPrice) are conflation-immune; trade matcher
        # is not. We accept lower coverage in exchange for correctness.
    except Exception as exc:
        logger.warning("polymarket_trade_verifier sweep failed", exc_info=exc)


async def _sync_position_marks_and_exit_registry() -> None:
    """Refresh PositionMarkState and exit evaluation registry with current open orders.

    Called periodically after reconciliation cycles to keep the event-driven
    infrastructure aware of which positions exist and which tokens to monitor.
    """
    global _last_position_mark_sync_at
    now = time.monotonic()
    if (now - _last_position_mark_sync_at) < _POSITION_MARK_SYNC_INTERVAL_SECONDS:
        return
    _last_position_mark_sync_at = now

    try:
        from models.database import TraderOrder
        from sqlalchemy import select

        from services.position_mark_state import get_position_mark_state
        from services.ws_feeds import get_feed_manager

        pms = get_position_mark_state()
        feed_manager = get_feed_manager()

        async with AsyncSessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(TraderOrder).where(
                            TraderOrder.mode == "live",
                            TraderOrder.status.in_(tuple(OPEN_ORDER_STATUSES)),
                        )
                    )
                )
                .scalars()
                .all()
            )
        rows = _dedupe_live_authority_rows(rows)

        # Build sets for tracking
        active_order_ids: set[str] = set()
        token_ids_to_subscribe: set[str] = set()
        exit_registry_by_token: dict[str, list[dict[str, Any]]] = {}

        for row in rows:
            order_id = str(row.id)
            active_order_ids.add(order_id)
            payload = dict(row.payload_json or {})
            live_market = payload.get("live_market") if isinstance(payload.get("live_market"), dict) else {}
            position_state = payload.get("position_state") if isinstance(payload.get("position_state"), dict) else {}

            # Resolve token_id
            token_id = str(
                payload.get("selected_token_id")
                or payload.get("token_id")
                or live_market.get("selected_token_id")
                or ""
            ).strip()
            if not token_id:
                continue

            market_id = str(row.market_id or "").strip()
            direction = str(row.direction or "yes").strip().lower()
            entry_price = float(row.effective_price or row.entry_price or 0)
            # Try payload fill data when row-level price is missing
            if entry_price <= 0:
                _recon = payload.get("provider_reconciliation")
                if isinstance(_recon, dict):
                    entry_price = float(_recon.get("average_fill_price") or 0)
                if entry_price <= 0:
                    _snap = _recon.get("snapshot") if isinstance(_recon, dict) else None
                    if isinstance(_snap, dict):
                        entry_price = float(_snap.get("limit_price") or _snap.get("average_fill_price") or 0)
            notional = float(row.notional_usd or 0)
            edge_pct = float(row.edge_percent or 0)

            if notional <= 0:
                continue

            token_ids_to_subscribe.add(token_id)

            # Register with PositionMarkState
            pms.register_position(
                order_id=order_id,
                market_id=market_id,
                token_id=token_id,
                direction=direction,
                entry_price=entry_price,
                notional=notional,
                edge_percent=edge_pct,
            )

            # Build exit evaluation registry entry
            if token_id not in exit_registry_by_token:
                exit_registry_by_token[token_id] = []

            strategy_exit_config = (
                payload.get("strategy_exit_config")
                if isinstance(payload.get("strategy_exit_config"), dict)
                else {}
            )
            pending_exit = payload.get("pending_live_exit") if isinstance(payload.get("pending_live_exit"), dict) else {}

            exit_registry_by_token[token_id].append({
                "order_id": order_id,
                "trader_id": str(row.trader_id or ""),
                "entry_price": entry_price,
                "has_pending_exit": bool(pending_exit.get("status") in ("submitted", "working")),
                "take_profit_pct": float(strategy_exit_config.get("take_profit_pct") or 0) or None,
                "stop_loss_pct": float(strategy_exit_config.get("stop_loss_pct") or 0) or None,
                "trailing_stop_pct": float(strategy_exit_config.get("trailing_stop_pct") or 0) or None,
                "min_hold_minutes": float(strategy_exit_config.get("min_hold_minutes") or 0),
                "highest_price": float(position_state.get("highest_price") or 0) or None,
                "age_anchor": str(
                    (row.executed_at or row.updated_at or row.created_at).isoformat()
                    if (row.executed_at or row.updated_at or row.created_at)
                    else ""
                ),
            })

        # Unregister closed positions from PositionMarkState
        current_marks = pms.get_marks()
        for existing_oid in list(current_marks.keys()):
            if existing_oid not in active_order_ids:
                pms.unregister_position(existing_oid)

        # Update exit evaluation registry
        all_registered_tokens = set(get_exit_registered_tokens())
        for token_id, orders in exit_registry_by_token.items():
            register_exit_orders(token_id, orders)
        for old_token in all_registered_tokens - set(exit_registry_by_token.keys()):
            unregister_exit_token(old_token)

        # Subscribe tokens to WS feed for price updates
        if token_ids_to_subscribe and getattr(feed_manager, "_started", False):
            try:
                await feed_manager.polymarket_feed.subscribe(sorted(token_ids_to_subscribe))
            except Exception:
                pass

        # Detect position changes and fire PG NOTIFY for the API process
        global _previous_active_order_ids
        opened_ids = active_order_ids - _previous_active_order_ids
        closed_ids = _previous_active_order_ids - active_order_ids
        if opened_ids or closed_ids:
            # Build order→token map for the notification payload
            order_token_map: dict[str, str] = {}
            for row in rows:
                oid = str(row.id)
                payload = dict(row.payload_json or {})
                lm = payload.get("live_market") if isinstance(payload.get("live_market"), dict) else {}
                tid = str(
                    payload.get("selected_token_id")
                    or payload.get("token_id")
                    or lm.get("selected_token_id")
                    or ""
                ).strip()
                order_token_map[oid] = tid
            try:
                await _pg_notify_position_changes(
                    new_active=active_order_ids,
                    closed_ids=closed_ids,
                    opened_ids=opened_ids,
                    order_token_map=order_token_map,
                )
            except Exception as exc:
                logger.debug("Position change notify failed: %s", exc)
            _previous_active_order_ids = set(active_order_ids)

        logger.debug(
            "Position mark sync: positions=%d tokens=%d",
            len(active_order_ids),
            len(token_ids_to_subscribe),
        )

    except Exception as exc:
        logger.warning("Position mark sync failed", exc_info=exc)


_recovery_next_attempt_at: float = 0.0
_RECOVERY_BACKOFF_SECONDS = 120.0  # back off 2 minutes on failure
_RECOVERY_SUCCESS_INTERVAL_SECONDS = 300.0
_RECOVERY_DB_PRESSURE_BACKOFF_SECONDS = 180.0
_AUTHORITY_RECOVERY_STATEMENT_TIMEOUT_MS = 8000
_AUTHORITY_RECOVERY_LOCK_TIMEOUT_MS = 1000
_live_order_cleanup_next_at: float = 0.0


async def _run_reconciliation_cycle(
    *,
    reason: str,
    emit_event: bool,
    provider_pass: bool,
    heartbeat_activity: str = "",
) -> dict[str, Any]:
    global _recovery_next_attempt_at
    summary = _empty_cycle_summary()
    now_mono = time.monotonic()
    if now_mono >= _recovery_next_attempt_at:
        if is_db_pressure_active():
            _recovery_next_attempt_at = time.monotonic() + _RECOVERY_DB_PRESSURE_BACKOFF_SECONDS
            logger.warning(
                "Skipping live order authority recovery under DB pressure; backing off %.0fs",
                _RECOVERY_DB_PRESSURE_BACKOFF_SECONDS,
                db_pressure=db_pressure_snapshot(),
            )
        else:
            try:
                async def _authority_recovery() -> None:
                    async with AsyncSessionLocal() as session:
                        # Fold both timeout SETs into a single round-trip.
                        # The production log captured authority_recovery
                        # holding a 16.89s transaction (uow_dirty=1,
                        # tables=execution_sessions) — the two preliminary
                        # SET LOCAL round-trips were small but billable
                        # contributors on every recovery cycle.  See
                        # ``shared_state._apply_local_db_timeouts`` for
                        # the same fold pattern with the same rationale.
                        await session.execute(
                            text(
                                "SELECT "
                                "set_config('statement_timeout', :stmt_ms, true), "
                                "set_config('lock_timeout', :lock_ms, true)"
                            ),
                            {
                                "stmt_ms": str(_AUTHORITY_RECOVERY_STATEMENT_TIMEOUT_MS),
                                "lock_ms": str(_AUTHORITY_RECOVERY_LOCK_TIMEOUT_MS),
                            },
                        )
                        await recover_missing_live_trader_orders(
                            session,
                            trader_ids=None,
                            commit=True,
                            broadcast=True,
                        )

                await _graceful_timeout(
                    _authority_recovery(),
                    label="authority_recovery",
                    timeout=_TRADER_RECONCILE_TIMEOUT_SECONDS,
                )
                _recovery_next_attempt_at = time.monotonic() + _RECOVERY_SUCCESS_INTERVAL_SECONDS
            except _TimedTaskStillRunningError:
                _recovery_next_attempt_at = time.monotonic() + _RECOVERY_BACKOFF_SECONDS
                logger.warning(
                    "Live order authority recovery is still finishing from the prior timeout; backing off %.0fs",
                    _RECOVERY_BACKOFF_SECONDS,
                )
            except asyncio.TimeoutError:
                _recovery_next_attempt_at = time.monotonic() + _RECOVERY_BACKOFF_SECONDS
                logger.warning(
                    "Live order authority recovery timed out after %.0fs; backing off %.0fs",
                    _TRADER_RECONCILE_TIMEOUT_SECONDS,
                    _RECOVERY_BACKOFF_SECONDS,
                )
            except Exception as exc:
                _recovery_next_attempt_at = time.monotonic() + _RECOVERY_BACKOFF_SECONDS
                logger.warning(
                    "Live order authority recovery failed; backing off %.0fs",
                    _RECOVERY_BACKOFF_SECONDS,
                    exc_info=exc,
                )
    # Periodic cleanup of stale live_trading_orders (once per hour).
    # These accumulate from CLOB order syncs and the recovery function
    # loads ALL of them — keeping the table small prevents memory spikes.
    global _live_order_cleanup_next_at
    now_mono2 = time.monotonic()
    if now_mono2 >= _live_order_cleanup_next_at:
        _live_order_cleanup_next_at = now_mono2 + 3600.0  # 1 hour
        try:
            async with AsyncSessionLocal() as cleanup_session:
                from sqlalchemy import text as sa_text
                result = await cleanup_session.execute(
                    sa_text("""
                        DELETE FROM live_trading_orders
                        WHERE created_at < now() - interval '7 days'
                        AND lower(coalesce(status, '')) NOT IN ('open', 'pending', 'partially_filled')
                    """)
                )
                deleted = result.rowcount
                await cleanup_session.commit()
                if deleted > 0:
                    logger.info("Cleaned up %d stale live_trading_orders (>7 days old)", deleted)
        except Exception as exc:
            logger.warning("Failed to clean up stale live_trading_orders", exc_info=exc)

    async with AsyncSessionLocal() as session:
        traders = await list_traders(session)
        orchestrator_control = await read_orchestrator_control(session)
    control_settings = dict(orchestrator_control.get("settings") or {})

    summary["traders_seen"] = len(traders)

    async def _reconcile_one(trader: dict[str, Any]) -> dict[str, Any] | None:
        trader_id = str(trader.get("id") or "").strip()
        if not trader_id:
            return None
        for attempt in range(_TRADER_RECONCILE_ATTEMPTS):
            try:
                return await _graceful_timeout(
                    _reconcile_live_state_for_trader(
                        trader,
                        provider_pass=provider_pass,
                        control_settings=control_settings,
                    ),
                    label=f"reconcile:{trader_id}",
                    timeout=_TRADER_RECONCILE_TIMEOUT_SECONDS,
                )
            except _TimedTaskStillRunningError:
                logger.warning(
                    "Live reconciliation skipped because prior timed-out cleanup is still finishing for trader=%s reason=%s",
                    trader_id,
                    reason,
                )
                summary["failures"] = int(summary["failures"]) + 1
                break
            except asyncio.TimeoutError:
                emit, suppressed = _should_emit_reconcile_timeout_warning(trader_id)
                if emit:
                    logger.warning(
                        "Live reconciliation timed out for trader=%s reason=%s attempt=%d/%d timeout=%.1fs suppressed_since_last=%d",
                        trader_id,
                        reason,
                        attempt + 1,
                        _TRADER_RECONCILE_ATTEMPTS,
                        _TRADER_RECONCILE_TIMEOUT_SECONDS,
                        suppressed,
                    )
                else:
                    logger.debug(
                        "Live reconciliation timed out (suppressed) trader=%s reason=%s attempt=%d/%d",
                        trader_id,
                        reason,
                        attempt + 1,
                        _TRADER_RECONCILE_ATTEMPTS,
                    )
                summary["failures"] = int(summary["failures"]) + 1
                break
            except StaleDataError as exc:
                if attempt < _TRADER_RECONCILE_ATTEMPTS - 1:
                    logger.warning(
                        "Live reconciliation stale-row conflict for trader=%s reason=%s attempt=%d/%d; retrying",
                        trader_id,
                        reason,
                        attempt + 1,
                        _TRADER_RECONCILE_ATTEMPTS,
                        exc_info=exc,
                    )
                    await asyncio.sleep(_reconcile_retry_delay_seconds(attempt))
                    continue
                summary["failures"] = int(summary["failures"]) + 1
                logger.warning(
                    "Live reconciliation failed for trader=%s reason=%s error_type=%s retryable_db=%s",
                    trader_id,
                    reason,
                    type(exc).__name__,
                    False,
                    exc_info=exc,
                )
            except Exception as exc:
                retryable_db = _is_retryable_db_error(exc)
                if retryable_db and attempt < _TRADER_RECONCILE_ATTEMPTS - 1:
                    logger.warning(
                        "Live reconciliation retrying trader=%s reason=%s attempt=%d/%d due retryable DB error (%s)",
                        trader_id,
                        reason,
                        attempt + 1,
                        _TRADER_RECONCILE_ATTEMPTS,
                        type(exc).__name__,
                        exc_info=exc,
                    )
                    await asyncio.sleep(_reconcile_retry_delay_seconds(attempt))
                    continue
                summary["failures"] = int(summary["failures"]) + 1
                # Capture the full traceback (formatted) inline.  The
                # structured logger's ``||`` summary line drops most
                # frames, so for diagnosing async/greenlet issues like
                # MissingGreenlet we need the file:line of the originating
                # call site preserved verbatim.
                traceback_text = traceback.format_exc()
                logger.warning(
                    "Live reconciliation failed for trader=%s reason=%s error_type=%s retryable_db=%s traceback=%s",
                    trader_id,
                    reason,
                    type(exc).__name__,
                    retryable_db,
                    traceback_text,
                    exc_info=exc,
                )
            break
        return None

    # Run traders sequentially (concurrency=1) so we can update heartbeat between each.
    # Sleep between traders to release DB pool pressure — the reconciliation holds
    # a connection for 7-15s of Polymarket API I/O per trader, which starves the
    # trader orchestrator of connections for signal processing.
    inter_trader_sleep_seconds = _inter_trader_sleep_seconds(reason=reason)
    results: list[dict[str, Any] | None | BaseException] = []
    for idx, trader in enumerate(traders):
        try:
            result = await _reconcile_one(trader)
        except Exception as exc:
            result = exc
        results.append(result)
        if inter_trader_sleep_seconds > 0.0 and idx < len(traders) - 1:
            await asyncio.sleep(inter_trader_sleep_seconds)
        # Update heartbeat between traders to prevent stale detection
        if heartbeat_activity and idx < len(traders) - 1:
            try:
                async with AsyncSessionLocal() as hb_session:
                    processed = int(summary["traders_processed"]) + (1 if result and not isinstance(result, BaseException) else 0)
                    await write_worker_snapshot(
                        hb_session,
                        WORKER_NAME,
                        running=True,
                        enabled=True,
                        current_activity=(
                            f"{heartbeat_activity} {processed}/"
                            f"{len(traders)} traders"
                        ),
                        interval_seconds=DEFAULT_INTERVAL_SECONDS,
                        last_run_at=utcnow(),
                        stats={
                            "cycle_reason": reason,
                            "provider_pass": bool(provider_pass),
                            "traders_seen": len(traders),
                            "traders_processed": processed,
                            "provider_updates": int(summary["provider_updates"]),
                            "positions_closed": int(summary["positions_closed"]),
                            "inventory_open_positions": int(summary["inventory_open_positions"]),
                            **_wallet_monitor_snapshot_stats(),
                        },
                    )
            except Exception:
                pass
    for result in results:
        if isinstance(result, BaseException) or result is None:
            continue
        provider = result.get("provider") or {}
        lifecycle = result.get("lifecycle") or {}
        inventory = result.get("inventory") or {}
        summary["traders_processed"] = int(summary["traders_processed"]) + 1
        if not bool(provider.get("provider_ready", True)):
            summary["provider_unavailable_traders"] = int(summary["provider_unavailable_traders"]) + 1
        summary["provider_active_seen"] = int(summary["provider_active_seen"]) + int(
            provider.get("active_seen", 0) or 0
        )
        summary["provider_status_changes"] = int(summary["provider_status_changes"]) + int(
            provider.get("status_changes", 0) or 0
        )
        summary["provider_updates"] = int(summary["provider_updates"]) + int(provider.get("updated_orders", 0) or 0)
        summary["provider_notional_updates"] = int(summary["provider_notional_updates"]) + int(
            provider.get("notional_updates", 0) or 0
        )
        summary["provider_price_updates"] = int(summary["provider_price_updates"]) + int(
            provider.get("price_updates", 0) or 0
        )
        summary["positions_would_close"] = int(summary["positions_would_close"]) + int(
            lifecycle.get("would_close", 0) or 0
        )
        summary["positions_closed"] = int(summary["positions_closed"]) + int(lifecycle.get("closed", 0) or 0)
        summary["positions_held"] = int(summary["positions_held"]) + int(lifecycle.get("held", 0) or 0)
        summary["positions_skipped"] = int(summary["positions_skipped"]) + int(lifecycle.get("skipped", 0) or 0)
        summary["inventory_open_positions"] = int(summary["inventory_open_positions"]) + int(
            inventory.get("open_positions", 0) or 0
        )
        summary["inventory_updates"] = int(summary["inventory_updates"]) + int(inventory.get("updates", 0) or 0)
        summary["inventory_inserts"] = int(summary["inventory_inserts"]) + int(inventory.get("inserts", 0) or 0)
        summary["inventory_closures"] = int(summary["inventory_closures"]) + int(inventory.get("closures", 0) or 0)

    if emit_event:
        async with AsyncSessionLocal() as session:
            await create_trader_event(
                session,
                trader_id=None,
                event_type="live_reconciliation_cycle",
                source="worker",
                message=(
                    f"Live reconciliation {reason}: "
                    f"processed={int(summary['traders_processed'])}/{int(summary['traders_seen'])}, "
                    f"provider_updates={int(summary['provider_updates'])}, "
                    f"positions_closed={int(summary['positions_closed'])}"
                ),
                payload={
                    "reason": reason,
                    "provider_pass": bool(provider_pass),
                    **summary,
                    **_wallet_monitor_snapshot_stats(),
                },
                commit=True,
            )
    return summary


async def run_worker_loop() -> None:
    logger.info("Starting trader reconciliation worker loop")

    # Cycle 5 of the perf-harness loop hit a ``stale_heartbeat`` watchdog
    # restart at +35s of new-worker runtime.  Root cause: the previous
    # worker process left ``last_run_at`` in the ``worker_snapshots``
    # row from N minutes ago.  The new worker spawned, the watchdog
    # observed the old timestamp and killed the new process before it
    # ever wrote its first heartbeat (the first write happens at line
    # 1751 below, AFTER the startup reconciliation cycle which itself
    # can take up to 180s).  Stamping ``last_run_at = now()`` here
    # ensures the watchdog sees a fresh row within the first ~50ms of
    # worker startup, eliminating the race entirely.  Activity string
    # is intentionally generic — the cycle tail will overwrite it with
    # the real "Reconciled X/Y traders ..." status.
    try:
        async with AsyncSessionLocal() as _bootstrap_session:
            await write_worker_snapshot(
                _bootstrap_session,
                WORKER_NAME,
                running=True,
                enabled=True,
                current_activity="Worker booting",
                interval_seconds=DEFAULT_INTERVAL_SECONDS,
                last_run_at=utcnow(),
            )
    except Exception as exc:
        logger.warning(
            "Failed to write startup heartbeat snapshot; watchdog may misread the prior process's row",
            exc_info=exc,
        )

    # Wallet cache reseeder MUST start before any DB-dependent startup
    # work.  If we hang on ``ensure_all_strategies_seeded`` or any other
    # slow query at boot, traders block on ``no_rest_seed_yet`` because
    # the reseeder task was never created.  Decouple the wallet
    # bootstrap entirely from strategy/DB startup — they are independent
    # and the wallet path uses HTTP, not the DB pool that may be
    # contended.
    wallet_cache_reseeder_stop = asyncio.Event()
    wallet_cache_reseeder_task = asyncio.create_task(
        _run_wallet_cache_reseeder_loop(wallet_cache_reseeder_stop),
        name="wallet-cache-reseeder",
    )

    # Strategy seeding/refresh: run as a SEPARATE BACKGROUND TASK so
    # the wait_for timeout's cancellation chain cannot propagate into
    # ``run_worker_loop``'s parent task and silently kill sibling tasks
    # like the wallet reseeder.  Production caught this on Python 3.14:
    # the strategy seed's wait_for(30s) timed out exactly when the
    # reseeder cycle was 29.6-29.9s in, and the reseeder cycle was
    # cancelled — confirmed by the watchdog.  The strategy seed is
    # non-critical here (the host bootstrap also seeds strategies),
    # so worst-case isolation is the right call.
    async def _seed_strategies_in_background() -> None:
        try:
            async with AsyncSessionLocal() as session:
                await asyncio.wait_for(
                    ensure_all_strategies_seeded(session),
                    timeout=30.0,
                )
            await asyncio.wait_for(
                refresh_strategy_runtime_if_needed(source_keys=None, force=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Reconciliation strategy startup sync timed out; "
                "continuing without — host bootstrap also seeds strategies."
            )
        except Exception as exc:
            logger.warning("Reconciliation strategy startup sync failed", exc_info=exc)

    asyncio.create_task(
        _seed_strategies_in_background(),
        name="reconciliation-strategy-seed-bg",
    )

    consecutive_db_failures = 0
    wallet_monitor_wallet = ""
    wallet_monitor_refresh_at = 0.0
    wallet_monitor_wallet = await _sync_live_wallet_monitor_source(wallet_monitor_wallet)
    wallet_monitor_refresh_at = time.monotonic() + _WALLET_MONITOR_REFRESH_SECONDS

    startup_summary = _empty_cycle_summary()
    try:
        startup_summary = await _graceful_timeout(
            _run_reconciliation_cycle(
                reason="startup",
                emit_event=True,
                provider_pass=True,
                heartbeat_activity="Startup reconciling",
            ),
            label="cycle:startup",
            timeout=_RECONCILIATION_CYCLE_TIMEOUT_SECONDS,
        )
    except _TimedTaskStillRunningError:
        logger.warning("Startup reconciliation cycle is still finishing from the prior timeout")
    except asyncio.TimeoutError:
        logger.warning(
            "Startup reconciliation cycle timed out after %.0fs",
            _RECONCILIATION_CYCLE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("Startup full reconciliation failed", exc_info=exc)

    try:
        async with AsyncSessionLocal() as session:
            await write_worker_snapshot(
                session,
                WORKER_NAME,
                running=True,
                enabled=True,
                current_activity=(
                    f"Startup full sync processed={int(startup_summary['traders_processed'])}/"
                    f"{int(startup_summary['traders_seen'])}"
                ),
                interval_seconds=DEFAULT_INTERVAL_SECONDS,
                last_run_at=utcnow(),
                stats={
                    **startup_summary,
                    **_wallet_monitor_snapshot_stats(),
                    "cycle_reason": "startup",
                    "provider_pass": True,
                },
            )
    except Exception as exc:
        logger.warning("Failed to persist startup reconciliation snapshot", exc_info=exc)

    interval_seconds = DEFAULT_INTERVAL_SECONDS
    is_enabled = True
    is_paused = False
    requested_run = False
    last_open_positions = int(startup_summary.get("inventory_open_positions", 0) or 0)
    last_provider_pass_at = time.monotonic()
    control_refresh_at = 0.0
    trigger_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAXSIZE)

    async def _on_runtime_event(event_type: str, data: dict[str, Any]) -> None:
        _ = data
        if event_type not in _RECONCILE_TRIGGER_EVENTS:
            return
        try:
            trigger_queue.put_nowait(event_type)
        except asyncio.QueueFull:
            logger.warning("Reconciliation event queue full; dropping trigger", event_type=event_type)

    await event_bus.start()
    for event_type in sorted(_RECONCILE_TRIGGER_EVENTS):
        event_bus.subscribe(event_type, _on_runtime_event)

    try:
        while True:
            try:
                now_mono = time.monotonic()
                if now_mono >= control_refresh_at:
                    async with AsyncSessionLocal() as session:
                        control = await read_worker_control(
                            session,
                            WORKER_NAME,
                            default_interval=DEFAULT_INTERVAL_SECONDS,
                        )
                        interval_seconds = max(1, int(control.get("interval_seconds") or DEFAULT_INTERVAL_SECONDS))
                        is_enabled = bool(control.get("is_enabled", True))
                        is_paused = bool(control.get("is_paused", False))
                        requested_run = bool(control.get("requested_run_at") is not None)
                        if requested_run:
                            await clear_worker_run_request(session, WORKER_NAME)
                    try:
                        await refresh_strategy_runtime_if_needed(source_keys=None)
                    except Exception as exc:
                        logger.warning("Reconciliation strategy runtime refresh failed", exc_info=exc)

                    if not is_enabled or is_paused:
                        async with AsyncSessionLocal() as snap_session:
                            await write_worker_snapshot(
                                snap_session,
                                WORKER_NAME,
                                running=False,
                                enabled=bool(is_enabled and not is_paused),
                                current_activity="Paused" if is_paused else "Disabled",
                                interval_seconds=interval_seconds,
                                stats={
                                    "requested_run": bool(requested_run),
                                    "open_positions_last_cycle": int(last_open_positions),
                                    **_wallet_monitor_snapshot_stats(),
                                },
                            )

                    control_refresh_at = now_mono + _CONTROL_REFRESH_SECONDS

                if now_mono >= wallet_monitor_refresh_at:
                    wallet_monitor_wallet = await _sync_live_wallet_monitor_source(wallet_monitor_wallet)
                    wallet_monitor_refresh_at = now_mono + _WALLET_MONITOR_REFRESH_SECONDS

                if not is_enabled or is_paused:
                    await asyncio.sleep(max(_IDLE_SLEEP_SECONDS, interval_seconds))
                    continue

                cycle_reason = "scheduled"
                provider_pass = True
                emit_event = False

                if requested_run:
                    cycle_reason = "requested"
                    provider_pass = True
                    emit_event = True
                    requested_run = False
                else:
                    wait_seconds = _ACTIVE_POSITION_TICK_SECONDS if last_open_positions > 0 else float(interval_seconds)
                    try:
                        event_type = await asyncio.wait_for(
                            trigger_queue.get(),
                            timeout=max(0.05, wait_seconds),
                        )
                        drained_event_types = [event_type]
                        while len(drained_event_types) < _MAX_TRIGGER_DRAIN_PER_CYCLE:
                            try:
                                drained_event_types.append(trigger_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                        unique_event_types = list(dict.fromkeys(str(item or "").strip() for item in drained_event_types if item))
                        drained_count = len(drained_event_types) - 1
                        cycle_reason = f"event:{unique_event_types[0]}"
                        if len(unique_event_types) > 1:
                            cycle_reason = f"{cycle_reason}+{len(unique_event_types) - 1}_types"
                        elif drained_count > 0:
                            cycle_reason = f"{cycle_reason}+{drained_count}_queued"
                        provider_pass = True
                    except asyncio.TimeoutError:
                        if last_open_positions > 0:
                            elapsed_provider_seconds = time.monotonic() - last_provider_pass_at
                            if elapsed_provider_seconds >= float(interval_seconds):
                                cycle_reason = "scheduled"
                                provider_pass = True
                            else:
                                cycle_reason = "position_tick"
                                provider_pass = False
                        else:
                            cycle_reason = "scheduled"
                            provider_pass = True

                try:
                    cycle_summary = await _graceful_timeout(
                        _run_reconciliation_cycle(
                            reason=cycle_reason,
                            emit_event=emit_event,
                            provider_pass=provider_pass,
                            heartbeat_activity="Reconciling",
                        ),
                        label="cycle:main",
                        timeout=_RECONCILIATION_CYCLE_TIMEOUT_SECONDS,
                    )
                except _TimedTaskStillRunningError:
                    logger.warning(
                        "Reconciliation cycle skipped because the prior timed-out cycle is still finishing reason=%s",
                        cycle_reason,
                    )
                    cycle_summary = _empty_cycle_summary()
                except asyncio.TimeoutError:
                    logger.warning(
                        "Reconciliation cycle timed out after %.0fs reason=%s",
                        _RECONCILIATION_CYCLE_TIMEOUT_SECONDS,
                        cycle_reason,
                    )
                    cycle_summary = _empty_cycle_summary()
                consecutive_db_failures = 0

                # Mandatory cooldown after each cycle to release DB pool pressure
                # and let the trader orchestrator process signals uncontested.
                cooldown_seconds = _post_cycle_cooldown_seconds(
                    reason=cycle_reason,
                    provider_pass=provider_pass,
                )
                if cooldown_seconds > 0.0:
                    await asyncio.sleep(cooldown_seconds)

                # Sync position marks and exit registry for event-driven updates.
                # Cycle 4 of the perf-harness loop showed a 360s heartbeat-stale
                # incident with the activity stuck on "Reconciled 9/9 traders" —
                # i.e. the cycle tail or next cycle's pre-trader phase hung past
                # the 180s watchdog.  Each unbounded await below is now wrapped
                # in a hard timeout so a single hung call (e.g. a stuck WS
                # subscribe in feed_manager) cannot trip the watchdog and force
                # a worker restart.  Per-call timeouts sum to <180s so the full
                # post-cycle phase always completes inside the watchdog budget.
                try:
                    await asyncio.wait_for(
                        _sync_position_marks_and_exit_registry(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "post-cycle: _sync_position_marks_and_exit_registry timed out after 30s"
                    )
                except Exception as exc:
                    logger.warning(
                        "post-cycle: _sync_position_marks_and_exit_registry failed",
                        exc_info=exc,
                    )
                # Refresh the live wallet position snapshot from the data API
                # so the in-memory ledger does not silently drift from chain
                # state when off-app activity touches the wallet.  Spawn as
                # a background task instead of awaiting — the function has
                # internal 30s throttling so it won't pile up, and a slow
                # Polymarket /positions response (saw 20s timeouts in prod)
                # must not block the reconciliation loop.
                _spawn_background(
                    _sync_live_wallet_positions(),
                    label="sync_live_wallet_positions",
                )
                # Cancel CLOB orders that are clearly abandoned (old + far
                # off-market or below the Polymarket per-order minimum so they
                # cannot fill).  Throttled internally to ~5 minutes.  The
                # function's internal call sites already wrap Polymarket I/O
                # in asyncio.wait_for, but wrap the whole thing in a
                # belt-and-suspenders 30s outer timeout so a missed inner
                # timeout cannot stall the cycle tail.
                try:
                    await asyncio.wait_for(_sweep_stale_open_orders(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "post-cycle: _sweep_stale_open_orders timed out after 30s"
                    )
                except Exception as exc:
                    logger.warning("stale-order sweep failed", exc_info=exc)
                # Surveillance pass: scan for orders in
                # blocked-terminal exit states whose underlying position
                # is verifiable on-chain.  Pure observation — never
                # writes to trader_orders.  Emits Telegram alerts for
                # operator-intervention rows; rows that the redeemer
                # or verifier will recover automatically are silent.
                # Throttled internally to once per scan interval.
                try:
                    await asyncio.wait_for(
                        _scan_stuck_positions_throttled(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "post-cycle: _scan_stuck_positions_throttled timed out after 60s"
                    )
                except Exception as exc:
                    logger.warning("stuck-position monitor failed", exc_info=exc)
                # Reconcile reported realized P&L against Polymarket truth.
                # Walks today's closed orders, fetches actual on-chain SELL
                # trades from the data API, FIFO-matches and writes the
                # verified P&L back to each row. Polymarket's wallet
                # trades are the SINGLE
                # source of truth for realized P&L (covers bot SELLs,
                # manual user sells via the Polymarket UI, and any other
                # on-chain exit that touches the proxy wallet).
                try:
                    await asyncio.wait_for(
                        _verify_realized_pnl_against_wallet_trades(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "post-cycle: _verify_realized_pnl_against_wallet_trades timed out after 60s"
                    )
                except Exception as exc:
                    logger.warning(
                        "post-cycle: _verify_realized_pnl_against_wallet_trades failed",
                        exc_info=exc,
                    )
                last_open_positions = int(cycle_summary.get("inventory_open_positions", 0) or 0)
                if provider_pass:
                    last_provider_pass_at = time.monotonic()

                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        WORKER_NAME,
                        running=True,
                        enabled=True,
                        current_activity=(
                            f"Reconciled {int(cycle_summary['traders_processed'])}/"
                            f"{int(cycle_summary['traders_seen'])} traders, "
                            f"provider_updates={int(cycle_summary['provider_updates'])}, "
                            f"positions_closed={int(cycle_summary['positions_closed'])}, "
                            f"provider_unavailable={int(cycle_summary['provider_unavailable_traders'])}"
                        ),
                        interval_seconds=interval_seconds,
                        last_run_at=utcnow(),
                        stats={
                            **cycle_summary,
                            "cycle_reason": cycle_reason,
                            "provider_pass": bool(provider_pass),
                            **_wallet_monitor_snapshot_stats(),
                        },
                    )
            except DBAPIError as exc:
                if not _is_retryable_db_error(exc):
                    raise
                consecutive_db_failures += 1
                logger.warning(
                    "Trader reconciliation cycle hit transient DB error (failure=%d)",
                    consecutive_db_failures,
                    exc_info=exc,
                )
                if consecutive_db_failures >= _MAX_CONSECUTIVE_DB_FAILURES:
                    await recover_pool()
                    consecutive_db_failures = 0
                    logger.warning("Recovered DB connection pool after trader reconciliation disconnects")
                control_refresh_at = 0.0
                wallet_monitor_refresh_at = 0.0
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
            except Exception as exc:
                logger.exception("Trader reconciliation worker cycle failed: %s", exc)
                try:
                    async with AsyncSessionLocal() as session:
                        await write_worker_snapshot(
                            session,
                            WORKER_NAME,
                            running=False,
                            enabled=True,
                            current_activity="Worker error",
                            interval_seconds=interval_seconds,
                            last_error=str(exc),
                        )
                except Exception:
                    pass
                control_refresh_at = 0.0
                wallet_monitor_refresh_at = 0.0
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
    finally:
        for event_type in sorted(_RECONCILE_TRIGGER_EVENTS):
            event_bus.unsubscribe(event_type, _on_runtime_event)
        # Stop the wallet-cache reseeder cleanly so it doesn't outlive
        # the worker process.  The signal is idempotent.
        wallet_cache_reseeder_stop.set()
        if not wallet_cache_reseeder_task.done():
            try:
                await asyncio.wait_for(wallet_cache_reseeder_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                wallet_cache_reseeder_task.cancel()
            except Exception:
                pass


async def start_loop() -> None:
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader reconciliation worker shutting down")


async def main() -> None:
    await init_database()
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader reconciliation worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
