from __future__ import annotations

import asyncio
import logging
import time as _time_mod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import bindparam, func, inspect, or_, select, text as _sa_text, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AsyncSessionLocal, LiveTradingOrder, LiveTradingPosition, TradeSignal, TraderOrder, release_conn
from services.polymarket import polymarket_client
from services.runtime_signal_queue import publish_signal_batch
from services.signal_bus import make_dedupe_key, upsert_trade_signal
from services.simulation import simulation_service
from services.strategy_sdk import StrategySDK
from services.live_execution_service import live_execution_service
from services.live_pressure import is_db_pressure_active
from services.trader_order_verification import (
    TRADER_ORDER_VERIFICATION_LOCAL,
    append_trader_order_verification_event,
    apply_trader_order_verification,
    derive_trader_order_verification,
)
from services.trader_orchestrator_state import _dedupe_live_authority_rows, _extract_copy_source_wallet_from_payload
from services.trader_orchestrator import exit_executor
from services.trader_orchestrator.hot_path import allow_polymarket_rest_call, is_hot_path_no_rest
from services.wallet_state_cache import get_wallet_state_cache
from utils.utcnow import utcnow
from utils.converters import safe_float
import services.trader_hot_state as hot_state

logger = logging.getLogger("position_lifecycle")

SHADOW_ACTIVE_STATUSES = {"submitted", "executed", "completed", "open", "pending", "placing", "queued"}
LIVE_ACTIVE_STATUSES = {"submitted", "executed", "completed", "open", "pending", "placing", "queued"}
_FAILED_EXIT_MAX_RETRIES = 5
_FAILED_EXIT_MIN_RETRY_INTERVAL_SECONDS = 15
# Absolute ceiling for the soft-bypass path.  Without this, an exit whose
# provider-side submit keeps timing out can retry indefinitely while the
# wallet position is still open -- the log has shown orders at attempt=67+.
# Past this many attempts the exit is marked blocked_retry_exhausted_hard
# and an operator must intervene.
_FAILED_EXIT_HARD_RETRY_CEILING = 100
# A short circuit specifically for ``asyncio.TimeoutError`` on the exit
# submission path.  A timeout means the CLOB call hung for the full
# ``_LIVE_EXIT_RETRY_TIMEOUT_SECONDS`` budget; each one chews ~10s of
# Python compute (cancellation tears asyncpg state, my retry fix then
# invalidates the connection) and the next reconcile cycle picks the
# same order back up.  N consecutive timeouts is a strong signal the
# path is fundamentally stuck (drifted ledger, dead provider session,
# or a market the CLOB simply will not accept) and continuing to retry
# is just burning CPU/connection state.  Reset the counter on any
# non-timeout outcome so transient timeouts don't poison healthy
# orders.
_FAILED_EXIT_PERSISTENT_TIMEOUT_THRESHOLD = 3

# How long a ``blocked_persistent_timeout`` entry must sit before we
# auto-reset it to ``failed`` and let the normal retry path try once
# more.  Three consecutive submission timeouts can be caused by a
# transient event-loop stall (heavy strategy bursts, GC, RPC hiccup);
# pinning the row in blocked-terminal forever and waking an operator
# is the wrong response when the underlying cause has cleared.  The
# hard retry ceiling (_FAILED_EXIT_HARD_RETRY_CEILING = 100) still
# protects against truly persistent failures — those eventually
# escalate to ``blocked_retry_exhausted_hard`` which still requires
# operator action.
# Cooldown for auto-recovering ``blocked_persistent_timeout`` rows.
#
# This was 30 minutes pre-incident; the long delay was a defensive
# response to a flaky environment.  After the SELL-timeout bump
# (5s→15s in commit a8ef524) and the DB-pressure cascade fix in commit
# 4a319cd, transient client-side failures clear in seconds rather than
# minutes — there is no reason to wait half an hour before retrying.
#
# The hard retry ceiling (``_FAILED_EXIT_HARD_RETRY_CEILING = 100``)
# still bounds total attempts, so a genuinely stuck market still
# escalates eventually; this just makes us notice it tradable again
# sooner once the venue / our infra recovers.
_BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_AFTER_SECONDS = 5 * 60
# After this many cumulative retries on a single order we no longer
# cycle it every 5 minutes; we apply exponential backoff up to a 1-hour
# ceiling.  Without this, two orders whose venue-side problem isn't
# going to resolve (e.g. ``balance: $X, sum of active orders: $X``
# rejection that needs operator attention to clear) eat ~30s of the
# 30s reconcile budget every 5 minutes — blocking every other order's
# retry path indefinitely.  The 2026-04-28 cascade had bed1b35d at
# attempt=13 and 76e3a73e at attempt=12 doing exactly this.
_BLOCKED_PERSISTENT_TIMEOUT_QUARANTINE_THRESHOLD_RETRIES = 10
_BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_MAX_SECONDS = 60 * 60

# ── Short-lived cache for wallet data to avoid redundant Polymarket API
# calls when multiple traders share the same execution wallet.
_WALLET_CACHE_TTL_SECONDS = 15.0
_WALLET_HISTORY_CACHE_TTL_SECONDS = 180.0
_WALLET_HISTORY_GRACE_SECONDS = 120.0
_WALLET_POSITIONS_LOAD_TIMEOUT_SECONDS = 6.0
_WALLET_HISTORY_LOAD_TIMEOUT_SECONDS = 6.0
_WALLET_HISTORY_MAX_CLOSED_POSITIONS = 300
_WALLET_HISTORY_MAX_TRADES = 300
_WALLET_HISTORY_TRADE_PAGE_SIZE = 200
_WALLET_HISTORY_MAX_ACTIVITY_ITEMS = 500
_WALLET_HISTORY_ACTIVITY_PAGE_SIZE = 200
_wallet_positions_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})
_wallet_positions_last_refresh_succeeded = False
_wallet_closed_positions_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})
_wallet_closed_positions_last_refresh_succeeded = False
_wallet_trades_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
_wallet_buy_trades_cache: tuple[float, dict[str, list[dict[str, Any]]]] = (0.0, {})
_wallet_sell_trades_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})
_wallet_activity_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})
_wallet_activity_last_refresh_succeeded = False

# 2026-05-10: single-flight locks for the wallet-history loaders.
# ``reconcile_live_positions`` runs an ``asyncio.gather`` of 4 of these
# loaders.  ``_load_execution_wallet_recent_buy_trades_by_token`` and
# ``_load_execution_wallet_recent_sell_trades_by_token`` BOTH call
# ``_load_execution_wallet_trade_history`` — so a single cache miss
# was firing TWO paginated Polymarket calls concurrently, plus the
# closed-positions and activity calls in parallel.  Soak measured
# ``external_io=12-20s`` against an API that responds in ms; the
# cause was thundering-herd on cache expiry, not API latency.
#
# The locks are stored per-running-loop so pytest's per-test loop
# rotation doesn't leave us holding a Lock bound to a closed loop
# (which would silently raise during the test that triggers the
# next acquire).  Production has exactly one loop for the worker
# process lifetime, so the dict only ever holds one entry there.
_wallet_locks_loop_id: Optional[int] = None
_wallet_locks: dict[str, asyncio.Lock] = {}


def _get_wallet_lock(name: str) -> asyncio.Lock:
    global _wallet_locks_loop_id, _wallet_locks
    loop_id = id(asyncio.get_running_loop())
    if loop_id != _wallet_locks_loop_id:
        # New event loop (typically a fresh pytest test) — abandon
        # the old locks, which were bound to a now-defunct loop.
        _wallet_locks_loop_id = loop_id
        _wallet_locks = {}
    lock = _wallet_locks.get(name)
    if lock is None:
        lock = asyncio.Lock()
        _wallet_locks[name] = lock
    return lock


def _get_wallet_closed_positions_lock() -> asyncio.Lock:
    return _get_wallet_lock("closed_positions")


def _get_wallet_trades_lock() -> asyncio.Lock:
    return _get_wallet_lock("trades")


def _get_wallet_buy_trades_lock() -> asyncio.Lock:
    return _get_wallet_lock("buy_trades")


def _get_wallet_sell_trades_lock() -> asyncio.Lock:
    return _get_wallet_lock("sell_trades")


def _get_wallet_activity_lock() -> asyncio.Lock:
    return _get_wallet_lock("activity")
_WALLET_SIZE_EPSILON = 1e-9
_MARK_TOUCH_INTERVAL_SECONDS = 0.5
_MAX_LIVE_EXIT_FALLBACK_MARK_AGE_SECONDS = 120.0
_POST_END_EXTREME_MARK_GRACE_SECONDS = 900.0
_TERMINAL_EXTREME_MARK_THRESHOLD = 0.001
# Per-attempt budget for SELL submissions on the slow path.  This is
# the OUTER ``asyncio.wait_for`` that wraps ``execute_live_order``;
# the SDK itself has ``_ORDER_SUBMIT_TIMEOUT_SECONDS=20s`` for the
# CLOB roundtrip, and ``execute_live_order`` adds:
#
#   * VPN check                     (~50ms healthy, can spike to 1s)
#   * pre-submit DB write           (~50ms healthy, seconds under load)
#   * sell allowance prep           (~200-500ms; on-chain check)
#   * SDK signature                 (~50ms)
#   * HTTP POST to Polymarket       (~200-800ms healthy)
#   * server-side processing        (~200-2000ms)
#   * post-submit DB write          (~50ms)
#
# Total realistic budget: 5-10s under healthy conditions.  The prior
# 5s value caused recurring false TimeoutErrors that the lifecycle
# treated as venue rejections, which the operator stuck-position
# alert then surfaced as "needs operator review".  The 2026-04-28
# incident's bulk writeoff fired against rows whose ONLY problem was
# this too-tight outer timeout.
#
# Outer ``asyncio.wait_for`` budget around ``execute_live_order`` for
# entry submissions.  This MUST exceed the sum of the SDK's per-call
# timeouts (``_ORDER_SUBMIT_TIMEOUT_SECONDS`` × 2 = 16s) plus a small
# buffer for the ``release_conn`` + post-result book-keeping; if it
# is tighter the outer ``wait_for`` will cancel the SDK
# ``asyncio.to_thread`` worker mid-protocol.  Cancelling
# ``to_thread`` is best-effort in CPython (the thread keeps running
# in the background and may complete the half-submitted request),
# which is exactly the pathology that produced the 2026-04-28 ghost
# orders / asyncpg state corruption cascade.
#
# Sized to fit the 30s per-trader reconcile cycle:
#   prep (6s) + outer (20s) + bookkeeping (~3s) ≈ 29s.
_LIVE_EXIT_ORDER_TIMEOUT_SECONDS = 20.0
# Same constraint for the retry path.  Was 12.0 — strictly less than
# the SDK's prior 20.0 timeout, so every retry that hit Polymarket
# gateway latency cancelled the SDK call and produced false
# TimeoutErrors that bumped ``consecutive_blocked_failure_count`` and
# eventually marked the row ``blocked_persistent_timeout`` even
# though the venue was healthy.  Bumped to 20.0 in lockstep with
# the SDK timeout drop to 8.0 so the inner can complete cleanly.
_LIVE_EXIT_RETRY_TIMEOUT_SECONDS = 20.0
# Cap the number of exits we submit per reconcile pass.  With each
# attempt now budgeting up to ~29s (6s prep + 20s outer + bookkeeping),
# a 2-per-pass cap blows the reconcile worker's 30s per-trader
# timeout.  Lowered to 1 always — the deferred work picks up on the
# next reconcile cycle anyway, and 1-per-pass matches the previous
# "under DB pressure" path (which never regressed throughput in
# production).
_LIVE_EXIT_MAX_SUBMISSIONS_PER_PASS = 1
_LIVE_EXIT_PRESSURE_MAX_SUBMISSIONS_PER_PASS = 1
# Bound on ``prepare_sell_balance_allowance`` (chain RPC, up to 3
# signature types tried sequentially).  Pre-incident this was
# unbounded and could happily run for 15-25s before the outer
# ``wait_for`` even started its clock.  Cap at 6s so a degraded
# Polygon RPC can't blow the per-attempt budget; the allowance
# refresh isn't required for correctness — failure here just means
# the SDK falls through to the existing on-the-fly approval path.
_LIVE_EXIT_PREP_ALLOWANCE_TIMEOUT_SECONDS = 6.0
_MARKET_INFO_LOAD_TIMEOUT_SECONDS = 2.5
_ORDER_SNAPSHOT_LOAD_TIMEOUT_SECONDS = 2.0
_RECONCILE_TIMING_WARN_SECONDS = 20.0
_TERMINAL_REOPEN_LOOKBACK_HOURS = 72.0
_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS = 72.0
_TERMINAL_REOPEN_AUDIT_COOLDOWN_SECONDS = 1800.0
_TERMINAL_REOPEN_AUDIT_MAX_ROWS_PER_PASS = 32
_STALE_PENDING_EXIT_WALLET_ABSENT_SECONDS = 600.0
_LIVE_FALLBACK_MANUAL_EXIT_REASONS = {
    "circuit_breaker_safe_exit",
    "terminal_market_watchdog",
}
_LIVE_FALLBACK_MANUAL_EXIT_TRIGGERS = {
    "manual_mark_to_market",
    "grouped:manual_mark_to_market",
}


def _wallet_position_outcome_index(outcome: Any) -> Optional[int]:
    outcome_text = str(outcome or "").strip().lower()
    if outcome_text in {"yes", "buy_yes"}:
        return 0
    if outcome_text in {"no", "buy_no"}:
        return 1
    return None


def _live_trading_position_to_wallet_position(row: LiveTradingPosition) -> Optional[dict[str, Any]]:
    token_id = str(row.token_id or "").strip()
    if not token_id:
        return None
    size = max(0.0, safe_float(row.size, 0.0) or 0.0)
    if size <= 0.0:
        return None
    market_id = str(row.market_id or "").strip() or token_id
    average_cost = safe_float(row.average_cost, None)
    current_price = safe_float(row.current_price, None)
    initial_value = (
        float(size * average_cost)
        if average_cost is not None and average_cost > 0.0
        else None
    )
    # Fix LL: ``current_price`` of 0.0 is a VALID post-resolution
    # settlement value (the losing side of a binary market), not a
    # missing-data sentinel.  Nullifying it here masked losers from
    # the downstream settlement-price extraction, which then fell
    # into a fallback heuristic that treated "no price" as "won at
    # $1" — flipping every loser into a phantom win.  Preserve 0.0
    # explicitly; only treat strictly-negative values as missing.
    current_value = (
        float(size * current_price)
        if current_price is not None and current_price >= 0.0
        else None
    )
    outcome_text = str(row.outcome or "").strip()
    outcome_index = _wallet_position_outcome_index(outcome_text)
    payload: dict[str, Any] = {
        "asset": token_id,
        "asset_id": token_id,
        "token_id": token_id,
        "tokenId": token_id,
        "conditionId": market_id,
        "condition_id": market_id,
        "market": market_id,
        "market_id": market_id,
        "marketId": market_id,
        "title": str(row.market_question or "").strip(),
        "market_question": str(row.market_question or "").strip(),
        "question": str(row.market_question or "").strip(),
        "outcome": outcome_text,
        "position_side": outcome_text.lower(),
        "side": outcome_text.lower(),
        "size": float(size),
        "positionSize": float(size),
        "shares": float(size),
        "avgPrice": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "average_price": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "avgCost": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "average_cost": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "currentPrice": float(current_price) if current_price is not None and current_price >= 0.0 else None,
        "current_price": float(current_price) if current_price is not None and current_price >= 0.0 else None,
        "curPrice": float(current_price) if current_price is not None and current_price >= 0.0 else None,
        "cur_price": float(current_price) if current_price is not None and current_price >= 0.0 else None,
        "markPrice": float(current_price) if current_price is not None and current_price >= 0.0 else None,
        "mark_price": float(current_price) if current_price is not None and current_price >= 0.0 else None,
        "initialValue": initial_value,
        "currentValue": current_value,
        "unrealizedPnl": safe_float(row.unrealized_pnl, None),
        "cashPnl": safe_float(row.unrealized_pnl, None),
        "redeemable": bool(getattr(row, "redeemable", False)),
        "counts_as_open": _safe_bool(getattr(row, "counts_as_open", True), True),
        "endDate": str(getattr(row, "end_date", "") or "").strip() or None,
        "end_date": str(getattr(row, "end_date", "") or "").strip() or None,
    }
    if outcome_index is not None:
        payload["outcomeIndex"] = outcome_index
        payload["outcome_index"] = outcome_index
    return payload


def _iso_utc(value: datetime) -> str:
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mark_touch_interval_seconds(params: dict[str, Any], *, mode: str) -> float:
    mode_key = str(mode or "").strip().lower()
    aliases: tuple[str, ...]
    if mode_key == "shadow":
        aliases = ("shadow_mark_touch_interval_seconds", "mark_touch_interval_seconds")
    else:
        aliases = ("live_mark_touch_interval_seconds", "mark_touch_interval_seconds")
    for key in aliases:
        parsed = safe_float(params.get(key))
        if parsed is None:
            continue
        return max(0.05, min(5.0, float(parsed)))
    return _MARK_TOUCH_INTERVAL_SECONDS


def _live_exit_submission_cap() -> int:
    if is_db_pressure_active():
        return _LIVE_EXIT_PRESSURE_MAX_SUBMISSIONS_PER_PASS
    return _LIVE_EXIT_MAX_SUBMISSIONS_PER_PASS


def _terminal_reopen_audit_recent(payload: dict[str, Any], now_naive: datetime) -> bool:
    marker = payload.get("terminal_reopen_audit")
    if not isinstance(marker, dict):
        return False
    audited_at = _parse_iso_utc_naive(marker.get("audited_at"))
    if audited_at is None:
        return False
    return (now_naive - audited_at).total_seconds() < _TERMINAL_REOPEN_AUDIT_COOLDOWN_SECONDS


def _pending_exit_stale_without_provider(pending_exit: Any, now_naive: datetime) -> bool:
    if not isinstance(pending_exit, dict):
        return False
    status = str(pending_exit.get("status") or "").strip().lower()
    if status not in {"pending", "submitted", "failed", "blocked_min_notional"}:
        return False
    if _pending_exit_provider_clob_id(pending_exit):
        return False
    retry_count = int(safe_float(pending_exit.get("retry_count"), 0.0) or 0)
    if retry_count >= _FAILED_EXIT_MAX_RETRIES:
        return True
    last_attempt = _parse_iso_utc_naive(pending_exit.get("last_attempt_at") or pending_exit.get("triggered_at"))
    if last_attempt is None:
        return False
    return (now_naive - last_attempt).total_seconds() >= _STALE_PENDING_EXIT_WALLET_ABSENT_SECONDS


def _mark_terminal_reopen_audited(row: TraderOrder, payload: dict[str, Any], now: datetime, reason: str) -> None:
    payload["terminal_reopen_audit"] = {
        "audited_at": _iso_utc(now),
        "status": str(getattr(row, "status", "") or ""),
        "reason": str(reason or "no_reopen"),
    }
    row.payload_json = payload
    row.updated_at = now


def _terminal_row_requires_reopen_audit(row: TraderOrder, now_naive: datetime) -> bool:
    payload = dict(getattr(row, "payload_json", None) or {})
    position_close = payload.get("position_close")
    pending_exit = payload.get("pending_live_exit")
    status_key = str(getattr(row, "status", None) or "").strip().lower()
    close_trigger = ""
    close_price_source = ""
    if isinstance(position_close, dict):
        close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
        close_price_source = str(position_close.get("price_source") or "").strip().lower()
    pending_status = ""
    if isinstance(pending_exit, dict):
        pending_status = str(pending_exit.get("status") or "").strip().lower()
    age_anchor = getattr(row, "updated_at", None) or getattr(row, "executed_at", None) or getattr(row, "created_at", None)
    if not isinstance(age_anchor, datetime):
        return False
    if age_anchor.tzinfo is not None:
        age_anchor = age_anchor.astimezone(timezone.utc).replace(tzinfo=None)
    if _terminal_row_needs_wallet_fill_repair_audit(row, now_naive):
        return True
    if _terminal_reopen_audit_recent(payload, now_naive):
        return False
    if isinstance(position_close, dict) and (
        _position_close_underfills_entry(row, payload, position_close)
        or close_trigger in {"wallet_activity", "wallet_summary_recovery", "external_wallet_flatten"}
        or close_price_source in {"wallet_activity", "wallet_trade"}
    ):
        return age_anchor >= (now_naive - timedelta(hours=_TERMINAL_REOPEN_LOOKBACK_HOURS))
    if (
        close_trigger in {"wallet_absent_close", "wallet_flat_override"}
        or pending_status == "filled"
        or close_price_source in {"provider_exit_fill", "resolved_settlement"}
    ):
        return age_anchor >= (now_naive - timedelta(hours=_TERMINAL_REOPEN_LOOKBACK_HOURS))
    verification_status = str(getattr(row, "verification_status", None) or "").strip().lower()
    if status_key in {"resolved", "resolved_win", "resolved_loss", "closed_win", "closed_loss"}:
        if verification_status in {"venue_order", "venue_fill", "wallet_position"}:
            return age_anchor >= (now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS))
        if _provider_reconciliation_has_order_evidence(row, payload):
            return age_anchor >= (now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS))
        return False
    if status_key not in {"cancelled", "failed", "rejected", "error"}:
        return False
    if verification_status in {"venue_order", "venue_fill", "wallet_position"}:
        return age_anchor >= (now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS))
    if str(getattr(row, "provider_clob_order_id", None) or "").strip():
        return age_anchor >= (now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS))
    if str(getattr(row, "provider_order_id", None) or "").strip():
        return age_anchor >= (now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS))
    return _provider_reconciliation_has_order_evidence(row, payload) and age_anchor >= (
        now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS)
    )


def _terminal_row_needs_wallet_fill_repair_audit(row: TraderOrder, now_naive: datetime) -> bool:
    payload = dict(getattr(row, "payload_json", None) or {})
    entry_fill_recovery = payload.get("entry_fill_recovery")
    if (
        isinstance(entry_fill_recovery, dict)
        and str(entry_fill_recovery.get("source") or "").strip().lower() == "wallet_trade_history"
    ):
        return False
    position_close = payload.get("position_close")
    if not isinstance(position_close, dict):
        return False
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if close_trigger not in {"wallet_activity", "wallet_summary_recovery", "external_wallet_flatten"}:
        return False
    if not _row_can_recover_entry_from_wallet_history(row, payload):
        return False
    filled_notional, filled_size, _ = _extract_live_fill_metrics(payload)
    if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
        return False
    age_anchor = getattr(row, "updated_at", None) or getattr(row, "executed_at", None) or getattr(row, "created_at", None)
    if not isinstance(age_anchor, datetime):
        return False
    if age_anchor.tzinfo is not None:
        age_anchor = age_anchor.astimezone(timezone.utc).replace(tzinfo=None)
    return age_anchor >= (now_naive - timedelta(hours=_NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS))


async def _strategy_exit_instance(session: AsyncSession, strategy_slug: str) -> Any | None:
    slug = str(strategy_slug or "").strip().lower()
    if not slug:
        return None
    from services.strategy_loader import strategy_loader

    loaded = strategy_loader.get_strategy(slug)
    if loaded is None:
        try:
            await strategy_loader.reload_from_db(slug, session)
        except Exception as exc:
            logger.debug("Failed lazy strategy reload for exit decision (%s): %s", slug, exc)
        loaded = strategy_loader.get_strategy(slug)
    if loaded is None:
        return None
    instance = getattr(loaded, "instance", None)
    if instance is None or not hasattr(instance, "should_exit"):
        return None
    return instance


async def _publish_trader_order_updates(rows: list[TraderOrder]) -> None:
    if not rows:
        return
    from services.event_bus import event_bus
    from services.trader_orchestrator_state import _serialize_order

    seen: set[str] = set()
    for row in rows:
        row_id = str(getattr(row, "id", "") or "").strip()
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        try:
            await event_bus.publish("trader_order", _serialize_order(row))
        except Exception:
            continue


def _failed_exit_retry_delay_seconds(last_error: Any) -> int:
    error_text = str(last_error or "").strip().lower()
    if "status_code=429" in error_text or "error 1015" in error_text or "too many requests" in error_text:
        return 90
    if "status_code=425" in error_text or "service not ready" in error_text:
        return 45
    if "invalid signature" in error_text:
        return 120
    if "not enough balance / allowance" in error_text or "allowance" in error_text:
        return 8
    if (
        "vpn check failed" in error_text
        or "trading proxy unreachable" in error_text
        or "invalid username/password" in error_text
    ):
        return 90
    if (
        "below minimum" in error_text
        or "lower than minimum" in error_text
        or "exit_notional_below_min" in error_text
    ):
        return 20
    if "missing token_id or fill_size" in error_text:
        return 30
    return _FAILED_EXIT_MIN_RETRY_INTERVAL_SECONDS


def _is_allowance_error(error: Any) -> bool:
    error_text = str(error or "").strip().lower()
    return "not enough balance / allowance" in error_text or "allowance" in error_text


def _is_zero_balance_error(error: Any) -> bool:
    error_text = str(error or "").strip().lower()
    return "available_shares=0" in error_text or "balance_shares=0" in error_text


def _is_rate_limited_error(error: Any) -> bool:
    error_text = str(error or "").strip().lower()
    return "status_code=429" in error_text or "error 1015" in error_text or "too many requests" in error_text


def _is_invalid_signature_error(error: Any) -> bool:
    error_text = str(error or "").strip().lower()
    return "invalid signature" in error_text


def _bump_allowance_error_counter(pending_exit: dict[str, Any], error: Any) -> None:
    if _is_allowance_error(error):
        pending_exit["allowance_error_count"] = int(pending_exit.get("allowance_error_count", 0) or 0) + 1
        return
    pending_exit["allowance_error_count"] = 0


def _format_exit_error(error: Any) -> str:
    """Produce a human-readable error string for an exit retry failure.

    ``str(error or "unknown")`` is wrong when error is a truthy exception
    with an empty message (e.g. ``asyncio.TimeoutError()``) -- it renders
    as "" and we lose all signal about why the retry failed.  Always return
    at least the exception class name.
    """
    if error is None:
        return "unknown"
    if isinstance(error, BaseException):
        cls = type(error).__name__
        msg = str(error).strip()
        return f"{cls}: {msg}" if msg else cls
    text = str(error).strip()
    return text or "unknown"


def _is_orderbook_gone_error(error_text: str) -> bool:
    """``the orderbook <token_id> does not exist`` is what Polymarket
    returns when the market has been resolved/removed and the CLOB no
    longer accepts orders for it.  No retry will ever succeed — the
    market is gone.  The position must be settled via the on-chain
    redemption path (handled by ``redeemer_worker``), not via CLOB.
    """
    text = (error_text or "").lower()
    return "the orderbook" in text and "does not exist" in text


def _is_fak_no_match_error(error_text: str) -> bool:
    """``no orders found to match with FAK order`` means the orderbook
    has no liquidity at the requested price.  Single occurrences are
    recoverable (the book changes), but a sustained streak is a strong
    signal that the bot's price ladder is permanently off-market for
    this side of this token — retrying just burns 10s/attempt without
    progress until the 100-attempt hard ceiling.  Treat as
    "blocked-class" so the 3-strike quarantine fires.
    """
    text = (error_text or "").lower()
    return "no orders found to match with fak order" in text


def _apply_failed_exit_state(pending_exit: dict[str, Any], *, error: Any, now: datetime, retry_count: int) -> None:
    error_text = _format_exit_error(error)
    pending_exit["last_error"] = error_text
    pending_exit["last_attempt_at"] = _iso_utc(now)
    if _is_zero_balance_error(error_text):
        pending_exit["status"] = "blocked_no_inventory"
        pending_exit["retry_count"] = retry_count
        pending_exit["exhausted_at"] = _iso_utc(now)
        pending_exit["next_retry_at"] = None
        pending_exit["consecutive_timeout_count"] = 0
        pending_exit["consecutive_blocked_failure_count"] = 0
        return
    # Hard-terminal: market's orderbook is gone (resolved/removed).
    # No CLOB retry can succeed — the position settles on-chain
    # via the redeemer.  Stop the retry loop immediately.  Without
    # this, we observed exit retries spinning at 10-18s of CPU per
    # cycle, starving the entire orchestrator (cascading into fast-
    # trader db_list_signals slowness, WS pong timeouts, etc.).
    if _is_orderbook_gone_error(error_text):
        pending_exit["status"] = "blocked_orderbook_gone"
        pending_exit["retry_count"] = retry_count
        pending_exit["exhausted_at"] = _iso_utc(now)
        pending_exit["next_retry_at"] = None
        pending_exit["consecutive_timeout_count"] = 0
        pending_exit["consecutive_blocked_failure_count"] = 0
        return
    is_timeout = isinstance(error, (asyncio.TimeoutError, TimeoutError))
    # A "blocked-class" failure is one we know retrying with the same
    # exit_size cannot fix on its own: a CLOB submission timeout, OR
    # a balance/allowance rejection (V1 returns the format
    # ``not enough balance / allowance: the balance is not enough ->
    # balance: X, order amount: Y``; V2 returns a bare
    # ``not enough balance / allowance`` with no numbers because
    # pUSD/V2 approvals were never set up).  Both shapes mean the
    # next retry will fail the same way until something external
    # (operator, sync_positions reset, V2 migration completion)
    # changes the wallet state.  Track the streak across both kinds
    # so alternating timeouts and balance errors don't reset the
    # counter and let the order spin forever.
    #
    # FAK ``no orders found to match`` is added to the blocked-class
    # bucket: each FAK rejection costs ~10s on the asyncio.wait_for
    # budget, and a sustained streak (>3) means our price ladder is
    # off-market.  Letting it grind to attempt 100 burns ~17 minutes
    # of orchestrator time; the 3-strike quarantine drops it to ~30s
    # before the row enters ``blocked_persistent_timeout`` and only
    # auto-retries on the 5-minute cooldown.
    error_text_lower = (error_text or "").lower()
    is_balance_failure = "not enough balance" in error_text_lower or "not enough balance / allowance" in error_text_lower
    is_fak_no_match = _is_fak_no_match_error(error_text_lower)
    is_blocked_class = is_timeout or is_balance_failure or is_fak_no_match
    if is_blocked_class:
        blocked_streak = int(pending_exit.get("consecutive_blocked_failure_count") or 0) + 1
        pending_exit["consecutive_blocked_failure_count"] = blocked_streak
        # Keep the legacy timeout counter in sync for callers that
        # still read it.
        if is_timeout:
            pending_exit["consecutive_timeout_count"] = (
                int(pending_exit.get("consecutive_timeout_count") or 0) + 1
            )
        if blocked_streak >= _FAILED_EXIT_PERSISTENT_TIMEOUT_THRESHOLD:
            # Stop pumping reconcile cycles into an exit that can't
            # currently succeed.  Each attempt burns ~10s on the
            # asyncio.wait_for budget, cancels asyncpg mid-protocol,
            # and forces a connection invalidation; retrying again
            # will do the same thing.  An operator (or a successful
            # periodic ``sync_positions`` reset, or completion of
            # the V2 wallet migration) can clear this state.
            pending_exit["status"] = "blocked_persistent_timeout"
            pending_exit["retry_count"] = retry_count
            pending_exit["exhausted_at"] = _iso_utc(now)
            pending_exit["next_retry_at"] = None
            return
    else:
        pending_exit["consecutive_timeout_count"] = 0
        pending_exit["consecutive_blocked_failure_count"] = 0
    pending_exit["status"] = "failed"
    pending_exit["retry_count"] = retry_count
    _bump_allowance_error_counter(pending_exit, error_text)
    pending_exit["next_retry_at"] = _iso_utc(now + timedelta(seconds=_failed_exit_retry_delay_seconds(error_text)))


def _persistent_timeout_auto_retry_seconds(retry_count: int) -> float:
    """Cooldown for the next auto-recovery attempt of a
    ``blocked_persistent_timeout`` row.

    Up to ``_BLOCKED_PERSISTENT_TIMEOUT_QUARANTINE_THRESHOLD_RETRIES``
    cumulative retries we use the base 5-minute cadence — that's the
    "transient venue/infra blip" recovery path.

    Past the threshold we double the cooldown for each additional
    retry, capped at
    ``_BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_MAX_SECONDS`` (1 hour).
    The progression for retry_count = 10..14 is:

        10 → 5 min
        11 → 10 min
        12 → 20 min
        13 → 40 min
        14+→ 60 min

    This quarantines orders whose venue-side problem isn't going to
    fix itself within minutes (balance/allowance mismatches, condition
    deactivations, etc.) so they don't burn the reconcile worker's
    30s per-trader budget every 5 minutes; an operator who comes to
    look will still find them tradable, just on hourly cadence.
    """
    base = float(_BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_AFTER_SECONDS)
    threshold = int(_BLOCKED_PERSISTENT_TIMEOUT_QUARANTINE_THRESHOLD_RETRIES)
    cap = float(_BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_MAX_SECONDS)
    if retry_count <= threshold:
        return base
    excess = max(0, retry_count - threshold)
    # 2** is fast even for huge retry_count because we cap; clamp
    # excess at log2(cap/base) + 1 to avoid pathological exponents.
    bounded_excess = min(excess, 20)
    return min(cap, base * (2 ** bounded_excess))


def _maybe_clear_persistent_timeout(pending_exit: dict[str, Any], now: datetime) -> bool:
    """Auto-recover a ``blocked_persistent_timeout`` row whose cooldown elapsed.

    Three consecutive ``asyncio.TimeoutError`` on exit submission can be
    caused by transient event-loop saturation (heavy strategy bursts,
    GC, RPC slowness) — once that clears, the same exit will usually
    submit cleanly.  Pinning the row forever forces an unnecessary
    operator alert and leaves real money sitting idle.

    On reset we flip the status back to ``"failed"`` so the normal
    failed-exit retry path picks it up on the next reconcile cycle.
    The retry counter is preserved (so the hard ceiling still binds);
    the consecutive-streak counters are zeroed so transient timeouts
    don't immediately re-latch the blocked state on the first new
    failure.

    The cooldown applied here is exponential past
    ``_BLOCKED_PERSISTENT_TIMEOUT_QUARANTINE_THRESHOLD_RETRIES`` so a
    single doomed order doesn't dominate every 5-minute reconcile
    cycle indefinitely.  See
    ``_persistent_timeout_auto_retry_seconds``.

    Returns True iff the row was reset.
    """
    if str(pending_exit.get("status") or "").strip().lower() != "blocked_persistent_timeout":
        return False
    exhausted_at = _parse_iso_utc_naive(pending_exit.get("exhausted_at"))
    if exhausted_at is None:
        # No exhaustion timestamp — fall back to ``last_attempt_at``.
        exhausted_at = _parse_iso_utc_naive(pending_exit.get("last_attempt_at"))
    if exhausted_at is None:
        return False
    now_naive = now.replace(tzinfo=None) if now.tzinfo is not None else now
    elapsed_seconds = (now_naive - exhausted_at).total_seconds()
    retry_count = int(pending_exit.get("retry_count") or 0)
    cooldown = _persistent_timeout_auto_retry_seconds(retry_count)
    if elapsed_seconds < cooldown:
        return False
    prior_error = str(pending_exit.get("last_error") or "").strip() or "unknown"
    pending_exit["status"] = "failed"
    pending_exit["consecutive_blocked_failure_count"] = 0
    pending_exit["consecutive_timeout_count"] = 0
    pending_exit["next_retry_at"] = _iso_utc(now)
    pending_exit["last_error"] = (
        f"persistent_timeout_auto_recovery_after_{int(elapsed_seconds)}s "
        f"(cooldown_was={int(cooldown)}s, retry_count={retry_count}, "
        f"prior: {prior_error[:80]})"
    )
    return True


def _take_profit_price_floor(pending_exit: dict[str, Any]) -> Optional[float]:
    """Compute the minimum price for a take-profit-limit exit.

    Returns None if the TP should be abandoned (market moved against us
    below entry price — caller should convert to a market sell).
    """
    kind = str(pending_exit.get("kind") or "").strip().lower()
    if kind != "take_profit_limit":
        return None
    close_price = safe_float(pending_exit.get("close_price"))
    entry_fill_price = safe_float(pending_exit.get("entry_fill_price"))

    current_price = safe_float(pending_exit.get("current_mid_price"))
    if current_price is not None and entry_fill_price is not None and entry_fill_price > 0.0:
        if current_price < entry_fill_price * 0.995:
            return None

    target_take_profit_pct = max(0.0, safe_float(pending_exit.get("target_take_profit_pct"), 0.0) or 0.0)
    fee_buffer_pct = max(0.0, safe_float(pending_exit.get("take_profit_fee_buffer_pct"), 0.35) or 0.35)
    floor_candidates: list[float] = []
    if close_price is not None and close_price > 0.0:
        floor_candidates.append(close_price)
    if entry_fill_price is not None and entry_fill_price > 0.0:
        floor_candidates.append(entry_fill_price * (1.0 + (target_take_profit_pct / 100.0)))
        floor_candidates.append(entry_fill_price * (1.0 + (fee_buffer_pct / 100.0)))
    if not floor_candidates:
        return None
    return min(0.999, max(0.01, max(floor_candidates)))


def _aggressive_exit_price_decrement(retry_count: int) -> float:
    """Cumulative decrement (in 0–1 binary units) to subtract from a
    SELL limit price after ``retry_count`` failures.

    Progression:

      retry 0 → 0.00   (original price)
      retry 1 → 0.01   (1 ¢ off)
      retry 2 → 0.02
      retry 3 → 0.05
      retry 4 → 0.05
      retry 5 → 0.06
      retry 6 → 0.07
      …
      retry ≥ 9 → 0.10 (cap)

    Tuned so a healthy market clears at the original price, a slightly
    thin market clears within 1–3 retries, and a persistently illiquid
    market eventually walks down enough to accept what is there
    instead of spinning at the same rejected limit forever.  Capped at
    10 ¢ to bound the give-away on a non-take-profit exit; if the
    market truly doesn't pay 10 ¢ less than our target, the retry
    ceiling will eventually escalate it.

    Caller MUST skip this when ``_take_profit_price_floor`` returns a
    non-None value — for take-profit exits the limit price is set
    from the gain target, not from market liquidity, so walking it
    down is just leaving money on the table.
    """
    if retry_count <= 0:
        return 0.0
    if retry_count == 1:
        return 0.01
    if retry_count == 2:
        return 0.02
    if retry_count <= 4:
        return 0.05
    return min(0.10, 0.05 + 0.01 * (retry_count - 4))


async def _prepare_sell_allowance_bounded(token_id: str) -> bool:
    """Wrap ``live_execution_service.prepare_sell_balance_allowance``
    in an ``asyncio.wait_for`` so a degraded Polygon RPC can't run
    unbounded before the SDK call's outer timeout starts its clock.

    The underlying allowance refresh tries up to 3 signature types
    sequentially with chain RPC reads in each — pre-incident this was
    completely unbounded and observed at 15-25s per token in the
    user-visible cascade.

    Failure here is non-fatal: the SDK call has its own on-the-fly
    approval fall-through, so a swallowed exception (timeout or
    otherwise) just means the SELL submission picks up the slack.
    """
    if not token_id:
        return False
    # 2026-05-10: surface long-but-not-timed-out calls so the next
    # capture can attribute ``reconcile_live_positions processing=10s``
    # to specific tokens.  This function is called 5+ places in the
    # per-candidate loop and goes through ``_client_io_lock`` (the
    # documented "30+ s offender" in ``live_execution_service``); a
    # single stuck candidate making 3-4 of these calls under SDK lock
    # contention can stack into the 7-10s per-iteration slowness
    # observed for ``dcec142b`` in the 2026-05-10 00:48-00:59 capture.
    _started_mono = _time_mod.monotonic()
    try:
        return bool(
            await asyncio.wait_for(
                live_execution_service.prepare_sell_balance_allowance(token_id),
                timeout=_LIVE_EXIT_PREP_ALLOWANCE_TIMEOUT_SECONDS,
            )
        )
    except asyncio.TimeoutError:
        logger.warning(
            "prepare_sell_balance_allowance bounded timeout exceeded "
            "(token_id=%s, budget=%.1fs) — proceeding to SDK without refresh",
            token_id,
            _LIVE_EXIT_PREP_ALLOWANCE_TIMEOUT_SECONDS,
        )
        return False
    except Exception:
        return False
    finally:
        _elapsed = _time_mod.monotonic() - _started_mono
        if _elapsed >= 2.0:
            logger.warning(
                "prepare_sell_balance_allowance slow (token_id=%s, elapsed=%.2fs)"
                " — likely _client_io_lock contention; check ps_submit_order /"
                " reconcile_live_positions processing slowest-candidate logs",
                token_id,
                _elapsed,
            )


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


# ----------------------------------------------------------------------------
# StrategySDK: partial-exit (action="reduce") consumer plumbing
# ----------------------------------------------------------------------------
# ``ExitDecision(action="reduce", reduce_fraction=X)`` is part of the public
# strategy SDK contract — it lets a strategy ask for a partial close (e.g. a
# scale-out ladder). Historically the live + shadow lifecycle sites treated
# ``reduce`` as a no-op hold, which made the contract live only in the
# backtester. The helpers below give both sites a generic, strategy-agnostic
# way to honor a reduce decision: submit a partial sell sized by the
# fraction, persist a ``partial_exit_history`` audit entry, and leave the
# row's status as-is so subsequent ticks continue to manage the remainder.


def _coerce_reduce_fraction(decision: Any) -> Optional[float]:
    """Validate ``reduce_fraction`` from an ExitDecision.

    Returns the clamped fraction in (0, 1) when the decision is a reduce
    with a usable fraction. Returns ``None`` when the decision is not a
    reduce, the fraction is missing, malformed, or outside (0, 1) — the
    caller treats that as "fall back to a full close" so a misconfigured
    strategy can never silently no-op.
    """
    if decision is None or getattr(decision, "action", None) != "reduce":
        return None
    raw = getattr(decision, "reduce_fraction", None)
    try:
        frac = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    if frac is None or not (0.0 < frac < 1.0):
        return None
    return frac


def _record_partial_exit(
    payload: dict[str, Any],
    *,
    reason: str,
    reduce_fraction: float,
    exit_size: float,
    close_price: Optional[float],
    price_source: Optional[str],
    now: datetime,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Append a partial-exit audit entry. Bounded history (last 16 entries)."""
    history = payload.get("partial_exit_history")
    if not isinstance(history, list):
        history = []
    entry: dict[str, Any] = {
        "submitted_at": _iso_utc(now),
        "reason": reason,
        "reduce_fraction": float(reduce_fraction),
        "exit_size": float(exit_size),
        "close_price": float(close_price) if close_price is not None else None,
        "price_source": price_source,
    }
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key not in entry:
                entry[key] = value
    history.append(entry)
    payload["partial_exit_history"] = history[-16:]


async def _submit_live_partial_exit(
    *,
    session: AsyncSession,
    row: TraderOrder,
    payload: dict[str, Any],
    decision: Any,
    close_price: Optional[float],
    price_source: Optional[str],
    filled_size: float,
    notional_usd: float,
    entry_price: float,
    params: dict[str, Any],
    now: datetime,
) -> tuple[bool, Optional[str]]:
    """Submit a partial sell sized by ``decision.reduce_fraction``.

    Returns ``(submitted, error_message)``. The caller is responsible for
    NOT flipping ``row.status`` — the position remains open and the next
    provider reconciliation cycle picks up the smaller residual size.

    Generic across strategies: this consumer doesn't know about scale-out
    tiers; it just honors the SDK's reduce primitive.
    """
    fraction = _coerce_reduce_fraction(decision)
    if fraction is None:
        return False, "invalid_reduce_fraction"

    token_id = _extract_live_token_id(payload)
    if not token_id:
        return False, "missing_token_id"

    base_size = filled_size if filled_size > 0.0 else (
        notional_usd / entry_price if entry_price > 0 else 0.0
    )
    exit_size = base_size * fraction
    if exit_size <= _WALLET_SIZE_EPSILON:
        return False, "partial_size_below_epsilon"

    base_min_order_size_usd = _resolve_position_min_order_size_usd(
        trader_params=params,
        payload=payload,
        mode="live",
    )
    min_order_size_usd = _effective_exit_min_order_size_usd(
        base_min_order_size_usd,
        "strategy_partial_exit",
    )
    exit_notional_estimate = exit_size * float(max(close_price or 0.0, 0.0))
    if exit_notional_estimate + 1e-9 < min_order_size_usd:
        # Don't over-fragment: if this slice can't clear the venue floor,
        # bail and let the strategy escalate to a full close on a later tick.
        return False, "partial_below_min_notional"

    reason = str(getattr(decision, "reason", "") or "strategy_reduce")
    try:
        from services.live_execution_adapter import execute_live_order

        async with release_conn(session):
            await _prepare_sell_allowance_bounded(token_id)
            exec_result = await asyncio.wait_for(
                execute_live_order(
                    token_id=token_id,
                    side="SELL",
                    size=float(exit_size),
                    fallback_price=close_price,
                    min_order_size_usd=min_order_size_usd,
                    time_in_force="IOC",
                    resolve_live_price=True,
                    enforce_fallback_bound=False,
                ),
                timeout=_LIVE_EXIT_ORDER_TIMEOUT_SECONDS,
            )
    except Exception as exc:
        _record_partial_exit(
            payload,
            reason=reason,
            reduce_fraction=fraction,
            exit_size=exit_size,
            close_price=close_price,
            price_source=price_source,
            now=now,
            extra={"status": "exception", "error": _format_exit_error(exc)},
        )
        logger.warning(
            "Partial exit exception for order=%s reason=%s: %s: %s",
            row.id,
            reason,
            type(exc).__name__,
            str(exc) or "<no message>",
            exc_info=exc,
        )
        return False, _format_exit_error(exc)

    submitted = exec_result.status in {"executed", "open", "submitted"}
    extra: dict[str, Any] = {
        "status": "submitted" if submitted else "failed",
        "exec_status": str(exec_result.status or ""),
    }
    if submitted:
        extra["exit_order_id"] = exec_result.order_id
        clob_id = ""
        if isinstance(exec_result.payload, dict):
            clob_id = str(exec_result.payload.get("clob_order_id") or "")
        if clob_id:
            extra["provider_clob_order_id"] = clob_id
    else:
        extra["error"] = exec_result.error_message or ""
    _record_partial_exit(
        payload,
        reason=reason,
        reduce_fraction=fraction,
        exit_size=exit_size,
        close_price=close_price,
        price_source=price_source,
        now=now,
        extra=extra,
    )
    if submitted:
        logger.info(
            "Partial exit submitted for order=%s fraction=%.3f size=%.4f reason=%s",
            row.id,
            fraction,
            exit_size,
            reason,
        )
    else:
        logger.warning(
            "Partial exit failed for order=%s fraction=%.3f error=%s",
            row.id,
            fraction,
            exec_result.error_message,
        )
    return submitted, None if submitted else (exec_result.error_message or "exec_failed")


async def _submit_shadow_partial_exit(
    *,
    payload: dict[str, Any],
    decision: Any,
    close_price: Optional[float],
    price_source: Optional[str],
    notional_usd: float,
    entry_price: float,
    now: datetime,
) -> bool:
    """Record a partial-exit ledger entry for shadow mode positions.

    Shadow positions don't go through the live venue, so we just adjust
    the in-payload notional/cost-basis accounting and audit the slice.
    Returns True when a partial exit was recorded.
    """
    fraction = _coerce_reduce_fraction(decision)
    if fraction is None:
        return False
    if entry_price <= 0.0 or notional_usd <= 0.0:
        return False
    if close_price is None or close_price <= 0.0:
        return False

    sliced_notional = notional_usd * fraction
    realized_quantity = sliced_notional / entry_price
    realized_proceeds = realized_quantity * float(close_price)
    realized_pnl = realized_proceeds - sliced_notional

    # Mutate the position state so the residual notional shrinks for the
    # next tick. We deliberately don't rewrite ``entry_price`` — the cost
    # basis per share is unchanged on the residual.
    payload["effective_notional_usd"] = max(0.0, notional_usd - sliced_notional)
    realized_history = payload.get("partial_realized_pnl_history")
    if not isinstance(realized_history, list):
        realized_history = []
    realized_history.append(
        {
            "at": _iso_utc(now),
            "realized_pnl": realized_pnl,
            "sliced_notional_usd": sliced_notional,
            "fraction": fraction,
            "close_price": float(close_price),
        }
    )
    payload["partial_realized_pnl_history"] = realized_history[-16:]

    _record_partial_exit(
        payload,
        reason=str(getattr(decision, "reason", "") or "strategy_reduce"),
        reduce_fraction=fraction,
        exit_size=realized_quantity,
        close_price=close_price,
        price_source=price_source,
        now=now,
        extra={
            "mode": "shadow",
            "realized_pnl": realized_pnl,
            "sliced_notional_usd": sliced_notional,
        },
    )
    return True


def _strategy_hold_blocks_default_exit(exit_decision: Any) -> bool:
    if exit_decision is None:
        return False
    action = str(getattr(exit_decision, "action", "") or "").strip().lower()
    if action != "hold":
        return False
    payload = getattr(exit_decision, "payload", None)
    if not isinstance(payload, dict):
        return False
    return _safe_bool(payload.get("skip_default_exit"), False)


def _is_rapid_close_trigger(close_trigger: Any) -> bool:
    trigger = str(close_trigger or "").strip().lower()
    if not trigger:
        return False
    normalized = " ".join(trigger.replace("_", " ").replace("-", " ").split())
    rapid_markers = (
        "rapid",
        "stop loss",
        "trailing stop",
        "adaptive backside",
        "executable notional guard",
        "executable hazard exit",
        "underwater",
        "resolution risk flatten",
        "force flatten",
        "manual mark to market",
        "risk position cap",
    )
    return any(marker in normalized for marker in rapid_markers)


def _extract_leg_token_id(payload: Any) -> str:
    """Pull the leg's token_id out of a TraderOrder payload.

    Mirrors the lookup ``_shadow_ledger_token_id`` performs in the
    worker so the lifecycle reconciler can resolve a bare
    ``direction='buy'`` order back to a binary outcome index without
    threading the worker helper through.
    """

    if not isinstance(payload, dict):
        return ""
    candidates = (
        payload.get("token_id"),
        payload.get("selected_token_id"),
        payload.get("yes_token_id") if "yes" in str(payload.get("direction") or "").lower() else None,
        payload.get("no_token_id") if "no" in str(payload.get("direction") or "").lower() else None,
    )
    for raw in candidates:
        text = str(raw or "").strip()
        if text:
            return text
    leg = payload.get("leg")
    if isinstance(leg, dict):
        text = str(leg.get("token_id") or "").strip()
        if text:
            return text
    return ""


def _direction_outcome_index(
    direction: Any,
    *,
    market_info: Optional[dict[str, Any]] = None,
    token_id: Any = None,
) -> Optional[int]:
    """Map a leg ``direction`` to the binary outcome index (0=YES, 1=NO).

    Canonical fast path: ``buy_yes`` / ``sell_yes`` → 0, ``buy_no`` /
    ``sell_no`` → 1 (perf + back-compat).  The outcome index identifies
    the side of the binary market the leg sits on, regardless of the
    buy/sell action — closing a YES position is still index 0.
    Defensive widening: when the direction is bare ``buy``/``sell``
    (emitted by ``traders_copy_trade`` when the leader trade lands on a
    non-canonical outcome label) and a ``token_id`` plus binary
    ``market_info`` are both available, resolve via the market's
    ``token_ids`` array.  Truly multi-outcome single-market structures
    (>2 tokens) and missing tokens return ``None`` so the caller skips
    the row instead of silently misclassifying it.
    """

    normalized = str(direction or "").strip().lower()
    if normalized in {"buy_yes", "sell_yes"}:
        return 0
    if normalized in {"buy_no", "sell_no"}:
        return 1
    if normalized in {"buy", "sell"}:
        token_text = str(token_id or "").strip()
        if not token_text:
            return None
        tokens = _extract_market_token_ids(market_info)
        if len(tokens) != 2:
            return None
        try:
            return tokens.index(token_text)
        except ValueError:
            return None
    return None


def _opposite_direction(direction: Any) -> str:
    return StrategySDK.opposite_direction(direction, default="")


def _resolve_position_min_order_size_usd(
    *,
    trader_params: dict[str, Any],
    payload: dict[str, Any],
    mode: str,
) -> float:
    merged_params: dict[str, Any] = {}
    merged_params.update(dict(trader_params or {}))

    strategy_exit_config = payload.get("strategy_exit_config")
    if isinstance(strategy_exit_config, dict):
        merged_params.update(strategy_exit_config)

    strategy_context = payload.get("strategy_context")
    if isinstance(strategy_context, dict):
        strategy_context_params = strategy_context.get("params")
        if isinstance(strategy_context_params, dict):
            merged_params.update(strategy_context_params)

    payload_strategy_params = payload.get("strategy_params")
    if isinstance(payload_strategy_params, dict):
        merged_params.update(payload_strategy_params)

    return StrategySDK.resolve_min_order_size_usd(merged_params, mode=mode, fallback=1.0)


def _payload_strategy_exit_config(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("strategy_exit_config")
    return dict(raw) if isinstance(raw, dict) else {}


def _payload_exit_param(
    payload: dict[str, Any],
    *,
    prefix_key: str,
    name: str,
    default: Any = None,
) -> Any:
    exit_config = _payload_strategy_exit_config(payload)
    prefixed_key = f"{prefix_key}_{name}"
    if prefixed_key in exit_config:
        return exit_config.get(prefixed_key)
    if name in exit_config:
        return exit_config.get(name)
    return default


def _extract_market_token_ids(market_info: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(market_info, dict):
        return []
    token_ids = market_info.get("token_ids")
    if not isinstance(token_ids, list):
        token_ids = market_info.get("tokenIds")
    if not isinstance(token_ids, list):
        token_ids = market_info.get("clob_token_ids")
    if not isinstance(token_ids, list):
        token_ids = market_info.get("clobTokenIds")
    if not isinstance(token_ids, list):
        return []
    out: list[str] = []
    for token_id in token_ids:
        token_text = str(token_id or "").strip()
        if token_text:
            out.append(token_text)
    return out


def _extract_market_token_id(market_info: Optional[dict[str, Any]], outcome_idx: int) -> str:
    if outcome_idx not in (0, 1):
        return ""
    token_ids = _extract_market_token_ids(market_info)
    if not token_ids:
        return ""
    outcomes_raw = (market_info or {}).get("outcomes")
    if isinstance(outcomes_raw, list) and len(outcomes_raw) == len(token_ids):
        for idx, outcome_label in enumerate(outcomes_raw):
            label = str(outcome_label or "").strip().lower()
            if label == "yes" and outcome_idx == 0:
                return token_ids[idx]
            if label == "no" and outcome_idx == 1:
                return token_ids[idx]
    if len(token_ids) > outcome_idx:
        return token_ids[outcome_idx]
    return ""


def _normalize_reverse_entry(
    *,
    reverse_intent: Any,
    current_direction: str,
    current_price: Optional[float],
    market_seconds_left: Optional[float],
    market_info: Optional[dict[str, Any]],
    strategy_type: str,
    source: str,
    now: datetime,
) -> dict[str, Any] | None:
    normalized = StrategySDK.normalize_reverse_intent(
        reverse_intent,
        fallback_direction=_opposite_direction(current_direction),
        default_signal_type=f"{str(source or '').strip().lower() or 'strategy'}_reverse",
    )
    if not isinstance(normalized, dict):
        return None

    direction = str(normalized.get("direction") or "").strip().lower()
    if direction not in {"buy_yes", "buy_no"}:
        return None
    if direction == str(current_direction or "").strip().lower():
        return None

    min_seconds_left = max(0.0, safe_float(normalized.get("min_seconds_left"), 0.0) or 0.0)
    if market_seconds_left is not None and market_seconds_left < min_seconds_left:
        return None

    min_price_headroom = max(0.0, min(1.0, safe_float(normalized.get("min_price_headroom"), 0.0) or 0.0))
    if current_price is not None and current_price < min_price_headroom:
        return None

    outcome_idx = _direction_outcome_index(direction)
    if outcome_idx is None:
        return None

    entry_price = safe_float(normalized.get("entry_price"), None)
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        entry_price = _extract_market_side_price(market_info, outcome_idx)
    if (entry_price is None or entry_price <= 0.0 or entry_price >= 1.0) and current_price is not None:
        entry_price = 1.0 - current_price
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        return None

    token_id = str(normalized.get("token_id") or "").strip()
    if not token_id:
        token_id = _extract_market_token_id(market_info, outcome_idx)

    strategy_key = str(normalized.get("strategy_type") or strategy_type or "").strip().lower()
    signal_type = str(normalized.get("signal_type") or f"{source}_reverse").strip().lower()
    if not signal_type:
        signal_type = "strategy_reverse"

    expires_in_seconds = max(5.0, safe_float(normalized.get("expires_in_seconds"), 60.0) or 60.0)
    expires_at = now + timedelta(seconds=expires_in_seconds)
    out = dict(normalized)
    out["direction"] = direction
    out["entry_price"] = float(max(0.001, min(0.999, entry_price)))
    out["token_id"] = token_id
    out["strategy_type"] = strategy_key
    out["signal_type"] = signal_type
    out["source"] = str(source or "").strip().lower()
    out["expires_at"] = _iso_utc(expires_at)
    return out


def _arm_reverse_entry_from_exit(
    *,
    row: TraderOrder,
    payload: dict[str, Any],
    strategy_exit: Any,
    market_info: Optional[dict[str, Any]],
    market_seconds_left: Optional[float],
    current_price: Optional[float],
    now: datetime,
) -> dict[str, Any] | None:
    strategy_payload = getattr(strategy_exit, "payload", None)
    if not isinstance(strategy_payload, dict):
        return None
    reverse_intent = strategy_payload.get("reverse_intent")
    if not isinstance(reverse_intent, dict):
        return None

    existing = payload.get("pending_reverse_entry")
    existing = dict(existing) if isinstance(existing, dict) else {}
    existing_status = str(existing.get("status") or "").strip().lower()
    if existing_status in {"emitted", "submitted"} and str(existing.get("signal_id") or "").strip():
        return None

    prepared = _normalize_reverse_entry(
        reverse_intent=reverse_intent,
        current_direction=str(row.direction or ""),
        current_price=current_price,
        market_seconds_left=market_seconds_left,
        market_info=market_info,
        strategy_type=str(payload.get("strategy_type") or ""),
        source=str(row.source or ""),
        now=now,
    )
    if not isinstance(prepared, dict):
        return None

    attempt_index = int(safe_float(existing.get("attempt_index"), 0) or 0) + 1
    prepared["status"] = "armed"
    prepared["armed_at"] = _iso_utc(now)
    prepared["attempt_index"] = attempt_index
    prepared["source_order_id"] = str(row.id or "")
    prepared["source_signal_id"] = str(row.signal_id or "")
    prepared["trigger_reason"] = str(getattr(strategy_exit, "reason", "") or "").strip()
    payload["pending_reverse_entry"] = prepared
    return prepared


async def _emit_armed_reverse_signal(
    session: AsyncSession,
    *,
    row: TraderOrder,
    payload: dict[str, Any],
    signal_payload: dict[str, Any],
    market_info: Optional[dict[str, Any]],
    close_trigger: Optional[str],
    realized_pnl: Optional[float],
    now: datetime,
) -> tuple[str | None, str | None]:
    pending = payload.get("pending_reverse_entry")
    if not isinstance(pending, dict):
        return None, None
    status = str(pending.get("status") or "").strip().lower()
    if status not in {"armed", "failed", "ready"}:
        return None, None
    trigger_key = str(close_trigger or "").strip().lower()
    if "resolution" in trigger_key:
        pending["status"] = "skipped_resolution"
        pending["skipped_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    expires_at = _parse_iso_utc_naive(pending.get("expires_at"))
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    if expires_at is not None and now_naive > expires_at:
        pending["status"] = "expired"
        pending["expired_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    direction = str(pending.get("direction") or "").strip().lower()
    if direction not in {"buy_yes", "buy_no"}:
        pending["status"] = "failed"
        pending["last_error"] = "invalid_reverse_direction"
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None
    outcome_idx = _direction_outcome_index(direction)
    if outcome_idx is None:
        pending["status"] = "failed"
        pending["last_error"] = "invalid_reverse_outcome_idx"
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    entry_price = safe_float(pending.get("entry_price"), None)
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        entry_price = _extract_market_side_price(market_info, outcome_idx)
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        close_price = safe_float((payload.get("position_close") or {}).get("close_price"), None)
        if close_price is not None:
            entry_price = 1.0 - close_price
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        pending["status"] = "failed"
        pending["last_error"] = "missing_reverse_entry_price"
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    token_id = str(pending.get("token_id") or "").strip()
    if not token_id:
        token_id = _extract_market_token_id(market_info, outcome_idx)
        if token_id:
            pending["token_id"] = token_id

    source = str(row.source or "").strip().lower() or "scanner"
    strategy_type = str(pending.get("strategy_type") or payload.get("strategy_type") or "").strip().lower() or None
    signal_type = str(pending.get("signal_type") or f"{source}_reverse").strip().lower()
    if not signal_type:
        signal_type = "strategy_reverse"
    confidence = safe_float(pending.get("confidence"), None)
    edge_percent = safe_float(pending.get("edge_percent"), None)
    liquidity = safe_float(
        pending.get("liquidity"),
        safe_float((signal_payload or {}).get("liquidity"), safe_float(getattr(row, "notional_usd", None), 0.0)),
    )

    payload_json = dict(signal_payload or {})
    payload_json["strategy_origin"] = str(payload_json.get("strategy_origin") or "crypto_worker")
    payload_json["reverse_entry"] = {
        "enabled": True,
        "source_order_id": str(row.id or ""),
        "source_signal_id": str(row.signal_id or ""),
        "close_trigger": trigger_key,
        "realized_pnl": realized_pnl,
        "direction": direction,
        "entry_price": float(entry_price),
        "size_multiplier": safe_float(pending.get("size_multiplier"), 1.0),
        "size_usd": safe_float(pending.get("size_usd")),
        "metadata": dict(pending.get("metadata") or {}) if isinstance(pending.get("metadata"), dict) else {},
    }
    payload_json["reverse_entry"]["emitted_at"] = _iso_utc(now)
    payload_json["selected_token_id"] = token_id or payload_json.get("selected_token_id")
    payload_json["token_id"] = token_id or payload_json.get("token_id")
    payload_json["signal_emitted_at"] = _iso_utc(now)
    payload_json["source_observed_at"] = _iso_utc(now)

    strategy_context = payload.get("strategy_context")
    strategy_context_json = dict(strategy_context) if isinstance(strategy_context, dict) else {}
    strategy_context_json["reverse_entry"] = dict(payload_json["reverse_entry"])
    strategy_context_json["source_order_id"] = str(row.id or "")
    strategy_context_json["source_signal_id"] = str(row.signal_id or "")
    strategy_context_json["selected_direction"] = direction
    if token_id:
        strategy_context_json["selected_token_id"] = token_id

    attempt_index = int(safe_float(pending.get("attempt_index"), 1) or 1)
    dedupe_key = make_dedupe_key(
        "reverse_entry",
        str(row.id or ""),
        attempt_index,
        direction,
        str(pending.get("armed_at") or ""),
    )
    source_item_id = str(pending.get("source_signal_id") or row.signal_id or row.id or "")
    market_question = (
        str(
            payload_json.get("market_question") or payload_json.get("question") or getattr(row, "market_id", "") or ""
        ).strip()
        or None
    )
    expires_at_dt = _parse_iso_utc_naive(pending.get("expires_at"))
    if expires_at_dt is None:
        expires_seconds = max(5.0, safe_float(pending.get("expires_in_seconds"), 60.0) or 60.0)
        expires_at_dt = now_naive + timedelta(seconds=expires_seconds)

    try:
        signal_row = await upsert_trade_signal(
            session,
            source=source,
            source_item_id=source_item_id,
            signal_type=signal_type,
            strategy_type=strategy_type,
            market_id=str(row.market_id or ""),
            market_question=market_question,
            direction=direction,
            entry_price=float(max(0.001, min(0.999, entry_price))),
            edge_percent=edge_percent,
            confidence=confidence,
            liquidity=liquidity,
            expires_at=expires_at_dt,
            payload_json=payload_json,
            strategy_context_json=strategy_context_json or None,
            dedupe_key=dedupe_key,
            commit=False,
        )
    except Exception as exc:
        pending["status"] = "failed"
        pending["last_error"] = str(exc)
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    signal_id = str(getattr(signal_row, "id", "") or "").strip()
    pending["status"] = "emitted"
    pending["emitted_at"] = _iso_utc(now)
    pending["signal_id"] = signal_id
    pending["dedupe_key"] = dedupe_key
    payload["pending_reverse_entry"] = pending
    return signal_id or None, source


async def _publish_reverse_signal_batches(signal_ids_by_source: dict[str, list[str]], *, emitted_at: datetime) -> None:
    if not signal_ids_by_source:
        return
    from services.event_bus import event_bus

    emitted_at_iso = emitted_at.isoformat()
    for source_key, signal_ids in signal_ids_by_source.items():
        ids = [str(signal_id or "").strip() for signal_id in signal_ids if str(signal_id or "").strip()]
        if not ids:
            continue
        try:
            await event_bus.publish(
                "trade_signal_batch",
                {
                    "event_type": "reverse_entry",
                    "source": str(source_key or "").strip().lower(),
                    "signal_count": int(len(ids)),
                    "signal_ids": ids[:500],
                    "emitted_at": emitted_at_iso,
                    "trigger": "position_lifecycle_reverse",
                },
            )
        except Exception:
            continue
        try:
            await publish_signal_batch(
                event_type="reverse_entry",
                source=str(source_key or "").strip().lower(),
                signal_ids=ids,
                trigger="position_lifecycle_reverse",
                emitted_at=emitted_at_iso,
            )
        except Exception:
            continue


def _extract_signal_side_price(payload: dict[str, Any], outcome_idx: int) -> Optional[float]:
    side_keys = ("yes",) if outcome_idx == 0 else ("no",)
    for prefix in side_keys:
        for key in (
            f"{prefix}_price",
            f"{prefix}Price",
            f"best_{prefix}",
            f"best{prefix.title()}",
            f"{prefix}_mid",
            f"{prefix}Mid",
        ):
            parsed = safe_float(payload.get(key))
            if parsed is not None and parsed >= 0:
                return parsed

    prices = payload.get("outcome_prices")
    if not isinstance(prices, list):
        prices = payload.get("outcomePrices")
    if isinstance(prices, list) and len(prices) > outcome_idx:
        parsed = safe_float(prices[outcome_idx])
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_market_side_price(market_info: Optional[dict[str, Any]], outcome_idx: int) -> Optional[float]:
    if not isinstance(market_info, dict):
        return None
    key = "yes_price" if outcome_idx == 0 else "no_price"
    parsed = safe_float(market_info.get(key))
    if parsed is not None and parsed >= 0:
        return parsed
    prices = market_info.get("outcome_prices")
    if isinstance(prices, list) and len(prices) > outcome_idx:
        parsed = safe_float(prices[outcome_idx])
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_winning_outcome_index(market_info: Optional[dict[str, Any]]) -> Optional[int]:
    if not isinstance(market_info, dict):
        return None

    winner_raw = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") not in (None, "")
        else market_info.get("winner")
    )
    if winner_raw in (None, ""):
        return None

    outcomes_raw = market_info.get("outcomes")
    outcomes: list[str] = []
    if isinstance(outcomes_raw, list):
        outcomes = [str(item or "").strip().lower() for item in outcomes_raw if str(item or "").strip()]

    try:
        idx = int(winner_raw)
        if idx in (0, 1):
            return idx
    except Exception:
        pass

    winner_text = str(winner_raw).strip().lower()
    if winner_text == "yes":
        return 0
    if winner_text == "no":
        return 1
    if outcomes:
        for idx, label in enumerate(outcomes):
            if label == winner_text and idx in (0, 1):
                return idx
    return None


def _market_has_terminal_resolution_signal(market_info: Optional[dict[str, Any]]) -> bool:
    if not isinstance(market_info, dict):
        return False

    if _safe_bool(market_info.get("closed"), False):
        return True
    if _safe_bool(market_info.get("archived"), False):
        return True

    resolved_value = (
        market_info.get("resolved")
        if market_info.get("resolved") is not None
        else (
            market_info.get("isResolved")
            if market_info.get("isResolved") is not None
            else market_info.get("is_resolved")
        )
    )
    if _safe_bool(resolved_value, False):
        return True

    winner = market_info.get("winner")
    if winner not in (None, ""):
        return True

    winning_outcome = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") is not None
        else market_info.get("winningOutcome")
    )
    if winning_outcome not in (None, ""):
        return True

    raw_status = (
        market_info.get("status")
        if market_info.get("status") is not None
        else (
            market_info.get("market_status")
            if market_info.get("market_status") is not None
            else market_info.get("marketStatus")
        )
    )
    status_text = str(raw_status or "").strip().lower()
    if status_text:
        normalized = status_text.replace("_", " ").replace("-", " ")
        blocked_terms = (
            "in review",
            "review",
            "in dispute",
            "dispute",
            "final",
            "resolved",
            "settled",
            "closed",
            "expired",
            "cancelled",
            "canceled",
        )
        if any(term in normalized for term in blocked_terms):
            return True

    uma_statuses: list[str] = []
    raw_uma_status = (
        market_info.get("uma_resolution_status")
        if market_info.get("uma_resolution_status") is not None
        else market_info.get("umaResolutionStatus")
    )
    if isinstance(raw_uma_status, list):
        uma_statuses.extend(str(item or "").strip() for item in raw_uma_status if str(item or "").strip())
    elif raw_uma_status not in (None, ""):
        uma_statuses.append(str(raw_uma_status).strip())

    raw_uma_statuses = (
        market_info.get("uma_resolution_statuses")
        if market_info.get("uma_resolution_statuses") is not None
        else market_info.get("umaResolutionStatuses")
    )
    if isinstance(raw_uma_statuses, list):
        uma_statuses.extend(str(item or "").strip() for item in raw_uma_statuses if str(item or "").strip())
    elif raw_uma_statuses not in (None, ""):
        uma_statuses.append(str(raw_uma_statuses).strip())

    if uma_statuses:
        blocked_terms = (
            "proposed",
            "review",
            "dispute",
            "final",
            "resolved",
            "settled",
            "closed",
            "expired",
            "cancelled",
            "canceled",
        )
        for raw in uma_statuses:
            normalized = raw.lower().replace("_", " ").replace("-", " ")
            if any(term in normalized for term in blocked_terms):
                return True

    end_dt = polymarket_client._coerce_datetime(
        market_info.get("end_date") if market_info.get("end_date") is not None else market_info.get("endDate")
    )
    ended = end_dt is not None and end_dt <= utcnow()
    accepting_orders_value = (
        market_info.get("accepting_orders")
        if market_info.get("accepting_orders") is not None
        else market_info.get("acceptingOrders")
    )
    enable_order_book_value = (
        market_info.get("enable_order_book")
        if market_info.get("enable_order_book") is not None
        else market_info.get("enableOrderBook")
    )
    if ended and (
        _safe_bool(accepting_orders_value, True) is False
        or _safe_bool(enable_order_book_value, True) is False
    ):
        return True

    return False


def _extract_winning_outcome_index_from_prices(
    market_info: Optional[dict[str, Any]],
    *,
    market_tradable: bool,
    settle_floor: float,
) -> Optional[int]:
    if not isinstance(market_info, dict):
        return None
    if market_tradable:
        return None
    if not _market_has_terminal_resolution_signal(market_info):
        return None

    settle_floor = min(1.0, max(0.5, settle_floor))
    settle_ceiling = max(0.0, 1.0 - settle_floor)

    yes_price = safe_float(market_info.get("yes_price"))
    no_price = safe_float(market_info.get("no_price"))
    if yes_price is None or no_price is None:
        prices = market_info.get("outcome_prices")
        if isinstance(prices, list) and len(prices) >= 2:
            outcomes = market_info.get("outcomes")
            if isinstance(outcomes, list) and len(outcomes) == len(prices):
                for idx, raw_label in enumerate(outcomes):
                    label = str(raw_label or "").strip().lower()
                    parsed = safe_float(prices[idx])
                    if parsed is None:
                        continue
                    if yes_price is None and label == "yes":
                        yes_price = parsed
                    if no_price is None and label == "no":
                        no_price = parsed
            if yes_price is None:
                yes_price = safe_float(prices[0])
            if no_price is None:
                no_price = safe_float(prices[1])

    yes_price = _state_price_floor(yes_price)
    no_price = _state_price_floor(no_price)
    if yes_price is None or no_price is None:
        return None
    if yes_price >= settle_floor and no_price <= settle_ceiling:
        return 0
    if no_price >= settle_floor and yes_price <= settle_ceiling:
        return 1
    return None


def _missing_market_info_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _normalize_market_info_aliases(market_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(market_info, dict):
        return {}

    normalized = dict(market_info)
    if _missing_market_info_value(normalized.get("id")) and not _missing_market_info_value(normalized.get("market_id")):
        normalized["id"] = normalized.get("market_id")
    if _missing_market_info_value(normalized.get("market_id")) and not _missing_market_info_value(normalized.get("id")):
        normalized["market_id"] = normalized.get("id")

    if _missing_market_info_value(normalized.get("yes_price")):
        live_yes_price = safe_float(normalized.get("live_yes_price"))
        if live_yes_price is not None:
            normalized["yes_price"] = live_yes_price
    if _missing_market_info_value(normalized.get("no_price")):
        live_no_price = safe_float(normalized.get("live_no_price"))
        if live_no_price is not None:
            normalized["no_price"] = live_no_price

    selected_price = safe_float(normalized.get("live_selected_price"))
    selected_outcome = str(normalized.get("selected_outcome") or "").strip().lower()
    if selected_price is not None:
        if selected_outcome == "yes" and _missing_market_info_value(normalized.get("yes_price")):
            normalized["yes_price"] = selected_price
        elif selected_outcome == "no" and _missing_market_info_value(normalized.get("no_price")):
            normalized["no_price"] = selected_price

    if _missing_market_info_value(normalized.get("end_date")) and not _missing_market_info_value(
        normalized.get("market_end_time")
    ):
        normalized["end_date"] = normalized.get("market_end_time")
    if _missing_market_info_value(normalized.get("resolved")) and not _missing_market_info_value(
        normalized.get("market_resolved")
    ):
        normalized["resolved"] = normalized.get("market_resolved")
    if _missing_market_info_value(normalized.get("closed")) and not _missing_market_info_value(
        normalized.get("market_closed")
    ):
        normalized["closed"] = normalized.get("market_closed")

    return normalized


def _merge_market_info(primary: Optional[dict[str, Any]], fallback: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(primary, dict):
        normalized_fallback = _normalize_market_info_aliases(fallback)
        return normalized_fallback or None

    merged = _normalize_market_info_aliases(fallback)
    normalized_primary = _normalize_market_info_aliases(primary)
    for key, value in normalized_primary.items():
        if not _missing_market_info_value(value) or key not in merged:
            merged[key] = value
    return merged


def _terminal_extreme_selected_price(price: Optional[float]) -> Optional[float]:
    normalized = _state_price_floor(price)
    if normalized is None:
        return None
    if normalized <= _TERMINAL_EXTREME_MARK_THRESHOLD:
        return 0.0
    if normalized >= (1.0 - _TERMINAL_EXTREME_MARK_THRESHOLD):
        return 1.0
    return None


def _infer_post_end_terminal_price(
    *,
    market_info: Optional[dict[str, Any]],
    current_price: Optional[float],
    current_price_source: Optional[str],
    previous_mark_price: Optional[float],
    wallet_mark_price: Optional[float],
    now: datetime,
) -> tuple[Optional[float], Optional[str]]:
    end_time = _extract_market_end_time_naive(market_info)
    if end_time is None:
        return None, None

    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    if (now_naive - end_time).total_seconds() < _POST_END_EXTREME_MARK_GRACE_SECONDS:
        return None, None

    live_price_source = str(current_price_source or "").strip().lower()
    if current_price is not None and live_price_source in {"ws_mid", "clob_midpoint", "wallet_mark"}:
        terminal_price = _terminal_extreme_selected_price(current_price)
        if terminal_price is not None:
            return terminal_price, live_price_source
        return None, None

    for source, price in (
        (live_price_source or "market_mark", current_price),
        ("wallet_mark", wallet_mark_price),
        ("position_state_mark", previous_mark_price),
    ):
        terminal_price = _terminal_extreme_selected_price(price)
        if terminal_price is not None:
            return terminal_price, source
    return None, None


def _state_price_floor(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _status_for_close(*, pnl: float, close_trigger: Optional[str]) -> str:
    normalized_pnl = 0.0 if abs(float(pnl or 0.0)) <= 1e-9 else float(pnl)
    trigger = str(close_trigger or "").strip().lower()
    is_resolution = trigger in {"resolution", "resolution_inferred", "resolution_extreme_mark"}
    if is_resolution:
        return "resolved_win" if normalized_pnl >= 0 else "resolved_loss"
    return "closed_win" if normalized_pnl >= 0 else "closed_loss"


def _extract_position_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("position_state")
    return state if isinstance(state, dict) else {}


def _first_float_from_candidates(candidates: list[Any], keys: tuple[str, ...]) -> Optional[float]:
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            parsed = safe_float(candidate.get(key))
            if parsed is not None:
                return float(parsed)
    return None


def _entry_requested_size_cap(payload: dict[str, Any]) -> Optional[float]:
    wallet_position_aggregate = payload.get("wallet_position_aggregate_authority")
    if isinstance(wallet_position_aggregate, dict):
        aggregate_size = safe_float(wallet_position_aggregate.get("aggregate_size"))
        if aggregate_size is not None and aggregate_size > _WALLET_SIZE_EPSILON:
            return float(aggregate_size)

    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    submission_intent = payload.get("submission_intent")
    if not isinstance(submission_intent, dict):
        submission_intent = {}
    cap = _first_float_from_candidates(
        [snapshot, provider_reconciliation, submission_intent, payload],
        (
            "size",
            "original_size",
            "initial_size",
            "requested_shares",
            "requested_size",
            "shares",
        ),
    )
    if cap is None or cap <= _WALLET_SIZE_EPSILON:
        return None
    return float(cap)


def _live_position_notional_cap(params: dict[str, Any]) -> Optional[float]:
    caps: list[float] = []
    for key in ("max_position_notional_usd", "max_per_market_exposure_usd"):
        cap = safe_float(params.get(key))
        if cap is not None and cap > 0.0:
            caps.append(float(cap))
    if not caps:
        return None
    return min(caps)


def _extract_live_fill_metrics(payload: dict[str, Any]) -> tuple[float, float, Optional[float]]:
    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    candidates: list[Any] = [provider_reconciliation, snapshot, payload]
    filled_size = max(
        0.0,
        _first_float_from_candidates(
            candidates,
            (
                "filled_size",
                "size_matched",
                "sizeMatched",
                "matched_size",
                "filled_shares",
                "executed_size",
            ),
        )
        or 0.0,
    )
    average_fill_price = _first_float_from_candidates(
        candidates,
        (
            "average_fill_price",
            "avg_fill_price",
            "avg_price",
            "avgFillPrice",
            "matched_price",
            "price",
            "limit_price",
        ),
    )
    filled_notional = max(
        0.0,
        _first_float_from_candidates(
            candidates,
            (
                "filled_notional_usd",
                "filled_notional",
                "matched_notional",
                "matched_amount",
                "executed_notional",
            ),
        )
        or 0.0,
    )
    if filled_notional <= 0.0 and filled_size > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_notional = filled_size * average_fill_price
    if filled_size <= 0.0 and filled_notional > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_size = filled_notional / average_fill_price
    requested_size_cap = _entry_requested_size_cap(payload)
    if requested_size_cap is not None and filled_size > requested_size_cap:
        filled_size = requested_size_cap
        if average_fill_price is not None and average_fill_price > 0.0:
            filled_notional = filled_size * average_fill_price
    if (
        filled_notional > 0.0
        and filled_size > 0.0
        and average_fill_price is not None
        and average_fill_price > 0.0
    ):
        capped_notional = filled_size * average_fill_price
        if filled_notional > capped_notional:
            filled_notional = capped_notional
    return filled_notional, filled_size, average_fill_price


def _provider_reconciliation_parts(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    return provider_reconciliation, snapshot


def _provider_reconciliation_has_order_evidence(row: TraderOrder, payload: dict[str, Any]) -> bool:
    if str(getattr(row, "provider_clob_order_id", None) or "").strip():
        return True
    if str(getattr(row, "provider_order_id", None) or "").strip():
        return True
    filled_notional, filled_size, _average_fill_price = _extract_live_fill_metrics(payload)
    if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
        return True
    provider_reconciliation, snapshot = _provider_reconciliation_parts(payload)
    raw_snapshot = snapshot.get("raw")
    candidates = [payload, provider_reconciliation, snapshot]
    if isinstance(raw_snapshot, dict):
        candidates.append(raw_snapshot)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in (
            "clob_order_id",
            "order_id",
            "provider_order_id",
            "provider_clob_order_id",
            "live_trading_order_id",
            "exchange_order_id",
        ):
            if str(candidate.get(key) or "").strip():
                return True
    live_wallet_authority = payload.get("live_wallet_authority")
    if isinstance(live_wallet_authority, dict):
        authority_source = str(live_wallet_authority.get("source") or "").strip().lower()
        if authority_source in {"wallet_positions_api", "wallet_trade_history"}:
            return True
    return False


def _normalize_provider_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if not status:
        return ""
    if status in {"live", "active", "working", "partially_filled", "partially_matched"}:
        return "open"
    if status in {"matched", "executed"}:
        return "filled"
    if status in {"pending", "placing", "queued", "submitted"}:
        return "pending"
    if status == "canceled":
        return "cancelled"
    return status


def _raw_snapshot_status(snapshot: dict[str, Any]) -> str:
    raw_snapshot = snapshot.get("raw")
    raw_status = snapshot.get("status") or snapshot.get("raw_status")
    if not raw_status and isinstance(raw_snapshot, dict):
        raw_status = raw_snapshot.get("status") or raw_snapshot.get("raw_status")
    return _normalize_provider_status(raw_status)


def _resolved_snapshot_status(snapshot: Optional[dict[str, Any]], *status_candidates: Any) -> str:
    snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
    raw_status = _raw_snapshot_status(snapshot_dict)
    resolved_status = ""
    for candidate in status_candidates:
        normalized_status = _normalize_provider_status(candidate)
        if normalized_status:
            resolved_status = normalized_status
            break
    if resolved_status in {"open", "pending"} and raw_status in {
        "filled",
        "cancelled",
        "expired",
        "failed",
        "rejected",
        "error",
    }:
        return raw_status
    return resolved_status or raw_status


def _provider_snapshot_status(payload: dict[str, Any]) -> str:
    provider_reconciliation, snapshot = _provider_reconciliation_parts(payload)
    return _resolved_snapshot_status(
        snapshot,
        snapshot.get("normalized_status"),
        provider_reconciliation.get("snapshot_status"),
        provider_reconciliation.get("mapped_status"),
    )


def _live_trading_order_snapshot(row: LiveTradingOrder) -> dict[str, Any]:
    raw_status = str(row.status or "").strip().lower()
    normalized_status = raw_status
    if raw_status in {"matched", "executed", "filled"}:
        normalized_status = "filled"
    elif raw_status in {"open", "working", "active", "live", "partially_filled", "partially_matched"}:
        normalized_status = "open"
    elif raw_status in {"pending", "placing", "queued", "submitted"}:
        normalized_status = "pending"
    elif raw_status in {"cancelled", "canceled", "expired"}:
        normalized_status = "cancelled"
    elif raw_status in {"failed", "rejected", "error"}:
        normalized_status = "failed"

    filled_size = max(0.0, safe_float(row.filled_size, 0.0) or 0.0)
    average_fill_price = safe_float(row.average_fill_price)
    limit_price = safe_float(row.price)
    filled_notional_usd = 0.0
    if filled_size > 0.0 and average_fill_price is not None and average_fill_price > 0.0:
        filled_notional_usd = filled_size * average_fill_price
    elif filled_size > 0.0 and limit_price is not None and limit_price > 0.0:
        filled_notional_usd = filled_size * limit_price

    return {
        "clob_order_id": str(row.clob_order_id or "").strip() or None,
        "normalized_status": normalized_status,
        "raw_status": raw_status or None,
        "size": safe_float(row.size),
        "filled_size": filled_size,
        "remaining_size": None,
        "average_fill_price": average_fill_price,
        "limit_price": limit_price,
        "filled_notional_usd": filled_notional_usd if filled_notional_usd > 0.0 else None,
        "raw": {
            "status": raw_status or None,
            "side": str(row.side or "").strip().upper() or None,
            "token_id": str(row.token_id or "").strip() or None,
            "updated_at": _iso_utc(row.updated_at if row.updated_at is not None else utcnow()),
        },
    }


def _provider_snapshot_remaining_size(payload: dict[str, Any]) -> Optional[float]:
    provider_reconciliation, snapshot = _provider_reconciliation_parts(payload)
    remaining_size = safe_float(
        snapshot.get("remaining_size"),
        safe_float(provider_reconciliation.get("remaining_size"), None),
    )
    if remaining_size is not None:
        return max(0.0, float(remaining_size))
    requested_size = _first_float_from_candidates(
        [snapshot, provider_reconciliation, payload],
        ("size", "original_size", "requested_size", "shares", "requested_shares"),
    )
    filled_size = _first_float_from_candidates(
        [snapshot, provider_reconciliation, payload],
        ("filled_size", "size_matched", "matched_size", "filled_shares", "executed_size"),
    )
    if requested_size is None:
        return None
    return max(0.0, float(requested_size) - max(0.0, float(filled_size or 0.0)))


def _provider_entry_order_still_working(payload: dict[str, Any]) -> bool:
    snapshot_status = _provider_snapshot_status(payload)
    if snapshot_status not in {"open", "submitted", "pending"}:
        return False
    remaining_size = _provider_snapshot_remaining_size(payload)
    return remaining_size is None or remaining_size > _WALLET_SIZE_EPSILON


def _extract_live_token_id(payload: dict[str, Any]) -> str:
    token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
    if token_id:
        return token_id
    provider_reconciliation = payload.get("provider_reconciliation")
    if isinstance(provider_reconciliation, dict):
        snapshot = provider_reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            token_id = str(snapshot.get("asset_id") or snapshot.get("asset") or snapshot.get("token_id") or "").strip()
            if token_id:
                return token_id
    return ""


def _extract_condition_outcome_key(payload: dict[str, Any], direction: str) -> str:
    """Build a conditionId:outcomeIndex fallback key for wallet position lookup.

    Polymarket can change a market's token_ids (token versioning) while the
    stored order still references the original.  The conditionId + outcomeIndex
    pair is stable across versions, so it serves as a reliable fallback when
    exact token_id lookup fails.
    """
    # Try live_market.condition_id first (set during live execution)
    cid = ""
    live_market = payload.get("live_market")
    if isinstance(live_market, dict):
        cid = str(live_market.get("condition_id") or live_market.get("conditionId") or "").strip()
    # Fall back to provider reconciliation snapshot
    if not cid:
        pr = payload.get("provider_reconciliation")
        if isinstance(pr, dict):
            snapshot = pr.get("snapshot")
            if isinstance(snapshot, dict):
                raw = snapshot.get("raw")
                if isinstance(raw, dict):
                    cid = str(raw.get("market") or raw.get("condition_id") or raw.get("conditionId") or "").strip()
    if not cid:
        return ""
    # Determine outcome index from direction
    dir_lower = str(direction or "").strip().lower()
    if dir_lower in ("buy_no", "no", "sell_yes"):
        outcome_idx = 1
    elif dir_lower in ("buy_yes", "yes", "sell_no"):
        outcome_idx = 0
    else:
        # Try to get from leg metadata
        leg = payload.get("leg")
        if isinstance(leg, dict):
            outcome = str(leg.get("outcome") or "").strip().lower()
            if outcome in ("no", "down"):
                outcome_idx = 1
            elif outcome in ("yes", "up"):
                outcome_idx = 0
            else:
                return ""
        else:
            return ""
    return f"{cid}:{outcome_idx}"


def _extract_wallet_settlement_price(wallet_position: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(wallet_position, dict):
        return None
    if not _safe_bool(wallet_position.get("redeemable"), False):
        return None
    mark = safe_float(wallet_position.get("curPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("currentPrice"))
    if mark is None:
        # redeemable=True already filters out losers, so counts_as_open=False
        # here reliably means the position settled as a winner at $1.
        if not _safe_bool(wallet_position.get("counts_as_open"), True):
            return 1.0
        return None
    if mark <= 0.001:
        return 0.0
    if mark >= 0.999:
        return 1.0
    return None


def _extract_wallet_position_size(wallet_position: Optional[dict[str, Any]]) -> float:
    if not isinstance(wallet_position, dict):
        return 0.0
    size = safe_float(wallet_position.get("size"))
    if size is None:
        size = safe_float(wallet_position.get("positionSize"))
    if size is None:
        size = safe_float(wallet_position.get("shares"))
    return max(0.0, float(size or 0.0))


def _extract_wallet_mark_price(wallet_position: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(wallet_position, dict):
        return None
    mark = safe_float(wallet_position.get("curPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("currentPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("markPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("price"))
    if mark is None or mark < 0:
        return None
    return float(mark)


def _extract_wallet_entry_price(wallet_position: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(wallet_position, dict):
        return None
    price = safe_float(wallet_position.get("avgPrice"))
    if price is None:
        price = safe_float(wallet_position.get("average_price"))
    if price is None:
        initial_value = safe_float(wallet_position.get("initialValue"))
        size = _extract_wallet_position_size(wallet_position)
        if initial_value is not None and initial_value > 0.0 and size > _WALLET_SIZE_EPSILON:
            price = initial_value / size
    if price is None or price <= 0.0:
        return None
    return float(price)


def _lookup_wallet_position_for_payload(
    wallet_positions_by_token: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    direction: str,
) -> tuple[str, Optional[dict[str, Any]]]:
    token_id = _extract_live_token_id(payload)
    if token_id:
        wallet_position = wallet_positions_by_token.get(token_id) or wallet_positions_by_token.get(token_id.lower())
        if isinstance(wallet_position, dict):
            return token_id, wallet_position

    if token_id:
        cond_key = _extract_condition_outcome_key(payload, direction)
        if cond_key:
            wallet_position = wallet_positions_by_token.get(cond_key) or wallet_positions_by_token.get(cond_key.lower())
            if isinstance(wallet_position, dict):
                return cond_key, wallet_position

    return token_id, None


def _pending_exit_blocks_wallet_position_collapse(payload: dict[str, Any]) -> bool:
    pending_exit = payload.get("pending_live_exit")
    if not isinstance(pending_exit, dict):
        return False
    status_key = str(pending_exit.get("status") or "").strip().lower()
    if not status_key:
        return False
    return status_key in {"pending", "submitted", "working", "open", "partially_filled", "filled"}


def _wallet_position_notional(wallet_position: dict[str, Any]) -> tuple[float, float, Optional[float], Optional[float]]:
    wallet_size = _extract_wallet_position_size(wallet_position)
    wallet_entry_price = _extract_wallet_entry_price(wallet_position)
    wallet_mark_price = _extract_wallet_mark_price(wallet_position)
    price = wallet_entry_price if wallet_entry_price is not None and wallet_entry_price > 0.0 else wallet_mark_price
    notional = wallet_size * price if price is not None and price > 0.0 else 0.0
    return wallet_size, notional, wallet_entry_price, wallet_mark_price


def _payload_execution_plan(payload: dict[str, Any]) -> dict[str, Any]:
    execution_plan = payload.get("execution_plan")
    if isinstance(execution_plan, dict):
        return execution_plan
    strategy_context = payload.get("strategy_context")
    if isinstance(strategy_context, dict):
        nested_plan = strategy_context.get("execution_plan")
        if isinstance(nested_plan, dict):
            return nested_plan
    return {}


def _required_bundle_token_ids(payload: dict[str, Any], signal_payload: dict[str, Any]) -> list[str]:
    token_ids: list[str] = []
    for source_payload in (payload, signal_payload):
        execution_plan = _payload_execution_plan(source_payload)
        raw_legs = execution_plan.get("legs")
        raw_legs = raw_legs if isinstance(raw_legs, list) else []
        for leg in raw_legs:
            if not isinstance(leg, dict):
                continue
            token_id = str(leg.get("token_id") or "").strip()
            if token_id and token_id not in token_ids:
                token_ids.append(token_id)
    for source_payload in (signal_payload, payload):
        raw_positions = source_payload.get("positions_to_take")
        raw_positions = raw_positions if isinstance(raw_positions, list) else []
        for position in raw_positions:
            if not isinstance(position, dict):
                continue
            token_id = str(position.get("token_id") or "").strip()
            if token_id and token_id not in token_ids:
                token_ids.append(token_id)
    return token_ids


def _full_bundle_execution_required(payload: dict[str, Any], signal_payload: dict[str, Any]) -> bool:
    for source_payload in (payload, signal_payload):
        execution_plan = _payload_execution_plan(source_payload)
        metadata = execution_plan.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if _safe_bool(metadata.get("full_bundle_execution_required"), False):
            return True
    if not _safe_bool(signal_payload.get("is_guaranteed"), False):
        return False
    required_token_ids = _required_bundle_token_ids(payload, signal_payload)
    return len(required_token_ids) >= 2


def _bundle_position_shortfall(
    *,
    token_id: str | None,
    payload: dict[str, Any],
    signal_payload: dict[str, Any],
    wallet_positions_by_token: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    required_token_ids = _required_bundle_token_ids(payload, signal_payload)
    if not _full_bundle_execution_required(payload, signal_payload):
        return None
    if len(required_token_ids) < 2:
        return None
    normalized_token_id = str(token_id or "").strip()
    if normalized_token_id and normalized_token_id not in required_token_ids:
        return None
    observed_token_ids = [
        required_token_id
        for required_token_id in required_token_ids
        if _extract_wallet_position_size(wallet_positions_by_token.get(required_token_id)) > _WALLET_SIZE_EPSILON
    ]
    missing_token_ids = [required_token_id for required_token_id in required_token_ids if required_token_id not in observed_token_ids]
    if not observed_token_ids or not missing_token_ids:
        return None
    execution_plan = _payload_execution_plan(payload)
    if not execution_plan:
        execution_plan = _payload_execution_plan(signal_payload)
    metadata = execution_plan.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "required_token_ids": required_token_ids,
        "observed_token_ids": observed_token_ids,
        "missing_token_ids": missing_token_ids,
        "full_bundle_execution_mode": str(metadata.get("full_bundle_execution_mode") or "").strip() or None,
        "plan_id": str(execution_plan.get("plan_id") or "").strip() or None,
    }


def _pending_exit_fill_threshold(pending_exit: dict[str, Any]) -> float:
    parsed = safe_float(
        pending_exit.get("fill_ratio_threshold")
        if pending_exit.get("fill_ratio_threshold") is not None
        else pending_exit.get("fill_threshold_ratio")
    )
    if parsed is not None:
        return max(0.5, min(1.0, float(parsed)))

    required_exit_size = max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0)
    retry_count = int(safe_float(pending_exit.get("retry_count"), 0) or 0)
    close_trigger = str(pending_exit.get("close_trigger") or "").strip().lower()

    if required_exit_size <= 10.0:
        threshold = 0.88
    elif required_exit_size <= 25.0:
        threshold = 0.90
    elif required_exit_size <= 75.0:
        threshold = 0.93
    elif required_exit_size <= 200.0:
        threshold = 0.95
    else:
        threshold = 0.97

    if _is_rapid_close_trigger(close_trigger):
        threshold = min(threshold, 0.92)
    if retry_count >= 2:
        threshold = max(0.85, threshold - 0.03)
    return max(0.5, min(1.0, float(threshold)))


def _effective_exit_min_order_size_usd(min_order_size_usd: Any, close_trigger: Any) -> float:
    return 0.01


def _remaining_exit_size(
    *,
    required_exit_size: float,
    pending_exit: dict[str, Any],
    wallet_position_size: float,
) -> float:
    required = max(0.0, float(required_exit_size))
    already_filled = max(0.0, safe_float(pending_exit.get("filled_size"), 0.0) or 0.0)
    # Laddered exits accumulate fills across child orders; the parent's
    # ``filled_size`` is updated via aggregation but may lag a single tick.
    # Prefer the child aggregate when it's larger.
    children = pending_exit.get("children") if isinstance(pending_exit, dict) else None
    if isinstance(children, list) and children:
        agg = exit_executor.aggregate_children_fills(children)
        already_filled = max(already_filled, agg["filled_size"])
    remaining = max(0.0, required - already_filled)
    if required > 0.0 and remaining <= 0.0 and already_filled <= 0.0:
        remaining = required
    wallet_cap = wallet_position_size if wallet_position_size > _WALLET_SIZE_EPSILON else 0.0
    if wallet_cap > 0.0:
        if remaining <= 0.0:
            remaining = wallet_cap
        else:
            remaining = min(remaining, wallet_cap)
    return max(0.0, remaining)


def _pending_exit_provider_clob_id(pending_exit: dict[str, Any]) -> str:
    direct = str(pending_exit.get("provider_clob_order_id") or pending_exit.get("exit_order_clob_id") or "").strip()
    if direct:
        return direct
    fallback = str(pending_exit.get("exit_order_id") or "").strip()
    if not fallback:
        return ""
    if fallback.startswith("0x") or fallback.isdigit():
        return fallback
    try:
        cached = live_execution_service.get_order(fallback)
    except Exception:
        cached = None
    if cached is None:
        return ""
    return str(getattr(cached, "clob_order_id", "") or "").strip()


def _pending_exit_fill_evidence(pending_exit: dict[str, Any]) -> tuple[str, float, float, Optional[float]]:
    snapshot = pending_exit.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    candidates: list[Any] = [snapshot, pending_exit]
    provider_status = _resolved_snapshot_status(
        snapshot,
        snapshot.get("normalized_status"),
        pending_exit.get("provider_status"),
    )
    filled_size = max(
        0.0,
        _first_float_from_candidates(
            candidates,
            ("filled_size", "size_matched", "matched_size", "filled_shares", "executed_size"),
        )
        or 0.0,
    )
    filled_notional = max(
        0.0,
        _first_float_from_candidates(
            candidates,
            ("filled_notional_usd", "filled_notional", "executed_notional_usd", "matched_notional_usd"),
        )
        or 0.0,
    )
    average_fill_price = _first_float_from_candidates(
        candidates,
        ("average_fill_price", "avg_fill_price", "executed_price", "matched_price", "price"),
    )
    if filled_notional <= 0.0 and filled_size > 0.0 and average_fill_price is not None and average_fill_price > 0.0:
        filled_notional = filled_size * average_fill_price
    return provider_status, filled_size, filled_notional, average_fill_price


def _pending_exit_has_verified_terminal_fill(pending_exit: dict[str, Any]) -> bool:
    children = pending_exit.get("children") if isinstance(pending_exit, dict) else None
    if isinstance(children, list) and children:
        # Laddered exit: terminal when the executor reports completion AND
        # aggregated child fills account for at least the target size, OR
        # at least one child has filled and the rest are terminal.
        agg = exit_executor.aggregate_children_fills(children)
        if agg["filled_size"] > _WALLET_SIZE_EPSILON and exit_executor.is_exit_complete(pending_exit):
            return True
    provider_status, filled_size, filled_notional, average_fill_price = _pending_exit_fill_evidence(pending_exit)
    if provider_status not in {"filled", "matched", "executed"}:
        return False
    if filled_size > _WALLET_SIZE_EPSILON:
        return True
    if filled_notional > 0.0:
        return True
    return average_fill_price is not None and average_fill_price > 0.0


def _position_close_wallet_trade_id(position_close: dict[str, Any]) -> str:
    return str(position_close.get("wallet_trade_id") or "").strip()


def _position_close_has_terminal_external_authority(position_close: dict[str, Any]) -> bool:
    price_source = str(position_close.get("price_source") or "").strip().lower()
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if price_source in {"wallet_activity", "wallet_trade", "wallet_flat_override"}:
        return True
    if close_trigger in {"wallet_activity", "wallet_summary_recovery", "external_wallet_flatten"}:
        return True
    for key in (
        "wallet_activity_transaction_hash",
        "wallet_trade_id",
        "wallet_activity_id",
        "wallet_trade_timestamp",
        "wallet_activity_timestamp",
        "wallet_closed_position_timestamp",
    ):
        if str(position_close.get(key) or "").strip():
            return True
    return False


def _entry_size_for_terminal_close(row: TraderOrder, payload: dict[str, Any]) -> float:
    filled_notional, filled_size, fill_price = _extract_live_fill_metrics(payload)
    if filled_size > _WALLET_SIZE_EPSILON:
        return filled_size

    price = fill_price if fill_price is not None and fill_price > 0.0 else safe_float(getattr(row, "effective_price", None))
    if price is None or price <= 0.0:
        price = safe_float(getattr(row, "entry_price", None))
    notional = filled_notional if filled_notional > 0.0 else safe_float(getattr(row, "notional_usd", None))
    if notional is not None and notional > 0.0 and price is not None and price > 0.0:
        return max(0.0, float(notional) / float(price))
    return 0.0


def _position_close_filled_size(position_close: dict[str, Any]) -> float:
    return max(
        0.0,
        _first_float_from_candidates(
            [position_close],
            (
                "filled_size",
                "wallet_activity_size",
                "wallet_trade_size",
                "closed_size",
                "size",
                "shares",
            ),
        )
        or 0.0,
    )


def _position_close_evidence_size(position_close: dict[str, Any]) -> float:
    evidence_size = _first_float_from_candidates(
        [position_close],
        (
            "wallet_activity_total_size",
            "wallet_trade_total_size",
            "wallet_closed_position_total_size",
            "evidence_size",
            "total_size",
            "wallet_activity_size",
            "wallet_trade_size",
        ),
    )
    if evidence_size is not None and evidence_size > 0.0:
        return float(evidence_size)
    return _position_close_filled_size(position_close)


def _position_close_underfills_entry(
    row: TraderOrder,
    payload: dict[str, Any],
    position_close: dict[str, Any],
) -> bool:
    price_source = str(position_close.get("price_source") or "").strip().lower()
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if price_source not in {"wallet_activity", "wallet_trade", "provider_exit_fill"} and close_trigger not in {
        "wallet_activity",
        "wallet_summary_recovery",
        "external_wallet_flatten",
        "live_exit_fill",
        "manual_mark_to_market",
    }:
        return False

    close_size = _position_close_filled_size(position_close)
    if close_size <= _WALLET_SIZE_EPSILON:
        return False
    entry_size = _entry_size_for_terminal_close(row, payload)
    if entry_size <= _WALLET_SIZE_EPSILON:
        return False
    return close_size + _WALLET_SIZE_EPSILON < entry_size * 0.995


def _position_close_missing_terminal_fill_authority(
    row: TraderOrder,
    payload: dict[str, Any],
    position_close: dict[str, Any],
) -> bool:
    status_key = str(getattr(row, "status", None) or "").strip().lower()
    if status_key not in {"closed_win", "closed_loss"}:
        return False

    price_source = str(position_close.get("price_source") or "").strip().lower()
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if close_trigger in {"resolution", "resolution_inferred", "resolution_extreme_mark"}:
        return False
    if price_source in {"resolved_settlement", "wallet_redeemable_mark"}:
        return False

    pending_exit = payload.get("pending_live_exit")
    pending_exit = pending_exit if isinstance(pending_exit, dict) else {}
    if _pending_exit_has_verified_terminal_fill(pending_exit):
        return False
    if (
        _position_close_has_terminal_external_authority(position_close)
        and not _position_close_missing_durable_wallet_identity(position_close)
        and _position_close_filled_size(position_close) > _WALLET_SIZE_EPSILON
    ):
        return False
    return True


def _position_close_evidence_key(
    row: TraderOrder,
    payload: dict[str, Any],
    position_close: dict[str, Any],
) -> str:
    token_id = _extract_live_token_id(payload) or str(getattr(row, "market_id", "") or "").strip()
    trade_id = str(position_close.get("wallet_trade_id") or "").strip()
    trade_timestamp = str(position_close.get("wallet_trade_timestamp") or "").strip()
    if trade_id or trade_timestamp:
        parts = [f"trade_id:{trade_id}"] if trade_id else []
        if trade_timestamp:
            parts.append(f"timestamp:{trade_timestamp}")
        return "|".join(("wallet_trade", token_id, *parts))

    activity_tx = str(position_close.get("wallet_activity_transaction_hash") or "").strip()
    activity_id = str(position_close.get("wallet_activity_id") or "").strip()
    activity_timestamp = str(position_close.get("wallet_activity_timestamp") or "").strip()
    if activity_tx or activity_id or activity_timestamp:
        parts = [f"transactionHash:{activity_tx}"] if activity_tx else []
        if activity_id:
            parts.append(f"id:{activity_id}")
        if activity_timestamp:
            parts.append(f"timestamp:{activity_timestamp}")
        return "|".join(("wallet_activity", token_id, *parts))

    closed_position_timestamp = str(position_close.get("wallet_closed_position_timestamp") or "").strip()
    closed_position_total = safe_float(position_close.get("wallet_closed_position_total_size"))
    if closed_position_timestamp or (closed_position_total is not None and closed_position_total > 0.0):
        parts = [f"timestamp:{closed_position_timestamp}"] if closed_position_timestamp else []
        if closed_position_total is not None and closed_position_total > 0.0:
            parts.append(f"total:{closed_position_total:.8f}")
        return "|".join(("wallet_closed_position", token_id, *parts))
    return ""


def _position_close_missing_durable_wallet_identity(position_close: dict[str, Any]) -> bool:
    price_source = str(position_close.get("price_source") or "").strip().lower()
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if price_source == "wallet_trade" or close_trigger == "external_wallet_flatten":
        return not bool(
            str(
                position_close.get("wallet_trade_id")
                or position_close.get("wallet_activity_transaction_hash")
                or ""
            ).strip()
        )
    if price_source == "wallet_activity" or close_trigger == "wallet_activity":
        return not bool(
            str(
                position_close.get("wallet_activity_transaction_hash")
                or position_close.get("wallet_activity_id")
                or position_close.get("wallet_trade_id")
                or ""
            ).strip()
        )
    return False


def _pending_exit_is_live_fallback_manual_exit(pending_exit: dict[str, Any]) -> bool:
    trigger = str(pending_exit.get("close_trigger") or "").strip().lower()
    if trigger not in _LIVE_FALLBACK_MANUAL_EXIT_TRIGGERS:
        return False
    reason = str(pending_exit.get("reason") or "").strip().lower()
    if reason in _LIVE_FALLBACK_MANUAL_EXIT_REASONS:
        return True
    if trigger.startswith("grouped:") and "manual_mark_to_market" in trigger:
        return True
    return False


def _wallet_close_evidence_key(kind: str, token_id: str, evidence: dict[str, Any]) -> str:
    identity_parts: list[str] = []
    for key in (
        "transactionHash",
        "txHash",
        "tx_hash",
        "trade_id",
        "tradeID",
        "id",
        "order_id",
        "orderId",
        "timestamp",
        "created_at",
        "createdAt",
        "time",
    ):
        value = evidence.get(key)
        if isinstance(value, datetime):
            value = _iso_utc(value)
        value_text = str(value or "").strip()
        if value_text:
            identity_parts.append(f"{key}:{value_text}")
    if not identity_parts:
        return ""
    return "|".join((kind, str(token_id or "").strip(), *identity_parts))


def _extract_wallet_trade_token_id(trade: dict[str, Any]) -> str:
    return str(
        trade.get("asset_id") or trade.get("asset") or trade.get("token_id") or trade.get("tokenId") or ""
    ).strip()


def _extract_wallet_trade_side(trade: dict[str, Any]) -> str:
    return str(trade.get("side") or trade.get("trade_side") or trade.get("type") or "").strip().lower()


def _extract_wallet_trade_size(trade: dict[str, Any]) -> float:
    size = safe_float(trade.get("size"))
    if size is None:
        size = safe_float(trade.get("amount"))
    if size is None:
        size = safe_float(trade.get("shares"))
    if size is None:
        size = safe_float(trade.get("matched_size"))
    return max(0.0, float(size or 0.0))


def _extract_wallet_trade_price(trade: dict[str, Any]) -> Optional[float]:
    price = safe_float(trade.get("price"))
    if price is None:
        price = safe_float(trade.get("avg_price"))
    if price is None:
        price = safe_float(trade.get("limit_price"))
    if price is None or price < 0:
        return None
    return float(price)


def _parse_iso_utc_naive(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_wallet_trade_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    return _parse_iso_utc_naive(text)


def _terminal_reopen_blocks_wallet_close(
    payload: dict[str, Any],
    *,
    source: str,
    evidence_key: str = "",
    evidence_timestamp: Any = None,
) -> bool:
    if source not in {"wallet_activity", "wallet_trade", "closed_positions_api"}:
        return False
    marker = payload.get("terminal_close_reopen")
    if not isinstance(marker, dict):
        return False
    if str(marker.get("reopen_reason") or "").strip().lower() != "terminal_close_underfilled_entry":
        return False

    rejected_key = str(marker.get("rejected_close_evidence_key") or "").strip()
    if rejected_key and evidence_key and rejected_key == evidence_key:
        return True

    reopened_at = _parse_iso_utc_naive(marker.get("reopened_at"))
    if reopened_at is None:
        return True

    evidence_at: Optional[datetime]
    if isinstance(evidence_timestamp, datetime):
        evidence_at = (
            evidence_timestamp.astimezone(timezone.utc).replace(tzinfo=None)
            if evidence_timestamp.tzinfo is not None
            else evidence_timestamp
        )
    else:
        evidence_at = _parse_wallet_trade_time(evidence_timestamp)
    if evidence_at is None:
        return True
    return evidence_at <= reopened_at + timedelta(seconds=1.0)


def _row_entry_anchor_naive(row: TraderOrder) -> Optional[datetime]:
    for value in (getattr(row, "created_at", None), getattr(row, "executed_at", None), getattr(row, "updated_at", None)):
        if not isinstance(value, datetime):
            continue
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    return None


def _wallet_history_identity_key(payload: dict[str, Any], direction: str) -> str:
    cond_key = _extract_condition_outcome_key(payload, direction)
    if cond_key:
        return cond_key
    return _extract_live_token_id(payload)


def _row_has_wallet_trade_fill_authority(row: TraderOrder, payload: dict[str, Any]) -> bool:
    filled_notional, filled_size, _average_fill_price = _extract_live_fill_metrics(payload)
    if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
        return True
    entry_fill_recovery = payload.get("entry_fill_recovery")
    if isinstance(entry_fill_recovery, dict):
        recovery_source = str(entry_fill_recovery.get("source") or "").strip().lower()
        recovery_size = max(0.0, safe_float(entry_fill_recovery.get("size"), 0.0) or 0.0)
        recovery_price = safe_float(entry_fill_recovery.get("price"))
        if recovery_source == "wallet_trade_history" and recovery_size > _WALLET_SIZE_EPSILON and recovery_price:
            return True
    wallet_entry_fill_repair = payload.get("wallet_entry_fill_repair")
    if isinstance(wallet_entry_fill_repair, dict):
        repair_size = max(0.0, safe_float(wallet_entry_fill_repair.get("size"), 0.0) or 0.0)
        repair_price = safe_float(wallet_entry_fill_repair.get("price"))
        if repair_size > _WALLET_SIZE_EPSILON and repair_price:
            return True
    verification_status = str(getattr(row, "verification_status", None) or "").strip().lower()
    return verification_status == "venue_fill"


def _row_can_recover_entry_from_wallet_history(row: TraderOrder, payload: dict[str, Any]) -> bool:
    if _row_has_wallet_trade_fill_authority(row, payload):
        return True
    if str(getattr(row, "provider_clob_order_id", None) or payload.get("provider_clob_order_id") or "").strip():
        return True
    if str(getattr(row, "provider_order_id", None) or payload.get("provider_order_id") or "").strip():
        return True
    live_wallet_authority = payload.get("live_wallet_authority")
    if isinstance(live_wallet_authority, dict) and str(live_wallet_authority.get("live_trading_order_id") or "").strip():
        return True
    verification_status = str(getattr(row, "verification_status", None) or "").strip().lower()
    return verification_status in {"venue_order", "wallet_position"}


def _select_aggregate_wallet_backfill_order_ids(rows: list[TraderOrder]) -> set[str]:
    counts_by_identity: dict[str, int] = {}
    identity_by_order: dict[str, str] = {}
    for row in rows:
        payload = dict(getattr(row, "payload_json", None) or {})
        identity_key = _wallet_history_identity_key(payload, str(getattr(row, "direction", None) or ""))
        if not identity_key:
            continue
        order_id = str(getattr(row, "id", None) or "").strip()
        if not order_id:
            continue
        identity_by_order[order_id] = identity_key
        counts_by_identity[identity_key] = counts_by_identity.get(identity_key, 0) + 1
    return {
        order_id
        for order_id, identity_key in identity_by_order.items()
        if counts_by_identity.get(identity_key, 0) == 1
    }


def _build_wallet_buy_trade_matches(
    rows: list[TraderOrder],
    wallet_buy_trades_by_token: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    rows_by_token: dict[str, list[TraderOrder]] = {}
    for row in rows:
        payload = dict(getattr(row, "payload_json", None) or {})
        token_id = _extract_live_token_id(payload)
        if not token_id or not _row_can_recover_entry_from_wallet_history(row, payload):
            continue
        filled_notional, filled_size, _ = _extract_live_fill_metrics(payload)
        if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
            continue
        rows_by_token.setdefault(token_id, []).append(row)

    anchor_tolerance = timedelta(seconds=30)
    for token_id, token_rows in rows_by_token.items():
        trades = [dict(trade) for trade in (wallet_buy_trades_by_token.get(token_id) or []) if isinstance(trade, dict)]
        if not trades:
            continue
        sorted_rows = sorted(
            token_rows,
            key=lambda row: (
                _row_entry_anchor_naive(row) or datetime.min,
                str(getattr(row, "id", None) or ""),
            ),
        )
        sorted_trades = sorted(
            trades,
            key=lambda trade: (
                _parse_wallet_trade_time(
                    trade.get("timestamp") or trade.get("created_at") or trade.get("createdAt") or trade.get("time")
                )
                or datetime.min,
                str(trade.get("transactionHash") or trade.get("id") or ""),
            ),
        )
        trade_index = 0
        for row in sorted_rows:
            anchor = _row_entry_anchor_naive(row)
            while trade_index < len(sorted_trades):
                trade = sorted_trades[trade_index]
                trade_ts = _parse_wallet_trade_time(
                    trade.get("timestamp") or trade.get("created_at") or trade.get("createdAt") or trade.get("time")
                )
                if anchor is not None and trade_ts is not None and trade_ts < (anchor - anchor_tolerance):
                    trade_index += 1
                    continue
                matches[str(getattr(row, "id", None) or "")] = trade
                trade_index += 1
                break
    return matches


def _terminal_wallet_fill_repair_state(
    row: TraderOrder,
    *,
    wallet_buy_trade_by_order_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], float, float, str] | None:
    payload = dict(getattr(row, "payload_json", None) or {})
    position_close = payload.get("position_close")
    if not isinstance(position_close, dict):
        return None
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if close_trigger not in {"wallet_activity", "wallet_summary_recovery", "external_wallet_flatten"}:
        return None
    if not _row_can_recover_entry_from_wallet_history(row, payload):
        return None
    filled_notional, filled_size, _ = _extract_live_fill_metrics(payload)
    if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
        return None
    wallet_entry_trade = wallet_buy_trade_by_order_id.get(str(getattr(row, "id", None) or ""))
    if not isinstance(wallet_entry_trade, dict):
        return None
    wallet_entry_trade_size = _extract_wallet_trade_size(wallet_entry_trade)
    if wallet_entry_trade_size <= _WALLET_SIZE_EPSILON:
        return None
    recovered_price = _extract_wallet_trade_price(wallet_entry_trade)
    if recovered_price is None or recovered_price <= 0.0:
        recovered_price = safe_float(getattr(row, "effective_price", None))
    if recovered_price is None or recovered_price <= 0.0:
        recovered_price = safe_float(getattr(row, "entry_price", None))
    if recovered_price is None or recovered_price <= 0.0:
        return None
    recovered_notional = wallet_entry_trade_size * recovered_price
    entry_fill_recovery = payload.get("entry_fill_recovery")
    recovered_source = ""
    if isinstance(entry_fill_recovery, dict):
        recovered_source = str(entry_fill_recovery.get("source") or "").strip().lower()
    existing_notional = safe_float(getattr(row, "notional_usd", None), 0.0) or 0.0
    existing_effective_price = safe_float(getattr(row, "effective_price", None), 0.0) or 0.0
    if (
        recovered_source == "wallet_trade_history"
        and abs(existing_notional - recovered_notional) <= 1e-9
        and abs(existing_effective_price - recovered_price) <= 1e-9
    ):
        return None
    return payload, wallet_entry_trade, recovered_notional, recovered_price, close_trigger


def _extract_wallet_activity_token_id(activity: dict[str, Any]) -> str:
    return str(
        activity.get("asset")
        or activity.get("asset_id")
        or activity.get("token_id")
        or activity.get("tokenId")
        or ""
    ).strip()


def _extract_wallet_activity_type(activity: dict[str, Any]) -> str:
    return str(activity.get("type") or activity.get("activityType") or "").strip().lower()


def _extract_wallet_activity_side(activity: dict[str, Any]) -> str:
    return str(activity.get("side") or activity.get("trade_side") or "").strip().lower()


def _extract_wallet_activity_size(activity: dict[str, Any]) -> float:
    size = safe_float(activity.get("size"))
    if size is None:
        size = safe_float(activity.get("amount"))
    if size is None:
        size = safe_float(activity.get("shares"))
    return max(0.0, float(size or 0.0))


def _extract_wallet_activity_price(activity: dict[str, Any]) -> Optional[float]:
    price = safe_float(activity.get("price"))
    if price is None:
        price = safe_float(activity.get("avgPrice"))
    if price is None:
        price = safe_float(activity.get("avg_price"))
    if price is None or price < 0.0:
        return None
    return float(price)


def _extract_wallet_activity_tx_hash(activity: dict[str, Any]) -> str:
    return str(activity.get("transactionHash") or activity.get("txHash") or activity.get("tx_hash") or "").strip()


def _parse_wallet_activity_time(activity: dict[str, Any]) -> Optional[datetime]:
    return _parse_wallet_trade_time(
        activity.get("timestamp")
        if activity.get("timestamp") is not None
        else activity.get("createdAt") if activity.get("createdAt") is not None
        else activity.get("created_at")
    )


def _wallet_activity_condition_outcome_key(activity: dict[str, Any]) -> str:
    condition_id = str(activity.get("conditionId") or activity.get("condition_id") or "").strip()
    if not condition_id:
        return ""
    outcome_idx = activity.get("outcomeIndex")
    if outcome_idx is None:
        outcome_idx = activity.get("outcome_index")
    if outcome_idx is None:
        return ""
    return f"{condition_id}:{outcome_idx}"


def _wallet_activity_is_close_authority(activity: dict[str, Any]) -> bool:
    activity_type = _extract_wallet_activity_type(activity)
    if activity_type == "trade":
        return _extract_wallet_activity_side(activity) == "sell" and _extract_wallet_activity_size(activity) > _WALLET_SIZE_EPSILON
    return activity_type in {"merge", "redeem", "claim", "convert", "conversion"}


def _apply_position_close_verification(
    session: AsyncSession,
    *,
    row: TraderOrder,
    now: datetime,
    event_type: str,
    payload_json: dict[str, Any],
) -> None:
    previous_status = str(row.verification_status or "")
    previous_source = str(row.verification_source or "")
    previous_reason = row.verification_reason
    previous_order_id = str(row.provider_order_id or "")
    previous_clob_order_id = str(row.provider_clob_order_id or "")
    previous_wallet = str(row.execution_wallet_address or "")
    previous_tx_hash = str(row.verification_tx_hash or "")

    derived = derive_trader_order_verification(
        mode=row.mode,
        status=row.status,
        reason=row.reason,
        payload=payload_json,
    )
    next_status = str(derived.get("verification_status") or TRADER_ORDER_VERIFICATION_LOCAL)
    next_source = str(derived.get("verification_source") or "").strip()
    next_reason = str(derived.get("verification_reason") or "").strip() or None
    next_order_id = str(derived.get("provider_order_id") or "").strip()
    next_clob_order_id = str(derived.get("provider_clob_order_id") or "").strip()
    next_wallet = str(derived.get("execution_wallet_address") or "").strip().lower()
    next_tx_hash = str(derived.get("verification_tx_hash") or "").strip()
    material_changed = (
        previous_status != next_status
        or previous_source != next_source
        or previous_reason != next_reason
        or (next_order_id and previous_order_id != next_order_id)
        or (next_clob_order_id and previous_clob_order_id != next_clob_order_id)
        or (next_wallet and previous_wallet != next_wallet)
        or (next_tx_hash and previous_tx_hash != next_tx_hash)
    )
    # Don't downgrade Polymarket-truth-verified rows. The general
    # close-verification path uses force=True which would clobber
    # wallet_activity status set by polymarket_trade_verifier.
    _row_status_lower = str(getattr(row, "status", None) or "").strip().lower()
    _is_resolved_status_for_guard = _row_status_lower in {
        "resolved", "resolved_win", "resolved_loss",
        "closed_win", "closed_loss", "win", "loss",
    }
    _existing_is_verified = (
        previous_status.strip().lower() == "wallet_activity"
        and _is_resolved_status_for_guard
    )
    if material_changed and _existing_is_verified:
        material_changed = False
    if material_changed or row.verified_at is None:
        apply_trader_order_verification(
            row,
            verification_status=next_status,
            verification_source=next_source or None,
            verification_reason=next_reason,
            provider_order_id=next_order_id or None,
            provider_clob_order_id=next_clob_order_id or None,
            execution_wallet_address=next_wallet or None,
            verification_tx_hash=next_tx_hash or None,
            verified_at=now,
            force=True,
        )
    if material_changed:
        append_trader_order_verification_event(
            session,
            trader_order_id=str(row.id),
            verification_status=str(row.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
            source=row.verification_source,
            event_type=event_type,
            reason=row.verification_reason,
            provider_order_id=row.provider_order_id,
            provider_clob_order_id=row.provider_clob_order_id,
            execution_wallet_address=row.execution_wallet_address,
            tx_hash=row.verification_tx_hash,
            payload_json={},
            created_at=now,
        )


def _active_row_has_unfilled_terminal_provider_failure(row: TraderOrder, payload: dict[str, Any]) -> bool:
    wallet_entry_fill_repair = payload.get("wallet_entry_fill_repair")
    if isinstance(wallet_entry_fill_repair, dict):
        return False
    entry_fill_recovery = payload.get("entry_fill_recovery")
    if (
        isinstance(entry_fill_recovery, dict)
        and str(entry_fill_recovery.get("source") or "").strip().lower() == "wallet_trade_history"
    ):
        return False
    filled_notional, filled_size, _average_fill_price = _extract_live_fill_metrics(payload)
    if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
        return False

    provider_status = _provider_snapshot_status(payload)
    if provider_status not in {"failed", "cancelled", "expired", "rejected", "error"}:
        return False

    verification_status = str(getattr(row, "verification_status", None) or "").strip().lower()
    if verification_status in {"venue_fill", "wallet_position"}:
        return False

    live_wallet_authority = payload.get("live_wallet_authority")
    if isinstance(live_wallet_authority, dict):
        authority_source = str(live_wallet_authority.get("source") or "").strip().lower()
        if authority_source == "wallet_positions_api":
            return False

    return True


def _provider_failure_terminal_status(row: TraderOrder, payload: dict[str, Any]) -> str:
    provider_status = _provider_snapshot_status(payload)
    error_text = str(getattr(row, "error_message", None) or payload.get("error_message") or "").strip().lower()
    if "no orders found to match with fak order" in error_text:
        return "rejected"
    if provider_status in {"cancelled", "expired"}:
        return "cancelled"
    return "failed"


def _failed_terminal_row_can_reopen_from_wallet_position(row: TraderOrder, payload: dict[str, Any]) -> bool:
    filled_notional, filled_size, _average_fill_price = _extract_live_fill_metrics(payload)
    live_wallet_authority = payload.get("live_wallet_authority")
    authority_source = ""
    if isinstance(live_wallet_authority, dict):
        authority_source = str(live_wallet_authority.get("source") or "").strip().lower()
    if _active_row_has_unfilled_terminal_provider_failure(row, payload):
        return False

    if filled_notional > 0.0 or filled_size > _WALLET_SIZE_EPSILON:
        return True

    verification_status = str(getattr(row, "verification_status", None) or "").strip().lower()
    if verification_status in {"venue_fill", "wallet_position"}:
        return True

    if not isinstance(live_wallet_authority, dict):
        return False

    if authority_source == "wallet_positions_api":
        return True

    provider_status = _provider_snapshot_status(payload)
    return provider_status in {"filled", "partially_filled", "open", "submitted", "pending"}


def _terminal_row_has_wallet_position_authority(row: TraderOrder, payload: dict[str, Any]) -> bool:
    verification_status = str(getattr(row, "verification_status", None) or "").strip().lower()
    if verification_status in {"venue_order", "venue_fill", "wallet_position"}:
        return True

    verification_source = str(getattr(row, "verification_source", None) or "").strip().lower()
    if verification_source in {"live_order_ack", "wallet_positions_api", "wallet_activity_api", "wallet_trade_history"}:
        return True

    live_wallet_authority = payload.get("live_wallet_authority")
    if isinstance(live_wallet_authority, dict):
        authority_source = str(live_wallet_authority.get("source") or "").strip().lower()
        if authority_source in {"live_trading_orders", "wallet_positions_api", "wallet_trade_history"}:
            return True

    if str(getattr(row, "provider_clob_order_id", None) or "").strip():
        return True
    if str(getattr(row, "provider_order_id", None) or "").strip():
        return True
    return _provider_reconciliation_has_order_evidence(row, payload)


def _apply_wallet_position_reopen(
    *,
    row: TraderOrder,
    payload: dict[str, Any],
    wallet_position: dict[str, Any],
    wallet_key: str,
    now: datetime,
    reopen_reason: str,
    allocated_size: Optional[float] = None,
    allocated_notional_usd: Optional[float] = None,
) -> tuple[float, float, Optional[float], Optional[float], dict[str, Any]]:
    wallet_size, wallet_notional, wallet_entry_price, wallet_mark_price = _wallet_position_notional(wallet_position)
    if wallet_size <= _WALLET_SIZE_EPSILON:
        return wallet_size, wallet_notional, wallet_entry_price, wallet_mark_price, payload

    # Don't reopen rows that polymarket_trade_verifier already
    # closed and verified against actual on-chain settlement /
    # trade data. Wallet position lingering after resolution
    # (winning shares pending redemption to USDC) is normal — the
    # bot used to interpret that as "position is still open" and
    # reopen it, which downgraded verification_status from
    # wallet_activity to wallet_position. The DB-layer guard then
    # NULL'd actual_profit, erasing the verified P&L.
    existing_verification = str(getattr(row, "verification_status", None) or "").strip().lower()
    existing_status = str(getattr(row, "status", None) or "").strip().lower()
    is_resolved_status = existing_status in {
        "resolved", "resolved_win", "resolved_loss",
        "closed_win", "closed_loss", "win", "loss",
    }
    if existing_verification == "wallet_activity" and is_resolved_status:
        return wallet_size, wallet_notional, wallet_entry_price, wallet_mark_price, payload

    reopened_size = float(allocated_size) if allocated_size is not None and allocated_size > 0.0 else wallet_size
    reopened_size = min(wallet_size, reopened_size)
    if allocated_notional_usd is not None and allocated_notional_usd > 0.0:
        reopened_notional = float(allocated_notional_usd)
    elif wallet_entry_price is not None and wallet_entry_price > 0.0:
        reopened_notional = reopened_size * wallet_entry_price
    elif wallet_size > _WALLET_SIZE_EPSILON and wallet_notional > 0.0:
        reopened_notional = wallet_notional * min(1.0, reopened_size / wallet_size)
    else:
        reopened_notional = wallet_notional

    if reopened_notional > 0.0:
        row.notional_usd = float(reopened_notional)
    if wallet_entry_price is not None and wallet_entry_price > 0.0:
        row.entry_price = float(wallet_entry_price)
        row.effective_price = float(wallet_entry_price)
    row.status = "open"
    row.actual_profit = None
    row.error_message = None
    row.updated_at = now
    row.verification_status = "wallet_position"
    row.verification_source = "wallet_positions_api"

    provider_reconciliation = payload.get("provider_reconciliation")
    provider_reconciliation = provider_reconciliation if isinstance(provider_reconciliation, dict) else {}
    snapshot = provider_reconciliation.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    snapshot.update(
        {
            "normalized_status": "filled",
            "status": "filled",
            "asset_id": str(wallet_position.get("token_id") or wallet_position.get("asset") or wallet_key),
            "token_id": str(wallet_position.get("token_id") or wallet_position.get("asset") or wallet_key),
            "filled_size": float(reopened_size),
            "filled_notional_usd": float(reopened_notional),
            "average_fill_price": float(wallet_entry_price) if wallet_entry_price is not None else None,
            "source": "wallet_position_aggregate",
        }
    )
    provider_reconciliation.update(
        {
            "snapshot": snapshot,
            "snapshot_status": "filled",
            "mapped_status": "executed",
            "filled_size": float(reopened_size),
            "filled_notional_usd": float(reopened_notional),
            "average_fill_price": float(wallet_entry_price) if wallet_entry_price is not None else None,
            "source": "wallet_positions_api",
            "reconciled_at": _iso_utc(now),
        }
    )
    token_id = str(wallet_position.get("token_id") or wallet_position.get("asset") or wallet_key).strip()
    payload["provider_reconciliation"] = provider_reconciliation
    payload["live_wallet_authority"] = {
        "source": "wallet_positions_api",
        "token_id": token_id or wallet_key,
        "wallet_address": str(
            wallet_position.get("wallet_address")
            or wallet_position.get("proxyWallet")
            or payload.get("execution_wallet_address")
            or ""
        ).strip()
        or None,
        "recovered_at": _iso_utc(now),
    }
    payload["wallet_position_reopen"] = {
        "reopened_at": _iso_utc(now),
        "reopen_reason": reopen_reason,
        "wallet_key": wallet_key,
        "aggregate_size": float(wallet_size),
        "aggregate_notional_usd": float(wallet_notional),
        "reopened_size": float(reopened_size),
        "reopened_notional_usd": float(reopened_notional),
        "average_entry_price": float(wallet_entry_price) if wallet_entry_price is not None else None,
        "current_mark_price": float(wallet_mark_price) if wallet_mark_price is not None else None,
    }
    if token_id:
        payload["token_id"] = token_id
        payload["selected_token_id"] = token_id
    position_state = payload.get("position_state")
    position_state = position_state if isinstance(position_state, dict) else {}
    if wallet_mark_price is not None and wallet_mark_price >= 0.0:
        position_state.update(
            {
                "last_mark_price": float(wallet_mark_price),
                "last_mark_source": "wallet_position",
                "last_marked_at": _iso_utc(now),
            }
        )
        payload["position_state"] = position_state
    payload.pop("position_close", None)
    row.payload_json = payload
    return reopened_size, reopened_notional, wallet_entry_price, wallet_mark_price, payload


def _parse_market_end_time_naive(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    return _parse_iso_utc_naive(value)


def _extract_market_end_time_naive(market_info: Any) -> Optional[datetime]:
    if not isinstance(market_info, dict):
        return None
    for key in (
        "end_time",
        "endTime",
        "end_date",
        "endDate",
        "market_end_time",
        "expiration_time",
    ):
        parsed = _parse_market_end_time_naive(market_info.get(key))
        if parsed is not None:
            return parsed
    return None


def _market_seconds_left(market_info: Any, now: datetime) -> Optional[float]:
    if not isinstance(market_info, dict):
        return None
    for key in ("seconds_left", "secondsLeft", "seconds_until_close", "secondsUntilClose"):
        parsed = safe_float(market_info.get(key), None)
        if parsed is not None and parsed >= 0.0:
            return float(parsed)
    end_time = _extract_market_end_time_naive(market_info)
    if end_time is None:
        return None
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    return max(0.0, (end_time - now_naive).total_seconds())


def _market_end_time_iso(market_info: Any) -> Optional[str]:
    end_time = _extract_market_end_time_naive(market_info)
    if end_time is None:
        return None
    return _iso_utc(end_time)


async def _resolve_execution_wallet_address() -> str:
    wallet = ""
    try:
        await live_execution_service.ensure_initialized()
    except Exception:
        pass
    try:
        wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet
    try:
        runtime_signature_type = getattr(live_execution_service, "_balance_signature_type", None)
        signature_type = (
            int(runtime_signature_type)
            if isinstance(runtime_signature_type, int)
            else int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1))
        )
    except Exception:
        signature_type = 1
    try:
        wallet = str(live_execution_service._funder_for_signature_type(signature_type) or "").strip()
    except Exception:
        wallet = ""
    if not wallet:
        wallet = str(live_execution_service._get_wallet_address() or "").strip()
    # DB fallback: look up wallet from live_trading_positions if service not initialized
    if not wallet:
        try:
            from models.database import AsyncSessionLocal, LiveTradingPosition
            from sqlalchemy import select

            async with AsyncSessionLocal() as _session:
                row = (
                    await _session.execute(
                        select(LiveTradingPosition.wallet_address)
                        .where(LiveTradingPosition.wallet_address.isnot(None))
                        .limit(1)
                    )
                ).scalar()
                if row:
                    wallet = str(row).strip()
                    logger.info("wallet address resolved from DB fallback: %s", wallet[:12] + "...")
        except Exception:
            pass
    return wallet


async def _load_execution_wallet_positions_by_token() -> dict[str, dict[str, Any]]:
    global _wallet_positions_cache, _wallet_positions_last_refresh_succeeded

    # Hot-path read from WalletStateCache.  When the orchestrator is
    # in the no-REST scope, we serve cached state regardless of TTL —
    # the reconciliation worker is the single REST writer that keeps
    # the cache fresh on its own cadence.
    if is_hot_path_no_rest():
        ws_cache = get_wallet_state_cache()
        snapshot = ws_cache.positions_by_token()
        if snapshot:
            return snapshot
        # Fall through to existing in-process TTL cache (DB-backed,
        # not REST) — last-resort fallback for the very-first cycle
        # before reconciliation has seeded.  Never call REST.
        return cached_data if (cached_data := _wallet_positions_cache[1]) else {}

    cached_at, cached_data = _wallet_positions_cache
    if _wallet_positions_last_refresh_succeeded and (_time_mod.monotonic() - cached_at) < _WALLET_CACHE_TTL_SECONDS:
        return cached_data

    wallet = await _resolve_execution_wallet_address()
    if not wallet:
        _wallet_positions_last_refresh_succeeded = False
        return {}

    wallet_key = str(wallet).strip().lower()
    try:
        async with AsyncSessionLocal() as session:
            position_rows = list(
                (
                    await session.execute(
                        select(LiveTradingPosition)
                        .where(func.lower(func.coalesce(LiveTradingPosition.wallet_address, "")) == wallet_key)
                        .order_by(LiveTradingPosition.updated_at.desc(), LiveTradingPosition.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )
    except Exception:
        _wallet_positions_last_refresh_succeeded = False
        return dict(cached_data or {})

    by_token: dict[str, dict[str, Any]] = {}
    for row in position_rows:
        position = _live_trading_position_to_wallet_position(row)
        if not isinstance(position, dict):
            continue
        token_id = str(position.get("asset") or position.get("asset_id") or position.get("token_id") or "").strip()
        if not token_id:
            continue
        by_token[token_id] = position
        cid = str(position.get("conditionId") or position.get("condition_id") or "").strip()
        outcome_idx = position.get("outcomeIndex")
        if cid and outcome_idx is not None:
            by_token[f"{cid}:{outcome_idx}"] = position
    _wallet_positions_cache = (_time_mod.monotonic(), by_token)
    _wallet_positions_last_refresh_succeeded = True
    return by_token


async def _load_execution_wallet_closed_positions_by_token() -> dict[str, dict[str, Any]]:
    global _wallet_closed_positions_cache, _wallet_closed_positions_last_refresh_succeeded

    # Hot-path read from WalletStateCache.
    if is_hot_path_no_rest():
        ws_cache = get_wallet_state_cache()
        snapshot = ws_cache.closed_positions_by_token()
        if snapshot:
            return snapshot
        return _wallet_closed_positions_cache[1] or {}

    cached_at, cached_data = _wallet_closed_positions_cache
    if _wallet_closed_positions_last_refresh_succeeded and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
        return cached_data

    # Single-flight: serialize concurrent cache misses so we don't
    # fire duplicate paginated Polymarket calls under the
    # ``asyncio.gather`` that drives ``external_io``.
    async with _get_wallet_closed_positions_lock():
        # Re-check cache after acquiring the lock — a peer caller may
        # have just populated it while we were queued.
        cached_at, cached_data = _wallet_closed_positions_cache
        if _wallet_closed_positions_last_refresh_succeeded and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
            return cached_data

        # Hot-path gate: orchestrator must not block on REST.  Return
        # whatever cache we have (even stale or empty) — the reconciliation
        # worker is responsible for refreshing this off the hot path.
        if not allow_polymarket_rest_call("closed_positions"):
            return cached_data or {}

        wallet = await _resolve_execution_wallet_address()
        if not wallet:
            _wallet_closed_positions_last_refresh_succeeded = False
            return {}

        try:
            positions = await polymarket_client.get_closed_positions_paginated(
                wallet,
                max_positions=_WALLET_HISTORY_MAX_CLOSED_POSITIONS,
            )
        except Exception:
            _wallet_closed_positions_last_refresh_succeeded = False
            return {}

        by_token: dict[str, dict[str, Any]] = {}
        for position in positions:
            if not isinstance(position, dict):
                continue
            token_id = str(position.get("asset") or position.get("asset_id") or position.get("token_id") or "").strip()
            if token_id:
                by_token[token_id] = position
            cid = str(position.get("conditionId") or position.get("condition_id") or "").strip()
            outcome_idx = position.get("outcomeIndex")
            if cid and outcome_idx is not None:
                by_token[f"{cid}:{outcome_idx}"] = position
        _wallet_closed_positions_cache = (_time_mod.monotonic(), by_token)
        _wallet_closed_positions_last_refresh_succeeded = True
        return by_token


async def _load_execution_wallet_trade_history() -> list[dict[str, Any]]:
    global _wallet_trades_cache
    cached_at, cached_data = _wallet_trades_cache
    if cached_data and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
        return list(cached_data)

    # Single-flight: ``_load_execution_wallet_recent_buy_trades_by_token``
    # and ``_load_execution_wallet_recent_sell_trades_by_token`` both
    # call this concurrently inside ``reconcile_live_positions``'s
    # ``asyncio.gather``.  Without serialization, a single cache miss
    # fired TWO paginated ``get_wallet_trades_paginated`` calls in
    # parallel against an already-loaded API.
    async with _get_wallet_trades_lock():
        cached_at, cached_data = _wallet_trades_cache
        if cached_data and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
            return list(cached_data)

        # Hot-path gate: never issue REST from the orchestrator hot loop.
        if not allow_polymarket_rest_call("wallet_trades"):
            return list(cached_data) if cached_data else []

        wallet = await _resolve_execution_wallet_address()
        if not wallet:
            return []
        try:
            trades = await polymarket_client.get_wallet_trades_paginated(
                wallet,
                max_trades=_WALLET_HISTORY_MAX_TRADES,
                page_size=_WALLET_HISTORY_TRADE_PAGE_SIZE,
            )
        except Exception:
            return []
        if not isinstance(trades, list):
            return []

        normalized = [dict(trade) for trade in trades if isinstance(trade, dict)]
        _wallet_trades_cache = (_time_mod.monotonic(), normalized)
        return list(normalized)


async def _load_execution_wallet_recent_buy_trades_by_token() -> dict[str, list[dict[str, Any]]]:
    global _wallet_buy_trades_cache
    cached_at, cached_data = _wallet_buy_trades_cache
    if cached_data and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
        return cached_data

    trades = await _load_execution_wallet_trade_history()
    if not isinstance(trades, list):
        return {}

    market_cache_by_condition: dict[str, Optional[dict[str, Any]]] = {}
    condition_ids_to_fetch: set[str] = set()
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if not _extract_wallet_trade_token_id(trade):
            cid = str(trade.get("conditionId") or trade.get("condition_id") or trade.get("market") or "").strip()
            if cid:
                condition_ids_to_fetch.add(cid)

    if condition_ids_to_fetch and allow_polymarket_rest_call("market_by_condition_id_batch_buys"):
        batch_size = 10

        async def _fetch_condition(cid: str) -> tuple[str, Optional[dict[str, Any]]]:
            try:
                return cid, await polymarket_client.get_market_by_condition_id(cid)
            except Exception:
                return cid, None

        cid_list = sorted(condition_ids_to_fetch)
        for index in range(0, len(cid_list), batch_size):
            batch = cid_list[index : index + batch_size]
            results = await asyncio.gather(*[_fetch_condition(condition_id) for condition_id in batch])
            for cid, info in results:
                market_cache_by_condition[cid] = info

    def _infer_trade_token_id(trade: dict[str, Any]) -> str:
        condition_id = str(trade.get("conditionId") or trade.get("condition_id") or trade.get("market") or "").strip()
        if not condition_id:
            return ""

        market_info = market_cache_by_condition.get(condition_id)
        if not isinstance(market_info, dict):
            return ""

        token_ids_raw = market_info.get("token_ids")
        if not isinstance(token_ids_raw, list):
            token_ids_raw = market_info.get("tokenIds")
        if not isinstance(token_ids_raw, list):
            return ""
        token_ids = [str(token_id or "").strip() for token_id in token_ids_raw]
        if not token_ids:
            return ""

        outcomes_raw = market_info.get("outcomes")
        outcomes: list[str] = []
        if isinstance(outcomes_raw, list):
            outcomes = [str(outcome or "").strip().lower() for outcome in outcomes_raw]

        outcome_idx = safe_float(
            trade.get("outcomeIndex") if trade.get("outcomeIndex") is not None else trade.get("outcome_index")
        )
        if outcome_idx is not None:
            idx = int(outcome_idx)
            if 0 <= idx < len(token_ids):
                return token_ids[idx]

        outcome_text = str(trade.get("outcome") or trade.get("token_outcome") or "").strip().lower()
        if outcome_text and outcomes:
            for idx, outcome_label in enumerate(outcomes):
                if outcome_label == outcome_text and idx < len(token_ids):
                    return token_ids[idx]

        return ""

    buys_by_token: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        token_id = _extract_wallet_trade_token_id(trade)
        if not token_id:
            token_id = _infer_trade_token_id(trade)
        if not token_id:
            continue
        if _extract_wallet_trade_side(trade) != "buy":
            continue
        size = _extract_wallet_trade_size(trade)
        if size <= 0.0:
            continue
        price = _extract_wallet_trade_price(trade)
        buys_by_token.setdefault(token_id, []).append(
            {
                "token_id": token_id,
                "size": size,
                "price": price,
                "timestamp": _parse_wallet_trade_time(
                    trade.get("timestamp") or trade.get("created_at") or trade.get("createdAt") or trade.get("time")
                ),
                "transactionHash": str(
                    trade.get("transactionHash") or trade.get("txHash") or trade.get("tx_hash") or ""
                ).strip()
                or None,
                "raw": dict(trade),
            }
        )

    for token_id, token_trades in buys_by_token.items():
        token_trades.sort(
            key=lambda trade: (
                trade.get("timestamp") or datetime.min,
                str(trade.get("transactionHash") or ""),
            )
        )

    _wallet_buy_trades_cache = (_time_mod.monotonic(), buys_by_token)
    return buys_by_token


async def _load_execution_wallet_recent_sell_trades_by_token() -> dict[str, dict[str, Any]]:
    global _wallet_sell_trades_cache
    cached_at, cached_data = _wallet_sell_trades_cache
    if cached_data and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
        return cached_data

    trades = await _load_execution_wallet_trade_history()
    if not isinstance(trades, list):
        return {}

    latest_by_token: dict[str, dict[str, Any]] = {}
    market_cache_by_condition: dict[str, Optional[dict[str, Any]]] = {}

    # Pre-fetch all unique condition_ids in parallel to avoid sequential API calls.
    condition_ids_to_fetch: set[str] = set()
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if not _extract_wallet_trade_token_id(trade):
            cid = str(trade.get("conditionId") or trade.get("condition_id") or trade.get("market") or "").strip()
            if cid:
                condition_ids_to_fetch.add(cid)

    if condition_ids_to_fetch and allow_polymarket_rest_call("market_by_condition_id_batch_sells"):
        _BATCH = 10

        async def _fetch_condition(cid: str) -> tuple[str, Optional[dict[str, Any]]]:
            try:
                return cid, await polymarket_client.get_market_by_condition_id(cid)
            except Exception:
                return cid, None

        cid_list = sorted(condition_ids_to_fetch)
        for i in range(0, len(cid_list), _BATCH):
            batch = cid_list[i : i + _BATCH]
            results = await asyncio.gather(*[_fetch_condition(c) for c in batch])
            for cid, info in results:
                market_cache_by_condition[cid] = info

    def _infer_trade_token_id(trade: dict[str, Any]) -> str:
        condition_id = str(trade.get("conditionId") or trade.get("condition_id") or trade.get("market") or "").strip()
        if not condition_id:
            return ""

        market_info = market_cache_by_condition.get(condition_id)
        if not isinstance(market_info, dict):
            return ""

        token_ids_raw = market_info.get("token_ids")
        if not isinstance(token_ids_raw, list):
            token_ids_raw = market_info.get("tokenIds")
        if not isinstance(token_ids_raw, list):
            return ""
        token_ids = [str(token_id or "").strip() for token_id in token_ids_raw]
        if not token_ids:
            return ""

        outcomes_raw = market_info.get("outcomes")
        outcomes: list[str] = []
        if isinstance(outcomes_raw, list):
            outcomes = [str(outcome or "").strip().lower() for outcome in outcomes_raw]

        outcome_idx = safe_float(
            trade.get("outcomeIndex") if trade.get("outcomeIndex") is not None else trade.get("outcome_index")
        )
        if outcome_idx is not None:
            idx = int(outcome_idx)
            if 0 <= idx < len(token_ids):
                return token_ids[idx]

        outcome_text = str(trade.get("outcome") or trade.get("token_outcome") or "").strip().lower()
        if outcome_text and outcomes:
            for idx, outcome_label in enumerate(outcomes):
                if outcome_label == outcome_text and idx < len(token_ids):
                    return token_ids[idx]

        return ""

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        token_id = _extract_wallet_trade_token_id(trade)
        if not token_id:
            token_id = _infer_trade_token_id(trade)
        if not token_id:
            continue
        if _extract_wallet_trade_side(trade) != "sell":
            continue
        size = _extract_wallet_trade_size(trade)
        if size <= 0.0:
            continue
        price = _extract_wallet_trade_price(trade)
        timestamp = _parse_wallet_trade_time(
            trade.get("timestamp") or trade.get("created_at") or trade.get("createdAt") or trade.get("time")
        )
        record = {
            "trade_id": str(trade.get("id") or trade.get("order_id") or trade.get("orderId") or "").strip(),
            "token_id": token_id,
            "size": size,
            "price": price,
            "timestamp": timestamp,
            "transactionHash": str(
                trade.get("transactionHash") or trade.get("txHash") or trade.get("tx_hash") or ""
            ).strip()
            or None,
        }
        existing = latest_by_token.get(token_id)
        if existing is None:
            latest_by_token[token_id] = record
            continue
        existing_ts = existing.get("timestamp")
        if timestamp is not None and (
            existing_ts is None or (isinstance(existing_ts, datetime) and timestamp > existing_ts)
        ):
            latest_by_token[token_id] = record
    _wallet_sell_trades_cache = (_time_mod.monotonic(), latest_by_token)
    return latest_by_token


async def _load_execution_wallet_recent_close_activity_by_token() -> dict[str, dict[str, Any]]:
    global _wallet_activity_cache, _wallet_activity_last_refresh_succeeded
    cached_at, cached_data = _wallet_activity_cache
    if _wallet_activity_last_refresh_succeeded and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
        return cached_data

    # Single-flight (see ``_load_execution_wallet_closed_positions_by_token``).
    async with _get_wallet_activity_lock():
        cached_at, cached_data = _wallet_activity_cache
        if _wallet_activity_last_refresh_succeeded and (_time_mod.monotonic() - cached_at) < _WALLET_HISTORY_CACHE_TTL_SECONDS:
            return cached_data

        # Hot-path gate: orchestrator hot loop must not issue REST.
        if not allow_polymarket_rest_call("wallet_activity"):
            return cached_data or {}

        wallet = await _resolve_execution_wallet_address()
        if not wallet:
            _wallet_activity_last_refresh_succeeded = False
            return {}

        try:
            activities = await polymarket_client.get_wallet_activity_paginated(
                wallet,
                max_items=_WALLET_HISTORY_MAX_ACTIVITY_ITEMS,
                page_size=_WALLET_HISTORY_ACTIVITY_PAGE_SIZE,
                activity_types=("TRADE", "MERGE", "REDEEM", "CLAIM", "CONVERSION", "CONVERT"),
            )
        except Exception:
            _wallet_activity_last_refresh_succeeded = False
            return {}
        if not isinstance(activities, list):
            _wallet_activity_last_refresh_succeeded = False
            return {}

        latest_by_token: dict[str, dict[str, Any]] = {}

        def _remember(key: str, activity: dict[str, Any]) -> None:
            if not key:
                return
            timestamp = _parse_wallet_activity_time(activity)
            current = latest_by_token.get(key)
            current_timestamp = _parse_wallet_activity_time(current) if isinstance(current, dict) else None
            if current_timestamp is None or (timestamp is not None and timestamp >= current_timestamp):
                latest_by_token[key] = activity

        for activity in activities:
            if not isinstance(activity, dict) or not _wallet_activity_is_close_authority(activity):
                continue
            token_id = _extract_wallet_activity_token_id(activity)
            if token_id:
                _remember(token_id, activity)
            cond_key = _wallet_activity_condition_outcome_key(activity)
            if cond_key:
                _remember(cond_key, activity)

        _wallet_activity_cache = (_time_mod.monotonic(), latest_by_token)
        _wallet_activity_last_refresh_succeeded = True
        return latest_by_token


async def _load_mapping_with_timeout(
    loader,
    *,
    timeout: float,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(loader(), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except Exception:
        return dict(fallback or {})


def _payload_market_info_for_order(order: Any, market_id: str, live_market: dict[str, Any]) -> dict[str, Any]:
    payload = dict(getattr(order, "payload_json", {}) or {})
    fallback = _normalize_market_info_aliases(live_market)

    matched_market: dict[str, Any] = {}
    raw_markets = payload.get("markets")
    if isinstance(raw_markets, list):
        for market in raw_markets:
            if not isinstance(market, dict):
                continue
            candidate_market_id = str(market.get("market_id") or market.get("id") or "").strip()
            if candidate_market_id == market_id:
                matched_market = dict(market)
                break
    if matched_market:
        fallback = _merge_market_info(matched_market, fallback) or {}

    direction = str(getattr(order, "direction", "") or payload.get("direction") or "").strip().lower()
    preferred_outcome = "yes" if direction.endswith("yes") else ("no" if direction.endswith("no") else "")
    matched_position: dict[str, Any] = {}
    first_position: dict[str, Any] = {}
    raw_positions = payload.get("positions_to_take")
    if isinstance(raw_positions, list):
        for position in raw_positions:
            if not isinstance(position, dict):
                continue
            candidate_market_id = str(position.get("market_id") or "").strip()
            if candidate_market_id != market_id:
                continue
            if not first_position:
                first_position = dict(position)
            outcome = str(position.get("outcome") or "").strip().lower()
            if preferred_outcome and outcome == preferred_outcome:
                matched_position = dict(position)
                break
        if not matched_position and first_position:
            matched_position = first_position

    derived: dict[str, Any] = {"market_id": market_id, "id": market_id}

    question = (
        payload.get("market_question")
        or matched_market.get("question")
        or matched_market.get("market_question")
        or fallback.get("question")
        or fallback.get("market_question")
    )
    if not _missing_market_info_value(question):
        derived["question"] = question
        derived["market_question"] = question

    condition_id = payload.get("condition_id") or matched_market.get("condition_id") or fallback.get("condition_id")
    if not _missing_market_info_value(condition_id):
        derived["condition_id"] = condition_id

    selected_outcome = str(
        fallback.get("selected_outcome")
        or matched_position.get("outcome")
        or preferred_outcome
        or ""
    ).strip().lower()
    if selected_outcome in {"yes", "no"}:
        derived["selected_outcome"] = selected_outcome

    matched_market_token_ids = matched_market.get("token_ids")
    if not isinstance(matched_market_token_ids, list):
        matched_market_token_ids = matched_market.get("clob_token_ids")
    matched_market_token_ids = matched_market_token_ids if isinstance(matched_market_token_ids, list) else []

    selected_token_id = (
        payload.get("selected_token_id")
        or payload.get("token_id")
        or matched_position.get("token_id")
        or fallback.get("selected_token_id")
    )
    if not _missing_market_info_value(selected_token_id):
        derived["selected_token_id"] = str(selected_token_id)

    yes_token_id = payload.get("yes_token_id") or fallback.get("yes_token_id")
    no_token_id = payload.get("no_token_id") or fallback.get("no_token_id")
    if _missing_market_info_value(yes_token_id) and len(matched_market_token_ids) >= 1:
        yes_token_id = matched_market_token_ids[0]
    if _missing_market_info_value(no_token_id) and len(matched_market_token_ids) >= 2:
        no_token_id = matched_market_token_ids[1]
    if not _missing_market_info_value(yes_token_id):
        derived["yes_token_id"] = str(yes_token_id)
    if not _missing_market_info_value(no_token_id):
        derived["no_token_id"] = str(no_token_id)

    token_ids: list[str] = []

    def _append_token(value: Any) -> None:
        token = str(value or "").strip()
        if token and token not in token_ids:
            token_ids.append(token)

    for key in ("token_ids", "clob_token_ids"):
        raw_tokens = fallback.get(key)
        if isinstance(raw_tokens, list):
            for token in raw_tokens:
                _append_token(token)
    for token in matched_market_token_ids:
        _append_token(token)
    _append_token(derived.get("yes_token_id"))
    _append_token(derived.get("no_token_id"))
    _append_token(derived.get("selected_token_id"))
    _append_token(matched_position.get("token_id"))
    if token_ids:
        derived["token_ids"] = token_ids

    selected_price = safe_float(matched_position.get("current_price"))
    if selected_price is None:
        selected_price = safe_float(matched_position.get("price"))
    if selected_price is None:
        selected_price = safe_float(payload.get("entry_price"))
    if selected_price is not None:
        derived["live_selected_price"] = float(selected_price)

    resolution_time = (
        payload.get("resolution_date")
        or payload.get("market_end_time")
        or matched_market.get("resolution_date")
        or matched_market.get("end_date")
        or matched_market.get("market_end_time")
        or fallback.get("end_date")
        or fallback.get("market_end_time")
    )
    if not _missing_market_info_value(resolution_time):
        derived["end_date"] = resolution_time
        derived["market_end_time"] = resolution_time

    return _normalize_market_info_aliases(_merge_market_info(derived, fallback) or derived)


def _market_lookup_candidates_for_order(order: Any) -> tuple[str, list[str], dict[str, Any]]:
    def _append_lookup_candidate(target: list[str], value: Any) -> None:
        candidate = str(value or "").strip()
        if candidate and candidate not in target:
            target.append(candidate)

    market_id = str(getattr(order, "market_id", "") or "").strip()
    if not market_id:
        return "", [], {}
    payload = dict(getattr(order, "payload_json", {}) or {})
    live_market = payload.get("live_market") if isinstance(payload.get("live_market"), dict) else {}
    payload_market_info = _payload_market_info_for_order(order, market_id, live_market)
    candidates: list[str] = []
    _append_lookup_candidate(candidates, market_id)
    _append_lookup_candidate(candidates, payload.get("condition_id"))
    _append_lookup_candidate(candidates, payload.get("token_id"))
    _append_lookup_candidate(candidates, payload.get("selected_token_id"))
    _append_lookup_candidate(candidates, payload.get("yes_token_id"))
    _append_lookup_candidate(candidates, payload.get("no_token_id"))
    if payload_market_info:
        _append_lookup_candidate(candidates, payload_market_info.get("id"))
        _append_lookup_candidate(candidates, payload_market_info.get("market_id"))
        _append_lookup_candidate(candidates, payload_market_info.get("condition_id"))
        _append_lookup_candidate(candidates, payload_market_info.get("selected_token_id"))
        _append_lookup_candidate(candidates, payload_market_info.get("yes_token_id"))
        _append_lookup_candidate(candidates, payload_market_info.get("no_token_id"))
        token_ids = payload_market_info.get("token_ids")
        if isinstance(token_ids, list):
            for token_id in token_ids:
                _append_lookup_candidate(candidates, token_id)
    return market_id, candidates, payload_market_info


def _fallback_market_info_for_orders(orders: list[TraderOrder]) -> dict[str, dict[str, Any]]:
    fallback: dict[str, dict[str, Any]] = {}
    for order in orders:
        market_id, candidates, payload_market_info = _market_lookup_candidates_for_order(order)
        if not market_id:
            continue
        if payload_market_info:
            fallback[market_id] = payload_market_info
            continue
        for candidate in candidates:
            cached_entry = _market_info_cache.get(candidate)
            if cached_entry is None:
                continue
            cached_info = cached_entry[1]
            if isinstance(cached_info, dict):
                fallback[market_id] = dict(cached_info)
                break
    return fallback


def _cached_order_snapshots_by_clob_ids(clob_order_ids: list[str]) -> dict[str, dict[str, Any]]:
    requested = {str(order_id or "").strip() for order_id in clob_order_ids if str(order_id or "").strip()}
    if not requested:
        return {}
    snapshots: dict[str, dict[str, Any]] = {}
    for order in live_execution_service._orders.values():
        snapshot = live_execution_service._snapshot_from_cached_order(order)
        if snapshot is None:
            continue
        clob_order_id = str(snapshot.get("clob_order_id") or "").strip()
        if clob_order_id in requested:
            snapshots[clob_order_id] = snapshot
    return snapshots


# Module-level TTL cache for market metadata to avoid redundant REST API calls.
# Each entry maps lookup_id -> (fetched_at_monotonic, market_info_dict).
_market_info_cache: dict[str, tuple[float, Optional[dict[str, Any]]]] = {}
_MARKET_INFO_CACHE_TTL_SECONDS = 180.0
_MARKET_INFO_NEAR_RESOLUTION_SECONDS = 300.0  # 5 minutes
_MARKET_INFO_CACHE_MAX_SIZE = 2000
_market_info_cache_last_eviction: float = 0.0


def _evict_stale_market_info_cache(now_mono: float) -> None:
    """Remove expired entries and enforce max size to prevent unbounded growth."""
    global _market_info_cache_last_eviction
    if (now_mono - _market_info_cache_last_eviction) < 30.0:
        return
    _market_info_cache_last_eviction = now_mono
    expired_keys = [
        k for k, (fetched_at, _) in _market_info_cache.items()
        if (now_mono - fetched_at) > _MARKET_INFO_CACHE_TTL_SECONDS * 5
    ]
    for k in expired_keys:
        _market_info_cache.pop(k, None)
    if len(_market_info_cache) > _MARKET_INFO_CACHE_MAX_SIZE:
        sorted_entries = sorted(_market_info_cache.items(), key=lambda x: x[1][0])
        for k, _ in sorted_entries[: len(_market_info_cache) - _MARKET_INFO_CACHE_MAX_SIZE]:
            _market_info_cache.pop(k, None)


def _market_info_needs_refresh(lookup_id: str, now_mono: float) -> bool:
    _evict_stale_market_info_cache(now_mono)
    entry = _market_info_cache.get(lookup_id)
    if entry is None:
        return True
    fetched_at, info = entry
    age = now_mono - fetched_at
    if age >= _MARKET_INFO_CACHE_TTL_SECONDS:
        return True
    if isinstance(info, dict):
        if info.get("winning_outcome") is not None:
            return True
        end_time = _extract_market_end_time_naive(info)
        if end_time is not None:
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            seconds_left = (end_time - now_utc).total_seconds()
            if seconds_left <= _MARKET_INFO_NEAR_RESOLUTION_SECONDS:
                return True
    return False


async def load_market_info_for_orders(orders: list[TraderOrder]) -> dict[str, Optional[dict[str, Any]]]:
    lookup_candidates_by_market_id: dict[str, list[str]] = {}
    fallback_info_by_market_id = _fallback_market_info_for_orders(orders)

    for order in orders:
        market_id, candidates, _ = _market_lookup_candidates_for_order(order)
        if not market_id:
            continue
        lookup_candidates_by_market_id[market_id] = candidates

    lookup_ids = sorted({candidate for candidates in lookup_candidates_by_market_id.values() for candidate in candidates})
    if not lookup_ids:
        return fallback_info_by_market_id

    now_mono = _time_mod.monotonic()

    async def _fetch(lookup_id: str) -> tuple[str, Optional[dict[str, Any]]]:
        needs_refresh = _market_info_needs_refresh(lookup_id, now_mono)
        if not needs_refresh:
            _, cached_info = _market_info_cache[lookup_id]
            return lookup_id, cached_info

        # Hot-path gate: never fetch market info from REST when the
        # orchestrator is in the no-REST scope.  Fall back to the
        # in-memory market_cache_service / feed_manager view.
        if not allow_polymarket_rest_call("market_info_lookup"):
            existing = _market_info_cache.get(lookup_id)
            return lookup_id, (existing[1] if existing else None)

        info: Optional[dict[str, Any]] = None
        if lookup_id.startswith("0x"):
            info = await polymarket_client.get_market_by_condition_id(lookup_id, force_refresh=True)
            if info is None:
                info = await polymarket_client.get_market_by_condition_id(lookup_id)
        if info is None:
            info = await polymarket_client.get_market_by_token_id(lookup_id)

        _market_info_cache[lookup_id] = (now_mono, info)
        return lookup_id, info

    # Limit concurrency to avoid overwhelming the Gamma API rate limiter.
    _fetch_sem = asyncio.Semaphore(8)

    async def _fetch_limited(lookup_id: str) -> tuple[str, Optional[dict[str, Any]]]:
        async with _fetch_sem:
            return await _fetch(lookup_id)

    pairs = await asyncio.gather(*[_fetch_limited(lookup_id) for lookup_id in lookup_ids], return_exceptions=True)
    info_by_lookup_id: dict[str, Optional[dict[str, Any]]] = {}
    for item in pairs:
        if isinstance(item, Exception):
            continue
        lookup_id, info = item
        info_by_lookup_id[lookup_id] = info

    out: dict[str, Optional[dict[str, Any]]] = {}
    for market_id, candidates in lookup_candidates_by_market_id.items():
        info = _normalize_market_info_aliases(fallback_info_by_market_id.get(market_id))
        for candidate in candidates:
            candidate_info = info_by_lookup_id.get(candidate)
            if candidate_info is not None:
                info = _merge_market_info(candidate_info, info)
                break
        out[market_id] = info
    return out


async def reconcile_shadow_positions(
    session: AsyncSession,
    *,
    trader_id: str,
    trader_params: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    force_mark_to_market: bool = False,
    max_age_hours: Optional[int] = None,
    order_ids: Optional[list[str]] = None,
    reason: str = "shadow_position_lifecycle",
    enable_simulation_ledger: bool = False,
) -> dict[str, Any]:
    params = dict(trader_params or {})
    mode_key = "shadow"
    prefix_key = "shadow"

    def _trader_param(name: str, default: Any = None) -> Any:
        prefixed_key = f"{prefix_key}_{name}"
        if prefixed_key in params:
            return params.get(prefixed_key)
        if name in params:
            return params.get(name)
        return default

    resolution_infer_from_prices = _safe_bool(_trader_param("resolution_infer_from_prices"), True)
    resolution_settle_floor = min(
        1.0,
        max(0.5, safe_float(_trader_param("resolution_settle_floor")) or 0.98),
    )

    candidates = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key,
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(SHADOW_ACTIVE_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    requested_order_ids: set[str] = set()
    requested_order_ids = {str(value or "").strip() for value in (order_ids or []) if str(value or "").strip()}
    if order_ids:
        if requested_order_ids:
            candidates = [row for row in candidates if str(row.id or "") in requested_order_ids]
        else:
            candidates = []

    if max_age_hours is not None:
        cutoff = utcnow() - timedelta(hours=max(1, int(max_age_hours)))
        candidates = [
            row
            for row in candidates
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]
    candidates = _dedupe_live_authority_rows(candidates)

    signal_ids = [str(row.signal_id) for row in candidates if row.signal_id]
    signal_payloads: dict[str, dict[str, Any]] = {}
    if signal_ids:
        signal_rows = (
            await session.execute(
                select(TradeSignal.id, TradeSignal.payload_json).where(TradeSignal.id.in_(signal_ids))
            )
        ).all()
        signal_payloads = {str(row.id): dict(row.payload_json or {}) for row in signal_rows}

    from models.database import release_conn

    async with release_conn(session):
        market_info_by_id = await load_market_info_for_orders(candidates)

    now = utcnow()
    would_close = 0
    closed = 0
    held = 0
    skipped = 0
    total_realized_pnl = 0.0
    by_status = {"resolved_win": 0, "resolved_loss": 0, "closed_win": 0, "closed_loss": 0}
    skipped_reasons: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    state_updates = 0
    reverse_signal_ids_by_source: dict[str, list[str]] = {}

    for row in candidates:
        entry_price = safe_float(row.effective_price)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price)
        notional = safe_float(row.notional_usd) or 0.0
        row_market_info = market_info_by_id.get(str(row.market_id or ""))
        row_payload_for_idx = dict(row.payload_json or {})
        row_token_id_for_idx = _extract_leg_token_id(row_payload_for_idx)
        outcome_idx = _direction_outcome_index(
            row.direction,
            market_info=row_market_info,
            token_id=row_token_id_for_idx,
        )
        if outcome_idx is None or entry_price is None or entry_price <= 0 or notional <= 0:
            payload = dict(row.payload_json or {})
            provider_snapshot_status = _provider_snapshot_status(payload)
            raw_filled_notional, raw_filled_size, _ = _extract_live_fill_metrics(payload)
            wallet_position_size = 0.0
            provider_status_key = str(provider_snapshot_status or "").strip().lower()
            if (
                wallet_position_size <= _WALLET_SIZE_EPSILON
                and raw_filled_notional <= 0.0
                and raw_filled_size <= 0.0
                and provider_status_key in {"filled", "partially_filled", "matched", "executed"}
            ):
                next_status = "cancelled"
                if not dry_run:
                    row.status = next_status
                    row.actual_profit = 0.0
                    row.notional_usd = 0.0
                    row.executed_at = None
                    row.updated_at = now
                    payload["position_close"] = {
                        "close_price": 0.0,
                        "price_source": "terminal_unfilled_provider_snapshot",
                        "close_trigger": "terminal_unfilled_provider_snapshot",
                        "realized_pnl": 0.0,
                        "provider_status": provider_status_key,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    payload["provider_failure_finalized"] = {
                        "finalized_at": _iso_utc(now),
                        "provider_status": provider_status_key,
                        "reason": "provider_snapshot_without_fill_or_wallet_position",
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=next_status,
                        actual_profit=0.0,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "terminal_unfilled_provider_snapshot",
                        "close_price": 0.0,
                        "realized_pnl": 0.0,
                        "provider_status": provider_status_key,
                        "next_status": next_status,
                    }
                )
                continue
            skipped += 1
            skipped_reasons["invalid_entry"] = int(skipped_reasons.get("invalid_entry", 0)) + 1
            continue

        signal_payload = signal_payloads.get(str(row.signal_id), {})
        market_info = market_info_by_id.get(str(row.market_id or ""))
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        market_seconds_left = _market_seconds_left(market_info, now)
        market_end_time = _market_end_time_iso(market_info)
        winning_idx = _extract_winning_outcome_index(market_info)
        winning_idx_inferred = False
        if winning_idx is None and resolution_infer_from_prices:
            inferred_idx = _extract_winning_outcome_index_from_prices(
                market_info,
                market_tradable=market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_idx is not None:
                winning_idx = inferred_idx
                winning_idx_inferred = True
        market_side_price = _extract_market_side_price(market_info, outcome_idx)
        snapshot_side_price = _extract_signal_side_price(signal_payload, outcome_idx)

        close_price: Optional[float] = None
        close_trigger: Optional[str] = None
        price_source: Optional[str] = None
        trailing_trigger_price: Optional[float] = None

        current_price = market_side_price if market_side_price is not None else snapshot_side_price
        current_price = _state_price_floor(current_price)
        current_price_source = (
            "market_mark"
            if market_side_price is not None
            else ("signal_snapshot_mark" if snapshot_side_price is not None else None)
        )

        age_anchor = row.executed_at or row.updated_at or row.created_at
        age_minutes = None
        if age_anchor is not None:
            age_minutes = max(0.0, (now - age_anchor).total_seconds() / 60.0)

        payload = dict(row.payload_json or {})
        take_profit_pct = safe_float(_payload_exit_param(payload, prefix_key=prefix_key, name="take_profit_pct"))
        stop_loss_pct = safe_float(_payload_exit_param(payload, prefix_key=prefix_key, name="stop_loss_pct"))
        max_hold_minutes = safe_float(_payload_exit_param(payload, prefix_key=prefix_key, name="max_hold_minutes"))
        min_hold_minutes = max(
            0.0,
            safe_float(_payload_exit_param(payload, prefix_key=prefix_key, name="min_hold_minutes")) or 0.0,
        )
        trailing_stop_pct = safe_float(_payload_exit_param(payload, prefix_key=prefix_key, name="trailing_stop_pct"))
        resolve_only = _safe_bool(_payload_exit_param(payload, prefix_key=prefix_key, name="resolve_only"), False)
        close_on_inactive_market = _safe_bool(
            _payload_exit_param(payload, prefix_key=prefix_key, name="close_on_inactive_market"),
            False,
        )
        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes
        _exit_instance = None
        position_state = _extract_position_state(payload)
        prev_high = safe_float(position_state.get("highest_price"))
        prev_low = safe_float(position_state.get("lowest_price"))
        prev_last_mark = safe_float(position_state.get("last_mark_price"))
        prev_mark_source = str(position_state.get("last_mark_source") or "")
        prev_marked_at = _parse_iso_utc_naive(position_state.get("last_marked_at"))
        highest_price = prev_high
        lowest_price = prev_low
        if current_price is not None:
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)

        mark_updated_at_value = (
            _iso_utc(now)
            if current_price is not None
            else (
                _iso_utc(prev_marked_at.replace(tzinfo=timezone.utc))
                if prev_marked_at is not None
                else ""
            )
        )
        next_state = {
            "highest_price": _state_price_floor(highest_price),
            "lowest_price": _state_price_floor(lowest_price),
            "last_mark_price": current_price if current_price is not None else _state_price_floor(prev_last_mark),
            "last_mark_source": str(current_price_source or prev_mark_source),
            "last_marked_at": mark_updated_at_value,
            "last_exit_evaluated_at": _iso_utc(now),
        }

        if winning_idx is not None:
            close_price = 1.0 if winning_idx == outcome_idx else 0.0
            close_trigger = "resolution_inferred" if winning_idx_inferred else "resolution"
            price_source = "resolved_settlement"
        else:
            pnl_pct = None
            if current_price is not None and entry_price > 0:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            if force_mark_to_market and current_price is not None:
                close_price = current_price
                close_trigger = "manual_mark_to_market"
                price_source = current_price_source
            else:
                # ── Strategy-based exit check ──────────────────────────
                # If the strategy that opened this position has a
                # should_exit() method, call it first and respect its
                # decision before falling through to default TP/SL/etc.
                strategy_slug = (payload.get("strategy_type") or "").strip().lower()
                strategy_exit = None
                _exit_instance = await _strategy_exit_instance(session, strategy_slug) if strategy_slug else None
                if _exit_instance is not None:
                    try:

                        class _ShadowPositionView:
                            pass

                        pos_view = _ShadowPositionView()
                        pos_view.entry_price = entry_price
                        pos_view.current_price = current_price
                        pos_view.highest_price = highest_price
                        pos_view.lowest_price = lowest_price
                        pos_view.age_minutes = age_minutes
                        pos_view.pnl_percent = pnl_pct
                        pos_view.filled_size = (notional / entry_price) if entry_price > 0 else 0.0
                        pos_view.notional_usd = notional
                        if "strategy_context" not in payload:
                            payload["strategy_context"] = {}
                        strategy_context_payload = (
                            payload["strategy_context"] if isinstance(payload.get("strategy_context"), dict) else {}
                        )
                        pos_view.strategy_context = payload["strategy_context"]
                        pos_view.config = payload.get("strategy_exit_config", {})
                        pos_view.outcome_idx = outcome_idx

                        token_id = _extract_live_token_id(payload)
                        min_order_size_usd = _resolve_position_min_order_size_usd(
                            trader_params=params,
                            payload=payload,
                            mode=mode_key,
                        )
                        market_state_dict = {
                            "current_price": current_price,
                            "market_tradable": market_tradable,
                            "is_resolved": False,
                            "winning_outcome": None,
                            "seconds_left": market_seconds_left,
                            "end_time": market_end_time,
                            "token_id": token_id,
                            "mark_source": current_price_source,
                            "min_order_size_usd": min_order_size_usd,
                            "notional_usd": notional,
                            "oracle_price": strategy_context_payload.get("oracle_price"),
                            "price_to_beat": strategy_context_payload.get("price_to_beat"),
                            "oracle_age_seconds": strategy_context_payload.get("oracle_age_seconds"),
                            "confirmed_stage_triggered": strategy_context_payload.get("confirmed_stage_triggered"),
                            "strategy_stage": strategy_context_payload.get("stage"),
                        }

                        exit_decision = _exit_instance.should_exit(pos_view, market_state_dict)
                        exit_action = getattr(exit_decision, "action", None) if exit_decision is not None else None
                        if exit_action == "close":
                            strategy_exit = exit_decision
                        elif exit_action == "reduce":
                            strategy_exit = exit_decision
                        elif _strategy_hold_blocks_default_exit(exit_decision):
                            strategy_exit = exit_decision
                    except Exception as exc:
                        logger.warning(
                            "Strategy should_exit() error for %s: %s",
                            strategy_slug,
                            exc,
                        )

                if strategy_exit is not None and getattr(strategy_exit, "action", None) == "reduce":
                    # SDK contract: ``reduce`` means realize a partial slice
                    # of the shadow position now and keep the residual open.
                    if not dry_run:
                        await _submit_shadow_partial_exit(
                            payload=payload,
                            decision=strategy_exit,
                            close_price=current_price,
                            price_source=current_price_source,
                            notional_usd=notional,
                            entry_price=entry_price,
                            now=now,
                        )
                        payload["position_state"] = next_state
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue

                if _strategy_hold_blocks_default_exit(strategy_exit):
                    if not dry_run:
                        payload["position_state"] = next_state
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue

                if strategy_exit is not None:
                    _arm_reverse_entry_from_exit(
                        row=row,
                        payload=payload,
                        strategy_exit=strategy_exit,
                        market_info=market_info,
                        market_seconds_left=market_seconds_left,
                        current_price=current_price,
                        now=now,
                    )
                    close_price = strategy_exit.close_price if strategy_exit.close_price is not None else current_price
                    close_trigger = f"strategy:{strategy_exit.reason}"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and take_profit_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct >= take_profit_pct
                ):
                    close_price = current_price
                    close_trigger = "take_profit"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and stop_loss_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct <= -abs(stop_loss_pct)
                ):
                    close_price = current_price
                    close_trigger = "stop_loss"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and trailing_stop_pct is not None
                    and trailing_stop_pct > 0
                    and current_price is not None
                    and highest_price is not None
                    and min_hold_passed
                ):
                    trailing_trigger_price = highest_price * (1.0 - (trailing_stop_pct / 100.0))
                    if highest_price > entry_price and current_price <= trailing_trigger_price:
                        close_price = current_price
                        close_trigger = "trailing_stop"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and max_hold_minutes is not None
                    and age_minutes is not None
                    and age_minutes >= max_hold_minutes
                ):
                    if current_price is not None:
                        close_price = current_price
                        close_trigger = "max_hold"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and close_on_inactive_market
                    and not market_tradable
                    and current_price is not None
                    and min_hold_passed
                ):
                    close_price = current_price
                    close_trigger = "market_inactive"
                    price_source = current_price_source

        if close_price is None:
            state_changed = False
            if current_price is not None:
                state_changed = (
                    prev_last_mark is None
                    or abs(prev_last_mark - current_price) > 1e-9
                    or prev_high is None
                    or prev_low is None
                    or abs((prev_high or 0.0) - (highest_price or 0.0)) > 1e-9
                    or abs((prev_low or 0.0) - (lowest_price or 0.0)) > 1e-9
                    or prev_mark_source != str(current_price_source or "")
                )
            if not dry_run and state_changed:
                payload["position_state"] = next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        quantity = notional / entry_price
        proceeds = quantity * close_price
        pnl = proceeds - notional
        next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)

        simulation_close: dict[str, Any] | None = None
        simulation_ledger = payload.get("simulation_ledger")
        if not dry_run and enable_simulation_ledger and isinstance(simulation_ledger, dict):
            sim_account_id = str(simulation_ledger.get("account_id") or "").strip()
            sim_trade_id = str(simulation_ledger.get("trade_id") or "").strip()
            sim_position_id = str(simulation_ledger.get("position_id") or "").strip()
            if sim_account_id and sim_trade_id and sim_position_id:
                try:
                    simulation_close = await simulation_service.close_orchestrator_shadow_fill(
                        account_id=sim_account_id,
                        trade_id=sim_trade_id,
                        position_id=sim_position_id,
                        close_price=float(close_price),
                        close_trigger=close_trigger,
                        price_source=price_source,
                        reason=reason,
                        session=session,
                        commit=False,
                    )
                    if simulation_close.get("closed"):
                        proceeds = float(simulation_close.get("actual_payout", proceeds))
                        pnl = float(simulation_close.get("actual_pnl", pnl))
                        next_status = str(simulation_close.get("trade_status") or next_status)
                    elif simulation_close.get("already_closed"):
                        existing_status = str(simulation_close.get("trade_status") or "")
                        if existing_status:
                            next_status = existing_status
                        pnl = float(simulation_close.get("actual_pnl", pnl))
                        proceeds = float(simulation_close.get("actual_payout", proceeds))
                except Exception as exc:
                    skipped += 1
                    skipped_reasons["simulation_close_error"] = (
                        int(skipped_reasons.get("simulation_close_error", 0)) + 1
                    )
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "direction": row.direction,
                            "close_trigger": close_trigger,
                            "reason": "simulation_close_error",
                            "error": str(exc),
                        }
                    )
                    continue

        total_realized_pnl += pnl
        by_status[next_status] = int(by_status.get(next_status, 0)) + 1

        detail = {
            "order_id": row.id,
            "market_id": row.market_id,
            "direction": row.direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "market_tradable": market_tradable,
            "notional_usd": notional,
            "quantity": quantity,
            "realized_pnl": pnl,
            "next_status": next_status,
            "age_minutes": age_minutes,
            "min_hold_minutes": min_hold_minutes,
            "trailing_stop_trigger_price": trailing_trigger_price,
            "highest_price_seen": _state_price_floor(highest_price),
            "lowest_price_seen": _state_price_floor(lowest_price),
            "simulation_close": simulation_close,
        }
        details.append(detail)
        would_close += 1

        if dry_run:
            continue

        row.status = next_status
        row.actual_profit = pnl
        row.updated_at = now
        payload["position_state"] = next_state
        payload["position_close"] = {
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "realized_pnl": pnl,
            "market_tradable": market_tradable,
            "age_minutes": age_minutes,
            "closed_at": _iso_utc(now),
            "reason": reason,
        }
        if simulation_close is not None:
            payload["position_close"]["simulation_close"] = simulation_close

        if _exit_instance is not None and hasattr(_exit_instance, "record_trade_outcome"):
            try:
                _exit_instance.record_trade_outcome(won=next_status in {"closed_win", "resolved_win"})
            except Exception:
                pass

        reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
            session,
            row=row,
            payload=payload,
            signal_payload=signal_payload,
            market_info=market_info,
            close_trigger=close_trigger,
            realized_pnl=pnl,
            now=now,
        )
        if reverse_signal_id and reverse_source:
            reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)

        row.payload_json = payload
        if reason:
            if row.reason:
                row.reason = f"{row.reason} | {reason}:{close_trigger}"
            else:
                row.reason = f"{reason}:{close_trigger}"
        hot_state.record_order_resolved(
            trader_id=trader_id,
            mode=str(row.mode or ""),
            order_id=str(row.id or ""),
            market_id=str(row.market_id or ""),
            direction=str(row.direction or ""),
            source=str(row.source or ""),
            status=next_status,
            actual_profit=pnl,
            payload=payload,
            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
        )
        closed += 1

    if not dry_run and (closed > 0 or state_updates > 0):
        touched_rows = [row for row in candidates if row.updated_at == now]
        if touched_rows:
            publish_rows = list(touched_rows)
            stmt = (
                update(TraderOrder)
                .where(TraderOrder.id == bindparam("_id"))
                .values(
                    status=bindparam("_status"),
                    actual_profit=bindparam("_actual_profit"),
                    updated_at=bindparam("_updated_at"),
                    payload_json=bindparam("_payload_json"),
                    reason=bindparam("_reason"),
                )
                .execution_options(synchronize_session=None)
            )
            params = [
                {
                    "id": row.id,
                    "_id": row.id,
                    "_status": row.status,
                    "_actual_profit": row.actual_profit,
                    "_updated_at": row.updated_at,
                    "_payload_json": row.payload_json,
                    "_reason": row.reason,
                }
                for row in publish_rows
            ]
            for row in touched_rows:
                if inspect(row).session is session.sync_session:
                    session.sync_session.expunge(row)
            with session.no_autoflush:
                await session.execute(stmt, params)
        else:
            publish_rows = []
        await session.commit()
        await _publish_trader_order_updates(publish_rows)
        if reverse_signal_ids_by_source:
            await _publish_reverse_signal_batches(reverse_signal_ids_by_source, emitted_at=now)

    return {
        "trader_id": trader_id,
        "mode": mode_key,
        "dry_run": bool(dry_run),
        "matched": len(candidates),
        "would_close": would_close,
        "closed": closed,
        "held": held,
        "skipped": skipped,
        "state_updates": state_updates,
        "total_realized_pnl": total_realized_pnl,
        "by_status": by_status,
        "skipped_reasons": skipped_reasons,
        "reverse_signals_emitted": sum(len(ids) for ids in reverse_signal_ids_by_source.values()),
        "details": details,
    }


async def reconcile_live_positions(
    session: AsyncSession,
    *,
    trader_id: str,
    trader_params: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    force_mark_to_market: bool = False,
    max_age_hours: Optional[int] = None,
    order_ids: Optional[list[str]] = None,
    reason: str = "live_position_lifecycle",
) -> dict[str, Any]:
    """Lifecycle management for live positions.

    Mirrors reconcile_shadow_positions but operates on mode='live' orders.
    Handles: stop-loss, take-profit, trailing stop, max hold, market
    inactivity, and resolution detection.  Does NOT interact with the
    simulation ledger (that is shadow-only).
    """
    params = dict(trader_params or {})
    resolution_infer_from_prices = _safe_bool(params.get("live_resolution_infer_from_prices"), True)
    resolution_settle_floor = min(
        1.0,
        max(0.5, safe_float(params.get("live_resolution_settle_floor")) or 0.98),
    )
    mark_touch_interval_seconds = _mark_touch_interval_seconds(params, mode="live")

    import time as _time
    _lc_t0 = _time.monotonic()

    candidates = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    TraderOrder.mode == "live",
                    TraderOrder.status.in_(tuple(LIVE_ACTIVE_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    requested_order_ids = {str(value or "").strip() for value in (order_ids or []) if str(value or "").strip()}
    if order_ids:
        if requested_order_ids:
            candidates = [row for row in candidates if str(row.id or "") in requested_order_ids]
        else:
            candidates = []

    if max_age_hours is not None:
        cutoff = utcnow() - timedelta(hours=max(1, int(max_age_hours)))
        candidates = [
            row
            for row in candidates
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]

    global _wallet_positions_last_refresh_succeeded, _wallet_closed_positions_last_refresh_succeeded, _wallet_activity_last_refresh_succeeded

    now = utcnow()
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    would_close = 0
    closed = 0
    held = 0
    skipped = 0
    total_realized_pnl = 0.0
    by_status: dict[str, int] = {"resolved_win": 0, "resolved_loss": 0, "closed_win": 0, "closed_loss": 0}
    skipped_reasons: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    state_updates = 0
    extra_touched_rows: list[TraderOrder] = []
    reverse_signal_ids_by_source: dict[str, list[str]] = {}

    candidate_ids = {str(row.id) for row in candidates}
    terminal_age_expr = func.coalesce(
        TraderOrder.updated_at,
        TraderOrder.executed_at,
        TraderOrder.created_at,
    )
    terminal_stmt = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        TraderOrder.mode == "live",
        TraderOrder.status.in_(
            (
                "closed_win",
                "closed_loss",
                "resolved",
                "resolved_win",
                "resolved_loss",
                "cancelled",
                "failed",
                "rejected",
                "error",
            )
        ),
    ).order_by(terminal_age_expr.asc(), TraderOrder.id.asc())
    if requested_order_ids:
        terminal_stmt = terminal_stmt.where(TraderOrder.id.in_(requested_order_ids))
    if max_age_hours is not None:
        cutoff = now - timedelta(hours=max(1, int(max_age_hours)))
        terminal_stmt = terminal_stmt.where(terminal_age_expr <= cutoff)
    else:
        audit_cutoff = now_naive - timedelta(
            hours=max(_TERMINAL_REOPEN_LOOKBACK_HOURS, _NONACTIVE_WALLET_REOPEN_LOOKBACK_HOURS)
        )
        terminal_stmt = terminal_stmt.where(terminal_age_expr >= audit_cutoff)
        # Bound the SQL fetch.  Without a LIMIT, traders with hundreds of
        # terminal orders in the 72h lookback window were returning the
        # full set every reconciliation cycle (driving pre_io >10s).  We
        # take the OLDEST first and cap well above the Python-side
        # _TERMINAL_REOPEN_AUDIT_MAX_ROWS_PER_PASS so the post-filter
        # still has room.
        terminal_stmt = terminal_stmt.limit(_TERMINAL_REOPEN_AUDIT_MAX_ROWS_PER_PASS * 8)
    terminal_rows = list(((await session.execute(terminal_stmt)).scalars().all()))
    if max_age_hours is None:
        terminal_rows = [row for row in terminal_rows if _terminal_row_requires_reopen_audit(row, now_naive)]
        terminal_rows = terminal_rows[:_TERMINAL_REOPEN_AUDIT_MAX_ROWS_PER_PASS]
    terminal_rows = _dedupe_live_authority_rows(terminal_rows)

    # 2026-05-09: detach-then-rollback pattern. Slow_tx and
    # _capture_slow_commit_diagnostic showed commit_ms=2155 / 4329 with
    # dirty_rows=0 here — i.e. an empty COMMIT was costing 2-5s of WAL
    # fsync under pool contention. Plain ``rollback()`` on its own would
    # detach the loaded ORM rows AND expire their column attributes,
    # raising DetachedInstanceError when the loop below reads
    # ``row.id`` / ``row.payload_json`` / ``row.direction``.
    #
    # Instead we ``expunge`` the rows we loaded ourselves first.
    # ``expunge`` removes a row from the session WITHOUT expiring its
    # already-loaded columns, so the SELECTed values remain readable
    # from the now-detached instance. Refresh would still fail, but we
    # never trigger one — every downstream access is a column read of
    # a value the SELECT already materialized. ``rollback`` then closes
    # the read-only transaction without WAL fsync.
    for row in candidates:
        session.expunge(row)
    for row in terminal_rows:
        session.expunge(row)
    await session.rollback()
    async with release_conn(session):
        wallet_positions_by_token = await _load_mapping_with_timeout(
            _load_execution_wallet_positions_by_token,
            timeout=_WALLET_POSITIONS_LOAD_TIMEOUT_SECONDS,
            fallback=dict(_wallet_positions_cache[1] or {}),
        )
    # Two distinct concepts:
    #
    #   wallet_positions_loaded         — the load attempt succeeded; the
    #                                     wallet snapshot is meaningful
    #                                     enough to AUDIT against.
    #   wallet_positions_authoritative  — the load returned at least one
    #                                     row, so an absent token here is
    #                                     genuine evidence the wallet does
    #                                     not hold it.
    #
    # The wallet-absent close path uses ``authoritative`` rather than just
    # ``loaded`` because an empty cache is indistinguishable from "the
    # data API blipped empty".  The 2026-04-28 incident saw 3 positions
    # wrongly resolved as ``superseded_wallet_absent_no_provider`` after
    # such a blip wiped ``live_trading_positions``; requiring at least
    # one observed entry before treating an absent token as authoritative
    # eliminates that misclassification while still letting the path fire
    # for a wallet that genuinely held N positions of which this one is
    # gone.
    wallet_positions_loaded = bool(_wallet_positions_last_refresh_succeeded or wallet_positions_by_token)
    wallet_positions_authoritative = bool(wallet_positions_by_token)
    wallet_closed_positions_by_token: dict[str, dict[str, Any]] = {}
    wallet_sell_trades_by_token: dict[str, dict[str, Any]] = {}
    wallet_close_activity_by_token: dict[str, dict[str, Any]] = {}
    wallet_closed_positions_loaded = False
    wallet_terminal_close_consumed_size_by_key: dict[str, float] = {}
    wallet_position_reopen_consumed_size_by_key: dict[str, float] = {}

    for row in terminal_rows:
        row_id = str(row.id)
        if row_id in candidate_ids:
            continue
        payload = dict(row.payload_json or {})
        token_id = _extract_live_token_id(payload)
        wallet_position = wallet_positions_by_token.get(token_id) if token_id else None
        if not isinstance(wallet_position, dict) and token_id:
            cond_key = _extract_condition_outcome_key(payload, str(row.direction or ""))
            if cond_key:
                wallet_position = wallet_positions_by_token.get(cond_key)
        wallet_position_size = _extract_wallet_position_size(wallet_position)
        terminal_wallet_settlement_price = _extract_wallet_settlement_price(wallet_position)
        status_key = str(row.status or "").strip().lower()
        position_close = payload.get("position_close")
        terminal_close_trigger = ""
        terminal_close_price_source = ""
        if isinstance(position_close, dict):
            terminal_close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
            terminal_close_price_source = str(position_close.get("price_source") or "").strip().lower()
        generic_wallet_reopen_allowed = not (
            terminal_close_price_source in {"resolved_settlement", "provider_exit_fill", "wallet_absent_best_available"}
            or terminal_close_trigger in {"wallet_absent_close", "wallet_flat_override"}
        )
        if (
            generic_wallet_reopen_allowed
            and wallet_positions_loaded
            and wallet_position_size > _WALLET_SIZE_EPSILON
            and terminal_wallet_settlement_price is None
        ):
            failed_terminal_status = status_key in {"cancelled", "failed", "rejected", "error"}
            wallet_position_token_id = str(
                wallet_position.get("asset")
                or wallet_position.get("asset_id")
                or wallet_position.get("token_id")
                or wallet_position.get("tokenId")
                or ""
            ).strip()
            wallet_position_entry_price = _extract_wallet_entry_price(wallet_position)
            if failed_terminal_status and not (
                _failed_terminal_row_can_reopen_from_wallet_position(row, payload)
                or (
                    _terminal_row_has_wallet_position_authority(row, payload)
                    and wallet_position_token_id
                    and wallet_position_entry_price is not None
                )
            ):
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "next_status": row.status,
                        "wallet_position_size": wallet_position_size,
                        "note": "kept_terminal_row_specific_authority_missing",
                    }
                )
                continue
            wallet_key = token_id or _extract_condition_outcome_key(payload, str(row.direction or "")) or str(row.market_id or "")
            wallet_size, wallet_notional, wallet_entry_price, _wallet_mark_price = _wallet_position_notional(wallet_position)
            consumed_reopen_size = wallet_position_reopen_consumed_size_by_key.get(wallet_key, 0.0)
            available_reopen_size = max(0.0, wallet_size - consumed_reopen_size)
            if available_reopen_size <= _WALLET_SIZE_EPSILON:
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "next_status": row.status,
                        "wallet_position_size": wallet_size,
                        "consumed_wallet_position_size": consumed_reopen_size,
                        "note": "kept_terminal_wallet_position_already_allocated",
                    }
                )
                continue
            row_entry_size = _entry_size_for_terminal_close(row, payload)
            reopen_size = (
                min(row_entry_size, available_reopen_size)
                if row_entry_size > _WALLET_SIZE_EPSILON
                else available_reopen_size
            )
            if wallet_entry_price is not None and wallet_entry_price > 0.0:
                reopen_notional = reopen_size * wallet_entry_price
            elif wallet_size > _WALLET_SIZE_EPSILON and wallet_notional > 0.0:
                reopen_notional = wallet_notional * min(1.0, reopen_size / wallet_size)
            else:
                reopen_notional = wallet_notional
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "next_status": "open",
                    "wallet_position_size": wallet_size,
                    "wallet_position_notional_usd": wallet_notional,
                    "reopened_wallet_position_size": reopen_size,
                    "reopened_wallet_position_notional_usd": reopen_notional,
                    "note": "reopened_wallet_position_authority",
                }
            )
            wallet_position_reopen_consumed_size_by_key[wallet_key] = consumed_reopen_size + reopen_size
            if not dry_run:
                wallet_size, wallet_notional, wallet_entry_price, _wallet_mark_price, payload = _apply_wallet_position_reopen(
                    row=row,
                    payload=payload,
                    wallet_position=wallet_position,
                    wallet_key=wallet_key,
                    now=now,
                    reopen_reason="wallet_position_still_held",
                    allocated_size=reopen_size,
                    allocated_notional_usd=reopen_notional,
                )
                hot_state.upsert_active_order(
                    trader_id=trader_id,
                    mode=str(row.mode or ""),
                    order_id=str(row.id or ""),
                    status=str(row.status or ""),
                    market_id=str(row.market_id or ""),
                    direction=str(row.direction or ""),
                    source=str(row.source or ""),
                    notional_usd=float(wallet_notional),
                    entry_price=float(wallet_entry_price or row.effective_price or row.entry_price or 0.0),
                    token_id=str(payload.get("token_id") or wallet_key),
                    filled_shares=float(wallet_size),
                    payload=payload,
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                )
                state_updates += 1
            candidates.append(row)
            candidate_ids.add(row_id)
            continue
        _exit_instance = None
        if isinstance(position_close, dict):
            close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
            close_price_source = str(position_close.get("price_source") or "").strip().lower()
            close_size = _position_close_filled_size(position_close)
            close_evidence_size = _position_close_evidence_size(position_close)
            entry_size = _entry_size_for_terminal_close(row, payload)
            close_evidence_key = _position_close_evidence_key(row, payload, position_close)
            close_uses_wallet_evidence = (
                close_price_source in {"wallet_activity", "wallet_trade"}
                or close_trigger in {"wallet_activity", "wallet_summary_recovery", "external_wallet_flatten"}
            )
            close_has_external_identity = _position_close_has_terminal_external_authority(position_close)
            close_underfills_entry = False
            needs_wallet_fill_repair = _terminal_row_needs_wallet_fill_repair_audit(row, now_naive)
            provider_entry_notional, provider_entry_size, _provider_entry_price = _extract_live_fill_metrics(payload)
            can_repair_entry_from_wallet_history = (
                provider_entry_notional <= 0.0
                and provider_entry_size <= _WALLET_SIZE_EPSILON
                and _row_can_recover_entry_from_wallet_history(row, payload)
            )
            if close_uses_wallet_evidence and not needs_wallet_fill_repair and not can_repair_entry_from_wallet_history:
                close_underfills_entry = _position_close_underfills_entry(
                    row,
                    payload,
                    position_close,
                ) or _position_close_missing_durable_wallet_identity(position_close)
                if not close_has_external_identity:
                    close_underfills_entry = close_underfills_entry or _position_close_missing_terminal_fill_authority(
                        row,
                        payload,
                        position_close,
                    )
            if (
                close_uses_wallet_evidence
                and not needs_wallet_fill_repair
                and not can_repair_entry_from_wallet_history
                and close_evidence_key
                and close_evidence_size > _WALLET_SIZE_EPSILON
                and entry_size > _WALLET_SIZE_EPSILON
            ):
                consumed_size = wallet_terminal_close_consumed_size_by_key.get(close_evidence_key, 0.0)
                available_size = max(0.0, close_evidence_size - consumed_size)
                if available_size + _WALLET_SIZE_EPSILON < entry_size * 0.995:
                    close_underfills_entry = True
                else:
                    close_underfills_entry = False
                    wallet_terminal_close_consumed_size_by_key[close_evidence_key] = consumed_size + min(
                        entry_size,
                        close_size if close_size > _WALLET_SIZE_EPSILON else close_evidence_size,
                    )
            if close_underfills_entry:
                pending_exit = payload.get("pending_live_exit")
                pending_exit = pending_exit if isinstance(pending_exit, dict) else {}
                if (
                    wallet_positions_loaded
                    # Same blip-defence as the no-provider wallet-absent path
                    # below (line ~6814) — refuse to flip a row to a wallet-
                    # absent variant when the wallet cache is empty (could be
                    # a Polymarket data-API blip rather than a genuinely
                    # flat wallet).
                    and wallet_positions_authoritative
                    and token_id
                    and wallet_position_size <= _WALLET_SIZE_EPSILON
                ):
                    resolved_at = _iso_utc(now)
                    provider_status = _provider_snapshot_status(payload)
                    provider_remaining_size = _provider_snapshot_remaining_size(payload)
                    previous_status = str(row.status or "")
                    if not dry_run:
                        row.status = "resolved"
                        row.actual_profit = None
                        row.updated_at = now
                        payload.pop("position_close", None)
                        if pending_exit:
                            pending_exit["status"] = "superseded_wallet_absent_underfilled_close"
                            pending_exit["resolved_at"] = resolved_at
                            pending_exit["terminal_close_filled_size"] = float(close_size)
                            pending_exit["required_exit_size"] = float(entry_size)
                            pending_exit["provider_status"] = provider_status or pending_exit.get("provider_status")
                            if provider_remaining_size is not None:
                                pending_exit["provider_remaining_size"] = provider_remaining_size
                            payload["pending_live_exit"] = pending_exit
                        payload["wallet_absent_resolution"] = {
                            "resolved_at": resolved_at,
                            "reason": "terminal_close_underfilled_entry_but_wallet_flat",
                            "previous_status": previous_status,
                            "close_trigger": close_trigger,
                            "price_source": close_price_source,
                            "close_filled_size": float(close_size),
                            "close_evidence_size": float(close_evidence_size),
                            "required_entry_size": float(entry_size),
                            "provider_status": provider_status,
                            "provider_remaining_size": provider_remaining_size,
                        }
                        row.payload_json = payload
                        hot_state.record_order_cancelled(
                            trader_id=trader_id,
                            mode=str(row.mode or ""),
                            order_id=str(row.id or ""),
                            market_id=str(row.market_id or ""),
                            source=str(row.source or ""),
                            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                        )
                        state_updates += 1
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "price_source": close_price_source,
                            "filled_size": close_size,
                            "required_entry_size": entry_size,
                            "next_status": "resolved",
                            "note": "neutralized_terminal_close_underfilled_entry_wallet_flat",
                        }
                    )
                    continue
                reopened_at = _iso_utc(now)
                if not dry_run:
                    row.status = "open"
                    row.actual_profit = None
                    row.updated_at = now
                    payload.pop("position_close", None)
                    if pending_exit:
                        pending_exit["status"] = (
                            "submitted"
                            if str(pending_exit.get("exit_order_id") or "").strip()
                            or str(pending_exit.get("provider_clob_order_id") or "").strip()
                            else "pending"
                        )
                        pending_exit["reopened_at"] = reopened_at
                        pending_exit["reopen_reason"] = "terminal_close_underfilled_entry"
                        pending_exit["terminal_close_filled_size"] = float(close_size)
                        pending_exit["required_exit_size"] = float(entry_size)
                        payload["pending_live_exit"] = pending_exit
                    payload["terminal_close_reopen"] = {
                        "reopened_at": reopened_at,
                        "reopen_reason": "terminal_close_underfilled_entry",
                        "close_trigger": close_trigger,
                        "price_source": close_price_source,
                        "close_filled_size": float(close_size),
                        "close_evidence_size": float(close_evidence_size),
                        "required_entry_size": float(entry_size),
                        "rejected_close_evidence_key": close_evidence_key or None,
                    }
                    row.payload_json = payload
                    state_updates += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "price_source": close_price_source,
                        "filled_size": close_size,
                        "required_entry_size": entry_size,
                        "next_status": "open",
                        "note": "reopened_terminal_close_underfilled_entry",
                    }
                )
                candidates.append(row)
                candidate_ids.add(row_id)
                continue
            if (
                close_price_source == "resolved_settlement"
                and terminal_wallet_settlement_price is None
                and wallet_position_size > _WALLET_SIZE_EPSILON
            ):
                reopened_at = _iso_utc(now)
                if not dry_run:
                    row.status = "open"
                    row.actual_profit = None
                    row.updated_at = now
                    payload.pop("position_close", None)
                    payload["resolution_reopen"] = {
                        "reopened_at": reopened_at,
                        "reopen_reason": "resolved_settlement_not_wallet_terminal",
                        "wallet_position_size": wallet_position_size,
                    }
                    row.payload_json = payload
                    state_updates += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "next_status": "open",
                        "note": "reopened_resolved_settlement_not_wallet_terminal",
                    }
                )
                candidates.append(row)
                candidate_ids.add(row_id)
                continue
            if close_price_source == "provider_exit_fill" and wallet_position_size > _WALLET_SIZE_EPSILON:
                pending_exit = payload.get("pending_live_exit")
                pending_exit = pending_exit if isinstance(pending_exit, dict) else {}
                provider_status = _provider_snapshot_status(payload)
                provider_remaining_size = _provider_snapshot_remaining_size(payload)
                reopened_at = _iso_utc(now)
                if not dry_run:
                    row.status = "open"
                    row.actual_profit = None
                    row.updated_at = now
                    payload.pop("position_close", None)
                    if pending_exit:
                        pending_exit["status"] = (
                            "submitted"
                            if str(pending_exit.get("exit_order_id") or "").strip()
                            or str(pending_exit.get("provider_clob_order_id") or "").strip()
                            else "pending"
                        )
                        pending_exit["reopened_at"] = reopened_at
                        pending_exit["reopen_reason"] = "provider_exit_fill_not_wallet_flat"
                        pending_exit["provider_status"] = provider_status or pending_exit.get("provider_status")
                        if provider_remaining_size is not None:
                            pending_exit["provider_remaining_size"] = provider_remaining_size
                        payload["pending_live_exit"] = pending_exit
                    payload["provider_exit_reopen"] = {
                        "reopened_at": reopened_at,
                        "reopen_reason": "provider_exit_fill_not_wallet_flat",
                        "provider_status": provider_status,
                        "provider_remaining_size": provider_remaining_size,
                    }
                    row.payload_json = payload
                    state_updates += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "provider_status": provider_status,
                        "provider_remaining_size": provider_remaining_size,
                        "next_status": "open",
                        "note": "reopened_provider_exit_fill_not_wallet_flat",
                    }
                )
                candidates.append(row)
                candidate_ids.add(row_id)
                continue
            if close_trigger in {"wallet_absent_close", "wallet_flat_override"}:
                pending_exit = payload.get("pending_live_exit")
                pending_exit = pending_exit if isinstance(pending_exit, dict) else {}
                provider_status = _provider_snapshot_status(payload)
                provider_remaining_size = _provider_snapshot_remaining_size(payload)
                fallback_close_verified = (
                    _pending_exit_has_verified_terminal_fill(pending_exit)
                    or bool(_position_close_wallet_trade_id(position_close))
                )
                reopen_reason = ""
                if not fallback_close_verified:
                    if close_trigger == "wallet_absent_close" and _provider_entry_order_still_working(payload):
                        reopen_reason = "provider_entry_order_still_open"
                    else:
                        reopen_reason = "fallback_close_not_verified"
                if (
                    reopen_reason == "fallback_close_not_verified"
                    and wallet_positions_loaded
                    and token_id
                    and wallet_position_size <= _WALLET_SIZE_EPSILON
                ):
                    if not dry_run:
                        pending_exit["status"] = "superseded_wallet_absent_unverified_close"
                        pending_exit["resolved_at"] = _iso_utc(now)
                        payload["pending_live_exit"] = pending_exit
                        payload.pop("position_close", None)
                        payload["wallet_absent_resolution"] = {
                            "resolved_at": _iso_utc(now),
                            "reason": f"{close_trigger}_unverified_but_wallet_flat",
                            "previous_status": row.status,
                            "provider_status": provider_status,
                            "provider_remaining_size": provider_remaining_size,
                        }
                        row.status = "resolved"
                        row.actual_profit = None
                        row.updated_at = now
                        row.payload_json = payload
                        hot_state.record_order_cancelled(
                            trader_id=trader_id,
                            mode=str(row.mode or ""),
                            order_id=str(row.id or ""),
                            market_id=str(row.market_id or ""),
                            source=str(row.source or ""),
                            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                        )
                        state_updates += 1
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "provider_status": provider_status,
                            "provider_remaining_size": provider_remaining_size,
                            "next_status": "resolved",
                            "note": "resolved_terminal_unverified_close_wallet_absent",
                        }
                    )
                    continue
                if reopen_reason:
                    reopen_key = "wallet_absent_reopen" if close_trigger == "wallet_absent_close" else "wallet_flat_reopen"
                    if not dry_run:
                        row.status = "open"
                        row.actual_profit = None
                        row.updated_at = now
                        payload.pop("position_close", None)
                        payload[reopen_key] = {
                            "reopened_at": _iso_utc(now),
                            "reopen_reason": reopen_reason,
                            "provider_status": provider_status,
                            "provider_remaining_size": provider_remaining_size,
                        }
                        row.payload_json = payload
                        state_updates += 1
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "provider_status": provider_status,
                            "provider_remaining_size": provider_remaining_size,
                            "next_status": "open",
                            "note": f"reopened_{close_trigger}_{reopen_reason}",
                        }
                    )
                    candidates.append(row)
                    candidate_ids.add(row_id)
                    continue
        pending_exit = payload.get("pending_live_exit")
        if not isinstance(pending_exit, dict):
            continue
        if isinstance(position_close, dict) and _position_close_has_terminal_external_authority(position_close):
            continue
        pending_status = str(pending_exit.get("status") or "").strip().lower()
        if pending_status != "filled":
            continue
        if _safe_bool(pending_exit.get("allow_partial_fill_terminal"), False):
            continue
        close_trigger = str(pending_exit.get("close_trigger") or "").strip().lower()
        if "resolution" in close_trigger:
            continue
        required_exit_size = max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0)
        _entry_notional, _entry_size, _entry_price = _extract_live_fill_metrics(payload)
        if _entry_size <= 0.0 and _entry_notional > 0.0:
            _fallback_entry_price = (
                _entry_price if _entry_price and _entry_price > 0 else safe_float(row.effective_price)
            )
            if _fallback_entry_price is None or _fallback_entry_price <= 0:
                _fallback_entry_price = safe_float(row.entry_price)
            if _fallback_entry_price and _fallback_entry_price > 0:
                _entry_size = _entry_notional / _fallback_entry_price
        if _entry_size > 0.0:
            required_exit_size = max(required_exit_size, _entry_size)
        if required_exit_size <= 0.0:
            continue
        filled_exit_size = max(0.0, safe_float(pending_exit.get("filled_size"), 0.0) or 0.0)
        fill_ratio = safe_float(pending_exit.get("fill_ratio"))
        if fill_ratio is None:
            fill_ratio = (filled_exit_size / required_exit_size) if required_exit_size > 0.0 else 0.0
        threshold_ratio = _pending_exit_fill_threshold(pending_exit)
        if fill_ratio + 1e-9 >= threshold_ratio:
            continue
        if not dry_run:
            row.status = "open"
            row.actual_profit = None
            row.updated_at = now
            pending_exit["status"] = (
                "submitted"
                if str(pending_exit.get("exit_order_id") or "").strip()
                or str(pending_exit.get("provider_clob_order_id") or "").strip()
                else "pending"
            )
            pending_exit["reopened_at"] = _iso_utc(now)
            pending_exit["reopen_reason"] = "partial_exit_fill_below_threshold"
            pending_exit["fill_ratio"] = float(fill_ratio)
            payload["pending_live_exit"] = pending_exit
            payload.pop("position_close", None)
            row.payload_json = payload
            state_updates += 1
        details.append(
            {
                "order_id": row.id,
                "market_id": row.market_id,
                "close_trigger": str(pending_exit.get("close_trigger") or "live_exit_fill"),
                "filled_size": filled_exit_size,
                "required_exit_size": required_exit_size,
                "fill_ratio": float(fill_ratio),
                "fill_ratio_threshold": threshold_ratio,
                "next_status": "open",
                "note": "reopened_partial_exit_fill_below_threshold",
            }
        )
        candidates.append(row)
        candidate_ids.add(row_id)

    if not dry_run and terminal_rows:
        for row in terminal_rows:
            if str(row.id) in candidate_ids:
                continue
            payload = dict(row.payload_json or {})
            _mark_terminal_reopen_audited(row, payload, now, "no_reopen_authority")

    if wallet_positions_loaded and candidates:
        active_by_wallet_key: dict[str, list[tuple[TraderOrder, dict[str, Any], dict[str, Any]]]] = {}
        for row in candidates:
            payload = dict(row.payload_json or {})
            if _pending_exit_blocks_wallet_position_collapse(payload):
                continue
            wallet_key, wallet_position = _lookup_wallet_position_for_payload(
                wallet_positions_by_token,
                payload,
                str(row.direction or ""),
            )
            if not wallet_key or not isinstance(wallet_position, dict):
                continue
            if _extract_wallet_position_size(wallet_position) <= _WALLET_SIZE_EPSILON:
                continue
            active_by_wallet_key.setdefault(wallet_key, []).append((row, payload, wallet_position))

        collapsed_duplicate_ids: set[str] = set()
        for wallet_key, entries in active_by_wallet_key.items():
            if len(entries) <= 1:
                continue
            entries.sort(
                key=lambda item: (
                    _row_entry_anchor_naive(item[0]) or datetime.min,
                    str(item[0].id or ""),
                )
            )
            canonical_row, canonical_payload, wallet_position = entries[-1]
            wallet_size, wallet_notional, wallet_entry_price, wallet_mark_price = _wallet_position_notional(wallet_position)
            if wallet_size <= _WALLET_SIZE_EPSILON or wallet_notional <= 0.0:
                continue
            duplicate_entries = entries[:-1]
            duplicate_order_ids = [str(row.id or "") for row, _, _ in duplicate_entries if str(row.id or "")]
            duplicate_clob_ids = [
                str(row.provider_clob_order_id or "")
                for row, _, _ in duplicate_entries
                if str(row.provider_clob_order_id or "").strip()
            ]
            duplicate_clob_ids.extend(
                str(payload.get("provider_clob_order_id") or "")
                for _, payload, _ in duplicate_entries
                if str(payload.get("provider_clob_order_id") or "").strip()
            )
            duplicate_clob_ids = sorted({value for value in duplicate_clob_ids if value})
            canonical_clob_ids = [
                str(canonical_row.provider_clob_order_id or "").strip(),
                str(canonical_payload.get("provider_clob_order_id") or "").strip(),
            ]
            canonical_clob_ids = sorted({value for value in canonical_clob_ids if value})
            # Same guard as _apply_wallet_position_reopen — don't
            # downgrade rows already verified against Polymarket
            # truth. See that function for the full rationale.
            _existing_verification = str(getattr(canonical_row, "verification_status", None) or "").strip().lower()
            _existing_status = str(getattr(canonical_row, "status", None) or "").strip().lower()
            _is_resolved_status = _existing_status in {
                "resolved", "resolved_win", "resolved_loss",
                "closed_win", "closed_loss", "win", "loss",
            }
            if _existing_verification == "wallet_activity" and _is_resolved_status:
                continue
            if not dry_run:
                _previous_status = str(canonical_row.status or "").strip().lower()
                canonical_row.status = "executed"
                canonical_row.notional_usd = float(wallet_notional)
                if wallet_entry_price is not None and wallet_entry_price > 0.0:
                    canonical_row.entry_price = float(wallet_entry_price)
                    canonical_row.effective_price = float(wallet_entry_price)
                canonical_row.verification_status = "wallet_position"
                canonical_row.verification_source = "wallet_positions_api"
                canonical_row.updated_at = now

                # Strategy hook: fire on_fill for the asynchronously-
                # discovered live fill.  This is the canonical
                # ground-truth point — the venue confirmed our order
                # actually filled and the wallet now holds the
                # position.  Skip if we're transitioning from an
                # already-executed status (no new fill event).
                if _previous_status not in {"executed", "closed_win", "closed_loss", "resolved", "resolved_win", "resolved_loss"}:
                    try:
                        _strategy_slug = str(canonical_row.strategy_key or "").strip()
                        if _strategy_slug:
                            from services.strategy_callbacks import dispatch_on_fill

                            asyncio.create_task(
                                dispatch_on_fill(
                                    strategy_slug=_strategy_slug,
                                    order=canonical_row,
                                    mode="live",
                                    filled_shares=float(wallet_size or 0.0),
                                    average_price=float(wallet_entry_price or 0.0),
                                    notional_usd=float(wallet_notional or 0.0),
                                    ensemble_snapshot=None,
                                )
                            )
                    except Exception:
                        pass  # never raise from a fill notification
                provider_reconciliation = canonical_payload.get("provider_reconciliation")
                provider_reconciliation = provider_reconciliation if isinstance(provider_reconciliation, dict) else {}
                snapshot = provider_reconciliation.get("snapshot")
                snapshot = snapshot if isinstance(snapshot, dict) else {}
                snapshot.update(
                    {
                        "normalized_status": "filled",
                        "status": "filled",
                        "filled_size": float(wallet_size),
                        "filled_notional_usd": float(wallet_notional),
                        "average_fill_price": float(wallet_entry_price) if wallet_entry_price is not None else None,
                        "source": "wallet_position_aggregate",
                    }
                )
                provider_reconciliation.update(
                    {
                        "snapshot": snapshot,
                        "snapshot_status": "filled",
                        "mapped_status": "executed",
                        "filled_size": float(wallet_size),
                        "filled_notional_usd": float(wallet_notional),
                        "average_fill_price": float(wallet_entry_price) if wallet_entry_price is not None else None,
                        "source": "wallet_position_aggregate",
                    }
                )
                canonical_payload["provider_reconciliation"] = provider_reconciliation
                canonical_payload["live_wallet_authority"] = {
                    "source": "wallet_positions_api",
                    "token_id": str(wallet_position.get("token_id") or wallet_position.get("asset") or wallet_key),
                    "wallet_address": str(
                        wallet_position.get("wallet_address")
                        or wallet_position.get("proxyWallet")
                        or canonical_payload.get("execution_wallet_address")
                        or ""
                    ).strip()
                    or None,
                    "recovered_at": _iso_utc(now),
                }
                canonical_payload["wallet_position_aggregate_authority"] = {
                    "wallet_key": wallet_key,
                    "aggregate_size": float(wallet_size),
                    "aggregate_notional_usd": float(wallet_notional),
                    "average_entry_price": float(wallet_entry_price) if wallet_entry_price is not None else None,
                    "current_mark_price": float(wallet_mark_price) if wallet_mark_price is not None else None,
                    "canonical_order_id": str(canonical_row.id or ""),
                    "canonical_provider_clob_order_ids": canonical_clob_ids,
                    "collapsed_order_ids": duplicate_order_ids,
                    "collapsed_provider_clob_order_ids": duplicate_clob_ids,
                    "collapsed_at": _iso_utc(now),
                }
                canonical_row.payload_json = canonical_payload
                extra_touched_rows.append(canonical_row)
                hot_state.upsert_active_order(
                    trader_id=trader_id,
                    mode=str(canonical_row.mode or ""),
                    order_id=str(canonical_row.id or ""),
                    status=str(canonical_row.status or ""),
                    market_id=str(canonical_row.market_id or ""),
                    direction=str(canonical_row.direction or ""),
                    source=str(canonical_row.source or ""),
                    notional_usd=float(wallet_notional),
                    entry_price=float(wallet_entry_price or canonical_row.effective_price or canonical_row.entry_price or 0.0),
                    token_id=str(canonical_payload.get("token_id") or wallet_position.get("token_id") or wallet_key),
                    filled_shares=float(wallet_size),
                    payload=canonical_payload,
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(canonical_payload),
                )
                state_updates += 1

            for duplicate_row, duplicate_payload, _duplicate_wallet_position in duplicate_entries:
                duplicate_id = str(duplicate_row.id or "")
                if duplicate_id:
                    collapsed_duplicate_ids.add(duplicate_id)
                if not dry_run:
                    duplicate_row.status = "resolved"
                    duplicate_row.notional_usd = 0.0
                    duplicate_row.actual_profit = None
                    duplicate_row.updated_at = now
                    duplicate_payload["duplicate_wallet_position_collapse"] = {
                        "wallet_key": wallet_key,
                        "canonical_order_id": str(canonical_row.id or ""),
                        "collapsed_at": _iso_utc(now),
                        "reason": "same wallet position already represented by canonical order",
                    }
                    duplicate_row.payload_json = duplicate_payload
                    extra_touched_rows.append(duplicate_row)
                    hot_state.record_order_cancelled(
                        trader_id=trader_id,
                        mode=str(duplicate_row.mode or ""),
                        order_id=str(duplicate_row.id or ""),
                        market_id=str(duplicate_row.market_id or ""),
                        source=str(duplicate_row.source or ""),
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(duplicate_payload),
                    )
                    state_updates += 1

            details.append(
                {
                    "order_id": canonical_row.id,
                    "market_id": canonical_row.market_id,
                    "wallet_key": wallet_key,
                    "aggregate_size": wallet_size,
                    "aggregate_notional_usd": wallet_notional,
                    "collapsed_order_ids": duplicate_order_ids,
                    "next_status": "executed",
                    "note": "collapsed_duplicate_wallet_position_rows",
                }
            )

        if collapsed_duplicate_ids:
            candidates = [row for row in candidates if str(row.id or "") not in collapsed_duplicate_ids]
            candidate_ids.difference_update(collapsed_duplicate_ids)

    def _candidate_needs_wallet_history(row: TraderOrder) -> bool:
        payload = dict(row.payload_json or {})
        pending_exit = payload.get("pending_live_exit")
        if isinstance(pending_exit, dict):
            pending_status = str(pending_exit.get("status") or "").strip().lower()
            if pending_status in {"pending", "submitted", "working", "filled"}:
                return True
            # Laddered exits: any non-terminal child means we need wallet
            # history to verify partial fills against wallet activity.
            children = pending_exit.get("children")
            if isinstance(children, list) and children:
                for child in children:
                    if not isinstance(child, dict):
                        continue
                    cs = str(child.get("status") or "").strip().lower()
                    if cs in {"submitted", "partial", "planned", "escalated"}:
                        return True
        token_id = _extract_live_token_id(payload)
        if not token_id:
            return False
        filled_notional, filled_size, _ = _extract_live_fill_metrics(payload)
        wallet_position = wallet_positions_by_token.get(token_id)
        if not isinstance(wallet_position, dict):
            cond_key = _extract_condition_outcome_key(payload, str(row.direction or ""))
            if cond_key:
                wallet_position = wallet_positions_by_token.get(cond_key)
        if isinstance(wallet_position, dict) and _extract_wallet_position_size(wallet_position) <= _WALLET_SIZE_EPSILON:
            return True
        age_anchor = getattr(row, "updated_at", None) or getattr(row, "executed_at", None) or getattr(row, "created_at", None)
        if not isinstance(age_anchor, datetime):
            return False
        if age_anchor.tzinfo is not None:
            age_anchor = age_anchor.astimezone(timezone.utc).replace(tzinfo=None)
        order_age_seconds = max(0.0, (now_naive - age_anchor).total_seconds())
        if order_age_seconds < _WALLET_HISTORY_GRACE_SECONDS:
            return False
        if filled_notional <= 0.0 and filled_size <= _WALLET_SIZE_EPSILON and _row_can_recover_entry_from_wallet_history(row, payload):
            return True
        if not isinstance(wallet_position, dict):
            return True
        return _extract_wallet_position_size(wallet_position) <= _WALLET_SIZE_EPSILON

    signal_ids = [str(row.signal_id) for row in candidates if row.signal_id]
    signal_payloads: dict[str, dict[str, Any]] = {}
    if signal_ids:
        signal_rows = (
            await session.execute(
                select(TradeSignal.id, TradeSignal.payload_json).where(TradeSignal.id.in_(signal_ids))
            )
        ).all()
        signal_payloads = {str(row.id): dict(row.payload_json or {}) for row in signal_rows}

    _lc_t1 = _time.monotonic()
    wallet_buy_trades_by_token: dict[str, list[dict[str, Any]]] = {}
    terminal_rows_needing_wallet_fill_repair = [
        row for row in terminal_rows if _terminal_row_needs_wallet_fill_repair_audit(row, now_naive)
    ]
    if any(_candidate_needs_wallet_history(row) for row in candidates) or terminal_rows_needing_wallet_fill_repair:
        async with release_conn(session):
            (
                wallet_closed_positions_by_token,
                wallet_buy_trades_by_token,
                wallet_sell_trades_by_token,
                wallet_close_activity_by_token,
            ) = await asyncio.gather(
                _load_mapping_with_timeout(
                    _load_execution_wallet_closed_positions_by_token,
                    timeout=_WALLET_HISTORY_LOAD_TIMEOUT_SECONDS,
                    fallback=dict(_wallet_closed_positions_cache[1] or {}),
                ),
                _load_mapping_with_timeout(
                    _load_execution_wallet_recent_buy_trades_by_token,
                    timeout=_WALLET_HISTORY_LOAD_TIMEOUT_SECONDS,
                    fallback=dict(_wallet_buy_trades_cache[1] or {}),
                ),
                _load_mapping_with_timeout(
                    _load_execution_wallet_recent_sell_trades_by_token,
                    timeout=_WALLET_HISTORY_LOAD_TIMEOUT_SECONDS,
                    fallback=dict(_wallet_sell_trades_cache[1] or {}),
                ),
                _load_mapping_with_timeout(
                    _load_execution_wallet_recent_close_activity_by_token,
                    timeout=_WALLET_HISTORY_LOAD_TIMEOUT_SECONDS,
                    fallback=dict(_wallet_activity_cache[1] or {}),
                ),
            )
        wallet_closed_positions_loaded = bool(
            _wallet_closed_positions_last_refresh_succeeded or wallet_closed_positions_by_token
        )
    wallet_buy_trade_match_rows = list(candidates)
    wallet_buy_trade_match_rows.extend(
        row
        for row in terminal_rows_needing_wallet_fill_repair
        if str(row.id or "") not in candidate_ids
    )
    wallet_buy_trade_by_order_id = _build_wallet_buy_trade_matches(wallet_buy_trade_match_rows, wallet_buy_trades_by_token)
    aggregate_wallet_backfill_order_ids = _select_aggregate_wallet_backfill_order_ids(candidates)

    for row in terminal_rows_needing_wallet_fill_repair:
        row_id = str(row.id or "")
        if row_id in candidate_ids:
            continue
        repair_state = _terminal_wallet_fill_repair_state(
            row,
            wallet_buy_trade_by_order_id=wallet_buy_trade_by_order_id,
        )
        if repair_state is None:
            continue
        payload, wallet_entry_trade, recovered_notional, recovered_price, close_trigger = repair_state
        token_id = _extract_live_token_id(payload)
        closed_position = wallet_closed_positions_by_token.get(token_id) if token_id else None
        wallet_close_activity = wallet_close_activity_by_token.get(token_id) if token_id else None
        latest_wallet_sell_trade = wallet_sell_trades_by_token.get(token_id) if token_id else None
        if not isinstance(wallet_close_activity, dict) and token_id:
            cond_key = _extract_condition_outcome_key(payload, str(row.direction or ""))
            if cond_key:
                closed_position = wallet_closed_positions_by_token.get(cond_key)
                wallet_close_activity = wallet_close_activity_by_token.get(cond_key)
        if (
            not isinstance(wallet_close_activity, dict)
            and not isinstance(closed_position, dict)
            and not isinstance(latest_wallet_sell_trade, dict)
        ):
            continue
        details.append(
            {
                "order_id": row.id,
                "market_id": row.market_id,
                "next_status": "open",
                "prior_close_trigger": close_trigger,
                "recovered_notional": recovered_notional,
                "recovered_price": recovered_price,
                "note": "reopened_wallet_entry_fill_repair",
            }
        )
        if not dry_run:
            row.status = "open"
            row.actual_profit = None
            row.updated_at = now
            payload.pop("position_close", None)
            payload["wallet_entry_fill_repair"] = {
                "reopened_at": _iso_utc(now),
                "reopen_reason": "wallet_trade_entry_fill_repair",
                "prior_close_trigger": close_trigger,
                "size": _extract_wallet_trade_size(wallet_entry_trade),
                "price": _extract_wallet_trade_price(wallet_entry_trade),
                "timestamp": (
                    _iso_utc(wallet_entry_trade.get("timestamp"))
                    if isinstance(wallet_entry_trade.get("timestamp"), datetime)
                    else None
                ),
                "transaction_hash": str(
                    wallet_entry_trade.get("transactionHash")
                    or wallet_entry_trade.get("txHash")
                    or wallet_entry_trade.get("tx_hash")
                    or ""
                ).strip()
                or None,
            }
            row.payload_json = payload
            state_updates += 1
        candidates.append(row)
        candidate_ids.add(row_id)

    # Release the DB connection back to the pool while we do external I/O
    # (Polymarket API, WS cache, CLOB midpoints).  The session
    # lazily reconnects on the next DB operation.
    fallback_market_info_by_id = _fallback_market_info_for_orders(candidates)
    async with release_conn(session):
        market_info_by_id = await _load_mapping_with_timeout(
            lambda: load_market_info_for_orders(candidates),
            timeout=_MARKET_INFO_LOAD_TIMEOUT_SECONDS,
            fallback=fallback_market_info_by_id,
        )
        ws_mid_prices: dict[str, float] = {}
        clob_mid_prices: dict[str, float] = {}
        token_ids_for_prices = sorted(
            {
                token_id
                for token_id in (_extract_live_token_id(dict(row.payload_json or {})) for row in candidates)
                if token_id
            }
        )
        if token_ids_for_prices:
            strict_stale_seconds = max(0.05, float(settings.WS_EXECUTION_PRICE_STALE_SECONDS))
            relaxed_stale_seconds = max(strict_stale_seconds, float(settings.WS_PRICE_STALE_SECONDS))
            try:
                from services.ws_feeds import get_feed_manager

                feed_manager = get_feed_manager()
            except Exception:
                feed_manager = None

            def _collect_ws_mid_prices(token_ids: list[str], *, stale_seconds: float) -> dict[str, float]:
                if feed_manager is None or not getattr(feed_manager, "_started", False):
                    return {}
                cache = getattr(feed_manager, "cache", None)
                if cache is None:
                    return {}
                out: dict[str, float] = {}
                for token_id in token_ids:
                    try:
                        mid = safe_float(cache.get_mid_price(token_id))
                        age_s = safe_float(cache.staleness(token_id))
                    except Exception:
                        continue
                    if mid is None or mid < 0:
                        continue
                    if age_s is not None and age_s > stale_seconds:
                        continue
                    out[str(token_id).strip()] = float(mid)
                return out

            ws_mid_prices.update(_collect_ws_mid_prices(token_ids_for_prices, stale_seconds=strict_stale_seconds))
            unresolved_token_ids = [token_id for token_id in token_ids_for_prices if token_id not in ws_mid_prices]
            if unresolved_token_ids and relaxed_stale_seconds > strict_stale_seconds + 1e-9:
                ws_mid_prices.update(
                    _collect_ws_mid_prices(
                        unresolved_token_ids,
                        stale_seconds=relaxed_stale_seconds,
                    )
                )
            unresolved_token_ids = [token_id for token_id in token_ids_for_prices if token_id not in ws_mid_prices]
            if unresolved_token_ids and allow_polymarket_rest_call("midpoints_batch"):
                # ONE HTTP request for all unresolved tokens via the
                # CLOB /midpoints batch endpoint instead of N parallel
                # single-token GETs.  Skipped on the orchestrator hot
                # path — the WS feed is supposed to have these
                # tokens; if it doesn't we use the cached/live midpoint
                # already in ``ws_mid_prices`` and accept the gap.
                try:
                    batch = await asyncio.wait_for(
                        polymarket_client.get_midpoints_batch(unresolved_token_ids),
                        timeout=4.0,
                    )
                except Exception:
                    batch = {}
            elif unresolved_token_ids:
                batch = {}
                for token_id, midpoint in batch.items():
                    parsed = safe_float(midpoint)
                    if parsed is None or parsed < 0:
                        continue
                    clob_mid_prices[str(token_id).strip()] = float(parsed)

        # Collect parent + child clob_order_ids that we need provider snapshots
        # for. A laddered exit has no provider id on the parent — the venue
        # ids live on each child.
        _pending_provider_id_set: set[str] = set()
        _pending_order_id_set: set[str] = set()
        for _row in candidates:
            _pe = (dict((_row.payload_json or {})).get("pending_live_exit"))
            if not isinstance(_pe, dict):
                continue
            _status = str(_pe.get("status") or "").strip().lower()
            _children = _pe.get("children") if isinstance(_pe.get("children"), list) else None
            if _children:
                # Laddered exit: collect every child's ids, regardless of the
                # parent's status (which is purely a high-level marker).
                for _child in _children:
                    if not isinstance(_child, dict):
                        continue
                    _cid = str(_child.get("provider_clob_order_id") or "").strip()
                    if _cid:
                        _pending_provider_id_set.add(_cid)
                    _oid = str(_child.get("exit_order_id") or "").strip()
                    if _oid:
                        _pending_order_id_set.add(_oid)
            else:
                if _status in {"submitted", "pending"}:
                    _pid = _pending_exit_provider_clob_id(_pe)
                    if _pid:
                        _pending_provider_id_set.add(_pid)
                    _oid = str(_pe.get("exit_order_id") or "").strip()
                    if _oid:
                        _pending_order_id_set.add(_oid)
        pending_exit_provider_ids = sorted(_pending_provider_id_set)
        pending_exit_order_ids = sorted(_pending_order_id_set)
        pending_exit_snapshots: dict[str, dict[str, Any]] = {}
        if pending_exit_provider_ids:
            pending_exit_snapshots = await _load_mapping_with_timeout(
                lambda: live_execution_service.get_order_snapshots_by_clob_ids(pending_exit_provider_ids),
                timeout=_ORDER_SNAPSHOT_LOAD_TIMEOUT_SECONDS,
                fallback=_cached_order_snapshots_by_clob_ids(pending_exit_provider_ids),
            )

        # Backfill: fetch entry fill snapshots for orders missing entry price.
        # IOC/FAK orders may have average_fill_price=0 in LiveTradingOrder because
        # the venue response didn't include fill price and the order completed before
        # the sync loop caught it.
        entry_backfill_snapshots: dict[str, dict[str, Any]] = {}
        entry_backfill_clob_ids: list[str] = []
        for row in candidates:
            _bf_payload = dict(row.payload_json or {})
            _bf_price = safe_float(row.effective_price) or safe_float(row.entry_price) or 0.0
            if _bf_price > 0:
                continue
            _bf_recon = _bf_payload.get("provider_reconciliation")
            if isinstance(_bf_recon, dict):
                _bf_price = safe_float(_bf_recon.get("average_fill_price")) or 0.0
            if _bf_price > 0:
                continue
            _bf_clob = str(_bf_payload.get("provider_clob_order_id") or "").strip()
            if _bf_clob:
                entry_backfill_clob_ids.append(_bf_clob)
        if entry_backfill_clob_ids:
            entry_backfill_snapshots = await _load_mapping_with_timeout(
                lambda: live_execution_service.get_order_snapshots_by_clob_ids(entry_backfill_clob_ids),
                timeout=_ORDER_SNAPSHOT_LOAD_TIMEOUT_SECONDS,
                fallback=_cached_order_snapshots_by_clob_ids(entry_backfill_clob_ids),
            )

    _lc_t2 = _time.monotonic()

    pending_exit_snapshot_fallbacks: dict[str, dict[str, Any]] = {}
    if pending_exit_provider_ids or pending_exit_order_ids:
        pending_exit_live_rows = (
            await session.execute(
                select(LiveTradingOrder).where(
                    or_(
                        LiveTradingOrder.clob_order_id.in_(pending_exit_provider_ids) if pending_exit_provider_ids else False,
                        LiveTradingOrder.id.in_(pending_exit_order_ids) if pending_exit_order_ids else False,
                    )
                )
            )
        ).scalars().all()
        for live_row in pending_exit_live_rows:
            snapshot = _live_trading_order_snapshot(live_row)
            clob_order_id = str(snapshot.get("clob_order_id") or "").strip()
            if clob_order_id and clob_order_id not in pending_exit_snapshots:
                pending_exit_snapshot_fallbacks[clob_order_id] = snapshot

    # Per-pass exit-submission budget.  Each execute_live_order call can take
    # up to _LIVE_EXIT_ORDER_TIMEOUT_SECONDS (12s); serial submits over 20+
    # candidates would blow the 30s per-trader reconciliation window and
    # starve the fast-tier pool.  We cap submissions per pass; the rest are
    # held and picked up on the next cycle in age-priority order.  Counter
    # is wrapped in a list so the per-row closures can mutate it.
    exit_submissions_this_pass: list[int] = [0]
    live_position_notional_cap = _live_position_notional_cap(params)

    # Per-candidate timing: capture wall time spent on each row so the
    # timing log can surface the slowest candidates and operators can
    # attribute the ``processing=Xs`` bucket to specific orders.  We
    # bracket iterations using the loop boundary itself rather than
    # try/finally because the processing block has many ``continue``
    # exits.
    _per_candidate_timings: list[tuple[str, float]] = []
    _last_iter_start: float = _time.monotonic()
    _last_iter_row_id: str | None = None

    for row in candidates:
        _now_iter = _time.monotonic()
        if _last_iter_row_id is not None:
            _per_candidate_timings.append((_last_iter_row_id, _now_iter - _last_iter_start))
        _last_iter_row_id = str(getattr(row, "id", "") or "")
        _last_iter_start = _now_iter
        # The per-candidate processing block below is pure Python and
        # historically runs ~3s/candidate × 7 candidates = ~20s of
        # synchronous compute, which blocks the asyncio loop and
        # starves the fast-tier trader cycle running concurrently.
        # Yield once per candidate so other tasks can interleave; this
        # does not change total reconcile time, only the latency
        # other tasks see while it runs.
        await asyncio.sleep(0)
        payload = dict(row.payload_json or {})
        # Bundle-residual zombie short-circuit.  When the session_engine's
        # bundle recovery path leaves residual wallet exposure on a leg
        # that did fill, retrying the flatten every reconcile cycle will
        # never succeed (the V2 wallet/balance state that caused the
        # original failure is the same).  Each attempt costs ~10s on the
        # ``asyncio.wait_for`` budget for the SELL submit, tears asyncpg
        # mid-protocol, and saturates the per-trader cycle.  Detect the
        # marker (or the legacy reason-text on rows persisted before the
        # marker existed) and terminalize the order as a 100% loss
        # immediately so it stops being a candidate.
        bundle_recovery_residual_marker = payload.get("bundle_recovery_residual")
        legacy_reason_match = (
            isinstance(getattr(row, "reason", None), str)
            and "residual exposure remains" in row.reason
            and str(row.status or "").strip().lower() == "executed"
        )
        if (
            (
                isinstance(bundle_recovery_residual_marker, dict)
                and str(bundle_recovery_residual_marker.get("status") or "").strip().lower() == "unrecoverable"
            )
            or legacy_reason_match
        ):
            close_trigger = "bundle_residual_unrecoverable"
            filled_notional = safe_float(row.notional_usd, 0.0) or 0.0
            pnl = -filled_notional
            next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "close_trigger": close_trigger,
                    "next_status": next_status,
                    "note": "terminalized_bundle_residual_unrecoverable",
                }
            )
            if not dry_run:
                row.status = next_status
                row.actual_profit = float(pnl)
                row.updated_at = now
                payload["position_close"] = {
                    "close_price": 0.0,
                    "price_source": "bundle_residual_unrecoverable",
                    "close_trigger": close_trigger,
                    "realized_pnl": pnl,
                    "filled_notional_usd": filled_notional,
                    "closed_at": _iso_utc(now),
                    "reason": (
                        "Bundle recovery left residual wallet exposure that the "
                        "session-engine flatten could not unwind. Terminalized to "
                        "stop reconcile-cycle retry storms; operator must clean "
                        "the residual position manually."
                    ),
                }
                row.payload_json = payload
                hot_state.record_order_resolved(
                    trader_id=trader_id,
                    mode=str(row.mode or ""),
                    order_id=str(row.id or ""),
                    market_id=str(row.market_id or ""),
                    direction=str(row.direction or ""),
                    source=str(row.source or ""),
                    status=next_status,
                    actual_profit=float(pnl),
                    payload=payload,
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                )
                closed += 1
            total_realized_pnl += pnl
            by_status[next_status] = int(by_status.get(next_status, 0)) + 1
            would_close += 1
            continue
        terminal_position_close = payload.get("position_close")
        if isinstance(terminal_position_close, dict) and _position_close_has_terminal_external_authority(
            terminal_position_close
        ):
            close_trigger = str(terminal_position_close.get("close_trigger") or "position_close").strip() or "position_close"
            pnl = safe_float(
                terminal_position_close.get("realized_pnl"),
                safe_float(row.actual_profit, 0.0) or 0.0,
            ) or 0.0
            next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "close_trigger": close_trigger,
                    "next_status": next_status,
                    "note": "terminalized_active_row_with_position_close",
                }
            )
            if not dry_run:
                row.status = next_status
                row.actual_profit = float(pnl)
                row.updated_at = now
                terminal_position_close.setdefault("closed_at", _iso_utc(now))
                pending_exit = payload.get("pending_live_exit")
                if isinstance(pending_exit, dict):
                    pending_status = str(pending_exit.get("status") or "").strip().lower()
                    if pending_status not in {
                        "filled",
                        "superseded_resolution",
                        "superseded_wallet_activity",
                        "superseded_wallet_flat",
                        "superseded_external",
                        "cancelled",
                    }:
                        pending_exit["status"] = "superseded_position_close"
                        pending_exit["resolved_at"] = terminal_position_close["closed_at"]
                        payload["pending_live_exit"] = pending_exit
                payload["position_close"] = terminal_position_close
                row.payload_json = payload
                _apply_position_close_verification(
                    session,
                    row=row,
                    now=now,
                    event_type="position_close_status_repair",
                    payload_json=payload,
                )
                hot_state.record_order_resolved(
                    trader_id=trader_id,
                    mode=str(row.mode or ""),
                    order_id=str(row.id or ""),
                    market_id=str(row.market_id or ""),
                    direction=str(row.direction or ""),
                    source=str(row.source or ""),
                    status=next_status,
                    actual_profit=pnl,
                    payload=payload,
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                )
                closed += 1
            total_realized_pnl += pnl
            by_status[next_status] = int(by_status.get(next_status, 0)) + 1
            would_close += 1
            continue
        _exit_instance = None
        token_id = _extract_live_token_id(payload)
        take_profit_pct = safe_float(_payload_exit_param(payload, prefix_key="live", name="take_profit_pct"))
        stop_loss_pct = safe_float(_payload_exit_param(payload, prefix_key="live", name="stop_loss_pct"))
        max_hold_minutes = safe_float(_payload_exit_param(payload, prefix_key="live", name="max_hold_minutes"))
        min_hold_minutes = max(
            0.0,
            safe_float(_payload_exit_param(payload, prefix_key="live", name="min_hold_minutes")) or 0.0,
        )
        trailing_stop_pct = safe_float(_payload_exit_param(payload, prefix_key="live", name="trailing_stop_pct"))
        resolve_only = _safe_bool(_payload_exit_param(payload, prefix_key="live", name="resolve_only"), False)
        close_on_inactive_market = _safe_bool(
            _payload_exit_param(payload, prefix_key="live", name="close_on_inactive_market"),
            False,
        )
        wallet_position = wallet_positions_by_token.get(token_id) if token_id else None
        closed_position = wallet_closed_positions_by_token.get(token_id) if token_id else None
        wallet_close_activity = wallet_close_activity_by_token.get(token_id) if token_id else None
        # Fallback: if exact token_id lookup fails (token versioning), try
        # conditionId:outcomeIndex which is stable across token versions.
        if not isinstance(wallet_position, dict) and token_id:
            cond_key = _extract_condition_outcome_key(payload, str(row.direction or ""))
            if cond_key:
                wallet_position = wallet_positions_by_token.get(cond_key)
                closed_position = wallet_closed_positions_by_token.get(cond_key)
                wallet_close_activity = wallet_close_activity_by_token.get(cond_key)
        wallet_position_observed = isinstance(wallet_position, dict)
        wallet_position_size = _extract_wallet_position_size(wallet_position)
        wallet_mark_price = _extract_wallet_mark_price(wallet_position)
        wallet_settlement_price = _extract_wallet_settlement_price(wallet_position)
        latest_wallet_sell_trade = wallet_sell_trades_by_token.get(token_id) if token_id else None
        latest_wallet_sell_trade_reopen_blocked = False
        if token_id and isinstance(latest_wallet_sell_trade, dict):
            latest_wallet_sell_trade_reopen_blocked = _terminal_reopen_blocks_wallet_close(
                payload,
                source="wallet_trade",
                evidence_key=_wallet_close_evidence_key("wallet_trade", token_id, latest_wallet_sell_trade),
                evidence_timestamp=latest_wallet_sell_trade.get("timestamp")
                or latest_wallet_sell_trade.get("created_at")
                or latest_wallet_sell_trade.get("createdAt")
                or latest_wallet_sell_trade.get("time"),
            )
        closed_position_reopen_blocked = False
        if token_id and isinstance(closed_position, dict):
            closed_position_reopen_blocked = _terminal_reopen_blocks_wallet_close(
                payload,
                source="closed_positions_api",
                evidence_timestamp=closed_position.get("timestamp")
                or closed_position.get("created_at")
                or closed_position.get("createdAt")
                or closed_position.get("time"),
            )
        provider_reconciliation = payload.get("provider_reconciliation")
        provider_reconciliation = provider_reconciliation if isinstance(provider_reconciliation, dict) else {}
        provider_snapshot = provider_reconciliation.get("snapshot")
        provider_snapshot = provider_snapshot if isinstance(provider_snapshot, dict) else {}
        provider_raw = provider_snapshot.get("raw")
        provider_raw = provider_raw if isinstance(provider_raw, dict) else {}
        provider_side = str(provider_raw.get("side") or provider_snapshot.get("side") or "").strip().upper()
        is_recovered_sell_authority = (
            str(row.reason or "").strip().lower() == "recovered from live venue authority"
            and provider_side == "SELL"
        )
        wallet_entry_trade = wallet_buy_trade_by_order_id.get(str(row.id or ""))
        wallet_entry_trade_size = _extract_wallet_trade_size(wallet_entry_trade) if isinstance(wallet_entry_trade, dict) else 0.0
        wallet_entry_trade_price = (
            _extract_wallet_trade_price(wallet_entry_trade) if isinstance(wallet_entry_trade, dict) else None
        )
        wallet_entry_trade_timestamp = (
            _parse_wallet_trade_time(
                wallet_entry_trade.get("timestamp")
                or wallet_entry_trade.get("created_at")
                or wallet_entry_trade.get("createdAt")
                or wallet_entry_trade.get("time")
            )
            if isinstance(wallet_entry_trade, dict)
            else None
        )
        aggregate_wallet_backfill_allowed = str(row.id or "") in aggregate_wallet_backfill_order_ids
        if _active_row_has_unfilled_terminal_provider_failure(
            row,
            payload,
        ) and not _failed_terminal_row_can_reopen_from_wallet_position(row, payload):
            terminal_status = _provider_failure_terminal_status(row, payload)
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "next_status": terminal_status,
                    "note": "finalized_unfilled_provider_failure",
                }
            )
            if not dry_run:
                row.status = terminal_status
                row.notional_usd = 0.0
                row.actual_profit = None
                row.executed_at = None
                row.updated_at = now
                payload["provider_failure_finalized"] = {
                    "finalized_at": _iso_utc(now),
                    "provider_status": _provider_snapshot_status(payload),
                    "reason": "unfilled_provider_failure",
                }
                row.payload_json = payload
                hot_state.record_order_cancelled(
                    trader_id=trader_id,
                    mode=str(row.mode or ""),
                    order_id=str(row.id or ""),
                    market_id=str(row.market_id or ""),
                    source=str(row.source or ""),
                    copy_source_wallet=None,
                )
                state_updates += 1
            skipped += 1
            skipped_reasons["unfilled_provider_failure"] = int(skipped_reasons.get("unfilled_provider_failure", 0)) + 1
            continue
        entry_fill_notional, entry_fill_size, entry_fill_price = _extract_live_fill_metrics(payload)
        if entry_fill_price is None or entry_fill_price <= 0.0:
            entry_fill_price = safe_float(row.effective_price)
        if entry_fill_price is None or entry_fill_price <= 0.0:
            entry_fill_price = safe_float(row.entry_price)
        # Backfill entry price from venue snapshot for orders that were
        # persisted without fill price data (e.g. IOC/FAK orders whose fill
        # price was not in the placement response).
        if entry_fill_price is None or entry_fill_price <= 0.0:
            _bf_clob = str(payload.get("provider_clob_order_id") or "").strip()
            _bf_snap = entry_backfill_snapshots.get(_bf_clob) if _bf_clob else None
            if isinstance(_bf_snap, dict):
                _bf_avg = safe_float(_bf_snap.get("average_fill_price"))
                _bf_filled = max(0.0, safe_float(_bf_snap.get("filled_size"), 0.0) or 0.0)
                _bf_notional = max(0.0, safe_float(_bf_snap.get("filled_notional_usd"), 0.0) or 0.0)
                if _bf_avg is not None and _bf_avg > 0:
                    entry_fill_price = _bf_avg
                elif _bf_notional > 0 and _bf_filled > 0:
                    entry_fill_price = _bf_notional / _bf_filled
                else:
                    _bf_limit = safe_float(_bf_snap.get("limit_price"))
                    if _bf_limit is not None and _bf_limit > 0:
                        entry_fill_price = _bf_limit
                if entry_fill_price is not None and entry_fill_price > 0:
                    if _bf_filled > 0:
                        entry_fill_size = max(entry_fill_size, _bf_filled)
                    if _bf_notional > 0:
                        entry_fill_notional = max(entry_fill_notional, _bf_notional)
                    # Persist backfilled price so future cycles don't need to
                    # re-fetch from the venue.
                    if not dry_run:
                        row.effective_price = float(entry_fill_price)
                        if row.entry_price is None or float(row.entry_price or 0) <= 0:
                            row.entry_price = float(entry_fill_price)
                        recon = payload.get("provider_reconciliation")
                        if isinstance(recon, dict) and not safe_float(recon.get("average_fill_price")):
                            recon["average_fill_price"] = float(entry_fill_price)
                            if _bf_filled > 0:
                                recon["filled_size"] = _bf_filled
                            if _bf_notional > 0:
                                recon["filled_notional_usd"] = _bf_notional
                            payload["provider_reconciliation"] = recon
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                        logger.info(
                            "Backfilled entry price for order=%s price=%.4f source=venue_snapshot",
                            row.id, entry_fill_price,
                        )
        provider_fill_missing = (
            max(0.0, safe_float(provider_reconciliation.get("filled_size"), 0.0) or 0.0) <= _WALLET_SIZE_EPSILON
            and max(0.0, safe_float(provider_reconciliation.get("filled_notional_usd"), 0.0) or 0.0) <= 0.0
        )
        if wallet_entry_trade_size > _WALLET_SIZE_EPSILON and provider_fill_missing and (
            entry_fill_size <= _WALLET_SIZE_EPSILON or entry_fill_notional <= 0.0
        ):
            entry_fill_size = wallet_entry_trade_size
            if wallet_entry_trade_price is not None and wallet_entry_trade_price > 0.0:
                entry_fill_price = wallet_entry_trade_price
                entry_fill_notional = wallet_entry_trade_size * wallet_entry_trade_price
            elif entry_fill_price is not None and entry_fill_price > 0.0:
                entry_fill_notional = wallet_entry_trade_size * entry_fill_price
            if not dry_run:
                backfilled_from_wallet_trade = False
                if wallet_entry_trade_price is not None and wallet_entry_trade_price > 0.0:
                    if row.entry_price is None or abs(float(row.entry_price or 0.0) - float(wallet_entry_trade_price)) > 1e-9:
                        row.entry_price = float(wallet_entry_trade_price)
                        backfilled_from_wallet_trade = True
                    if (
                        row.effective_price is None
                        or abs(float(row.effective_price or 0.0) - float(wallet_entry_trade_price)) > 1e-9
                    ):
                        row.effective_price = float(wallet_entry_trade_price)
                        backfilled_from_wallet_trade = True
                if entry_fill_notional > 0.0 and abs((safe_float(row.notional_usd) or 0.0) - entry_fill_notional) > 1e-9:
                    row.notional_usd = float(entry_fill_notional)
                    backfilled_from_wallet_trade = True
                recovered_entry = {
                    "source": "wallet_trade_history",
                    "size": float(wallet_entry_trade_size),
                    "price": float(wallet_entry_trade_price) if wallet_entry_trade_price is not None else None,
                    "timestamp": (
                        _iso_utc(wallet_entry_trade_timestamp)
                        if isinstance(wallet_entry_trade_timestamp, datetime)
                        else None
                    ),
                    "transaction_hash": str(
                        wallet_entry_trade.get("transactionHash")
                        or wallet_entry_trade.get("txHash")
                        or wallet_entry_trade.get("tx_hash")
                        or ""
                    ).strip()
                    or None,
                }
                if payload.get("entry_fill_recovery") != recovered_entry:
                    payload["entry_fill_recovery"] = recovered_entry
                    backfilled_from_wallet_trade = True
                if backfilled_from_wallet_trade:
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
        if (
            wallet_position_size > _WALLET_SIZE_EPSILON
            and aggregate_wallet_backfill_allowed
            and wallet_entry_trade_size <= _WALLET_SIZE_EPSILON
        ):
            wallet_entry_price = _extract_wallet_entry_price(wallet_position)
            if entry_fill_size <= 0.0:
                entry_fill_size = wallet_position_size
            if (entry_fill_price is None or entry_fill_price <= 0.0) and wallet_entry_price is not None:
                entry_fill_price = wallet_entry_price
            if entry_fill_notional <= 0.0 and entry_fill_price is not None and entry_fill_price > 0.0 and entry_fill_size > 0.0:
                entry_fill_notional = entry_fill_size * entry_fill_price
            if not dry_run and wallet_entry_price is not None and wallet_entry_price > 0.0:
                backfilled_from_wallet = False
                if row.entry_price is None or float(row.entry_price or 0.0) <= 0.0:
                    row.entry_price = float(wallet_entry_price)
                    backfilled_from_wallet = True
                if row.effective_price is None or float(row.effective_price or 0.0) <= 0.0:
                    row.effective_price = float(wallet_entry_price)
                    backfilled_from_wallet = True
                if (safe_float(row.notional_usd) or 0.0) <= 0.0 and entry_fill_notional > 0.0:
                    row.notional_usd = float(entry_fill_notional)
                    backfilled_from_wallet = True
                if backfilled_from_wallet:
                    row.updated_at = now
                    state_updates += 1
        if entry_fill_notional <= 0.0:
            entry_fill_notional = max(0.0, safe_float(row.notional_usd) or 0.0)
        if entry_fill_size <= 0.0 and entry_fill_notional > 0.0 and entry_fill_price and entry_fill_price > 0.0:
            entry_fill_size = entry_fill_notional / entry_fill_price
        if entry_fill_notional > 0.0 and abs((safe_float(row.notional_usd) or 0.0) - entry_fill_notional) > 1e-9:
            if not dry_run:
                row.notional_usd = float(entry_fill_notional)
                row.updated_at = now
                state_updates += 1
        entry_anchor = wallet_entry_trade_timestamp or _row_entry_anchor_naive(row)
        wallet_activity_close_underfilled_entry = False
        terminal_reopen_blocked_wallet_close = False
        if closed_position_reopen_blocked:
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "close_trigger": "wallet_summary_recovery",
                    "next_status": row.status,
                    "note": "ignored_rejected_terminal_close_closed_position",
                }
            )
        if latest_wallet_sell_trade_reopen_blocked:
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "close_trigger": "external_wallet_flatten",
                    "next_status": row.status,
                    "note": "ignored_rejected_terminal_close_wallet_trade",
                }
            )
        if token_id and wallet_position_size <= _WALLET_SIZE_EPSILON and isinstance(wallet_close_activity, dict):
            activity_timestamp = _parse_wallet_activity_time(wallet_close_activity)
            activity_after_entry = True
            if isinstance(activity_timestamp, datetime) and isinstance(entry_anchor, datetime):
                activity_after_entry = activity_timestamp >= entry_anchor
            if activity_after_entry:
                activity_type = _extract_wallet_activity_type(wallet_close_activity)
                aggregate_realized_pnl = (
                    safe_float(closed_position.get("realizedPnl"), None)
                    if isinstance(closed_position, dict)
                    else None
                )
                close_price = _extract_wallet_activity_price(wallet_close_activity)
                if close_price is None and wallet_settlement_price is not None:
                    close_price = wallet_settlement_price
                if close_price is None and entry_fill_price is not None and entry_fill_price > 0.0:
                    close_price = entry_fill_price
                wallet_activity_total_size = _extract_wallet_activity_size(wallet_close_activity)
                wallet_activity_size = wallet_activity_total_size
                wallet_activity_evidence_key = _wallet_close_evidence_key(
                    "wallet_activity",
                    token_id,
                    wallet_close_activity,
                )
                if wallet_activity_evidence_key and wallet_activity_size > _WALLET_SIZE_EPSILON:
                    wallet_activity_size = max(
                        0.0,
                        wallet_activity_size
                        - wallet_terminal_close_consumed_size_by_key.get(wallet_activity_evidence_key, 0.0),
                    )
                allocated_wallet_activity_size = (
                    min(entry_fill_size, wallet_activity_size)
                    if entry_fill_size > _WALLET_SIZE_EPSILON and wallet_activity_size > _WALLET_SIZE_EPSILON
                    else wallet_activity_size
                )
                wallet_activity_position_close = {
                    "price_source": "wallet_activity",
                    "close_trigger": "wallet_activity",
                    "filled_size": allocated_wallet_activity_size,
                    "wallet_activity_size": allocated_wallet_activity_size,
                    "wallet_activity_total_size": wallet_activity_total_size,
                }
                wallet_activity_close_underfilled_entry = _position_close_underfills_entry(
                    row,
                    payload,
                    wallet_activity_position_close,
                )
                if _terminal_reopen_blocks_wallet_close(
                    payload,
                    source="wallet_activity",
                    evidence_key=wallet_activity_evidence_key,
                    evidence_timestamp=activity_timestamp,
                ):
                    terminal_reopen_blocked_wallet_close = True
                    wallet_activity_close_underfilled_entry = True
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": "wallet_activity",
                            "wallet_activity_type": activity_type,
                            "next_status": row.status,
                            "note": "ignored_rejected_terminal_close_wallet_activity",
                        }
                    )
                elif wallet_activity_close_underfilled_entry:
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": "wallet_activity",
                            "wallet_activity_type": activity_type,
                            "filled_size": _position_close_filled_size(wallet_activity_position_close),
                            "required_entry_size": _entry_size_for_terminal_close(row, payload),
                            "next_status": row.status,
                            "note": "ignored_wallet_activity_underfilled_entry",
                        }
                    )
                elif is_recovered_sell_authority:
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": "wallet_activity_auxiliary_exit",
                            "next_status": "resolved",
                        }
                    )
                    if not dry_run:
                        row.status = "resolved"
                        row.updated_at = now
                        payload["authority_recovery_state"] = {
                            "kind": "auxiliary_exit",
                            "source": "wallet_activity_api",
                            "resolved_at": _iso_utc(now),
                            "activity_type": activity_type,
                            "transaction_hash": _extract_wallet_activity_tx_hash(wallet_close_activity) or None,
                        }
                        row.payload_json = payload
                        _apply_position_close_verification(
                            session,
                            row=row,
                            now=now,
                            event_type="wallet_activity_auxiliary_exit",
                            payload_json=payload,
                        )
                        state_updates += 1
                    held += 1
                    continue
                realized_pnl = None
                if (
                    not wallet_activity_close_underfilled_entry
                    and close_price is not None
                    and allocated_wallet_activity_size > _WALLET_SIZE_EPSILON
                ):
                    allocated_cost_basis = entry_fill_notional
                    if entry_fill_size > _WALLET_SIZE_EPSILON:
                        allocated_cost_basis = entry_fill_notional * min(
                            1.0,
                            allocated_wallet_activity_size / entry_fill_size,
                        )
                    realized_pnl = (allocated_wallet_activity_size * close_price) - allocated_cost_basis
                elif (
                    not wallet_activity_close_underfilled_entry
                    and aggregate_realized_pnl is not None
                    and allocated_wallet_activity_size > _WALLET_SIZE_EPSILON
                    and wallet_activity_total_size > _WALLET_SIZE_EPSILON
                ):
                    realized_pnl = aggregate_realized_pnl * min(
                        1.0,
                        allocated_wallet_activity_size / wallet_activity_total_size,
                    )
                if not wallet_activity_close_underfilled_entry and realized_pnl is not None:
                    close_trigger = "wallet_activity"
                    next_status = _status_for_close(pnl=realized_pnl, close_trigger=close_trigger)
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "close_price": close_price,
                            "realized_pnl": realized_pnl,
                            "next_status": next_status,
                            "wallet_activity_type": activity_type,
                            "wallet_activity_size": allocated_wallet_activity_size,
                        }
                    )
                    total_realized_pnl += realized_pnl
                    by_status[next_status] = int(by_status.get(next_status, 0)) + 1
                    would_close += 1
                    if not dry_run:
                        row.status = next_status
                        row.actual_profit = realized_pnl
                        if entry_fill_notional > 0.0:
                            row.notional_usd = float(entry_fill_notional)
                            await session.execute(
                                update(TraderOrder)
                                .where(TraderOrder.id == row.id)
                                .values(notional_usd=float(entry_fill_notional))
                            )
                        row.updated_at = now
                        pending_exit = payload.get("pending_live_exit")
                        pending_exit = pending_exit if isinstance(pending_exit, dict) else {}
                        if pending_exit:
                            pending_exit["status"] = "superseded_wallet_activity"
                            pending_exit["resolved_at"] = _iso_utc(now)
                            payload["pending_live_exit"] = pending_exit
                        payload["position_close"] = {
                            "close_price": close_price,
                            "price_source": "wallet_activity",
                            "close_trigger": close_trigger,
                            "realized_pnl": realized_pnl,
                            "filled_size": allocated_wallet_activity_size,
                            "closed_at": _iso_utc(now),
                            "reason": reason,
                            "wallet_activity_type": activity_type,
                            "wallet_activity_timestamp": (
                                _iso_utc(activity_timestamp)
                                if isinstance(activity_timestamp, datetime)
                                else None
                            ),
                            "wallet_activity_transaction_hash": _extract_wallet_activity_tx_hash(wallet_close_activity) or None,
                            "wallet_activity_id": str(wallet_close_activity.get("id") or "").strip() or None,
                            "wallet_activity_size": allocated_wallet_activity_size,
                            "wallet_activity_total_size": wallet_activity_total_size,
                            "wallet_trade_id": (
                                str(wallet_close_activity.get("tradeID") or wallet_close_activity.get("trade_id") or "").strip()
                                or None
                            ),
                        }
                        if wallet_activity_evidence_key and wallet_activity_size > _WALLET_SIZE_EPSILON:
                            wallet_terminal_close_consumed_size_by_key[wallet_activity_evidence_key] = (
                                    wallet_terminal_close_consumed_size_by_key.get(wallet_activity_evidence_key, 0.0)
                                + allocated_wallet_activity_size
                            )
                        row.payload_json = payload
                        _apply_position_close_verification(
                            session,
                            row=row,
                            now=now,
                            event_type="wallet_activity_close",
                            payload_json=payload,
                        )
                        hot_state.record_order_resolved(
                            trader_id=trader_id,
                            mode=str(row.mode or ""),
                            order_id=str(row.id or ""),
                            market_id=str(row.market_id or ""),
                            direction=str(row.direction or ""),
                            source=str(row.source or ""),
                            status=next_status,
                            actual_profit=realized_pnl,
                            payload=payload,
                            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                        )
                        closed += 1
                    continue
        if (
            token_id
            and wallet_position_size <= _WALLET_SIZE_EPSILON
            and wallet_closed_positions_loaded
            and isinstance(closed_position, dict)
            and not wallet_activity_close_underfilled_entry
            and not terminal_reopen_blocked_wallet_close
            and not closed_position_reopen_blocked
        ):
            # ARCHITECTURE — Polymarket-truth requirement:
            #
            # We are here because the wallet shows zero balance for this
            # token, the position is in Polymarket's closed_positions API,
            # but we do NOT have a matching on-chain trade record (a
            # `wallet_close_activity`) — that path was checked above. The
            # earlier implementation tried to back into a close_price by
            # falling through `currentPrice` / `curPrice` (live market
            # prices, NOT realized) and then attributed wallet-aggregate
            # realizedPnl to this single order via token-id matching. That
            # produced phantom P&L (the +$82 phantom on Flash Crash that
            # didn't match the user's actual ~-$100 Polymarket account).
            #
            # The only legitimate sources of realized P&L are:
            #   1. A confirmed on-chain trade record (wallet_close_activity
            #      path above — already handled with the real fill price)
            #   2. A market-resolution payout ($1 winning / $0 losing)
            #
            # If we got here, neither source has fired yet. We mark the
            # position closed (so it doesn't sit "open" forever) but
            # record actual_profit=NULL and verification_status=summary_only
            # so the aggregation queries (already filtered) exclude it
            # from displayed P&L. The continuous trade verifier will fill
            # in the real number on the next sweep when wallet_trades data
            # becomes available.
            close_price = None
            realized_pnl = None
            closed_position_total_size = max(
                0.0,
                safe_float(
                    closed_position.get("totalBought"),
                    safe_float(
                        closed_position.get("size"),
                        safe_float(closed_position.get("amount"), 0.0),
                    ),
                )
                or 0.0,
            )
            close_trigger = "wallet_summary_recovery"
            next_status = "resolved" if is_recovered_sell_authority else _status_for_close(pnl=0.0, close_trigger=close_trigger)
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "close_trigger": close_trigger,
                    "close_price": close_price,
                    "realized_pnl": realized_pnl,
                    "next_status": next_status,
                    "summary_only": True,
                    "wallet_closed_position_total_size": closed_position_total_size,
                }
            )
            if realized_pnl is not None:
                total_realized_pnl += realized_pnl
            by_status[next_status] = int(by_status.get(next_status, 0)) + 1
            would_close += 1
            if not dry_run:
                row.status = next_status
                if realized_pnl is not None:
                    row.actual_profit = realized_pnl
                if entry_fill_notional > 0.0:
                    row.notional_usd = float(entry_fill_notional)
                    await session.execute(
                        update(TraderOrder)
                        .where(TraderOrder.id == row.id)
                        .values(notional_usd=float(entry_fill_notional))
                    )
                row.updated_at = now
                pending_exit = payload.get("pending_live_exit")
                pending_exit = pending_exit if isinstance(pending_exit, dict) else {}
                if pending_exit:
                    pending_exit["status"] = "superseded_closed_position"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                payload["position_close"] = {
                    "close_price": close_price,
                    "price_source": "closed_positions_api",
                    "close_trigger": close_trigger,
                    "realized_pnl": realized_pnl,
                    "filled_size": entry_fill_size if entry_fill_size > 0.0 else safe_float(closed_position.get("totalBought"), None),
                    "wallet_closed_position_total_size": closed_position_total_size,
                    "closed_at": _iso_utc(now),
                    "reason": reason,
                    "wallet_closed_position_timestamp": safe_float(closed_position.get("timestamp"), None),
                }
                row.payload_json = payload
                _apply_position_close_verification(
                    session,
                    row=row,
                    now=now,
                    event_type="summary_close_recovery",
                    payload_json=payload,
                )
                hot_state.record_order_resolved(
                    trader_id=trader_id,
                    mode=str(row.mode or ""),
                    order_id=str(row.id or ""),
                    market_id=str(row.market_id or ""),
                    direction=str(row.direction or ""),
                    source=str(row.source or ""),
                    status=next_status,
                    actual_profit=realized_pnl,
                    payload=payload,
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                )
                closed += 1
            continue
        pending_market_info = market_info_by_id.get(str(row.market_id or ""))
        pending_outcome_idx = _direction_outcome_index(
            row.direction,
            market_info=pending_market_info,
            token_id=_extract_leg_token_id(row.payload_json),
        )
        pending_market_tradable = polymarket_client.is_market_tradable(pending_market_info, now=now)
        pending_winning_idx = _extract_winning_outcome_index(pending_market_info)

        # ── Permanent-block short-circuit ──────────────────────────────
        # Order 350ebe8b on 2026-04-27 spent 18.4s of every reconcile
        # cycle re-attempting a SELL into a market the CLOB no longer
        # accepts (market_tradable=false), eating the 5s execute_live_order
        # timeout twice (allowance prep + submit) plus retry-state
        # bookkeeping.  Each attempt failed with TimeoutError, the
        # blocked-streak counter ratcheted to N, status was set to
        # blocked_persistent_timeout — but the trigger evaluation
        # at the END of the loop kept re-firing "Max hold exceeded"
        # and creating a fresh pending_live_exit on every cycle,
        # effectively bypassing the persistent-timeout guard.
        #
        # Short-circuit here: if the CURRENT pending_exit is in any
        # terminal-blocked state AND we have market info AND the
        # market is no longer tradable, just hold the row and skip
        # the rest of the per-candidate work.  An operator (or a
        # market resolution → the resolution branches at line 8005
        # and 7635 will pick it up on a future cycle when wallet/
        # market data confirms settlement) can clear this state.
        _pending_exit_now = payload.get("pending_live_exit")
        # Auto-recover transient blocked_persistent_timeout rows whose
        # cooldown has elapsed.  Three timeouts during an event-loop
        # stall should not pin a position forever — the hard retry
        # ceiling still bounds true-permanent failures.
        if isinstance(_pending_exit_now, dict) and not dry_run:
            if _maybe_clear_persistent_timeout(_pending_exit_now, now):
                payload["pending_live_exit"] = _pending_exit_now
                row.payload_json = payload
                row.updated_at = now
        if (
            isinstance(_pending_exit_now, dict)
            and str(_pending_exit_now.get("status") or "").strip().lower() in {
                "blocked_persistent_timeout",
                "blocked_no_inventory",
                "blocked_retry_exhausted",
                "blocked_retry_exhausted_hard",
                "blocked_orderbook_gone",
            }
            and pending_market_info is not None
            and not pending_market_tradable
            # No pending resolution evidence — if there is, the
            # downstream resolution branches will close the row.
            and pending_winning_idx is None
            and wallet_settlement_price is None
            # Allow through if a post-end extreme mark can close the position.
            and _infer_post_end_terminal_price(
                market_info=pending_market_info,
                current_price=None,
                current_price_source=None,
                previous_mark_price=safe_float(_extract_position_state(payload).get("last_mark_price")),
                wallet_mark_price=None,
                now=now,
            )[0] is None
        ):
            held += 1
            skipped_reasons["blocked_terminal_untradable"] = (
                int(skipped_reasons.get("blocked_terminal_untradable", 0)) + 1
            )
            continue

        pending_winning_idx_inferred = False
        if pending_winning_idx is None and resolution_infer_from_prices:
            inferred_pending_idx = _extract_winning_outcome_index_from_prices(
                pending_market_info,
                market_tradable=pending_market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_pending_idx is not None:
                pending_winning_idx = inferred_pending_idx
                pending_winning_idx_inferred = True

        pending_signal_payload = signal_payloads.get(str(row.signal_id), {})
        pending_market_side_price = (
            _extract_market_side_price(pending_market_info, pending_outcome_idx)
            if pending_outcome_idx is not None
            else None
        )
        pending_ws_side_price = ws_mid_prices.get(token_id) if token_id else None
        pending_clob_side_price = clob_mid_prices.get(token_id) if token_id else None
        pending_current_price = (
            pending_ws_side_price
            if pending_ws_side_price is not None
            else (
                pending_clob_side_price
                if pending_clob_side_price is not None
                else (pending_market_side_price if pending_market_side_price is not None else wallet_mark_price)
            )
        )
        pending_current_price = _state_price_floor(pending_current_price)
        pending_current_price_source = (
            "ws_mid"
            if pending_ws_side_price is not None
            else (
                "clob_midpoint"
                if pending_clob_side_price is not None
                else (
                    "market_mark"
                    if pending_market_side_price is not None
                    else ("wallet_mark" if wallet_mark_price is not None else None)
                )
            )
        )
        pending_position_state = _extract_position_state(payload)
        pending_prev_high = safe_float(pending_position_state.get("highest_price"))
        pending_prev_low = safe_float(pending_position_state.get("lowest_price"))
        pending_prev_last_mark = safe_float(pending_position_state.get("last_mark_price"))
        pending_prev_mark_source = str(pending_position_state.get("last_mark_source") or "")
        pending_prev_marked_at = _parse_iso_utc_naive(pending_position_state.get("last_marked_at"))
        pending_highest_price = pending_prev_high
        pending_lowest_price = pending_prev_low
        if pending_current_price is not None:
            if pending_highest_price is None:
                pending_highest_price = pending_current_price
            else:
                pending_highest_price = max(pending_highest_price, pending_current_price)
            if pending_lowest_price is None:
                pending_lowest_price = pending_current_price
            else:
                pending_lowest_price = min(pending_lowest_price, pending_current_price)
        pending_mark_updated_at_value = (
            _iso_utc(now)
            if pending_current_price is not None
            else (
                _iso_utc(pending_prev_marked_at.replace(tzinfo=timezone.utc))
                if pending_prev_marked_at is not None
                else ""
            )
        )
        pending_next_state = {
            "highest_price": _state_price_floor(pending_highest_price),
            "lowest_price": _state_price_floor(pending_lowest_price),
            "last_mark_price": (
                pending_current_price
                if pending_current_price is not None
                else _state_price_floor(pending_prev_last_mark)
            ),
            "last_mark_source": str(pending_current_price_source or pending_prev_mark_source),
            "last_marked_at": pending_mark_updated_at_value,
            "last_exit_evaluated_at": _iso_utc(now),
        }
        pending_state_changed = False
        if pending_current_price is not None:
            pending_mark_stale = (
                pending_prev_marked_at is None
                or (now_naive - pending_prev_marked_at).total_seconds() >= mark_touch_interval_seconds
            )
            pending_state_changed = (
                pending_prev_last_mark is None
                or abs(pending_prev_last_mark - pending_current_price) > 1e-9
                or pending_prev_high is None
                or pending_prev_low is None
                or abs((pending_prev_high or 0.0) - (pending_highest_price or 0.0)) > 1e-9
                or abs((pending_prev_low or 0.0) - (pending_lowest_price or 0.0)) > 1e-9
                or pending_prev_mark_source != str(pending_current_price_source or "")
                or pending_mark_stale
            )

        # External/manual flatten convergence:
        # if wallet inventory is flat and we have concrete wallet SELL evidence
        # for this token after entry, treat the position as externally closed.
        if (
            token_id
            and entry_fill_size > 0.0
            and wallet_position_size <= _WALLET_SIZE_EPSILON
            and pending_winning_idx is None
            and wallet_settlement_price is None
            and isinstance(latest_wallet_sell_trade, dict)
            and not latest_wallet_sell_trade_reopen_blocked
        ):
            latest_wallet_sell_trade_has_identity = bool(
                str(
                    latest_wallet_sell_trade.get("trade_id")
                    or latest_wallet_sell_trade.get("transactionHash")
                    or latest_wallet_sell_trade.get("txHash")
                    or latest_wallet_sell_trade.get("tx_hash")
                    or ""
                ).strip()
            )
            if not latest_wallet_sell_trade_has_identity:
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "external_wallet_flatten",
                        "next_status": row.status,
                        "note": "ignored_external_wallet_flatten_missing_trade_identity",
                    }
                )
                held += 1
                continue
            trade_ts = latest_wallet_sell_trade.get("timestamp")
            trade_after_entry = True
            if isinstance(trade_ts, datetime) and isinstance(entry_anchor, datetime):
                trade_after_entry = trade_ts >= entry_anchor
            if trade_after_entry:
                close_price = safe_float(latest_wallet_sell_trade.get("price"))
                if close_price is None or close_price <= 0.0:
                    close_price = (
                        wallet_mark_price
                        if wallet_mark_price is not None and wallet_mark_price > 0.0
                        else entry_fill_price
                    )
                if close_price is not None and close_price > 0.0:
                    close_qty = entry_fill_size
                    latest_wallet_sell_trade_size = _extract_wallet_trade_size(latest_wallet_sell_trade)
                    latest_wallet_sell_trade_evidence_key = _wallet_close_evidence_key(
                        "wallet_trade",
                        token_id,
                        latest_wallet_sell_trade,
                    )
                    latest_wallet_sell_trade_available_size = latest_wallet_sell_trade_size
                    if latest_wallet_sell_trade_evidence_key and latest_wallet_sell_trade_size > _WALLET_SIZE_EPSILON:
                        latest_wallet_sell_trade_available_size = max(
                            0.0,
                            latest_wallet_sell_trade_size
                            - wallet_terminal_close_consumed_size_by_key.get(
                                latest_wallet_sell_trade_evidence_key,
                                0.0,
                            ),
                        )
                    if latest_wallet_sell_trade_available_size + _WALLET_SIZE_EPSILON < close_qty * 0.995:
                        details.append(
                            {
                                "order_id": row.id,
                                "market_id": row.market_id,
                                "close_trigger": "external_wallet_flatten",
                                "wallet_trade_size": latest_wallet_sell_trade_available_size,
                                "required_entry_size": close_qty,
                                "next_status": row.status,
                                "note": "ignored_external_wallet_flatten_underfilled_entry",
                            }
                        )
                        if wallet_position_size > _WALLET_SIZE_EPSILON:
                            held += 1
                        elif not wallet_positions_authoritative:
                            # Empty wallet cache could be a Polymarket data-
                            # API blip; don't trust ``size <= 0`` here.  Hold
                            # and let the next reconcile cycle re-evaluate
                            # against a fresh load.  See the
                            # wallet_positions_authoritative comment above
                            # the load.
                            held += 1
                        else:
                            details.append(
                                {
                                    "order_id": row.id,
                                    "market_id": row.market_id,
                                    "next_status": "resolved",
                                    "note": "external_wallet_flatten_underfilled_and_wallet_absent_resolved_neutral",
                                }
                            )
                            if not dry_run:
                                pending_exit_local = payload.get("pending_live_exit")
                                if isinstance(pending_exit_local, dict):
                                    pending_exit_local["status"] = "superseded_wallet_absent"
                                    pending_exit_local["resolved_at"] = _iso_utc(now)
                                    payload["pending_live_exit"] = pending_exit_local
                                payload["wallet_absent_resolution"] = {
                                    "resolved_at": _iso_utc(now),
                                    "reason": "external_wallet_flatten_underfilled_and_wallet_absent",
                                    "wallet_trade_size_available": float(latest_wallet_sell_trade_available_size),
                                    "required_entry_size": float(close_qty),
                                }
                                row.status = "resolved"
                                row.actual_profit = None
                                row.updated_at = now
                                row.payload_json = payload
                                hot_state.record_order_cancelled(
                                    trader_id=trader_id,
                                    mode=str(row.mode or ""),
                                    order_id=str(row.id or ""),
                                    market_id=str(row.market_id or ""),
                                    source=str(row.source or ""),
                                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                                )
                                state_updates += 1
                            skipped += 1
                            skipped_reasons["wallet_absent_without_provider_or_close_evidence"] = (
                                int(skipped_reasons.get("wallet_absent_without_provider_or_close_evidence", 0)) + 1
                            )
                        continue
                    close_notional = close_qty * close_price
                    close_trigger = "external_wallet_flatten"
                    pnl = close_notional - entry_fill_notional
                    next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "close_price": close_price,
                            "realized_pnl": pnl,
                            "next_status": next_status,
                            "wallet_trade_id": latest_wallet_sell_trade.get("trade_id"),
                        }
                    )
                    would_close += 1
                    total_realized_pnl += pnl
                    by_status[next_status] = int(by_status.get(next_status, 0)) + 1
                    if not dry_run:
                        pending_exit_local = payload.get("pending_live_exit")
                        if isinstance(pending_exit_local, dict):
                            pending_exit_local["status"] = "superseded_external"
                            pending_exit_local["resolved_at"] = _iso_utc(now)
                            payload["pending_live_exit"] = pending_exit_local
                        row.status = next_status
                        row.actual_profit = pnl
                        if entry_fill_notional > 0.0:
                            row.notional_usd = float(entry_fill_notional)
                            await session.execute(
                                update(TraderOrder)
                                .where(TraderOrder.id == row.id)
                                .values(notional_usd=float(entry_fill_notional))
                            )
                        row.updated_at = now
                        payload["position_close"] = {
                            "close_price": close_price,
                            "price_source": "wallet_trade",
                            "close_trigger": close_trigger,
                            "realized_pnl": pnl,
                            "filled_size": close_qty,
                            "closed_at": _iso_utc(now),
                            "reason": reason,
                            "wallet_trade_id": str(latest_wallet_sell_trade.get("trade_id") or ""),
                            "wallet_trade_size": close_qty,
                            "wallet_trade_total_size": latest_wallet_sell_trade_size,
                            "wallet_trade_timestamp": (
                                _iso_utc(latest_wallet_sell_trade.get("timestamp"))
                                if isinstance(latest_wallet_sell_trade.get("timestamp"), datetime)
                                else None
                            ),
                            "wallet_activity_transaction_hash": (
                                str(
                                    latest_wallet_sell_trade.get("transactionHash")
                                    or latest_wallet_sell_trade.get("txHash")
                                    or latest_wallet_sell_trade.get("tx_hash")
                                    or ""
                                ).strip()
                                or None
                            ),
                        }
                        if latest_wallet_sell_trade_evidence_key and latest_wallet_sell_trade_available_size > _WALLET_SIZE_EPSILON:
                            wallet_terminal_close_consumed_size_by_key[latest_wallet_sell_trade_evidence_key] = (
                                wallet_terminal_close_consumed_size_by_key.get(
                                    latest_wallet_sell_trade_evidence_key,
                                    0.0,
                                )
                                + min(latest_wallet_sell_trade_available_size, close_qty)
                            )
                        reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
                            session,
                            row=row,
                            payload=payload,
                            signal_payload=pending_signal_payload,
                            market_info=pending_market_info,
                            close_trigger=close_trigger,
                            realized_pnl=pnl,
                            now=now,
                        )
                        if reverse_signal_id and reverse_source:
                            reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)
                        row.payload_json = payload
                        _apply_position_close_verification(
                            session,
                            row=row,
                            now=now,
                            event_type="wallet_trade_close",
                            payload_json=payload,
                        )
                        hot_state.record_order_resolved(
                            trader_id=trader_id,
                            mode=str(row.mode or ""),
                            order_id=str(row.id or ""),
                            market_id=str(row.market_id or ""),
                            direction=str(row.direction or ""),
                            source=str(row.source or ""),
                            status=next_status,
                            actual_profit=pnl,
                            payload=payload,
                            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                        )
                        closed += 1
                    continue

        # ── Wallet-absent close: position vanished from on-chain wallet ──
        # If wallet data loaded successfully but this token is completely
        # absent (size=0 / redeemed), and there is no sell trade (i.e. auto-
        # redemption or external close), close the order.  Use the best
        # available price: market mark, wallet mark, or entry price.
        # Skip when there is already an active exit submitted to the provider
        # (identified by a CLOB order ID) — the pending-exit reconciliation
        # path handles that case and knows the real fill price.
        _pending_exit_tmp = payload.get("pending_live_exit")
        _fallback_manual_pending_exit = (
            isinstance(_pending_exit_tmp, dict)
            and _pending_exit_is_live_fallback_manual_exit(_pending_exit_tmp)
        )
        _has_active_provider_exit = isinstance(_pending_exit_tmp, dict) and bool(
            _pending_exit_provider_clob_id(_pending_exit_tmp)
        ) and not _fallback_manual_pending_exit
        _has_working_provider_entry = _provider_entry_order_still_working(payload)
        _has_exit_management = isinstance(_pending_exit_tmp, dict) or bool(payload.get("strategy_exit_config"))
        # A missing wallet row is not treated as close authority here.
        # Live positions remain open until explicit exit evidence exists.
        # Guard: do not wallet-absent-close orders placed in the last 120
        # seconds.  Polymarket's data API can lag behind the CLOB — a just-
        # placed buy may not appear in get_wallet_positions() yet, causing a
        # false wallet_absent_close that records a phantom loss and triggers
        # an immediate re-buy of the same market.
        if (
            token_id
            and entry_fill_size > 0.0
            and wallet_positions_loaded
            # ``wallet_positions_authoritative`` requires the load to have
            # returned at least one row.  An empty wallet snapshot is the
            # exact shape produced by a transient Polymarket data-API blip
            # (HTTP 200 with empty ``data`` array) — and that's the failure
            # mode that produced the 2026-04-28 incident.  A wallet that
            # GENUINELY holds zero positions can be reconciled by other
            # paths (resolution detection, settlement); the wallet-absent
            # close path's purpose is recovering positions whose token
            # disappeared from a wallet that still holds others.  See
            # ``services/live_execution_service._persist_positions`` for
            # the corresponding write-side guard against wiping the cache
            # on an empty result.
            and wallet_positions_authoritative
            and not wallet_position_observed
            and pending_winning_idx is None
            and wallet_settlement_price is None
            and (not _has_exit_management or _pending_exit_stale_without_provider(_pending_exit_tmp, now_naive))
            and not _has_active_provider_exit
            and not _has_working_provider_entry
        ):
            if _pending_exit_stale_without_provider(_pending_exit_tmp, now_naive):
                if not dry_run:
                    stale_pending_exit = dict(_pending_exit_tmp) if isinstance(_pending_exit_tmp, dict) else {}
                    stale_pending_exit["status"] = "superseded_wallet_absent_no_provider"
                    stale_pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = stale_pending_exit
                    payload["wallet_absent_resolution"] = {
                        "resolved_at": _iso_utc(now),
                        "reason": "stale_pending_exit_no_provider_and_wallet_absent",
                        "retry_count": int(safe_float(stale_pending_exit.get("retry_count"), 0.0) or 0),
                        "last_error": stale_pending_exit.get("last_error"),
                    }
                    row.status = "resolved"
                    row.actual_profit = None
                    row.updated_at = now
                    row.payload_json = payload
                    hot_state.record_order_cancelled(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        source=str(row.source or ""),
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    state_updates += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "next_status": "resolved",
                        "note": "resolved_stale_pending_exit_wallet_absent",
                    }
                )
                skipped += 1
                skipped_reasons["stale_pending_exit_wallet_absent"] = (
                    int(skipped_reasons.get("stale_pending_exit_wallet_absent", 0)) + 1
                )
                continue
            details.append(
                {
                    "order_id": row.id,
                    "market_id": row.market_id,
                    "next_status": str(row.status or "").strip().lower(),
                    "note": "wallet_absent_without_verified_close_evidence_held",
                }
            )
            held += 1
            continue

        pending_exit = payload.get("pending_live_exit")
        pending_exit_status = (
            str(pending_exit.get("status") or "").strip().lower() if isinstance(pending_exit, dict) else ""
        )
        pending_exit_kind = (
            str(pending_exit.get("kind") or "").strip().lower() if isinstance(pending_exit, dict) else ""
        )

        def _attach_pending_state(target_payload: dict[str, Any]) -> None:
            if pending_state_changed:
                target_payload["position_state"] = pending_next_state

        # ── Laddered (multi-child) exit reconcile ─────────────────────────
        # When pending_live_exit carries a ``children`` list, the orchestrator
        # delegates submission/escalation/reprice to ``exit_executor``. The
        # legacy single-order branches below are skipped for this row this
        # pass — they assume one provider order id on the parent.
        _children = (
            pending_exit.get("children")
            if isinstance(pending_exit, dict) and isinstance(pending_exit.get("children"), list)
            else None
        )
        if _children and not force_mark_to_market:
            # 1. Apply provider snapshots to each child
            for _child in _children:
                if not isinstance(_child, dict):
                    continue
                _cid = str(_child.get("provider_clob_order_id") or "").strip()
                _snap = pending_exit_snapshots.get(_cid) if _cid else None
                if _snap is None and _cid:
                    _snap = pending_exit_snapshot_fallbacks.get(_cid)
                if not isinstance(_snap, dict):
                    continue
                _snap_status = _resolved_snapshot_status(
                    _snap, _snap.get("normalized_status"), _child.get("provider_status")
                )
                _snap_filled = max(0.0, safe_float(_snap.get("filled_size"), 0.0) or 0.0)
                _snap_price = safe_float(_snap.get("average_fill_price"))
                if _snap_price is None or _snap_price <= 0:
                    _snap_price = safe_float(_snap.get("limit_price"))
                if _snap_filled > 0:
                    _child["filled_size"] = float(max(_snap_filled, safe_float(_child.get("filled_size"), 0.0) or 0.0))
                if _snap_price is not None and _snap_price > 0:
                    _child["average_fill_price"] = float(_snap_price)
                if _snap_status == "filled" or (
                    _snap_status in {"matched", "executed"}
                    and _child["filled_size"] + 1e-9 >= float(_child.get("size", 0.0) or 0.0)
                ):
                    _child["status"] = exit_executor.CHILD_FILLED
                elif _snap_status in {"cancelled", "expired"}:
                    _child["status"] = exit_executor.CHILD_CANCELLED
                elif _snap_status in {"failed"}:
                    _child["status"] = exit_executor.CHILD_FAILED
                elif _snap_filled > 0:
                    _child["status"] = exit_executor.CHILD_PARTIAL
                _child["provider_status"] = _snap_status

            # 2. Check completion. Terminalize if done.
            if exit_executor.is_exit_complete(pending_exit):
                _agg = exit_executor.aggregate_children_fills(_children)
                _filled_sz = _agg["filled_size"]
                _avg_px = _agg["average_fill_price"]
                pending_exit["filled_size"] = float(_filled_sz)
                pending_exit["average_fill_price"] = float(_avg_px) if _avg_px > 0 else pending_exit.get("average_fill_price")
                pending_exit["children_filled_size"] = float(_filled_sz)
                pending_exit["children_filled_notional"] = float(_agg["filled_notional"])
                pending_exit["status"] = "filled"
                _close_trigger = str(pending_exit.get("close_trigger") or "ladder_exit_complete")
                _required = max(0.0, safe_float(pending_exit.get("target_size"), 0.0) or 0.0)
                _entry_not = safe_float(pending_exit.get("entry_notional_usd")) or (safe_float(row.notional_usd) or 0.0)
                _entry_qty = safe_float(pending_exit.get("entry_filled_size")) or (
                    safe_float(_payload_strategy_exit_config(payload).get("filled_size"))
                )
                if not _entry_qty or _entry_qty <= 0:
                    _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                    _entry_qty = _fill_sz if _fill_sz > 0 else (
                        (_fill_not / _fill_px) if _fill_px and _fill_px > 0 else 0.0
                    )
                    if _entry_not <= 0:
                        _entry_not = _fill_not
                _close_qty = min(_filled_sz, _entry_qty) if _entry_qty > 0 else _filled_sz
                _cost_basis = (
                    _entry_not * (_close_qty / _entry_qty) if _entry_qty > 0 else _entry_not
                )
                _proceeds = _close_qty * _avg_px if _close_qty > 0 and _avg_px > 0 else 0.0
                _pnl = _proceeds - _cost_basis if _cost_basis > 0 else 0.0
                _ns = _status_for_close(pnl=_pnl, close_trigger=_close_trigger)
                if not dry_run:
                    row.status = _ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": _avg_px if _avg_px > 0 else (safe_float(pending_exit.get("close_price")) or 0.0),
                        "price_source": "ladder_aggregate",
                        "close_trigger": _close_trigger,
                        "realized_pnl": _pnl,
                        "cost_basis_usd": _cost_basis,
                        "settlement_proceeds_usd": _proceeds,
                        "filled_size": _close_qty,
                        "filled_notional_usd": _close_qty * _avg_px if _avg_px > 0 else 0.0,
                        "ladder_children": [
                            {
                                "child_id": c.get("child_id"),
                                "price": c.get("price"),
                                "filled_size": c.get("filled_size"),
                                "average_fill_price": c.get("average_fill_price"),
                                "status": c.get("status"),
                            }
                            for c in _children
                            if isinstance(c, dict)
                        ],
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=_ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[_ns] = int(by_status.get(_ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": _close_trigger,
                        "close_price": _avg_px,
                        "realized_pnl": _pnl,
                        "next_status": _ns,
                        "ladder_filled_size": _filled_sz,
                        "ladder_child_count": len(_children),
                    }
                )
                continue

            # 3. Run another executor pass: submit planned, escalate stale,
            #    reprice on drift. Submission counter is shared with legacy
            #    paths so we don't double-spend the per-trader budget.
            _ladder_token_id = (
                _extract_live_token_id(payload)
                or str(pending_exit.get("token_id") or "")
            )
            if not _ladder_token_id:
                # Token id missing; fail-safe and try again next pass.
                pending_exit["last_error"] = "ladder_missing_token_id"
                if not dry_run:
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            _ladder_min_order_size_usd = _resolve_position_min_order_size_usd(
                trader_params=params, payload=payload, mode="live",
            )
            _ladder_remaining_budget = max(
                0, _live_exit_submission_cap() - exit_submissions_this_pass[0]
            )
            if _ladder_remaining_budget <= 0:
                # No budget for this row this pass. The remaining children
                # are still tracked by the snapshot loop and will be picked
                # up on the next reconcile cycle.
                if not dry_run:
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            _ladder_budget = exit_executor.PassBudget(
                submits=_ladder_remaining_budget,
                escalations=_ladder_remaining_budget,
                reprices=_ladder_remaining_budget,
            )

            from services.live_execution_adapter import execute_live_order as _ladder_place_order

            async def _ladder_cancel(_clob_id: str) -> bool:
                async with release_conn(session):
                    try:
                        return bool(await live_execution_service.cancel_order(_clob_id))
                    except Exception:
                        return False

            try:
                async with release_conn(session):
                    await _prepare_sell_allowance_bounded(_ladder_token_id)
                    _ladder_report = await exit_executor.run_exit_pass(
                        pending_exit=pending_exit,
                        token_id=_ladder_token_id,
                        side="SELL",
                        min_order_size_usd=_ladder_min_order_size_usd,
                        place_order=_ladder_place_order,
                        cancel_fn=_ladder_cancel,
                        current_mid=pending_current_price,
                        tick_size=exit_executor.DEFAULT_TICK_SIZE,
                        budget=_ladder_budget,
                        now=now,
                    )
                exit_submissions_this_pass[0] += int(
                    _ladder_report.get("submitted", 0)
                    + _ladder_report.get("escalated", 0)
                    + _ladder_report.get("repriced", 0)
                )
                pending_exit["last_pass_at"] = _iso_utc(now)
                pending_exit["last_pass_report"] = dict(_ladder_report)
                if _ladder_report.get("complete"):
                    # Will terminalize on the next reconcile cycle via the
                    # completion branch above; persist and re-iterate.
                    pending_exit["status"] = pending_exit.get("status") or "submitted"
                else:
                    pending_exit["status"] = "submitted"
            except Exception as _ladder_exc:
                logger.warning(
                    "Laddered exit pass exception for order=%s: %s",
                    row.id,
                    _format_exit_error(_ladder_exc),
                    exc_info=_ladder_exc,
                )
                pending_exit["last_error"] = _format_exit_error(_ladder_exc)

            if not dry_run:
                payload["pending_live_exit"] = pending_exit
                _attach_pending_state(payload)
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        if (
            isinstance(pending_exit, dict)
            and pending_exit_status in {"pending", "submitted", "failed"}
            and _pending_exit_is_live_fallback_manual_exit(pending_exit)
        ):
            provider_clob_id = _pending_exit_provider_clob_id(pending_exit)
            if provider_clob_id and pending_exit_status in {"submitted", "pending"}:
                try:
                    async with release_conn(session):
                        await live_execution_service.cancel_order(provider_clob_id)
                except Exception:
                    pass
            if not dry_run:
                pending_exit["status"] = "superseded_fallback_exit"
                pending_exit["superseded_at"] = _iso_utc(now)
                payload["superseded_pending_exit"] = pending_exit
                payload.pop("pending_live_exit", None)
                _attach_pending_state(payload)
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            pending_exit = None
            pending_exit_status = ""
            pending_exit_kind = ""

        # ── Force mark-to-market: supersede any existing pending exit ──
        # When a manual sell is requested, override whatever exit state
        # exists so the force_mark_to_market path is always reached below.
        if force_mark_to_market and isinstance(pending_exit, dict) and pending_exit_status:
            provider_clob_id = _pending_exit_provider_clob_id(pending_exit)
            if provider_clob_id and pending_exit_status in {"submitted", "pending"}:
                try:
                    async with release_conn(session):
                        await live_execution_service.cancel_order(provider_clob_id)
                except Exception:
                    pass
            pending_exit["status"] = "superseded_manual_sell"
            pending_exit["superseded_at"] = _iso_utc(now)
            payload["superseded_pending_exit"] = pending_exit
            payload.pop("pending_live_exit", None)
            if not dry_run:
                row.payload_json = payload
                row.updated_at = now
            pending_exit = None
            pending_exit_status = ""
            pending_exit_kind = ""

        # ── Submitted/pending exit reconciliation ──────────────────────
        # For live exits we trust provider fill truth; a terminal local close
        # only happens once provider fill size reaches the required threshold.
        if isinstance(pending_exit, dict) and pending_exit_status in {"submitted", "pending"}:
            wallet_resolution_confirmed = wallet_settlement_price is not None
            if pending_winning_idx is not None and pending_outcome_idx is not None and wallet_resolution_confirmed:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = 1.0 if pending_winning_idx == pending_outcome_idx else 0.0
                close_trigger = "resolution_inferred" if pending_winning_idx_inferred else "resolution"
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "resolved_settlement",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            if wallet_settlement_price is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = wallet_settlement_price
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger="resolution")
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "wallet_redeemable_mark",
                        "close_trigger": "resolution",
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "resolution",
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            provider_clob_order_id = _pending_exit_provider_clob_id(pending_exit)
            if provider_clob_order_id:
                pending_exit["provider_clob_order_id"] = provider_clob_order_id
            snapshot = pending_exit_snapshots.get(provider_clob_order_id) if provider_clob_order_id else None
            if snapshot is None and provider_clob_order_id:
                snapshot = pending_exit_snapshot_fallbacks.get(provider_clob_order_id)
            snapshot_status = _resolved_snapshot_status(
                snapshot,
                (snapshot or {}).get("normalized_status"),
                pending_exit.get("provider_status"),
            )
            snapshot_filled_size = max(0.0, safe_float((snapshot or {}).get("filled_size"), 0.0) or 0.0)
            snapshot_fill_price = safe_float((snapshot or {}).get("average_fill_price"))
            if snapshot_fill_price is None or snapshot_fill_price <= 0:
                snapshot_fill_price = safe_float((snapshot or {}).get("limit_price"))
            if snapshot_filled_size <= 0.0:
                snapshot_filled_size = max(0.0, safe_float(pending_exit.get("filled_size"), 0.0) or 0.0)
            if snapshot_fill_price is None or snapshot_fill_price <= 0.0:
                snapshot_fill_price = safe_float(pending_exit.get("average_fill_price"))
            required_exit_size = max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0)
            reopened_partial_exit = str(pending_exit.get("reopen_reason") or "").strip().lower() in {
                "partial_exit_fill_below_threshold",
                "terminal_close_underfilled_entry",
            }
            _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
            _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
            if _ep is None or _ep <= 0:
                _ep = safe_float(row.entry_price)
            _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
            entry_position_size = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
            if entry_position_size > 0.0:
                required_exit_size = max(required_exit_size, entry_position_size)
            if required_exit_size > 0.0:
                pending_exit["exit_size"] = float(required_exit_size)

            threshold_ratio = _pending_exit_fill_threshold(pending_exit)
            fill_ratio = (
                (snapshot_filled_size / required_exit_size)
                if required_exit_size > 0.0
                else (1.0 if snapshot_status == "filled" and snapshot_filled_size > 0.0 else 0.0)
            )
            pending_exit["provider_status"] = snapshot_status or pending_exit.get("provider_status")
            if snapshot_filled_size > 0.0:
                pending_exit["filled_size"] = float(snapshot_filled_size)
            if snapshot_fill_price is not None and snapshot_fill_price > 0.0:
                pending_exit["average_fill_price"] = float(snapshot_fill_price)
            if fill_ratio > 0.0:
                pending_exit["fill_ratio"] = float(fill_ratio)
            if snapshot is not None:
                pending_exit["last_snapshot_at"] = _iso_utc(now)

            terminal_provider_status = snapshot_status in {"filled", "matched", "executed"}
            provider_status_unknown = snapshot_status in {"", "missing", "invalid", "unknown"}
            wallet_flat_confirmed = wallet_position_size <= _WALLET_SIZE_EPSILON and (
                wallet_position_observed
                or isinstance(latest_wallet_sell_trade, dict)
                or isinstance(wallet_close_activity, dict)
                or isinstance(closed_position, dict)
                or (
                    wallet_positions_loaded
                    and bool(wallet_positions_by_token)
                    and not wallet_position_observed
                    and terminal_provider_status
                )
            )
            close_fill_threshold_met = (
                required_exit_size > 0.0
                and terminal_provider_status
                and snapshot_filled_size >= (required_exit_size * threshold_ratio)
                and wallet_flat_confirmed
            )
            close_fill_threshold_with_wallet_confirmation = (
                required_exit_size > 0.0
                and provider_status_unknown
                and snapshot_filled_size >= (required_exit_size * threshold_ratio)
                and wallet_flat_confirmed
            )
            close_fill_terminal_with_wallet_confirmation = (
                required_exit_size > 0.0
                and terminal_provider_status
                and snapshot_filled_size > 0.0
                and wallet_flat_confirmed
                and not reopened_partial_exit
            )
            close_fill_unknown_but_wallet_flat = (
                required_exit_size <= 0.0 and terminal_provider_status and wallet_flat_confirmed
            )
            if (
                close_fill_threshold_met
                or close_fill_threshold_with_wallet_confirmation
                or close_fill_terminal_with_wallet_confirmation
                or close_fill_unknown_but_wallet_flat
            ):
                base_qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                close_qty = snapshot_filled_size if snapshot_filled_size > 0 else required_exit_size
                if close_qty <= 0.0:
                    close_qty = base_qty
                if base_qty <= 0.0 or _not <= 0.0:
                    held += 1
                    if not dry_run:
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    continue
                close_qty = min(base_qty, close_qty if close_qty > 0.0 else base_qty)
                cost_basis = _not * (close_qty / base_qty) if base_qty > 0 else _not
                cp = snapshot_fill_price
                if cp is None or cp <= 0.0:
                    cp = safe_float(pending_exit.get("close_price")) or _ep
                close_trigger = str(pending_exit.get("close_trigger") or "live_exit_fill")
                _pnl = (close_qty * cp) - cost_basis
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "filled"
                    if close_fill_terminal_with_wallet_confirmation and not close_fill_threshold_met:
                        pending_exit["allow_partial_fill_terminal"] = True
                    pending_exit["filled_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "provider_exit_fill",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "cost_basis_usd": cost_basis,
                        "settlement_proceeds_usd": close_qty * cp,
                        "filled_size": close_qty,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
                        session,
                        row=row,
                        payload=payload,
                        signal_payload=pending_signal_payload,
                        market_info=pending_market_info,
                        close_trigger=close_trigger,
                        realized_pnl=_pnl,
                        now=now,
                    )
                    if reverse_signal_id and reverse_source:
                        reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)
                    row.payload_json = payload
                    _apply_position_close_verification(
                        session,
                        row=row,
                        now=now,
                        event_type="provider_exit_fill",
                        payload_json=payload,
                    )
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            pending_close_trigger = str(pending_exit.get("close_trigger") or "").strip().lower()
            working_provider_status = snapshot_status in {
                "open",
                "working",
                "active",
                "live",
                "unmatched",
                "partially_filled",
            }
            rapid_exit_requote_enabled = _is_rapid_close_trigger(pending_close_trigger)
            take_profit_requote_enabled = pending_exit_kind == "take_profit_limit"
            partial_fill_terminal_requires_retry = (
                required_exit_size > 0.0
                and terminal_provider_status
                and snapshot_filled_size > 0.0
                and fill_ratio + 1e-9 < threshold_ratio
                and wallet_position_size > _WALLET_SIZE_EPSILON
            )
            if partial_fill_terminal_requires_retry:
                retry_delay_seconds = (
                    0.5
                    if rapid_exit_requote_enabled or take_profit_requote_enabled
                    else max(1.0, _failed_exit_retry_delay_seconds("provider_partial_fill_terminal"))
                )
                if not dry_run:
                    pending_exit["status"] = "failed"
                    pending_exit["retry_count"] = int(pending_exit.get("retry_count", 0) or 0) + 1
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = (
                        "provider_partial_fill_terminal:"
                        f"{snapshot_status or 'unknown'}:{snapshot_filled_size:.6f}/{required_exit_size:.6f}"
                    )
                    pending_exit["next_retry_at"] = _iso_utc(now + timedelta(seconds=retry_delay_seconds))
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            reprice_attempts = int(safe_float(pending_exit.get("reprice_attempts"), 0) or 0)
            max_reprice_attempts_default = (
                8
                if rapid_exit_requote_enabled
                else (4 if take_profit_requote_enabled else 6)
            )
            max_reprice_attempts = max(
                1,
                int(safe_float(pending_exit.get("max_reprice_attempts"), max_reprice_attempts_default) or max_reprice_attempts_default),
            )
            last_attempt_dt = _parse_iso_utc_naive(
                pending_exit.get("last_attempt_at") or pending_exit.get("triggered_at")
            )
            reprice_cooldown_default = 0.5 if rapid_exit_requote_enabled else (1.0 if take_profit_requote_enabled else 2.0)
            reprice_cooldown_seconds = max(
                0.5,
                safe_float(pending_exit.get("reprice_cooldown_seconds"), reprice_cooldown_default) or reprice_cooldown_default,
            )
            cooldown_elapsed = (
                last_attempt_dt is None or (now_naive - last_attempt_dt).total_seconds() >= reprice_cooldown_seconds
            )
            stale_requote_after_seconds_default = (
                1.0
                if rapid_exit_requote_enabled
                else (2.0 if take_profit_requote_enabled else 6.0)
            )
            stale_requote_after_seconds = max(
                0.5,
                safe_float(
                    pending_exit.get("reprice_stale_after_seconds"),
                    stale_requote_after_seconds_default,
                )
                or stale_requote_after_seconds_default,
            )
            stale_requote_due = last_attempt_dt is not None and (
                (now_naive - last_attempt_dt).total_seconds() >= stale_requote_after_seconds
            )
            reprice_enabled = rapid_exit_requote_enabled or take_profit_requote_enabled or stale_requote_due
            if (
                reprice_enabled
                and working_provider_status
                and required_exit_size > 0.0
                and fill_ratio + 1e-9 < threshold_ratio
                and reprice_attempts < max_reprice_attempts
                and cooldown_elapsed
            ):
                remaining_exit_size = _remaining_exit_size(
                    required_exit_size=required_exit_size,
                    pending_exit=pending_exit,
                    wallet_position_size=wallet_position_size,
                )
                if token_id and remaining_exit_size > 0.0:
                    if exit_submissions_this_pass[0] >= _live_exit_submission_cap():
                        # Budget blown — defer requote.  Do NOT cancel the
                        # working provider order: cancelling here without
                        # resubmitting would leave the position without an
                        # active exit.  The existing provider order keeps
                        # working until we can requote on a later pass.
                        held += 1
                        continue
                    exit_submissions_this_pass[0] += 1
                    cancel_target = provider_clob_order_id
                    cancel_ok = True
                    if cancel_target:
                        async with release_conn(session):
                            try:
                                cancel_ok = bool(await live_execution_service.cancel_order(cancel_target))
                            except Exception:
                                cancel_ok = False

                    pending_exit["reprice_attempts"] = reprice_attempts + 1
                    pending_exit["last_reprice_at"] = _iso_utc(now)
                    pending_exit["last_attempt_at"] = _iso_utc(now)

                    if cancel_ok:
                        try:
                            from services.live_execution_adapter import execute_live_order

                            base_min_order_size_usd = _resolve_position_min_order_size_usd(
                                trader_params=params,
                                payload=payload,
                                mode="live",
                            )
                            min_order_size_usd = _effective_exit_min_order_size_usd(
                                base_min_order_size_usd,
                                pending_close_trigger,
                            )
                            pending_exit["current_mid_price"] = pending_current_price
                            take_profit_floor_price = _take_profit_price_floor(pending_exit)
                            fallback_exit_price = (
                                pending_current_price
                                if pending_current_price is not None and pending_current_price > 0.0
                                else (
                                    safe_float(pending_exit.get("close_price"))
                                    if safe_float(pending_exit.get("close_price")) is not None
                                    else 0.01
                                )
                            )
                            if take_profit_floor_price is None and pending_exit_kind == "take_profit_limit":
                                pending_exit["close_trigger"] = "tp_underwater_market_sell"
                                pending_exit["post_only"] = False
                                async with release_conn(session):
                                    await _prepare_sell_allowance_bounded(token_id)
                                    exec_result = await asyncio.wait_for(
                                        execute_live_order(
                                            token_id=token_id,
                                            side="SELL",
                                            size=remaining_exit_size,
                                            fallback_price=fallback_exit_price,
                                            min_order_size_usd=min_order_size_usd,
                                            time_in_force="IOC",
                                            post_only=False,
                                            resolve_live_price=True,
                                            enforce_fallback_bound=False,
                                        ),
                                        timeout=_LIVE_EXIT_ORDER_TIMEOUT_SECONDS,
                                    )
                            else:
                                if take_profit_floor_price is not None:
                                    fallback_exit_price = max(fallback_exit_price, take_profit_floor_price)
                                    pending_exit["close_price"] = float(fallback_exit_price)
                                    pending_exit["post_only"] = True
                                requote_as_rapid_exit = rapid_exit_requote_enabled
                                async with release_conn(session):
                                    await _prepare_sell_allowance_bounded(token_id)
                                    exec_result = await asyncio.wait_for(
                                        execute_live_order(
                                            token_id=token_id,
                                            side="SELL",
                                            size=remaining_exit_size,
                                            fallback_price=fallback_exit_price,
                                            min_order_size_usd=min_order_size_usd,
                                            time_in_force="IOC" if requote_as_rapid_exit else "GTC",
                                            post_only=bool(take_profit_requote_enabled and not requote_as_rapid_exit),
                                            resolve_live_price=requote_as_rapid_exit,
                                            enforce_fallback_bound=not requote_as_rapid_exit,
                                        ),
                                        timeout=_LIVE_EXIT_ORDER_TIMEOUT_SECONDS,
                                    )
                            if exec_result.status in {"executed", "open", "submitted"}:
                                pending_exit["status"] = "submitted"
                                pending_exit["allowance_error_count"] = 0
                                pending_exit["exit_order_id"] = str(exec_result.order_id or "")
                                pending_exit["provider_clob_order_id"] = str(
                                    (exec_result.payload or {}).get("clob_order_id") or ""
                                )
                                pending_exit["provider_status"] = str(
                                    (exec_result.payload or {}).get("trading_status") or ""
                                ).strip().lower()
                                incremental_fill = max(
                                    0.0,
                                    safe_float((exec_result.payload or {}).get("filled_size"), 0.0) or 0.0,
                                )
                                if incremental_fill > 0.0:
                                    prior_filled = max(
                                        0.0,
                                        safe_float(pending_exit.get("filled_size"), 0.0) or 0.0,
                                    )
                                    pending_exit["filled_size"] = float(prior_filled + incremental_fill)
                                    pending_exit["fill_ratio"] = float(
                                        (prior_filled + incremental_fill) / required_exit_size
                                    )
                                reprice_fill_price = safe_float((exec_result.payload or {}).get("average_fill_price"))
                                if reprice_fill_price is not None and reprice_fill_price > 0.0:
                                    pending_exit["average_fill_price"] = float(reprice_fill_price)
                                pending_exit["next_retry_at"] = None
                            else:
                                pending_exit["status"] = "failed"
                                pending_exit["retry_count"] = int(pending_exit.get("retry_count", 0) or 0) + 1
                                pending_exit["last_error"] = _format_exit_error(
                                    exec_result.error_message
                                ) if exec_result.error_message else "exit_requote_failed"
                                _bump_allowance_error_counter(pending_exit, pending_exit["last_error"])
                                pending_exit["next_retry_at"] = _iso_utc(
                                    now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                                )
                        except Exception as exc:
                            pending_exit["status"] = "failed"
                            pending_exit["retry_count"] = int(pending_exit.get("retry_count", 0) or 0) + 1
                            # Use _format_exit_error so an empty-message
                            # exception (TimeoutError()) still produces
                            # a useful last_error instead of "".
                            pending_exit["last_error"] = _format_exit_error(exc)
                            _bump_allowance_error_counter(pending_exit, pending_exit["last_error"])
                            pending_exit["next_retry_at"] = _iso_utc(
                                now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                            )
                    else:
                        pending_exit["status"] = "failed"
                        pending_exit["retry_count"] = int(pending_exit.get("retry_count", 0) or 0) + 1
                        pending_exit["last_error"] = "exit_requote_cancel_failed"
                        _bump_allowance_error_counter(pending_exit, pending_exit["last_error"])
                        pending_exit["next_retry_at"] = _iso_utc(
                            now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                        )

                    if not dry_run:
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue

            if snapshot_status in {"cancelled", "expired", "failed"}:
                if not dry_run:
                    pending_exit["status"] = "failed"
                    pending_exit["retry_count"] = int(pending_exit.get("retry_count", 0) or 0) + 1
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = f"provider_exit_status:{snapshot_status}"
                    pending_exit["next_retry_at"] = _iso_utc(
                        now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                    )
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            if not dry_run and snapshot is not None:
                payload["pending_live_exit"] = pending_exit
                _attach_pending_state(payload)
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1

            if pending_exit_kind != "take_profit_limit":
                if not dry_run and snapshot is None and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

        # ── Failed exit retry logic ────────────────────────────────────
        # If a previous cycle recorded a pending_live_exit that failed,
        # retry submitting the sell order (with cooldown & max retries).
        if isinstance(pending_exit, dict) and pending_exit.get("status") == "failed":
            wallet_resolution_confirmed = wallet_settlement_price is not None
            if pending_winning_idx is not None and pending_outcome_idx is not None and wallet_resolution_confirmed:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = 1.0 if pending_winning_idx == pending_outcome_idx else 0.0
                close_trigger = "resolution_inferred" if pending_winning_idx_inferred else "resolution"
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "resolved_settlement",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            if wallet_settlement_price is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = wallet_settlement_price
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger="resolution")
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "wallet_redeemable_mark",
                        "close_trigger": "resolution",
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "resolution",
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            retry_count = int(pending_exit.get("retry_count", 0) or 0)
            last_attempt_iso = pending_exit.get("last_attempt_at") or pending_exit.get("triggered_at")
            last_attempt_dt = _parse_iso_utc_naive(last_attempt_iso)
            next_retry_iso = pending_exit.get("next_retry_at")
            next_retry_dt = _parse_iso_utc_naive(next_retry_iso)
            pending_close_trigger = str(pending_exit.get("close_trigger") or "").strip().lower()
            pending_exit_kind = str(pending_exit.get("kind") or "").strip().lower()
            force_rapid_retry = _is_rapid_close_trigger(pending_close_trigger)
            if force_rapid_retry:
                min_retry_seconds = max(0.5, safe_float(pending_exit.get("rapid_retry_seconds"), 1.0) or 1.0)
            else:
                min_retry_seconds = max(
                    _FAILED_EXIT_MIN_RETRY_INTERVAL_SECONDS,
                    _failed_exit_retry_delay_seconds(pending_exit.get("last_error")),
                )
            now_naive = now.astimezone(timezone.utc).replace(tzinfo=None)
            seconds_since_attempt = (
                (now_naive - last_attempt_dt).total_seconds() if last_attempt_dt is not None else float("inf")
            )

            soft_retry_exhausted_bypass = bool(
                retry_count >= _FAILED_EXIT_MAX_RETRIES
                and retry_count < _FAILED_EXIT_HARD_RETRY_CEILING
                and wallet_position_size > _WALLET_SIZE_EPSILON
                and token_id
            )
            if retry_count >= _FAILED_EXIT_HARD_RETRY_CEILING and not dry_run:
                # Hard ceiling reached -- stop burning DB on an exit that has
                # failed far beyond the soft-bypass threshold.  An operator
                # must review last_error and either reset retry_count or
                # close the position out-of-band.
                logger.error(
                    "Exit retry HARD CEILING reached for order=%s retry_count=%d wallet_position_size=%.6f last_error=%s",
                    row.id,
                    retry_count,
                    wallet_position_size,
                    _format_exit_error(pending_exit.get("last_error")),
                )
                pending_exit["status"] = "blocked_retry_exhausted_hard"
                pending_exit["exhausted_at"] = _iso_utc(now)
                pending_exit["retry_count"] = retry_count
                pending_exit["last_attempt_at"] = _iso_utc(now)
                pending_exit["last_error"] = _format_exit_error(
                    pending_exit.get("last_error") or "hard_ceiling_reached"
                )
                pending_exit["next_retry_at"] = None
                payload["pending_live_exit"] = pending_exit
                _attach_pending_state(payload)
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
                held += 1
                continue
            if retry_count >= _FAILED_EXIT_MAX_RETRIES and not soft_retry_exhausted_bypass:
                if not dry_run:
                    pending_exit["status"] = "blocked_retry_exhausted"
                    pending_exit["exhausted_at"] = _iso_utc(now)
                    pending_exit["retry_count"] = retry_count
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = str(pending_exit.get("last_error") or "exit_retry_exhausted")
                    pending_exit["next_retry_at"] = None
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            if next_retry_dt is not None and now_naive < next_retry_dt:
                if not dry_run and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            if seconds_since_attempt < min_retry_seconds:
                # Cooldown not elapsed — hold, don't spam.
                if not dry_run and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            # Retry: refresh allowance and re-place exit order
            _fill_not_r, _fill_sz_r, _ = _extract_live_fill_metrics(payload)
            exit_size = max(
                max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0),
                _fill_sz_r if _fill_sz_r > 0 else 0.0,
            )
            wallet_exit_size_cap = wallet_position_size if wallet_position_size > _WALLET_SIZE_EPSILON else 0.0
            if exit_size <= 0.0 and wallet_exit_size_cap > 0.0:
                exit_size = wallet_exit_size_cap
            if exit_size > 0.0:
                pending_exit["exit_size"] = float(exit_size)
            exit_size = _remaining_exit_size(
                required_exit_size=exit_size,
                pending_exit=pending_exit,
                wallet_position_size=wallet_exit_size_cap,
            )
            if exit_size > 0.0:
                pending_exit["remaining_size"] = float(exit_size)
            exit_price = safe_float(pending_exit.get("close_price")) or 0.01
            take_profit_floor_price = _take_profit_price_floor(pending_exit)
            if take_profit_floor_price is not None:
                exit_price = max(exit_price, take_profit_floor_price)
                pending_exit["close_price"] = float(exit_price)
                pending_exit["post_only"] = True
            else:
                # No take-profit floor → this is an SL / time-stop /
                # default-exit retry whose job is to force a fill.  If
                # the venue rejected a SELL at ``exit_price`` once, it
                # will keep rejecting at the same price; walk the
                # limit down based on ``retry_count`` so a thin market
                # eventually gets a price it can fill.  Capped at
                # 10 ¢ off the original; see
                # ``_aggressive_exit_price_decrement``.
                walk_decrement = _aggressive_exit_price_decrement(retry_count)
                if walk_decrement > 0.0:
                    walked_price = max(0.01, exit_price - walk_decrement)
                    if walked_price < exit_price - 1e-9:
                        pending_exit["aggressive_walk_down_applied"] = {
                            "retry_count": retry_count,
                            "original_close_price": float(exit_price),
                            "walked_close_price": float(walked_price),
                            "decrement": float(walk_decrement),
                        }
                        exit_price = walked_price
                        pending_exit["close_price"] = float(exit_price)
            base_min_order_size_usd = _resolve_position_min_order_size_usd(
                trader_params=params,
                payload=payload,
                mode="live",
            )
            min_order_size_usd = _effective_exit_min_order_size_usd(
                base_min_order_size_usd,
                pending_exit.get("close_trigger"),
            )

            if token_id and exit_size > 0:
                exit_notional_estimate = float(exit_size) * float(max(exit_price, 0.0))
                if exit_notional_estimate + 1e-9 < min_order_size_usd:
                    if not dry_run:
                        pending_exit["status"] = "blocked_min_notional"
                        pending_exit["retry_count"] = retry_count + 1
                        pending_exit["exhausted_at"] = _iso_utc(now)
                        pending_exit["last_attempt_at"] = _iso_utc(now)
                        pending_exit["last_error"] = (
                            f"exit_notional_below_min:{exit_notional_estimate:.4f}<{min_order_size_usd:.4f}"
                        )
                        pending_exit["next_retry_at"] = None
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue
                if exit_submissions_this_pass[0] >= _live_exit_submission_cap():
                    # Defer: this pass already submitted the per-pass cap.
                    # Leave pending_live_exit unchanged so the row picks up on
                    # the next reconcile cycle.  Prevents a 20-order retry
                    # wave from blowing the 30s per-trader budget.
                    held += 1
                    continue
                exit_submissions_this_pass[0] += 1
                try:
                    from services.live_execution_adapter import execute_live_order

                    rapid_retry_exit = force_rapid_retry
                    post_only_retry = bool(pending_exit_kind == "take_profit_limit" and not rapid_retry_exit)
                    async with release_conn(session):
                        await _prepare_sell_allowance_bounded(token_id)
                        exec_result = await asyncio.wait_for(
                            execute_live_order(
                                token_id=token_id,
                                side="SELL",
                                size=exit_size,
                                fallback_price=exit_price,
                                min_order_size_usd=min_order_size_usd,
                                time_in_force="IOC" if rapid_retry_exit else "GTC",
                                post_only=post_only_retry,
                                resolve_live_price=rapid_retry_exit,
                                enforce_fallback_bound=not rapid_retry_exit,
                            ),
                            timeout=_LIVE_EXIT_RETRY_TIMEOUT_SECONDS,
                        )
                    if exec_result.status in {"executed", "open", "submitted"}:
                        logger.info(
                            "Exit retry succeeded for order=%s attempt=%d status=%s",
                            row.id,
                            retry_count + 1,
                            exec_result.status,
                        )
                        if not dry_run:
                            pending_exit["status"] = "submitted"
                            pending_exit["retry_count"] = retry_count if soft_retry_exhausted_bypass else retry_count + 1
                            pending_exit["allowance_error_count"] = 0
                            pending_exit["last_attempt_at"] = _iso_utc(now)
                            pending_exit["next_retry_at"] = None
                            pending_exit["exit_order_id"] = exec_result.order_id
                            pending_exit["provider_clob_order_id"] = str(
                                (exec_result.payload or {}).get("clob_order_id") or ""
                            )
                            payload["pending_live_exit"] = pending_exit
                            _attach_pending_state(payload)
                            row.payload_json = payload
                            row.updated_at = now
                            state_updates += 1
                        held += 1
                        continue
                    logger.warning(
                        "Exit retry failed for order=%s attempt=%d status=%s error=%s",
                        row.id,
                        retry_count + 1,
                        exec_result.status,
                        _format_exit_error(exec_result.error_message),
                    )
                    if not dry_run:
                        _apply_failed_exit_state(
                            pending_exit,
                            error=exec_result.error_message,
                            now=now,
                            retry_count=retry_count + 1,
                        )
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue
                except Exception as exc:
                    logger.warning(
                        "Exit retry exception for order=%s attempt=%d: %s",
                        row.id,
                        retry_count + 1,
                        _format_exit_error(exc),
                        exc_info=exc,
                    )
                    if not dry_run:
                        _apply_failed_exit_state(
                            pending_exit,
                            error=exc,
                            now=now,
                            retry_count=retry_count + 1,
                        )
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue
            else:
                # No token_id or size — can't retry, mark exhausted
                if not dry_run:
                    pending_exit["status"] = "failed"
                    pending_exit["exhausted_at"] = _iso_utc(now)
                    pending_exit["retry_count"] = retry_count + 1
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = "missing token_id or fill_size"
                    pending_exit["next_retry_at"] = _iso_utc(
                        now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                    )
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

        if isinstance(pending_exit, dict) and pending_exit.get("status") in {
            "blocked_no_inventory",
            "blocked_min_notional",
            "blocked_retry_exhausted",
            "blocked_retry_exhausted_hard",
            "blocked_persistent_timeout",
            "blocked_orderbook_gone",
        }:
            pending_wallet_resolution_confirmed = wallet_settlement_price is not None
            if pending_winning_idx is not None and pending_outcome_idx is not None and pending_wallet_resolution_confirmed:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = 1.0 if pending_winning_idx == pending_outcome_idx else 0.0
                close_trigger = "resolution_inferred" if pending_winning_idx_inferred else "resolution"
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "resolved_settlement",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue
            if wallet_settlement_price is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = wallet_settlement_price
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger="resolution")
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "wallet_redeemable_mark",
                        "close_trigger": "resolution",
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "resolution",
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue
            pending_expired_terminal_price, pending_expired_terminal_source = _infer_post_end_terminal_price(
                market_info=pending_market_info,
                current_price=pending_current_price,
                current_price_source=pending_current_price_source,
                previous_mark_price=pending_prev_last_mark,
                wallet_mark_price=wallet_mark_price,
                now=now,
            )
            if pending_expired_terminal_price is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = pending_expired_terminal_price
                close_trigger = "resolution_extreme_mark"
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": pending_expired_terminal_source,
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=ns,
                        actual_profit=_pnl,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue
            pending_provider_status = str(pending_exit.get("provider_status") or "").strip().lower()
            provider_close_verified = _pending_exit_has_verified_terminal_fill(pending_exit)
            wallet_flat_by_snapshot = wallet_position_observed and wallet_position_size <= _WALLET_SIZE_EPSILON
            latest_wallet_sell_trade_size = (
                _extract_wallet_trade_size(latest_wallet_sell_trade)
                if isinstance(latest_wallet_sell_trade, dict)
                else 0.0
            )
            latest_wallet_sell_trade_evidence_key = (
                _wallet_close_evidence_key("wallet_trade", token_id or "", latest_wallet_sell_trade)
                if isinstance(latest_wallet_sell_trade, dict)
                else ""
            )
            latest_wallet_sell_trade_available_size = latest_wallet_sell_trade_size
            if latest_wallet_sell_trade_evidence_key and latest_wallet_sell_trade_size > _WALLET_SIZE_EPSILON:
                latest_wallet_sell_trade_available_size = max(
                    0.0,
                    latest_wallet_sell_trade_size
                    - wallet_terminal_close_consumed_size_by_key.get(latest_wallet_sell_trade_evidence_key, 0.0),
                )
            latest_wallet_sell_trade_has_identity = (
                isinstance(latest_wallet_sell_trade, dict)
                and bool(
                    str(
                        latest_wallet_sell_trade.get("trade_id")
                        or latest_wallet_sell_trade.get("transactionHash")
                        or latest_wallet_sell_trade.get("txHash")
                        or latest_wallet_sell_trade.get("tx_hash")
                        or ""
                    ).strip()
                )
            )
            wallet_trade_confirms_exit = (
                isinstance(latest_wallet_sell_trade, dict)
                and latest_wallet_sell_trade_available_size > _WALLET_SIZE_EPSILON
                and latest_wallet_sell_trade_has_identity
            )
            wallet_flat_by_trade = (
                wallet_position_size <= _WALLET_SIZE_EPSILON
                and wallet_trade_confirms_exit
            )
            wallet_flat_override = (
                (wallet_flat_by_snapshot or wallet_flat_by_trade)
                and (provider_close_verified or wallet_trade_confirms_exit)
            )
            if wallet_flat_override:
                close_trigger = "wallet_flat_override"
                close_price = (
                    safe_float(latest_wallet_sell_trade.get("price"))
                    if isinstance(latest_wallet_sell_trade, dict)
                    else None
                )
                if close_price is None or close_price < 0.0:
                    close_price = safe_float(pending_exit.get("average_fill_price"))
                if close_price is None or close_price < 0.0:
                    close_price = pending_current_price if pending_current_price is not None else pending_market_side_price
                if close_price is None or close_price < 0.0:
                    close_price = 0.0

                filled_notional, filled_size, fill_price = _extract_live_fill_metrics(payload)
                entry_price = fill_price if fill_price is not None and fill_price > 0 else safe_float(row.effective_price)
                if entry_price is None or entry_price <= 0:
                    entry_price = safe_float(row.entry_price)
                notional = filled_notional if filled_notional > 0.0 else (safe_float(row.notional_usd) or 0.0)
                quantity = filled_size if filled_size > 0.0 else (notional / entry_price if entry_price and entry_price > 0 else 0.0)

                if quantity > 0.0 and notional > 0.0:
                    _provider_status, provider_filled_size, _provider_filled_notional, _provider_fill_price = _pending_exit_fill_evidence(
                        pending_exit
                    )
                    wallet_trade_covers_entry = (
                        latest_wallet_sell_trade_available_size > _WALLET_SIZE_EPSILON
                        and latest_wallet_sell_trade_available_size + _WALLET_SIZE_EPSILON >= quantity * 0.995
                    )
                    provider_close_covers_entry = (
                        provider_filled_size > _WALLET_SIZE_EPSILON
                        and provider_filled_size + _WALLET_SIZE_EPSILON >= quantity * 0.995
                    )
                    if not (wallet_trade_covers_entry or provider_close_covers_entry):
                        details.append(
                            {
                                "order_id": row.id,
                                "market_id": row.market_id,
                                "close_trigger": close_trigger,
                                "wallet_trade_size": latest_wallet_sell_trade_available_size,
                                "provider_filled_size": provider_filled_size,
                                "required_entry_size": quantity,
                                "next_status": row.status,
                                "note": "ignored_wallet_flat_override_underfilled_entry",
                            }
                        )
                        held += 1
                        continue
                    proceeds = float(quantity) * float(close_price)
                    pnl = proceeds - float(notional)
                    next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)
                    if not dry_run:
                        row.status = next_status
                        row.actual_profit = pnl
                        row.updated_at = now
                        pending_exit["status"] = "superseded_wallet_flat"
                        pending_exit["resolved_at"] = _iso_utc(now)
                        pending_exit["provider_status"] = pending_provider_status or pending_exit.get("provider_status")
                        payload["pending_live_exit"] = pending_exit
                        payload["position_state"] = pending_next_state
                        payload["position_close"] = {
                            "close_price": close_price,
                            "price_source": "wallet_flat_override",
                            "close_trigger": close_trigger,
                            "realized_pnl": pnl,
                            "cost_basis_usd": notional,
                            "settlement_proceeds_usd": proceeds,
                            "filled_size": quantity,
                            "filled_notional_usd": notional,
                            "market_tradable": pending_market_tradable,
                            "age_minutes": pending_exit.get("age_minutes"),
                            "closed_at": _iso_utc(now),
                            "reason": reason,
                            "wallet_trade_id": (
                                str(latest_wallet_sell_trade.get("trade_id") or "")
                                if isinstance(latest_wallet_sell_trade, dict)
                                else ""
                            ),
                            "wallet_trade_size": (
                                _extract_wallet_trade_size(latest_wallet_sell_trade)
                                if isinstance(latest_wallet_sell_trade, dict)
                                else None
                            ),
                            "wallet_trade_timestamp": (
                                _iso_utc(latest_wallet_sell_trade.get("timestamp"))
                                if isinstance(latest_wallet_sell_trade, dict)
                                and isinstance(latest_wallet_sell_trade.get("timestamp"), datetime)
                                else None
                            ),
                            "wallet_activity_transaction_hash": (
                                str(
                                    latest_wallet_sell_trade.get("transactionHash")
                                    or latest_wallet_sell_trade.get("txHash")
                                    or latest_wallet_sell_trade.get("tx_hash")
                                    or ""
                                ).strip()
                                if isinstance(latest_wallet_sell_trade, dict)
                                else ""
                            ) or None,
                        }
                        if latest_wallet_sell_trade_evidence_key and latest_wallet_sell_trade_available_size > _WALLET_SIZE_EPSILON:
                            wallet_terminal_close_consumed_size_by_key[latest_wallet_sell_trade_evidence_key] = (
                                wallet_terminal_close_consumed_size_by_key.get(latest_wallet_sell_trade_evidence_key, 0.0)
                                + min(latest_wallet_sell_trade_available_size, quantity)
                            )
                        row.payload_json = payload
                        _apply_position_close_verification(
                            session,
                            row=row,
                            now=now,
                            event_type="wallet_flat_close",
                            payload_json=payload,
                        )
                        hot_state.record_order_resolved(
                            trader_id=trader_id,
                            mode=str(row.mode or ""),
                            order_id=str(row.id or ""),
                            market_id=str(row.market_id or ""),
                            direction=str(row.direction or ""),
                            source=str(row.source or ""),
                            status=next_status,
                            actual_profit=pnl,
                            payload=payload,
                            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                        )
                        closed += 1
                    total_realized_pnl += pnl
                    by_status[next_status] = int(by_status.get(next_status, 0)) + 1
                    would_close += 1
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "close_price": close_price,
                            "realized_pnl": pnl,
                            "next_status": next_status,
                        }
                    )
                    continue
            if not dry_run and pending_state_changed:
                payload["pending_live_exit"] = pending_exit
                payload["position_state"] = pending_next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        # If pending_live_exit is in-flight (submitted), skip normal processing
        if (
            isinstance(pending_exit, dict)
            and pending_exit.get("status") in {"submitted", "pending"}
            and str(pending_exit.get("kind") or "").strip().lower() != "take_profit_limit"
        ):
            if not dry_run and pending_state_changed:
                payload["position_state"] = pending_next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        filled_notional = max(0.0, float(entry_fill_notional or 0.0))
        filled_size = max(0.0, float(entry_fill_size or 0.0))
        fill_price = entry_fill_price if entry_fill_price is not None and entry_fill_price > 0.0 else None
        status_key = str(row.status or "").strip().lower()
        raw_filled_notional, raw_filled_size, _raw_fill_price = _extract_live_fill_metrics(payload)
        provider_snapshot_status = _provider_snapshot_status(payload)
        if (
            status_key in {"open", "submitted", "pending", "placing", "queued"}
            and raw_filled_notional <= 0.0
            and raw_filled_size <= 0.0
            and wallet_position_size <= _WALLET_SIZE_EPSILON
            and provider_snapshot_status not in {"filled", "partially_filled", "matched", "executed"}
        ):
            if pending_market_info is not None and not pending_market_tradable:
                next_status = "cancelled"
                if not dry_run:
                    row.status = next_status
                    row.actual_profit = 0.0
                    row.updated_at = now
                    payload["position_close"] = {
                        "close_price": 0.0,
                        "price_source": "terminal_unfilled_cancel",
                        "close_trigger": "terminal_unfilled_cancel",
                        "realized_pnl": 0.0,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=next_status,
                        actual_profit=0.0,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "terminal_unfilled_cancel",
                        "close_price": 0.0,
                        "realized_pnl": 0.0,
                        "next_status": next_status,
                    }
                )
                continue
            skipped += 1
            skipped_reasons["awaiting_fill"] = int(skipped_reasons.get("awaiting_fill", 0)) + 1
            continue
        entry_price = fill_price if fill_price is not None and fill_price > 0 else safe_float(row.effective_price)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price)
        notional = filled_notional if filled_notional > 0.0 else (safe_float(row.notional_usd) or 0.0)
        outcome_idx = _direction_outcome_index(
            row.direction,
            market_info=pending_market_info,
            token_id=_extract_leg_token_id(payload),
        )
        if outcome_idx is None or entry_price is None or entry_price <= 0 or notional <= 0:
            if (
                wallet_position_size <= _WALLET_SIZE_EPSILON
                and raw_filled_notional <= 0.0
                and raw_filled_size <= 0.0
                and provider_snapshot_status in {"filled", "partially_filled", "matched", "executed"}
            ):
                next_status = "cancelled"
                if not dry_run:
                    row.status = next_status
                    row.actual_profit = 0.0
                    row.notional_usd = 0.0
                    row.executed_at = None
                    row.updated_at = now
                    payload["position_close"] = {
                        "close_price": 0.0,
                        "price_source": "terminal_unfilled_provider_snapshot",
                        "close_trigger": "terminal_unfilled_provider_snapshot",
                        "realized_pnl": 0.0,
                        "provider_status": provider_snapshot_status,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    payload["provider_failure_finalized"] = {
                        "finalized_at": _iso_utc(now),
                        "provider_status": provider_snapshot_status,
                        "reason": "provider_snapshot_without_fill_or_wallet_position",
                    }
                    row.payload_json = payload
                    hot_state.record_order_resolved(
                        trader_id=trader_id,
                        mode=str(row.mode or ""),
                        order_id=str(row.id or ""),
                        market_id=str(row.market_id or ""),
                        direction=str(row.direction or ""),
                        source=str(row.source or ""),
                        status=next_status,
                        actual_profit=0.0,
                        payload=payload,
                        copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                    )
                    closed += 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "terminal_unfilled_provider_snapshot",
                        "close_price": 0.0,
                        "realized_pnl": 0.0,
                        "provider_status": provider_snapshot_status,
                        "next_status": next_status,
                    }
                )
                continue
            skipped += 1
            skipped_reasons["invalid_entry"] = int(skipped_reasons.get("invalid_entry", 0)) + 1
            continue

        signal_payload = signal_payloads.get(str(row.signal_id), {})
        market_info = market_info_by_id.get(str(row.market_id or ""))
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        market_seconds_left = _market_seconds_left(market_info, now)
        market_end_time = _market_end_time_iso(market_info)
        winning_idx = _extract_winning_outcome_index(market_info)
        winning_idx_inferred = False
        if winning_idx is None and resolution_infer_from_prices:
            inferred_idx = _extract_winning_outcome_index_from_prices(
                market_info,
                market_tradable=market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_idx is not None:
                winning_idx = inferred_idx
                winning_idx_inferred = True
        market_side_price = _extract_market_side_price(market_info, outcome_idx)
        ws_side_price = ws_mid_prices.get(token_id) if token_id else None
        clob_side_price = clob_mid_prices.get(token_id) if token_id else None

        close_price: Optional[float] = None
        close_trigger: Optional[str] = None
        price_source: Optional[str] = None
        trailing_trigger_price: Optional[float] = None

        current_price = (
            ws_side_price
            if ws_side_price is not None
            else (
                clob_side_price
                if clob_side_price is not None
                else (market_side_price if market_side_price is not None else wallet_mark_price)
            )
        )
        current_price = _state_price_floor(current_price)
        current_price_source = (
            "ws_mid"
            if ws_side_price is not None
            else (
                "clob_midpoint"
                if clob_side_price is not None
                else (
                    "market_mark"
                    if market_side_price is not None
                    else ("wallet_mark" if wallet_mark_price is not None else None)
                )
            )
        )

        age_anchor = row.executed_at or row.updated_at or row.created_at
        age_minutes = None
        if age_anchor is not None:
            age_minutes = max(0.0, (now - age_anchor).total_seconds() / 60.0)
        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes

        position_state = _extract_position_state(payload)
        prev_high = safe_float(position_state.get("highest_price"))
        prev_low = safe_float(position_state.get("lowest_price"))
        prev_last_mark = safe_float(position_state.get("last_mark_price"))
        prev_mark_source = str(position_state.get("last_mark_source") or "")
        prev_marked_at = _parse_iso_utc_naive(position_state.get("last_marked_at"))
        highest_price = prev_high
        lowest_price = prev_low
        if current_price is not None:
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)

        mark_updated_at_value = (
            _iso_utc(now)
            if current_price is not None
            else (
                _iso_utc(prev_marked_at.replace(tzinfo=timezone.utc))
                if prev_marked_at is not None
                else ""
            )
        )
        next_state = {
            "highest_price": _state_price_floor(highest_price),
            "lowest_price": _state_price_floor(lowest_price),
            "last_mark_price": current_price if current_price is not None else _state_price_floor(prev_last_mark),
            "last_mark_source": str(current_price_source or prev_mark_source),
            "last_marked_at": mark_updated_at_value,
            "last_exit_evaluated_at": _iso_utc(now),
        }
        prev_mark_age_seconds = (
            max(0.0, (now_naive - prev_marked_at).total_seconds()) if prev_marked_at is not None else None
        )
        fallback_mark_fresh = bool(
            prev_last_mark is not None
            and prev_mark_age_seconds is not None
            and prev_mark_age_seconds <= _MAX_LIVE_EXIT_FALLBACK_MARK_AGE_SECONDS
        )
        if current_price is not None:
            exit_eval_price = current_price
            exit_eval_price_source = current_price_source
        elif fallback_mark_fresh:
            exit_eval_price = _state_price_floor(prev_last_mark)
            exit_eval_price_source = "position_state_mark"
        else:
            exit_eval_price = None
            exit_eval_price_source = None

        wallet_resolution_confirmed = wallet_settlement_price is not None
        if winning_idx is not None and wallet_resolution_confirmed:
            close_price = 1.0 if winning_idx == outcome_idx else 0.0
            close_trigger = "resolution_inferred" if winning_idx_inferred else "resolution"
            price_source = "resolved_settlement"
        elif wallet_settlement_price is not None:
            close_price = wallet_settlement_price
            close_trigger = "resolution"
            price_source = "wallet_redeemable_mark"
        else:
            expired_terminal_price, expired_terminal_source = _infer_post_end_terminal_price(
                market_info=market_info,
                current_price=exit_eval_price,
                current_price_source=exit_eval_price_source,
                previous_mark_price=prev_last_mark,
                wallet_mark_price=wallet_mark_price,
                now=now,
            )
            if expired_terminal_price is not None:
                close_price = expired_terminal_price
                close_trigger = "resolution_extreme_mark"
                price_source = expired_terminal_source
            else:
                bundle_position_shortfall = _bundle_position_shortfall(
                    token_id=token_id,
                    payload=payload,
                    signal_payload=signal_payload,
                    wallet_positions_by_token=wallet_positions_by_token,
                )
                pnl_pct = None
                if exit_eval_price is not None and entry_price > 0:
                    pnl_pct = ((exit_eval_price - entry_price) / entry_price) * 100.0

                if bundle_position_shortfall is not None:
                    close_price = exit_eval_price
                    if close_price is None or close_price <= 0.0:
                        close_price = _state_price_floor(entry_price)
                    close_trigger = "force_flatten_bundle_residual"
                    price_source = exit_eval_price_source or ("entry_price" if close_price is not None else None)
                    if not dry_run:
                        payload["bundle_execution_state"] = {
                            "status": "incomplete_live_wallet_bundle",
                            "required_token_ids": list(bundle_position_shortfall["required_token_ids"]),
                            "observed_token_ids": list(bundle_position_shortfall["observed_token_ids"]),
                            "missing_token_ids": list(bundle_position_shortfall["missing_token_ids"]),
                            "full_bundle_execution_mode": bundle_position_shortfall.get("full_bundle_execution_mode"),
                            "plan_id": bundle_position_shortfall.get("plan_id"),
                            "detected_at": _iso_utc(now),
                        }
                elif force_mark_to_market and exit_eval_price is not None:
                    close_price = exit_eval_price
                    close_trigger = "manual_mark_to_market"
                    price_source = exit_eval_price_source
                elif (
                    not resolve_only
                    and live_position_notional_cap is not None
                    and notional > live_position_notional_cap + max(0.05, live_position_notional_cap * 0.01)
                    and market_tradable
                    and exit_eval_price is not None
                ):
                    close_price = exit_eval_price
                    close_trigger = "risk_position_cap"
                    price_source = exit_eval_price_source
                    if not dry_run:
                        payload["position_cap_exit"] = {
                            "triggered_at": _iso_utc(now),
                            "notional_usd": float(notional),
                            "cap_usd": float(live_position_notional_cap),
                            "current_price": float(exit_eval_price),
                            "price_source": price_source,
                        }
                else:
                    # ── Strategy-based exit check ──────────────────────────
                    # If the strategy that opened this position has a
                    # should_exit() method, call it first and respect its
                    # decision before falling through to default TP/SL/etc.
                    strategy_slug = (payload.get("strategy_type") or "").strip().lower()
                    strategy_exit = None
                    _exit_instance = await _strategy_exit_instance(session, strategy_slug) if strategy_slug else None
                    if _exit_instance is not None:
                        try:

                            class _LivePositionView:
                                pass

                            pos_view = _LivePositionView()
                            pos_view.entry_price = entry_price
                            pos_view.current_price = exit_eval_price
                            pos_view.highest_price = highest_price
                            pos_view.lowest_price = lowest_price
                            pos_view.age_minutes = age_minutes
                            pos_view.pnl_percent = pnl_pct
                            pos_view.filled_size = (
                                filled_size
                                if filled_size > 0.0
                                else (notional / entry_price if entry_price > 0 else 0.0)
                            )
                            pos_view.notional_usd = notional
                            if "strategy_context" not in payload:
                                payload["strategy_context"] = {}
                            strategy_context_payload = (
                                payload["strategy_context"] if isinstance(payload.get("strategy_context"), dict) else {}
                            )
                            pos_view.strategy_context = payload["strategy_context"]
                            pos_view.config = payload.get("strategy_exit_config", {})
                            pos_view.outcome_idx = outcome_idx

                            min_order_size_usd = _resolve_position_min_order_size_usd(
                                trader_params=params,
                                payload=payload,
                                mode="live",
                            )
                            market_state_dict = {
                                "current_price": exit_eval_price,
                                "market_tradable": market_tradable,
                                "is_resolved": False,
                                "winning_outcome": None,
                                "seconds_left": market_seconds_left,
                                "end_time": market_end_time,
                                "token_id": token_id,
                                "mark_source": exit_eval_price_source,
                                "min_order_size_usd": min_order_size_usd,
                                "notional_usd": notional,
                                "oracle_price": strategy_context_payload.get("oracle_price"),
                                "price_to_beat": strategy_context_payload.get("price_to_beat"),
                                "oracle_age_seconds": strategy_context_payload.get("oracle_age_seconds"),
                                "confirmed_stage_triggered": strategy_context_payload.get("confirmed_stage_triggered"),
                                "strategy_stage": strategy_context_payload.get("stage"),
                            }

                            exit_decision = _exit_instance.should_exit(pos_view, market_state_dict)
                            exit_action = getattr(exit_decision, "action", None) if exit_decision is not None else None
                            if exit_action == "close":
                                strategy_exit = exit_decision
                            elif exit_action == "reduce":
                                strategy_exit = exit_decision
                            elif _strategy_hold_blocks_default_exit(exit_decision):
                                strategy_exit = exit_decision
                        except Exception as exc:
                            logger.warning(
                                "Strategy should_exit() error for %s: %s",
                                strategy_slug,
                                exc,
                            )

                    if strategy_exit is not None and getattr(strategy_exit, "action", None) == "reduce":
                        # SDK contract: ``reduce`` means submit a partial sell
                        # sized by reduce_fraction and keep the position open.
                        # An invalid/missing fraction silently no-ops (we
                        # treat the synthetic floor / next-tick guards as
                        # the safety net).
                        if not dry_run:
                            await _submit_live_partial_exit(
                                session=session,
                                row=row,
                                payload=payload,
                                decision=strategy_exit,
                                close_price=exit_eval_price,
                                price_source=exit_eval_price_source,
                                filled_size=filled_size,
                                notional_usd=notional,
                                entry_price=entry_price,
                                params=params,
                                now=now,
                            )
                            payload["position_state"] = next_state
                            row.payload_json = payload
                            row.updated_at = now
                            state_updates += 1
                        held += 1
                        continue

                    if _strategy_hold_blocks_default_exit(strategy_exit):
                        if not dry_run:
                            payload["position_state"] = next_state
                            row.payload_json = payload
                            row.updated_at = now
                            state_updates += 1
                        held += 1
                        continue

                    active_take_profit_limit = (
                        isinstance(pending_exit, dict)
                        and str(pending_exit.get("kind") or "").strip().lower() == "take_profit_limit"
                        and str(pending_exit.get("status") or "").strip().lower() in {"submitted", "pending"}
                    )
                    if strategy_exit is not None:
                        _arm_reverse_entry_from_exit(
                            row=row,
                            payload=payload,
                            strategy_exit=strategy_exit,
                            market_info=market_info,
                            market_seconds_left=market_seconds_left,
                            current_price=exit_eval_price,
                            now=now,
                        )
                        close_price = (
                            strategy_exit.close_price if strategy_exit.close_price is not None else exit_eval_price
                        )
                        close_trigger = f"strategy:{strategy_exit.reason}"
                        price_source = exit_eval_price_source
                    elif (
                        not resolve_only
                        and take_profit_pct is not None
                        and pnl_pct is not None
                        and min_hold_passed
                        and not active_take_profit_limit
                        and pnl_pct >= take_profit_pct
                    ):
                        close_price = exit_eval_price
                        close_trigger = "take_profit"
                        price_source = exit_eval_price_source
                    elif (
                        not resolve_only
                        and stop_loss_pct is not None
                        and pnl_pct is not None
                        and min_hold_passed
                        and pnl_pct <= -abs(stop_loss_pct)
                    ):
                        close_price = exit_eval_price
                        close_trigger = "stop_loss"
                        price_source = exit_eval_price_source
                    elif (
                        not resolve_only
                        and trailing_stop_pct is not None
                        and trailing_stop_pct > 0
                        and exit_eval_price is not None
                        and highest_price is not None
                        and min_hold_passed
                    ):
                        trailing_trigger_price = highest_price * (1.0 - (trailing_stop_pct / 100.0))
                        if highest_price > entry_price and exit_eval_price <= trailing_trigger_price:
                            close_price = exit_eval_price
                            close_trigger = "trailing_stop"
                            price_source = exit_eval_price_source
                    elif (
                        not resolve_only
                        and max_hold_minutes is not None
                        and age_minutes is not None
                        and age_minutes >= max_hold_minutes
                    ):
                        if exit_eval_price is not None:
                            close_price = exit_eval_price
                            close_trigger = "max_hold"
                            price_source = exit_eval_price_source
                    elif (
                        not resolve_only
                        and close_on_inactive_market
                        and not market_tradable
                        and exit_eval_price is not None
                        and min_hold_passed
                    ):
                        close_price = exit_eval_price
                        close_trigger = "market_inactive"
                        price_source = exit_eval_price_source

        close_is_resolution = close_trigger in {"resolution", "resolution_inferred", "resolution_extreme_mark"}

        if close_price is None:
            state_changed = False
            if current_price is not None:
                mark_stale = (
                    prev_marked_at is None
                    or (now_naive - prev_marked_at).total_seconds() >= mark_touch_interval_seconds
                )
                state_changed = (
                    prev_last_mark is None
                    or abs(prev_last_mark - current_price) > 1e-9
                    or prev_high is None
                    or prev_low is None
                    or abs((prev_high or 0.0) - (highest_price or 0.0)) > 1e-9
                    or abs((prev_low or 0.0) - (lowest_price or 0.0)) > 1e-9
                    or prev_mark_source != str(current_price_source or "")
                    or mark_stale
                )
            if not dry_run and state_changed:
                payload["position_state"] = next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        quantity = filled_size if filled_size > 0.0 else (notional / entry_price if entry_price > 0 else 0.0)
        cost_basis = filled_notional if filled_notional > 0.0 else notional
        if quantity <= 0.0 or cost_basis <= 0.0:
            skipped += 1
            skipped_reasons["invalid_fill_state"] = int(skipped_reasons.get("invalid_fill_state", 0)) + 1
            continue
        proceeds = quantity * close_price
        pnl = proceeds - cost_basis
        next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)

        detail = {
            "order_id": row.id,
            "market_id": row.market_id,
            "direction": row.direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "market_tradable": market_tradable,
            "notional_usd": notional,
            "cost_basis_usd": cost_basis,
            "filled_size": filled_size,
            "filled_notional_usd": filled_notional,
            "quantity": quantity,
            "next_status": next_status,
            "age_minutes": age_minutes,
            "min_hold_minutes": min_hold_minutes,
            "trailing_stop_trigger_price": trailing_trigger_price,
            "highest_price_seen": _state_price_floor(highest_price),
            "lowest_price_seen": _state_price_floor(lowest_price),
        }

        if not close_is_resolution:
            detail["next_status"] = str(row.status or "").strip().lower()
            detail["realized_pnl"] = None
            detail["hypothetical_pnl"] = pnl
            details.append(detail)
            would_close += 1

            if not dry_run:
                payload["position_state"] = next_state
                existing_tp_limit = (
                    pending_exit
                    if isinstance(pending_exit, dict)
                    and str(pending_exit.get("kind") or "").strip().lower() == "take_profit_limit"
                    and str(pending_exit.get("status") or "").strip().lower() in {"submitted", "pending"}
                    else None
                )
                if existing_tp_limit is not None:
                    cancel_target = _pending_exit_provider_clob_id(existing_tp_limit)
                    if not cancel_target:
                        cancel_target = str(existing_tp_limit.get("exit_order_id") or "").strip()
                    cancel_success = False
                    if cancel_target:
                        try:
                            async with release_conn(session):
                                cancel_success = bool(await live_execution_service.cancel_order(cancel_target))
                        except Exception:
                            cancel_success = False
                    existing_tp_limit["cancelled_for_override_at"] = _iso_utc(now)
                    existing_tp_limit["override_cancel_target"] = cancel_target
                    existing_tp_limit["override_cancel_success"] = cancel_success
                    if cancel_success:
                        existing_tp_limit["status"] = "cancelled"
                    payload["superseded_take_profit_exit"] = existing_tp_limit

                exit_record: dict[str, Any] = {
                    "triggered_at": _iso_utc(now),
                    "close_trigger": close_trigger,
                    "close_price": close_price,
                    "price_source": price_source,
                    "market_tradable": market_tradable,
                    "hypothetical_pnl": pnl,
                    "age_minutes": age_minutes,
                    "reason": reason,
                    "retry_count": 0,
                    "status": "pending",
                }

                # Immediately attempt to place the sell order
                token_id = _extract_live_token_id(payload)
                exit_size = filled_size if filled_size > 0.0 else quantity
                if exit_size > 0.0:
                    exit_record["exit_size"] = float(exit_size)
                wallet_exit_size_cap = wallet_position_size if wallet_position_size > _WALLET_SIZE_EPSILON else 0.0
                if wallet_exit_size_cap > 0.0:
                    if exit_size <= 0.0:
                        exit_size = wallet_exit_size_cap
                    else:
                        exit_size = min(exit_size, wallet_exit_size_cap)
                    exit_record["exit_size"] = float(exit_size)
                base_min_order_size_usd = _resolve_position_min_order_size_usd(
                    trader_params=params,
                    payload=payload,
                    mode="live",
                )
                min_order_size_usd = _effective_exit_min_order_size_usd(
                    base_min_order_size_usd,
                    close_trigger,
                )
                rapid_close_exit = _is_rapid_close_trigger(close_trigger)

                # ── Resolve advanced exit policy ──────────────────────────
                # Prefer per-decision override on the strategy's ExitDecision;
                # else look up the strategy's per-trigger ``exit_policies``
                # map. ``_exit_instance`` and ``strategy_exit`` were set in
                # the strategy-exit block above (else: None / never raised).
                _resolved_policy = None
                _local_exit_instance = locals().get("_exit_instance")
                _local_strategy_exit = locals().get("strategy_exit")
                if _local_exit_instance is not None and hasattr(
                    _local_exit_instance, "resolve_exit_policy"
                ):
                    try:
                        _resolved_policy = _local_exit_instance.resolve_exit_policy(
                            _local_strategy_exit, str(close_trigger or "")
                        )
                    except Exception as _policy_exc:
                        logger.warning(
                            "Strategy resolve_exit_policy() error for %s: %s",
                            payload.get("strategy_type"),
                            _policy_exc,
                        )
                        _resolved_policy = None

                if _resolved_policy is not None and token_id and exit_size > 0:
                    _exit_id, _children_records = exit_executor.build_initial_children(
                        target_size=float(exit_size),
                        trigger_price=float(close_price) if close_price else 0.0,
                        side="SELL",
                        policy=_resolved_policy,
                        tick_size=exit_executor.DEFAULT_TICK_SIZE,
                    )
                    if _children_records:
                        _entry_filled_not, _entry_filled_sz, _entry_filled_px = (
                            _extract_live_fill_metrics(payload)
                        )
                        exit_record["exit_id"] = _exit_id
                        exit_record["children"] = _children_records
                        exit_record["policy"] = exit_executor.serialize_policy(_resolved_policy)
                        exit_record["target_size"] = float(exit_size)
                        exit_record["trigger_price"] = float(close_price or 0.0)
                        exit_record["entry_notional_usd"] = float(_entry_filled_not)
                        exit_record["entry_filled_size"] = float(_entry_filled_sz)
                        exit_record["status"] = "submitted"
                        exit_record["last_attempt_at"] = _iso_utc(now)
                        # First executor pass — submit the inside-most rungs.
                        _initial_budget_room = max(
                            0,
                            _live_exit_submission_cap()
                            - exit_submissions_this_pass[0],
                        )
                        if _initial_budget_room > 0:
                            from services.live_execution_adapter import (
                                execute_live_order as _ladder_first_place_order,
                            )

                            async def _ladder_first_cancel(_clob_id: str) -> bool:
                                async with release_conn(session):
                                    try:
                                        return bool(
                                            await live_execution_service.cancel_order(_clob_id)
                                        )
                                    except Exception:
                                        return False

                            try:
                                async with release_conn(session):
                                    await _prepare_sell_allowance_bounded(token_id)
                                    _first_report = await exit_executor.run_exit_pass(
                                        pending_exit=exit_record,
                                        token_id=token_id,
                                        side="SELL",
                                        min_order_size_usd=min_order_size_usd,
                                        place_order=_ladder_first_place_order,
                                        cancel_fn=_ladder_first_cancel,
                                        current_mid=close_price,
                                        tick_size=exit_executor.DEFAULT_TICK_SIZE,
                                        budget=exit_executor.PassBudget(
                                            submits=_initial_budget_room,
                                            escalations=_initial_budget_room,
                                            reprices=_initial_budget_room,
                                        ),
                                        now=now,
                                    )
                                exit_submissions_this_pass[0] += int(
                                    _first_report.get("submitted", 0)
                                    + _first_report.get("escalated", 0)
                                    + _first_report.get("repriced", 0)
                                )
                                logger.info(
                                    "Laddered exit initialized for order=%s trigger=%s children=%d submitted=%d",
                                    row.id,
                                    close_trigger,
                                    len(_children_records),
                                    int(_first_report.get("submitted", 0) or 0),
                                )
                            except Exception as _first_exc:
                                exit_record["last_error"] = _format_exit_error(_first_exc)
                                logger.warning(
                                    "Laddered exit initial pass exception for order=%s: %s",
                                    row.id,
                                    _format_exit_error(_first_exc),
                                    exc_info=_first_exc,
                                )
                        payload["pending_live_exit"] = exit_record
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                        held += 1
                        continue

                if token_id and exit_size > 0:
                    exit_size = _remaining_exit_size(
                        required_exit_size=exit_size,
                        pending_exit=exit_record,
                        wallet_position_size=wallet_exit_size_cap,
                    )
                    if exit_size > 0.0:
                        exit_record["remaining_size"] = float(exit_size)
                    exit_notional_estimate = float(exit_size) * float(max(close_price, 0.0))
                    if exit_notional_estimate + 1e-9 < min_order_size_usd:
                        exit_record["status"] = "blocked_min_notional"
                        exit_record["retry_count"] = 1
                        exit_record["exhausted_at"] = _iso_utc(now)
                        exit_record["last_error"] = (
                            f"exit_notional_below_min:{exit_notional_estimate:.4f}<{min_order_size_usd:.4f}"
                        )
                        exit_record["last_attempt_at"] = _iso_utc(now)
                        exit_record["next_retry_at"] = None
                        payload["pending_live_exit"] = exit_record
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                        held += 1
                        continue
                    if exit_submissions_this_pass[0] >= _live_exit_submission_cap():
                        # Defer: persist the exit_record as status="pending"
                        # (no provider id yet); the failed-exit retry branch
                        # will pick it up on the next cycle.  But first convert
                        # to status="failed" with retry_count=0 + a marker
                        # last_error so the retry path actually reaches it
                        # (status="pending" with no provider id is the stuck
                        # state we fixed in the grouped-exit commit).
                        exit_record["status"] = "failed"
                        exit_record["retry_count"] = 0
                        exit_record["last_error"] = "deferred_per_pass_cap"
                        exit_record["next_retry_at"] = None
                        payload["pending_live_exit"] = exit_record
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                        held += 1
                        continue
                    exit_submissions_this_pass[0] += 1
                    try:
                        from services.live_execution_adapter import execute_live_order

                        async with release_conn(session):
                            await _prepare_sell_allowance_bounded(token_id)
                            exec_result = await asyncio.wait_for(
                                execute_live_order(
                                    token_id=token_id,
                                    side="SELL",
                                    size=exit_size,
                                    fallback_price=close_price,
                                    min_order_size_usd=min_order_size_usd,
                                    time_in_force="IOC" if rapid_close_exit else "GTC",
                                    resolve_live_price=rapid_close_exit,
                                    enforce_fallback_bound=not rapid_close_exit,
                                ),
                                timeout=_LIVE_EXIT_ORDER_TIMEOUT_SECONDS,
                            )
                        if exec_result.status in {"executed", "open", "submitted"}:
                            exit_record["status"] = "submitted"
                            exit_record["exit_order_id"] = exec_result.order_id
                            exit_record["provider_clob_order_id"] = str(
                                (exec_result.payload or {}).get("clob_order_id") or ""
                            )
                            exit_record["last_attempt_at"] = _iso_utc(now)
                            logger.info(
                                "Exit order placed for order=%s trigger=%s status=%s",
                                row.id,
                                close_trigger,
                                exec_result.status,
                            )
                        else:
                            _apply_failed_exit_state(
                                exit_record,
                                error=exec_result.error_message,
                                now=now,
                                retry_count=1,
                            )
                            logger.warning(
                                "Exit order failed for order=%s trigger=%s error=%s",
                                row.id,
                                close_trigger,
                                exec_result.error_message,
                            )
                    except Exception as exc:
                        _apply_failed_exit_state(exit_record, error=exc, now=now, retry_count=1)
                        # Always include the exception class name so empty-str
                        # exceptions (e.g. ``Exception()``) don't trail off as
                        # "trigger=...: " with nothing after the colon, and
                        # attach exc_info so the traceback survives.
                        logger.warning(
                            "Exit order exception for order=%s trigger=%s: %s: %s",
                            row.id,
                            close_trigger,
                            type(exc).__name__,
                            str(exc) or "<no message>",
                            exc_info=exc,
                        )
                else:
                    exit_record["status"] = "failed"
                    exit_record["last_error"] = "missing token_id or fill_size"
                    exit_record["retry_count"] = 1
                    exit_record["next_retry_at"] = _iso_utc(
                        now + timedelta(seconds=_failed_exit_retry_delay_seconds(exit_record["last_error"]))
                    )

                payload["pending_live_exit"] = exit_record
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        # NOTE: No simulation ledger interaction for live positions.
        # Live positions settle against real exchange state.
        total_realized_pnl += pnl
        by_status[next_status] = int(by_status.get(next_status, 0)) + 1
        detail["realized_pnl"] = pnl
        details.append(detail)
        would_close += 1

        if dry_run:
            continue

        row.status = next_status
        row.actual_profit = pnl
        row.updated_at = now
        payload["position_state"] = next_state
        payload["position_close"] = {
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "realized_pnl": pnl,
            "cost_basis_usd": cost_basis,
            "settlement_proceeds_usd": proceeds,
            "filled_size": filled_size,
            "filled_notional_usd": filled_notional,
            "market_tradable": market_tradable,
            "age_minutes": age_minutes,
            "closed_at": _iso_utc(now),
            "reason": reason,
        }
        if _exit_instance is not None and hasattr(_exit_instance, "record_trade_outcome"):
            try:
                _exit_instance.record_trade_outcome(won=next_status in {"closed_win", "resolved_win"})
            except Exception:
                pass

        reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
            session,
            row=row,
            payload=payload,
            signal_payload=signal_payload,
            market_info=market_info,
            close_trigger=close_trigger,
            realized_pnl=pnl,
            now=now,
        )
        if reverse_signal_id and reverse_source:
            reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)
        row.payload_json = payload
        if reason:
            if row.reason:
                row.reason = f"{row.reason} | {reason}:{close_trigger}"
            else:
                row.reason = f"{reason}:{close_trigger}"
        hot_state.record_order_resolved(
            trader_id=trader_id,
            mode=str(row.mode or ""),
            order_id=str(row.id or ""),
            market_id=str(row.market_id or ""),
            direction=str(row.direction or ""),
            source=str(row.source or ""),
            status=next_status,
            actual_profit=pnl,
            payload=payload,
            copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
        )
        closed += 1

    # Grouped exit: when any order in a (market, direction) group gets an exit
    # triggered, propagate the exit to all siblings in the same group so stacked
    # orders exit together.
    if not dry_run:
        order_groups: dict[tuple[str, str], list] = {}
        for row in candidates:
            gk = (str(row.market_id or "").strip(), str(row.direction or "").strip().lower())
            order_groups.setdefault(gk, []).append(row)
        for group_key, group_rows in order_groups.items():
            if len(group_rows) < 2:
                continue
            trigger_source: dict[str, Any] | None = None
            for row in group_rows:
                payload = dict(row.payload_json or {})
                pe = payload.get("pending_live_exit")
                if isinstance(pe, dict) and pe.get("status") in ("pending", "submitted"):
                    trigger_source = pe
                    break
                if row.status in ("closed_win", "closed_loss", "resolved_win", "resolved_loss"):
                    pc = payload.get("position_close", {})
                    trigger_source = {
                        "close_trigger": str(pc.get("close_trigger") or "grouped_exit"),
                        "close_price": pc.get("close_price"),
                        "price_source": pc.get("price_source"),
                    }
                    break
            if trigger_source is None:
                continue
            for row in group_rows:
                if row.status not in LIVE_ACTIVE_STATUSES:
                    continue
                payload = dict(row.payload_json or {})
                existing_pe = payload.get("pending_live_exit")
                # Skip if the sibling already has any exit state the standard
                # reconcile paths are handling.  Previously we only skipped
                # "pending"/"submitted"/"filled", which caused grouped-exit to
                # overwrite a sibling that was mid-retry ("failed" with
                # retry_count>0), resetting its progress on every cycle.
                if isinstance(existing_pe, dict) and existing_pe.get("status") in (
                    "pending",
                    "submitted",
                    "filled",
                    "failed",
                    "blocked_min_notional",
                    "blocked_retry_exhausted",
                    "blocked_retry_exhausted_hard",
                    "superseded_resolution",
                    "superseded_manual_sell",
                ):
                    continue
                token_id = _extract_live_token_id(payload)
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                exit_size = _fill_sz if _fill_sz > 0 else ((_fill_not / _fill_px) if _fill_px and _fill_px > 0 else 0.0)
                if exit_size <= 0.0:
                    # Zero-fill sibling (entry was cancelled before any fill).
                    # Do not stamp a phantom pending_live_exit that can never be
                    # submitted — it would leave the row stuck at status=pending
                    # with retry_count=0 forever.  Reconciliation will settle the
                    # row's terminal status through its own path.
                    continue
                source_trigger = str(trigger_source.get("close_trigger") or "grouped_exit")
                # Write as status="failed" with retry_count=0 so the failed-exit
                # retry block picks this up on the next cycle and actually submits
                # via execute_live_order.  The original bug was stamping status=
                # "pending" without submitting, which left the row orphaned (the
                # pending/submitted reconciliation branch assumes a provider order
                # id already exists).
                payload["pending_live_exit"] = {
                    "status": "failed",
                    "close_trigger": f"grouped:{source_trigger}" if not source_trigger.startswith("grouped:") else source_trigger,
                    "close_price": trigger_source.get("close_price"),
                    "price_source": trigger_source.get("price_source"),
                    "token_id": token_id,
                    "exit_size": float(exit_size),
                    "triggered_at": _iso_utc(now),
                    "reason": "Grouped exit: sibling order in same market exited",
                    "retry_count": 0,
                    "last_error": "grouped_exit_pending_submit",
                }
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1

    # Capture the final candidate's elapsed (the loop boundary based
    # tracker only records on the *next* iteration, so the last row
    # would be missed without this).
    if _last_iter_row_id is not None:
        _per_candidate_timings.append(
            (_last_iter_row_id, _time.monotonic() - _last_iter_start)
        )
    _lc_t3 = _time.monotonic()

    if not dry_run and (closed > 0 or state_updates > 0):
        touched_by_id: dict[str, TraderOrder] = {}
        for row in candidates:
            if row.updated_at == now:
                touched_by_id[str(row.id or "")] = row
        for row in extra_touched_rows:
            touched_by_id[str(row.id or "")] = row
        touched_rows = [row for row_id, row in touched_by_id.items() if row_id]
        publish_rows = list(touched_rows)
        if touched_rows:
            _lc_t3a = _time.monotonic()
            # Roadmap #3 — bound the bulk UPDATE's worst case under
            # contention with concurrent TraderOrder writers (orchestrator
            # submission path, audit flush). Without this, the executemany
            # below could wait the full statement_timeout (30s) on row
            # locks held by another writer; production logs showed
            # "executemany=17.8s(n=7)" repeatedly. lock_timeout=2s makes
            # the UPDATE fail-fast on contention; the next reconcile
            # cycle will retry. SET LOCAL is scoped to this transaction
            # and reverts on commit/rollback.
            await session.execute(_sa_text("SET LOCAL lock_timeout = '2000'"))
            stmt = (
                update(TraderOrder)
                .where(TraderOrder.id == bindparam("_id"))
                .values(
                    status=bindparam("_status"),
                    actual_profit=bindparam("_actual_profit"),
                    notional_usd=bindparam("_notional_usd"),
                    entry_price=bindparam("_entry_price"),
                    effective_price=bindparam("_effective_price"),
                    verification_status=bindparam("_verification_status"),
                    verification_source=bindparam("_verification_source"),
                    updated_at=bindparam("_updated_at"),
                    payload_json=bindparam("_payload_json"),
                    reason=bindparam("_reason"),
                )
                .execution_options(synchronize_session=None)
            )
            params = [
                {
                    "id": row.id,
                    "_id": row.id,
                    "_status": row.status,
                    "_actual_profit": row.actual_profit,
                    "_notional_usd": row.notional_usd,
                    "_entry_price": row.entry_price,
                    "_effective_price": row.effective_price,
                    "_verification_status": row.verification_status,
                    "_verification_source": row.verification_source,
                    "_updated_at": row.updated_at,
                    "_payload_json": row.payload_json,
                    "_reason": row.reason,
                }
                for row in touched_rows
            ]
            for row in touched_rows:
                if inspect(row).session is session.sync_session:
                    session.sync_session.expunge(row)
            # asyncpg raises LockNotAvailableError when another writer
            # holds the row lock past lock_timeout. The next reconcile
            # cycle re-reads venue truth and retries these rows, so treat
            # that contention as deferred work rather than a hard failure.
            try:
                with session.no_autoflush:
                    await session.execute(stmt, params)
                _lc_t3b = _time.monotonic()
                await session.commit()
            except Exception as _bulk_exc:
                _bulk_orig = getattr(_bulk_exc, "orig", None)
                _bulk_cause = getattr(_bulk_exc, "__cause__", None)
                _bulk_context = getattr(_bulk_exc, "__context__", None)
                _bulk_sqlstate = str(
                    getattr(_bulk_orig, "sqlstate", "")
                    or getattr(_bulk_cause, "sqlstate", "")
                    or getattr(_bulk_context, "sqlstate", "")
                    or getattr(_bulk_exc, "sqlstate", "")
                    or ""
                ).strip()
                _bulk_message = " ".join(
                    str(part or "")
                    for part in (
                        type(_bulk_exc).__name__,
                        _bulk_exc,
                        type(_bulk_orig).__name__ if _bulk_orig is not None else "",
                        _bulk_orig,
                        _bulk_cause,
                        _bulk_context,
                    )
                ).lower()
                _is_lock_timeout = (
                    _bulk_sqlstate == "55P03"
                    or "locknotavailable" in _bulk_message
                    or "lock not available" in _bulk_message
                    or "lock timeout" in _bulk_message
                    or "lock_timeout" in _bulk_message
                    or "canceling statement due to lock timeout" in _bulk_message
                )
                try:
                    await session.rollback()
                except Exception:
                    pass
                if _is_lock_timeout:
                    logger.warning(
                        "reconcile_live_positions bulk UPDATE hit lock_timeout; "
                        "next reconcile cycle will retry",
                        extra={"trader_id": trader_id, "rows": len(touched_rows)},
                    )
                    return {
                        "trader_id": trader_id,
                        "mode": "live",
                        "dry_run": bool(dry_run),
                        "matched": len(candidates),
                        "would_close": would_close,
                        "closed": 0,
                        "held": held,
                        "skipped": skipped + closed,
                        "state_updates": 0,
                        "total_realized_pnl": 0.0,
                        "by_status": by_status,
                        "skipped_reasons": {**skipped_reasons, "lock_timeout": closed + state_updates},
                        "reverse_signals_emitted": 0,
                        "details": details,
                        "deferred_due_to_lock_timeout": True,
                    }
                raise
            _lc_t3c = _time.monotonic()
            _total_elapsed = _lc_t3c - _lc_t0
            # Top-3 slowest candidates (by elapsed wall time inside the
            # processing loop).  Surfacing the order ids tells us which
            # rows hit which expensive branch — the data we need to
            # decide whether the 20s ``processing=`` is concentrated in
            # a few outliers or spread evenly.
            _slowest = sorted(_per_candidate_timings, key=lambda item: item[1], reverse=True)[:3]
            _slowest_str = ", ".join(f"{rid[:8]}={dt:.2f}s" for rid, dt in _slowest)
            _timing_msg = (
                f"  lifecycle_detail: pre_io={_lc_t1 - _lc_t0:.1f}s "
                f"external_io={_lc_t2 - _lc_t1:.1f}s processing={_lc_t3 - _lc_t2:.1f}s "
                f"executemany={_lc_t3b - _lc_t3a:.1f}s(n={len(touched_rows)}) "
                f"commit={_lc_t3c - _lc_t3b:.1f}s total={_total_elapsed:.1f}s "
                f"(candidates={len(candidates)} terminal_audit={len(terminal_rows)}"
                f" slowest=[{_slowest_str}])"
            )
            if _total_elapsed >= _RECONCILE_TIMING_WARN_SECONDS:
                logger.warning("reconcile_live_positions timing: %s", _timing_msg)
            else:
                logger.debug("reconcile_live_positions timing: %s", _timing_msg)
        else:
            await session.commit()
        await _publish_trader_order_updates(publish_rows)
        if reverse_signal_ids_by_source:
            await _publish_reverse_signal_batches(reverse_signal_ids_by_source, emitted_at=now)

    return {
        "trader_id": trader_id,
        "mode": "live",
        "dry_run": bool(dry_run),
        "matched": len(candidates),
        "would_close": would_close,
        "closed": closed,
        "held": held,
        "skipped": skipped,
        "state_updates": state_updates,
        "total_realized_pnl": total_realized_pnl,
        "by_status": by_status,
        "skipped_reasons": skipped_reasons,
        "reverse_signals_emitted": sum(len(ids) for ids in reverse_signal_ids_by_source.values()),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Event-driven exit evaluation
# ---------------------------------------------------------------------------
# In-memory registry of open orders by token_id for event-driven exit checks.
# Populated by the reconciliation worker safety net to keep the registry fresh.
_open_orders_by_token: dict[str, list[dict[str, Any]]] = {}


def register_open_orders(token_id: str, orders: list[dict[str, Any]]) -> None:
    _open_orders_by_token[token_id] = orders


def unregister_token(token_id: str) -> None:
    _open_orders_by_token.pop(token_id, None)


def get_registered_token_ids() -> list[str]:
    return list(_open_orders_by_token.keys())


async def evaluate_exit_for_token(
    token_id: str,
    mid_price: float,
    bid: float,
    ask: float,
) -> dict[str, Any]:
    """Event-driven exit evaluation triggered by significant price changes.

    Called by the FeedManager on_change callback (>0.5% price move) for tokens
    that have open positions. This replaces the polling-based exit checks for
    latency-sensitive triggers (stop-loss, take-profit, trailing stop).

    Returns a summary dict with orders that triggered exit conditions.
    No heavy DB work is performed here; the actual exit submission is delegated
    to the reconciliation safety-net or a direct live execution call.
    """
    orders = _open_orders_by_token.get(token_id)
    if not orders:
        return {"token_id": token_id, "evaluated": 0, "triggered": []}

    now = utcnow()
    triggered: list[dict[str, Any]] = []

    for order in orders:
        order_id = str(order.get("order_id") or "").strip()
        if not order_id:
            continue

        entry_price = safe_float(order.get("entry_price"))
        if entry_price is None or entry_price <= 0:
            continue

        # Skip orders that already have a pending exit in flight
        if order.get("has_pending_exit"):
            continue

        take_profit_pct = safe_float(order.get("take_profit_pct"))
        stop_loss_pct = safe_float(order.get("stop_loss_pct"))
        trailing_stop_pct = safe_float(order.get("trailing_stop_pct"))
        min_hold_minutes = max(0.0, safe_float(order.get("min_hold_minutes")) or 0.0)
        highest_price = safe_float(order.get("highest_price"))

        age_anchor_iso = order.get("age_anchor")
        age_minutes: Optional[float] = None
        if age_anchor_iso:
            age_anchor = _parse_iso_utc_naive(age_anchor_iso)
            if age_anchor is not None:
                now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo else now
                age_minutes = max(0.0, (now_naive - age_anchor).total_seconds() / 60.0)

        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes
        if not min_hold_passed:
            continue

        pnl_pct = ((mid_price - entry_price) / entry_price) * 100.0

        close_trigger: Optional[str] = None

        if take_profit_pct is not None and pnl_pct >= take_profit_pct:
            close_trigger = "take_profit"
        elif stop_loss_pct is not None and pnl_pct <= -abs(stop_loss_pct):
            close_trigger = "stop_loss"
        elif (
            trailing_stop_pct is not None
            and trailing_stop_pct > 0
            and highest_price is not None
            and highest_price > entry_price
        ):
            # Update highest price in the registry opportunistically
            if mid_price > highest_price:
                order["highest_price"] = mid_price
                highest_price = mid_price
            trailing_trigger = highest_price * (1.0 - (trailing_stop_pct / 100.0))
            if mid_price <= trailing_trigger:
                close_trigger = "trailing_stop"

        if close_trigger is not None:
            triggered.append({
                "order_id": order_id,
                "trader_id": order.get("trader_id", ""),
                "token_id": token_id,
                "close_trigger": close_trigger,
                "mid_price": mid_price,
                "bid": bid,
                "ask": ask,
                "entry_price": entry_price,
                "pnl_pct": round(pnl_pct, 4),
                "highest_price": highest_price,
                "age_minutes": round(age_minutes, 2) if age_minutes is not None else None,
            })

    return {
        "token_id": token_id,
        "evaluated": len(orders),
        "triggered": triggered,
    }
