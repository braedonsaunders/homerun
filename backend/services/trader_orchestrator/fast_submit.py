"""Single-leg direct submission path for fast-tier traders.

``execute_fast_signal`` is the narrow fast-path alternative to the full
``ExecutionSessionEngine.execute_signal`` orchestration.  It is designed
for ``latency_class='fast'`` traders that trade one leg on one market at
a time (sub-second crypto binaries are the canonical use case).

Key differences vs. SessionEngine:

* No ``execution_sessions`` / ``execution_session_orders`` /
  ``execution_session_legs`` / ``execution_session_events`` rows.
* No pre-submit placeholder with a 45s ack timeout.
* No leg-wave / reprice orchestration — one attempt, one result.
* A single short DB transaction writes one ``TraderOrder`` row and its
  verification event; inventory rebuilds are left to the slower workers.

The low-level submission to the provider still flows through the
well-tested ``submit_execution_leg`` primitive in ``order_manager``, so
every token-resolution, buy pre-submit gate, shadow-mode microstructure-fill and
live-provider submission path remains identical to the orchestrated one.

A fast trader *must* be single-leg single-market.  If the signal has no
``positions_to_take`` or multiple positions, we refuse and return a
``blocked`` result — the trader config is the bug, not the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import TraderOrder, release_conn
import services.trader_hot_state as hot_state
from services.execution_latency_metrics import execution_latency_metrics
from services.trader_order_verification import (
    TRADER_ORDER_VERIFICATION_LOCAL,
    append_trader_order_verification_event,
)
from services.trader_orchestrator.fast_idempotency import derive_fast_idempotency_key
from services.trader_orchestrator.order_manager import LegSubmitResult, submit_execution_leg
from services.signal_bus import set_trade_signal_status
from services.trader_orchestrator_state import (
    _is_active_order_status,
    _now,
    build_trader_order_row,
    create_trader_decision,
    record_signal_consumption,
    upsert_trader_signal_cursor,
)
from utils.converters import safe_float
from utils.logger import get_logger
from utils.signal_helpers import normalize_position_side
from utils.utcnow import utcnow

# Marker key + values stored in TraderOrder.payload_json so we don't need a
# new status enum value (which would ripple through every status filter in
# trader_orchestrator_state). Reading these values is the cleanup-sweep /
# reconcile job's responsibility — the in-flight skeleton looks like a
# normal "submitted" row to existing code.
_SUBMISSION_STATE_KEY = "fast_submission_state"
_SUBMISSION_STATE_IN_FLIGHT = "in_flight"
_SUBMISSION_STATE_COMPLETED = "completed"
_SUBMISSION_STATE_CLOB_RAISED = "clob_exception"
_SUBMISSION_STATE_POST_UPDATE_FAILED = "post_update_failed"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    return parsed

logger = get_logger(__name__)


@dataclass
class FastSubmitResult:
    """Shape-compatible with ``SessionExecutionResult`` for downstream consumers."""

    session_id: str  # always "" for the fast path (no session was created)
    status: str
    effective_price: float | None
    error_message: str | None
    orders_written: int
    payload: dict[str, Any]
    created_orders: list[dict[str, Any]] = field(default_factory=list)


def _extract_single_position(signal: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Pull the single leg's position dict off a trade signal payload.

    Returns ``(position, error)`` — exactly one of the two is non-None.
    """
    payload = getattr(signal, "payload_json", None)
    if not isinstance(payload, dict):
        return None, "Signal has no payload_json — fast path requires positions_to_take[0]."
    positions = payload.get("positions_to_take") or payload.get("positions")
    if not isinstance(positions, list) or not positions:
        return None, "Signal has no positions_to_take — fast path requires exactly one leg."
    if len(positions) > 1:
        # Fast path is single-leg by design.  A multi-leg signal belongs on the
        # session_engine path (latency_class='normal' or 'slow').  Refusing
        # here is safer than silently dropping legs.
        return None, (
            f"Fast path refuses multi-leg signal (got {len(positions)} legs). "
            "Set trader latency_class to 'normal' or split the strategy."
        )
    pos = positions[0]
    if not isinstance(pos, dict):
        return None, "Signal positions_to_take[0] is not a dict."
    return pos, None


def _leg_from_position(position: dict[str, Any], signal: Any) -> dict[str, Any]:
    """Build the ``leg`` dict that ``submit_execution_leg`` expects."""
    action = str(position.get("action") or position.get("side") or "").strip()
    side = normalize_position_side(action)
    outcome = str(position.get("outcome") or "").strip().upper()
    market_id = str(position.get("market_id") or "").strip() or str(getattr(signal, "market_id", "") or "").strip()
    price = safe_float(position.get("price"), None)
    token_id = str(position.get("token_id") or "").strip() or None
    return {
        "leg_id": f"fast-{getattr(signal, 'id', 'unknown')}-0",
        "leg_index": 0,
        "market_id": market_id,
        "market_question": position.get("market_question") or getattr(signal, "market_question", None),
        "outcome": outcome or None,
        "side": side,
        "token_id": token_id,
        "price": price,
        # Fast-tier defaults: aggressive taker-limit crossing the book for an
        # immediate fill or nothing.  No post-only, no chase, no reprice.
        "price_policy": "taker_limit",
        "time_in_force": "FAK",
        "post_only": False,
        "metadata": {"fast_tier": True},
    }


def _result_payload_for_trader_order(
    *,
    leg_result: LegSubmitResult,
    now_iso: str,
) -> dict[str, Any]:
    """Build the TraderOrder.payload_json for a fast-tier fill.

    Must populate the fields that downstream consumers read:
    * ``provider_reconciliation.snapshot`` — used by
      ``_extract_live_fill_metrics`` for realized P&L and fill tracking.
    * ``position_state`` — used by ``reconcile_live_positions`` for mark
      price bookkeeping when the position is later closed.
    """
    base_payload = dict(leg_result.payload or {})
    filled_size = safe_float(leg_result.shares, 0.0) or 0.0
    filled_notional = safe_float(leg_result.notional_usd, 0.0) or 0.0
    avg_price = leg_result.effective_price if leg_result.effective_price is not None else None

    # Mirror into provider_reconciliation so reconciler + position_lifecycle
    # pick up the fill immediately without waiting for an async reconcile pass.
    snapshot: dict[str, Any] = {
        "filled_size": filled_size,
        "average_fill_price": avg_price,
        "filled_notional_usd": filled_notional,
        "normalized_status": leg_result.status,
    }
    if leg_result.provider_clob_order_id:
        snapshot["clob_order_id"] = leg_result.provider_clob_order_id
    if leg_result.provider_order_id:
        snapshot["provider_order_id"] = leg_result.provider_order_id

    base_payload.setdefault("provider_reconciliation", {})
    base_payload["provider_reconciliation"].update({"snapshot": snapshot})

    base_payload.setdefault(
        "position_state",
        {
            "last_mark_price": avg_price,
            "last_mark_source": "fast_submit_fill",
            "last_marked_at": now_iso,
        },
    )

    if leg_result.provider_order_id:
        base_payload.setdefault("provider_order_id", leg_result.provider_order_id)
    if leg_result.provider_clob_order_id:
        base_payload.setdefault("provider_clob_order_id", leg_result.provider_clob_order_id)

    base_payload["fast_tier"] = True
    return base_payload


async def execute_fast_signal(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: Any,
    decision_id: str | None,
    decision_audit: dict[str, Any] | None = None,
    strategy_key: str | None,
    strategy_version: int | None,
    strategy_params: dict[str, Any] | None,
    mode: str,
    size_usd: float,
    reason: str | None,
) -> FastSubmitResult:
    """Fast-tier single-leg direct submit.

    Writes exactly one ``TraderOrder`` row plus its verification event.
    It deliberately bypasses ``create_trader_order`` because that general
    helper performs a full inventory rebuild, which is too expensive for
    the fast-tier pool.
    """
    now = _now()
    now_iso = now.isoformat()

    mode_key = str(mode or "").strip().lower() or "shadow"
    notional = float(max(0.0, safe_float(size_usd, 0.0) or 0.0))

    # ----- Idempotency guard --------------------------------------------------
    # If a TraderOrder already exists for this (trader_id, signal_id), the
    # signal was previously submitted — typically a crash/retry where the
    # cursor advance silently failed (advance_fast_trader_cursor swallows
    # errors so a stale cursor doesn't block future trades). Refusing here
    # blocks the "DB row exists but cursor wasn't advanced" duplicate case.
    #
    # NOTE: this check does NOT protect against the rarer "CLOB order
    # placed but DB row never written" case (process killed between
    # submit_execution_leg succeeding and session.flush()). Closing that
    # gap requires a deterministic provider client_order_id derived from
    # the signal so a restart can reconcile against the provider; tracked
    # as a follow-up — see ``docs/fast-lane-idempotency.md``.
    # ----- Fast-tier risk cap ------------------------------------------------
    # Fix JJ: the fast tier was bypassing the risk_manager entirely, so a
    # configured ``max_open_orders`` cap (visible in the trader-config
    # UI) had ZERO effect on this code path.  Production observed 8
    # open orders against a 2-order cap.  Apply a single-row count
    # check here using the trader's persisted risk_limits.  The query
    # is bounded to active statuses and indexed on ``trader_id``, so
    # it adds a few ms — far cheaper than a runaway-order-fan.  Skip
    # the check entirely if no limit is configured.
    if isinstance(strategy_params, dict):
        # ``strategy_params`` is the per-source config passed in by the
        # fast trader runtime; it carries whatever risk overrides the
        # operator put on the source-config row.  Fall through to the
        # trader-wide risk_limits otherwise.
        risk_open_cap = strategy_params.get("max_open_orders")
    else:
        risk_open_cap = None
    if risk_open_cap is None:
        # Pull from the trader's risk_limits_json, lazy-fetched once
        # per call — small cost on the few-ms scale.
        try:
            from models.database import Trader
            trader_row = await session.get(Trader, trader_id)
            if trader_row is not None:
                trader_risk = getattr(trader_row, "risk_limits_json", None) or {}
                if isinstance(trader_risk, str):
                    import json as _json
                    try:
                        trader_risk = _json.loads(trader_risk)
                    except Exception:
                        trader_risk = {}
                risk_open_cap = trader_risk.get("max_open_orders") if isinstance(trader_risk, dict) else None
        except Exception:
            risk_open_cap = None
    try:
        risk_open_cap_int = int(risk_open_cap) if risk_open_cap is not None else None
    except Exception:
        risk_open_cap_int = None
    if risk_open_cap_int is not None and risk_open_cap_int > 0:
        from sqlalchemy import func as _sa_func
        active_statuses = ("submitted", "open", "partial", "pending", "working")
        open_count = (
            await session.execute(
                select(_sa_func.count(TraderOrder.id))
                .where(TraderOrder.trader_id == trader_id)
                .where(TraderOrder.status.in_(active_statuses))
            )
        ).scalar_one() or 0
        if open_count >= risk_open_cap_int:
            logger.info(
                "Fast-tier blocking submission: trader at max_open_orders cap",
                trader_id=trader_id,
                open_count=open_count,
                max_open_orders=risk_open_cap_int,
            )
            return FastSubmitResult(
                session_id="",
                status="skipped",
                effective_price=None,
                error_message=f"max_open_orders cap reached ({open_count}/{risk_open_cap_int})",
                orders_written=0,
                payload={
                    "fast_tier": True,
                    "reason": "max_open_orders_cap",
                    "open_count": open_count,
                    "cap": risk_open_cap_int,
                },
            )

    signal_id = str(getattr(signal, "id", "") or "").strip()
    if signal_id:
        existing_id = (
            await session.execute(
                select(TraderOrder.id)
                .where(TraderOrder.trader_id == trader_id)
                .where(TraderOrder.signal_id == signal_id)
                .limit(1)
            )
        ).scalar_one_or_none()
        if existing_id:
            logger.info(
                "Fast-tier refusing duplicate submission",
                trader_id=trader_id,
                signal_id=signal_id,
                existing_order_id=str(existing_id),
            )
            return FastSubmitResult(
                session_id="",
                status="skipped",
                effective_price=None,
                error_message=f"trader_order already exists for signal {signal_id}",
                orders_written=0,
                payload={
                    "fast_tier": True,
                    "reason": "duplicate_signal_existing_order",
                    "existing_trader_order_id": str(existing_id),
                },
            )

    position, parse_error = _extract_single_position(signal)
    if parse_error is not None or position is None:
        return FastSubmitResult(
            session_id="",
            status="failed",
            effective_price=None,
            error_message=parse_error,
            orders_written=0,
            payload={"fast_tier": True, "reason": "bad_signal_shape"},
        )

    leg = _leg_from_position(position, signal)

    # ----- Deterministic idempotency key --------------------------------------
    # Derived from (trader_id, signal_id) so a retry produces the same key.
    # Stamped into the venue's order metadata field AND onto the skeleton
    # row's payload_json so the orphan-reconcile sweep can match a venue
    # order back to its TraderOrder when the post-submit flush is lost.
    #
    # Note on the field name: we set ``leg["clob_idempotency_key"]`` rather
    # than overloading ``leg["metadata"]`` (which is the
    # ExecutionPlan-bookkeeping dict consumed by readers like
    # ``order_manager._resolve_execution_price_bounds``). Conflating those
    # two meanings is what produced the production hex-error pattern; the
    # dedicated field ends that overload for good.
    idempotency_key = derive_fast_idempotency_key(
        trader_id=trader_id,
        signal_id=signal_id,
    )
    if mode_key == "live" and idempotency_key:
        leg = {**leg, "clob_idempotency_key": idempotency_key}

    # ----- Pre-submit skeleton row (idempotency lock) -------------------------
    # Write a TraderOrder row marked in-flight BEFORE the CLOB submission so
    # that if the process dies between CLOB success and the post-submit DB
    # update, the next runtime cycle's duplicate-check guard sees this row
    # and refuses to re-submit. The row is mutated to its final state on
    # CLOB return — we never INSERT twice. The marker lives in payload_json
    # so existing status filters (UNFILLED_ORDER_STATUSES, cleanup sweeps)
    # continue to work unchanged.
    pre_submit_payload: dict[str, Any] = {
        "fast_tier": True,
        _SUBMISSION_STATE_KEY: _SUBMISSION_STATE_IN_FLIGHT,
        "pre_submit_at_iso": now_iso,
        "fast_idempotency_key": idempotency_key,
    }
    # Persist the runtime strategy_params so the UI's per-bot performance
    # view can attribute historical orders to the exact configuration that
    # produced them (rather than falling back to the trader's *current*
    # config, which would drift after every retune).  The session_engine
    # path persists this under ``payload["strategy_params"]``; mirror that
    # here so fast-tier orders show up the same way.
    if strategy_params:
        pre_submit_payload["strategy_params"] = dict(strategy_params)
    try:
        if decision_audit is not None and decision_id:
            await create_trader_decision(
                session,
                decision_id=decision_id,
                trader_id=trader_id,
                signal=signal,
                strategy_key=strategy_key or str(getattr(signal, "strategy_type", "") or ""),
                strategy_version=strategy_version,
                decision=str(decision_audit.get("decision") or "selected"),
                reason=decision_audit.get("reason"),
                score=decision_audit.get("score"),
                checks_summary=decision_audit.get("checks_summary"),
                risk_snapshot=decision_audit.get("risk_snapshot"),
                payload=decision_audit.get("payload"),
                trace_id=decision_audit.get("trace_id"),
                commit=False,
            )
        order = build_trader_order_row(
            trader_id=trader_id,
            signal=signal,
            decision_id=decision_id,
            strategy_key=strategy_key,
            strategy_version=strategy_version,
            mode=mode_key,
            status="submitted",
            notional_usd=notional,
            effective_price=None,
            reason=reason,
            payload=pre_submit_payload,
        )
        session.add(order)
        # Commit the skeleton so the lock is durable across the CLOB call.
        # ``release_conn`` (used below to free the pool slot during the
        # network I/O) calls ``session.reset()`` which would otherwise drop
        # any unflushed/uncommitted state. Committing here is the price of
        # crash-survivability — without it, a process kill mid-CLOB leaves
        # the venue with an order and our DB with nothing, defeating the
        # idempotency guard.
        await session.commit()
    except Exception as exc:
        try:
            await session.rollback()
        except Exception as rollback_exc:
            logger.debug("Fast-tier pre-submit rollback failed", trader_id=trader_id, exc_info=rollback_exc)
        logger.error("Fast-tier pre-submit row write failed", trader_id=trader_id, exc_info=exc)
        return FastSubmitResult(
            session_id="",
            status="failed",
            effective_price=None,
            error_message=f"fast pre-submit raised: {type(exc).__name__}: {exc}",
            orders_written=0,
            payload={"fast_tier": True, "reason": "pre_submit_persist_failed"},
        )
    pre_submit_order_id = str(order.id)

    # ----- CLOB submission ----------------------------------------------------
    # Release the DB connection (if the session has one checked out)
    # while the external CLOB submission is in flight.  The submission
    # is pure network IO and does not touch ``session``; holding a
    # fast-tier pool slot across a 300-500ms upstream roundtrip both
    # starves the pool and creates a window in which an outer
    # cancellation can tear the asyncpg extended-protocol state in
    # half (the ``cannot switch to state X; another operation in
    # progress`` pattern).  ``release_conn`` is a no-op when the
    # session is still lazy.
    submit_started_at = utcnow()
    try:
        async with release_conn(session):
            leg_result = await submit_execution_leg(
                mode=mode_key,
                signal=signal,
                leg=leg,
                notional_usd=notional,
                strategy_params=strategy_params,
            )
    except Exception as exc:
        logger.warning(
            "Fast-tier leg submit raised",
            trader_id=trader_id,
            pre_submit_order_id=pre_submit_order_id,
            exc_info=exc,
        )
        # CLOB call raised — we don't know if the venue accepted the order.
        # Mark the skeleton row as failed-with-clob-exception so the reconcile
        # sweep can match it against any orders the venue actually has, and
        # so the duplicate-check guard still blocks re-submission.
        try:
            # ``release_conn`` detached the pre-submit ``order`` ORM object;
            # re-attach it before mutating so the UPDATE actually flushes.
            refetched_after_raise = await session.get(TraderOrder, pre_submit_order_id)
            target = refetched_after_raise if refetched_after_raise is not None else order
            target.status = "failed"
            target.error_message = f"submit_execution_leg raised: {type(exc).__name__}: {exc}"
            target.payload_json = {
                **(target.payload_json or {}),
                _SUBMISSION_STATE_KEY: _SUBMISSION_STATE_CLOB_RAISED,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            await session.flush()
        except Exception as flush_exc:
            logger.debug(
                "Fast-tier flush of clob-exception marker failed",
                trader_id=trader_id,
                exc_info=flush_exc,
            )
        return FastSubmitResult(
            session_id="",
            status="failed",
            effective_price=None,
            error_message=f"submit_execution_leg raised: {type(exc).__name__}: {exc}",
            orders_written=1,
            payload={
                "fast_tier": True,
                "reason": "submit_exception",
                "trader_order_id": pre_submit_order_id,
            },
            created_orders=[{"id": pre_submit_order_id}],
        )
    submit_completed_at = utcnow()

    # ``release_conn`` resets the session, which detaches the pre-submit
    # ``order`` ORM object. Re-attach by fetching it back so subsequent
    # attribute mutations propagate to an UPDATE on flush.
    #
    # The refetch can race with pool saturation under burst load (the
    # CLOB call held no DB connection but other consumers may have
    # filled the fast pool while we were on the wire).  If the get
    # raises a pool / asyncpg-state error the session is unusable for
    # the rest of this function, so we rollback and fall back to
    # building a transient row that mirrors the pre-submit state — the
    # CLOB-side trade is durable already (covered by the pre-submit
    # row's idempotency lock); this just keeps the post-submit update
    # path running rather than crashing the trader.
    try:
        refetched = await session.get(TraderOrder, pre_submit_order_id)
    except Exception as _refetch_exc:
        logger.warning(
            "Fast-tier post-submit refetch failed; will skip ORM update",
            trader_id=trader_id,
            pre_submit_order_id=pre_submit_order_id,
            exc_info=_refetch_exc,
        )
        try:
            await session.rollback()
        except Exception:
            pass
        refetched = None
    if refetched is not None:
        order = refetched

    # Map leg result status onto trader_order status.  submit_execution_leg
    # returns: executed | failed | skipped | cancelled.  "skipped" is a
    # pre-submit gate rejection (e.g. buy gate) and produces no order.
    leg_status = str(leg_result.status or "").strip().lower()
    if leg_status == "skipped":
        # The CLOB never received an order — the buy-gate (or similar) rejected
        # before submission. Repurpose the skeleton row to reflect that so we
        # don't leave a "submitted" ghost; cancelled is the closest taxonomy.
        try:
            order.status = "cancelled"
            order.error_message = leg_result.error_message
            order.payload_json = {
                **(order.payload_json or {}),
                _SUBMISSION_STATE_KEY: _SUBMISSION_STATE_COMPLETED,
                "reason": "pre_submit_gate",
                "leg": leg_result.payload,
            }
            await session.flush()
        except Exception as flush_exc:
            logger.debug(
                "Fast-tier flush of pre-submit-gate marker failed",
                trader_id=trader_id,
                exc_info=flush_exc,
            )
        return FastSubmitResult(
            session_id="",
            status="skipped",
            effective_price=leg_result.effective_price,
            error_message=leg_result.error_message,
            orders_written=1,
            payload={
                "fast_tier": True,
                "reason": "pre_submit_gate",
                "leg": leg_result.payload,
                "trader_order_id": pre_submit_order_id,
            },
            created_orders=[{"id": pre_submit_order_id}],
        )

    order_status = {
        "executed": "executed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(leg_status, "failed")

    order_payload = _result_payload_for_trader_order(leg_result=leg_result, now_iso=now_iso)
    order_payload[_SUBMISSION_STATE_KEY] = _SUBMISSION_STATE_COMPLETED
    order_payload["fast_tier"] = True
    order_payload["fast_idempotency_key"] = idempotency_key
    # Carry the runtime strategy_params forward into the post-submit row.
    # The post-submit update overwrites payload_json wholesale, so without
    # this the params we stamped onto the pre-submit row would be lost.
    if strategy_params:
        order_payload["strategy_params"] = dict(strategy_params)
    submit_completed_iso = submit_completed_at.isoformat()
    order_payload.setdefault("submit_started_at_iso", submit_started_at.isoformat())
    order_payload.setdefault("submit_completed_at_iso", submit_completed_iso)

    # ----- Post-submit update -------------------------------------------------
    # Mutate the same skeleton row in place (no second INSERT) so the
    # idempotency lock survives even if this update fails. Critically, we
    # do *not* roll back on failure — the CLOB has already executed, and
    # rolling back the lock would let the next runtime cycle re-submit.
    try:
        order.status = order_status
        order.notional_usd = leg_result.notional_usd if leg_result.notional_usd is not None else notional
        order.effective_price = leg_result.effective_price
        order.error_message = leg_result.error_message
        order.payload_json = order_payload
        order.provider_order_id = leg_result.provider_order_id or order.provider_order_id
        order.provider_clob_order_id = leg_result.provider_clob_order_id or order.provider_clob_order_id

        event_payload = dict(order.payload_json or {})
        event_payload.update({"status": str(order.status or ""), "mode": str(order.mode or "")})
        append_trader_order_verification_event(
            session,
            trader_order_id=pre_submit_order_id,
            verification_status=str(order.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
            source=order.verification_source,
            event_type="order_created",
            reason=order.verification_reason,
            provider_order_id=order.provider_order_id,
            provider_clob_order_id=order.provider_clob_order_id,
            execution_wallet_address=order.execution_wallet_address,
            tx_hash=order.verification_tx_hash,
            payload_json=event_payload,
            created_at=now,
        )
        await session.flush()
        if _is_active_order_status(mode_key, order_status):
            hot_state.upsert_active_order(
                trader_id=trader_id,
                mode=mode_key,
                order_id=pre_submit_order_id,
                status=order_status,
                market_id=str(order.market_id or ""),
                direction=str(order.direction or ""),
                source=str(order.source or ""),
                notional_usd=safe_float(leg_result.notional_usd, notional) or 0.0,
                entry_price=safe_float(leg_result.effective_price, safe_float(order.entry_price, 0.0)) or 0.0,
                token_id=str(leg.get("token_id") or ""),
                filled_shares=safe_float(leg_result.shares, 0.0) or 0.0,
                payload=order_payload,
            )
            # Wire the fresh order into PositionMarkState so the WS
            # ``position_marks_update`` push channel updates U-P&L on
            # the very next price tick (≤ 100 ms typically) instead
            # of waiting for the trader_reconciliation_worker to
            # register it on its 30 s cycle.  The slow orchestrator's
            # position-monitor loop normally does this; the fast tier
            # had to wait, leaving fresh fast-tier fills with $0
            # marks in the UI for up to half a minute.  Live mode
            # only — shadow trades have no real position to track.
            if mode_key == "live" and order_status == "executed":
                try:
                    fill_token_id = str(leg.get("token_id") or "").strip()
                    fill_price = (
                        safe_float(leg_result.effective_price, None)
                        or safe_float(order.entry_price, None)
                    )
                    fill_notional = safe_float(leg_result.notional_usd, notional) or 0.0
                    if fill_token_id and fill_price and fill_price > 0 and fill_notional > 0:
                        from services.position_mark_state import get_position_mark_state
                        from services.ws_feeds import get_feed_manager

                        pms = get_position_mark_state()
                        pms.register_position(
                            order_id=pre_submit_order_id,
                            market_id=str(order.market_id or ""),
                            token_id=fill_token_id,
                            direction=str(order.direction or "yes").strip().lower(),
                            entry_price=float(fill_price),
                            notional=float(fill_notional),
                            edge_percent=safe_float(order.edge_percent, 0.0) or 0.0,
                        )
                        # Subscribe the WS feed for this token if it
                        # isn't already subscribed — without this the
                        # PriceCache.on_update callbacks that drive
                        # ``pms.on_price_update`` never fire for the
                        # new token, so U-P&L stays at the initial
                        # entry-price-equals-mark (0 P&L) state.
                        try:
                            feed_manager = get_feed_manager()
                            if getattr(feed_manager, "_started", False):
                                # Fire-and-forget — subscribe is fast
                                # but we don't want to block the
                                # post-submit hot path on any WS
                                # network jitter.
                                import asyncio as _asyncio
                                _asyncio.create_task(
                                    feed_manager.polymarket_feed.subscribe([fill_token_id]),
                                    name=f"fast-submit-ws-subscribe-{fill_token_id[:12]}",
                                )
                        except Exception as _sub_exc:
                            logger.debug(
                                "Fast-tier WS subscribe after fill failed (non-fatal)",
                                token_id=fill_token_id,
                                exc_info=_sub_exc,
                            )
                except Exception as _pms_exc:
                    logger.debug(
                        "Fast-tier PositionMarkState registration failed (non-fatal)",
                        order_id=pre_submit_order_id,
                        exc_info=_pms_exc,
                    )
    except Exception as exc:
        # CLOB has already executed — DO NOT rollback. Mark the row so the
        # reconcile sweep knows post-update is incomplete and can repair
        # the missing fields against the venue snapshot.
        logger.error(
            "Fast-tier trader_order post-submit update failed",
            trader_id=trader_id,
            pre_submit_order_id=pre_submit_order_id,
            exc_info=exc,
        )
        try:
            order.payload_json = {
                **(order.payload_json or {}),
                _SUBMISSION_STATE_KEY: _SUBMISSION_STATE_POST_UPDATE_FAILED,
                "post_update_error_type": type(exc).__name__,
                "post_update_error_message": str(exc),
            }
            await session.flush()
        except Exception as flush_exc:
            logger.debug(
                "Fast-tier flush of post-update marker failed",
                trader_id=trader_id,
                exc_info=flush_exc,
            )
        return FastSubmitResult(
            session_id="",
            status="failed",
            effective_price=leg_result.effective_price,
            error_message=f"fast order post-update raised: {type(exc).__name__}: {exc}",
            orders_written=1,
            payload={
                "fast_tier": True,
                "reason": "post_update_failed",
                "trader_order_id": pre_submit_order_id,
            },
            created_orders=[{"id": pre_submit_order_id}],
        )

    # ----- Latency telemetry --------------------------------------------------
    # Hook into the existing execution_latency_metrics buffer so the fast
    # path shows up in the same SLO dashboard as the orchestrated slow
    # tier. We populate the stage keys this path can actually measure;
    # the slow-tier-only stages (wake_to_context_ready_ms etc.) stay None.
    try:
        signal_payload_dict = getattr(signal, "payload_json", None)
        signal_payload_dict = signal_payload_dict if isinstance(signal_payload_dict, dict) else {}
        emitted_at = (
            _parse_iso(str(signal_payload_dict.get("signal_emitted_at") or signal_payload_dict.get("ingested_at") or ""))
            or getattr(signal, "created_at", None)
        )
        if emitted_at is not None and getattr(emitted_at, "tzinfo", None) is None:
            from datetime import timezone as _tz
            emitted_at = emitted_at.replace(tzinfo=_tz.utc)

        def _delta_ms(start: Any, end: Any) -> int | None:
            if start is None or end is None:
                return None
            try:
                return max(0, int((end - start).total_seconds() * 1000))
            except Exception:
                return None

        latency_payload = {
            "submit_round_trip_ms": _delta_ms(submit_started_at, submit_completed_at),
            "emit_to_submit_start_ms": _delta_ms(emitted_at, submit_started_at),
        }
        await execution_latency_metrics.record(
            trader_id=trader_id,
            source=str(getattr(signal, "source", "") or ""),
            strategy_key=str(strategy_key or getattr(signal, "strategy_type", "") or ""),
            payload=latency_payload,
        )
    except Exception as exc:
        # Telemetry is best-effort; never let a metric failure roll back a
        # successful trade.
        logger.debug("Fast-tier latency record failed", trader_id=trader_id, exc_info=exc)

    return FastSubmitResult(
        session_id="",
        status=order_status,
        effective_price=leg_result.effective_price,
        error_message=leg_result.error_message,
        orders_written=1,
        payload={
            "fast_tier": True,
            "trader_order_id": pre_submit_order_id,
            "leg": leg_result.payload,
            "mode": mode_key,
            "submitted_at": now_iso,
            "submit_started_at_iso": submit_started_at.isoformat(),
            "submit_completed_at_iso": submit_completed_iso,
        },
        created_orders=[{"id": pre_submit_order_id}],
    )


async def advance_fast_trader_cursor(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: Any,
    decision_id: str | None,
    outcome: str,
    reason: str | None,
) -> None:
    """Advance the trader signal cursor + mark consumption after fast submit.

    Called by the fast-tier runtime once it has committed a TraderOrder so
    the signal is not re-evaluated on the next cycle.  Any DB error here
    is logged but swallowed — leaving the cursor stale is a recoverable
    annoyance, not a money-losing bug (duplicate submission is already
    gated by the occupancy check).
    """
    signal_id = str(getattr(signal, "id", "") or "").strip()
    if not signal_id:
        return
    try:
        await set_trade_signal_status(session, signal_id, outcome, commit=False)
    except Exception as exc:
        logger.debug("Fast-tier set_trade_signal_status failed", exc_info=exc)
    try:
        await record_signal_consumption(
            session,
            trader_id=trader_id,
            signal_id=signal_id,
            outcome=outcome,
            reason=reason,
            decision_id=decision_id,
            commit=False,
        )
    except Exception as exc:
        logger.debug("Fast-tier record_signal_consumption failed", exc_info=exc)
    try:
        created_at = getattr(signal, "created_at", None) or utcnow()
        runtime_sequence = getattr(signal, "runtime_sequence", None)
        normalized_runtime_sequence = int(runtime_sequence) if runtime_sequence is not None else None
        hot_state.update_signal_cursor(
            trader_id,
            "live",
            created_at,
            signal_id,
            normalized_runtime_sequence,
        )
        await upsert_trader_signal_cursor(
            session,
            trader_id=trader_id,
            last_signal_created_at=created_at,
            last_signal_id=signal_id,
            last_runtime_sequence=normalized_runtime_sequence,
            commit=False,
        )
    except Exception as exc:
        logger.debug("Fast-tier upsert_trader_signal_cursor failed", exc_info=exc)
