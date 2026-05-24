"""Single-leg direct submission path for fast-tier traders.

``execute_fast_signal`` is the narrow fast-path alternative to the full
``ExecutionSessionEngine.execute_signal`` orchestration.  It is designed
for ``latency_class='fast'`` traders that trade one leg on one market at
a time (sub-second crypto binaries are the canonical use case).

Journal-first durability (2026-05 cutover)
------------------------------------------
There is **no database write before the wire**.  The durable "about to
fire signal X" record that prevents a crash-restart double-submit is an
``fsync``'d append to the local :mod:`intent_journal` (tens of
microseconds), not a networked Postgres commit (whose tail spiked to
seconds under pool contention).  Dedup and the ``max_open_orders`` cap
are answered entirely in-memory (journal index + ``trader_hot_state``).
The ``TraderOrder`` row is written **after** the venue round-trip, so DB
latency never extends time-to-market.

Crash-recovery contract:

* ``record_intent`` (fsync) happens before the CLOB call.
* On a *definite* venue outcome we persist the order row, commit, then
  ``record_result`` to close the intent.
* On an *ambiguous* failure (timeout / cancel / raised — the venue may or
  may not have the order) we deliberately leave the intent OPEN and write
  no result.  Startup / periodic reconcile then queries the venue by the
  deterministic ``clob_idempotency_key`` and either backfills the row or
  closes the intent as failed.

Single-process ownership invariant: one fast-worker process owns a
disjoint set of traders, so the in-memory dedup index + per-trader
``asyncio.Lock`` are a complete guard.  Two processes owning one trader
is not a supported topology (the pre-cutover DB row didn't truly guard
that case either — the race was always closed by the in-process lock).

A fast trader *must* be single-leg single-market.  If the signal has no
``positions_to_take`` or multiple positions, we refuse and return a
``blocked`` result — the trader config is the bug, not the runtime.
"""

from __future__ import annotations

import asyncio as _asyncio
from dataclasses import dataclass, field
from typing import Any

from datetime import datetime, timedelta, timezone

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
from services.trader_orchestrator.intent_journal import get_intent_journal
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

# Marker key + values stored in TraderOrder.payload_json so existing
# status filters / reconcile sweeps can reason about a fast order's
# submission state without a dedicated status enum value.
_SUBMISSION_STATE_KEY = "fast_submission_state"
_SUBMISSION_STATE_COMPLETED = "completed"
_SUBMISSION_STATE_POST_PERSIST_FAILED = "post_persist_failed"

# Hard ceiling on the CLOB submit roundtrip from the fast tier.  The
# underlying client retries internally on transient disconnects, which
# under degraded CLOB health pushed the call to 30-47s — well past the
# fast trader's 3s hard cycle budget.  5s is conservative: typical
# successful submits land in 300-700ms; on timeout we fall through to
# the ambiguous-failure path (intent left open) and the orphan reconcile
# resolves it against the venue.
_FAST_LEG_SUBMIT_TIMEOUT_SECONDS = 5.0


# Per-trader async lock for the dedup-check + cap-check + intent-record
# critical section.  Without it, concurrent ``execute_fast_signal`` calls
# for the same trader can race the cap check.  The lock is per-trader so
# unrelated traders don't serialise, and it is RELEASED before the CLOB
# network call so the cap-bound serialisation doesn't extend over the
# wire round-trip.
_per_trader_submit_locks: dict[str, _asyncio.Lock] = {}


def _get_per_trader_submit_lock(trader_id: str) -> _asyncio.Lock:
    lock = _per_trader_submit_locks.get(trader_id)
    if lock is None:
        lock = _asyncio.Lock()
        _per_trader_submit_locks[trader_id] = lock
    return lock


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
    """Build the ``leg`` dict that ``submit_execution_leg`` expects.

    ``price_policy`` / ``time_in_force`` / ``post_only`` are driven by the
    strategy's position payload, falling back to fast-tier defaults
    (taker_limit / FAK / not post-only) only when the strategy didn't
    specify.
    """
    action = str(position.get("action") or position.get("side") or "").strip()
    side = normalize_position_side(action)
    outcome = str(position.get("outcome") or "").strip().upper()
    market_id = str(position.get("market_id") or "").strip() or str(getattr(signal, "market_id", "") or "").strip()
    price = safe_float(position.get("price"), None)
    token_id = str(position.get("token_id") or "").strip() or None

    # Strategy-driven execution policy with fast-tier fallbacks.  If the
    # strategy supplied an empty string we still fall through to the
    # default (the or-chain is intentional).
    price_policy = str(position.get("price_policy") or "").strip().lower() or "taker_limit"
    time_in_force = str(position.get("time_in_force") or "").strip().upper() or "FAK"
    post_only_raw = position.get("post_only")
    if post_only_raw is None:
        post_only = False
    else:
        post_only = bool(post_only_raw)

    return {
        "leg_id": f"fast-{getattr(signal, 'id', 'unknown')}-0",
        "leg_index": 0,
        "market_id": market_id,
        "market_question": position.get("market_question") or getattr(signal, "market_question", None),
        "outcome": outcome or None,
        "side": side,
        "token_id": token_id,
        "price": price,
        "price_policy": price_policy,
        "time_in_force": time_in_force,
        "post_only": post_only,
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


def _resolve_open_order_cap(
    strategy_params: dict[str, Any] | None,
    risk_limits: dict[str, Any] | None,
) -> int | None:
    """Resolve ``max_open_orders`` from in-memory config only (no DB).

    The runtime passes the trader's ``risk_limits`` in-memory, so the
    pre-cutover ``session.get(Trader, ...)`` DB fallback is gone — the cap
    check stays entirely off the wire.
    """
    raw = None
    if isinstance(strategy_params, dict):
        raw = strategy_params.get("max_open_orders")
    if raw is None and isinstance(risk_limits, dict):
        raw = risk_limits.get("max_open_orders")
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def _skip_result(reason: str, *, error_message: str | None = None, extra: dict[str, Any] | None = None) -> FastSubmitResult:
    payload: dict[str, Any] = {"fast_tier": True, "reason": reason}
    if extra:
        payload.update(extra)
    return FastSubmitResult(
        session_id="",
        status="skipped",
        effective_price=None,
        error_message=error_message,
        orders_written=0,
        payload=payload,
    )


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
    risk_limits: dict[str, Any] | None = None,
) -> FastSubmitResult:
    """Fast-tier single-leg direct submit (journal-first, DB after the wire).

    Pre-wire: dedup + cap (in-memory) → durable intent (fsync).  Wire:
    one ``submit_execution_leg`` attempt under a 5s ceiling.  Post-wire:
    one ``TraderOrder`` INSERT + commit, then close the journal intent.
    """
    now = _now()
    now_iso = now.isoformat()

    mode_key = str(mode or "").strip().lower() or "shadow"
    notional = float(max(0.0, safe_float(size_usd, 0.0) or 0.0))
    signal_id = str(getattr(signal, "id", "") or "").strip()
    journal = get_intent_journal()

    # ----- Parse the single leg (needed before recording intent) -------------
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

    # ----- Deterministic idempotency key -------------------------------------
    # Derived from (trader_id, signal_id) so a retry produces the same key.
    # Stamped into the venue's order metadata so the orphan-reconcile sweep
    # can match a venue order back to its intent when a post-wire DB write
    # is lost.
    idempotency_key = derive_fast_idempotency_key(trader_id=trader_id, signal_id=signal_id)
    if mode_key == "live" and idempotency_key:
        leg = {**leg, "clob_idempotency_key": idempotency_key}

    risk_open_cap_int = _resolve_open_order_cap(strategy_params, risk_limits)

    # ----- Critical section: dedup + cap + durable intent (NO DB) ------------
    # The per-trader lock serialises dedup+cap+intent so concurrent signals
    # for the same trader can't both pass the cap.  It is released BEFORE
    # the wire so the network round-trip is never serialised.
    _submit_lock = _get_per_trader_submit_lock(trader_id)
    await _submit_lock.acquire()
    _lock_released = False

    def _release_submit_lock_if_held() -> None:
        nonlocal _lock_released
        if _lock_released:
            return
        _lock_released = True
        try:
            _submit_lock.release()
        except RuntimeError:
            pass

    try:
        # Dedup: the journal is the authority (durable across restart via
        # its on-boot ``load``).  Any prior record for this signal — open
        # or resolved — blocks a re-fire.
        if signal_id and journal.has_intent(trader_id, signal_id):
            logger.info(
                "Fast-tier refusing duplicate submission",
                trader_id=trader_id,
                signal_id=signal_id,
            )
            return _skip_result(
                "duplicate_signal_existing_intent",
                error_message=f"intent already journalled for signal {signal_id}",
            )

        # Cap: in-flight intents (journal) + open orders (hot_state).  This
        # reproduces the pre-cutover behaviour where the committed
        # ``placing`` skeleton counted against the cap, but without a DB
        # round-trip.
        if risk_open_cap_int is not None and risk_open_cap_int > 0:
            open_count = (
                hot_state.get_open_order_count(trader_id, mode_key)
                + journal.open_intent_count(trader_id)
            )
            if open_count >= risk_open_cap_int:
                logger.info(
                    "Fast-tier blocking submission: trader at max_open_orders cap",
                    trader_id=trader_id,
                    open_count=open_count,
                    max_open_orders=risk_open_cap_int,
                )
                return _skip_result(
                    "max_open_orders_cap",
                    error_message=f"max_open_orders cap reached ({open_count}/{risk_open_cap_int})",
                    extra={"open_count": open_count, "cap": risk_open_cap_int},
                )

        # Durable pre-wire intent (fsync).  This is the single hot-path
        # stable-storage write and the crash-recovery anchor.
        if signal_id:
            journal.record_intent(
                trader_id=trader_id,
                signal_id=signal_id,
                key=idempotency_key,
                token_id=str(leg.get("token_id") or "") or None,
                side=str(leg.get("side") or "") or None,
                size_usd=notional,
                market_id=str(leg.get("market_id") or "") or None,
                mode=mode_key,
            )
    finally:
        # Release before the wire (and as a safety net for any raise above).
        _release_submit_lock_if_held()

    # ----- CLOB submission ---------------------------------------------------
    # ``release_conn`` frees the DB pool slot during the network round-trip
    # (the submission touches no DB).  No pre-submit row exists to lose.
    submit_started_at = utcnow()
    try:
        async with release_conn(session):
            leg_result = await _asyncio.wait_for(
                submit_execution_leg(
                    mode=mode_key,
                    signal=signal,
                    leg=leg,
                    notional_usd=notional,
                    strategy_params=strategy_params,
                    risk_limits=risk_limits,
                    trader_id=trader_id,
                ),
                timeout=_FAST_LEG_SUBMIT_TIMEOUT_SECONDS,
            )
    except (Exception, _asyncio.CancelledError) as exc:
        is_cancelled = isinstance(exc, _asyncio.CancelledError)
        is_timeout = isinstance(exc, _asyncio.TimeoutError)
        # AMBIGUOUS outcome — the venue may or may not hold the order.  We
        # deliberately leave the journal intent OPEN (write no result) so
        # the orphan-reconcile sweep queries the venue by the deterministic
        # key and either backfills the order row or closes the intent as
        # failed.  This preserves the exact crash-safety the pre-submit DB
        # row provided, anchored on the journal instead.
        if is_cancelled:
            logger.warning(
                "Fast-tier leg submit cancelled mid-flight (intent left open for reconcile)",
                trader_id=trader_id,
                signal_id=signal_id,
            )
            raise
        if is_timeout:
            logger.warning(
                "Fast-tier leg submit exceeded CLOB timeout (intent left open for reconcile)",
                trader_id=trader_id,
                signal_id=signal_id,
                timeout_seconds=_FAST_LEG_SUBMIT_TIMEOUT_SECONDS,
            )
        else:
            logger.warning(
                "Fast-tier leg submit raised (intent left open for reconcile)",
                trader_id=trader_id,
                signal_id=signal_id,
                exc_info=exc,
            )
        return FastSubmitResult(
            session_id="",
            status="failed",
            effective_price=None,
            error_message=(
                f"submit_execution_leg timed out after {_FAST_LEG_SUBMIT_TIMEOUT_SECONDS}s (CLOB degraded)"
                if is_timeout
                else f"submit_execution_leg raised: {type(exc).__name__}: {exc}"
            ),
            orders_written=0,
            payload={
                "fast_tier": True,
                "reason": "submit_timeout" if is_timeout else "submit_exception",
                "intent_open_for_reconcile": True,
                "fast_idempotency_key": idempotency_key,
            },
        )
    submit_completed_at = utcnow()

    leg_status = str(leg_result.status or "").strip().lower()

    # ----- Pre-submit gate rejection: venue never received the order ---------
    if leg_status == "skipped":
        # Terminal + no order row: the buy/spread gate rejected before
        # submission, so the venue holds nothing.  Close the intent.
        if signal_id:
            journal.record_result(trader_id=trader_id, signal_id=signal_id, status="skipped")
        return _skip_result(
            "pre_submit_gate",
            error_message=leg_result.error_message,
            extra={"leg": leg_result.payload},
        )

    order_status = {
        "executed": "executed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(leg_status, "failed")

    # ----- Post-wire persistence (off the time-to-market path) ---------------
    # One INSERT of the FINAL state — no skeleton, no second UPDATE.  We
    # commit here (after the wire) so the row is durable BEFORE we close
    # the journal intent; if this commit fails or the process dies first,
    # the intent stays open and the orphan reconcile backfills the row
    # from the venue snapshot.
    order_payload = _result_payload_for_trader_order(leg_result=leg_result, now_iso=now_iso)
    order_payload[_SUBMISSION_STATE_KEY] = _SUBMISSION_STATE_COMPLETED
    order_payload["fast_tier"] = True
    order_payload["fast_idempotency_key"] = idempotency_key
    if strategy_params:
        order_payload["strategy_params"] = dict(strategy_params)
    submit_completed_iso = submit_completed_at.isoformat()
    order_payload.setdefault("submit_started_at_iso", submit_started_at.isoformat())
    order_payload.setdefault("submit_completed_at_iso", submit_completed_iso)

    order_id: str | None = None
    persisted = False
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
            status=order_status,
            notional_usd=leg_result.notional_usd if leg_result.notional_usd is not None else notional,
            effective_price=leg_result.effective_price,
            reason=reason,
            payload=order_payload,
        )
        order.provider_order_id = leg_result.provider_order_id or order.provider_order_id
        order.provider_clob_order_id = leg_result.provider_clob_order_id or order.provider_clob_order_id
        order.error_message = leg_result.error_message
        session.add(order)
        await session.flush()
        order_id = str(order.id)

        event_payload = dict(order.payload_json or {})
        event_payload.update({"status": str(order.status or ""), "mode": str(order.mode or "")})
        append_trader_order_verification_event(
            session,
            trader_order_id=order_id,
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
        # Commit post-wire so the row is durable before we close the
        # intent.  Shielded so a cycle-budget cancellation can't tear a
        # write-back for an order the venue already executed.
        await _asyncio.shield(session.commit())
        persisted = True
    except (Exception, _asyncio.CancelledError) as exc:
        is_cancelled = isinstance(exc, _asyncio.CancelledError)
        # The venue already acted but our DB write failed.  Do NOT close
        # the journal intent — leave it open so the orphan reconcile
        # patches the row in from the venue snapshot (matched by the
        # deterministic key).  Roll back the half-written transaction.
        if is_cancelled:
            logger.warning(
                "Fast-tier post-wire persist cancelled (intent left open for reconcile)",
                trader_id=trader_id,
                signal_id=signal_id,
            )
        else:
            logger.error(
                "Fast-tier post-wire persist failed (intent left open for reconcile)",
                trader_id=trader_id,
                signal_id=signal_id,
                exc_info=exc,
            )
        try:
            await session.rollback()
        except Exception:
            pass
        if is_cancelled:
            raise
        return FastSubmitResult(
            session_id="",
            status=order_status,
            effective_price=leg_result.effective_price,
            error_message=f"fast post-wire persist raised: {type(exc).__name__}: {exc}",
            orders_written=0,
            payload={
                "fast_tier": True,
                "reason": "post_persist_failed",
                "intent_open_for_reconcile": True,
                "fast_idempotency_key": idempotency_key,
            },
        )

    # ----- Update in-memory state, THEN close the intent ---------------------
    # hot_state is upserted before ``record_result`` so the cap never
    # transiently under-counts an order during the index hand-off (an
    # over-count by one is the fail-safe direction for a risk cap).
    if persisted and _is_active_order_status(mode_key, order_status):
        hot_state.upsert_active_order(
            trader_id=trader_id,
            mode=mode_key,
            order_id=order_id,
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
        # ``position_marks_update`` push channel updates U-P&L on the next
        # price tick instead of waiting for the reconciliation worker.
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
                        order_id=order_id,
                        market_id=str(order.market_id or ""),
                        token_id=fill_token_id,
                        direction=str(order.direction or "yes").strip().lower(),
                        entry_price=float(fill_price),
                        notional=float(fill_notional),
                        edge_percent=safe_float(order.edge_percent, 0.0) or 0.0,
                    )
                    try:
                        feed_manager = get_feed_manager()
                        if getattr(feed_manager, "_started", False):
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
                    order_id=order_id,
                    exc_info=_pms_exc,
                )

    # Close the journal intent now that the order row is durable.
    if signal_id:
        try:
            journal.record_result(
                trader_id=trader_id,
                signal_id=signal_id,
                status=order_status,
                provider_clob_order_id=leg_result.provider_clob_order_id,
                provider_order_id=leg_result.provider_order_id,
            )
        except Exception as _jr_exc:
            # A lost result record is safe (replay re-reconciles), so never
            # let it fail a completed trade.
            logger.debug("Fast-tier journal result record failed", trader_id=trader_id, exc_info=_jr_exc)

    # ----- Latency telemetry -------------------------------------------------
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
        logger.debug("Fast-tier latency record failed", trader_id=trader_id, exc_info=exc)

    return FastSubmitResult(
        session_id="",
        status=order_status,
        effective_price=leg_result.effective_price,
        error_message=leg_result.error_message,
        orders_written=1,
        payload={
            "fast_tier": True,
            "trader_order_id": order_id,
            "leg": leg_result.payload,
            "mode": mode_key,
            "submitted_at": now_iso,
            "submit_started_at_iso": submit_started_at.isoformat(),
            "submit_completed_at_iso": submit_completed_iso,
        },
        created_orders=[{"id": order_id}],
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
    gated by the journal dedup check).
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


def _recovery_created_at(rec: dict[str, Any]) -> datetime:
    """Naive-UTC ``created_at`` for a recovery row, backdated to the intent.

    Backdating to the original intent timestamp makes the row immediately
    eligible for ``reconcile_orphaned_fast_submissions`` (which skips rows
    younger than its ``min_age_seconds``), so a crash orphan resolves on
    the first sweep after restart instead of waiting a full cycle.
    """
    ts = rec.get("ts")
    try:
        if ts is not None:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        pass
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=120)


async def recover_open_intents(
    session: AsyncSession,
    *,
    journal: Any = None,
) -> dict[str, Any]:
    """Bridge crash-orphaned journal intents into the DB reconcile path.

    Run at fast-worker startup and on the periodic orphan sweep.  For every
    OPEN journal intent — an ambiguous wire failure, or a crash before the
    post-wire persist — ensure there is a ``TraderOrder`` recovery row the
    tested ``reconcile_orphaned_fast_submissions`` sweep can resolve against
    the venue by the deterministic key.  Intents that already have a
    persisted row, or that are shadow / keyless (no venue to reconcile),
    are simply closed.  Closing records a terminal ``recovered`` result so
    the signal stays deduped but is never re-processed.
    """
    from services.trader_orchestrator.fast_idempotency import is_fast_idempotency_key

    j = journal if journal is not None else get_intent_journal()
    intents = j.open_intents()
    materialized = 0
    closed_existing = 0
    closed_noop = 0

    for rec in intents:
        trader_id = str(rec.get("tr") or "").strip()
        signal_id = str(rec.get("sig") or "").strip()
        if not trader_id or not signal_id:
            continue
        mode_key = str(rec.get("md") or "").strip().lower() or "shadow"
        key = str(rec.get("key") or "").strip()

        # Shadow / keyless intents have no venue counterpart to reconcile.
        if mode_key != "live" or not is_fast_idempotency_key(key):
            j.record_result(trader_id=trader_id, signal_id=signal_id, status="recovered")
            closed_noop += 1
            continue

        existing = (
            await session.execute(
                select(TraderOrder)
                .where(TraderOrder.trader_id == trader_id)
                .where(TraderOrder.signal_id == signal_id)
                .limit(1)
            )
        ).scalar_one_or_none()
        if existing is not None:
            # The post-wire persist landed before the crash — the DB row and
            # the existing reconcile/lifecycle own it now.
            j.record_result(
                trader_id=trader_id,
                signal_id=signal_id,
                status="recovered",
                provider_clob_order_id=existing.provider_clob_order_id,
            )
            closed_existing += 1
            continue

        # No row — the crash hit the ambiguous wire window.  Materialize a
        # recovery row the orphan sweep will resolve by key, then hand it
        # ownership and close the intent.
        recovery_payload = {
            "fast_tier": True,
            _SUBMISSION_STATE_KEY: "in_flight",
            "fast_idempotency_key": key,
            "recovered_from_journal": True,
        }
        await session.merge(
            TraderOrder(
                id=f"fastjrnl-{trader_id}-{signal_id}",
                trader_id=trader_id,
                signal_id=signal_id,
                source=str(rec.get("src") or "fast"),
                market_id=str(rec.get("mkt") or ""),
                mode="live",
                status="placing",
                notional_usd=safe_float(rec.get("usd"), 0.0) or 0.0,
                payload_json=recovery_payload,
                created_at=_recovery_created_at(rec),
            )
        )
        j.record_result(trader_id=trader_id, signal_id=signal_id, status="recovered")
        materialized += 1

    if materialized or closed_existing or closed_noop:
        try:
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                pass
            raise

    return {
        "open_intents": len(intents),
        "materialized": materialized,
        "closed_existing": closed_existing,
        "closed_noop": closed_noop,
    }
