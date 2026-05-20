"""Venue-side gates run pre-DB by the session engine's preflight.

These are the predicates the Phase 1 hoist pulled out of ``submit_leg``:

* ``buy_collateral_gate`` — for live BUYs, check the USDC wallet has
  enough notional to cover the leg.  Skipped for shadow mode and for
  SELLs (which return shares from a token balance, not USDC).
* ``max_spread_bps_gate`` — for both shadow and live, reject when the
  current best-bid/best-ask spread exceeds the configured cap.

Both predicates produce :class:`GateResult` instances whose ``detail``
carries the keys the existing :func:`_synthesize_preflight_skipped_result`
helper in ``session_engine`` needs to build a ``LegSubmitResult`` payload
identical to the one the old inline code produced.  Phase 2 must not
change any payload shapes — only the wiring.

The two predicates are module-level functions (not closures) so future
phases (StrategySDK custom gates, Phase 3) can import and compose them
directly.
"""

from __future__ import annotations

from typing import Any

from services import live_execution_service as live_execution_service_module
from services.trader_orchestrator.gate_pipeline import GateContext, GateResult
from services.trader_orchestrator.order_manager import (
    _check_max_spread_bps,
    _resolve_shadow_book_and_tape,
    _resolve_token_id_for_leg,
    _safe_live_context,
    _safe_signal_payload,
)
from utils.converters import safe_float
from utils.logger import get_logger

logger = get_logger(__name__)


# Gate names — exported so callers and tests reference the same string.
GATE_NAME_BUY_COLLATERAL = "buy_collateral"
GATE_NAME_MAX_SPREAD_BPS = "max_spread_bps"

# Reason codes — matches the strings the Phase 1 inline code wrote into
# the ``payload_extras['reason']`` field.  Downstream code (rollups,
# Slack alerts, the orchestrator dashboard) keys on these.
REASON_BUY_PRE_SUBMIT_GATE = "buy_pre_submit_gate"
REASON_MAX_SPREAD_BPS_EXCEEDED = "max_spread_bps_exceeded"


def _normalize_side(leg: dict[str, Any] | None) -> str:
    """Return ``BUY`` or ``SELL`` for the leg.  Defaults to BUY."""
    if not isinstance(leg, dict):
        return "BUY"
    side_key = str(leg.get("side") or "buy").strip().lower()
    return "SELL" if side_key == "sell" else "BUY"


def _leg_requested_notional(leg: dict[str, Any] | None) -> float:
    if not isinstance(leg, dict):
        return 0.0
    return float(max(0.0, safe_float(leg.get("requested_notional_usd"), 0.0) or 0.0))


def _pass_result() -> GateResult:
    return GateResult(passed=True)


async def buy_collateral_gate(ctx: GateContext) -> GateResult:
    """L1_CACHED gate: ensure the BUY leg can be funded.

    Skips with ``passed=True`` (no-op pass) for shadow mode or for SELL
    legs — the USDC balance is irrelevant in those cases.  Skips with
    pass when the leg lacks a resolvable token id (matches the inline
    Phase 1 fallback: let ``submit_leg`` do the authoritative check).

    On an exception inside the live-execution gate call we likewise
    return pass — preflight failure must never block trading.  The
    post-DB inline gate inside ``submit_leg`` is the authoritative
    backstop.
    """

    leg = ctx.leg
    mode_key = str(ctx.mode or "").strip().lower()
    if mode_key != "live":
        return _pass_result()
    if _normalize_side(leg) != "BUY":
        return _pass_result()
    if not isinstance(leg, dict):
        return _pass_result()

    payload = _safe_signal_payload(ctx.runtime_signal)
    live_context_dict = (
        ctx.live_context
        if isinstance(ctx.live_context, dict)
        else _safe_live_context(ctx.runtime_signal, payload)
    )

    token_id, _src, _attempts = _resolve_token_id_for_leg(
        leg=leg,
        payload=payload,
        live_context=live_context_dict,
    )
    if not token_id:
        # Match Phase 1 behaviour: no token id ⇒ skip gate, let
        # submit_leg do the authoritative check.
        return _pass_result()

    requested_shares = safe_float(leg.get("requested_shares"), None)
    limit_price = safe_float(leg.get("limit_price"), None)
    requested_notional = _leg_requested_notional(leg)

    try:
        buy_gate_ok, buy_gate_error = (
            await live_execution_service_module.live_execution_service.check_buy_pre_submit_gate(
                token_id=token_id,
                required_notional_usd=requested_notional,
            )
        )
    except Exception as exc:
        logger.warning(
            "buy_collateral_gate: live gate raised %r; falling back to submit_leg gate",
            exc,
        )
        return _pass_result()

    if buy_gate_ok:
        return _pass_result()

    return GateResult(
        passed=False,
        reason=REASON_BUY_PRE_SUBMIT_GATE,
        error_message=buy_gate_error or "BUY pre-submit gate failed.",
        detail={
            "token_id": token_id,
            "side": "BUY",
            "requested_notional_usd": requested_notional,
            "effective_notional_usd": requested_notional,
            "requested_shares": requested_shares,
            "effective_price": limit_price,
            "leg": dict(leg),
            "payload_extras": {
                "mode": mode_key,
                "submission": "skipped",
                "reason": REASON_BUY_PRE_SUBMIT_GATE,
                "token_id": token_id,
                "leg": dict(leg),
                "requested_shares": requested_shares,
                "requested_notional_usd": requested_notional,
                "effective_notional_usd": requested_notional,
                "preflight_rejected": True,
            },
        },
    )


async def max_spread_bps_gate(ctx: GateContext) -> GateResult:
    """L1_CACHED gate: reject when book spread exceeds the configured cap.

    Mirrors the inline check in ``submit_leg``.  Runs for both shadow and
    live so the shadow preview produces the same decision the live path
    would.  Skips with pass when the cap is unset/zero (knob off) or the
    book is unavailable — those are not rejections, they're "data
    missing, defer to post-DB gate".
    """

    leg = ctx.leg
    if not isinstance(leg, dict):
        return _pass_result()

    payload = _safe_signal_payload(ctx.runtime_signal)
    live_context_dict = (
        ctx.live_context
        if isinstance(ctx.live_context, dict)
        else _safe_live_context(ctx.runtime_signal, payload)
    )

    token_id, _src, _attempts = _resolve_token_id_for_leg(
        leg=leg,
        payload=payload,
        live_context=live_context_dict,
    )

    try:
        book_payload, _trades, _book_age_ms, _quote_source, _quote_err = (
            await _resolve_shadow_book_and_tape(
                token_id=token_id,
                live_context=live_context_dict,
            )
        )
    except Exception as exc:
        logger.warning(
            "max_spread_bps_gate: book resolution raised %r; falling back to submit_leg gate",
            exc,
        )
        return _pass_result()

    spread_rejected, spread_bps, spread_cap = _check_max_spread_bps(
        book_payload=book_payload,
        risk_limits=ctx.risk_limits,
    )
    if not spread_rejected:
        return _pass_result()

    mode_key = str(ctx.mode or "").strip().lower()
    requested_notional = _leg_requested_notional(leg)
    requested_shares = safe_float(leg.get("requested_shares"), None)
    limit_price = safe_float(leg.get("limit_price"), None)

    return GateResult(
        passed=False,
        reason=REASON_MAX_SPREAD_BPS_EXCEEDED,
        error_message=(
            f"Book spread {spread_bps:.1f} bps exceeds "
            f"max_spread_bps {spread_cap:.1f}."
        ),
        detail={
            "token_id": token_id,
            "spread_bps": round(float(spread_bps), 2),
            "max_spread_bps": float(spread_cap),
            "requested_notional_usd": requested_notional,
            "effective_notional_usd": 0.0,
            "requested_shares": requested_shares,
            "effective_price": limit_price,
            "leg": dict(leg),
            "payload_extras": {
                "mode": mode_key,
                "submission": "rejected",
                "reason": REASON_MAX_SPREAD_BPS_EXCEEDED,
                "spread_bps": round(float(spread_bps), 2),
                "max_spread_bps": float(spread_cap),
                "token_id": token_id,
                "leg": dict(leg),
                "requested_notional_usd": requested_notional,
                "effective_notional_usd": 0.0,
                "preflight_rejected": True,
            },
        },
    )


__all__ = [
    "GATE_NAME_BUY_COLLATERAL",
    "GATE_NAME_MAX_SPREAD_BPS",
    "REASON_BUY_PRE_SUBMIT_GATE",
    "REASON_MAX_SPREAD_BPS_EXCEEDED",
    "buy_collateral_gate",
    "max_spread_bps_gate",
]
