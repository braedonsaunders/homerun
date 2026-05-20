"""Platform-side gate predicates extracted from ``apply_platform_decision_gates``.

Phase 2 migrates the *cheapest* L0_MEMORY gates from the monolithic
``apply_platform_decision_gates`` function into the new :class:`Gate` /
:class:`GatePipeline` protocol.  The remaining gates inside
``decision_gates.apply_platform_decision_gates`` stay inline for now —
Phase 3 will migrate more incrementally.

Side-effect contract
--------------------
``apply_platform_decision_gates`` historically does FOUR things for each
gate:

1. Append a row to ``platform_gates`` describing the outcome.
2. Optionally append a row to ``checks_payload``.
3. On a block, mutate ``final_decision`` / ``final_reason``.
4. On a block (when ``invoke_hooks`` is true), call
   ``strategy.on_blocked(...)``.

This module exposes pure predicates returning :class:`GateResult`.  The
adapter inside ``decision_gates`` replays those side effects from the
:attr:`GateResult.detail` dict, so the legacy payload shapes are byte-
identical to the inline pre-Phase-2 code.

Each ``GateResult.detail`` follows this shape:

```
{
    "platform_gate": {"gate": ..., "status": ..., "detail": ..., "payload": ...?},
    "checks_payload": {...}?,           # optional, only when applicable
    "on_blocked": {                     # only present on block
        "reason": BlockReason.X,        # string constant
        "context": {...},               # passed to strategy.on_blocked
    },
}
```

Gates extracted in Phase 2
--------------------------
* ``strategy_demoted_gate``       L0_MEMORY
* ``signal_staleness_gate``       L0_MEMORY
* ``trading_schedule_gate``       L0_MEMORY
* ``execution_plan_token_conflict_gate``  L0_MEMORY
* ``stacking_guard_gate``         L0_MEMORY  (also known as occupied_market)

Gates NOT extracted (deferred to Phase 3) — see report notes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from services.data_events import BlockReason
from services.trader_orchestrator.gate_pipeline import GateContext, GateResult
from utils.converters import safe_float

# Gate name constants — exported for callers and tests.
GATE_NAME_STRATEGY_DEMOTED = "strategy_demoted"
GATE_NAME_SIGNAL_STALENESS = "signal_staleness"
GATE_NAME_TRADING_SCHEDULE = "trading_schedule"
GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT = "execution_plan_token_conflict"
GATE_NAME_STACKING_GUARD = "stacking_guard"


# ---------------------------------------------------------------------------
# strategy_demoted_gate
# ---------------------------------------------------------------------------


def strategy_demoted_gate(ctx: GateContext) -> GateResult:
    """L0_MEMORY: block when the strategy_type is on the demoted list.

    Requires ``ctx.extras['demoted_strategy_types']`` (a set).  Skips
    silently (passes) when the set is empty or absent.
    """

    demoted: set[str] = ctx.extras.get("demoted_strategy_types") or set()
    if not demoted:
        return GateResult(passed=True)

    strategy_type = str(
        getattr(ctx.runtime_signal, "strategy_type", "") or ""
    ).strip().lower()
    if not strategy_type or strategy_type not in demoted:
        return GateResult(passed=True)

    reason = (
        f"Strategy demoted under validation guardrail (strategy_type={strategy_type})"
    )
    return GateResult(
        passed=False,
        reason=reason,
        detail={
            "platform_gate": {
                "gate": GATE_NAME_STRATEGY_DEMOTED,
                "status": "blocked",
                "detail": reason,
            },
            "on_blocked": {
                "reason": BlockReason.STRATEGY_DEMOTED,
                "context": {"strategy_type": strategy_type},
                # Legacy strategy_demoted block wrapped on_blocked in
                # try/except.  Preserve that defensive behavior here so
                # a misbehaving strategy hook can't break decision
                # evaluation.
                "swallow_errors": True,
            },
        },
    )


# ---------------------------------------------------------------------------
# signal_staleness_gate
# ---------------------------------------------------------------------------


def _parse_datetime_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _staleness_anchor(runtime_signal: Any) -> datetime | None:
    payload = getattr(runtime_signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}

    for candidate in (
        payload.get("signal_emitted_at"),
        payload.get("execution_armed_at"),
        payload.get("ingested_at"),
        getattr(runtime_signal, "updated_at", None),
        getattr(runtime_signal, "created_at", None),
    ):
        parsed = _parse_datetime_utc(candidate)
        if parsed is not None:
            return parsed
    return None


def signal_staleness_gate(ctx: GateContext) -> GateResult:
    """L0_MEMORY: reject signals older than ``max_signal_age_seconds``.

    Opt-in per strategy.  Passes when the strategy doesn't declare
    a cutoff, or when no staleness anchor timestamp is available
    (no source = no rejection).
    """

    max_age_seconds = safe_float(
        (ctx.strategy_params or {}).get("max_signal_age_seconds"),
        None,
    )
    if max_age_seconds is None or max_age_seconds <= 0.0:
        # Gate not configured ⇒ no platform_gate row, no checks_payload row.
        return GateResult(passed=True)

    anchor = _staleness_anchor(ctx.runtime_signal)
    if anchor is None:
        # No anchor ⇒ silent pass (matches inline behavior).
        return GateResult(passed=True)

    age_seconds = (datetime.now(timezone.utc) - anchor).total_seconds()
    if age_seconds <= float(max_age_seconds):
        return GateResult(
            passed=True,
            detail={
                "platform_gate": {
                    "gate": GATE_NAME_SIGNAL_STALENESS,
                    "status": "passed",
                    "detail": f"age={age_seconds:.1f}s <= max={float(max_age_seconds):.1f}s",
                },
            },
        )

    reason = f"Signal stale: age={age_seconds:.1f}s > max={float(max_age_seconds):.1f}s"
    return GateResult(
        passed=False,
        reason=reason,
        detail={
            "platform_gate": {
                "gate": GATE_NAME_SIGNAL_STALENESS,
                "status": "blocked",
                "detail": reason,
            },
            "on_blocked": {
                "reason": BlockReason.STALE_SIGNAL,
                "context": {
                    "age_seconds": age_seconds,
                    "max_age_seconds": float(max_age_seconds),
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# trading_schedule_gate
# ---------------------------------------------------------------------------


def trading_schedule_gate(ctx: GateContext) -> GateResult:
    """L0_MEMORY: enforce the configured UTC trading-window.

    Requires ``ctx.extras['trading_schedule_ok']`` (bool) and
    ``ctx.extras['trading_schedule_config']`` (dict, optional).
    The schedule is computed by the orchestrator before this gate runs
    (cheap, in-memory) — we just check the result.
    """

    trading_schedule_ok = bool(ctx.extras.get("trading_schedule_ok", True))
    trading_schedule_config = ctx.extras.get("trading_schedule_config") or {}

    if trading_schedule_ok:
        return GateResult(
            passed=True,
            detail={
                "platform_gate": {
                    "gate": GATE_NAME_TRADING_SCHEDULE,
                    "status": "passed",
                    "detail": "Inside configured UTC trading schedule",
                },
            },
        )

    reason = "Outside configured trading schedule (UTC)"
    return GateResult(
        passed=False,
        reason=reason,
        detail={
            "platform_gate": {
                "gate": GATE_NAME_TRADING_SCHEDULE,
                "status": "blocked",
                "detail": reason,
            },
            "on_blocked": {
                "reason": BlockReason.TRADING_WINDOW,
                "context": {"trading_schedule": trading_schedule_config},
            },
        },
    )


# ---------------------------------------------------------------------------
# execution_plan_token_conflict_gate
# ---------------------------------------------------------------------------


def execution_plan_token_conflict_gate(ctx: GateContext) -> GateResult:
    """L0_MEMORY: reject plans with duplicate buy legs or self-crossing quotes.

    The original gate also appended a row to ``checks_payload`` regardless
    of pass/fail — we replay that via ``detail['checks_payload']``.

    We import the helpers from ``decision_gates`` lazily inside the
    function body so this module does not create a top-level circular
    import (decision_gates imports this module to wire the pipeline).
    """

    # Lazy import to avoid module-load circular dependency.
    from services.trader_orchestrator.decision_gates import (
        _execution_plan_token_conflict,
        _runtime_signal_execution_plan,
    )

    plan, payload = _runtime_signal_execution_plan(ctx.runtime_signal)
    conflict = _execution_plan_token_conflict(plan, payload)
    passed = conflict is None
    plan_id = str(plan.get("plan_id") or "").strip() or None

    checks_payload_row = {
        "check_key": "execution_plan_token_conflict_guard",
        "check_label": "Execution plan token conflict guard",
        "passed": passed,
        "score": None,
        "detail": (
            "No duplicate buy or self-crossing token legs"
            if passed
            else "Execution plan has duplicate buy legs or self-crossing quotes for one token"
        ),
        "payload": {
            "plan_id": plan_id,
            "violation": conflict,
        },
    }

    if passed:
        return GateResult(
            passed=True,
            detail={
                "platform_gate": {
                    "gate": GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT,
                    "status": "passed",
                    "detail": "No duplicate buy or self-crossing token legs",
                },
                "checks_payload": checks_payload_row,
            },
        )

    violation_reason = str((conflict or {}).get("reason") or "token_conflict")
    block_reason = f"Execution plan token conflict guard blocked: {violation_reason}"
    return GateResult(
        passed=False,
        reason=block_reason,
        detail={
            "platform_gate": {
                "gate": GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT,
                "status": "blocked",
                "detail": block_reason,
                "payload": conflict,
            },
            "checks_payload": checks_payload_row,
            "on_blocked": {
                "reason": BlockReason.RISK_TRADE_NOTIONAL,
                "context": conflict,
            },
        },
    )


# ---------------------------------------------------------------------------
# stacking_guard_gate (a.k.a. occupied_market)
# ---------------------------------------------------------------------------


def stacking_guard_gate(ctx: GateContext) -> GateResult:
    """L0_MEMORY: enforce one active live entry per market.

    Requires:

    * ``ctx.extras['occupied_market_ids']`` — set[str].
    * ``ctx.extras['allow_averaging']``     — bool.
    * ``ctx.extras['execution_mode']``       — str, defaults to ctx.mode.

    Matches the inline ``stacking_guard`` block at the bottom of
    ``apply_platform_decision_gates``.
    """

    occupied: set[str] = ctx.extras.get("occupied_market_ids") or set()
    allow_averaging = bool(ctx.extras.get("allow_averaging", False))
    execution_mode = str(
        ctx.extras.get("execution_mode") or ctx.mode or "live"
    ).strip().lower()
    live_single_market_guard = execution_mode == "live"

    if not (live_single_market_guard or not allow_averaging):
        return GateResult(passed=True)

    signal_market_id = str(getattr(ctx.runtime_signal, "market_id", "") or "").strip()
    stacking_blocked = bool(signal_market_id) and signal_market_id in occupied

    checks_payload_row = {
        "check_key": "stacking_guard",
        "check_label": (
            "One active live entry per market"
            if live_single_market_guard
            else "One active entry per market"
        ),
        "passed": not stacking_blocked,
        "score": None,
        "detail": (
            "live execution permits only one active entry per market"
            if stacking_blocked and live_single_market_guard
            else "allow_averaging=false and market is already occupied by an open position or active order"
            if stacking_blocked
            else "Market is not occupied"
        ),
        "payload": {
            "allow_averaging": allow_averaging,
            "live_single_market_guard": live_single_market_guard,
            "market_id": signal_market_id or None,
        },
    }

    if not stacking_blocked:
        return GateResult(
            passed=True,
            detail={
                "platform_gate": {
                    "gate": GATE_NAME_STACKING_GUARD,
                    "status": "passed",
                    "detail": "No existing occupied market for signal",
                },
                "checks_payload": checks_payload_row,
            },
        )

    reason = (
        "Live exposure guard: market already occupied"
        if live_single_market_guard
        else "Stacking guard: market already occupied while allow_averaging=false"
    )
    return GateResult(
        passed=False,
        reason=reason,
        detail={
            "platform_gate": {
                "gate": GATE_NAME_STACKING_GUARD,
                "status": "blocked",
                "detail": reason,
            },
            "checks_payload": checks_payload_row,
            "on_blocked": {
                "reason": BlockReason.STACKING_GUARD,
                "context": {"market_id": signal_market_id},
            },
        },
    )


__all__ = [
    "GATE_NAME_STRATEGY_DEMOTED",
    "GATE_NAME_SIGNAL_STALENESS",
    "GATE_NAME_TRADING_SCHEDULE",
    "GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT",
    "GATE_NAME_STACKING_GUARD",
    "strategy_demoted_gate",
    "signal_staleness_gate",
    "trading_schedule_gate",
    "execution_plan_token_conflict_gate",
    "stacking_guard_gate",
]
