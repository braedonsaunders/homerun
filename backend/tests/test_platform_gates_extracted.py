"""Tests for the Phase 2 platform gates extracted from
``apply_platform_decision_gates``.

Each extracted predicate is exercised with representative inputs and the
``GateResult`` is compared against what the legacy inline code would
have appended to ``platform_gates`` / ``checks_payload`` / what
``strategy.on_blocked`` would have been called with.

Three gates are wired into the pipeline today (strategy_demoted,
signal_staleness, trading_schedule).  The remaining two extracted
predicates (execution_plan_token_conflict, stacking_guard) are exercised
here too so they're ready for Phase 3 to wire in incrementally.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.data_events import BlockReason
from services.trader_orchestrator.gate_pipeline import GateContext
from services.trader_orchestrator.platform_gates import (
    GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT,
    GATE_NAME_SIGNAL_STALENESS,
    GATE_NAME_STACKING_GUARD,
    GATE_NAME_STRATEGY_DEMOTED,
    GATE_NAME_TRADING_SCHEDULE,
    execution_plan_token_conflict_gate,
    signal_staleness_gate,
    stacking_guard_gate,
    strategy_demoted_gate,
    trading_schedule_gate,
)


def _signal(*, strategy_type: str = "generic_strategy", market_id: str = "market-1", payload: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_type=strategy_type,
        market_id=market_id,
        direction="buy_yes",
        payload_json=payload or {},
        source="scanner",
    )


def _ctx(**kwargs) -> GateContext:
    """Build a default GateContext, overriding fields via kwargs."""
    base = dict(
        runtime_signal=_signal(),
        decision=None,
        leg=None,
        live_context=None,
        risk_limits={},
        strategy_params={},
        mode="live",
        extras={},
    )
    base.update(kwargs)
    return GateContext(**base)


# ---------------------------------------------------------------------------
# strategy_demoted_gate
# ---------------------------------------------------------------------------


def test_strategy_demoted_gate_passes_when_demoted_set_empty():
    """No demoted strategies ⇒ passes silently (no platform_gate row)."""
    result = strategy_demoted_gate(_ctx(extras={"demoted_strategy_types": set()}))
    assert result.passed is True
    assert "platform_gate" not in (result.detail or {})


def test_strategy_demoted_gate_passes_when_strategy_not_in_set():
    """Strategy not in demoted set ⇒ passes silently."""
    result = strategy_demoted_gate(
        _ctx(
            runtime_signal=_signal(strategy_type="generic_strategy"),
            extras={"demoted_strategy_types": {"other_strategy"}},
        )
    )
    assert result.passed is True
    assert "platform_gate" not in (result.detail or {})


def test_strategy_demoted_gate_blocks_when_strategy_demoted():
    """Strategy IS demoted ⇒ blocked with platform_gate row + on_blocked."""
    result = strategy_demoted_gate(
        _ctx(
            runtime_signal=_signal(strategy_type="weather_strategy"),
            extras={"demoted_strategy_types": {"weather_strategy", "other"}},
        )
    )
    assert result.passed is False
    assert "weather_strategy" in result.reason

    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_STRATEGY_DEMOTED
    assert platform_gate["status"] == "blocked"
    assert "weather_strategy" in platform_gate["detail"]

    on_blocked = result.detail["on_blocked"]
    assert on_blocked["reason"] == BlockReason.STRATEGY_DEMOTED
    assert on_blocked["context"] == {"strategy_type": "weather_strategy"}
    # Legacy block wrapped strategy.on_blocked in try/except; we preserve
    # that via the swallow_errors flag.
    assert on_blocked["swallow_errors"] is True


# ---------------------------------------------------------------------------
# signal_staleness_gate
# ---------------------------------------------------------------------------


def test_signal_staleness_gate_passes_silently_when_not_configured():
    """No max_signal_age_seconds ⇒ no row, no block."""
    result = signal_staleness_gate(_ctx(strategy_params={}))
    assert result.passed is True
    assert "platform_gate" not in (result.detail or {})


def test_signal_staleness_gate_passes_silently_when_no_anchor():
    """Configured but no anchor timestamp ⇒ no row, no block (matches legacy)."""
    result = signal_staleness_gate(
        _ctx(
            runtime_signal=_signal(payload={}),  # no signal_emitted_at, etc
            strategy_params={"max_signal_age_seconds": 5},
        )
    )
    assert result.passed is True
    assert "platform_gate" not in (result.detail or {})


def test_signal_staleness_gate_emits_passed_row_when_fresh():
    """Within max age ⇒ passed row, no on_blocked."""
    fresh = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat().replace("+00:00", "Z")
    result = signal_staleness_gate(
        _ctx(
            runtime_signal=_signal(payload={"signal_emitted_at": fresh}),
            strategy_params={"max_signal_age_seconds": 10},
        )
    )
    assert result.passed is True
    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_SIGNAL_STALENESS
    assert platform_gate["status"] == "passed"
    assert "platform_gate" in result.detail
    assert "on_blocked" not in (result.detail or {})


def test_signal_staleness_gate_blocks_when_stale():
    """Beyond max age ⇒ blocked row + on_blocked with age context."""
    stale = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    result = signal_staleness_gate(
        _ctx(
            runtime_signal=_signal(payload={"signal_emitted_at": stale}),
            strategy_params={"max_signal_age_seconds": 5},
        )
    )
    assert result.passed is False
    assert "Signal stale" in result.reason

    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_SIGNAL_STALENESS
    assert platform_gate["status"] == "blocked"

    on_blocked = result.detail["on_blocked"]
    assert on_blocked["reason"] == BlockReason.STALE_SIGNAL
    assert on_blocked["context"]["max_age_seconds"] == 5.0
    assert on_blocked["context"]["age_seconds"] >= 25.0  # rough lower bound


# ---------------------------------------------------------------------------
# trading_schedule_gate
# ---------------------------------------------------------------------------


def test_trading_schedule_gate_passes_when_ok():
    """trading_schedule_ok=True ⇒ passed row, no on_blocked."""
    result = trading_schedule_gate(_ctx(extras={"trading_schedule_ok": True}))
    assert result.passed is True
    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_TRADING_SCHEDULE
    assert platform_gate["status"] == "passed"


def test_trading_schedule_gate_blocks_when_outside_window():
    """trading_schedule_ok=False ⇒ blocked row + on_blocked with config."""
    config = {"start": "00:00", "end": "23:59"}
    result = trading_schedule_gate(
        _ctx(extras={"trading_schedule_ok": False, "trading_schedule_config": config})
    )
    assert result.passed is False
    assert "Outside" in result.reason

    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_TRADING_SCHEDULE
    assert platform_gate["status"] == "blocked"

    on_blocked = result.detail["on_blocked"]
    assert on_blocked["reason"] == BlockReason.TRADING_WINDOW
    assert on_blocked["context"] == {"trading_schedule": config}


# ---------------------------------------------------------------------------
# execution_plan_token_conflict_gate
# ---------------------------------------------------------------------------


def test_execution_plan_token_conflict_gate_passes_with_clean_plan():
    """Plan with no token collisions ⇒ passes; checks_payload row emitted."""
    payload = {
        "execution_plan": {
            "plan_id": "plan-clean",
            "legs": [
                {"leg_id": "l1", "market_id": "m1", "token_id": "t1", "side": "buy", "limit_price": 0.5},
                {"leg_id": "l2", "market_id": "m2", "token_id": "t2", "side": "buy", "limit_price": 0.5},
            ],
        },
    }
    result = execution_plan_token_conflict_gate(
        _ctx(runtime_signal=_signal(payload=payload))
    )
    assert result.passed is True

    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT
    assert platform_gate["status"] == "passed"

    checks = result.detail["checks_payload"]
    assert checks["check_key"] == "execution_plan_token_conflict_guard"
    assert checks["passed"] is True
    assert checks["payload"]["plan_id"] == "plan-clean"


def test_execution_plan_token_conflict_gate_blocks_on_duplicate_buy_legs():
    """Two buy legs on the same (market, token, outcome) ⇒ blocked."""
    payload = {
        "execution_plan": {
            "plan_id": "plan-dup",
            "legs": [
                {
                    "leg_id": "l1",
                    "market_id": "m1",
                    "token_id": "t1",
                    "outcome": "yes",
                    "side": "buy",
                    "limit_price": 0.5,
                },
                {
                    "leg_id": "l2",
                    "market_id": "m1",
                    "token_id": "t1",
                    "outcome": "yes",
                    "side": "buy",
                    "limit_price": 0.55,
                },
            ],
        },
    }
    result = execution_plan_token_conflict_gate(
        _ctx(runtime_signal=_signal(payload=payload))
    )
    assert result.passed is False
    assert "duplicate_buy_legs" in result.reason

    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_EXECUTION_PLAN_TOKEN_CONFLICT
    assert platform_gate["status"] == "blocked"
    assert platform_gate["payload"]["reason"] == "duplicate_buy_legs"

    checks = result.detail["checks_payload"]
    assert checks["passed"] is False

    on_blocked = result.detail["on_blocked"]
    assert on_blocked["reason"] == BlockReason.RISK_TRADE_NOTIONAL


# ---------------------------------------------------------------------------
# stacking_guard_gate
# ---------------------------------------------------------------------------


def test_stacking_guard_gate_passes_when_market_not_occupied():
    """Market not in occupied set ⇒ passes (live mode)."""
    result = stacking_guard_gate(
        _ctx(
            runtime_signal=_signal(market_id="market-free"),
            extras={
                "occupied_market_ids": {"market-other"},
                "allow_averaging": False,
                "execution_mode": "live",
            },
        )
    )
    assert result.passed is True
    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_STACKING_GUARD
    assert platform_gate["status"] == "passed"


def test_stacking_guard_gate_blocks_in_live_when_market_occupied():
    """Live mode + occupied market ⇒ blocked, regardless of allow_averaging."""
    result = stacking_guard_gate(
        _ctx(
            runtime_signal=_signal(market_id="market-busy"),
            extras={
                "occupied_market_ids": {"market-busy"},
                "allow_averaging": True,  # ignored in live mode
                "execution_mode": "live",
            },
        )
    )
    assert result.passed is False
    assert "already occupied" in result.reason.lower()

    platform_gate = result.detail["platform_gate"]
    assert platform_gate["gate"] == GATE_NAME_STACKING_GUARD
    assert platform_gate["status"] == "blocked"

    on_blocked = result.detail["on_blocked"]
    assert on_blocked["reason"] == BlockReason.STACKING_GUARD
    assert on_blocked["context"] == {"market_id": "market-busy"}


def test_stacking_guard_gate_passes_when_averaging_allowed_in_shadow():
    """Shadow mode + allow_averaging=True ⇒ gate is a no-op pass."""
    result = stacking_guard_gate(
        _ctx(
            runtime_signal=_signal(market_id="market-busy"),
            extras={
                "occupied_market_ids": {"market-busy"},
                "allow_averaging": True,
                "execution_mode": "shadow",
            },
        )
    )
    assert result.passed is True
    # When the gate is skipped entirely, no platform_gate row.
    assert "platform_gate" not in (result.detail or {})


# ---------------------------------------------------------------------------
# Integration: legacy apply_platform_decision_gates path still works
# ---------------------------------------------------------------------------


def test_apply_platform_decision_gates_emits_demoted_block_via_pipeline():
    """The refactored ``apply_platform_decision_gates`` returns the same
    ``platform_gates`` list shape for the strategy_demoted block as the
    legacy inline code did."""
    from services.trader_orchestrator.decision_gates import apply_platform_decision_gates

    runtime_signal = _signal(strategy_type="weather_strategy")
    decision_obj = SimpleNamespace(decision="selected", reason="ok", score=1.0, size_usd=10.0)
    strategy = MagicMock()

    result = apply_platform_decision_gates(
        decision_obj=decision_obj,
        runtime_signal=runtime_signal,
        strategy=strategy,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=False,
        occupied_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=lambda s: (SimpleNamespace(allowed=True, reason="ok", checks=[]), {}),
        invoke_hooks=True,
        execution_mode="live",
        demoted_strategy_types={"weather_strategy"},
    )

    assert result["final_decision"] == "blocked"
    assert "weather_strategy" in result["final_reason"]
    # The strategy_demoted row must be present.
    gate_names = [
        g.get("gate")
        for g in result["platform_gates"]
        if isinstance(g, dict)
    ]
    assert GATE_NAME_STRATEGY_DEMOTED in gate_names
    # The strategy.on_blocked hook fires (we passed invoke_hooks=True).
    strategy.on_blocked.assert_called_once()
    args, _kwargs = strategy.on_blocked.call_args
    # First arg is the runtime_signal, second is the BlockReason.STRATEGY_DEMOTED.
    assert args[1] == BlockReason.STRATEGY_DEMOTED


def test_apply_platform_decision_gates_emits_trading_schedule_block_via_pipeline():
    """Trading schedule outside window ⇒ pipeline emits the same payload
    the legacy code did, and the strategy hook fires with the schedule
    config."""
    from services.trader_orchestrator.decision_gates import apply_platform_decision_gates

    runtime_signal = _signal()
    decision_obj = SimpleNamespace(decision="selected", reason="ok", score=1.0, size_usd=10.0)
    strategy = MagicMock()
    schedule_config = {"start": "08:00", "end": "10:00", "days": ["Mon"]}

    result = apply_platform_decision_gates(
        decision_obj=decision_obj,
        runtime_signal=runtime_signal,
        strategy=strategy,
        checks_payload=[],
        trading_schedule_ok=False,
        trading_schedule_config=schedule_config,
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=False,
        occupied_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=lambda s: (SimpleNamespace(allowed=True, reason="ok", checks=[]), {}),
        invoke_hooks=True,
        execution_mode="live",
        demoted_strategy_types=set(),
    )

    assert result["final_decision"] == "blocked"
    assert "Outside" in result["final_reason"]
    gate_names = [
        g.get("gate")
        for g in result["platform_gates"]
        if isinstance(g, dict)
    ]
    assert GATE_NAME_TRADING_SCHEDULE in gate_names
    strategy.on_blocked.assert_called_once()
    args, _kwargs = strategy.on_blocked.call_args
    assert args[1] == BlockReason.TRADING_WINDOW
    assert args[2] == {"trading_schedule": schedule_config}
