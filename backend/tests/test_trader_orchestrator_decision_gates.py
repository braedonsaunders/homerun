import sys
from pathlib import Path
from types import SimpleNamespace


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.decision_gates import apply_platform_decision_gates


def _decision(size_usd: float = 50.0):
    return SimpleNamespace(
        decision="selected",
        reason="selected",
        score=1.0,
        size_usd=size_usd,
    )


def _runtime_signal():
    return SimpleNamespace(market_id="market-1")


def _risk_evaluator(size_for_eval: float):
    return (
        SimpleNamespace(
            allowed=True,
            reason=f"risk_ok:{size_for_eval:.2f}",
            checks=[],
        ),
        {},
    )


def test_portfolio_allocator_caps_selected_size_before_risk_gate():
    result = apply_platform_decision_gates(
        decision_obj=_decision(60.0),
        runtime_signal=_runtime_signal(),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=lambda _size: {
            "allowed": True,
            "reason": "Portfolio capped for source budget",
            "size_usd": 25.0,
            "target_gross_cap_usd": 3000.0,
            "remaining_gross_cap_usd": 25.0,
            "source_key": "crypto",
            "source_cap_usd": 900.0,
            "source_exposure_usd": 875.0,
            "source_remaining_usd": 25.0,
            "min_order_notional_usd": 10.0,
            "target_utilization_pct": 60.0,
            "max_source_exposure_pct": 30.0,
        },
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "selected"
    assert result["size_usd"] == 25.0
    assert any(g["gate"] == "portfolio" and g["status"] == "capped" for g in result["platform_gates"])
    portfolio_check = next(check for check in result["checks_payload"] if check["check_key"] == "portfolio_allocator")
    assert portfolio_check["passed"] is True
    assert "Portfolio capped" in str(portfolio_check["detail"])


def test_portfolio_allocator_blocks_signal_when_allocation_not_allowed():
    result = apply_platform_decision_gates(
        decision_obj=_decision(40.0),
        runtime_signal=_runtime_signal(),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=lambda _size: {
            "allowed": False,
            "reason": "Portfolio blocked: no remaining gross exposure budget",
            "size_usd": 0.0,
            "target_gross_cap_usd": 3000.0,
            "remaining_gross_cap_usd": 0.0,
            "source_key": "crypto",
            "source_cap_usd": 900.0,
            "source_exposure_usd": 875.0,
            "source_remaining_usd": 25.0,
            "min_order_notional_usd": 10.0,
            "target_utilization_pct": 60.0,
            "max_source_exposure_pct": 30.0,
        },
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "blocked"
    assert "Portfolio blocked" in result["final_reason"]
    assert any(g["gate"] == "portfolio" and g["status"] == "blocked" for g in result["platform_gates"])
    assert any(g["gate"] == "risk" and g["status"] == "skipped" for g in result["platform_gates"])
    portfolio_check = next(check for check in result["checks_payload"] if check["check_key"] == "portfolio_allocator")
    assert portfolio_check["passed"] is False
