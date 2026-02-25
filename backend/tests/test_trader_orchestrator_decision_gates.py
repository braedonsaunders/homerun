import sys
from pathlib import Path
from types import SimpleNamespace


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
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


def test_min_exit_notional_guard_blocks_under_min_feasible_size(monkeypatch):
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    result = apply_platform_decision_gates(
        decision_obj=_decision(2.0),
        runtime_signal=SimpleNamespace(market_id="market-1", entry_price=0.4, payload_json={}),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={"exit_price_ratio_floor": 0.25},
    )

    assert result["final_decision"] == "blocked"
    assert "Min-exit-notional guard blocked" in result["final_reason"]
    assert any(g["gate"] == "min_exit_notional" and g["status"] == "blocked" for g in result["platform_gates"])
    min_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "min_exit_notional_guard"
    )
    assert min_exit_check["passed"] is False
    assert min_exit_check["payload"]["conservative_exit_source"] == "configured_ratio_floor"
    assert float(min_exit_check["payload"]["required_size_usd"]) >= 4.0


def test_min_exit_notional_guard_respects_strategy_override_ratio(monkeypatch):
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    result = apply_platform_decision_gates(
        decision_obj=_decision(2.2),
        runtime_signal=SimpleNamespace(market_id="market-1", entry_price=0.4, payload_json={}),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={"exit_price_ratio_floor": 0.5},
    )

    assert result["final_decision"] == "selected"
    assert any(g["gate"] == "min_exit_notional" and g["status"] == "passed" for g in result["platform_gates"])
    min_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "min_exit_notional_guard"
    )
    assert min_exit_check["passed"] is True


def test_min_exit_notional_guard_can_be_disabled_by_strategy_config(monkeypatch):
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    result = apply_platform_decision_gates(
        decision_obj=_decision(2.0),
        runtime_signal=SimpleNamespace(market_id="market-1", entry_price=0.4, payload_json={}),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={
            "exit_price_ratio_floor": 0.25,
            "enforce_min_exit_notional": False,
        },
    )

    assert result["final_decision"] == "selected"
    assert any(g["gate"] == "min_exit_notional" and g["status"] == "skipped" for g in result["platform_gates"])
    min_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "min_exit_notional_guard"
    )
    assert min_exit_check["passed"] is True
    assert min_exit_check["payload"]["enabled"] is False
    assert min_exit_check["payload"]["conservative_exit_source"] == "guard_disabled"


def test_min_exit_notional_guard_prefers_stop_loss_price_when_available(monkeypatch):
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    result = apply_platform_decision_gates(
        decision_obj=_decision(3.0),
        runtime_signal=SimpleNamespace(market_id="market-1", entry_price=0.345, payload_json={}),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={"stop_loss_pct": 5.0, "exit_price_ratio_floor": 0.25},
    )

    assert result["final_decision"] == "selected"
    min_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "min_exit_notional_guard"
    )
    assert min_exit_check["passed"] is True
    assert min_exit_check["payload"]["conservative_exit_source"] == "stop_loss_pct"


def test_min_exit_notional_guard_uses_ratio_floor_when_stop_loss_is_near_close_only(monkeypatch):
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    result = apply_platform_decision_gates(
        decision_obj=_decision(3.0),
        runtime_signal=SimpleNamespace(
            market_id="market-1",
            entry_price=0.345,
            payload_json={"strategy_context": {"seconds_left": 240}},
        ),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={
            "stop_loss_pct": 5.0,
            "stop_loss_policy": "near_close_only",
            "stop_loss_activation_seconds": 60,
            "exit_price_ratio_floor": 0.25,
        },
    )

    assert result["final_decision"] == "blocked"
    min_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "min_exit_notional_guard"
    )
    assert min_exit_check["passed"] is False
    assert min_exit_check["payload"]["stop_loss_armed"] is False
    assert min_exit_check["payload"]["conservative_exit_source"] == "configured_ratio_floor"


def test_min_exit_notional_guard_arms_stop_loss_when_inside_close_window(monkeypatch):
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    result = apply_platform_decision_gates(
        decision_obj=_decision(3.0),
        runtime_signal=SimpleNamespace(
            market_id="market-1",
            entry_price=0.345,
            payload_json={"strategy_context": {"seconds_left": 30}},
        ),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={
            "stop_loss_pct": 5.0,
            "stop_loss_policy": "near_close_only",
            "stop_loss_activation_seconds": 60,
            "exit_price_ratio_floor": 0.25,
        },
    )

    assert result["final_decision"] == "selected"
    min_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "min_exit_notional_guard"
    )
    assert min_exit_check["passed"] is True
    assert min_exit_check["payload"]["stop_loss_armed"] is True
    assert min_exit_check["payload"]["conservative_exit_source"] == "stop_loss_pct"


def test_pending_live_exit_guard_blocks_new_entry_when_exit_is_in_flight():
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=_runtime_signal(),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=2,
        pending_live_exit_summary={
            "count": 2,
            "order_ids": ["order-1", "order-2"],
            "market_ids": ["market-1"],
            "statuses": {"submitted": 2},
        },
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "blocked"
    assert "Pending live exit guard blocked" in result["final_reason"]
    assert any(g["gate"] == "pending_live_exit_guard" and g["status"] == "blocked" for g in result["platform_gates"])
    pending_exit_check = next(
        check for check in result["checks_payload"] if check["check_key"] == "pending_live_exit_guard"
    )
    assert pending_exit_check["passed"] is False
    assert pending_exit_check["payload"]["count"] == 2


def test_pending_live_exit_identity_guard_blocks_matching_market_direction_signal():
    runtime_signal = SimpleNamespace(
        id="signal-1",
        market_id="market-1",
        direction="buy_yes",
        payload_json={},
    )
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=runtime_signal,
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=0,
        pending_live_exit_summary={
            "count": 0,
            "order_ids": [],
            "market_ids": [],
            "statuses": {},
            "identities": [
                {
                    "order_id": "order-1",
                    "market_id": "market-1",
                    "direction": "buy_yes",
                    "signal_id": "signal-1",
                    "status": "submitted",
                }
            ],
            "identity_keys": ["market-1|buy_yes|signal-1"],
        },
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "blocked"
    assert "identity guard" in result["final_reason"].lower()
    assert any(
        gate["gate"] == "pending_live_exit_identity_guard" and gate["status"] == "blocked"
        for gate in result["platform_gates"]
    )


def test_pending_live_exit_guard_allows_configured_inflight_limit():
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=_runtime_signal(),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=2,
        pending_live_exit_summary={
            "count": 2,
            "order_ids": ["order-1", "order-2"],
            "market_ids": ["market-1"],
            "statuses": {"submitted": 2},
        },
        pending_live_exit_max_allowed=2,
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "selected"
    guard_check = next(check for check in result["checks_payload"] if check["check_key"] == "pending_live_exit_guard")
    assert guard_check["passed"] is True
    assert guard_check["payload"]["max_allowed"] == 2


def test_pending_live_exit_identity_guard_can_be_disabled():
    runtime_signal = SimpleNamespace(
        id="signal-1",
        market_id="market-1",
        direction="buy_yes",
        payload_json={},
    )
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=runtime_signal,
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=0,
        pending_live_exit_summary={
            "count": 0,
            "order_ids": [],
            "market_ids": [],
            "statuses": {},
            "identities": [
                {
                    "order_id": "order-1",
                    "market_id": "market-1",
                    "direction": "buy_yes",
                    "signal_id": "signal-1",
                    "status": "submitted",
                }
            ],
            "identity_keys": ["market-1|buy_yes|signal-1"],
        },
        pending_live_exit_identity_guard_enabled=False,
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "selected"
    assert any(
        gate["gate"] == "pending_live_exit_identity_guard" and gate["status"] == "skipped"
        for gate in result["platform_gates"]
    )


def test_directional_min_timeframe_blocks_crypto_sub_5m_signal():
    runtime_signal = SimpleNamespace(
        id="signal-crypto-1",
        market_id="market-1",
        direction="buy_yes",
        source="crypto",
        payload_json={"strategy_context": {"timeframe": "1m"}},
    )
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=runtime_signal,
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=0,
        pending_live_exit_summary={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}},
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={"enforce_market_data_freshness": False},
    )

    assert result["final_decision"] == "blocked"
    assert "timeframe guard blocked" in result["final_reason"].lower()
    assert any(
        gate["gate"] == "directional_min_timeframe" and gate["status"] == "blocked" for gate in result["platform_gates"]
    )


def test_max_risk_score_guard_blocks_high_risk_signal():
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=SimpleNamespace(
            id="signal-risk-1",
            market_id="market-1",
            risk_score=0.82,
            payload_json={},
        ),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=0,
        pending_live_exit_summary={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}},
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={
            "enforce_market_data_freshness": False,
            "max_risk_score": 0.4,
        },
    )

    assert result["final_decision"] == "blocked"
    assert "Max-risk guard blocked" in result["final_reason"]
    assert any(g["gate"] == "max_risk_score" and g["status"] == "blocked" for g in result["platform_gates"])
    risk_check = next(check for check in result["checks_payload"] if check["check_key"] == "max_risk_score_guard")
    assert risk_check["passed"] is False
    assert float(risk_check["payload"]["signal_risk_score"]) == 0.82


def test_max_risk_score_guard_skips_when_signal_risk_unavailable():
    result = apply_platform_decision_gates(
        decision_obj=_decision(25.0),
        runtime_signal=SimpleNamespace(
            id="signal-risk-2",
            market_id="market-1",
            payload_json={},
        ),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        pending_live_exit_count=0,
        pending_live_exit_summary={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}},
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={
            "enforce_market_data_freshness": False,
            "max_risk_score": 0.4,
        },
    )

    assert result["final_decision"] == "selected"
    assert any(g["gate"] == "max_risk_score" and g["status"] == "passed" for g in result["platform_gates"])
    risk_check = next(check for check in result["checks_payload"] if check["check_key"] == "max_risk_score_guard")
    assert risk_check["passed"] is True
    assert risk_check["payload"]["signal_risk_score"] is None
