import sys
from datetime import datetime, timedelta, timezone
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
        strategy_params={
            "enforce_market_data_freshness": False,
            "require_live_market_revalidation": False,
        },
    )

    assert result["final_decision"] == "blocked"
    assert "timeframe guard blocked" in result["final_reason"].lower()
    assert any(
        gate["gate"] == "directional_min_timeframe" and gate["status"] == "blocked" for gate in result["platform_gates"]
    )


def test_market_data_freshness_blocks_stale_scanner_signal():
    runtime_signal = SimpleNamespace(
        id="signal-scanner-stale-1",
        market_id="market-1",
        direction="buy_yes",
        source="scanner",
        payload_json={
            "market_data_age_ms": 3500,
            "source_observed_at": "2026-02-28T00:00:00Z",
            "strategy_context": {"source": "scanner"},
        },
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
        strategy_params={
            "max_market_data_age_ms": 1000,
            "require_live_market_revalidation": False,
        },
    )

    assert result["final_decision"] == "blocked"
    assert "freshness gate blocked" in result["final_reason"].lower()
    assert any(
        gate["gate"] == "market_data_freshness" and gate["status"] == "blocked"
        for gate in result["platform_gates"]
    )


def test_live_market_revalidation_blocks_crypto_when_freshness_is_unprovable():
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_signal = SimpleNamespace(
        id="signal-crypto-unprovable-live",
        market_id="market-1",
        direction="buy_yes",
        source="crypto",
        payload_json={
            "strategy_context": {"timeframe": "5m"},
            "live_market": {
                "live_selected_price": 0.52,
                "fetched_at": now_iso,
                "market_data_source": "market_snapshot",
            },
        },
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
        strategy_params={"max_market_data_age_ms": 1000},
    )

    assert result["final_decision"] == "blocked"
    assert "live market revalidation required" in result["final_reason"].lower()
    assert any(
        gate["gate"] == "live_market_revalidation" and gate["status"] == "blocked"
        for gate in result["platform_gates"]
    )


def test_live_market_revalidation_allows_crypto_with_trusted_live_source():
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_signal = SimpleNamespace(
        id="signal-crypto-trusted-live",
        market_id="market-1",
        direction="buy_yes",
        source="crypto",
        payload_json={
            "strategy_context": {"timeframe": "5m"},
            "live_market": {
                "live_selected_price": 0.52,
                "fetched_at": now_iso,
                "market_data_source": "http_batch",
            },
        },
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
        strategy_params={"max_market_data_age_ms": 1000},
    )

    assert result["final_decision"] == "selected"
    assert any(
        gate["gate"] == "live_market_revalidation" and gate["status"] == "passed"
        for gate in result["platform_gates"]
    )


def test_live_market_revalidation_keeps_explicit_zero_age_without_fallback():
    runtime_signal = SimpleNamespace(
        id="signal-scanner-zero-age",
        market_id="market-1",
        direction="buy_yes",
        source="scanner",
        payload_json={
            "live_market": {
                "live_selected_price": 0.52,
                "market_data_source": "http_batch",
                "market_data_age_ms": 0.0,
                "source_observed_at": "2020-01-01T00:00:00Z",
                "fetched_at": "2020-01-01T00:00:00Z",
            }
        },
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
        strategy_params={"max_market_data_age_ms": 1000},
    )

    assert result["final_decision"] == "selected"
    assert any(
        gate["gate"] == "live_market_revalidation" and gate["status"] == "passed"
        for gate in result["platform_gates"]
    )


def test_strict_ws_pricing_blocks_non_ws_market_source():
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_signal = SimpleNamespace(
        id="signal-strict-ws-blocked",
        market_id="market-1",
        direction="buy_yes",
        source="crypto",
        payload_json={
            "strategy_context": {"timeframe": "5m"},
            "live_market": {
                "live_selected_price": 0.52,
                "fetched_at": now_iso,
                "market_data_source": "http_batch",
                "market_data_age_ms": 5,
            },
        },
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
        strategy_params={
            "require_strict_ws_pricing": True,
            "strict_ws_price_sources": ["ws_strict"],
            "max_market_data_age_ms": 1000,
        },
    )

    assert result["final_decision"] == "blocked"
    assert "strict ws pricing required" in result["final_reason"].lower()
    assert any(g["gate"] == "strict_ws_pricing" and g["status"] == "blocked" for g in result["platform_gates"])


def test_strict_ws_pricing_allows_ws_source_with_fresh_age():
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_signal = SimpleNamespace(
        id="signal-strict-ws-passed",
        market_id="market-1",
        direction="buy_yes",
        source="crypto",
        payload_json={
            "strategy_context": {"timeframe": "5m"},
            "live_market": {
                "live_selected_price": 0.52,
                "fetched_at": now_iso,
                "market_data_source": "ws_strict",
                "market_data_age_ms": 12,
            },
        },
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
        strategy_params={
            "require_strict_ws_pricing": True,
            "strict_ws_price_sources": ["ws_strict"],
            "max_market_data_age_ms": 1000,
        },
    )

    assert result["final_decision"] == "selected"
    assert any(g["gate"] == "strict_ws_pricing" and g["status"] == "passed" for g in result["platform_gates"])


def test_strict_ws_pricing_blocks_scanner_signal_when_age_exceeds_strategy_budget():
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_signal = SimpleNamespace(
        id="signal-strict-ws-scanner-stale",
        market_id="market-1",
        direction="buy_no",
        source="scanner",
        payload_json={
            "strategy_context": {"source": "scanner"},
            "live_market": {
                "live_selected_price": 0.905,
                "fetched_at": now_iso,
                "source_observed_at": now_iso,
                "market_data_source": "ws_strict",
                "market_data_age_ms": 16000,
            },
        },
    )
    result = apply_platform_decision_gates(
        decision_obj=_decision(2.25),
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
        strategy_params={
            "require_strict_ws_pricing": True,
            "require_live_market_revalidation": True,
            "strict_ws_price_sources": ["ws_strict"],
            "max_market_data_age_ms": 15000,
        },
    )

    assert result["final_decision"] == "blocked"
    assert "strict ws pricing required" in result["final_reason"].lower()
    assert any(g["gate"] == "strict_ws_pricing" and g["status"] == "blocked" for g in result["platform_gates"])


def test_backtest_mode_skips_live_execution_freshness_gates():
    runtime_signal = SimpleNamespace(
        id="signal-backtest-freshness",
        market_id="market-1",
        direction="buy_yes",
        source="scanner",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
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
        pending_live_exit_summary={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}},
        portfolio_allocator=None,
        risk_evaluator=_risk_evaluator,
        invoke_hooks=False,
        strategy_params={
            "require_strict_ws_pricing": True,
            "require_live_market_revalidation": True,
            "max_market_data_age_ms": 1000,
        },
        execution_mode="backtest",
    )

    assert result["final_decision"] == "selected"
    assert any(g["gate"] == "strict_ws_pricing" and g["status"] == "skipped" for g in result["platform_gates"])
    assert any(
        g["gate"] == "live_market_revalidation" and g["status"] == "skipped"
        for g in result["platform_gates"]
    )
    assert any(
        g["gate"] == "market_data_freshness" and g["status"] == "skipped"
        for g in result["platform_gates"]
    )
    strict_ws_check = next(check for check in result["checks_payload"] if check["check_key"] == "strict_ws_pricing")
    assert strict_ws_check["passed"] is True
    assert "Skipped in backtest mode" in str(strict_ws_check["detail"])


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


def test_generic_skipped_reason_includes_failed_check_detail():
    decision_obj = SimpleNamespace(
        decision="skipped",
        reason="Crypto worker filters not met",
        score=0.2,
        size_usd=10.0,
    )
    result = apply_platform_decision_gates(
        decision_obj=decision_obj,
        runtime_signal=_runtime_signal(),
        strategy=None,
        checks_payload=[
            {
                "check_key": "confidence",
                "check_label": "Confidence",
                "passed": False,
                "score": 0.41,
                "detail": "min=0.43",
                "payload": {},
            }
        ],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=None,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "skipped"
    assert result["final_reason"].startswith("Crypto worker filters not met")
    assert "Confidence: min=0.43" in result["final_reason"]


def test_generic_risk_block_reason_includes_failed_risk_detail():
    def _blocked_risk_evaluator(_: float):
        return (
            SimpleNamespace(
                allowed=False,
                reason="Risk blocked: trader_open_positions",
                checks=[
                    SimpleNamespace(
                        key="trader_open_positions",
                        passed=False,
                        score=2.0,
                        detail="next=2 max=1",
                    )
                ],
            ),
            {},
        )

    result = apply_platform_decision_gates(
        decision_obj=_decision(20.0),
        runtime_signal=_runtime_signal(),
        strategy=None,
        checks_payload=[],
        trading_schedule_ok=True,
        trading_schedule_config={},
        global_limits={"max_gross_exposure_usd": 5000.0},
        effective_risk_limits={"max_trade_notional_usd": 1000.0},
        allow_averaging=True,
        open_market_ids=set(),
        portfolio_allocator=None,
        risk_evaluator=_blocked_risk_evaluator,
        invoke_hooks=False,
    )

    assert result["final_decision"] == "blocked"
    assert result["final_reason"].startswith("Risk blocked: trader_open_positions")
    assert "next=2 max=1" in result["final_reason"]
