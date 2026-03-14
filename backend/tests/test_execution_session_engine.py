import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator import session_engine as session_engine_module
from utils.utcnow import utcnow


def _leg_result(
    *,
    leg_id: str,
    status: str,
    notional_usd: float = 0.0,
    shares: float = 0.0,
    provider_order_id: str | None = None,
    provider_clob_order_id: str | None = None,
    error_message: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        leg_id=leg_id,
        status=status,
        effective_price=0.41,
        error_message=(
            error_message
            if error_message is not None
            else (None if status != "failed" else "submission_failed")
        ),
        payload={"provider": "test"},
        provider_order_id=provider_order_id,
        provider_clob_order_id=provider_clob_order_id,
        shares=shares,
        notional_usd=notional_usd,
    )


@pytest.mark.asyncio
async def test_execute_signal_aborts_before_order_writes_on_pair_lock_violation(monkeypatch):
    db = SimpleNamespace()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "SEQUENTIAL_HEDGE", "plan_id": "plan-1"}
    legs = [
        {
            "leg_id": "leg-1",
            "side": "buy",
            "requested_notional_usd": 100.0,
            "limit_price": 0.41,
        }
    ]
    constraints = {
        "max_unhedged_notional_usd": 10.0,
        "hedge_timeout_seconds": 20,
    }
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    buffer_outcome_mock = AsyncMock()
    publish_signal_status_mock = AsyncMock()
    cancel_provider_mock = AsyncMock(return_value=True)

    monkeypatch.setattr(session_engine_module.hot_state, "buffer_execution_outcome", buffer_outcome_mock)
    monkeypatch.setattr(engine, "_publish_hot_signal_status", publish_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "cancel_live_provider_order", cancel_provider_mock)
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: True)
    monkeypatch.setattr(
        session_engine_module,
        "submit_execution_wave",
        AsyncMock(
            return_value=[
                _leg_result(
                    leg_id="leg-1",
                    status="submitted",
                    notional_usd=100.0,
                    shares=200.0,
                    provider_order_id="provider-1",
                    provider_clob_order_id="clob-1",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value={"session": {"unhedged_notional_usd": 100.0}}),
    )

    signal = SimpleNamespace(
        id="signal-1",
        source="scanner",
        trace_id="trace-1",
        strategy_type="tail_end_carry",
        strategy_context_json={},
        payload_json={},
        market_id="market-1",
        market_question="question",
        direction="buy_yes",
        entry_price=0.41,
        edge_percent=4.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-1",
        signal=signal,
        decision_id="decision-1",
        strategy_key="tail_end_carry",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=100.0,
        reason="test",
    )

    assert result.status == "failed"
    assert "Pair lock violation" in str(result.error_message)
    assert result.orders_written == 0
    assert cancel_provider_mock.await_count == 1
    assert buffer_outcome_mock.await_count == 1
    buffered = buffer_outcome_mock.await_args.kwargs
    assert buffered["session_row"].status == "failed"
    assert buffered["signal_status"] == "failed"
    assert buffered["trader_orders"] == []
    assert any(row.status == "cancelled" for row in buffered["leg_rows"])
    assert publish_signal_status_mock.await_args.kwargs["status"] == "failed"


@pytest.mark.asyncio
async def test_execute_signal_sets_hedging_timeout_payload(monkeypatch):
    db = SimpleNamespace()
    engine = session_engine_module.ExecutionSessionEngine(db)
    strategy_params = {
        "max_market_data_age_ms": 250,
        "take_profit_pct": 4.5,
    }

    plan = {"policy": "SEQUENTIAL_HEDGE", "plan_id": "plan-2"}
    legs = [
        {
            "leg_id": "leg-1",
            "side": "buy",
            "requested_notional_usd": 80.0,
            "limit_price": 0.4,
        },
        {
            "leg_id": "leg-2",
            "side": "sell",
            "requested_notional_usd": 80.0,
            "limit_price": 0.6,
        },
    ]
    constraints = {
        "max_unhedged_notional_usd": 0.0,
        "hedge_timeout_seconds": 33,
    }
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    buffer_outcome_mock = AsyncMock()
    publish_signal_status_mock = AsyncMock()

    monkeypatch.setattr(session_engine_module.hot_state, "buffer_execution_outcome", buffer_outcome_mock)
    monkeypatch.setattr(engine, "_publish_hot_signal_status", publish_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    monkeypatch.setattr(
        session_engine_module,
        "submit_execution_wave",
        AsyncMock(
            return_value=[
                _leg_result(leg_id="leg-1", status="executed", notional_usd=80.0, shares=195.0),
                _leg_result(leg_id="leg-2", status="failed", notional_usd=0.0, shares=0.0),
            ]
        ),
    )
    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value={"session": {"unhedged_notional_usd": 0.0}}),
    )

    signal = SimpleNamespace(
        id="signal-2",
        source="scanner",
        trace_id="trace-2",
        strategy_type="tail_end_carry",
        strategy_context_json={},
        payload_json={},
        market_id="market-2",
        market_question="question",
        direction="buy_yes",
        entry_price=0.41,
        edge_percent=4.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-2",
        signal=signal,
        decision_id="decision-2",
        strategy_key="tail_end_carry",
        strategy_version=None,
        strategy_params=strategy_params,
        risk_limits={},
        mode="paper",
        size_usd=160.0,
        reason="test-hedging",
    )

    assert result.status == "hedging"
    assert result.orders_written == 2
    assert buffer_outcome_mock.await_count == 1
    buffered = buffer_outcome_mock.await_args.kwargs
    assert buffered["session_row"].status == "hedging"
    assert len(buffered["trader_orders"]) == 2
    assert buffered["trader_orders"][0].payload_json["strategy_params"] == strategy_params
    payload_patch = buffered["session_row"].payload_json
    assert payload_patch["hedge_timeout_seconds"] == 33
    assert payload_patch["hedging_escalation"] == "auto_fail_on_timeout"
    assert isinstance(payload_patch.get("hedging_started_at"), str)
    assert isinstance(payload_patch.get("hedging_deadline_at"), str)
    assert publish_signal_status_mock.await_args.kwargs["status"] == "submitted"


@pytest.mark.asyncio
async def test_execute_signal_skips_position_cap_failures_without_order_writes(monkeypatch):
    db = SimpleNamespace()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "SINGLE_LEG", "plan_id": "plan-skip"}
    legs = [
        {
            "leg_id": "leg-1",
            "side": "buy",
            "requested_notional_usd": 100.0,
            "limit_price": 0.5,
        }
    ]
    constraints = {
        "max_unhedged_notional_usd": 0.0,
        "hedge_timeout_seconds": 20,
    }
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    buffer_outcome_mock = AsyncMock()
    publish_signal_status_mock = AsyncMock()

    monkeypatch.setattr(session_engine_module.hot_state, "buffer_execution_outcome", buffer_outcome_mock)
    monkeypatch.setattr(engine, "_publish_hot_signal_status", publish_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    monkeypatch.setattr(
        session_engine_module,
        "submit_execution_wave",
        AsyncMock(
            return_value=[
                _leg_result(
                    leg_id="leg-1",
                    status="failed",
                    notional_usd=0.0,
                    shares=0.0,
                    error_message="Maximum open positions reached",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value={"session": {"unhedged_notional_usd": 0.0}}),
    )

    signal = SimpleNamespace(
        id="signal-skip",
        source="scanner",
        trace_id="trace-skip",
        strategy_type="late_favorite_alpha",
        strategy_context_json={},
        payload_json={},
        market_id="market-skip",
        market_question="question",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=1.0,
        confidence=0.55,
    )
    result = await engine.execute_signal(
        trader_id="trader-skip",
        signal=signal,
        decision_id="decision-skip",
        strategy_key="late_favorite_alpha",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=100.0,
        reason="test-skip",
    )

    assert result.status == "skipped"
    assert result.orders_written == 0
    assert buffer_outcome_mock.await_count == 1
    buffered = buffer_outcome_mock.await_args.kwargs
    assert buffered["session_row"].status == "skipped"
    assert buffered["trader_orders"] == []
    assert any(row.status == "skipped" for row in buffered["leg_rows"])
    assert publish_signal_status_mock.await_args.kwargs["status"] == "skipped"


@pytest.mark.asyncio
async def test_reconcile_active_sessions_escalates_hedging_timeout(monkeypatch):
    db = SimpleNamespace(commit=AsyncMock())
    engine = session_engine_module.ExecutionSessionEngine(db)

    past_start = utcnow() - timedelta(seconds=90)
    active_row = SimpleNamespace(
        id="sess-timeout",
        status="hedging",
        expires_at=utcnow() + timedelta(minutes=30),
        payload_json={
            "hedge_timeout_seconds": 10,
            "hedging_started_at": past_start.isoformat().replace("+00:00", "Z"),
        },
        created_at=utcnow() - timedelta(minutes=5),
        updated_at=utcnow() - timedelta(minutes=4),
    )

    monkeypatch.setattr(
        session_engine_module,
        "list_active_execution_sessions",
        AsyncMock(return_value=[active_row]),
    )
    create_event_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "create_execution_session_event", create_event_mock)
    cancel_session_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(engine, "cancel_session", cancel_session_mock)
    leg_rollups_mock = AsyncMock(
        return_value={"sess-timeout": {"legs_total": 0, "legs_completed": 0, "legs_failed": 0, "legs_open": 0}}
    )
    monkeypatch.setattr(session_engine_module, "get_execution_session_leg_rollups", leg_rollups_mock)

    result = await engine.reconcile_active_sessions(mode="live", trader_id="trader-1")

    assert result["active_seen"] == 1
    assert result["failed"] == 1
    assert cancel_session_mock.await_count == 1
    assert cancel_session_mock.await_args.kwargs["terminal_status"] == "failed"
    assert "Hedge timeout exceeded" in cancel_session_mock.await_args.kwargs["reason"]
    assert create_event_mock.await_count == 1
    assert create_event_mock.await_args.kwargs["event_type"] == "hedge_timeout"


@pytest.mark.asyncio
async def test_cancel_session_skips_already_terminal_trader_orders(monkeypatch):
    db = SimpleNamespace(commit=AsyncMock(), execute=AsyncMock())
    engine = session_engine_module.ExecutionSessionEngine(db)

    session_detail = {
        "session": {"status": "working", "signal_id": "signal-1"},
        "orders": [
            {
                "status": "open",
                "trader_order_id": "order-1",
                "provider_order_id": "provider-1",
                "provider_clob_order_id": "clob-1",
            }
        ],
        "legs": [],
    }
    terminal_order = SimpleNamespace(
        id="order-1",
        status="cancelled",
        payload_json={},
        reason="cleanup:max_open_order_timeout:crypto",
        notional_usd=0.0,
        executed_at=None,
        updated_at=None,
    )
    db.execute = AsyncMock(
        return_value=SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [terminal_order])
        )
    )

    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value=session_detail),
    )
    cancel_provider_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(session_engine_module, "cancel_live_provider_order", cancel_provider_mock)
    update_leg_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "update_execution_leg", update_leg_mock)
    update_status_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "update_execution_session_status", update_status_mock)
    set_signal_status_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", set_signal_status_mock)
    create_event_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "create_execution_session_event", create_event_mock)
    publish_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module.event_bus, "publish", publish_mock)

    result = await engine.cancel_session(
        session_id="session-1",
        reason="Session timed out before all legs completed.",
        terminal_status="expired",
    )

    assert result is True
    assert cancel_provider_mock.await_count == 0
    assert db.execute.await_count == 1
    assert update_leg_mock.await_count == 0
    assert update_status_mock.await_count == 1
    assert set_signal_status_mock.await_count == 1
    assert create_event_mock.await_count == 1

