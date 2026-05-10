import asyncio
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
from services import intent_runtime as intent_runtime_module
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
    effective_price: float = 0.41,
    payload: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        leg_id=leg_id,
        status=status,
        effective_price=effective_price,
        error_message=(
            error_message
            if error_message is not None
            else (None if status != "failed" else "submission_failed")
        ),
        payload=payload if payload is not None else {"provider": "test"},
        provider_order_id=provider_order_id,
        provider_clob_order_id=provider_clob_order_id,
        shares=shares,
        notional_usd=notional_usd,
    )


class _FailureProjectionDb:
    def __init__(self) -> None:
        self.pending: list[object] = []
        self.persisted_session_ids: set[str] = set()
        self.persisted_leg_ids: set[str] = set()
        self.persisted_trader_order_ids: set[str] = set()
        self.persisted_rows_by_type: dict[str, list[object]] = {}
        self.flush_calls = 0
        self.commit_calls = 0

    def add(self, row: object) -> None:
        self.pending.append(row)

    async def flush(self) -> None:
        self.flush_calls += 1
        for row in self.pending:
            row_type = row.__class__.__name__
            persisted_rows = self.persisted_rows_by_type.setdefault(row_type, [])
            row_id = str(getattr(row, "id", "") or "").strip()
            if row_id:
                replaced = False
                for index, existing_row in enumerate(persisted_rows):
                    if str(getattr(existing_row, "id", "") or "").strip() == row_id:
                        persisted_rows[index] = row
                        replaced = True
                        break
                if not replaced:
                    persisted_rows.append(row)
            else:
                persisted_rows.append(row)
            if row_type == "ExecutionSession":
                self.persisted_session_ids.add(str(getattr(row, "id", "") or ""))
            elif row_type == "ExecutionSessionLeg":
                self.persisted_leg_ids.add(str(getattr(row, "id", "") or ""))
            elif row_type == "TraderOrder":
                self.persisted_trader_order_ids.add(str(getattr(row, "id", "") or ""))
        for row in self.pending:
            if not isinstance(row, session_engine_module.ExecutionSessionOrder):
                continue
            assert str(row.session_id or "") in self.persisted_session_ids
            assert str(row.leg_id or "") in self.persisted_leg_ids
            trader_order_id = str(row.trader_order_id or "").strip()
            if trader_order_id:
                assert trader_order_id in self.persisted_trader_order_ids
        self.pending.clear()

    async def commit(self) -> None:
        self.commit_calls += 1


class _ForeignKeyStrictProjectionDb(_FailureProjectionDb):
    async def flush(self) -> None:
        self.flush_calls += 1
        persisted_trader_order_ids_before_flush = set(self.persisted_trader_order_ids)
        for row in self.pending:
            if not isinstance(row, session_engine_module.ExecutionSessionOrder):
                continue
            trader_order_id = str(row.trader_order_id or "").strip()
            if trader_order_id:
                assert trader_order_id in persisted_trader_order_ids_before_flush
        for row in self.pending:
            row_type = row.__class__.__name__
            persisted_rows = self.persisted_rows_by_type.setdefault(row_type, [])
            row_id = str(getattr(row, "id", "") or "").strip()
            if row_id:
                replaced = False
                for index, existing_row in enumerate(persisted_rows):
                    if str(getattr(existing_row, "id", "") or "").strip() == row_id:
                        persisted_rows[index] = row
                        replaced = True
                        break
                if not replaced:
                    persisted_rows.append(row)
            else:
                persisted_rows.append(row)
            if row_type == "ExecutionSession":
                self.persisted_session_ids.add(str(getattr(row, "id", "") or ""))
            elif row_type == "ExecutionSessionLeg":
                self.persisted_leg_ids.add(str(getattr(row, "id", "") or ""))
            elif row_type == "TraderOrder":
                self.persisted_trader_order_ids.add(str(getattr(row, "id", "") or ""))
        for row in self.pending:
            if not isinstance(row, session_engine_module.ExecutionSessionOrder):
                continue
            assert str(row.session_id or "") in self.persisted_session_ids
            assert str(row.leg_id or "") in self.persisted_leg_ids
            trader_order_id = str(row.trader_order_id or "").strip()
            if trader_order_id:
                assert trader_order_id in self.persisted_trader_order_ids
        self.pending.clear()


def test_build_execution_session_rows_carries_market_roster_into_session_payload():
    signal = SimpleNamespace(
        id="signal-roster",
        source="scanner",
        market_id="market-home",
        market_question="Will home team win?",
        payload_json={
            "markets": [
                {"id": "market-home", "question": "Will home team win?"},
                {"id": "market-draw", "question": "Will the match end in a draw?"},
                {"id": "market-away", "question": "Will away team win?"},
            ],
            "market_roster": {
                "scope": "event",
                "event_id": "event-atomic",
                "event_slug": "event-atomic",
                "event_title": "Atomic roster event",
                "category": "Sports",
                "market_count": 3,
                "roster_hash": "hash1234",
                "markets": [
                    {"id": "market-home", "question": "Will home team win?"},
                    {"id": "market-draw", "question": "Will the match end in a draw?"},
                    {"id": "market-away", "question": "Will away team win?"},
                ],
            },
        },
    )

    row, leg_rows = session_engine_module.build_execution_session_rows(
        trader_id="trader-roster",
        signal=signal,
        decision_id="decision-roster",
        strategy_key="settlement_lag",
        strategy_version=None,
        mode="live",
        policy="PARALLEL_MAKER",
        plan_id="plan-roster",
        legs=[
            {
                "leg_id": "leg-home",
                "market_id": "market-home",
                "market_question": "Will home team win?",
                "token_id": "token-home",
                "side": "buy",
                "outcome": "yes",
                "limit_price": 0.4,
            }
        ],
        requested_notional_usd=10.0,
        max_unhedged_notional_usd=0.0,
        expires_at=utcnow() + timedelta(minutes=5),
        payload={"execution_plan": {"plan_id": "plan-roster", "policy": "PARALLEL_MAKER", "legs": []}},
        trace_id="trace-roster",
    )

    assert len(leg_rows) == 1
    assert row.payload_json["market_roster"]["market_count"] == 3
    assert {market["id"] for market in row.payload_json["market_roster"]["markets"]} == {
        "market-home",
        "market-draw",
        "market-away",
    }


def test_build_plan_normalizes_limit_sell_action_from_position_payload():
    engine = session_engine_module.ExecutionSessionEngine(_FailureProjectionDb())
    signal = SimpleNamespace(
        id="signal-side-normalization",
        source="scanner",
        strategy_type="generic_strategy",
        payload_json={
            "positions_to_take": [
                {
                    "action": "LIMIT_BUY",
                    "market_id": "market-1",
                    "token_id": "token-1",
                    "outcome": "YES",
                    "price": 0.44,
                },
                {
                    "action": "LIMIT_SELL",
                    "market_id": "market-1",
                    "token_id": "token-1",
                    "outcome": "YES",
                    "price": 0.56,
                },
            ],
            "execution_plan": {
                "policy": "PARALLEL_MAKER",
                "legs": [
                    {
                        "leg_id": "leg-1",
                        "market_id": "market-1",
                        "token_id": "token-1",
                        "side": "buy",
                        "outcome": "yes",
                        "limit_price": 0.44,
                        "metadata": {"position_index": 0, "raw_action": "limit_buy"},
                    },
                    {
                        "leg_id": "leg-2",
                        "market_id": "market-1",
                        "token_id": "token-1",
                        "side": "buy",
                        "outcome": "yes",
                        "limit_price": 0.56,
                        "metadata": {"position_index": 1, "raw_action": "limit_sell"},
                    },
                ],
            },
        },
        market_id="market-1",
        market_question="Will generic event happen?",
        direction="buy_yes",
        entry_price=0.44,
    )

    _plan, legs, _constraints = engine._build_plan(
        signal,
        strategy_key="generic_strategy",
        size_usd=10.0,
        risk_limits={},
    )

    assert [leg["side"] for leg in legs] == ["buy", "sell"]


def test_resolve_leg_direction_uses_explicit_direction_when_present():
    direction = session_engine_module._resolve_leg_direction(
        {"direction": "buy_yes", "side": "sell", "outcome": "no"},
        fallback_direction="buy_no",
    )
    assert direction == "buy_yes"


def test_resolve_leg_direction_builds_buy_yes_from_side_and_outcome_when_direction_empty():
    direction = session_engine_module._resolve_leg_direction(
        {"direction": "", "side": "buy", "outcome": "yes"},
        fallback_direction="",
    )
    assert direction == "buy_yes"


def test_resolve_leg_direction_builds_buy_no_from_side_and_outcome_when_direction_empty():
    direction = session_engine_module._resolve_leg_direction(
        {"direction": "", "side": "buy", "outcome": "no"},
        fallback_direction="",
    )
    assert direction == "buy_no"


def test_resolve_leg_direction_falls_back_to_bare_buy_for_non_binary_outcome():
    direction = session_engine_module._resolve_leg_direction(
        {"direction": "", "side": "buy", "outcome": "fighter a"},
        fallback_direction="",
    )
    assert direction == "buy"


@pytest.mark.asyncio
async def test_execute_signal_rejects_self_crossing_same_token_plan(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "PARALLEL_MAKER", "plan_id": "plan-self-cross", "metadata": {}}
    legs = [
        {
            "leg_id": "leg-1",
            "market_id": "market-self-cross",
            "market_question": "Will generic event happen?",
            "token_id": "token-self-cross",
            "side": "buy",
            "outcome": "yes",
            "requested_notional_usd": 5.0,
            "requested_shares": 10.0,
            "limit_price": 0.55,
            "price_policy": "maker_limit",
            "time_in_force": "GTC",
        },
        {
            "leg_id": "leg-2",
            "market_id": "market-self-cross",
            "market_question": "Will generic event happen?",
            "token_id": "token-self-cross",
            "side": "sell",
            "outcome": "yes",
            "requested_notional_usd": 5.0,
            "requested_shares": 10.0,
            "limit_price": 0.55,
            "price_policy": "maker_limit",
            "time_in_force": "GTC",
        },
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    submit_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(session_engine_module, "submit_execution_wave", submit_mock)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_publish_hot_signal_status", AsyncMock(return_value=None))

    signal = SimpleNamespace(
        id="signal-self-cross",
        source="scanner",
        trace_id="trace-self-cross",
        strategy_type="generic_strategy",
        strategy_context_json={},
        payload_json={},
        market_id="market-self-cross",
        market_question="Will generic event happen?",
        direction="buy_yes",
        entry_price=0.55,
        edge_percent=5.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-self-cross",
        signal=signal,
        decision_id="decision-self-cross",
        strategy_key="generic_strategy",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=10.0,
        reason="generic signal selected",
    )

    assert result.status == "failed"
    assert result.orders_written == 0
    assert result.payload["self_crossing_plan"]["reason"] == "self_crossing_quote"
    assert submit_mock.await_count == 0
    session_rows = db.persisted_rows_by_type.get("ExecutionSession") or []
    assert session_rows[0].payload_json["self_crossing_plan"]["reason"] == "self_crossing_quote"


@pytest.mark.asyncio
async def test_execute_signal_flushes_new_trader_orders_before_execution_orders(monkeypatch):
    db = _ForeignKeyStrictProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "SINGLE_LEG", "plan_id": "plan-strict-flush-order"}
    legs = [
        {
            "leg_id": "leg-strict-flush-order-1",
            "market_id": "market-strict-flush-order-1",
            "market_question": "Will strict flush ordering hold?",
            "token_id": "token-strict-flush-order-1",
            "side": "buy",
            "outcome": "yes",
            "requested_notional_usd": 10.0,
            "requested_shares": 20.0,
            "limit_price": 0.5,
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "post_only": False,
        }
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_publish_hot_signal_status", AsyncMock(return_value=None))
    monkeypatch.setattr(
        session_engine_module,
        "submit_execution_wave",
        AsyncMock(
            return_value=[
                _leg_result(
                    leg_id="leg-strict-flush-order-1",
                    status="executed",
                    notional_usd=10.0,
                    shares=20.0,
                    provider_order_id="provider-strict-flush-order-1",
                    provider_clob_order_id="clob-strict-flush-order-1",
                    effective_price=0.5,
                    payload={
                        "provider": "test",
                        "token_id": "token-strict-flush-order-1",
                        "filled_size": 20.0,
                        "average_fill_price": 0.5,
                        "filled_notional_usd": 10.0,
                    },
                )
            ]
        ),
    )

    signal = SimpleNamespace(
        id="signal-strict-flush-order",
        source="scanner",
        trace_id="trace-strict-flush-order",
        strategy_type="tail_end_carry",
        strategy_context_json={},
        payload_json={},
        market_id="market-strict-flush-order-1",
        market_question="Will strict flush ordering hold?",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=4.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-strict-flush-order",
        signal=signal,
        decision_id="decision-strict-flush-order",
        strategy_key="tail_end_carry",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="shadow",
        size_usd=10.0,
        reason="strict-flush-order",
    )

    assert result.status == "completed"
    assert result.orders_written == 1
    trader_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    execution_rows = db.persisted_rows_by_type.get("ExecutionSessionOrder") or []
    assert len(trader_rows) == 1
    assert len(execution_rows) == 1
    assert str(execution_rows[0].trader_order_id or "") == str(trader_rows[0].id or "")


@pytest.mark.asyncio
async def test_execute_signal_aborts_before_order_writes_on_pair_lock_violation(monkeypatch):
    db = _FailureProjectionDb()
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
    cancel_provider_mock = AsyncMock(return_value=True)

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
    set_signal_status_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", set_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

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
    assert db.flush_calls >= 1
    assert set_signal_status_mock.await_count == 1
    assert set_signal_status_mock.await_args.args[2] == "failed"
    assert intent_runtime.update_signal_status.await_count == 1
    assert intent_runtime.update_signal_status.await_args.kwargs["status"] == "failed"


@pytest.mark.asyncio
async def test_execute_signal_sets_hedging_timeout_payload(monkeypatch):
    db = _FailureProjectionDb()
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
    publish_signal_status_mock = AsyncMock()

    monkeypatch.setattr(engine, "_publish_hot_signal_status", publish_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
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
        explicit_strategy_params={"max_market_data_age_ms": 250},
        risk_limits={},
        mode="shadow",
        size_usd=160.0,
        reason="test-hedging",
    )

    assert result.status == "hedging"
    assert result.orders_written == 2
    session_rows = db.persisted_rows_by_type.get("ExecutionSession") or []
    order_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    assert len(session_rows) == 1
    assert len(order_rows) == 2
    assert session_rows[0].status == "hedging"
    assert order_rows[0].payload_json["strategy_params"] == strategy_params
    assert "take_profit_pct" not in order_rows[0].payload_json["strategy_exit_config"]
    assert order_rows[0].payload_json["strategy_exit_config"]["max_market_data_age_ms"] == 250
    payload_patch = session_rows[0].payload_json
    assert payload_patch["hedge_timeout_seconds"] == 33
    assert payload_patch["hedging_escalation"] == "auto_fail_on_timeout"
    assert isinstance(payload_patch.get("hedging_started_at"), str)
    assert isinstance(payload_patch.get("hedging_deadline_at"), str)
    assert publish_signal_status_mock.await_args.kwargs["status"] == "submitted"


@pytest.mark.asyncio
async def test_execute_signal_persists_live_submit_placeholder_before_provider_call(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "SINGLE_LEG", "plan_id": "plan-live-placeholder"}
    legs = [
        {
            "leg_id": "leg-live-1",
            "market_id": "market-live-1",
            "market_question": "Will Example close above 50?",
            "token_id": "token-live-1",
            "side": "buy",
            "outcome": "yes",
            "requested_notional_usd": 10.0,
            "requested_shares": 25.0,
            "limit_price": 0.4,
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "post_only": False,
        }
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_publish_hot_signal_status", AsyncMock(return_value=None))

    pre_submit_observed: dict[str, list[str]] = {}

    async def _submit_wave(**kwargs):
        trader_rows = db.persisted_rows_by_type.get("TraderOrder") or []
        execution_rows = db.persisted_rows_by_type.get("ExecutionSessionOrder") or []
        pre_submit_observed["trader_statuses"] = [str(row.status or "") for row in trader_rows]
        pre_submit_observed["submission_states"] = [
            str(dict(getattr(row, "payload_json", {}) or {}).get("submission_intent", {}).get("state") or "")
            for row in trader_rows
        ]
        pre_submit_observed["execution_statuses"] = [str(row.status or "") for row in execution_rows]
        assert any(str(row.status or "") == "placing" for row in trader_rows)
        assert any(str(row.status or "") == "placing" for row in execution_rows)
        return [
            _leg_result(
                leg_id="leg-live-1",
                status="executed",
                notional_usd=10.0,
                shares=25.0,
                provider_order_id="provider-live-1",
                provider_clob_order_id="clob-live-1",
                effective_price=0.4,
                payload={
                    "provider": "test",
                    "token_id": "token-live-1",
                    "filled_size": 25.0,
                    "average_fill_price": 0.4,
                    "filled_notional_usd": 10.0,
                },
            )
        ]

    monkeypatch.setattr(session_engine_module, "submit_execution_wave", AsyncMock(side_effect=_submit_wave))

    signal = SimpleNamespace(
        id="signal-live-placeholder",
        source="scanner",
        trace_id="trace-live-placeholder",
        strategy_type="tail_end_carry",
        strategy_context_json={},
        payload_json={},
        market_id="market-live-1",
        market_question="Will Example close above 50?",
        direction="buy_yes",
        entry_price=0.4,
        edge_percent=4.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-live-placeholder",
        signal=signal,
        decision_id="decision-live-placeholder",
        strategy_key="tail_end_carry",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=10.0,
        reason="Tail carry signal selected",
    )

    assert result.status == "completed"
    assert db.commit_calls >= 1
    assert pre_submit_observed["trader_statuses"] == ["placing"]
    assert pre_submit_observed["submission_states"] == ["pre_submit_persisted"]
    assert pre_submit_observed["execution_statuses"] == ["placing"]
    trader_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    execution_rows = db.persisted_rows_by_type.get("ExecutionSessionOrder") or []
    assert len({str(row.id or "") for row in trader_rows}) == 1
    assert len({str(row.id or "") for row in execution_rows}) == 1
    final_order = trader_rows[-1]
    final_execution_order = execution_rows[-1]
    assert str(final_order.status or "") == "open"
    assert str(final_order.provider_clob_order_id or "") == "clob-live-1"
    assert dict(final_order.payload_json or {}).get("submission_intent", {}).get("state") == "provider_acknowledged"
    assert str(final_execution_order.status or "") == "executed"
    assert str(final_execution_order.provider_clob_order_id or "") == "clob-live-1"


@pytest.mark.asyncio
async def test_execute_signal_does_not_preplace_take_profit_exit_from_runtime_defaults(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "PARALLEL_MAKER", "plan_id": "plan-explicit-only"}
    legs = [
        {
            "leg_id": "leg-1",
            "side": "buy",
            "token_id": "token-1",
            "requested_notional_usd": 12.0,
            "limit_price": 0.4,
        }
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
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
                    status="executed",
                    notional_usd=12.0,
                    shares=30.0,
                    payload={
                        "provider": "test",
                        "token_id": "token-1",
                        "filled_size": 30.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 12.0,
                    },
                )
            ]
        ),
    )
    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value={"session": {"unhedged_notional_usd": 0.0}}),
    )
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(
        session_engine_module,
        "_strategy_instance_for_execution",
        lambda _strategy_key: SimpleNamespace(supports_entry_take_profit_exit=True),
    )
    tp_submit_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "execute_live_order", tp_submit_mock)
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "prepare_sell_balance_allowance",
        AsyncMock(return_value=None),
    )
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-explicit-only",
        source="scanner",
        trace_id="trace-explicit-only",
        strategy_type="btc_eth_maker_quote",
        strategy_context_json={},
        payload_json={},
        market_id="market-explicit-only",
        market_question="question",
        direction="buy_yes",
        entry_price=0.4,
        edge_percent=6.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-explicit-only",
        signal=signal,
        decision_id="decision-explicit-only",
        strategy_key="btc_eth_maker_quote",
        strategy_version=None,
        strategy_params={"preplace_take_profit_exit": True, "take_profit_pct": 12.0},
        explicit_strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=12.0,
        reason="explicit-only",
    )

    assert result.status == "completed"
    assert result.orders_written == 1
    assert tp_submit_mock.await_count == 0
    order_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    assert len(order_rows) == 1
    assert "pending_live_exit" not in order_rows[0].payload_json


@pytest.mark.asyncio
async def test_execute_signal_finalizes_pre_submit_placeholder_when_live_submit_is_cancelled(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "PARALLEL_MAKER", "plan_id": "plan-live-cancel"}
    legs = [
        {
            "leg_id": "leg-live-cancel-1",
            "market_id": "market-live-cancel-1",
            "market_question": "Will Example cancel cleanly?",
            "token_id": "token-live-cancel-1",
            "side": "buy",
            "outcome": "yes",
            "requested_notional_usd": 9.0,
            "requested_shares": 18.0,
            "limit_price": 0.5,
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "post_only": False,
        }
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_publish_hot_signal_status", AsyncMock(return_value=None))

    submit_started = asyncio.Event()

    async def _submit_wave(**kwargs):
        trader_rows = db.persisted_rows_by_type.get("TraderOrder") or []
        execution_rows = db.persisted_rows_by_type.get("ExecutionSessionOrder") or []
        assert [str(row.status or "") for row in trader_rows] == ["placing"]
        assert [str(row.status or "") for row in execution_rows] == ["placing"]
        submit_started.set()
        await asyncio.Future()

    monkeypatch.setattr(session_engine_module, "submit_execution_wave", AsyncMock(side_effect=_submit_wave))

    signal = SimpleNamespace(
        id="signal-live-cancel",
        source="scanner",
        trace_id="trace-live-cancel",
        strategy_type="tail_end_carry",
        strategy_context_json={},
        payload_json={},
        market_id="market-live-cancel-1",
        market_question="Will Example cancel cleanly?",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=3.0,
        confidence=0.8,
    )
    task = asyncio.create_task(
        engine.execute_signal(
            trader_id="trader-live-cancel",
            signal=signal,
            decision_id="decision-live-cancel",
            strategy_key="tail_end_carry",
            strategy_version=None,
            strategy_params={},
            risk_limits={},
            mode="live",
            size_usd=9.0,
            reason="Tail carry signal selected",
        )
    )

    await submit_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    session_rows = db.persisted_rows_by_type.get("ExecutionSession") or []
    trader_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    execution_rows = db.persisted_rows_by_type.get("ExecutionSessionOrder") or []

    assert db.commit_calls >= 1
    assert len(session_rows) == 1
    assert len(trader_rows) == 1
    assert len(execution_rows) == 1
    assert str(session_rows[-1].status or "") == "failed"
    assert bool(dict(session_rows[-1].payload_json or {}).get("submit_cancelled")) is True
    assert str(trader_rows[-1].status or "") == "failed"
    assert dict(trader_rows[-1].payload_json or {}).get("submission_intent", {}).get("state") == "submit_cancelled"
    assert str(execution_rows[-1].status or "") == "failed"


@pytest.mark.asyncio
async def test_execute_signal_preplace_take_profit_exit_requires_explicit_enable(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "PARALLEL_MAKER", "plan_id": "plan-explicit-tp"}
    legs = [
        {
            "leg_id": "leg-1",
            "side": "buy",
            "token_id": "token-1",
            "requested_notional_usd": 12.0,
            "limit_price": 0.4,
        }
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
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
                    status="executed",
                    notional_usd=12.0,
                    shares=30.0,
                    payload={
                        "provider": "test",
                        "token_id": "token-1",
                        "filled_size": 30.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 12.0,
                    },
                )
            ]
        ),
    )
    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value={"session": {"unhedged_notional_usd": 0.0}}),
    )
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(
        session_engine_module,
        "_strategy_instance_for_execution",
        lambda _strategy_key: SimpleNamespace(supports_entry_take_profit_exit=True),
    )
    tp_submit_mock = AsyncMock(
        return_value=SimpleNamespace(
            status="submitted",
            order_id="tp-order-1",
            effective_price=0.448,
            error_message=None,
            payload={"clob_order_id": "tp-clob-1", "trading_status": "live"},
        )
    )
    monkeypatch.setattr(session_engine_module, "execute_live_order", tp_submit_mock)
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "prepare_sell_balance_allowance",
        AsyncMock(return_value=None),
    )
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-explicit-tp",
        source="scanner",
        trace_id="trace-explicit-tp",
        strategy_type="btc_eth_maker_quote",
        strategy_context_json={},
        payload_json={},
        market_id="market-explicit-tp",
        market_question="question",
        direction="buy_yes",
        entry_price=0.4,
        edge_percent=6.0,
        confidence=0.7,
    )
    result = await engine.execute_signal(
        trader_id="trader-explicit-tp",
        signal=signal,
        decision_id="decision-explicit-tp",
        strategy_key="btc_eth_maker_quote",
        strategy_version=None,
        strategy_params={"preplace_take_profit_exit": True, "take_profit_pct": 12.0},
        explicit_strategy_params={"preplace_take_profit_exit": True, "take_profit_pct": 12.0},
        risk_limits={},
        mode="live",
        size_usd=12.0,
        reason="explicit-tp",
    )

    assert result.status == "completed"
    assert result.orders_written == 1
    assert tp_submit_mock.await_count == 1
    order_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    assert len(order_rows) == 1
    pending_exit = order_rows[0].payload_json.get("pending_live_exit")
    assert isinstance(pending_exit, dict)
    assert pending_exit["reason"] == "entry_bracket_take_profit"
    assert pending_exit["target_take_profit_pct"] == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_execute_signal_skips_position_cap_failures_without_order_writes(monkeypatch):
    db = _FailureProjectionDb()
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
    set_signal_status_mock = AsyncMock()
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", set_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-skip",
        source="scanner",
        trace_id="trace-skip",
        strategy_type="scanner_strategy",
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
        strategy_key="scanner_strategy",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=100.0,
        reason="test-skip",
    )

    assert result.status == "skipped"
    assert result.orders_written == 0
    assert db.flush_calls >= 1
    assert set_signal_status_mock.await_count == 1
    assert set_signal_status_mock.await_args.args[2] == "skipped"
    assert intent_runtime.update_signal_status.await_count == 1
    assert intent_runtime.update_signal_status.await_args.kwargs["status"] == "skipped"


@pytest.mark.asyncio
async def test_execute_signal_persists_leg_specific_market_metadata(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "SEQUENTIAL_HEDGE", "plan_id": "plan-leg-metadata"}
    legs = [
        {
            "leg_id": "leg-atletico",
            "side": "buy",
            "outcome": "yes",
            "market_id": "market-atletico",
            "market_question": "Will Atletico Nacional win?",
            "token_id": "token-atletico",
            "requested_notional_usd": 80.26,
            "limit_price": 0.79,
        },
        {
            "leg_id": "leg-cucuta",
            "side": "buy",
            "outcome": "yes",
            "market_id": "market-cucuta",
            "market_question": "Will Cucuta Deportivo FC win?",
            "token_id": "token-cucuta",
            "requested_notional_usd": 7.11,
            "limit_price": 0.07,
        },
    ]
    constraints = {
        "max_unhedged_notional_usd": 0.0,
        "hedge_timeout_seconds": 20,
    }
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    monkeypatch.setattr(
        session_engine_module,
        "submit_execution_wave",
        AsyncMock(
            return_value=[
                _leg_result(leg_id="leg-atletico", status="executed", notional_usd=80.26, shares=101.6),
                _leg_result(leg_id="leg-cucuta", status="executed", notional_usd=7.11, shares=101.6),
            ]
        ),
    )
    monkeypatch.setattr(
        session_engine_module,
        "get_execution_session_detail",
        AsyncMock(return_value={"session": {"unhedged_notional_usd": 0.0}}),
    )
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-leg-metadata",
        source="scanner",
        trace_id="trace-leg-metadata",
        strategy_type="settlement_lag",
        strategy_context_json={},
        payload_json={
            "markets": [
                {"market_id": "market-atletico", "question": "Will Atletico Nacional win?"},
                {"market_id": "market-cucuta", "question": "Will Cucuta Deportivo FC win?"},
            ],
            "live_market": {
                "market_id": "market-parent",
                "market_question": "Parent market question",
            },
        },
        market_id="market-parent",
        market_question="Parent market question",
        direction="buy_no",
        entry_price=0.33,
        edge_percent=13.65,
        confidence=0.5,
    )
    result = await engine.execute_signal(
        trader_id="trader-leg-metadata",
        signal=signal,
        decision_id="decision-leg-metadata",
        strategy_key="settlement_lag",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=87.37,
        reason="persist-leg-metadata",
    )

    assert result.status == "completed"
    assert result.orders_written == 2
    order_rows = db.persisted_rows_by_type.get("TraderOrder") or []
    assert len(order_rows) == 2

    rows_by_market = {str(row.market_id): row for row in order_rows}
    assert set(rows_by_market.keys()) == {"market-atletico", "market-cucuta"}

    atletico = rows_by_market["market-atletico"]
    assert atletico.market_question == "Will Atletico Nacional win?"
    assert atletico.direction == "buy_yes"
    assert atletico.entry_price == pytest.approx(0.79)
    assert atletico.payload_json["market_question"] == "Will Atletico Nacional win?"
    assert atletico.payload_json["live_market"]["market_id"] == "market-atletico"
    assert atletico.payload_json["live_market"]["selected_outcome"] == "yes"

    cucuta = rows_by_market["market-cucuta"]
    assert cucuta.market_question == "Will Cucuta Deportivo FC win?"
    assert cucuta.direction == "buy_yes"
    assert cucuta.entry_price == pytest.approx(0.07)
    assert cucuta.payload_json["market_question"] == "Will Cucuta Deportivo FC win?"
    assert cucuta.payload_json["live_market"]["market_id"] == "market-cucuta"
    assert cucuta.payload_json["live_market"]["selected_outcome"] == "yes"


@pytest.mark.asyncio
async def test_execute_signal_failed_projection_flushes_parents_before_signal_status_update(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "SINGLE_LEG", "plan_id": "plan-failed-projection"}
    legs = [
        {
            "leg_id": "leg-1",
            "side": "buy",
            "requested_notional_usd": 4.525,
            "limit_price": 0.905,
        }
    ]
    constraints = {
        "max_unhedged_notional_usd": 0.0,
        "hedge_timeout_seconds": 20,
    }
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
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
                    notional_usd=4.525,
                    shares=5.0,
                    provider_order_id="provider-failed-1",
                    error_message="submission_failed",
                )
            ]
        ),
    )

    set_signal_status_mock = AsyncMock()

    async def _set_trade_signal_status(db_session, signal_id, status, effective_price=None, commit=False):
        del signal_id, status, effective_price, commit
        pending_execution_orders = [
            row for row in db_session.pending if isinstance(row, session_engine_module.ExecutionSessionOrder)
        ]
        assert pending_execution_orders == []

    set_signal_status_mock.side_effect = _set_trade_signal_status
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", set_signal_status_mock)
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-failed-projection",
        source="scanner",
        trace_id="trace-failed-projection",
        strategy_type="tail_end_carry",
        strategy_context_json={},
        payload_json={},
        market_id="market-failed-projection",
        market_question="question",
        direction="buy_no",
        entry_price=0.905,
        edge_percent=3.0,
        confidence=0.5,
    )
    result = await engine.execute_signal(
        trader_id="trader-failed-projection",
        signal=signal,
        decision_id="decision-failed-projection",
        strategy_key="tail_end_carry",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=2.25,
        reason="failed-projection-test",
    )

    assert result.status == "failed"
    assert result.orders_written == 1
    assert db.flush_calls >= 1
    assert set_signal_status_mock.await_count == 1
    assert intent_runtime.update_signal_status.await_count == 1


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
async def test_reconcile_active_sessions_fails_stale_pre_submit_placeholders(monkeypatch):
    db = SimpleNamespace(commit=AsyncMock())
    engine = session_engine_module.ExecutionSessionEngine(db)

    stale_row = SimpleNamespace(
        id="sess-stale-submit",
        status="placing",
        expires_at=utcnow() + timedelta(minutes=30),
        payload_json={},
        created_at=utcnow() - timedelta(minutes=3),
        updated_at=utcnow() - timedelta(minutes=2),
    )

    monkeypatch.setattr(
        session_engine_module,
        "list_active_execution_sessions",
        AsyncMock(return_value=[stale_row]),
    )
    cancel_session_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(engine, "cancel_session", cancel_session_mock)
    leg_rollups_mock = AsyncMock(
        return_value={"sess-stale-submit": {"legs_total": 0, "legs_completed": 0, "legs_failed": 0, "legs_open": 0}}
    )
    monkeypatch.setattr(session_engine_module, "get_execution_session_leg_rollups", leg_rollups_mock)

    result = await engine.reconcile_active_sessions(mode="live", trader_id="trader-1")

    assert result["active_seen"] == 1
    assert result["failed"] == 1
    assert cancel_session_mock.await_count == 1
    assert cancel_session_mock.await_args.kwargs["terminal_status"] == "failed"
    assert "did not persist provider acknowledgement" in cancel_session_mock.await_args.kwargs["reason"]


@pytest.mark.asyncio
async def test_reconcile_active_sessions_skips_authority_recovery_for_ancient_stale_pre_submit_placeholders(
    monkeypatch,
):
    db = SimpleNamespace(commit=AsyncMock(), execute=AsyncMock(), get=AsyncMock(return_value=None))
    engine = session_engine_module.ExecutionSessionEngine(db)

    stale_row = SimpleNamespace(
        id="sess-ancient-stale-submit",
        trader_id="trader-1",
        status="placing",
        expires_at=utcnow() + timedelta(minutes=30),
        payload_json={},
        created_at=utcnow() - timedelta(hours=2),
        updated_at=utcnow() - timedelta(hours=2),
    )

    monkeypatch.setattr(
        session_engine_module,
        "list_active_execution_sessions",
        AsyncMock(return_value=[stale_row]),
    )
    recover_mock = AsyncMock(return_value={"recovered_orders": 0})
    monkeypatch.setattr(session_engine_module, "recover_missing_live_trader_orders", recover_mock)
    cancel_session_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(engine, "cancel_session", cancel_session_mock)
    leg_rollups_mock = AsyncMock(
        return_value={
            "sess-ancient-stale-submit": {
                "legs_total": 0,
                "legs_completed": 0,
                "legs_failed": 0,
                "legs_open": 0,
            }
        }
    )
    monkeypatch.setattr(session_engine_module, "get_execution_session_leg_rollups", leg_rollups_mock)

    result = await engine.reconcile_active_sessions(mode="live", trader_id="trader-1")

    assert result["active_seen"] == 1
    assert result["failed"] == 1
    assert recover_mock.await_count == 0
    assert cancel_session_mock.await_count == 1
    assert cancel_session_mock.await_args.kwargs["terminal_status"] == "failed"
    assert "did not persist provider acknowledgement" in cancel_session_mock.await_args.kwargs["reason"]


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


@pytest.mark.asyncio
async def test_cancel_session_fails_pre_submit_placeholder_orders(monkeypatch):
    db = SimpleNamespace(commit=AsyncMock(), execute=AsyncMock())
    engine = session_engine_module.ExecutionSessionEngine(db)

    timeout_reason = "Live order submission did not persist provider acknowledgement within 45s."
    session_detail = {
        "session": {"status": "placing", "signal_id": "signal-1", "trader_id": "trader-1", "mode": "live"},
        "orders": [
            {
                "status": "placing",
                "trader_order_id": "order-1",
                "provider_order_id": None,
                "provider_clob_order_id": None,
                "leg_id": "leg-1",
            }
        ],
        "legs": [{"id": "leg-1", "status": "placing"}],
    }
    placing_order = SimpleNamespace(
        id="order-1",
        trader_id="trader-1",
        mode="live",
        market_id="market-1",
        source="scanner",
        payload_json={"submission_intent": {"state": "pre_submit_persisted"}},
        reason="Tail carry signal selected",
        notional_usd=None,
        executed_at=None,
        updated_at=None,
        error_message=None,
        status="placing",
    )
    db.execute = AsyncMock(
        return_value=SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [placing_order])
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
        reason=timeout_reason,
        terminal_status="failed",
    )

    assert result is True
    assert cancel_provider_mock.await_count == 0
    assert placing_order.status == "failed"
    assert placing_order.error_message == timeout_reason
    assert placing_order.payload_json["submission_intent"]["state"] == "submit_timeout"
    assert placing_order.payload_json["submission_intent"]["submit_status"] == "timed_out"
    assert update_leg_mock.await_count == 1
    assert update_leg_mock.await_args.kwargs["status"] == "failed"
    assert update_status_mock.await_count == 1
    assert update_status_mock.await_args.kwargs["status"] == "failed"
    assert set_signal_status_mock.await_count == 1
    assert set_signal_status_mock.await_args.kwargs["status"] == "failed"
    assert create_event_mock.await_count == 1
    assert db.commit.await_count == 1


@pytest.mark.asyncio
async def test_execute_signal_rejects_incomplete_event_roster_before_submission(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {
        "policy": "PARALLEL_MAKER",
        "plan_id": "plan-missing-roster-leg",
        "legs": [
            {"leg_id": "leg-home", "market_id": "market-home"},
            {"leg_id": "leg-away", "market_id": "market-away"},
        ],
    }
    legs = [
        {
            "leg_id": "leg-home",
            "market_id": "market-home",
            "side": "buy",
            "requested_notional_usd": 5.0,
            "requested_shares": 10.0,
            "limit_price": 0.5,
        },
        {
            "leg_id": "leg-away",
            "market_id": "market-away",
            "side": "buy",
            "requested_notional_usd": 5.0,
            "requested_shares": 10.0,
            "limit_price": 0.5,
        },
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    submit_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(session_engine_module, "submit_execution_wave", submit_mock)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-missing-roster-leg",
        source="scanner",
        trace_id="trace-missing-roster-leg",
        strategy_type="settlement_lag",
        strategy_context_json={},
        payload_json={
            "is_guaranteed": True,
            "market_roster": {
                "scope": "event",
                "market_count": 3,
                "markets": [
                    {"id": "market-home"},
                    {"id": "market-draw"},
                    {"id": "market-away"},
                ],
            },
        },
        market_id="market-home",
        market_question="Will home team win?",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=8.0,
        confidence=0.7,
    )

    result = await engine.execute_signal(
        trader_id="trader-missing-roster-leg",
        signal=signal,
        decision_id="decision-missing-roster-leg",
        strategy_key="settlement_lag",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=10.0,
        reason="roster-gap",
    )

    assert result.status == "failed"
    assert result.orders_written == 0
    assert submit_mock.await_count == 0
    session_rows = db.persisted_rows_by_type.get("ExecutionSession") or []
    assert len(session_rows) == 1
    assert session_rows[0].status == "failed"
    assert session_rows[0].payload_json["bundle_coverage"]["missing_market_ids"] == ["market-draw"]


@pytest.mark.asyncio
async def test_execute_signal_rejects_bundle_below_minimum_executable_size(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "PARALLEL_MAKER", "plan_id": "plan-too-small", "legs": []}
    legs = [
        {
            "leg_id": "leg-yes",
            "market_id": "market-binary",
            "side": "buy",
            "requested_notional_usd": 0.13,
            "requested_shares": 8.8,
            "limit_price": 0.0145,
        },
        {
            "leg_id": "leg-no",
            "market_id": "market-binary",
            "side": "buy",
            "requested_notional_usd": 8.29,
            "requested_shares": 8.8,
            "limit_price": 0.9395,
        },
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    submit_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(session_engine_module, "submit_execution_wave", submit_mock)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-too-small",
        source="scanner",
        trace_id="trace-too-small",
        strategy_type="settlement_lag",
        strategy_context_json={},
        payload_json={"is_guaranteed": True},
        market_id="market-binary",
        market_question="Google (GOOGL) Up or Down?",
        direction="buy_yes",
        entry_price=0.0145,
        edge_percent=4.0,
        confidence=0.5,
    )

    result = await engine.execute_signal(
        trader_id="trader-too-small",
        signal=signal,
        decision_id="decision-too-small",
        strategy_key="settlement_lag",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=8.42,
        reason="too-small",
    )

    assert result.status == "failed"
    assert result.orders_written == 0
    assert submit_mock.await_count == 0
    session_rows = db.persisted_rows_by_type.get("ExecutionSession") or []
    assert len(session_rows) == 1
    payload = session_rows[0].payload_json["bundle_preflight"]
    assert payload["reason"] == "bundle_below_minimum_executable_size"
    assert payload["minimum_total_notional_usd"] > 8.42


@pytest.mark.asyncio
async def test_execute_signal_requests_bundle_recovery_after_partial_fill(monkeypatch):
    db = _FailureProjectionDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    plan = {"policy": "PARALLEL_MAKER", "plan_id": "plan-recovery", "legs": []}
    legs = [
        {
            "leg_id": "leg-1",
            "market_id": "market-one",
            "token_id": "token-1",
            "side": "buy",
            "requested_notional_usd": 5.0,
            "requested_shares": 10.0,
            "limit_price": 0.5,
        },
        {
            "leg_id": "leg-2",
            "market_id": "market-two",
            "token_id": "token-2",
            "side": "buy",
            "requested_notional_usd": 5.0,
            "requested_shares": 10.0,
            "limit_price": 0.5,
        },
    ]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    monkeypatch.setattr(engine, "_build_plan", lambda *args, **kwargs: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda _policy, _constraints: False)
    submit_mock = AsyncMock(
        side_effect=[
            [
                _leg_result(
                    leg_id="leg-1",
                    status="executed",
                    notional_usd=1.1,
                    shares=5.0,
                    provider_order_id="provider-filled",
                    provider_clob_order_id="clob-filled",
                    effective_price=0.22,
                    payload={
                        "provider": "test",
                        "filled_size": 5.0,
                        "average_fill_price": 0.22,
                        "filled_notional_usd": 1.1,
                    },
                ),
                _leg_result(
                    leg_id="leg-2",
                    status="open",
                    notional_usd=0.0,
                    shares=10.0,
                    provider_order_id="provider-open",
                    provider_clob_order_id="clob-open",
                    effective_price=0.5,
                    payload={
                        "provider": "test",
                        "filled_size": 0.0,
                        "average_fill_price": None,
                        "filled_notional_usd": 0.0,
                    },
                ),
            ],
            [
                _leg_result(
                    leg_id="leg-2",
                    status="failed",
                    notional_usd=0.0,
                    shares=0.0,
                    provider_order_id="provider-rescue",
                    provider_clob_order_id="clob-rescue",
                    effective_price=0.51,
                    payload={
                        "provider": "test",
                        "filled_size": 0.0,
                        "average_fill_price": None,
                        "filled_notional_usd": 0.0,
                    },
                    error_message="rescue_unfilled",
                )
            ],
        ]
    )
    monkeypatch.setattr(session_engine_module, "submit_execution_wave", submit_mock)
    cancel_provider_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(session_engine_module, "cancel_live_provider_order", cancel_provider_mock)
    flatten_mock = AsyncMock(
        return_value=SimpleNamespace(
            status="executed",
            order_id="flatten-order",
            effective_price=0.21,
            error_message=None,
            payload={
                "clob_order_id": "flatten-clob",
                "filled_size": 5.0,
                "average_fill_price": 0.21,
                "filled_notional_usd": 1.05,
            },
        )
    )
    monkeypatch.setattr(session_engine_module, "execute_live_order", flatten_mock)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    intent_runtime = SimpleNamespace(update_signal_status=AsyncMock())
    monkeypatch.setattr(intent_runtime_module, "get_intent_runtime", lambda: intent_runtime)

    signal = SimpleNamespace(
        id="signal-recovery",
        source="scanner",
        trace_id="trace-recovery",
        strategy_type="settlement_lag",
        strategy_context_json={},
        payload_json={
            "is_guaranteed": True,
            "market_roster": {
                "scope": "event",
                "market_count": 2,
                "markets": [
                    {"id": "market-one"},
                    {"id": "market-two"},
                ],
            },
        },
        market_id="market-one",
        market_question="Will team one win?",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=9.0,
        confidence=0.8,
    )

    result = await engine.execute_signal(
        trader_id="trader-recovery",
        signal=signal,
        decision_id="decision-recovery",
        strategy_key="settlement_lag",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=10.0,
        reason="partial-fill",
    )

    assert result.status == "failed"
    assert cancel_provider_mock.await_count == 1
    assert submit_mock.await_count == 2
    assert flatten_mock.await_count == 1
    assert flatten_mock.await_args.kwargs["token_id"] == "token-1"
    assert flatten_mock.await_args.kwargs["side"] == "SELL"
    session_rows = db.persisted_rows_by_type.get("ExecutionSession") or []
    assert len(session_rows) == 1
    bundle_recovery = session_rows[0].payload_json["bundle_recovery"]
    assert bundle_recovery["recovered_to_flat"] is True
    assert bundle_recovery["residual_exposure"] is False

