from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.trader_orchestrator.live_market_context import (
    RuntimeTradeSignalView,
    build_live_signal_contexts,
)


@pytest.fixture(autouse=True)
def _patch_ws_price_sources(monkeypatch):
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.redis_price_cache.read_prices",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.get_feed_manager",
        lambda: SimpleNamespace(_started=False),
    )


@pytest.mark.asyncio
async def test_build_live_signal_contexts_uses_live_prices_and_history(monkeypatch):
    market_id = "0x" + ("1" * 64)
    yes_token = "123456789012345678"
    no_token = "987654321098765432"

    async def _fake_market_lookup(_market_id: str):
        return {
            "condition_id": market_id,
            "question": "Will it rain tomorrow?",
            "token_ids": [yes_token, no_token],
            "outcomes": ["Yes", "No"],
            "yes_price": 0.41,
            "no_price": 0.59,
        }

    async def _fake_prices_batch(token_ids: list[str]):
        assert set(token_ids) == {yes_token, no_token}
        return {
            yes_token: {"mid": 0.45},
            no_token: {"mid": 0.55},
        }

    async def _fake_history(
        token_id: str,
        interval=None,
        fidelity=None,
        start_ts=None,
        end_ts=None,
        use_trading_proxy: bool = False,
    ):
        del interval, fidelity, use_trading_proxy
        base = 0.40 if token_id == yes_token else 0.60
        return [
            {"t": (int(start_ts) + 5) * 1000, "p": base},
            {"t": (int(end_ts) - 5) * 1000, "p": base + 0.02},
        ]

    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_market_by_condition_id",
        _fake_market_lookup,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_batch",
        _fake_prices_batch,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_history",
        _fake_history,
    )

    signals = [
        SimpleNamespace(
            id="sig_yes",
            market_id=market_id,
            market_question="Will it rain tomorrow?",
            source="weather",
            direction="buy_yes",
            entry_price=0.40,
            edge_percent=10.0,
            payload_json={},
        ),
        SimpleNamespace(
            id="sig_no",
            market_id=market_id,
            market_question="Will it rain tomorrow?",
            source="weather",
            direction="buy_no",
            entry_price=0.60,
            edge_percent=10.0,
            payload_json={},
        ),
    ]

    contexts = await build_live_signal_contexts(
        signals,
        history_window_seconds=1800,
        history_fidelity_seconds=300,
        max_history_points=20,
        history_tail_points=3,
    )

    yes_ctx = contexts["sig_yes"]
    assert yes_ctx["available"] is True
    assert yes_ctx["selected_outcome"] == "yes"
    assert yes_ctx["live_selected_price"] == pytest.approx(0.45)
    assert yes_ctx["live_edge_percent"] == pytest.approx(5.0)
    assert yes_ctx["market_data_source"] == "http_batch"
    assert yes_ctx["market_data_age_ms"] is not None
    assert yes_ctx["history_summary"]["points"] == 2
    assert len(yes_ctx["history_tail"]) <= 3

    no_ctx = contexts["sig_no"]
    assert no_ctx["available"] is True
    assert no_ctx["selected_outcome"] == "no"
    assert no_ctx["live_selected_price"] == pytest.approx(0.55)
    assert no_ctx["live_edge_percent"] == pytest.approx(15.0)
    assert no_ctx["entry_price_delta"] == pytest.approx(-0.05)
    assert no_ctx["adverse_price_move"] is False


@pytest.mark.asyncio
async def test_build_live_signal_contexts_derives_model_probability_from_weather_payload(
    monkeypatch,
):
    market_id = "0x" + ("2" * 64)
    yes_token = "111111111111111111"
    no_token = "222222222222222222"

    async def _fake_market_lookup(_market_id: str):
        return {
            "condition_id": market_id,
            "question": "Highest temp?",
            "token_ids": [yes_token, no_token],
            "outcomes": ["Yes", "No"],
        }

    async def _fake_prices_batch(token_ids: list[str]):
        del token_ids
        return {no_token: {"mid": 0.55}}

    async def _fake_history(*args, **kwargs):
        del args, kwargs
        return []

    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_market_by_condition_id",
        _fake_market_lookup,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_batch",
        _fake_prices_batch,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_history",
        _fake_history,
    )

    signal = SimpleNamespace(
        id="sig_payload",
        market_id=market_id,
        market_question="Highest temp?",
        source="weather",
        direction="buy_no",
        entry_price=None,
        edge_percent=None,
        payload_json={"metadata": {"weather": {"consensus_probability": 0.3}}},
    )

    contexts = await build_live_signal_contexts([signal])
    ctx = contexts["sig_payload"]
    assert ctx["model_probability"] == pytest.approx(0.7)
    assert ctx["live_edge_percent"] == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_build_live_signal_contexts_uses_payload_market_hints_when_signal_market_id_is_non_lookup(
    monkeypatch,
):
    signal_market_id = "Bitcoin Up or Down - February 23, 9AM ET"
    condition_id = "0x" + ("3" * 64)
    yes_token = "33333333333333333333333333333333333333333333333333333333333333333"
    no_token = "44444444444444444444444444444444444444444444444444444444444444444"

    async def _fake_market_lookup(market_id: str):
        assert market_id == condition_id
        return {
            "condition_id": condition_id,
            "question": signal_market_id,
            "token_ids": [yes_token, no_token],
            "outcomes": ["Yes", "No"],
        }

    async def _fake_prices_batch(token_ids: list[str]):
        assert set(token_ids) == {yes_token, no_token}
        return {
            yes_token: {"mid": 0.08},
            no_token: {"mid": 0.92},
        }

    async def _fake_history(*args, **kwargs):
        del args, kwargs
        return []

    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_market_by_condition_id",
        _fake_market_lookup,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_batch",
        _fake_prices_batch,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_history",
        _fake_history,
    )

    signal = SimpleNamespace(
        id="sig_payload_hint",
        market_id=signal_market_id,
        market_question=signal_market_id,
        source="scanner",
        direction="buy_no",
        entry_price=0.95,
        edge_percent=5.0,
        payload_json={
            "markets": [
                {
                    "id": "1407632",
                    "condition_id": condition_id,
                    "clob_token_ids": [yes_token, no_token],
                    "outcome_labels": ["Yes", "No"],
                }
            ]
        },
    )

    contexts = await build_live_signal_contexts([signal])
    ctx = contexts["sig_payload_hint"]
    assert ctx["market_id"] == signal_market_id.lower()
    assert ctx["condition_id"] == condition_id
    assert ctx["available"] is True
    assert ctx["selected_outcome"] == "no"
    assert ctx["selected_token_id"] == no_token
    assert ctx["live_selected_price"] == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_build_live_signal_contexts_falls_back_to_payload_tokens_when_market_lookup_fails(
    monkeypatch,
):
    signal_market_id = "unresolvable-market-name"
    yes_token = "55555555555555555555555555555555555555555555555555555555555555555"
    no_token = "66666666666666666666666666666666666666666666666666666666666666666"

    async def _fake_market_lookup(_market_id: str):
        return None

    async def _fake_market_trades(*args, **kwargs):
        del args, kwargs
        return []

    async def _fake_prices_batch(token_ids: list[str]):
        assert set(token_ids) == {yes_token, no_token}
        return {
            no_token: {"mid": 0.88},
        }

    async def _fake_history(*args, **kwargs):
        del args, kwargs
        return []

    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_market_by_condition_id",
        _fake_market_lookup,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_market_trades",
        _fake_market_trades,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_batch",
        _fake_prices_batch,
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.live_market_context.polymarket_client.get_prices_history",
        _fake_history,
    )

    signal = SimpleNamespace(
        id="sig_market_lookup_fails",
        market_id=signal_market_id,
        market_question="Fallback token hint path",
        source="scanner",
        direction="buy_no",
        entry_price=0.9,
        edge_percent=3.0,
        payload_json={
            "markets": [
                {
                    "id": "1407632",
                    "clob_token_ids": [yes_token, no_token],
                    "outcome_labels": ["Yes", "No"],
                }
            ]
        },
    )

    contexts = await build_live_signal_contexts([signal])
    ctx = contexts["sig_market_lookup_fails"]
    assert ctx["available"] is True
    assert ctx["selected_outcome"] == "no"
    assert ctx["selected_token_id"] == no_token
    assert ctx["live_selected_price"] == pytest.approx(0.88)


def test_runtime_trade_signal_view_overrides_runtime_fields():
    base = SimpleNamespace(
        id="sig_runtime",
        source="weather",
        market_id="0xabc",
        entry_price=0.22,
        edge_percent=18.0,
        payload_json={"foo": "bar"},
    )
    runtime = RuntimeTradeSignalView(
        base,
        live_context={
            "live_selected_price": 0.31,
            "live_edge_percent": 7.2,
            "selected_outcome": "yes",
        },
    )

    assert runtime.id == "sig_runtime"
    assert runtime.entry_price == pytest.approx(0.31)
    assert runtime.edge_percent == pytest.approx(7.2)
    assert runtime.payload_json["foo"] == "bar"
    assert runtime.payload_json["live_market"]["selected_outcome"] == "yes"
