import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import signal_bus


class _DummySession:
    def __init__(self) -> None:
        self.commit = AsyncMock()


@pytest.mark.asyncio
async def test_emit_crypto_market_signals_emits_multistrategy_payload(monkeypatch):
    session = _DummySession()
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(signal_bus, "upsert_trade_signal", upsert_mock)
    monkeypatch.setattr(signal_bus, "refresh_trade_signal_snapshots", refresh_mock)

    now = datetime.now(timezone.utc)
    market = {
        "condition_id": "cond_1",
        "slug": "btc-15m-test",
        "question": "Will BTC be above X?",
        "asset": "BTC",
        "timeframe": "15min",
        "seconds_left": 120,
        "end_time": (now + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        "price_to_beat": 100.0,
        "oracle_price": 102.0,
        "up_price": 0.40,
        "down_price": 0.60,
        "spread": 0.01,
        "liquidity": 500000.0,
        "fees_enabled": False,
    }

    emitted = await signal_bus.emit_crypto_market_signals(session, [market])

    assert emitted == 1
    session.commit.assert_awaited_once()
    refresh_mock.assert_awaited_once()
    upsert_mock.assert_awaited_once()

    kwargs = upsert_mock.await_args.kwargs
    assert kwargs["source"] == "crypto"
    assert kwargs["signal_type"] == "crypto_worker_multistrat"
    assert kwargs["strategy_type"] == "crypto_15m"
    assert kwargs["direction"] in {"buy_yes", "buy_no"}
    assert kwargs["edge_percent"] > 1.0
    assert 0.0 <= kwargs["confidence"] <= 1.0

    payload = kwargs["payload_json"]
    assert payload["signal_version"] == "crypto_worker_v2"
    assert payload["strategy_origin"] == "crypto_worker"
    assert payload["regime"] in {"opening", "mid", "closing"}
    assert "component_edges" in payload
    assert "net_edges" in payload
    assert "execution_penalty_breakdown" in payload


@pytest.mark.asyncio
async def test_emit_crypto_market_signals_skips_low_quality_edges(monkeypatch):
    session = _DummySession()
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(signal_bus, "upsert_trade_signal", upsert_mock)
    monkeypatch.setattr(signal_bus, "refresh_trade_signal_snapshots", refresh_mock)

    market = {
        "condition_id": "cond_2",
        "slug": "btc-flat-test",
        "question": "Flat edge market",
        "asset": "BTC",
        "timeframe": "15min",
        "seconds_left": 450,
        "price_to_beat": 100.0,
        "oracle_price": 100.0,
        "up_price": 0.50,
        "down_price": 0.50,
        "spread": 0.03,
        "liquidity": 15000.0,
        "fees_enabled": True,
    }

    emitted = await signal_bus.emit_crypto_market_signals(session, [market])

    assert emitted == 0
    upsert_mock.assert_not_awaited()
    session.commit.assert_awaited_once()
    refresh_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_emit_crypto_market_signals_works_without_oracle_for_price_arb(monkeypatch):
    session = _DummySession()
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(signal_bus, "upsert_trade_signal", upsert_mock)
    monkeypatch.setattr(signal_bus, "refresh_trade_signal_snapshots", refresh_mock)

    market = {
        "condition_id": "cond_3",
        "slug": "btc-no-oracle-underround",
        "question": "No-oracle market",
        "asset": "BTC",
        "timeframe": "15min",
        "seconds_left": 240,
        "price_to_beat": None,
        "oracle_price": None,
        "up_price": 0.47,
        "down_price": 0.45,
        "spread": 0.005,
        "liquidity": 180000.0,
        "fees_enabled": False,
    }

    emitted = await signal_bus.emit_crypto_market_signals(session, [market])

    assert emitted == 1
    kwargs = upsert_mock.await_args.kwargs
    payload = kwargs["payload_json"]
    assert payload["oracle_available"] is False
    assert payload["strategy_origin"] == "crypto_worker"
    assert kwargs["edge_percent"] > 1.0
    assert kwargs["confidence"] <= 0.85
