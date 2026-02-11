import sys
from pathlib import Path
from types import SimpleNamespace
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
async def test_emit_insider_intent_signals_uses_insider_source(monkeypatch):
    session = _DummySession()
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(signal_bus, "upsert_trade_signal", upsert_mock)
    monkeypatch.setattr(signal_bus, "refresh_trade_signal_snapshots", refresh_mock)

    intent = SimpleNamespace(
        id="intent_1",
        signal_key="key_1",
        market_id="mkt_123",
        market_question="Will event X happen?",
        direction="buy_yes",
        entry_price=0.41,
        edge_percent=7.5,
        confidence=0.72,
        insider_score=0.81,
        wallet_addresses_json=["0x1", "0x2"],
        metadata_json={"market_liquidity": 1000.0},
        created_at=None,
    )

    emitted = await signal_bus.emit_insider_intent_signals(session, [intent], max_age_minutes=90)
    assert emitted == 1
    session.commit.assert_awaited()
    refresh_mock.assert_awaited_once()
    upsert_mock.assert_awaited_once()

    kwargs = upsert_mock.await_args.kwargs
    assert kwargs["source"] == "insider"
    assert kwargs["signal_type"] == "insider_intent"
    assert kwargs["strategy_type"] == "insider_edge"
    assert kwargs["market_id"] == "mkt_123"
    assert kwargs["direction"] == "buy_yes"
    assert kwargs["payload_json"]["wallet_addresses"] == ["0x1", "0x2"]
