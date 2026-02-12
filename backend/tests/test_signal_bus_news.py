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
async def test_emit_news_intent_signals_includes_reasoning_and_evidence(monkeypatch):
    session = _DummySession()
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(signal_bus, "upsert_trade_signal", upsert_mock)
    monkeypatch.setattr(signal_bus, "refresh_trade_signal_snapshots", refresh_mock)

    intent = SimpleNamespace(
        id="news_intent_1",
        signal_key="sig_abc",
        finding_id="finding_1",
        market_id="mkt_123",
        market_question="Will X happen?",
        direction="buy_yes",
        entry_price=0.42,
        edge_percent=12.5,
        confidence=0.71,
        metadata_json={
            "market": {"liquidity": 12345.0},
            "finding": {
                "reasoning": "LLM rationale",
                "evidence": {"llm": {"model": "gpt"}},
            },
        },
        created_at=None,
    )

    emitted = await signal_bus.emit_news_intent_signals(session, [intent], max_age_minutes=60)

    assert emitted == 1
    session.commit.assert_awaited_once()
    refresh_mock.assert_awaited_once()
    upsert_mock.assert_awaited_once()

    kwargs = upsert_mock.await_args.kwargs
    assert kwargs["source"] == "news"
    assert kwargs["liquidity"] == 12345.0
    assert kwargs["payload_json"]["reasoning"] == "LLM rationale"
    assert kwargs["payload_json"]["evidence"] == {"llm": {"model": "gpt"}}
