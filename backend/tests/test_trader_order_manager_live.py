import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.live_execution_adapter import LiveOrderExecution  # noqa: E402
from services.trader_orchestrator import order_manager  # noqa: E402


@pytest.mark.asyncio
async def test_submit_order_live_uses_execution_adapter(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.41,
            error_message=None,
            payload={"order_id": "ord-123"},
            order_id="ord-123",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    signal = SimpleNamespace(
        id="sig-1",
        market_id="123456789012345678",
        direction="buy_yes",
        entry_price=0.40,
        market_question="Will X happen?",
        payload_json={},
    )

    status, effective_price, error_message, payload = await order_manager.submit_order(
        mode="live",
        signal=signal,
        size_usd=41.0,
    )

    assert status == "open"
    assert effective_price == 0.41
    assert error_message is None
    assert payload["mode"] == "live"
    assert payload["market_id"] == "123456789012345678"
    assert payload["shares"] == pytest.approx(102.5, rel=1e-6)
    execution_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_order_live_fails_without_token_id():
    signal = SimpleNamespace(
        id="sig-2",
        market_id="0x" + ("a" * 64),  # condition id, not directly executable token id
        direction="buy_yes",
        entry_price=0.40,
        market_question="Will Y happen?",
        payload_json={},
    )

    status, _price, error_message, payload = await order_manager.submit_order(
        mode="live",
        signal=signal,
        size_usd=40.0,
    )

    assert status == "failed"
    assert "token_id" in str(error_message or "")
    assert payload["reason"] == "missing_token_id"


@pytest.mark.asyncio
async def test_submit_order_paper_still_simulates_execution():
    signal = SimpleNamespace(
        id="sig-3",
        market_id="m3",
        direction="buy_no",
        entry_price=0.55,
        market_question="Will Z happen?",
        payload_json={},
    )

    status, effective_price, error_message, payload = await order_manager.submit_order(
        mode="paper",
        signal=signal,
        size_usd=25.0,
    )

    assert status == "executed"
    assert effective_price == 0.55
    assert error_message is None
    assert payload["submission"] == "simulated"
