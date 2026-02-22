import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import live_execution_adapter
from services.trading import OrderStatus


@pytest.mark.asyncio
async def test_execute_live_order_blocks_when_trading_init_fails(monkeypatch):
    ensure_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(live_execution_adapter.trading_service, "ensure_initialized", ensure_mock)

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=5.0,
        fallback_price=0.45,
    )

    assert result.status == "failed"
    assert result.error_message == "Trading service is not initialized."
    assert result.payload.get("submission") == "not_ready"
    ensure_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_live_order_initializes_and_places_order(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    place_mock = AsyncMock(
        return_value=SimpleNamespace(
            status=OrderStatus.OPEN,
            error_message=None,
            average_fill_price=0.0,
            id="order-123",
            clob_order_id="clob-123",
            filled_size=0.0,
        )
    )
    monkeypatch.setattr(live_execution_adapter.trading_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.trading_service, "place_order", place_mock)
    monkeypatch.setattr(
        live_execution_adapter.polymarket_client,
        "get_price",
        AsyncMock(return_value=0.42),
    )

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=10.0,
        fallback_price=0.40,
        market_question="Will BTC close above 110k?",
        opportunity_id="opp-1",
    )

    assert result.status == "open"
    assert result.error_message is None
    assert result.order_id == "order-123"
    assert result.effective_price == pytest.approx(0.42)
    ensure_mock.assert_awaited_once()
    place_mock.assert_awaited_once()
