import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import live_execution_adapter
from services.live_execution_service import OrderStatus


@pytest.mark.asyncio
async def test_execute_live_order_blocks_when_trading_init_fails(monkeypatch):
    ensure_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)

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
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "place_order", place_mock)
    get_price_mock = AsyncMock(side_effect=[0.40, 0.42])
    monkeypatch.setattr(
        live_execution_adapter.polymarket_client,
        "get_price",
        get_price_mock,
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
    assert get_price_mock.await_count == 2
    assert get_price_mock.await_args_list[0].kwargs["side"] == "BUY"
    assert get_price_mock.await_args_list[1].kwargs["side"] == "SELL"


@pytest.mark.asyncio
async def test_execute_live_order_uses_fallback_when_live_quote_notional_below_strategy_floor(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    place_mock = AsyncMock(
        return_value=SimpleNamespace(
            status=OrderStatus.OPEN,
            error_message=None,
            average_fill_price=0.0,
            id="order-456",
            clob_order_id="clob-456",
            filled_size=0.0,
        )
    )
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "place_order", place_mock)
    get_price_mock = AsyncMock(side_effect=[0.70, 0.71])
    monkeypatch.setattr(
        live_execution_adapter.polymarket_client,
        "get_price",
        get_price_mock,
    )

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=5.0,
        fallback_price=0.80,
        min_order_size_usd=4.0,
    )

    assert result.status == "open"
    assert result.payload["price_resolution"] == "fallback_min_notional_guard"
    assert result.payload["resolved_price"] == pytest.approx(0.80)
    _, kwargs = place_mock.await_args
    assert kwargs["price"] == pytest.approx(0.80)
    assert get_price_mock.await_count == 2
