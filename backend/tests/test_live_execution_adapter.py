import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import live_execution_adapter
from services.live_execution_service import OrderStatus, live_execution_service


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
async def test_execute_live_order_blocks_auth_circuit_without_retrying_init(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "clob_auth_circuit_open", lambda: True)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "clob_auth_circuit_reason", lambda: "invalid username/password")
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=5.0,
        fallback_price=0.45,
    )

    assert result.status == "failed"
    assert result.error_message == "invalid username/password"
    assert result.payload.get("submission") == "auth_unavailable"
    ensure_mock.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_execute_live_order_clamps_explicit_limit_before_submission(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    place_mock = AsyncMock(
        return_value=SimpleNamespace(
            status=OrderStatus.OPEN,
            error_message=None,
            average_fill_price=0.0,
            id="order-clamped",
            clob_order_id="clob-clamped",
            filled_size=0.0,
        )
    )
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "place_order", place_mock)

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="SELL",
        size=5.0,
        fallback_price=0.9995,
        resolve_live_price=False,
    )

    assert result.status == "open"
    assert "binary_price_clamped" in str(result.payload.get("price_resolution") or "")
    assert result.payload["resolved_price"] == pytest.approx(0.99)
    _, kwargs = place_mock.await_args
    assert kwargs["price"] == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_execute_live_order_aggressive_quote_respects_max_execution_price(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    place_mock = AsyncMock(
        return_value=SimpleNamespace(
            status=OrderStatus.OPEN,
            error_message=None,
            average_fill_price=0.0,
            id="order-789",
            clob_order_id="clob-789",
            filled_size=0.0,
        )
    )
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "place_order", place_mock)
    get_price_mock = AsyncMock(side_effect=[0.94, 0.96])
    monkeypatch.setattr(
        live_execution_adapter.polymarket_client,
        "get_price",
        get_price_mock,
    )

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=10.0,
        fallback_price=0.95,
        quote_aggressively=True,
        max_execution_price=0.92,
    )

    assert result.status == "open"
    assert "bounded_by_max_execution" in str(result.payload.get("price_resolution") or "")
    assert result.payload["resolved_price"] == pytest.approx(0.92)
    _, kwargs = place_mock.await_args
    assert kwargs["price"] == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_execute_live_order_aggressive_buy_can_submit_as_gtc(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    place_mock = AsyncMock(
        return_value=SimpleNamespace(
            status=OrderStatus.OPEN,
            error_message=None,
            average_fill_price=0.0,
            id="order-gtc",
            clob_order_id="clob-gtc",
            filled_size=0.0,
        )
    )
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "place_order", place_mock)
    get_price_mock = AsyncMock(side_effect=[0.91, 0.93])
    monkeypatch.setattr(
        live_execution_adapter.polymarket_client,
        "get_price",
        get_price_mock,
    )

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=10.0,
        fallback_price=0.90,
        time_in_force="IOC",
        quote_aggressively=True,
        aggressive_limit_buy_submit_as_gtc=True,
    )

    assert result.status == "open"
    assert "aggressive_buy_submit_as_gtc" in str(result.payload.get("price_resolution") or "")
    _, kwargs = place_mock.await_args
    assert kwargs["order_type"].value == "GTC"


@pytest.mark.asyncio
async def test_execute_live_order_aggressive_buy_keeps_ioc_without_gtc_override(monkeypatch):
    ensure_mock = AsyncMock(return_value=True)
    place_mock = AsyncMock(
        return_value=SimpleNamespace(
            status=OrderStatus.OPEN,
            error_message=None,
            average_fill_price=0.0,
            id="order-ioc",
            clob_order_id="clob-ioc",
            filled_size=0.0,
        )
    )
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "ensure_initialized", ensure_mock)
    monkeypatch.setattr(live_execution_adapter.live_execution_service, "place_order", place_mock)
    get_price_mock = AsyncMock(side_effect=[0.91, 0.93])
    monkeypatch.setattr(
        live_execution_adapter.polymarket_client,
        "get_price",
        get_price_mock,
    )

    result = await live_execution_adapter.execute_live_order(
        token_id="123456789012345678901",
        side="BUY",
        size=10.0,
        fallback_price=0.90,
        time_in_force="IOC",
        quote_aggressively=True,
    )

    assert result.status == "open"
    assert "aggressive_buy_submit_as_gtc" not in str(result.payload.get("price_resolution") or "")
    _, kwargs = place_mock.await_args
    assert kwargs["order_type"].value == "IOC"


@pytest.mark.asyncio
async def test_live_execution_service_client_io_times_out():
    with pytest.raises(asyncio.TimeoutError):
        await live_execution_service._run_client_io(lambda: time.sleep(0.5), timeout=0.01)
