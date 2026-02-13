import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from services import trading
from services.trading import Order, OrderSide, OrderStatus, TradingService


class _SequencedClient:
    def __init__(self, outcomes: list[bool]):
        self._outcomes = list(outcomes)
        self._counter = 0

    def create_order(self, order_args):
        return {"order_args": order_args}

    def post_order(self, signed_order, order_type):
        self._counter += 1
        success = self._outcomes.pop(0) if self._outcomes else True
        if success:
            return {"success": True, "orderID": f"oid-{self._counter}"}
        return {"success": False, "errorMsg": "simulated failure"}


def _install_fake_clob_modules(monkeypatch) -> None:
    py_clob_client = types.ModuleType("py_clob_client")
    clob_types = types.ModuleType("py_clob_client.clob_types")
    order_builder = types.ModuleType("py_clob_client.order_builder")
    constants = types.ModuleType("py_clob_client.order_builder.constants")

    class OrderArgs:
        def __init__(self, price, size, side, token_id):
            self.price = price
            self.size = size
            self.side = side
            self.token_id = token_id

    clob_types.OrderArgs = OrderArgs
    constants.BUY = "BUY"
    constants.SELL = "SELL"

    monkeypatch.setitem(sys.modules, "py_clob_client", py_clob_client)
    monkeypatch.setitem(sys.modules, "py_clob_client.clob_types", clob_types)
    monkeypatch.setitem(sys.modules, "py_clob_client.order_builder", order_builder)
    monkeypatch.setitem(
        sys.modules,
        "py_clob_client.order_builder.constants",
        constants,
    )


def _configure_limits(monkeypatch) -> None:
    monkeypatch.setattr(settings, "TRADING_ENABLED", True)
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    monkeypatch.setattr(settings, "MAX_TRADE_SIZE_USD", 10_000.0)
    monkeypatch.setattr(settings, "MAX_DAILY_TRADE_VOLUME", 10_000.0)
    monkeypatch.setattr(settings, "MAX_OPEN_POSITIONS", 1000)
    monkeypatch.setattr(settings, "MAX_PER_MARKET_USD", 10_000.0)


@pytest.mark.asyncio
async def test_failed_order_rolls_back_reserved_volume(monkeypatch):
    _configure_limits(monkeypatch)
    _install_fake_clob_modules(monkeypatch)

    refresh_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(trading.global_pause_state, "refresh_from_db", refresh_mock)
    monkeypatch.setattr(trading, "pre_trade_vpn_check", AsyncMock(return_value=(True, "")))

    service = TradingService()
    service._initialized = True
    service._client = _SequencedClient([False, True])

    failed = await service.place_order(
        token_id="token-a",
        side=OrderSide.BUY,
        price=0.5,
        size=8.0,
    )
    assert failed.status == OrderStatus.FAILED
    assert service.get_stats().daily_volume == 0.0

    succeeded = await service.place_order(
        token_id="token-a",
        side=OrderSide.BUY,
        price=0.5,
        size=8.0,
    )
    assert succeeded.status == OrderStatus.OPEN
    assert service.get_stats().daily_volume == pytest.approx(4.0)
    refresh_mock.assert_any_call(force=True)


@pytest.mark.asyncio
async def test_sell_order_reduces_market_exposure(monkeypatch):
    _configure_limits(monkeypatch)
    _install_fake_clob_modules(monkeypatch)

    monkeypatch.setattr(
        trading.global_pause_state,
        "refresh_from_db",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(trading, "pre_trade_vpn_check", AsyncMock(return_value=(True, "")))

    service = TradingService()
    service._initialized = True
    service._client = _SequencedClient([True, True])

    buy = await service.place_order(
        token_id="token-b",
        side=OrderSide.BUY,
        price=0.5,
        size=10.0,
    )
    assert buy.status == OrderStatus.OPEN
    assert float(service._market_positions["token-b"]) == pytest.approx(5.0)

    sell = await service.place_order(
        token_id="token-b",
        side=OrderSide.SELL,
        price=0.5,
        size=10.0,
    )
    assert sell.status == OrderStatus.OPEN
    assert "token-b" not in service._market_positions


def test_order_cache_is_bounded():
    service = TradingService()
    service._max_order_history = 3

    open_order = Order(
        id="open-1",
        token_id="tok",
        side=OrderSide.BUY,
        price=0.5,
        size=1.0,
        status=OrderStatus.OPEN,
        created_at=datetime(2026, 1, 1, 0, 0, 0),
        updated_at=datetime(2026, 1, 1, 0, 0, 0),
    )
    service._remember_order(open_order)

    for i in range(4):
        failed = Order(
            id=f"failed-{i}",
            token_id="tok",
            side=OrderSide.BUY,
            price=0.5,
            size=1.0,
            status=OrderStatus.FAILED,
            created_at=datetime(2026, 1, 1, 0, 0, i + 1),
            updated_at=datetime(2026, 1, 1, 0, 0, i + 1),
        )
        service._remember_order(failed)

    assert len(service._orders) == 3
    assert "open-1" in service._orders
