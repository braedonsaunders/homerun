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
import services.live_execution_service as live_execution_module
from services.live_execution_service import Order, OrderSide, OrderStatus, LiveExecutionService


class _SequencedClient:
    def __init__(self, outcomes: list[bool]):
        self._outcomes = list(outcomes)
        self._counter = 0

    def create_order(self, order_args):
        return {"order_args": order_args}

    def post_order(self, signed_order, order_type, post_only=False):
        self._counter += 1
        success = self._outcomes.pop(0) if self._outcomes else True
        if success:
            return {"success": True, "orderID": f"oid-{self._counter}"}
        return {"success": False, "errorMsg": "simulated failure"}


class _ProviderSnapshotClient:
    def __init__(self):
        self.cancel_calls: list[str] = []

    def get_orders(self):
        return [
            {
                "id": "clob-1",
                "status": "partially_filled",
                "size": "100",
                "size_matched": "25",
                "avg_price": "0.44",
                "price": "0.45",
            }
        ]

    def get_order(self, clob_order_id: str):
        return {
            "id": clob_order_id,
            "status": "filled",
            "size": "30",
            "size_matched": "30",
            "avg_price": "0.52",
        }

    def cancel(self, clob_order_id: str):
        self.cancel_calls.append(clob_order_id)
        return {"canceled": [clob_order_id]}


class _ReinitSnapshotClient:
    def __init__(self):
        self.get_orders_calls = 0

    def get_orders(self):
        self.get_orders_calls += 1
        if self.get_orders_calls == 1:
            raise RuntimeError("session expired")
        return [
            {
                "id": "clob-reinit",
                "status": "filled",
                "size": "12",
                "size_matched": "12",
                "avg_price": "0.33",
            }
        ]

    def get_order(self, clob_order_id: str):
        return {"id": clob_order_id, "status": "filled", "size": "12", "size_matched": "12", "avg_price": "0.33"}


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
    monkeypatch.setattr(live_execution_module.global_pause_state, "refresh_from_db", refresh_mock)
    monkeypatch.setattr(trading, "pre_trade_vpn_check", AsyncMock(return_value=(True, "")))

    service = LiveExecutionService()
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
        live_execution_module.global_pause_state,
        "refresh_from_db",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(trading, "pre_trade_vpn_check", AsyncMock(return_value=(True, "")))

    service = LiveExecutionService()
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
    service = LiveExecutionService()
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


@pytest.mark.asyncio
async def test_cancel_order_accepts_provider_clob_id(monkeypatch):
    _configure_limits(monkeypatch)
    service = LiveExecutionService()
    service._initialized = True
    client = _ProviderSnapshotClient()
    service._client = client

    cancelled = await service.cancel_order("clob-provider-1")

    assert cancelled is True
    assert client.cancel_calls == ["clob-provider-1"]


@pytest.mark.asyncio
async def test_get_order_snapshots_parses_provider_fill_values(monkeypatch):
    _configure_limits(monkeypatch)
    service = LiveExecutionService()
    service._initialized = True
    service._client = _ProviderSnapshotClient()

    snapshots = await service.get_order_snapshots_by_clob_ids(["clob-1", "clob-2"])

    assert set(snapshots.keys()) == {"clob-1", "clob-2"}
    assert snapshots["clob-1"]["normalized_status"] == "partially_filled"
    assert snapshots["clob-1"]["filled_size"] == pytest.approx(25.0)
    assert snapshots["clob-1"]["average_fill_price"] == pytest.approx(0.44)
    assert snapshots["clob-1"]["filled_notional_usd"] == pytest.approx(11.0)
    assert snapshots["clob-2"]["normalized_status"] == "filled"
    assert snapshots["clob-2"]["filled_notional_usd"] == pytest.approx(15.6)


@pytest.mark.asyncio
async def test_get_order_snapshots_maps_invalid_status_to_failed(monkeypatch):
    _configure_limits(monkeypatch)

    class _InvalidStatusClient:
        def get_orders(self):
            return []

        def get_order(self, clob_order_id: str):
            return {
                "id": clob_order_id,
                "status": "INVALID",
                "size": "17.77",
                "size_matched": "0",
                "price": "0.24",
            }

    service = LiveExecutionService()
    service._initialized = True
    service._client = _InvalidStatusClient()
    service._remember_order(
        Order(
            id="order-invalid",
            token_id="tok-invalid",
            side=OrderSide.BUY,
            price=0.24,
            size=17.77,
            status=OrderStatus.OPEN,
            clob_order_id="clob-invalid",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )
    )

    snapshots = await service.get_order_snapshots_by_clob_ids(["clob-invalid"])

    assert snapshots["clob-invalid"]["normalized_status"] == "failed"
    assert service.get_order("order-invalid") is not None
    assert service.get_order("order-invalid").status == OrderStatus.FAILED


@pytest.mark.asyncio
async def test_get_order_snapshots_retries_after_reinitializing_client(monkeypatch):
    _configure_limits(monkeypatch)
    service = LiveExecutionService()
    service._initialized = True
    service._client = _ReinitSnapshotClient()
    ensure_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(service, "ensure_initialized", ensure_mock)

    snapshots = await service.get_order_snapshots_by_clob_ids(["clob-reinit"])

    assert snapshots["clob-reinit"]["normalized_status"] == "filled"
    assert snapshots["clob-reinit"]["filled_size"] == pytest.approx(12.0)
    assert ensure_mock.await_count == 1
