import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_traders  # noqa: E402


@pytest.mark.asyncio
async def test_delete_route_writes_deleted_event_without_trader_fk(monkeypatch):
    trader_id = "trader-route-delete-safe"
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value={"id": trader_id, "name": "Route Trader"}))
    monkeypatch.setattr(routes_traders, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(
        routes_traders,
        "get_open_position_summary_for_trader",
        AsyncMock(return_value={"live": 0, "paper": 2, "other": 0, "total": 2}),
    )
    monkeypatch.setattr(
        routes_traders,
        "get_open_order_summary_for_trader",
        AsyncMock(return_value={"live": 0, "paper": 2, "other": 0, "total": 2}),
    )
    monkeypatch.setattr(routes_traders, "delete_trader", AsyncMock(return_value=True))
    create_event = AsyncMock(return_value=None)
    monkeypatch.setattr(routes_traders, "create_trader_event", create_event)

    response = await routes_traders.delete_trader_route(
        trader_id=trader_id,
        action=routes_traders.TraderDeleteAction.block,
        session=object(),
    )

    assert response["status"] == "deleted"
    assert response["trader_id"] == trader_id
    kwargs = create_event.await_args.kwargs
    assert kwargs["trader_id"] is None
    assert kwargs["payload"]["deleted_trader_id"] == trader_id


@pytest.mark.asyncio
async def test_delete_route_blocks_live_exposure_without_force(monkeypatch):
    trader_id = "trader-route-live-block"
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value={"id": trader_id, "name": "Live Trader"}))
    monkeypatch.setattr(routes_traders, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(
        routes_traders,
        "get_open_position_summary_for_trader",
        AsyncMock(return_value={"live": 1, "paper": 0, "other": 0, "total": 1}),
    )
    monkeypatch.setattr(
        routes_traders,
        "get_open_order_summary_for_trader",
        AsyncMock(return_value={"live": 0, "paper": 0, "other": 0, "total": 0}),
    )
    delete_trader_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(routes_traders, "delete_trader", delete_trader_mock)
    monkeypatch.setattr(routes_traders, "create_trader_event", AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as excinfo:
        await routes_traders.delete_trader_route(
            trader_id=trader_id,
            action=routes_traders.TraderDeleteAction.block,
            session=object(),
        )

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["code"] == "open_live_exposure"
    delete_trader_mock.assert_not_awaited()
