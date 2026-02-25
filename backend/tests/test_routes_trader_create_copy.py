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
async def test_create_route_forwards_copy_from_trader_id(monkeypatch):
    session_obj = object()
    source_trader_id = "source-trader-1"
    created_trader = {
        "id": "new-trader-1",
        "name": "Copied Bot",
        "description": "copied",
        "mode": "paper",
        "source_configs": [],
        "risk_limits": {},
        "metadata": {},
        "is_enabled": True,
        "is_paused": False,
        "interval_seconds": 60,
    }

    create_trader_mock = AsyncMock(return_value=created_trader)
    create_config_revision_mock = AsyncMock(return_value=None)
    create_trader_event_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(routes_traders, "create_trader", create_trader_mock)
    monkeypatch.setattr(routes_traders, "create_config_revision", create_config_revision_mock)
    monkeypatch.setattr(routes_traders, "create_trader_event", create_trader_event_mock)

    request = routes_traders.TraderRequest(
        name="Copied Bot",
        copy_from_trader_id=source_trader_id,
        requested_by="tester",
    )
    result = await routes_traders.create_trader_route(request=request, session=session_obj)

    assert result["id"] == "new-trader-1"
    create_trader_mock.assert_awaited_once()
    payload = create_trader_mock.await_args.args[1]
    assert payload["name"] == "Copied Bot"
    assert payload["copy_from_trader_id"] == source_trader_id
    assert "source_configs" not in payload
    assert "interval_seconds" not in payload
    assert "risk_limits" not in payload
    assert "metadata" not in payload
    assert "is_enabled" not in payload
    assert "is_paused" not in payload
    assert "description" not in payload

    create_config_revision_mock.assert_awaited_once()
    create_trader_event_mock.assert_awaited_once()
    event_kwargs = create_trader_event_mock.await_args.kwargs
    assert event_kwargs["message"] == "Trader created from existing trader settings"
    assert event_kwargs["payload"]["copy_from_trader_id"] == source_trader_id


@pytest.mark.asyncio
async def test_create_route_returns_422_on_missing_copy_source(monkeypatch):
    monkeypatch.setattr(routes_traders, "create_trader", AsyncMock(side_effect=ValueError("Source trader not found")))

    request = routes_traders.TraderRequest(
        name="Copied Bot",
        copy_from_trader_id="missing-trader",
    )
    with pytest.raises(HTTPException) as excinfo:
        await routes_traders.create_trader_route(request=request, session=object())

    assert excinfo.value.status_code == 422
    assert excinfo.value.detail == "Source trader not found"


@pytest.mark.asyncio
async def test_create_route_forwards_mode(monkeypatch):
    session_obj = object()
    created_trader = {
        "id": "live-trader-1",
        "name": "Live Bot",
        "description": None,
        "mode": "live",
        "source_configs": [],
        "risk_limits": {},
        "metadata": {},
        "is_enabled": True,
        "is_paused": False,
        "interval_seconds": 60,
    }
    create_trader_mock = AsyncMock(return_value=created_trader)
    monkeypatch.setattr(routes_traders, "create_trader", create_trader_mock)
    monkeypatch.setattr(routes_traders, "create_config_revision", AsyncMock(return_value=None))
    monkeypatch.setattr(routes_traders, "create_trader_event", AsyncMock(return_value=None))

    request = routes_traders.TraderRequest(name="Live Bot", mode="live", requested_by="tester")
    result = await routes_traders.create_trader_route(request=request, session=session_obj)

    assert result["mode"] == "live"
    payload = create_trader_mock.await_args.args[1]
    assert payload["mode"] == "live"


@pytest.mark.asyncio
async def test_get_all_traders_forwards_mode_filter(monkeypatch):
    session_obj = object()
    list_traders_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(routes_traders, "list_traders", list_traders_mock)

    await routes_traders.get_all_traders(mode="live", session=session_obj)

    list_traders_mock.assert_awaited_once_with(session_obj, mode="live")


@pytest.mark.asyncio
async def test_get_all_traders_rejects_invalid_mode():
    with pytest.raises(HTTPException) as excinfo:
        await routes_traders.get_all_traders(mode="both", session=object())

    assert excinfo.value.status_code == 422
    assert excinfo.value.detail == "mode must be 'paper' or 'live'"
