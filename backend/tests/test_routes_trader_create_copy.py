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
async def test_get_all_traders_forwards_shadow_mode_filter(monkeypatch):
    session_obj = object()
    list_traders_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(routes_traders, "list_traders", list_traders_mock)

    await routes_traders.get_all_traders(mode="shadow", session=session_obj)

    list_traders_mock.assert_awaited_once_with(session_obj, mode="shadow")


@pytest.mark.asyncio
async def test_get_all_traders_rejects_invalid_mode():
    with pytest.raises(HTTPException) as excinfo:
        await routes_traders.get_all_traders(mode="both", session=object())

    assert excinfo.value.status_code == 422
    assert excinfo.value.detail == "mode must be 'paper', 'shadow', or 'live'"


@pytest.mark.asyncio
async def test_start_trader_enables_and_unpauses(monkeypatch):
    session_obj = object()
    trader_id = "trader-1"
    existing = {
        "id": trader_id,
        "name": "Any Trader",
        "is_enabled": False,
        "is_paused": True,
        "metadata": {"resume_policy": "resume_full"},
    }
    updated = {
        "id": trader_id,
        "name": "Any Trader",
        "is_enabled": True,
        "is_paused": False,
        "mode": "live",
    }
    update_trader_mock = AsyncMock(return_value=updated)
    read_control_mock = AsyncMock(return_value={"mode": "live"})
    sync_inventory_mock = AsyncMock(return_value=None)
    open_summary_mock = AsyncMock(return_value={"live": 0, "paper": 0})
    create_event_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(routes_traders, "_assert_not_globally_paused", lambda: None)
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value=existing))
    monkeypatch.setattr(routes_traders, "update_trader", update_trader_mock)
    monkeypatch.setattr(routes_traders, "read_orchestrator_control", read_control_mock)
    monkeypatch.setattr(routes_traders, "sync_trader_position_inventory", sync_inventory_mock)
    monkeypatch.setattr(routes_traders, "get_open_position_summary_for_trader", open_summary_mock)
    monkeypatch.setattr(routes_traders, "create_trader_event", create_event_mock)

    result = await routes_traders.start_trader(trader_id=trader_id, session=session_obj)

    assert result == updated
    update_trader_mock.assert_awaited_once()
    update_payload = update_trader_mock.await_args.args[2]
    assert update_payload["is_enabled"] is True
    assert update_payload["is_paused"] is False
    assert update_payload["metadata"]["loss_streak_reset_reason"] == "operator_start"
    assert isinstance(update_payload["metadata"]["loss_streak_reset_at"], str)
    assert update_payload["metadata"]["loss_streak_reset_at"]
    create_event_mock.assert_awaited_once()
    event_kwargs = create_event_mock.await_args.kwargs
    assert event_kwargs["event_type"] == "trader_started"
    assert event_kwargs["message"] == "Trader resumed"
    assert event_kwargs["payload"]["loss_streak_reset_at"] == update_payload["metadata"]["loss_streak_reset_at"]


@pytest.mark.asyncio
async def test_start_trader_returns_404_when_missing(monkeypatch):
    monkeypatch.setattr(routes_traders, "_assert_not_globally_paused", lambda: None)
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value=None))
    monkeypatch.setattr(routes_traders, "update_trader", AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as excinfo:
        await routes_traders.start_trader(trader_id="missing", session=object())

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Trader not found"


@pytest.mark.asyncio
async def test_update_trader_route_resume_resets_loss_streak(monkeypatch):
    session_obj = object()
    trader_id = "trader-1"
    before = {
        "id": trader_id,
        "name": "Any Trader",
        "is_enabled": False,
        "is_paused": False,
        "metadata": {"resume_policy": "resume_full"},
    }
    after = {
        "id": trader_id,
        "name": "Any Trader",
        "is_enabled": True,
        "is_paused": False,
        "metadata": {"resume_policy": "resume_full"},
    }
    update_trader_mock = AsyncMock(return_value=after)
    create_config_revision_mock = AsyncMock(return_value=None)
    create_event_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value=before))
    monkeypatch.setattr(routes_traders, "update_trader", update_trader_mock)
    monkeypatch.setattr(routes_traders, "create_config_revision", create_config_revision_mock)
    monkeypatch.setattr(routes_traders, "create_trader_event", create_event_mock)

    request = routes_traders.TraderPatchRequest(
        is_enabled=True,
        is_paused=False,
        requested_by="tester",
    )
    result = await routes_traders.update_trader_route(trader_id=trader_id, request=request, session=session_obj)

    assert result == after
    update_payload = update_trader_mock.await_args.args[2]
    assert update_payload["is_enabled"] is True
    assert update_payload["is_paused"] is False
    assert update_payload["metadata"]["loss_streak_reset_reason"] == "operator_resume"
    assert isinstance(update_payload["metadata"]["loss_streak_reset_at"], str)
    assert update_payload["metadata"]["loss_streak_reset_at"]
    event_payload = create_event_mock.await_args.kwargs["payload"]
    assert event_payload["loss_streak_reset_at"] == update_payload["metadata"]["loss_streak_reset_at"]


@pytest.mark.asyncio
async def test_get_trader_live_wallet_positions_forwards_include_managed(monkeypatch):
    session_obj = object()
    trader_id = "live-trader-1"
    response_payload = {
        "trader_id": trader_id,
        "wallet_address": "0xabc",
        "positions": [],
        "managed_token_ids": [],
        "managed_order_ids": [],
        "summary": {
            "total_positions": 0,
            "managed_positions": 0,
            "unmanaged_positions": 0,
            "returned_positions": 0,
        },
    }
    list_mock = AsyncMock(return_value=response_payload)
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value={"id": trader_id}))
    monkeypatch.setattr(routes_traders, "list_live_wallet_positions_for_trader", list_mock)

    result = await routes_traders.get_trader_live_wallet_positions(
        trader_id=trader_id,
        include_managed=False,
        session=session_obj,
    )

    assert result == response_payload
    list_mock.assert_awaited_once_with(
        session_obj,
        trader_id=trader_id,
        include_managed=False,
    )


@pytest.mark.asyncio
async def test_adopt_trader_live_wallet_position_creates_event(monkeypatch):
    session_obj = object()
    trader_id = "live-trader-1"
    adopt_result = {
        "status": "adopted",
        "trader_id": trader_id,
        "wallet_address": "0xabc",
        "token_id": "token-1",
        "market_id": "0x" + "1" * 64,
        "direction": "buy_yes",
        "order": {"id": "order-123"},
        "position_inventory": {"open_positions": 1},
    }
    adopt_mock = AsyncMock(return_value=adopt_result)
    create_event_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value={"id": trader_id}))
    monkeypatch.setattr(routes_traders, "adopt_live_wallet_position", adopt_mock)
    monkeypatch.setattr(routes_traders, "create_trader_event", create_event_mock)

    request = routes_traders.TraderLiveWalletPositionAdoptRequest(
        token_id="token-1",
        reason="adopt_manual_trade",
        requested_by="tester",
    )
    result = await routes_traders.adopt_trader_live_wallet_position(
        trader_id=trader_id,
        request=request,
        session=session_obj,
    )

    assert result == adopt_result
    adopt_mock.assert_awaited_once_with(
        session_obj,
        trader_id=trader_id,
        token_id="token-1",
        reason="adopt_manual_trade",
    )
    create_event_mock.assert_awaited_once()
    event_kwargs = create_event_mock.await_args.kwargs
    assert event_kwargs["event_type"] == "trader_live_position_adopted"
    assert event_kwargs["payload"]["order_id"] == "order-123"


@pytest.mark.asyncio
async def test_adopt_trader_live_wallet_position_maps_conflict(monkeypatch):
    monkeypatch.setattr(routes_traders, "get_trader", AsyncMock(return_value={"id": "live-trader-1"}))
    monkeypatch.setattr(
        routes_traders,
        "adopt_live_wallet_position",
        AsyncMock(side_effect=ValueError("Token 'token-1' is already managed by order order-123")),
    )

    with pytest.raises(HTTPException) as excinfo:
        await routes_traders.adopt_trader_live_wallet_position(
            trader_id="live-trader-1",
            request=routes_traders.TraderLiveWalletPositionAdoptRequest(token_id="token-1"),
            session=object(),
        )

    assert excinfo.value.status_code == 409
