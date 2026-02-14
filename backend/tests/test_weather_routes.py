import sys
from pathlib import Path
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_weather_workflow


@pytest.mark.asyncio
async def test_status_endpoint_includes_pause_and_pending(monkeypatch):
    fake_session = object()
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "get_weather_status_from_db",
        AsyncMock(
            return_value={
                "running": True,
                "enabled": True,
                "interval_seconds": 14400,
                "last_scan": "2026-01-01T00:00:00Z",
                "opportunities_count": 4,
                "current_activity": "ok",
                "stats": {},
            }
        ),
    )
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "list_weather_intents",
        AsyncMock(return_value=[object(), object(), object()]),
    )
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "read_weather_control",
        AsyncMock(
            return_value={
                "is_paused": True,
                "requested_scan_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ),
    )

    out = await routes_weather_workflow.get_weather_workflow_status(fake_session)
    assert out["paused"] is True
    assert out["pending_intents"] == 3
    assert out["requested_scan_at"] is not None


@pytest.mark.asyncio
async def test_update_settings_syncs_control_interval(monkeypatch):
    fake_session = object()
    update_mock = AsyncMock(
        return_value={
            "scan_interval_seconds": 7200,
            "enabled": True,
        }
    )
    set_interval_mock = AsyncMock()
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "update_weather_settings",
        update_mock,
    )
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "set_weather_interval",
        set_interval_mock,
    )

    req = routes_weather_workflow.WeatherWorkflowSettingsRequest(scan_interval_seconds=7200)
    out = await routes_weather_workflow.update_weather_workflow_settings(
        request=req,
        session=fake_session,
    )

    assert out["status"] == "success"
    update_mock.assert_awaited_once()
    set_interval_mock.assert_awaited_once_with(fake_session, 7200)


@pytest.mark.asyncio
async def test_update_settings_accepts_temperature_unit(monkeypatch):
    fake_session = object()
    update_mock = AsyncMock(return_value={"temperature_unit": "C"})
    set_interval_mock = AsyncMock()
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "update_weather_settings",
        update_mock,
    )
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "set_weather_interval",
        set_interval_mock,
    )

    req = routes_weather_workflow.WeatherWorkflowSettingsRequest(temperature_unit="C")
    out = await routes_weather_workflow.update_weather_workflow_settings(
        request=req,
        session=fake_session,
    )

    assert out["status"] == "success"
    update_mock.assert_awaited_once_with(fake_session, {"temperature_unit": "C"})
    set_interval_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_weather_workflow_executes_immediately(monkeypatch):
    fake_session = object()
    monkeypatch.setattr(routes_weather_workflow.global_pause_state, "_paused", False)
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "get_weather_settings",
        AsyncMock(return_value={"orchestrator_max_age_minutes": 240}),
    )
    run_cycle_mock = AsyncMock(return_value={"status": "completed", "markets": 5, "opportunities": 2, "intents": 1})
    monkeypatch.setattr(
        routes_weather_workflow.weather_workflow_orchestrator,
        "run_cycle",
        run_cycle_mock,
    )
    list_intents_mock = AsyncMock(return_value=[object(), object()])
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "list_weather_intents",
        list_intents_mock,
    )
    emit_mock = AsyncMock(return_value=2)
    monkeypatch.setattr(routes_weather_workflow, "emit_weather_intent_signals", emit_mock)
    clear_request_mock = AsyncMock()
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "clear_weather_scan_request",
        clear_request_mock,
    )
    monkeypatch.setattr(
        routes_weather_workflow,
        "_build_status_payload",
        AsyncMock(return_value={"running": True, "paused": False}),
    )

    out = await routes_weather_workflow.run_weather_workflow_once(fake_session)
    assert out["status"] == "completed"
    assert out["signals_emitted"] == 2
    assert out["cycle"]["status"] == "completed"
    run_cycle_mock.assert_awaited_once_with(fake_session)
    list_intents_mock.assert_awaited_once_with(fake_session, status_filter="pending", limit=2000)
    emit_mock.assert_awaited_once()
    clear_request_mock.assert_awaited_once_with(fake_session)


@pytest.mark.asyncio
async def test_get_weather_opportunities_enforces_tradability_filters(monkeypatch):
    fake_session = object()
    get_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "get_weather_opportunities_from_db",
        get_mock,
    )

    out = await routes_weather_workflow.get_weather_opportunities(
        session=fake_session,
        min_edge=12.0,
        direction="buy_no",
        max_entry=0.2,
        location="Wellington",
        limit=20,
        offset=0,
    )

    assert out["total"] == 0
    assert out["opportunities"] == []
    get_mock.assert_awaited_once_with(
        fake_session,
        min_edge_percent=12.0,
        direction="buy_no",
        max_entry_price=0.2,
        location_query="Wellington",
        target_date=None,
        require_tradable_markets=True,
        exclude_near_resolution=True,
        include_report_only=False,
    )


@pytest.mark.asyncio
async def test_get_weather_opportunities_passes_target_date(monkeypatch):
    fake_session = object()
    get_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "get_weather_opportunities_from_db",
        get_mock,
    )

    target = date(2026, 2, 13)
    out = await routes_weather_workflow.get_weather_opportunities(
        session=fake_session,
        target_date=target,
        limit=20,
        offset=0,
    )

    assert out["total"] == 0
    await_args = get_mock.await_args
    assert await_args is not None
    assert await_args.kwargs.get("target_date") == target


@pytest.mark.asyncio
async def test_get_weather_opportunity_dates_uses_shared_date_counts(monkeypatch):
    fake_session = object()
    date_mock = AsyncMock(
        return_value=[
            {"date": "2026-02-13", "count": 3},
            {"date": "2026-02-14", "count": 1},
        ]
    )
    monkeypatch.setattr(
        routes_weather_workflow.shared_state,
        "get_weather_target_date_counts_from_db",
        date_mock,
    )

    out = await routes_weather_workflow.get_weather_opportunity_dates(
        session=fake_session,
        min_edge=8.0,
        direction="buy_yes",
        max_entry=0.25,
        location="Wellington",
    )

    assert out["total_dates"] == 2
    assert out["dates"][0]["date"] == "2026-02-13"
    date_mock.assert_awaited_once_with(
        fake_session,
        min_edge_percent=8.0,
        direction="buy_yes",
        max_entry_price=0.25,
        location_query="Wellington",
        require_tradable_markets=True,
        exclude_near_resolution=True,
        include_report_only=False,
    )
