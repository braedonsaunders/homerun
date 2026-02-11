import sys
from pathlib import Path
from datetime import datetime, timezone
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

    req = routes_weather_workflow.WeatherWorkflowSettingsRequest(
        scan_interval_seconds=7200
    )
    out = await routes_weather_workflow.update_weather_workflow_settings(
        request=req,
        session=fake_session,
    )

    assert out["status"] == "success"
    update_mock.assert_awaited_once()
    set_interval_mock.assert_awaited_once_with(fake_session, 7200)
