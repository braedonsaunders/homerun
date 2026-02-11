import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import asyncio

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import weather_worker


class _DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_worker_respects_pause_without_manual_request(monkeypatch):
    monkeypatch.setattr(weather_worker, "AsyncSessionLocal", lambda: _DummySession())
    monkeypatch.setattr(
        weather_worker.shared_state, "write_weather_snapshot", AsyncMock()
    )
    monkeypatch.setattr(
        weather_worker.shared_state,
        "read_weather_control",
        AsyncMock(
            return_value={
                "is_paused": True,
                "requested_scan_at": None,
                "scan_interval_seconds": 14400,
            }
        ),
    )
    monkeypatch.setattr(
        weather_worker.shared_state,
        "get_weather_settings",
        AsyncMock(return_value={"enabled": True, "auto_run": True, "scan_interval_seconds": 14400}),
    )
    run_cycle_mock = AsyncMock(return_value={"markets": 0, "opportunities": 0, "intents": 0})
    monkeypatch.setattr(
        weather_worker.weather_workflow_orchestrator, "run_cycle", run_cycle_mock
    )
    monkeypatch.setattr(
        weather_worker.shared_state, "clear_weather_scan_request", AsyncMock()
    )
    monkeypatch.setattr(
        weather_worker.asyncio,
        "sleep",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await weather_worker._run_loop()

    run_cycle_mock.assert_not_called()


@pytest.mark.asyncio
async def test_worker_runs_once_when_manual_request_is_set(monkeypatch):
    monkeypatch.setattr(weather_worker, "AsyncSessionLocal", lambda: _DummySession())
    monkeypatch.setattr(
        weather_worker.shared_state, "write_weather_snapshot", AsyncMock()
    )
    monkeypatch.setattr(
        weather_worker.shared_state,
        "read_weather_control",
        AsyncMock(
            return_value={
                "is_paused": True,
                "requested_scan_at": datetime.now(timezone.utc),
                "scan_interval_seconds": 14400,
            }
        ),
    )
    monkeypatch.setattr(
        weather_worker.shared_state,
        "get_weather_settings",
        AsyncMock(return_value={"enabled": True, "auto_run": True, "scan_interval_seconds": 14400}),
    )
    run_cycle_mock = AsyncMock(return_value={"markets": 1, "opportunities": 1, "intents": 1})
    monkeypatch.setattr(
        weather_worker.weather_workflow_orchestrator, "run_cycle", run_cycle_mock
    )
    clear_mock = AsyncMock()
    monkeypatch.setattr(weather_worker.shared_state, "clear_weather_scan_request", clear_mock)
    monkeypatch.setattr(
        weather_worker.asyncio,
        "sleep",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await weather_worker._run_loop()

    run_cycle_mock.assert_awaited_once()
    clear_mock.assert_awaited_once()
