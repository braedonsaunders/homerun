import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models import database
from services import discovery_shared_state, pause_state, shared_state, worker_state
from services.news import shared_state as news_shared_state
from services.weather import shared_state as weather_shared_state


class _SessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _patch_common_controls(monkeypatch, *, all_paused: bool) -> None:
    paused_value = {"is_paused": True}
    running_value = {"is_paused": False}

    scanner = paused_value if all_paused else running_value
    monkeypatch.setattr(shared_state, "read_scanner_control", AsyncMock(return_value=scanner))
    monkeypatch.setattr(news_shared_state, "read_news_control", AsyncMock(return_value=paused_value))
    monkeypatch.setattr(
        weather_shared_state,
        "read_weather_control",
        AsyncMock(return_value=paused_value),
    )
    monkeypatch.setattr(
        discovery_shared_state,
        "read_discovery_control",
        AsyncMock(return_value=paused_value),
    )
    monkeypatch.setattr(
        "services.trader_orchestrator_state.read_orchestrator_control",
        AsyncMock(return_value=paused_value),
    )
    monkeypatch.setattr(
        worker_state,
        "read_worker_control",
        AsyncMock(return_value=paused_value),
    )


@pytest.mark.asyncio
async def test_refresh_from_db_sets_paused_when_all_controls_paused(monkeypatch):
    state = pause_state.GlobalPauseState(refresh_interval_seconds=0.01)

    monkeypatch.setattr(database, "AsyncSessionLocal", lambda: _SessionContext())
    _patch_common_controls(monkeypatch, all_paused=True)

    paused = await state.refresh_from_db(force=True)
    assert paused is True


@pytest.mark.asyncio
async def test_refresh_from_db_sets_not_paused_when_any_control_running(monkeypatch):
    state = pause_state.GlobalPauseState(refresh_interval_seconds=0.01)

    monkeypatch.setattr(database, "AsyncSessionLocal", lambda: _SessionContext())
    _patch_common_controls(monkeypatch, all_paused=False)

    paused = await state.refresh_from_db(force=True)
    assert paused is False
