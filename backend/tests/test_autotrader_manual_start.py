import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_auto_trader
from services import autotrader_state


class _DummySession:
    def __init__(self):
        self.committed = False

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_start_route_enables_unpauses_and_clears_kill_switch(monkeypatch):
    fake_session = object()
    update_mock = AsyncMock(return_value={"mode": "paper", "is_enabled": True, "is_paused": False, "kill_switch": False})
    monkeypatch.setattr(routes_auto_trader, "update_autotrader_control", update_mock)
    monkeypatch.setattr(routes_auto_trader, "create_autotrader_event", AsyncMock(return_value=None))

    out = await routes_auto_trader.start_auto_trader(mode="paper", session=fake_session)

    assert out["status"] == "started"
    update_mock.assert_awaited_once_with(
        fake_session,
        is_enabled=True,
        is_paused=False,
        mode="paper",
        kill_switch=False,
    )


@pytest.mark.asyncio
async def test_start_route_persists_paper_account_id(monkeypatch):
    fake_session = object()
    update_mock = AsyncMock(
        return_value={
            "mode": "paper",
            "is_enabled": True,
            "is_paused": False,
            "kill_switch": False,
            "settings": {"paper_account_id": "paper-123"},
        }
    )
    read_control_mock = AsyncMock(return_value={"settings": {"enabled_strategies": ["news"]}})
    create_event_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(routes_auto_trader, "read_autotrader_control", read_control_mock)
    monkeypatch.setattr(routes_auto_trader, "update_autotrader_control", update_mock)
    monkeypatch.setattr(routes_auto_trader, "create_autotrader_event", create_event_mock)

    out = await routes_auto_trader.start_auto_trader(
        mode="paper",
        account_id="paper-123",
        session=fake_session,
    )

    assert out["status"] == "started"
    read_control_mock.assert_awaited_once_with(fake_session)
    update_mock.assert_awaited_once_with(
        fake_session,
        is_enabled=True,
        is_paused=False,
        mode="paper",
        kill_switch=False,
        settings={"enabled_strategies": ["news"], "paper_account_id": "paper-123"},
    )
    create_event_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_route_disables_and_pauses(monkeypatch):
    fake_session = object()
    update_mock = AsyncMock(return_value={"is_enabled": False, "is_paused": True})
    monkeypatch.setattr(routes_auto_trader, "update_autotrader_control", update_mock)
    monkeypatch.setattr(routes_auto_trader, "create_autotrader_event", AsyncMock(return_value=None))

    out = await routes_auto_trader.stop_auto_trader(session=fake_session)

    assert out["status"] == "stopped"
    update_mock.assert_awaited_once_with(fake_session, is_enabled=False, is_paused=True)


@pytest.mark.asyncio
async def test_reset_autotrader_for_manual_start(monkeypatch):
    row = SimpleNamespace(
        is_enabled=True,
        is_paused=False,
        requested_run_at="2026-01-01T00:00:00Z",
        updated_at=None,
    )
    session = _DummySession()

    monkeypatch.setattr(
        autotrader_state,
        "ensure_autotrader_control",
        AsyncMock(return_value=row),
    )
    monkeypatch.setattr(
        autotrader_state,
        "read_autotrader_control",
        AsyncMock(return_value={
            "is_enabled": False,
            "is_paused": True,
            "requested_run_at": None,
        }),
    )

    out = await autotrader_state.reset_autotrader_for_manual_start(session)

    assert row.is_enabled is False
    assert row.is_paused is True
    assert row.requested_run_at is None
    assert row.updated_at is not None
    assert session.committed is True
    assert out["is_enabled"] is False
    assert out["is_paused"] is True
