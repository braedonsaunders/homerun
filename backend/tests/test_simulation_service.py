import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.simulation import SimulationService


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _locked_exc() -> OperationalError:
    return OperationalError(
        "INSERT INTO simulation_accounts ...",
        {},
        Exception("database is locked"),
    )


def test_create_account_retries_sqlite_lock_and_succeeds(monkeypatch):
    state = {
        "commits": 0,
        "rollbacks": 0,
        "sessions": 0,
    }

    class _Session:
        def __init__(self):
            self.account = None

        async def execute(self, *_args, **_kwargs):
            return None

        def add(self, account):
            self.account = account

        async def commit(self):
            state["commits"] += 1
            if state["commits"] <= 2:
                raise _locked_exc()

        async def refresh(self, _account):
            return None

        async def rollback(self):
            state["rollbacks"] += 1

    def _session_factory():
        state["sessions"] += 1
        return _SessionContext(_Session())

    sleep_mock = AsyncMock()
    monkeypatch.setattr("services.simulation.AsyncSessionLocal", _session_factory)
    monkeypatch.setattr("services.simulation.asyncio.sleep", sleep_mock)
    monkeypatch.setattr(
        "services.simulation.settings.DATABASE_URL",
        "sqlite+aiosqlite:////tmp/test.db",
        raising=False,
    )

    service = SimulationService()
    account = asyncio.run(
        service.create_account(
            name="retry-test",
            initial_capital=12345.0,
            max_position_pct=12.0,
            max_positions=7,
        )
    )

    assert account.name == "retry-test"
    assert account.initial_capital == 12345.0
    assert account.max_position_size_pct == 12.0
    assert account.max_open_positions == 7
    assert state["sessions"] == 3
    assert state["commits"] == 3
    assert state["rollbacks"] == 2
    assert sleep_mock.await_count == 2


def test_create_account_raises_after_retry_budget(monkeypatch):
    state = {
        "commits": 0,
        "rollbacks": 0,
    }

    class _Session:
        async def execute(self, *_args, **_kwargs):
            return None

        def add(self, _account):
            return None

        async def commit(self):
            state["commits"] += 1
            raise _locked_exc()

        async def refresh(self, _account):
            return None

        async def rollback(self):
            state["rollbacks"] += 1

    def _session_factory():
        return _SessionContext(_Session())

    sleep_mock = AsyncMock()
    monkeypatch.setattr("services.simulation.AsyncSessionLocal", _session_factory)
    monkeypatch.setattr("services.simulation.asyncio.sleep", sleep_mock)
    monkeypatch.setattr(
        "services.simulation.settings.DATABASE_URL",
        "sqlite+aiosqlite:////tmp/test.db",
        raising=False,
    )

    service = SimulationService()
    service.SQLITE_LOCK_RETRY_ATTEMPTS = 3
    service.SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS = 0
    service.SQLITE_LOCK_RETRY_MAX_DELAY_SECONDS = 0

    with pytest.raises(OperationalError):
        asyncio.run(service.create_account(name="retry-fail"))

    assert state["commits"] == 3
    assert state["rollbacks"] == 3
    assert sleep_mock.await_count == 2
