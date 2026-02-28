from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import shared_state


class _FakeScalarResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _FakeExecuteResult:
    def __init__(self, row):
        self._row = row

    def scalars(self):
        return _FakeScalarResult(self._row)


class _FakeBind:
    class _FakeDialect:
        name = "postgresql"

    dialect = _FakeDialect()


class _FakeSession:
    def __init__(self, row):
        self._row = row
        self.deleted = []
        self.last_stmt_sql = None

    def get_bind(self):
        return _FakeBind()

    async def execute(self, stmt):
        self.last_stmt_sql = str(stmt)
        return _FakeExecuteResult(self._row)

    async def delete(self, row):
        self.deleted.append(row)


@pytest.mark.asyncio
async def test_drop_oldest_pending_scanner_batch_filters_out_active_leases(monkeypatch) -> None:
    fake_row = SimpleNamespace(id="droppable-batch")
    fake_session = _FakeSession(fake_row)
    commit_spy = AsyncMock()
    monkeypatch.setattr(shared_state, "_commit_with_retry", commit_spy)

    dropped_id = await shared_state.drop_oldest_pending_scanner_batch(fake_session)

    assert dropped_id == "droppable-batch"
    assert fake_session.deleted == [fake_row]
    assert "lease_expires_at" in str(fake_session.last_stmt_sql or "")
    assert "processed_at IS NULL" in str(fake_session.last_stmt_sql or "")
    commit_spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_drop_oldest_pending_scanner_batch_returns_none_when_no_droppable_row(monkeypatch) -> None:
    fake_session = _FakeSession(None)
    commit_spy = AsyncMock()
    monkeypatch.setattr(shared_state, "_commit_with_retry", commit_spy)

    dropped_id = await shared_state.drop_oldest_pending_scanner_batch(fake_session)

    assert dropped_id is None
    assert fake_session.deleted == []
    commit_spy.assert_not_awaited()
