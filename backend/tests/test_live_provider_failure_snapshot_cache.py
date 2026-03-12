import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import trader_orchestrator_worker


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _Session:
    def __init__(self, rows):
        self._rows = rows
        self.execute_calls = 0
        self.fail = False

    async def execute(self, _query):
        self.execute_calls += 1
        if self.fail:
            raise RuntimeError("transient execute failure")
        return _Result(self._rows)


@pytest.mark.asyncio
async def test_live_provider_failure_snapshot_uses_cache_within_ttl(monkeypatch):
    now = datetime(2026, 3, 12, 19, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(trader_orchestrator_worker, "utcnow", lambda: now)

    row = SimpleNamespace(
        id="order-1",
        status="failed",
        error_message="connection refused",
        payload_json={},
        updated_at=now,
    )
    session = _Session([row])
    trader_orchestrator_worker._live_provider_failure_snapshot_cache.clear()
    try:
        first = await trader_orchestrator_worker._live_provider_failure_snapshot(
            session,
            trader_id="trader-1",
            window_seconds=120,
        )
        second = await trader_orchestrator_worker._live_provider_failure_snapshot(
            session,
            trader_id="trader-1",
            window_seconds=120,
        )

        assert session.execute_calls == 1
        assert first == second
        assert first["count"] == 1
    finally:
        trader_orchestrator_worker._live_provider_failure_snapshot_cache.clear()


@pytest.mark.asyncio
async def test_live_provider_failure_snapshot_uses_stale_cache_on_refresh_failure(monkeypatch):
    base = datetime(2026, 3, 12, 19, 0, tzinfo=timezone.utc)
    timeline = [base, base + timedelta(seconds=3)]
    monkeypatch.setattr(trader_orchestrator_worker, "utcnow", lambda: timeline.pop(0))

    row = SimpleNamespace(
        id="order-2",
        status="failed",
        error_message="gateway timeout",
        payload_json={},
        updated_at=base,
    )
    session = _Session([row])
    trader_orchestrator_worker._live_provider_failure_snapshot_cache.clear()
    try:
        baseline = await trader_orchestrator_worker._live_provider_failure_snapshot(
            session,
            trader_id="trader-2",
            window_seconds=120,
        )
        session.fail = True
        recovered = await trader_orchestrator_worker._live_provider_failure_snapshot(
            session,
            trader_id="trader-2",
            window_seconds=120,
        )

        assert session.execute_calls == 2
        assert recovered == baseline
    finally:
        trader_orchestrator_worker._live_provider_failure_snapshot_cache.clear()
