from datetime import datetime, timedelta

import services.trader_orchestrator_state as state
from models.database import TraderOrchestratorSnapshot


def _snapshot(*, running: bool, interval_seconds: int, last_run_at: datetime) -> TraderOrchestratorSnapshot:
    return TraderOrchestratorSnapshot(
        id="latest",
        running=running,
        enabled=True,
        interval_seconds=interval_seconds,
        last_run_at=last_run_at,
    )


def test_snapshot_running_when_heartbeat_fresh(monkeypatch):
    now = datetime(2026, 2, 17, 4, 0, 0)
    monkeypatch.setattr(state, "_now", lambda: now)

    row = _snapshot(
        running=True,
        interval_seconds=2,
        last_run_at=now - timedelta(seconds=5),
    )
    payload = state._serialize_snapshot(row)
    assert payload["running"] is True


def test_snapshot_not_running_when_heartbeat_stale(monkeypatch):
    now = datetime(2026, 2, 17, 4, 0, 0)
    monkeypatch.setattr(state, "_now", lambda: now)

    row = _snapshot(
        running=True,
        interval_seconds=2,
        last_run_at=now - timedelta(seconds=16),
    )
    payload = state._serialize_snapshot(row)
    assert payload["running"] is False
