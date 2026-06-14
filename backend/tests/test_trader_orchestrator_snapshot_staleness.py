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
        updated_at=last_run_at,
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
    assert payload["is_stale"] is False
    assert payload["heartbeat_lag_seconds"] == 5.0


def test_snapshot_not_running_when_heartbeat_stale(monkeypatch):
    now = datetime(2026, 2, 17, 4, 0, 0)
    monkeypatch.setattr(state, "_now", lambda: now)

    # Staleness threshold is max(120s min, interval * 30x multiplier).
    # With interval_seconds=2, threshold = max(120, 60) = 120s.
    # Lag must exceed 120s to be detected as stale.
    row = _snapshot(
        running=True,
        interval_seconds=2,
        last_run_at=now - timedelta(seconds=121),
    )
    payload = state._serialize_snapshot(row)
    assert payload["running"] is False
    assert payload["is_stale"] is True


def test_snapshot_staleness_uses_heartbeat_updated_at(monkeypatch):
    now = datetime(2026, 2, 17, 4, 0, 0)
    monkeypatch.setattr(state, "_now", lambda: now)

    row = _snapshot(
        running=True,
        interval_seconds=2,
        last_run_at=now - timedelta(seconds=300),
    )
    row.updated_at = now - timedelta(seconds=5)

    payload = state._serialize_snapshot(row)
    assert payload["running"] is True
    assert payload["is_stale"] is False
    assert payload["heartbeat_lag_seconds"] == 5.0


def test_runtime_state_marks_enabled_stale_as_stale_not_stopped():
    payload = state.compose_orchestrator_runtime_state(
        {
            "is_enabled": True,
            "is_paused": False,
            "kill_switch": False,
        },
        {
            "running": False,
            "is_stale": True,
            "heartbeat_lag_seconds": 180.0,
            "stale_after_seconds": 120.0,
            "current_activity": "Cycle[requested_run:general] signals=29 decisions=29 orders=0",
        },
    )

    assert payload["state"] == "stale"
    assert payload["label"] == "STALE"
    assert payload["desired_active"] is True
    assert payload["worker_stale"] is True
    assert payload["can_place_orders"] is False


def test_runtime_state_marks_running_without_entry_eligible_traders_manage_only():
    payload = state.compose_orchestrator_runtime_state(
        {
            "is_enabled": True,
            "is_paused": False,
            "kill_switch": False,
        },
        {
            "running": True,
            "is_stale": False,
            "heartbeat_lag_seconds": 5.0,
            "stale_after_seconds": 120.0,
            "current_activity": "Cycle[scheduled:general] signals=0 decisions=0 orders=0",
        },
        {
            "traders_total": 9,
            "traders_enabled": 8,
            "traders_running": 1,
            "entry_enabled_traders": 0,
            "blocked_running_traders": 1,
            "paused_traders": 7,
            "disabled_traders": 1,
            "can_process_entry_signals": False,
        },
    )

    assert payload["state"] == "running"
    assert payload["label"] == "MANAGE-ONLY"
    assert payload["can_place_orders"] is False
    assert payload["global_order_entry_open"] is True
    assert payload["entry_signal_processing_enabled"] is False
    assert payload["reason"] == "All running traders have per-bot block_new_orders enabled."
