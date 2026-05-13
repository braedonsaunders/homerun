import pytest

from services.autoresearch_service import AutoresearchService, _resolve_autoresearch_status


class _FakeSession:
    def __init__(self) -> None:
        self.committed = False

    async def commit(self) -> None:
        self.committed = True


class _FakeExperiment:
    def __init__(self, experiment_id: str, status: str) -> None:
        self.id = experiment_id
        self.status = status
        self.finished_at = None
        self.updated_at = None


def test_resolve_autoresearch_status_requires_matching_active_experiment() -> None:
    assert _resolve_autoresearch_status("exp-1", "running", "exp-1") == "running"
    assert _resolve_autoresearch_status("exp-1", "running", None) == "failed"
    assert _resolve_autoresearch_status("exp-1", "completed", None) == "completed"


@pytest.mark.asyncio
async def test_resolve_runtime_status_marks_orphaned_running_experiment_failed() -> None:
    service = AutoresearchService()
    service._active_experiments.clear()
    service._stop_flags.clear()
    session = _FakeSession()
    row = _FakeExperiment("exp-1", "running")

    status = await service._resolve_runtime_status(session, row, "strategy:tail-end-carry")

    assert status == "failed"
    assert row.status == "failed"
    assert row.finished_at is not None
    assert row.updated_at is not None
    assert session.committed


@pytest.mark.asyncio
async def test_resolve_runtime_status_keeps_owned_active_experiment_running() -> None:
    service = AutoresearchService()
    service._active_experiments.clear()
    service._stop_flags.clear()
    service._active_experiments["strategy:tail-end-carry"] = "exp-1"
    session = _FakeSession()
    row = _FakeExperiment("exp-1", "running")

    status = await service._resolve_runtime_status(session, row, "strategy:tail-end-carry")

    assert status == "running"
    assert row.status == "running"
    assert row.finished_at is None
    assert not session.committed
