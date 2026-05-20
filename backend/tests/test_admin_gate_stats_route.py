"""Tests for the Phase 4 admin gate-stats API endpoints.

These tests call the route handler functions directly rather than via a
TestClient — that pattern matches the rest of the suite (see
``test_routes_maintenance_flush.py``) and keeps the test fast without
spinning up the full FastAPI app.

Coverage:

* ``GET  /api/admin/gate-stats`` returns the live GateMetrics snapshot
  in the documented JSON shape.
* ``POST /api/admin/gate-stats/reset`` clears every gate.
* ``POST /api/admin/gate-stats/reset?gate=<name>`` clears only one gate.
* The reset response echoes how many gates existed before the reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_admin
from services.trader_orchestrator.gate_pipeline import (
    CostClass,
    GateResult,
    get_gate_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics_around_each_test():
    """Each test starts with a clean global GateMetrics so siblings can't
    leak state.  We restore it on teardown for the same reason."""
    metrics = get_gate_metrics()
    metrics.reset()
    yield
    metrics.reset()


def _record(name: str, *, passes: int = 0, rejects: int = 0, latency_ms: float = 5.0, budget_ms: float | None = None) -> None:
    m = get_gate_metrics()
    for _ in range(passes):
        m.record(name, CostClass.L1_CACHED, GateResult(passed=True, latency_ms=latency_ms), budget_ms=budget_ms)
    for _ in range(rejects):
        m.record(name, CostClass.L1_CACHED, GateResult(passed=False, latency_ms=latency_ms, reason="x"), budget_ms=budget_ms)


# ---------- GET /api/admin/gate-stats ----------


@pytest.mark.asyncio
async def test_get_gate_stats_empty_returns_no_gates():
    response = await routes_admin.get_gate_stats()
    assert response.gates == []
    assert response.snapshot_at  # ISO-8601 string present


@pytest.mark.asyncio
async def test_get_gate_stats_shape_matches_documented_response():
    _record("alpha", passes=8, rejects=2, latency_ms=3.0, budget_ms=50.0)
    response = await routes_admin.get_gate_stats()

    payload = response.model_dump()
    assert "snapshot_at" in payload
    assert isinstance(payload["gates"], list) and len(payload["gates"]) == 1

    entry = payload["gates"][0]
    expected_keys = {
        "name",
        "cost_class",
        "runs",
        "rejects",
        "reject_rate",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "budget_ms",
        "budget_violations",
        "auto_demoted",
    }
    assert expected_keys.issubset(entry.keys())
    assert entry["name"] == "alpha"
    assert entry["runs"] == 10
    assert entry["rejects"] == 2
    assert entry["reject_rate"] == pytest.approx(0.2, abs=1e-3)
    assert entry["budget_ms"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_get_gate_stats_sorted_by_runs_desc():
    _record("low_volume", passes=1)
    _record("high_volume", passes=50)
    _record("mid_volume", passes=10)

    response = await routes_admin.get_gate_stats()
    names = [g.name for g in response.gates]
    assert names == ["high_volume", "mid_volume", "low_volume"]


# ---------- POST /api/admin/gate-stats/reset ----------


@pytest.mark.asyncio
async def test_post_reset_with_no_query_clears_everything():
    _record("a", passes=5)
    _record("b", passes=10)
    metrics = get_gate_metrics()
    assert len(metrics.snapshot()) == 2

    payload = await routes_admin.reset_gate_stats(gate=None)
    assert payload["status"] == "ok"
    assert payload["reset_gate"] == "*"
    assert payload["gates_before_reset"] == 2
    assert metrics.snapshot() == []


@pytest.mark.asyncio
async def test_post_reset_with_gate_param_clears_only_one():
    _record("keep_me", passes=3)
    _record("drop_me", passes=4)

    payload = await routes_admin.reset_gate_stats(gate="drop_me")

    assert payload["reset_gate"] == "drop_me"
    assert payload["gates_before_reset"] == 2

    metrics = get_gate_metrics()
    remaining = {s.name for s in metrics.snapshot()}
    assert remaining == {"keep_me"}


@pytest.mark.asyncio
async def test_post_reset_with_unknown_gate_is_a_noop():
    _record("real_gate", passes=2)
    payload = await routes_admin.reset_gate_stats(gate="not_a_real_gate")
    assert payload["status"] == "ok"
    # The real gate is still there.
    metrics = get_gate_metrics()
    names = {s.name for s in metrics.snapshot()}
    assert names == {"real_gate"}


@pytest.mark.asyncio
async def test_post_reset_with_blank_gate_treats_as_full_reset():
    """A whitespace-only or empty gate param should reset everything,
    not look for a literally-named "" gate."""
    _record("a", passes=1)
    _record("b", passes=1)
    payload = await routes_admin.reset_gate_stats(gate="   ")
    assert payload["reset_gate"] == "*"
    assert get_gate_metrics().snapshot() == []
