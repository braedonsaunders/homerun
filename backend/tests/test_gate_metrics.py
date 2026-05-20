"""Tests for Phase 4 GateMetrics: per-gate latency + rejection aggregation.

Covers the public surface:

* ``record()`` updates runs / rejects / max_ms / latency window.
* ``snapshot()`` returns one entry per recorded gate with the correct
  p50/p95/p99 quantiles, run/reject counts, and reject_rate.
* ``reset()`` with no arg drops everything; with a ``gate_name`` it
  drops only that gate while leaving siblings untouched.
* Pipeline integration: running N signals through a 2-gate pipeline
  records ``runs=N`` for the first gate and ``runs <= N`` for the
  second (short-circuit aware).
* Budget-violation log fires when p95 over the last 32 samples exceeds
  ``2x`` the declared ``latency_budget_ms``.
* Rate-limit: two violations within 30 seconds log at most once.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.gate_pipeline import (
    CostClass,
    Gate,
    GateContext,
    GateMetrics,
    GateMetricsSnapshot,
    GatePipeline,
    GateResult,
    get_gate_metrics,
)


def _ctx() -> GateContext:
    return GateContext(runtime_signal=None, decision=None)


def _result(passed: bool, latency_ms: float, reason: str = "") -> GateResult:
    return GateResult(
        passed=passed,
        reason=reason or ("ok" if passed else "rejected"),
        latency_ms=float(latency_ms),
    )


# ---------- record() ----------


def test_record_increments_runs_and_rejects():
    m = GateMetrics()
    m.record("g", CostClass.L0_MEMORY, _result(True, 1.0))
    m.record("g", CostClass.L0_MEMORY, _result(False, 2.0))
    m.record("g", CostClass.L0_MEMORY, _result(False, 3.0))

    snaps = m.snapshot()
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.name == "g"
    assert snap.cost_class == "L0_MEMORY"
    assert snap.runs == 3
    assert snap.rejects == 2
    assert snap.reject_rate == pytest.approx(0.6667, abs=1e-3)
    assert snap.max_ms == pytest.approx(3.0)


def test_record_separate_gates_tracked_independently():
    m = GateMetrics()
    m.record("a", CostClass.L0_MEMORY, _result(True, 1.0))
    m.record("b", CostClass.L1_CACHED, _result(False, 5.0))
    m.record("b", CostClass.L1_CACHED, _result(False, 10.0))

    snaps = {s.name: s for s in m.snapshot()}
    assert set(snaps) == {"a", "b"}
    assert snaps["a"].runs == 1 and snaps["a"].rejects == 0
    assert snaps["b"].runs == 2 and snaps["b"].rejects == 2
    assert snaps["a"].cost_class == "L0_MEMORY"
    assert snaps["b"].cost_class == "L1_CACHED"


def test_record_zero_runs_means_empty_snapshot():
    m = GateMetrics()
    assert m.snapshot() == []


def test_record_keeps_budget_from_first_seen_or_updates_on_change():
    m = GateMetrics()
    m.record("g", CostClass.L0_MEMORY, _result(True, 1.0), budget_ms=50.0)
    m.record("g", CostClass.L0_MEMORY, _result(True, 2.0), budget_ms=75.0)
    snap = m.snapshot()[0]
    # Latest budget wins.
    assert snap.budget_ms == pytest.approx(75.0)


# ---------- percentile calculation ----------


def test_percentiles_match_known_sample():
    """For a uniform 1..100 ms sample, p50 ~= 50, p95 ~= 95, p99 ~= 99."""
    m = GateMetrics(window_size=128)
    for i in range(1, 101):
        m.record("g", CostClass.L0_MEMORY, _result(True, float(i)))
    snap = m.snapshot()[0]
    # statistics.quantiles with method='inclusive' over 1..100 places
    # the 50th percentile at 50.5, p95 at 95.05, p99 at 99.01.  We
    # allow a generous tolerance because the implementation can pick
    # either of two adjacent samples at the boundary.
    assert 49.0 <= snap.p50_ms <= 52.0
    assert 94.0 <= snap.p95_ms <= 96.0
    assert 98.0 <= snap.p99_ms <= 100.0
    assert snap.max_ms == pytest.approx(100.0)


def test_percentile_one_sample_returns_that_sample():
    m = GateMetrics()
    m.record("g", CostClass.L0_MEMORY, _result(True, 7.5))
    snap = m.snapshot()[0]
    assert snap.p50_ms == pytest.approx(7.5)
    assert snap.p95_ms == pytest.approx(7.5)
    assert snap.p99_ms == pytest.approx(7.5)


def test_window_caps_at_window_size():
    """Beyond window_size samples, percentiles reflect only the most
    recent window — older samples have aged out."""
    m = GateMetrics(window_size=10)
    # Fill window with 100ms samples …
    for _ in range(10):
        m.record("g", CostClass.L0_MEMORY, _result(True, 100.0))
    # … then overwrite with 1ms samples.
    for _ in range(10):
        m.record("g", CostClass.L0_MEMORY, _result(True, 1.0))

    snap = m.snapshot()[0]
    assert snap.runs == 20
    # Window is 1ms-only now → p99 should be ~1.0, not ~100.0.
    assert snap.p99_ms < 5.0
    # But max_ms is a lifetime stat, so the 100ms peak is preserved.
    assert snap.max_ms == pytest.approx(100.0)


# ---------- reset() ----------


def test_reset_all_clears_every_gate():
    m = GateMetrics()
    m.record("a", CostClass.L0_MEMORY, _result(True, 1.0))
    m.record("b", CostClass.L0_MEMORY, _result(False, 2.0))
    assert len(m.snapshot()) == 2
    m.reset()
    assert m.snapshot() == []


def test_reset_one_keeps_siblings():
    m = GateMetrics()
    m.record("a", CostClass.L0_MEMORY, _result(True, 1.0))
    m.record("b", CostClass.L0_MEMORY, _result(False, 2.0))
    m.reset("a")
    snaps = {s.name: s for s in m.snapshot()}
    assert set(snaps) == {"b"}
    assert snaps["b"].runs == 1


def test_reset_unknown_gate_is_a_noop():
    m = GateMetrics()
    m.record("a", CostClass.L0_MEMORY, _result(True, 1.0))
    m.reset("nonexistent")
    assert {s.name for s in m.snapshot()} == {"a"}


# ---------- pipeline integration ----------


@pytest.mark.asyncio
async def test_pipeline_records_each_gate_on_pass():
    """A pass-through pipeline records every gate for every run."""
    # Use a fresh metrics object via the module singleton so the
    # production code path under test (GatePipeline -> _metrics.record)
    # is exercised end-to-end.
    metrics = get_gate_metrics()
    metrics.reset()

    pipeline = GatePipeline(
        [
            Gate(name="m_a", cost_class=CostClass.L0_MEMORY, predicate=lambda c: GateResult(passed=True)),
            Gate(name="m_b", cost_class=CostClass.L0_MEMORY, predicate=lambda c: GateResult(passed=True)),
        ]
    )
    for _ in range(100):
        await pipeline.run(_ctx())

    snaps = {s.name: s for s in metrics.snapshot()}
    assert snaps["m_a"].runs == 100
    assert snaps["m_b"].runs == 100
    assert snaps["m_a"].rejects == 0
    assert snaps["m_b"].rejects == 0
    metrics.reset()


@pytest.mark.asyncio
async def test_pipeline_short_circuit_reflects_in_metrics():
    """When the first gate rejects on K of N runs, the second gate has
    runs == N - K (it didn't run during the short-circuited cycles)."""
    metrics = get_gate_metrics()
    metrics.reset()

    rejects_planned = {0, 3, 7, 11, 19, 42, 73}  # 7 rejections out of 100

    def _first(ctx: GateContext) -> GateResult:
        # Use a list-index-style counter via closure mutation.
        idx = _first.call_count  # type: ignore[attr-defined]
        _first.call_count = idx + 1  # type: ignore[attr-defined]
        if idx in rejects_planned:
            return GateResult(passed=False, reason="first_reject")
        return GateResult(passed=True)

    _first.call_count = 0  # type: ignore[attr-defined]

    pipeline = GatePipeline(
        [
            Gate(name="sc_first", cost_class=CostClass.L0_MEMORY, predicate=_first),
            Gate(
                name="sc_second",
                cost_class=CostClass.L0_MEMORY,
                predicate=lambda c: GateResult(passed=True),
            ),
        ]
    )
    for _ in range(100):
        await pipeline.run(_ctx())

    snaps = {s.name: s for s in metrics.snapshot()}
    assert snaps["sc_first"].runs == 100
    assert snaps["sc_first"].rejects == len(rejects_planned)
    assert snaps["sc_second"].runs == 100 - len(rejects_planned)
    assert snaps["sc_second"].rejects == 0
    metrics.reset()


def test_pipeline_run_sync_records_metrics():
    metrics = get_gate_metrics()
    metrics.reset()
    pipeline = GatePipeline(
        [
            Gate(
                name="sync_only",
                cost_class=CostClass.L0_MEMORY,
                predicate=lambda c: GateResult(passed=True),
            ),
        ]
    )
    for _ in range(5):
        pipeline.run_sync(_ctx())
    snaps = metrics.snapshot()
    assert any(s.name == "sync_only" and s.runs == 5 for s in snaps)
    metrics.reset()


# ---------- budget violations ----------


def test_budget_violation_logs_when_p95_exceeds_2x_budget():
    """With budget=10ms and 32 samples at 30ms, p95 = 30ms > 20ms → log."""
    m = GateMetrics()
    with patch(
        "services.trader_orchestrator.gate_pipeline.logger"
    ) as mock_logger:
        # Feed 32 samples all at 30ms → p95 is well above 2x the 10ms budget.
        for _ in range(32):
            m.record("slow", CostClass.L0_MEMORY, _result(True, 30.0), budget_ms=10.0)

        # At least one warning fired.
        warn_calls = [
            call for call in mock_logger.warning.call_args_list
            if call.args and "budget violation" in str(call.args[0]).lower()
        ]
        assert len(warn_calls) >= 1, f"Expected budget-violation warning, got: {mock_logger.warning.call_args_list}"

    snap = m.snapshot()[0]
    assert snap.budget_violations >= 1


def test_budget_violation_does_not_fire_under_threshold():
    """With budget=100ms and 32 samples at 30ms, ratio is 0.3x — no log."""
    m = GateMetrics()
    with patch(
        "services.trader_orchestrator.gate_pipeline.logger"
    ) as mock_logger:
        for _ in range(32):
            m.record("fast", CostClass.L0_MEMORY, _result(True, 30.0), budget_ms=100.0)

        warn_calls = [
            call for call in mock_logger.warning.call_args_list
            if call.args and "budget violation" in str(call.args[0]).lower()
        ]
        assert warn_calls == []

    snap = m.snapshot()[0]
    assert snap.budget_violations == 0


def test_budget_violation_does_not_fire_without_enough_samples():
    """Under 32 samples, no violation check happens even if every
    sample is way over budget — we wait until the lookback window is
    full to avoid false positives on a single-sample spike."""
    m = GateMetrics()
    with patch(
        "services.trader_orchestrator.gate_pipeline.logger"
    ) as mock_logger:
        for _ in range(31):
            m.record("warming_up", CostClass.L0_MEMORY, _result(True, 500.0), budget_ms=10.0)

        warn_calls = [
            call for call in mock_logger.warning.call_args_list
            if call.args and "budget violation" in str(call.args[0]).lower()
        ]
        assert warn_calls == []


def test_budget_violation_rate_limited_to_once_per_5_minutes():
    """Two violations within 30 seconds log at most once."""
    m = GateMetrics()
    with patch(
        "services.trader_orchestrator.gate_pipeline.logger"
    ) as mock_logger:
        # First batch of 32 samples → first violation log.
        for _ in range(32):
            m.record("noisy", CostClass.L0_MEMORY, _result(True, 100.0), budget_ms=10.0)
        first_warn_count = sum(
            1
            for call in mock_logger.warning.call_args_list
            if call.args and "budget violation" in str(call.args[0]).lower()
        )
        # Now feed more samples — each one re-evaluates the budget
        # check, but the rate-limit suppresses the log.
        for _ in range(32):
            m.record("noisy", CostClass.L0_MEMORY, _result(True, 100.0), budget_ms=10.0)
        second_warn_count = sum(
            1
            for call in mock_logger.warning.call_args_list
            if call.args and "budget violation" in str(call.args[0]).lower()
        )

    assert first_warn_count == 1, f"Expected exactly 1 warning after first batch, got {first_warn_count}"
    assert second_warn_count == 1, "Second batch within 5 minutes must not log again"
    # ... but the underlying budget_violations counter does keep climbing.
    snap = m.snapshot()[0]
    assert snap.budget_violations >= 2


def test_auto_demote_flag_off_by_default():
    m = GateMetrics()
    for _ in range(32):
        m.record("slow", CostClass.L0_MEMORY, _result(True, 100.0), budget_ms=10.0)
    snap = m.snapshot()[0]
    # auto_demote defaults to False → snapshot.auto_demoted stays False
    # even though the budget is being violated.
    assert snap.auto_demoted is False


def test_auto_demote_flag_on_marks_violator_demoted():
    m = GateMetrics(auto_demote=True)
    for _ in range(32):
        m.record("slow", CostClass.L0_MEMORY, _result(True, 100.0), budget_ms=10.0)
    snap = m.snapshot()[0]
    assert snap.auto_demoted is True


def test_set_auto_demote_toggles_runtime():
    m = GateMetrics(auto_demote=False)
    m.set_auto_demote(True)
    assert m.auto_demote is True
    for _ in range(32):
        m.record("slow", CostClass.L0_MEMORY, _result(True, 100.0), budget_ms=10.0)
    assert m.snapshot()[0].auto_demoted is True


# ---------- singleton accessor ----------


def test_get_gate_metrics_returns_singleton():
    a = get_gate_metrics()
    b = get_gate_metrics()
    assert a is b


def test_snapshot_returns_dataclass_instances():
    m = GateMetrics()
    m.record("g", CostClass.L0_MEMORY, _result(True, 1.0))
    snap = m.snapshot()[0]
    assert isinstance(snap, GateMetricsSnapshot)


# ---------- record() must never raise on the hot path ----------


def test_record_with_zero_latency_does_not_crash():
    m = GateMetrics()
    m.record("g", CostClass.L0_MEMORY, _result(True, 0.0))
    snap = m.snapshot()[0]
    assert snap.runs == 1
    assert snap.max_ms == 0.0


def test_record_with_identical_samples_handles_quantile_edge_case():
    """statistics.quantiles raises if every sample is identical and we
    don't guard.  Make sure we degrade gracefully."""
    m = GateMetrics()
    for _ in range(50):
        m.record("flat", CostClass.L0_MEMORY, _result(True, 1.0))
    snap = m.snapshot()[0]
    assert snap.p50_ms == pytest.approx(1.0)
    assert snap.p95_ms == pytest.approx(1.0)
    assert snap.p99_ms == pytest.approx(1.0)
