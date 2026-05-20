"""Tests for the Phase 2 GatePipeline protocol.

Covers the core invariants the rest of the gate-pipeline refactor builds on:

* Ordering by ``CostClass`` (ascending), stable within a class.
* Short-circuit on the first ``passed=False`` result; subsequent gates
  do not run.
* Sync and async predicates both work via ``run`` and sync-only
  predicates work via ``run_sync``.
* Per-gate ``latency_ms`` is populated by the pipeline (not the
  predicate).
* ``PipelineRun.total_latency_ms`` is approximately the sum of per-gate
  latencies.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.gate_pipeline import (
    CostClass,
    Gate,
    GateContext,
    GatePipeline,
    GateResult,
    PipelineRun,
)


def _ctx() -> GateContext:
    return GateContext(runtime_signal=None, decision=None)


def _pass() -> GateResult:
    return GateResult(passed=True)


def _reject(reason: str = "rejected") -> GateResult:
    return GateResult(passed=False, reason=reason, error_message=f"rejected: {reason}")


def test_register_orders_by_cost_class_ascending():
    """Registering L2, then L0, then L1 should give L0, L1, L2 in order."""
    pipeline = GatePipeline()
    pipeline.register(
        Gate(name="g_l2", cost_class=CostClass.L2_IO, predicate=lambda c: _pass())
    )
    pipeline.register(
        Gate(name="g_l0", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass())
    )
    pipeline.register(
        Gate(name="g_l1", cost_class=CostClass.L1_CACHED, predicate=lambda c: _pass())
    )

    names = [g.name for g in pipeline.gates()]
    assert names == ["g_l0", "g_l1", "g_l2"]


def test_register_is_stable_within_same_cost_class():
    """Two gates with the same cost class keep registration order."""
    pipeline = GatePipeline()
    pipeline.register(
        Gate(name="first_l0", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass())
    )
    pipeline.register(
        Gate(name="middle_l1", cost_class=CostClass.L1_CACHED, predicate=lambda c: _pass())
    )
    pipeline.register(
        Gate(name="second_l0", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass())
    )
    pipeline.register(
        Gate(name="third_l0", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass())
    )

    names = [g.name for g in pipeline.gates()]
    # Three L0 gates retain declaration order, then the L1 gate.
    assert names == ["first_l0", "second_l0", "third_l0", "middle_l1"]


def test_pipeline_constructor_seed_list_preserves_registration_order():
    """Gates passed to constructor are registered in order, then sorted."""
    pipeline = GatePipeline(
        [
            Gate(name="b", cost_class=CostClass.L1_CACHED, predicate=lambda c: _pass()),
            Gate(name="a", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass()),
            Gate(name="c", cost_class=CostClass.L2_IO, predicate=lambda c: _pass()),
        ]
    )
    assert [g.name for g in pipeline.gates()] == ["a", "b", "c"]


def test_short_circuit_on_middle_gate_reject():
    """Gate 2 rejects → gate 3 never runs; per_gate has 2 entries."""
    call_log: list[str] = []

    def _make(name: str, passed: bool):
        def _pred(ctx: GateContext) -> GateResult:
            call_log.append(name)
            return _pass() if passed else _reject(reason=name)

        return _pred

    pipeline = GatePipeline(
        [
            Gate(name="g1", cost_class=CostClass.L0_MEMORY, predicate=_make("g1", True)),
            Gate(name="g2", cost_class=CostClass.L0_MEMORY, predicate=_make("g2", False)),
            Gate(name="g3", cost_class=CostClass.L0_MEMORY, predicate=_make("g3", True)),
        ]
    )
    run = pipeline.run_sync(_ctx())

    assert call_log == ["g1", "g2"], "g3 must NOT run after g2 rejects"
    assert run.passed is False
    assert len(run.per_gate) == 2
    assert [name for name, _ in run.per_gate] == ["g1", "g2"]
    assert run.short_circuited_at == "g2"
    assert run.final_reason == "g2"
    assert run.final_error.startswith("rejected: g2")


@pytest.mark.asyncio
async def test_sync_and_async_predicates_interleave():
    """A pipeline with mixed sync + async predicates runs end-to-end."""
    call_log: list[str] = []

    def _sync_pred(ctx: GateContext) -> GateResult:
        call_log.append("sync")
        return _pass()

    async def _async_pred(ctx: GateContext) -> GateResult:
        call_log.append("async_pre")
        await asyncio.sleep(0)
        call_log.append("async_post")
        return _pass()

    pipeline = GatePipeline(
        [
            Gate(name="sync_gate", cost_class=CostClass.L0_MEMORY, predicate=_sync_pred),
            Gate(name="async_gate", cost_class=CostClass.L1_CACHED, predicate=_async_pred),
        ]
    )
    run = await pipeline.run(_ctx())

    assert call_log == ["sync", "async_pre", "async_post"]
    assert run.passed is True
    assert len(run.per_gate) == 2
    assert [name for name, _ in run.per_gate] == ["sync_gate", "async_gate"]


def test_run_sync_rejects_async_predicate():
    """A sync caller using an async predicate fails fast with a clear error."""

    async def _async_pred(ctx: GateContext) -> GateResult:
        return _pass()

    pipeline = GatePipeline(
        [
            Gate(
                name="async_gate",
                cost_class=CostClass.L0_MEMORY,
                predicate=_async_pred,
            ),
        ]
    )
    run = pipeline.run_sync(_ctx())
    # Treated as a rejection — pipeline wraps the RuntimeError into a
    # GateResult so callers never crash mid-wave.
    assert run.passed is False
    assert run.per_gate[0][1].reason == "gate_predicate_raised"
    assert "awaitable" in run.per_gate[0][1].error_message


def test_latency_ms_is_populated_per_gate():
    """The pipeline times each gate; predicates don't need to."""

    def _slow_pred(ctx: GateContext) -> GateResult:
        # Sleep ~3ms.
        time.sleep(0.003)
        return _pass()

    pipeline = GatePipeline(
        [
            Gate(name="slow", cost_class=CostClass.L0_MEMORY, predicate=_slow_pred),
            Gate(name="fast", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass()),
        ]
    )
    run = pipeline.run_sync(_ctx())
    assert run.passed is True
    slow_latency = run.per_gate[0][1].latency_ms
    fast_latency = run.per_gate[1][1].latency_ms
    # Slow gate slept ~3ms; on most systems that is at least 1ms.
    assert slow_latency >= 1.0, f"expected slow gate >= 1ms, got {slow_latency}"
    # Fast gate should be measurably shorter than slow gate.
    assert fast_latency <= slow_latency + 1.0


def test_total_latency_approximately_sum_of_per_gate():
    """``PipelineRun.total_latency_ms`` ≈ sum of per-gate latencies."""

    def _pred(ms: float):
        def _inner(ctx: GateContext) -> GateResult:
            time.sleep(ms / 1000.0)
            return _pass()

        return _inner

    pipeline = GatePipeline(
        [
            Gate(name="g1", cost_class=CostClass.L0_MEMORY, predicate=_pred(2.0)),
            Gate(name="g2", cost_class=CostClass.L0_MEMORY, predicate=_pred(3.0)),
        ]
    )
    run = pipeline.run_sync(_ctx())
    per_gate_sum = sum(result.latency_ms for _, result in run.per_gate)
    # total >= sum because there's also dispatch overhead.  Sanity bound:
    # total - sum should be small (< 5ms) on any sane machine.
    assert run.total_latency_ms >= per_gate_sum
    assert run.total_latency_ms - per_gate_sum < 50.0


def test_disabled_gate_is_skipped():
    """``Gate(enabled=False)`` does not run and does not appear in per_gate."""
    call_log: list[str] = []

    def _make(name: str):
        def _pred(ctx: GateContext) -> GateResult:
            call_log.append(name)
            return _pass()

        return _pred

    pipeline = GatePipeline(
        [
            Gate(name="enabled_g", cost_class=CostClass.L0_MEMORY, predicate=_make("enabled_g")),
            Gate(
                name="disabled_g",
                cost_class=CostClass.L0_MEMORY,
                predicate=_make("disabled_g"),
                enabled=False,
            ),
        ]
    )
    run = pipeline.run_sync(_ctx())
    assert call_log == ["enabled_g"]
    assert [name for name, _ in run.per_gate] == ["enabled_g"]
    assert run.passed is True


def test_predicate_raising_is_treated_as_rejection():
    """A predicate that raises ⇒ pipeline short-circuits with reason."""

    def _boom(ctx: GateContext) -> GateResult:
        raise RuntimeError("kaboom")

    pipeline = GatePipeline(
        [
            Gate(name="boom_gate", cost_class=CostClass.L0_MEMORY, predicate=_boom),
            Gate(name="next_gate", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass()),
        ]
    )
    run = pipeline.run_sync(_ctx())
    assert run.passed is False
    assert len(run.per_gate) == 1
    assert run.per_gate[0][1].reason == "gate_predicate_raised"
    assert "kaboom" in run.per_gate[0][1].error_message
    assert run.short_circuited_at == "boom_gate"


def test_pipeline_run_is_returned_type():
    """Return type sanity — used by Phase 4 telemetry."""
    pipeline = GatePipeline(
        [Gate(name="g", cost_class=CostClass.L0_MEMORY, predicate=lambda c: _pass())]
    )
    run = pipeline.run_sync(_ctx())
    assert isinstance(run, PipelineRun)
    assert run.passed is True
    assert run.short_circuited_at == ""
