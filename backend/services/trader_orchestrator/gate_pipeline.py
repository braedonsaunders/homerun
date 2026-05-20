"""Phase 2 of the gate-pipeline refactor: formal Gate + Pipeline protocol.

Every gate in the orchestrator (platform, venue, and — in Phase 3 —
strategy SDK) is represented as a first-class :class:`Gate` instance with
a declared :class:`CostClass`.  The :class:`GatePipeline` runs gates in
ascending cost-class order (stable within a class) and short-circuits on
the first ``passed=False`` result.

Why this matters
----------------
Before Phase 1, every signal paid the 5-13s placeholder DB write cost
even when the venue's collateral or spread gate would reject it post-DB.
Phase 1 hoisted those checks pre-DB but kept them as inline code in
``session_engine.pre_db_venue_preflight``.  Phase 2 generalizes the same
pattern: every gate declares its cost class, and the pipeline guarantees
the cheap gates run first.  Phase 4 will hook per-gate latency telemetry
into the orchestrator's slow-log.

No external deps — pure stdlib so the module sits cleanly inside the
trader_orchestrator package with no import cycles.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Union

from utils.logger import get_logger

logger = get_logger(__name__)


class CostClass(IntEnum):
    """Latency-budget tiers.  Gates run in ascending order.

    The numeric value is intentionally a sort key — :class:`GatePipeline`
    relies on ``int(cost_class)`` for ordering.  Keep ascending: cheapest
    first.
    """

    L0_MEMORY = 0  # < 5ms; pure in-memory, no IO
    L1_CACHED = 1  # < 50ms; cached value with TTL refresh
    L2_IO = 2  # < 500ms; network or DB read
    L3_COMMIT = 3  # seconds; DB writes / venue submit (not a real gate)


@dataclass(slots=True)
class GateContext:
    """Read-only context passed to every gate predicate.

    Callers may attach arbitrary keys via :attr:`extras`.  Predicates that
    need a specific key should declare it in :attr:`Gate.requires` so the
    pipeline can validate the context up front (future work — Phase 2
    keeps validation soft so existing call sites don't break).
    """

    runtime_signal: Any
    decision: Any
    leg: dict[str, Any] | None = None
    live_context: Any | None = None
    risk_limits: dict[str, Any] = field(default_factory=dict)
    strategy_params: dict[str, Any] = field(default_factory=dict)
    mode: str = "live"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateResult:
    """Result of running a single gate predicate.

    ``detail`` is the place to put gate-specific structured info
    (e.g. ``{"spread_bps": 1665.7, "cap": 75.0}``).  The pipeline does
    not interpret it.
    """

    passed: bool
    reason: str = ""
    error_message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


# A gate predicate.  Sync or async.  Receives :class:`GateContext`,
# returns :class:`GateResult`.
GatePredicate = Callable[[GateContext], Union[GateResult, Awaitable[GateResult]]]


@dataclass(slots=True)
class Gate:
    """Declarative gate definition registered with :class:`GatePipeline`.

    Parameters
    ----------
    name
        Stable identifier — used in telemetry and short-circuit tracking.
        Should match the string the legacy ``platform_gates`` list used
        when migrating an existing gate so log queries continue to work.
    cost_class
        Cost-class tier.  Gates run in ascending order.
    predicate
        Sync or async callable that receives a :class:`GateContext` and
        returns a :class:`GateResult`.
    requires
        Optional tuple of ``GateContext.extras`` keys this gate consumes.
        Phase 2 stores but does not enforce — Phase 4 will validate.
    enabled
        Disabled gates are skipped (don't appear in
        :attr:`PipelineRun.per_gate`).  Use this for feature-flagging
        rather than mutating the registered list.
    latency_budget_ms
        Optional declared upper bound on latency.  Phase 4 may auto-demote
        gates that repeatedly exceed budget to a slower cost class.
    """

    name: str
    cost_class: CostClass
    predicate: GatePredicate
    requires: tuple[str, ...] = ()
    enabled: bool = True
    latency_budget_ms: float | None = None


@dataclass(slots=True)
class PipelineRun:
    """Result of running one :class:`GateContext` through a pipeline.

    ``per_gate`` is the ordered list of ``(gate_name, GateResult)``
    tuples for every gate that *ran*.  A short-circuited rejection
    means the final entry is the rejecting gate; gates further down
    the list never ran and are absent.
    """

    passed: bool
    final_reason: str = ""
    final_error: str = ""
    short_circuited_at: str = ""
    per_gate: list[tuple[str, GateResult]] = field(default_factory=list)
    total_latency_ms: float = 0.0


# Internal record holding the gate plus its registration order so the
# pipeline can sort stably within a cost class.
@dataclass(slots=True)
class _RegisteredGate:
    gate: Gate
    registration_index: int


class GatePipeline:
    """Holds an ordered list of gates and runs them with telemetry.

    Ordering is by ``(cost_class, registration_index)`` so gates declared
    earlier within the same cost class run first.  Adding a new gate via
    :meth:`register` does not perturb the relative order of existing
    gates within a different cost class.

    The pipeline is intentionally simple: no DAG, no parallel groups, no
    retries.  Those belong at the caller layer — Phase 3 (StrategySDK)
    composes pipelines; Phase 4 adds the observability hooks.
    """

    def __init__(self, gates: list[Gate] | None = None) -> None:
        self._gates: list[_RegisteredGate] = []
        self._next_index: int = 0
        if gates:
            for g in gates:
                self.register(g)

    def register(self, gate: Gate) -> None:
        """Insert ``gate`` maintaining ``cost_class`` ascending order.

        Stable within a cost class: two gates with the same cost class
        retain their registration order.  Uses insertion-sort because
        the per-pipeline gate count is small (~5-15) — the O(n) cost is
        irrelevant compared to a single gate's evaluation time.
        """

        record = _RegisteredGate(gate=gate, registration_index=self._next_index)
        self._next_index += 1
        # Find the first existing slot whose cost_class is strictly
        # greater — insert there to preserve stable order within a class.
        insert_at = len(self._gates)
        new_key = int(gate.cost_class)
        for i, existing in enumerate(self._gates):
            if int(existing.gate.cost_class) > new_key:
                insert_at = i
                break
        self._gates.insert(insert_at, record)

    def gates(self) -> list[Gate]:
        """Return the registered gates in pipeline-run order."""
        return [rec.gate for rec in self._gates]

    async def run(self, ctx: GateContext) -> PipelineRun:
        """Run all enabled gates in order; short-circuit on first reject.

        Each gate is timed and its latency_ms populated on the
        :class:`GateResult` the pipeline records.  Sync predicates run
        directly; async predicates are awaited.  Predicates that raise
        are treated as a rejection — the exception message becomes the
        gate's ``error_message`` and the pipeline short-circuits.

        Telemetry: one structured log line on rejection (gate name,
        cost class, latency).  Passes are silent — too noisy at scale.
        """

        run_start = time.monotonic()
        per_gate: list[tuple[str, GateResult]] = []
        final_reason = ""
        final_error = ""
        short_circuited_at = ""
        passed_overall = True

        for record in self._gates:
            gate = record.gate
            if not gate.enabled:
                continue
            gate_start = time.monotonic()
            try:
                raw = gate.predicate(ctx)
                if inspect.isawaitable(raw):
                    result = await raw
                else:
                    result = raw
                if not isinstance(result, GateResult):
                    # Defensive — a predicate that returns the wrong type
                    # is a programming error.  Surface as a rejection so
                    # we don't crash the whole pipeline mid-wave.
                    result = GateResult(
                        passed=False,
                        reason="gate_predicate_returned_invalid_type",
                        error_message=(
                            f"Gate {gate.name!r} predicate returned "
                            f"{type(raw).__name__!r}, expected GateResult"
                        ),
                    )
            except Exception as exc:
                result = GateResult(
                    passed=False,
                    reason="gate_predicate_raised",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            elapsed_ms = max(0.0, (time.monotonic() - gate_start) * 1000.0)
            result.latency_ms = round(elapsed_ms, 3)
            per_gate.append((gate.name, result))
            if not result.passed:
                passed_overall = False
                final_reason = result.reason or ""
                final_error = result.error_message or ""
                short_circuited_at = gate.name
                logger.info(
                    "gate_pipeline reject gate=%s cost_class=%s latency_ms=%.3f reason=%s",
                    gate.name,
                    gate.cost_class.name,
                    result.latency_ms,
                    result.reason or "<unspecified>",
                )
                break

        total_ms = max(0.0, (time.monotonic() - run_start) * 1000.0)
        return PipelineRun(
            passed=passed_overall,
            final_reason=final_reason,
            final_error=final_error,
            short_circuited_at=short_circuited_at,
            per_gate=per_gate,
            total_latency_ms=round(total_ms, 3),
        )

    def run_sync(self, ctx: GateContext) -> PipelineRun:
        """Synchronous variant of :meth:`run` for sync callers.

        Asserts every predicate returns a non-awaitable.  Async predicates
        registered against a pipeline used via ``run_sync`` will raise a
        :class:`RuntimeError`.  Use this from sync callers like
        ``apply_platform_decision_gates`` which cannot await without
        breaking their interface.
        """

        run_start = time.monotonic()
        per_gate: list[tuple[str, GateResult]] = []
        final_reason = ""
        final_error = ""
        short_circuited_at = ""
        passed_overall = True

        for record in self._gates:
            gate = record.gate
            if not gate.enabled:
                continue
            gate_start = time.monotonic()
            try:
                raw = gate.predicate(ctx)
                if inspect.isawaitable(raw):
                    raise RuntimeError(
                        f"Gate {gate.name!r} predicate returned an awaitable; "
                        "use pipeline.run() (async) for async predicates"
                    )
                result = raw
                if not isinstance(result, GateResult):
                    result = GateResult(
                        passed=False,
                        reason="gate_predicate_returned_invalid_type",
                        error_message=(
                            f"Gate {gate.name!r} predicate returned "
                            f"{type(raw).__name__!r}, expected GateResult"
                        ),
                    )
            except Exception as exc:
                result = GateResult(
                    passed=False,
                    reason="gate_predicate_raised",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            elapsed_ms = max(0.0, (time.monotonic() - gate_start) * 1000.0)
            result.latency_ms = round(elapsed_ms, 3)
            per_gate.append((gate.name, result))
            if not result.passed:
                passed_overall = False
                final_reason = result.reason or ""
                final_error = result.error_message or ""
                short_circuited_at = gate.name
                logger.info(
                    "gate_pipeline reject gate=%s cost_class=%s latency_ms=%.3f reason=%s",
                    gate.name,
                    gate.cost_class.name,
                    result.latency_ms,
                    result.reason or "<unspecified>",
                )
                break

        total_ms = max(0.0, (time.monotonic() - run_start) * 1000.0)
        return PipelineRun(
            passed=passed_overall,
            final_reason=final_reason,
            final_error=final_error,
            short_circuited_at=short_circuited_at,
            per_gate=per_gate,
            total_latency_ms=round(total_ms, 3),
        )


__all__ = [
    "CostClass",
    "Gate",
    "GateContext",
    "GatePipeline",
    "GatePredicate",
    "GateResult",
    "PipelineRun",
]
