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
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Union

from utils.logger import get_logger

logger = get_logger(__name__)


# Per-gate rolling-window size for percentile calculation.  256 samples
# is enough to produce a stable p99 estimate (~3 samples in the tail)
# while keeping the per-gate memory at ~2KB for a deque of floats.
_DEFAULT_WINDOW_SIZE = 256

# How many recent samples to consider when evaluating budget violation.
# Smaller than the percentile window: we want to react quickly to a
# regression rather than wait for the full 256 samples to roll over.
_BUDGET_VIOLATION_LOOKBACK = 32

# Budget-violation alert rate-limit.  We don't want one per cycle for a
# gate that's persistently slow — once per 5 minutes per gate is plenty.
_BUDGET_VIOLATION_LOG_INTERVAL_SECONDS = 300.0

# Multiple of declared budget at which p95 triggers a violation log.
# 2x leaves room for the normal noise / cold-cache outliers a gate's
# author would already have accounted for in their declared budget.
_BUDGET_VIOLATION_RATIO = 2.0


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


@dataclass(slots=True)
class GateMetricsSnapshot:
    """Point-in-time stats for one gate.

    Returned by :meth:`GateMetrics.snapshot`.  Percentiles are computed
    from the rolling window of the most-recent ``window_size`` latency
    samples — older samples have aged out.  ``runs`` / ``rejects`` /
    ``budget_violations`` are lifetime counters since process start or
    last :meth:`GateMetrics.reset`.
    """

    name: str
    cost_class: str
    runs: int
    rejects: int
    reject_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    budget_ms: float | None
    budget_violations: int
    auto_demoted: bool


@dataclass(slots=True)
class _GateAggregator:
    """Per-gate aggregation state.  Internal to :class:`GateMetrics`."""

    name: str
    cost_class: CostClass
    budget_ms: float | None
    latencies: deque[float]
    runs: int = 0
    rejects: int = 0
    max_ms: float = 0.0
    budget_violations: int = 0
    auto_demoted: bool = False
    last_violation_log_mono: float = 0.0


class GateMetrics:
    """In-memory per-gate latency + rejection aggregator.

    One singleton per orchestrator process.  ``record()`` is called by
    :class:`GatePipeline` after every gate execution and updates the
    aggregator in O(1) (deque append + scalar increments + one constant-
    time monotonic-clock read).  ``snapshot()`` computes percentiles
    from the rolling window on demand — callers pay the percentile cost,
    not the hot path.

    Thread-safety
    -------------
    The backing dict and per-gate deque are updated from a single
    asyncio loop in the trader orchestrator process, so no lock is
    needed.  If we ever go multi-threaded, the dict mutation in
    ``_get_or_create`` and the deque append in ``record`` become
    hazardous — add a :class:`threading.Lock` at that point.  Reading
    via ``snapshot()`` from a different thread is best-effort: it may
    see slightly-stale data but cannot corrupt structures because each
    aggregator is read field-by-field.

    Memory bound
    ------------
    ``window_size`` per gate, each a float (~28B on CPython).  At 256
    samples per gate, a gate uses ~7KB.  Even 1000 distinct gates would
    fit in under 10MB.

    Auto-demotion
    -------------
    ``auto_demote`` is opt-in via constructor flag.  When ``True`` and
    a gate's observed p95 (over the last
    :data:`_BUDGET_VIOLATION_LOOKBACK` samples) exceeds its declared
    ``latency_budget_ms`` by :data:`_BUDGET_VIOLATION_RATIO`x, the gate
    is marked ``auto_demoted=True`` in subsequent snapshots and a
    structured warning is logged (rate-limited to once per gate per
    :data:`_BUDGET_VIOLATION_LOG_INTERVAL_SECONDS`).  Phase 4 LOGS the
    violation but does not actually move the gate to a slower cost
    class — that's a behavioral change the next phase will ship behind
    a separate flag once we trust the detection.
    """

    def __init__(
        self,
        *,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        auto_demote: bool = False,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self._window_size = int(window_size)
        self._auto_demote = bool(auto_demote)
        self._gates: dict[str, _GateAggregator] = {}

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def auto_demote(self) -> bool:
        return self._auto_demote

    def set_auto_demote(self, enabled: bool) -> None:
        """Toggle auto-demote logging at runtime (e.g. from settings)."""
        self._auto_demote = bool(enabled)

    def _get_or_create(self, gate_name: str, cost_class: CostClass, budget_ms: float | None) -> _GateAggregator:
        agg = self._gates.get(gate_name)
        if agg is None:
            agg = _GateAggregator(
                name=gate_name,
                cost_class=cost_class,
                budget_ms=float(budget_ms) if budget_ms is not None else None,
                latencies=deque(maxlen=self._window_size),
            )
            self._gates[gate_name] = agg
        else:
            # Update on every record so re-registering a gate at a
            # different cost class / budget reflects the latest values
            # without losing the rolling window.
            agg.cost_class = cost_class
            if budget_ms is not None:
                agg.budget_ms = float(budget_ms)
        return agg

    def record(
        self,
        gate_name: str,
        cost_class: CostClass,
        result: GateResult,
        *,
        budget_ms: float | None = None,
    ) -> None:
        """Update aggregates for a single gate execution.

        Called from :meth:`GatePipeline.run` and :meth:`run_sync` after
        each gate finishes.  Must stay O(1) on the hot path — no
        allocation beyond the bounded-size deque's internal storage.
        """

        agg = self._get_or_create(gate_name, cost_class, budget_ms)
        latency = float(result.latency_ms or 0.0)
        agg.runs += 1
        if not result.passed:
            agg.rejects += 1
        if latency > agg.max_ms:
            agg.max_ms = latency
        agg.latencies.append(latency)

        if agg.budget_ms is not None and len(agg.latencies) >= _BUDGET_VIOLATION_LOOKBACK:
            recent = list(agg.latencies)[-_BUDGET_VIOLATION_LOOKBACK:]
            recent_p95 = _percentile(recent, 95.0)
            if recent_p95 > agg.budget_ms * _BUDGET_VIOLATION_RATIO:
                agg.budget_violations += 1
                if self._auto_demote:
                    agg.auto_demoted = True
                now_mono = time.monotonic()
                if (
                    now_mono - agg.last_violation_log_mono
                    >= _BUDGET_VIOLATION_LOG_INTERVAL_SECONDS
                ):
                    agg.last_violation_log_mono = now_mono
                    logger.warning(
                        "Gate budget violation",
                        gate=agg.name,
                        declared_class=agg.cost_class.name,
                        declared_budget_ms=agg.budget_ms,
                        observed_p95_ms=round(recent_p95, 3),
                        sample_size=len(recent),
                    )

    def snapshot(self) -> list[GateMetricsSnapshot]:
        """Return current snapshot of all known gates.

        Percentiles are computed lazily — callers pay the cost.  Order
        matches insertion order (first time the gate was recorded).
        """
        out: list[GateMetricsSnapshot] = []
        for agg in self._gates.values():
            latencies = list(agg.latencies)
            p50 = _percentile(latencies, 50.0)
            p95 = _percentile(latencies, 95.0)
            p99 = _percentile(latencies, 99.0)
            reject_rate = (agg.rejects / agg.runs) if agg.runs else 0.0
            out.append(
                GateMetricsSnapshot(
                    name=agg.name,
                    cost_class=agg.cost_class.name,
                    runs=agg.runs,
                    rejects=agg.rejects,
                    reject_rate=round(reject_rate, 4),
                    p50_ms=round(p50, 3),
                    p95_ms=round(p95, 3),
                    p99_ms=round(p99, 3),
                    max_ms=round(agg.max_ms, 3),
                    budget_ms=agg.budget_ms,
                    budget_violations=agg.budget_violations,
                    auto_demoted=agg.auto_demoted,
                )
            )
        return out

    def reset(self, gate_name: str | None = None) -> None:
        """Clear stats.  If ``gate_name`` is given, only that gate is
        cleared.  Otherwise every gate's stats are dropped.  The gate
        registration itself isn't recreated — the next ``record()`` for
        that gate will re-create the aggregator.
        """
        if gate_name is None:
            self._gates.clear()
            return
        self._gates.pop(gate_name, None)


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a list of floats.

    Uses :func:`statistics.quantiles` with ``n=100`` so the result is
    the index ``int(pct) - 1`` of the returned list.  Returns 0.0 for
    fewer than two samples (quantiles requires at least n=2 distinct
    points to interpolate).  For a single sample, the sample itself is
    returned — useful so a gate that has only run once shows that
    latency rather than 0.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    # statistics.quantiles raises if all values are identical and n>1,
    # which can happen in tests with a stubbed-zero latency. Guard.
    try:
        quantiles = statistics.quantiles(values, n=100, method="inclusive")
    except statistics.StatisticsError:
        return float(values[-1])
    idx = max(0, min(98, int(pct) - 1))
    return float(quantiles[idx])


# Module-level singleton.  See :func:`get_gate_metrics`.
_metrics = GateMetrics()


def get_gate_metrics() -> GateMetrics:
    """Return the process-wide :class:`GateMetrics` singleton.

    Provided as a function (not a bare module attribute) so tests can
    monkeypatch the global by patching this getter, and so future
    refactors can swap in a per-orchestrator-instance metrics object
    without breaking every call site.
    """
    return _metrics


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
            try:
                _metrics.record(
                    gate.name,
                    gate.cost_class,
                    result,
                    budget_ms=gate.latency_budget_ms,
                )
            except Exception:
                # Metrics is a side channel — never let a bug here
                # block trading.  Drop the sample and continue.
                pass
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
            try:
                _metrics.record(
                    gate.name,
                    gate.cost_class,
                    result,
                    budget_ms=gate.latency_budget_ms,
                )
            except Exception:
                # See note in run() — never let metrics break trading.
                pass
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
    "GateMetrics",
    "GateMetricsSnapshot",
    "GatePipeline",
    "GatePredicate",
    "GateResult",
    "PipelineRun",
    "get_gate_metrics",
]
