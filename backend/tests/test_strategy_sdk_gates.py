"""Phase 3 tests: StrategySDK gate exposure + orchestrator integration.

The Phase 3 hook lets a user strategy declare custom pre-submit gates by
overriding ``BaseStrategy.get_pre_submit_gates()`` and returning a list
of :class:`Gate` instances.  The orchestrator's ``_build_venue_preflight_pipeline``
folds those gates into the same pipeline as the venue gates, with
:class:`CostClass` ordering ensuring L0 strategy gates short-circuit
*before* the L1 venue gates.

Three contract layers are exercised here:

1. ``BaseStrategy.get_pre_submit_gates()`` default returns ``[]``.
2. Strategy SDK re-exports — ``from services.strategy_sdk import Gate,
   CostClass, GateContext, GateResult`` works at runtime.
3. Orchestrator integration — strategy gates run in cost-class order
   before venue gates, raising strategies don't crash cycles, and
   invalid gate objects are skipped with a warning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.base import BaseStrategy
from services.strategy_sdk import (
    CostClass as SDKCostClass,
    Gate as SDKGate,
    GateContext as SDKGateContext,
    GateResult as SDKGateResult,
)
from services.trader_orchestrator import session_engine as session_engine_module
from services.trader_orchestrator import venue_gates as venue_gates_module
from services.trader_orchestrator.gate_pipeline import (
    CostClass,
    Gate,
    GateContext,
    GateResult,
)


# ---------------------------------------------------------------------
# 1. BaseStrategy default + override behaviour
# ---------------------------------------------------------------------


class _BareStrategy(BaseStrategy):
    """Minimal concrete subclass — does NOT override get_pre_submit_gates."""

    strategy_type = "bare"
    name = "bare"


class _GatedStrategy(BaseStrategy):
    """Subclass that declares two L0 gates."""

    strategy_type = "gated"
    name = "gated"

    def get_pre_submit_gates(self):
        from services.strategy_sdk import Gate, CostClass, GateResult

        def _pass(_ctx: GateContext) -> GateResult:
            return GateResult(passed=True)

        def _reject(_ctx: GateContext) -> GateResult:
            return GateResult(passed=False, reason="gated_test_reject")

        return [
            Gate(name="gated_pass", cost_class=CostClass.L0_MEMORY, predicate=_pass),
            Gate(name="gated_reject", cost_class=CostClass.L0_MEMORY, predicate=_reject),
        ]


def test_base_strategy_default_returns_empty_list():
    s = _BareStrategy()
    assert s.get_pre_submit_gates() == []


def test_strategy_override_returns_gate_list_intact():
    s = _GatedStrategy()
    gates = s.get_pre_submit_gates()
    assert len(gates) == 2
    assert [g.name for g in gates] == ["gated_pass", "gated_reject"]
    assert all(isinstance(g, Gate) for g in gates)
    assert all(g.cost_class == CostClass.L0_MEMORY for g in gates)


# ---------------------------------------------------------------------
# 2. SDK re-exports identity
# ---------------------------------------------------------------------


def test_sdk_reexports_are_the_orchestrator_types():
    """The SDK names must BE the orchestrator types, not copies — otherwise
    isinstance checks in the orchestrator's pipeline-builder would reject
    Gate instances declared by user strategies."""
    assert SDKGate is Gate
    assert SDKGateContext is GateContext
    assert SDKGateResult is GateResult
    assert SDKCostClass is CostClass


def test_sdk_import_path_for_user_strategies():
    """The path documented in the SDK docstring must actually import."""
    from services.strategy_sdk import (  # noqa: F401
        Gate,
        CostClass,
        GateContext,
        GateResult,
    )


# ---------------------------------------------------------------------
# 3. Orchestrator pipeline integration
# ---------------------------------------------------------------------


def _make_pipeline_builder():
    """Replicate the orchestrator's pipeline-builder under test.

    We can't reach into the closure inside ``execute_signal`` directly,
    but the integration logic is small and well-localized.  This local
    builder mirrors the one in session_engine — keeping them in sync
    is the test's job (and Phase 4's per-gate latency telemetry will
    have to update both).
    """
    from services.trader_orchestrator.gate_pipeline import GatePipeline
    from services.trader_orchestrator.venue_gates import (
        GATE_NAME_BUY_COLLATERAL,
        GATE_NAME_MAX_SPREAD_BPS,
        buy_collateral_gate,
        max_spread_bps_gate,
    )
    import logging

    logger = logging.getLogger("test_strategy_sdk_gates")

    def build(strategy):
        pipeline = GatePipeline(
            [
                Gate(
                    name=GATE_NAME_BUY_COLLATERAL,
                    cost_class=CostClass.L1_CACHED,
                    predicate=buy_collateral_gate,
                ),
                Gate(
                    name=GATE_NAME_MAX_SPREAD_BPS,
                    cost_class=CostClass.L1_CACHED,
                    predicate=max_spread_bps_gate,
                ),
            ]
        )
        if strategy is not None:
            try:
                strategy_gates = strategy.get_pre_submit_gates() or []
            except Exception as exc:
                logger.warning(
                    "Strategy %s get_pre_submit_gates raised; skipping",
                    type(strategy).__name__, exc_info=exc,
                )
                strategy_gates = []
            for g in strategy_gates:
                if not isinstance(g, Gate):
                    logger.warning(
                        "Strategy %s yielded non-Gate object %r; skipped",
                        type(strategy).__name__, g,
                    )
                    continue
                pipeline.register(g)
        return pipeline

    return build


def test_strategy_l0_gates_run_before_venue_l1_gates():
    """Cost-class ordering: L0 strategy gates must appear before L1 venue gates."""
    s = _GatedStrategy()
    build = _make_pipeline_builder()
    pipeline = build(s)
    names = [g.name for g in pipeline.gates()]
    # Two L0 strategy gates first, then two L1 venue gates.
    assert names == [
        "gated_pass",
        "gated_reject",
        "buy_collateral",
        "max_spread_bps",
    ]


@pytest.mark.asyncio
async def test_strategy_l0_reject_short_circuits_before_venue():
    """A failing L0 strategy gate stops the pipeline before any L1 venue
    gate runs.  Proves strategy gates pay zero L1 cost on rejection."""
    s = _GatedStrategy()
    build = _make_pipeline_builder()
    pipeline = build(s)
    ctx = GateContext(
        runtime_signal=SimpleNamespace(payload_json={}, source="scanner"),
        decision=None,
        leg={"leg_id": "leg-1", "side": "buy"},
        live_context={},
        risk_limits={},
        strategy_params={},
        mode="live",
        extras={"signal_payload": {}},
    )
    run = await pipeline.run(ctx)
    assert run.passed is False
    assert run.short_circuited_at == "gated_reject"
    # The L1 venue gates must NOT appear in per_gate.
    assert [name for name, _ in run.per_gate] == ["gated_pass", "gated_reject"]


def test_strategy_with_non_gate_objects_skipped_with_warning(caplog):
    """Invalid gate objects in the strategy's list are skipped without
    crashing the pipeline build."""

    class _BadStrategy(BaseStrategy):
        strategy_type = "bad"
        name = "bad"

        def get_pre_submit_gates(self):
            return [
                "not_a_gate",
                42,
                Gate(
                    name="bad_only_real_one",
                    cost_class=CostClass.L0_MEMORY,
                    predicate=lambda _ctx: GateResult(passed=True),
                ),
            ]

    build = _make_pipeline_builder()
    with caplog.at_level("WARNING"):
        pipeline = build(_BadStrategy())
    names = [g.name for g in pipeline.gates()]
    assert "bad_only_real_one" in names
    # The two invalid entries were dropped.
    assert "not_a_gate" not in names
    assert 42 not in names
    # And the venue gates are still present.
    assert "buy_collateral" in names
    assert "max_spread_bps" in names
    # Warnings were emitted for the two bad entries.
    warning_messages = [
        rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"
    ]
    assert sum("non-Gate object" in m for m in warning_messages) == 2


def test_strategy_get_pre_submit_gates_raising_falls_back_to_venue_only(caplog):
    """Strategy errors during gate collection must not crash — the
    pipeline degrades to venue-only gates."""

    class _ExplodingStrategy(BaseStrategy):
        strategy_type = "explode"
        name = "explode"

        def get_pre_submit_gates(self):
            raise RuntimeError("boom")

    build = _make_pipeline_builder()
    with caplog.at_level("WARNING"):
        pipeline = build(_ExplodingStrategy())
    names = [g.name for g in pipeline.gates()]
    assert names == ["buy_collateral", "max_spread_bps"]
    warning_messages = [
        rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"
    ]
    assert any("get_pre_submit_gates raised" in m for m in warning_messages)


# ---------------------------------------------------------------------
# 4. End-to-end via the real session-engine call site
# ---------------------------------------------------------------------


def _leg(leg_id: str, token_id: str) -> dict:
    return {
        "leg_id": leg_id,
        "market_id": "market-e2e",
        "market_question": "Will the strategy gate fire?",
        "token_id": token_id,
        "side": "buy",
        "outcome": "yes",
        "requested_notional_usd": 10.0,
        "requested_shares": 20.0,
        "limit_price": 0.5,
        "price_policy": "taker_limit",
        "time_in_force": "IOC",
        "post_only": False,
    }


def _signal_for_e2e() -> SimpleNamespace:
    return SimpleNamespace(
        id="signal-e2e",
        source="scanner",
        trace_id="trace-e2e",
        strategy_type="generic_strategy",
        strategy_context_json={},
        payload_json={
            "selected_token_id": "11111111111111111111",
            "live_market": {"selected_token_id": "11111111111111111111"},
            "should_reject": True,
        },
        market_id="market-e2e",
        market_question="Will the strategy gate fire?",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=5.0,
        confidence=0.7,
    )


class _RecordingDb:
    def __init__(self) -> None:
        self.pending: list[object] = []
        self.persisted_rows_by_type: dict[str, list[object]] = {}
        self.commit_snapshots: list[list[tuple[str, str]]] = []

    def add(self, row: object) -> None:
        self.pending.append(row)

    async def flush(self) -> None:
        for row in self.pending:
            self.persisted_rows_by_type.setdefault(
                row.__class__.__name__, []
            ).append(row)
        self.pending.clear()

    async def commit(self) -> None:
        trader_orders = self.persisted_rows_by_type.get("TraderOrder") or []
        self.commit_snapshots.append(
            [
                (str(getattr(r, "id", "")), str(getattr(r, "status", "")))
                for r in trader_orders
            ]
        )

    @property
    def pre_submit_commit_count(self) -> int:
        return sum(
            1
            for snap in self.commit_snapshots
            if any(status == "placing" for _id, status in snap)
        )


class _PayloadFlagStrategy(BaseStrategy):
    """Strategy whose L0 gate rejects when the signal payload carries
    ``should_reject=True`` — independent of the venue gates."""

    strategy_type = "payload_flag"
    name = "payload_flag"

    def get_pre_submit_gates(self):
        from services.strategy_sdk import Gate, CostClass, GateResult

        def predicate(ctx: GateContext) -> GateResult:
            payload = (ctx.extras or {}).get("signal_payload") or {}
            if payload.get("should_reject"):
                return GateResult(
                    passed=False,
                    reason="payload_flag_reject",
                    error_message="payload.should_reject is true",
                )
            return GateResult(passed=True)

        return [
            Gate(
                name="payload_flag_reject",
                cost_class=CostClass.L0_MEMORY,
                predicate=predicate,
            ),
        ]


@pytest.mark.asyncio
async def test_strategy_gate_short_circuits_inside_session_engine(monkeypatch):
    """End-to-end: a strategy L0 gate that rejects skips both venue gates
    AND the placeholder DB commit.  This is the whole reason Phase 3
    exists — strategy-side rejections avoid the DB-write cost."""
    db = _RecordingDb()
    engine = session_engine_module.ExecutionSessionEngine(db)
    legs = [_leg("leg-e2e-1", "11111111111111111111")]
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    plan = {"policy": "SINGLE_LEG", "plan_id": "plan-e2e", "metadata": {}}
    monkeypatch.setattr(engine, "_build_plan", lambda *a, **k: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _p: False)
    monkeypatch.setattr(
        session_engine_module, "execution_waves", lambda _p, leg_rows: [leg_rows]
    )
    monkeypatch.setattr(
        session_engine_module, "requires_pair_lock", lambda *a, **k: False
    )
    monkeypatch.setattr(
        session_engine_module,
        "set_trade_signal_status",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        session_engine_module,
        "sync_trader_position_inventory",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        session_engine_module.event_bus, "publish", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        engine, "_publish_hot_signal_status", AsyncMock(return_value=None)
    )
    from services import intent_runtime as intent_runtime_module

    monkeypatch.setattr(
        intent_runtime_module,
        "get_intent_runtime",
        lambda: SimpleNamespace(update_signal_status=AsyncMock()),
    )
    submit_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(session_engine_module, "submit_execution_wave", submit_mock)

    # Stub venue-side IO — these must NEVER be reached when the L0
    # strategy gate rejects.
    venue_buy_gate = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "check_buy_pre_submit_gate",
        venue_buy_gate,
    )
    book_resolver = AsyncMock(
        return_value=(None, [], None, "stub_no_book", None)
    )
    monkeypatch.setattr(
        session_engine_module, "_resolve_shadow_book_and_tape", book_resolver
    )
    monkeypatch.setattr(
        venue_gates_module, "_resolve_shadow_book_and_tape", book_resolver
    )

    # Inject our strategy.
    monkeypatch.setattr(
        session_engine_module,
        "_strategy_instance_for_execution",
        lambda _key: _PayloadFlagStrategy(),
    )

    result = await engine.execute_signal(
        trader_id="trader-e2e",
        signal=_signal_for_e2e(),
        decision_id="decision-e2e",
        strategy_key="payload_flag",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=10.0,
        reason="strategy-gate-test",
    )

    # The single leg was rejected → submit_execution_wave was called
    # with NO submittable legs (or not at all).  Either way, no
    # placeholder commit happened.
    assert db.pre_submit_commit_count == 0
    # Strategy gate ran; venue gates didn't have a chance.
    venue_buy_gate.assert_not_awaited()
    # Telemetry confirms the pre-rejection was counted.
    timing = (result.payload or {}).get("execution_timing_ms") or {}
    assert timing.get("preflight_legs_pre_rejected", 0) >= 1
