import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import (  # noqa: E402
    Base,
    TradeSignal,
    Trader,
    TraderDecision,
    TraderEvent,
    TraderOrder,
    TraderSignalConsumption,
)
from services.strategies.base import StrategyDecision  # noqa: E402
import services.trader_hot_state as hot_state  # noqa: E402
from services.trader_orchestrator import fast_submit  # noqa: E402
from services.trader_orchestrator.order_manager import LegSubmitResult  # noqa: E402
from services.trader_orchestrator_state import list_fast_traders  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402
from workers import fast_trader_runtime  # noqa: E402


def _fast_trader_config() -> dict:
    return {
        "id": "fast-trader",
        "name": "Fast Infrastructure Trader",
        "mode": "live",
        "risk_limits": {"max_trade_notional_usd": 7.5},
        "source_configs": [
            {
                "source_key": "generic-source",
                "strategy_key": "generic-fast-strategy",
                "enabled": True,
                "strategy_params": {"min_score": 1.0},
            }
        ],
        "is_enabled": True,
        "is_paused": False,
    }


def _trade_signal(signal_id: str = "signal-1") -> TradeSignal:
    now = utcnow().replace(tzinfo=None)
    return TradeSignal(
        id=signal_id,
        source="generic-source",
        signal_type="entry",
        strategy_type="generic-fast-strategy",
        market_id=f"market-{signal_id}",
        market_question="Generic fast market?",
        direction="buy_yes",
        entry_price=0.42,
        effective_price=0.42,
        edge_percent=4.2,
        confidence=0.77,
        liquidity=1000.0,
        status="pending",
        payload_json={
            "positions_to_take": [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": 0.42,
                    "market_id": f"market-{signal_id}",
                    "token_id": f"token-{signal_id}",
                }
            ]
        },
        dedupe_key=f"dedupe-{signal_id}",
        runtime_sequence=100,
        created_at=now,
        updated_at=now,
    )


async def _seed_trader_and_signal(session, signal_id: str = "signal-1") -> None:
    now = utcnow().replace(tzinfo=None)
    session.add(
        Trader(
            id="fast-trader",
            name="Fast Infrastructure Trader",
            source_configs_json=_fast_trader_config()["source_configs"],
            risk_limits_json={"max_trade_notional_usd": 7.5},
            metadata_json={},
            mode="live",
            latency_class="fast",
            is_enabled=True,
            is_paused=False,
            interval_seconds=1,
            created_at=now,
            updated_at=now,
        )
    )
    session.add(_trade_signal(signal_id))
    await session.commit()


@pytest.mark.asyncio
async def test_fast_trader_records_skipped_decision_and_consumption(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "fast_runtime_skipped_decision")
    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)

    class SkippingStrategy:
        def evaluate(self, signal, context):
            assert context["fast_tier"] is True
            return StrategyDecision(
                decision="skipped",
                reason="shared fast filters not met",
                score=8.25,
                size_usd=3.5,
                checks=[
                    {
                        "key": "generic_gate",
                        "label": "Generic gate",
                        "passed": False,
                        "score": 8.25,
                        "detail": "Rejected by shared infrastructure test.",
                    }
                ],
            )

    monkeypatch.setattr(fast_trader_runtime.strategy_loader, "get_instance", lambda key: SkippingStrategy())
    runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), asyncio.Event())

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session)
            signal = await session.get(TradeSignal, "signal-1")
            await runner._process_one(session, signal=signal, mode="live", default_size_usd=7.5)
            await session.commit()
            await hot_state.flush_audit_buffer()

        async with session_factory() as session:
            decision = (
                await session.execute(select(TraderDecision).where(TraderDecision.trader_id == "fast-trader"))
            ).scalar_one()
            consumption = (
                await session.execute(
                    select(TraderSignalConsumption).where(TraderSignalConsumption.trader_id == "fast-trader")
                )
            ).scalar_one()

        assert decision.decision == "skipped"
        # Generic "filters not met" reasons get enriched with the specific
        # failed-check detail so the bot trader terminal shows operators
        # which filter rejected the signal, mirroring the slow-tier path.
        assert decision.reason == (
            "shared fast filters not met | failed checks: Generic gate: Rejected by shared infrastructure test."
        )
        assert decision.strategy_key == "generic-fast-strategy"
        assert decision.payload_json["fast_tier"] is True
        assert decision.payload_json["evaluated_size_usd"] == 3.5
        assert decision.checks_summary_json["checks"][0]["key"] == "generic_gate"
        assert consumption.decision_id == decision.id
        assert consumption.outcome == "skipped"
    finally:
        await engine.dispose()


def test_fast_trader_enriches_generic_skip_reason_with_failed_checks():
    # Regression: fast tier bypasses apply_platform_decision_gates for
    # latency, but it must still surface which strategy filter rejected
    # the signal — otherwise the bot trader terminal shows opaque
    # "Crypto worker filters not met" without the specific gate.
    runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), asyncio.Event())

    decision = SimpleNamespace(
        decision="skipped",
        reason="Crypto worker filters not met",
        score=0.5,
        size_usd=10.0,
        checks=[
            SimpleNamespace(key="source", label="Crypto source", passed=True, score=None, detail="ok", payload={}),
            SimpleNamespace(
                key="oracle_freshness",
                label="Oracle readiness",
                passed=False,
                score=12000,
                detail="age=12s > max=5s",
                payload={},
            ),
            SimpleNamespace(
                key="spread",
                label="Maximum spread",
                passed=False,
                score=0.04,
                detail="spread=0.04 > max=0.02",
                payload={},
            ),
        ],
    )

    enriched = runner._enriched_strategy_reason(decision, decision.reason, "skipped")
    assert enriched.startswith("Crypto worker filters not met")
    assert "Oracle readiness: age=12s > max=5s" in enriched
    assert "Maximum spread: spread=0.04 > max=0.02" in enriched


def test_fast_trader_keeps_specific_skip_reason_unchanged():
    # When the strategy already provides a specific reason (not in the
    # generic-token allow list), the enrichment is a no-op: the base
    # reason is descriptive enough on its own.
    runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), asyncio.Event())

    decision = SimpleNamespace(
        decision="skipped",
        reason="Spread too narrow (0.001 < 0.005)",
        score=0.5,
        size_usd=10.0,
        checks=[
            SimpleNamespace(key="spread", label="Maximum spread", passed=False, score=0.001, detail="0.001 < 0.005", payload={}),
        ],
    )

    enriched = runner._enriched_strategy_reason(decision, decision.reason, "skipped")
    assert enriched == "Spread too narrow (0.001 < 0.005)"


@pytest.mark.asyncio
async def test_fast_trader_consumes_signal_from_unconfigured_strategy_without_decision(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "fast_runtime_strategy_filter")
    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)

    def _unexpected_strategy_lookup(key):
        raise AssertionError(f"unexpected strategy lookup for {key}")

    monkeypatch.setattr(fast_trader_runtime.strategy_loader, "get_instance", _unexpected_strategy_lookup)
    runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), asyncio.Event())

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-filtered")
            signal = await session.get(TradeSignal, "signal-filtered")
            signal.strategy_type = "other-fast-strategy"
            await session.commit()

            await runner._process_one(session, signal=signal, mode="live", default_size_usd=7.5)
            await session.commit()
            await hot_state.flush_audit_buffer()

        async with session_factory() as session:
            decision_count = (
                await session.execute(select(func.count()).select_from(TraderDecision))
            ).scalar_one()
            consumption = (
                await session.execute(
                    select(TraderSignalConsumption).where(TraderSignalConsumption.trader_id == "fast-trader")
                )
            ).scalar_one()

        assert decision_count == 0
        assert consumption.decision_id is None
        assert consumption.outcome == "skipped"
        assert str(consumption.reason or "").startswith("source_strategy_filter:")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fast_trader_selected_signal_records_no_order_failure(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "fast_runtime_no_order_decision")
    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)

    class SelectingStrategy:
        def evaluate(self, signal, context):
            return StrategyDecision(decision="selected", reason="selected by shared fast test", score=12.0)

    async def fake_execute_fast_signal(session, **kwargs):
        assert session is not None
        assert kwargs["decision_id"]
        existing_decisions = (
            await session.execute(select(func.count()).select_from(TraderDecision))
        ).scalar_one()
        assert existing_decisions == 0
        return SimpleNamespace(
            status="skipped",
            effective_price=None,
            error_message="pre-submit gate rejected order",
            orders_written=0,
            payload={"reason": "pre_submit_gate"},
            created_orders=[],
        )

    monkeypatch.setattr(fast_trader_runtime.strategy_loader, "get_instance", lambda key: SelectingStrategy())
    monkeypatch.setattr(fast_trader_runtime, "execute_fast_signal", fake_execute_fast_signal)
    runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), asyncio.Event())

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-2")
            signal = await session.get(TradeSignal, "signal-2")
            await runner._process_one(session, signal=signal, mode="live", default_size_usd=7.5)
            await session.commit()
            await hot_state.flush_audit_buffer()

        async with session_factory() as session:
            decision = (
                await session.execute(select(TraderDecision).where(TraderDecision.trader_id == "fast-trader"))
            ).scalar_one()
            event = (
                await session.execute(select(TraderEvent).where(TraderEvent.trader_id == "fast-trader"))
            ).scalar_one()
            consumption = (
                await session.execute(
                    select(TraderSignalConsumption).where(TraderSignalConsumption.trader_id == "fast-trader")
                )
            ).scalar_one()

        assert decision.decision == "skipped"
        assert decision.reason == "pre-submit gate rejected order"
        assert decision.payload_json["submit_result"]["reason"] == "pre_submit_gate"
        assert event.event_type == "fast_submit_no_order"
        assert consumption.decision_id == decision.id
        assert consumption.outcome == "skipped"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fast_trader_idle_cycle_updates_last_run_and_emits_heartbeat(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "fast_runtime_idle_heartbeat")
    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)
    runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), asyncio.Event())
    runner._last_idle_event_at = -1_000_000.0

    try:
        async with session_factory() as session:
            now = utcnow().replace(tzinfo=None)
            session.add(
                Trader(
                    id="fast-trader",
                    name="Fast Infrastructure Trader",
                    source_configs_json=_fast_trader_config()["source_configs"],
                    risk_limits_json={"max_trade_notional_usd": 7.5},
                    metadata_json={},
                    mode="live",
                    latency_class="fast",
                    is_enabled=True,
                    is_paused=False,
                    interval_seconds=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            touched = await runner._touch_trader_run(session, force=True)
            emitted = await runner._maybe_emit_idle_event(
                session,
                accepted_sources=["generic-source"],
                cursor_runtime_sequence=123,
                cursor_created_at=now,
                cursor_signal_id="signal-0",
            )
            await session.commit()
            await hot_state.flush_audit_buffer()

        async with session_factory() as session:
            trader = await session.get(Trader, "fast-trader")
            event = (
                await session.execute(select(TraderEvent).where(TraderEvent.trader_id == "fast-trader"))
            ).scalar_one()

        assert touched is True
        assert emitted is True
        assert trader.last_run_at is not None
        assert event.event_type == "fast_cycle_heartbeat"
        assert event.payload_json["accepted_sources"] == ["generic-source"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fast_runtime_restarts_dead_per_trader_task(monkeypatch):
    runtime = fast_trader_runtime._FastRuntime()
    old_wake = asyncio.Event()
    old_runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), old_wake)
    old_task = asyncio.get_running_loop().create_future()
    old_task.set_result(None)
    runtime._wake_events["fast-trader"] = old_wake
    runtime._task_objs["fast-trader"] = old_runner
    runtime._tasks["fast-trader"] = old_task

    class EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_list_fast_traders(session):
        assert session is not None
        return [_fast_trader_config()]

    created = {"count": 0}

    def fake_create_task(coro, *, name=None):
        assert name == "fast-trader-fast-trader"
        coro.close()
        created["count"] += 1
        return asyncio.get_running_loop().create_future()

    monkeypatch.setattr(fast_trader_runtime, "AsyncSessionLocal", lambda: EmptySession())

    async def fake_control_enabled(session):
        assert session is not None
        return {
            "is_enabled": True,
            "is_paused": False,
            "kill_switch": False,
            "mode": "live",
        }

    monkeypatch.setattr(fast_trader_runtime, "read_orchestrator_control", fake_control_enabled)
    monkeypatch.setattr(fast_trader_runtime, "list_fast_traders", fake_list_fast_traders)
    monkeypatch.setattr(fast_trader_runtime.asyncio, "create_task", fake_create_task)

    await runtime._refresh_roster()

    assert old_runner._stopped is True
    assert created["count"] == 1
    assert runtime._task_objs["fast-trader"] is not old_runner
    assert runtime._tasks["fast-trader"] is not old_task


@pytest.mark.asyncio
async def test_fast_runtime_stops_trader_tasks_when_orchestrator_is_paused(monkeypatch):
    runtime = fast_trader_runtime._FastRuntime()
    old_wake = asyncio.Event()
    old_runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), old_wake)
    old_task = asyncio.get_running_loop().create_future()
    runtime._wake_events["fast-trader"] = old_wake
    runtime._task_objs["fast-trader"] = old_runner
    runtime._tasks["fast-trader"] = old_task

    class EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_list_fast_traders(session):
        raise AssertionError("paused orchestrator must not list fast traders")

    monkeypatch.setattr(fast_trader_runtime, "AsyncSessionLocal", lambda: EmptySession())

    async def fake_control_paused(session):
        assert session is not None
        return {
            "is_enabled": False,
            "is_paused": True,
            "kill_switch": False,
            "mode": "live",
        }

    monkeypatch.setattr(fast_trader_runtime, "read_orchestrator_control", fake_control_paused)
    monkeypatch.setattr(fast_trader_runtime, "list_fast_traders", fake_list_fast_traders)

    await runtime._refresh_roster()

    assert old_runner._stopped is True
    assert old_task.cancelled() is True
    assert "fast-trader" not in runtime._task_objs
    assert "fast-trader" not in runtime._tasks
    assert "fast-trader" not in runtime._wake_events


@pytest.mark.asyncio
async def test_list_fast_traders_excludes_disabled_and_paused_traders():
    engine, session_factory = await build_postgres_session_factory(Base, "fast_runtime_enabled_filter")
    try:
        async with session_factory() as session:
            now = utcnow().replace(tzinfo=None)
            for trader_id, enabled, paused in (
                ("fast-enabled", True, False),
                ("fast-disabled", False, False),
                ("fast-paused", True, True),
            ):
                session.add(
                    Trader(
                        id=trader_id,
                        name=trader_id,
                        source_configs_json=_fast_trader_config()["source_configs"],
                        risk_limits_json={},
                        metadata_json={},
                        mode="live",
                        latency_class="fast",
                        is_enabled=enabled,
                        is_paused=paused,
                        interval_seconds=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
            await session.commit()

            traders = await list_fast_traders(session)

        assert [trader["id"] for trader in traders] == ["fast-enabled"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_fast_signal_pre_submits_skeleton_before_clob(monkeypatch):
    """The skeleton TraderOrder row must exist in the DB by the time the
    CLOB call runs — that's what makes the idempotency lock crash-safe.
    Verified by inspecting the DB from the mock submit_execution_leg, which
    fires while the fast path is mid-submit."""
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_pre_submit_skeleton")

    skeleton_observed = {"row": None}

    async def fake_submit_execution_leg(**_kwargs):
        # The pre-submit row must already be visible to a fresh session at
        # this point — i.e. it was committed before the CLOB call started.
        async with session_factory() as probe_session:
            row = (
                await probe_session.execute(
                    select(TraderOrder).where(TraderOrder.signal_id == "signal-skeleton")
                )
            ).scalar_one_or_none()
            skeleton_observed["row"] = row
        return LegSubmitResult(
            leg_id="leg-1",
            status="executed",
            effective_price=0.42,
            error_message=None,
            payload={"provider_status": "filled"},
            provider_order_id="provider-skel-1",
            provider_clob_order_id="clob-skel-1",
            shares=7.0,
            notional_usd=3.0,
        )

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-skeleton")
            signal = await session.get(TradeSignal, "signal-skeleton")
            result = await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id=None,
                decision_audit=None,
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="skeleton test",
            )
            await session.commit()

        # The skeleton observed mid-CLOB had the in-flight marker.
        assert skeleton_observed["row"] is not None
        assert skeleton_observed["row"].payload_json.get("fast_submission_state") == "in_flight"
        assert skeleton_observed["row"].status == "submitted"

        # After CLOB returned, the same row was UPDATED in place — there's
        # exactly one TraderOrder, with the final state and provider IDs.
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(TraderOrder).where(TraderOrder.signal_id == "signal-skeleton")
                )
            ).scalars().all()

        assert len(rows) == 1
        assert rows[0].id == skeleton_observed["row"].id
        assert rows[0].status == "executed"
        assert rows[0].provider_order_id == "provider-skel-1"
        assert rows[0].payload_json.get("fast_submission_state") == "completed"
        assert result.status == "executed"
        assert result.orders_written == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_fast_signal_keeps_lock_when_clob_raises_after_pre_submit(monkeypatch):
    """If the CLOB call raises after the pre-submit row was committed, the
    row stays so the duplicate-check guard blocks any retry. Marker flips
    to clob_exception so the reconcile sweep knows what to do with it."""
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_clob_exception_lock")

    async def fake_submit_execution_leg(**_kwargs):
        raise RuntimeError("transport error mid-call")

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-clob-raise")
            signal = await session.get(TradeSignal, "signal-clob-raise")
            result = await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id=None,
                decision_audit=None,
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="clob raise test",
            )
            await session.commit()

        async with session_factory() as session:
            row = (
                await session.execute(
                    select(TraderOrder).where(TraderOrder.signal_id == "signal-clob-raise")
                )
            ).scalar_one()

        assert row.status == "failed"
        assert row.payload_json.get("fast_submission_state") == "clob_exception"
        assert "transport error mid-call" in (row.payload_json.get("exception_message") or "")
        assert result.status == "failed"
        assert result.orders_written == 1  # row remains as the lock
        assert result.payload.get("trader_order_id") == row.id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_fast_signal_stamps_deterministic_idempotency_key(monkeypatch):
    """Every fast-tier submission must (1) derive the same key from
    (trader_id, signal_id) on every call, (2) attach it to the leg dict
    under the dedicated ``clob_idempotency_key`` field (NOT overloading
    ``leg["metadata"]``, which now stays cleanly the ExecutionPlan
    bookkeeping dict), so the CLOB layer forwards it as
    ``OrderArgsV2.metadata``, and (3) persist it on
    ``TraderOrder.payload_json["fast_idempotency_key"]`` so the
    orphan-reconcile sweep can match a venue order back to the row even
    after a crash that lost ``provider_clob_order_id``."""
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_idempotency_key")

    captured_legs: list[dict] = []

    async def fake_submit_execution_leg(*, leg, **_kwargs):
        captured_legs.append(dict(leg))
        return LegSubmitResult(
            leg_id="leg-1",
            status="executed",
            effective_price=0.42,
            error_message=None,
            payload={"provider_status": "filled"},
            provider_order_id="provider-1",
            provider_clob_order_id="clob-1",
            shares=7.0,
            notional_usd=3.0,
        )

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-idemp")
            signal = await session.get(TradeSignal, "signal-idemp")
            await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id=None,
                decision_audit=None,
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="idempotency stamp test",
            )
            await session.commit()

        # The key derivation is deterministic — re-derive and compare.
        from services.trader_orchestrator.fast_idempotency import derive_fast_idempotency_key

        expected_key = derive_fast_idempotency_key(trader_id="fast-trader", signal_id="signal-idemp")
        # Sanity: real key shape, not the all-zero default.
        assert expected_key.startswith("0x") and len(expected_key) == 66
        assert expected_key != "0x" + ("0" * 64)

        # (1) Stamped into the leg dict's dedicated CLOB-key field for the
        # venue submission. ``leg["metadata"]`` stays the bookkeeping dict
        # — the two used to share a slot, which produced the
        # "non-hexadecimal number found in fromhex() arg at position 43"
        # production crash; the dedicated field ends that overload.
        assert len(captured_legs) == 1
        assert captured_legs[0].get("clob_idempotency_key") == expected_key
        legacy_metadata = captured_legs[0].get("metadata")
        assert not isinstance(legacy_metadata, str), (
            "leg['metadata'] should not carry the idempotency key under the "
            "new schema; the dedicated clob_idempotency_key field carries it."
        )

        # (2) Persisted onto the TraderOrder row's payload.
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(TraderOrder).where(TraderOrder.signal_id == "signal-idemp")
                )
            ).scalar_one()
        assert row.payload_json.get("fast_idempotency_key") == expected_key
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_fast_signal_records_latency_via_existing_metrics(monkeypatch):
    """Fast path should hook into ``execution_latency_metrics`` — the same
    sink used by the slow tier — so the SLO dashboard sees both paths."""
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_latency_metric")

    recorded: list[dict] = []

    async def fake_record(**kwargs):
        recorded.append(kwargs)

    async def fake_submit_execution_leg(**_kwargs):
        return LegSubmitResult(
            leg_id="leg-1",
            status="executed",
            effective_price=0.42,
            error_message=None,
            payload={"provider_status": "filled"},
            provider_order_id="latency-provider",
            provider_clob_order_id="latency-clob",
            shares=7.0,
            notional_usd=3.0,
        )

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)
    monkeypatch.setattr(fast_submit.execution_latency_metrics, "record", fake_record)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-latency")
            signal = await session.get(TradeSignal, "signal-latency")
            await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id=None,
                decision_audit=None,
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="latency test",
            )
            await session.commit()
    finally:
        await engine.dispose()

    assert len(recorded) == 1
    sample = recorded[0]
    assert sample["trader_id"] == "fast-trader"
    payload = sample["payload"]
    # Stage we can actually measure on the fast path:
    assert payload.get("submit_round_trip_ms") is not None
    assert payload["submit_round_trip_ms"] >= 0


@pytest.mark.asyncio
async def test_execute_fast_signal_links_deferred_decision_to_order(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_deferred_decision_link")

    async def fake_submit_execution_leg(**_kwargs):
        return LegSubmitResult(
            leg_id="leg-1",
            status="failed",
            effective_price=0.42,
            error_message="venue rejected",
            payload={"provider_status": "rejected"},
            provider_order_id="provider-order-1",
            provider_clob_order_id="provider-clob-1",
            shares=7.0,
            notional_usd=3.0,
        )

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-linked")
            signal = await session.get(TradeSignal, "signal-linked")
            result = await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id="decision-linked",
                decision_audit={
                    "decision": "selected",
                    "reason": "selected after provider submit",
                    "score": 9.0,
                    "checks_summary": {"fast_tier": True, "checks": []},
                    "risk_snapshot": {"fast_tier": True},
                    "payload": {"fast_tier": True},
                },
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="selected after provider submit",
            )
            await session.commit()

        async with session_factory() as session:
            decision = await session.get(TraderDecision, "decision-linked")
            order = (
                await session.execute(select(TraderOrder).where(TraderOrder.signal_id == "signal-linked"))
            ).scalar_one()

        assert result.orders_written == 1
        assert result.status == "failed"
        assert decision is not None
        assert decision.decision == "selected"
        assert order.decision_id == "decision-linked"
        assert order.provider_order_id == "provider-order-1"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_fast_signal_refuses_duplicate_when_existing_order_for_signal(monkeypatch):
    """If a TraderOrder already exists for (trader_id, signal_id), the fast
    path must refuse to submit again — this is the recovery guard for the
    'cursor advance silently failed' case described in the audit."""
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_duplicate_guard")

    submit_calls = {"count": 0}

    async def fake_submit_execution_leg(**_kwargs):
        submit_calls["count"] += 1
        return LegSubmitResult(
            leg_id="leg-1",
            status="executed",
            effective_price=0.42,
            error_message=None,
            payload={"provider_status": "filled"},
            notional_usd=3.0,
        )

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-dup")
            # Pre-populate a TraderOrder for this signal as if the previous
            # submission attempt had succeeded but the cursor advance failed.
            session.add(
                TraderOrder(
                    id="prev-order-1",
                    trader_id="fast-trader",
                    signal_id="signal-dup",
                    source="generic-source",
                    market_id="market-signal-dup",
                    mode="live",
                    status="executed",
                    notional_usd=3.0,
                    payload_json={"fast_tier": True},
                )
            )
            await session.commit()

            signal = await session.get(TradeSignal, "signal-dup")
            result = await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id=None,
                decision_audit=None,
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="duplicate guard test",
            )

        assert result.status == "skipped"
        assert result.orders_written == 0
        assert result.payload.get("reason") == "duplicate_signal_existing_order"
        assert result.payload.get("existing_trader_order_id") == "prev-order-1"
        assert submit_calls["count"] == 0  # CLOB was NOT touched
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_fast_signal_rolls_back_partial_decision_when_order_persist_fails(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "fast_submit_persist_failure_rollback")

    async def fake_submit_execution_leg(**_kwargs):
        return LegSubmitResult(
            leg_id="leg-1",
            status="failed",
            effective_price=0.42,
            error_message="venue rejected",
            payload={"provider_status": "rejected"},
            notional_usd=3.0,
        )

    def fake_build_trader_order_row(**_kwargs):
        raise RuntimeError("order write failed")

    monkeypatch.setattr(fast_submit, "submit_execution_leg", fake_submit_execution_leg)
    monkeypatch.setattr(fast_submit, "build_trader_order_row", fake_build_trader_order_row)

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-rollback")
            signal = await session.get(TradeSignal, "signal-rollback")
            result = await fast_submit.execute_fast_signal(
                session,
                trader_id="fast-trader",
                signal=signal,
                decision_id="decision-rollback",
                decision_audit={
                    "decision": "selected",
                    "reason": "selected after provider submit",
                    "payload": {"fast_tier": True},
                },
                strategy_key="generic-fast-strategy",
                strategy_version=None,
                strategy_params={},
                mode="live",
                size_usd=3.0,
                reason="selected after provider submit",
            )
            await session.commit()

        async with session_factory() as session:
            decision = await session.get(TraderDecision, "decision-rollback")

        assert result.orders_written == 0
        assert result.status == "failed"
        assert "order write failed" in str(result.error_message)
        assert decision is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fast_runtime_restarts_stale_per_trader_task(monkeypatch):
    runtime = fast_trader_runtime._FastRuntime()
    old_wake = asyncio.Event()
    old_runner = fast_trader_runtime._FastTraderTask(_fast_trader_config(), old_wake)
    now_mono = fast_trader_runtime.time.monotonic()
    old_runner._started_at_mono = now_mono - fast_trader_runtime._FAST_TASK_STALE_SECONDS - 10.0
    old_runner._last_cycle_started_at = now_mono - fast_trader_runtime._FAST_TASK_STALE_SECONDS - 5.0
    old_runner._last_cycle_finished_at = 0.0
    old_task = asyncio.get_running_loop().create_future()
    runtime._wake_events["fast-trader"] = old_wake
    runtime._task_objs["fast-trader"] = old_runner
    runtime._tasks["fast-trader"] = old_task

    class EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_list_fast_traders(session):
        assert session is not None
        return [_fast_trader_config()]

    created = {"count": 0}

    def fake_create_task(coro, *, name=None):
        assert name == "fast-trader-fast-trader"
        coro.close()
        created["count"] += 1
        return asyncio.get_running_loop().create_future()

    monkeypatch.setattr(fast_trader_runtime, "AsyncSessionLocal", lambda: EmptySession())

    async def fake_control_enabled(session):
        assert session is not None
        return {
            "is_enabled": True,
            "is_paused": False,
            "kill_switch": False,
            "mode": "live",
        }

    monkeypatch.setattr(fast_trader_runtime, "read_orchestrator_control", fake_control_enabled)
    monkeypatch.setattr(fast_trader_runtime, "list_fast_traders", fake_list_fast_traders)
    monkeypatch.setattr(fast_trader_runtime.asyncio, "create_task", fake_create_task)

    await runtime._refresh_roster()

    assert old_runner._stopped is True
    assert old_task.cancelled() is True
    assert created["count"] == 1
    assert runtime._task_objs["fast-trader"] is not old_runner
    assert runtime._tasks["fast-trader"] is not old_task


def test_fast_trader_filters_signal_strategy_types_by_source_config():
    task = object.__new__(fast_trader_runtime._FastTraderTask)
    task._trader = {
        "source_configs": [
            {
                "source_key": "feed",
                "strategy_key": "configured",
                "requested_strategy_key": "requested",
                "strategy_params": {
                    "accepted_signal_strategy_types": ["alternate", "configured"],
                },
            },
            {
                "source_key": "disabled",
                "strategy_key": "ignored",
                "enabled": False,
            },
        ]
    }

    assert task._accepted_strategy_types_by_source() == {
        "feed": ["configured", "requested", "alternate"]
    }


class _FakeColdStartCache:
    """Minimal stand-in for ``signal_cache.SignalCache`` for cold-start
    regression tests. Records what the runtime hydrated and what it
    upserted; reports as not-yet-hydrated for the trader so the
    cold-start branch executes."""

    def __init__(self) -> None:
        self.hydrated_ids: list[str] | None = None
        self.upserts: list = []
        self._consumed: dict[str, set[str]] = {}

    def is_ready(self) -> bool:
        return True

    def is_trader_hydrated(self, trader_id: str) -> bool:
        return trader_id in self._consumed

    def hydrate_trader_consumed_ids(self, trader_id, signal_ids):
        ids = [str(sid) for sid in signal_ids]
        self.hydrated_ids = ids
        self._consumed[str(trader_id or "")] = set(ids)

    def consumed_ids_for(self, trader_id: str) -> frozenset[str]:
        return frozenset(self._consumed.get(str(trader_id or ""), set()))

    def get_unconsumed_signals(self, **_kwargs):
        return []

    def upsert(self, snapshot) -> None:
        self.upserts.append(snapshot)


def _shadow_fast_trader_config() -> dict:
    cfg = _fast_trader_config()
    cfg["mode"] = "shadow"
    return cfg


@pytest.mark.asyncio
async def test_fast_trader_precycle_hydrates_consumed_set_from_db(monkeypatch):
    """Plan 0032 Task 7: pre-cycle hydrate MUST populate the consumed-set
    from ``trader_signal_consumption`` BEFORE the first
    ``intent_runtime.list_unconsumed_signals`` call. Without this, the
    scanner's 180s reactivation cooldown re-presents every signal whose
    ``TraderOrder`` already exists and the trader writes a
    ``trader_order already exists`` decision per cycle."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "fast_runtime_coldstart_hydrate"
    )

    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)
    monkeypatch.setattr(fast_trader_runtime, "FastAsyncSessionLocal", session_factory)

    fake_cache = _FakeColdStartCache()
    import services.signal_cache as _signal_cache

    monkeypatch.setattr(_signal_cache, "get_signal_cache", lambda: fake_cache)

    captured_query: dict = {"called_with": None}
    real_fetch = fast_trader_runtime.fetch_recent_consumed_signal_ids

    async def spy_fetch(session, *, trader_id, hours, limit):
        captured_query["called_with"] = {
            "trader_id": trader_id,
            "hours": hours,
            "limit": limit,
        }
        return await real_fetch(session, trader_id=trader_id, hours=hours, limit=limit)

    monkeypatch.setattr(fast_trader_runtime, "fetch_recent_consumed_signal_ids", spy_fetch)

    class _FakeIntentRuntime:
        async def list_unconsumed_signals(self, **kwargs):
            return []

    monkeypatch.setattr(fast_trader_runtime, "get_intent_runtime", lambda: _FakeIntentRuntime())

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-coldstart")
            now = utcnow().replace(tzinfo=None)
            session.add(
                TraderSignalConsumption(
                    id="cons-coldstart",
                    trader_id="fast-trader",
                    signal_id="signal-already-consumed",
                    decision_id=None,
                    outcome="skipped",
                    reason="prior cycle",
                    payload_json={},
                    consumed_at=now,
                )
            )
            await session.commit()

        runner = fast_trader_runtime._FastTraderTask(
            _shadow_fast_trader_config(), asyncio.Event()
        )
        await runner._run_once_inner("fast-trader", ["generic-source"])

        assert captured_query["called_with"] is not None
        assert captured_query["called_with"]["trader_id"] == "fast-trader"
        assert captured_query["called_with"]["hours"] == 48
        assert captured_query["called_with"]["limit"] == 50_000

        assert fake_cache.hydrated_ids == ["signal-already-consumed"]
        assert "precycle_consumed_hydrate" in runner._last_stage_timings_ms
        assert runner._consumed_set_hydrated is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fast_trader_precycle_falls_back_when_hydrate_query_raises(monkeypatch):
    """Plan 0032 Task 7: hydrate query failure must NEVER block the
    cycle. Falling back to an empty hydrate keeps the worker live;
    ``fast_submit``'s ``(trader_id, signal_id)`` idempotency-guard
    re-absorbs duplicates."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "fast_runtime_coldstart_hydrate_fail"
    )

    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)
    monkeypatch.setattr(fast_trader_runtime, "FastAsyncSessionLocal", session_factory)

    fake_cache = _FakeColdStartCache()
    import services.signal_cache as _signal_cache

    monkeypatch.setattr(_signal_cache, "get_signal_cache", lambda: fake_cache)

    async def raising_fetch(*_args, **_kwargs):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr(fast_trader_runtime, "fetch_recent_consumed_signal_ids", raising_fetch)

    class _FakeIntentRuntime:
        async def list_unconsumed_signals(self, **kwargs):
            return []

    monkeypatch.setattr(fast_trader_runtime, "get_intent_runtime", lambda: _FakeIntentRuntime())

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "signal-coldstart-fail")

        runner = fast_trader_runtime._FastTraderTask(
            _shadow_fast_trader_config(), asyncio.Event()
        )
        await runner._run_once_inner("fast-trader", ["generic-source"])

        assert fake_cache.hydrated_ids == []
        assert "precycle_consumed_hydrate" in runner._last_stage_timings_ms
        assert runner._consumed_set_hydrated is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fast_trader_filters_intent_runtime_signals_by_consumed_set(monkeypatch):
    """Plan 0032 Task 7: signals returned from
    ``intent_runtime.list_unconsumed_signals`` must be post-filtered by
    the in-process ``signal_cache`` consumed-set. Without this, the
    scanner's 180s reactivation cooldown re-presents previously
    submitted signals every cooldown window and the trader emits a
    ``trader_order already exists`` decision for each."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "fast_runtime_intent_consumed_filter"
    )

    hot_state._snapshots.clear()
    monkeypatch.setattr(hot_state, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(hot_state, "AuditAsyncSessionLocal", session_factory)
    monkeypatch.setattr(fast_trader_runtime, "FastAsyncSessionLocal", session_factory)

    fake_cache = _FakeColdStartCache()
    fake_cache.hydrate_trader_consumed_ids("fast-trader", ["already-consumed-signal"])

    import services.signal_cache as _signal_cache

    monkeypatch.setattr(_signal_cache, "get_signal_cache", lambda: fake_cache)

    async def empty_fetch(*_args, **_kwargs):
        return ["already-consumed-signal"]

    monkeypatch.setattr(fast_trader_runtime, "fetch_recent_consumed_signal_ids", empty_fetch)

    intent_signals = [
        SimpleNamespace(
            id="already-consumed-signal",
            source="generic-source",
            market_id="market-A",
            runtime_sequence=10,
            status="pending",
            payload_json={},
            strategy_context_json={},
        ),
        SimpleNamespace(
            id="fresh-signal",
            source="generic-source",
            market_id="market-B",
            runtime_sequence=11,
            status="pending",
            payload_json={},
            strategy_context_json={},
        ),
    ]

    class _FakeIntentRuntime:
        async def list_unconsumed_signals(self, **kwargs):
            return list(intent_signals)

    monkeypatch.setattr(fast_trader_runtime, "get_intent_runtime", lambda: _FakeIntentRuntime())

    processed: list[list] = []

    async def spy_process(self, signals, *, mode, default_size_usd):
        processed.append(list(signals))

    monkeypatch.setattr(
        fast_trader_runtime._FastTraderTask,
        "_process_signals_parallel_by_market",
        spy_process,
    )

    try:
        async with session_factory() as session:
            await _seed_trader_and_signal(session, "fresh-signal")

        runner = fast_trader_runtime._FastTraderTask(
            _shadow_fast_trader_config(), asyncio.Event()
        )
        await runner._run_once_inner("fast-trader", ["generic-source"])

        assert len(processed) == 1
        forwarded_ids = [str(getattr(sig, "id", "")) for sig in processed[0]]
        assert forwarded_ids == ["fresh-signal"]
        assert runner._last_stage_timings_ms.get("consumed_set_filtered") == 1
    finally:
        await engine.dispose()
