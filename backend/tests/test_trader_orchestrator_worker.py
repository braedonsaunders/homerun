import asyncio
import contextlib
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncpg
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import trader_orchestrator_worker
from services import trader_orchestrator_state
from services.runtime_status import runtime_status
from services.trader_orchestrator.strategies.base import StrategyDecision


def test_supports_live_market_context_defaults_enabled():
    assert trader_orchestrator_worker._supports_live_market_context(SimpleNamespace(source="crypto")) is True
    assert trader_orchestrator_worker._supports_live_market_context(SimpleNamespace(source="weather")) is True


def test_worker_logger_accepts_structured_warning_kwargs():
    trader_orchestrator_worker.logger.warning(
        "structured worker warning",
        lane="general",
        error="timeout",
    )


def test_pending_runtime_snapshots_restamp_release_time():
    snapshots = {
        "scanner": {
            "sig-1": {
                "id": "sig-1",
                "payload_json": {"signal_emitted_at": "2000-01-01T00:00:00Z"},
            }
        }
    }

    restamped = trader_orchestrator_worker._restamp_trigger_signal_snapshots_by_source(
        snapshots,
        emitted_at_iso="2026-06-13T01:30:00+00:00",
    )

    assert restamped["scanner"]["sig-1"]["payload_json"]["signal_emitted_at"] == "2026-06-13T01:30:00+00:00"
    assert snapshots["scanner"]["sig-1"]["payload_json"]["signal_emitted_at"] == "2000-01-01T00:00:00Z"


def test_signal_db_transient_classifier_handles_connection_and_timeout_errors():
    assert trader_orchestrator_worker._is_transient_db_error(asyncio.TimeoutError())
    assert trader_orchestrator_worker._is_transient_db_error(
        trader_orchestrator_worker.InterfaceError(
            "select 1",
            {},
            RuntimeError("server disconnected"),
        )
    )
    assert trader_orchestrator_worker._is_transient_db_error(
        Exception("cannot switch to state 15; another operation is in progress")
    )
    assert not trader_orchestrator_worker._is_transient_db_error(
        IntegrityError("insert", {}, RuntimeError("foreign key violation"))
    )
    assert not trader_orchestrator_worker._is_transient_db_error(Exception("strategy validation failed"))


def test_enforce_strict_ws_strategy_params_preserves_tighter_existing_budget():
    params = trader_orchestrator_worker._enforce_strict_ws_strategy_params(
        {"max_market_data_age_ms": 15000},
        strict_age_budget_ms=30000,
        strict_ws_price_sources=["ws_strict"],
    )

    assert params["require_strict_ws_pricing"] is True
    assert params["strict_ws_price_sources"] == ["ws_strict"]
    assert params["max_market_data_age_ms"] == 15000


def test_enforce_strict_ws_strategy_params_caps_looser_existing_budget():
    params = trader_orchestrator_worker._enforce_strict_ws_strategy_params(
        {"max_market_data_age_ms": 45000, "strict_ws_price_sources": ["redis_strict"]},
        strict_age_budget_ms=30000,
        strict_ws_price_sources=["ws_strict"],
    )

    assert params["max_market_data_age_ms"] == 30000
    assert params["strict_ws_price_sources"] == ["redis_strict"]


def test_build_execution_latency_sample_uses_execution_armed_at_for_release_sla():
    armed_at = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    emitted_at = armed_at - timedelta(seconds=30)
    signal = SimpleNamespace(
        payload_json={
            "execution_armed_at": armed_at.isoformat().replace("+00:00", "Z"),
            "signal_emitted_at": emitted_at.isoformat().replace("+00:00", "Z"),
            "ingested_at": emitted_at.isoformat().replace("+00:00", "Z"),
        }
    )

    sample = trader_orchestrator_worker._build_execution_latency_sample(
        signal,
        wake_started_at=armed_at + timedelta(seconds=1),
        context_ready_at=armed_at + timedelta(seconds=2),
        decision_ready_at=armed_at + timedelta(seconds=3),
        submit_started_at=armed_at + timedelta(seconds=4),
        submit_completed_at=armed_at + timedelta(seconds=5),
    )

    assert sample["armed_to_ws_release_ms"] == 0
    assert sample["emit_to_queue_wake_ms"] == 31000
    assert sample["ws_release_to_submit_start_ms"] == 4000
    assert sample["emit_to_submit_start_ms"] == 34000


def test_build_execution_latency_sample_uses_latest_emit_time_when_requeued():
    armed_at = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    emitted_at = armed_at + timedelta(seconds=45)
    signal = SimpleNamespace(
        payload_json={
            "execution_armed_at": armed_at.isoformat().replace("+00:00", "Z"),
            "signal_emitted_at": emitted_at.isoformat().replace("+00:00", "Z"),
            "ingested_at": emitted_at.isoformat().replace("+00:00", "Z"),
        }
    )

    sample = trader_orchestrator_worker._build_execution_latency_sample(
        signal,
        wake_started_at=emitted_at + timedelta(seconds=1),
        context_ready_at=emitted_at + timedelta(seconds=2),
        decision_ready_at=emitted_at + timedelta(seconds=3),
        submit_started_at=emitted_at + timedelta(seconds=4),
        submit_completed_at=emitted_at + timedelta(seconds=5),
    )

    assert sample["armed_to_ws_release_ms"] == 45000
    assert sample["emit_to_queue_wake_ms"] == 1000
    assert sample["ws_release_to_submit_start_ms"] == 4000
    assert sample["emit_to_submit_start_ms"] == 4000


def test_json_safe_runtime_signal_value_serializes_nested_datetimes():
    observed_at = datetime(2026, 4, 16, 3, 1, 51, tzinfo=timezone.utc)

    payload = trader_orchestrator_worker._json_safe_runtime_signal_value(
        {
            "observed_at": observed_at,
            "nested": {
                "expires_at": observed_at + timedelta(minutes=5),
            },
            "history": [observed_at],
        }
    )

    assert payload == {
        "observed_at": "2026-04-16T03:01:51Z",
        "nested": {
            "expires_at": "2026-04-16T03:06:51Z",
        },
        "history": ["2026-04-16T03:01:51Z"],
    }


@pytest.mark.asyncio
async def test_ensure_runtime_signal_persisted_uses_on_conflict_do_nothing_for_runtime_duplicates():
    statements: list[object] = []

    class _Session:
        async def execute(self, stmt):
            statements.append(stmt)
            return None

    signal = SimpleNamespace(
        id="signal-runtime-1",
        source="scanner",
        source_item_id="scanner-item-1",
        signal_type="scanner_opportunity",
        strategy_type="prob_surface_arb",
        market_id="market-1",
        market_question="Will this avoid duplicate runtime inserts?",
        direction="buy_yes",
        entry_price=0.42,
        edge_percent=5.5,
        confidence=0.77,
        liquidity=1000.0,
        expires_at=datetime(2026, 4, 16, 5, 15, tzinfo=timezone.utc),
        payload_json={
            "observed_at": datetime(2026, 4, 16, 3, 15, tzinfo=timezone.utc),
            "history": [datetime(2026, 4, 16, 3, 16, tzinfo=timezone.utc)],
        },
        strategy_context_json={
            "ingested_at": datetime(2026, 4, 16, 3, 17, tzinfo=timezone.utc),
        },
        quality_passed=True,
        dedupe_key="dedupe-runtime-1",
        runtime_sequence=123,
        status="pending",
    )

    await trader_orchestrator_worker._ensure_runtime_signal_persisted(_Session(), signal)

    assert len(statements) == 1
    compiled = statements[0].compile(dialect=postgresql.dialect())
    assert "ON CONFLICT DO NOTHING" in str(compiled)
    assert compiled.params["payload_json"] == {
        "observed_at": "2026-04-16T03:15:00Z",
        "history": ["2026-04-16T03:16:00Z"],
    }
    assert compiled.params["strategy_context_json"] == {
        "ingested_at": "2026-04-16T03:17:00Z",
    }


@pytest.mark.asyncio
async def test_list_unconsumed_trade_signals_prefers_intent_runtime(monkeypatch):
    runtime_signal = SimpleNamespace(
        id="runtime-signal-1",
        source="scanner",
        strategy_type="generic-runtime-strategy",
        market_id="runtime-market-1",
        status="pending",
        runtime_sequence=120,
    )
    observed_kwargs: dict[str, object] = {}

    class _Runtime:
        async def list_unconsumed_signals(self, **kwargs):
            observed_kwargs.update(kwargs)
            return [runtime_signal]

    async def _authoritative_must_not_run(*_args, **_kwargs):
        raise AssertionError("standard orchestrator hot path must not query trade_signals when runtime has signals")

    monkeypatch.setattr(trader_orchestrator_worker, "get_intent_runtime", lambda: _Runtime())
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        _authoritative_must_not_run,
    )

    rows = await trader_orchestrator_worker.list_unconsumed_trade_signals(
        object(),
        trader_id="trader-1",
        sources=["scanner"],
        statuses=["pending", "selected"],
        strategy_types_by_source={"scanner": ["generic-runtime-strategy"]},
        cursor_runtime_sequence=100,
        limit=10,
    )

    assert rows == [runtime_signal]
    assert observed_kwargs["trader_id"] == "trader-1"
    assert observed_kwargs["cursor_runtime_sequence"] == 100


@pytest.mark.asyncio
async def test_triggered_signal_build_uses_snapshots_when_authoritative_db_is_busy(monkeypatch):
    async def _db_timeout(*_args, **_kwargs):
        raise asyncio.TimeoutError()

    pressure_events: list[str] = []

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        _db_timeout,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "maybe_mark_db_pressure",
        lambda _exc, *, component, ttl_seconds: pressure_events.append(component),
    )

    rows = await trader_orchestrator_worker._build_triggered_trade_signals(
        object(),
        trader_id="trader-1",
        signal_ids_by_source={"scanner": ["snapshot-signal-1"]},
        signal_snapshots_by_source={
            "scanner": {
                "snapshot-signal-1": {
                    "id": "snapshot-signal-1",
                    "source": "scanner",
                    "signal_type": "scanner_opportunity",
                    "strategy_type": "generic-runtime-strategy",
                    "market_id": "snapshot-market-1",
                    "market_question": "Snapshot-backed signal?",
                    "direction": "buy_yes",
                    "entry_price": 0.42,
                    "edge_percent": 6.0,
                    "confidence": 0.8,
                    "liquidity": 1000.0,
                    "status": "pending",
                    "payload_json": {},
                    "strategy_context_json": {},
                    "runtime_sequence": 130,
                    "created_at": datetime(2026, 4, 16, 3, 30, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 4, 16, 3, 30, tzinfo=timezone.utc),
                }
            }
        },
        sources=["scanner"],
        strategy_types_by_source={"scanner": ["generic-runtime-strategy"]},
        cursor_runtime_sequence=100,
        cursor_created_at=None,
        cursor_signal_id=None,
        statuses=["pending"],
        limit=10,
    )

    assert [row.id for row in rows] == ["snapshot-signal-1"]
    assert rows[0].market_id == "snapshot-market-1"
    assert pressure_events == ["trader_orchestrator_trigger_signals"]


def test_maybe_log_execution_latency_sla_breach_does_not_repeat_same_signature(monkeypatch):
    warnings: list[dict[str, object]] = []

    def _warning(message: str, **kwargs):
        warnings.append({"message": message, **kwargs})

    monkeypatch.setattr(trader_orchestrator_worker.logger, "warning", _warning)
    trader_orchestrator_worker._latency_sla_breach_logged_at.clear()
    trader_orchestrator_worker._latency_sla_breach_last_signature.clear()

    metrics = {
        "execution_latency": {
            "internal_sla_target_ms": 200,
            "sample_count": 1,
            "overall": {
                "ws_release_to_submit_start_ms": {
                    "p95": 18048,
                    "p99": 18048,
                }
            },
            "by_source": {},
            "by_strategy": {},
            "by_trader": {},
        }
    }

    trader_orchestrator_worker._maybe_log_execution_latency_sla_breach(lane="general", metrics=metrics)
    trader_orchestrator_worker._maybe_log_execution_latency_sla_breach(lane="general", metrics=metrics)

    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_write_orchestrator_snapshot_updates_runtime_status_before_db_write(monkeypatch):
    original_snapshot = runtime_status.get_orchestrator()
    calls: list[dict[str, object]] = []

    async def _fake_write_orchestrator_snapshot(_session, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "write_orchestrator_snapshot",
        _fake_write_orchestrator_snapshot,
    )
    trader_orchestrator_worker._orchestrator_snapshot_last_persist_mono.clear()

    last_run_at = datetime(2026, 4, 9, 8, 40, 0, tzinfo=timezone.utc)
    try:
        await trader_orchestrator_worker._write_orchestrator_snapshot_best_effort(
            object(),
            running=True,
            enabled=True,
            current_activity="Cycle[scheduled:general]",
            interval_seconds=5,
            last_run_at=last_run_at,
            last_error=None,
            stats={"open_orders": 2},
        )

        runtime_snapshot = runtime_status.get_orchestrator()
        assert calls and calls[0]["current_activity"] == "Cycle[scheduled:general]"
        assert runtime_snapshot["running"] is True
        assert runtime_snapshot["enabled"] is True
        assert runtime_snapshot["current_activity"] == "Cycle[scheduled:general]"
        assert runtime_snapshot["interval_seconds"] == 5
        assert runtime_snapshot["last_run_at"] == last_run_at.isoformat().replace("+00:00", "Z")
        assert runtime_snapshot["stats"] == {"open_orders": 2}
    finally:
        runtime_status._orchestrator = original_snapshot  # type: ignore[attr-defined]
        trader_orchestrator_worker._orchestrator_snapshot_last_persist_mono.clear()


@pytest.mark.asyncio
async def test_completed_orchestrator_snapshot_bypasses_persist_throttle(monkeypatch):
    calls: list[dict[str, object]] = []

    async def _fake_write_orchestrator_snapshot(_session, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "write_orchestrator_snapshot",
        _fake_write_orchestrator_snapshot,
    )
    trader_orchestrator_worker._orchestrator_snapshot_last_persist_mono["general"] = time.monotonic()
    trader_orchestrator_worker._orch_snapshot_last_persisted_state["general"] = (True, True, False)

    last_run_at = datetime(2026, 6, 12, 21, 9, 5, tzinfo=timezone.utc)
    try:
        await trader_orchestrator_worker._write_orchestrator_snapshot_best_effort(
            object(),
            running=True,
            enabled=True,
            current_activity="Cycle[scheduled:general] signals=0 decisions=0 orders=0",
            interval_seconds=5,
            last_run_at=last_run_at,
            last_error=None,
            stats={"open_orders": 0},
        )

        assert len(calls) == 1
        assert calls[0]["last_run_at"] == last_run_at
        assert calls[0]["current_activity"] == "Cycle[scheduled:general] signals=0 decisions=0 orders=0"
    finally:
        trader_orchestrator_worker._orchestrator_snapshot_last_persist_mono.clear()
        trader_orchestrator_worker._orch_snapshot_last_persisted_state.clear()


@pytest.mark.asyncio
async def test_write_orchestrator_snapshot_swallows_query_canceled_errors(monkeypatch):
    original_snapshot = runtime_status.get_orchestrator()
    rollback_calls: list[str] = []

    class _Session:
        async def rollback(self):
            rollback_calls.append("rollback")

    async def _raise_query_canceled(_session, **_kwargs):
        raise asyncpg.exceptions.QueryCanceledError("canceling statement due to statement timeout")

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "write_orchestrator_snapshot",
        _raise_query_canceled,
    )
    trader_orchestrator_worker._orchestrator_snapshot_last_persist_mono.clear()

    last_run_at = datetime(2026, 4, 10, 9, 15, 0, tzinfo=timezone.utc)
    try:
        await trader_orchestrator_worker._write_orchestrator_snapshot_best_effort(
            _Session(),
            running=True,
            enabled=True,
            current_activity="Cycle[runtime-trigger:general]",
            interval_seconds=5,
            last_run_at=last_run_at,
            last_error=None,
            stats={"open_orders": 1},
        )

        runtime_snapshot = runtime_status.get_orchestrator()
        assert rollback_calls == ["rollback"]
        assert runtime_snapshot["running"] is True
        assert runtime_snapshot["current_activity"] == "Cycle[runtime-trigger:general]"
        assert runtime_snapshot["last_run_at"] == last_run_at.isoformat().replace("+00:00", "Z")
    finally:
        runtime_status._orchestrator = original_snapshot  # type: ignore[attr-defined]
        trader_orchestrator_worker._orchestrator_snapshot_last_persist_mono.clear()


@pytest.mark.asyncio
async def test_submit_order_executes_without_forcing_decision_persistence():
    call_order: list[str] = []

    async def _fake_execute_signal(**kwargs):
        call_order.append("execute")
        assert kwargs["decision_id"] == "decision-1"
        return {"status": "ok"}

    session_engine = SimpleNamespace(execute_signal=_fake_execute_signal)
    result = await trader_orchestrator_worker.submit_order(
        session_engine=session_engine,
        trader_id="trader-1",
        signal=SimpleNamespace(id="signal-1"),
        decision_id="decision-1",
        strategy_key="tail_end_carry",
        strategy_version=1,
        strategy_params={},
        risk_limits={},
        mode="shadow",
        size_usd=25.0,
        reason="selected",
    )

    assert result == {"status": "ok"}
    assert call_order == ["execute"]


def test_supports_live_market_context_ignores_explicit_disable():
    assert (
        trader_orchestrator_worker._supports_live_market_context(
            SimpleNamespace(source="crypto"),
            source_config={
                "strategy_key": "btc_eth_maker_quote",
                "strategy_params": {"enable_live_market_context": False},
            },
        )
        is True
    )


def test_source_config_allows_new_entries_respects_strategy_param_overrides(monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda _source_config: None,
    )

    base_config = {
        "source_key": "crypto",
        "strategy_key": "btc_eth_maker_quote",
        "strategy_params": {},
    }

    disabled = dict(base_config)
    disabled["strategy_params"] = {"allow_new_entries": False}
    assert trader_orchestrator_worker._source_config_allows_new_entries(disabled) is False

    disabled_alias = dict(base_config)
    disabled_alias["strategy_params"] = {"disable_new_entries": True}
    assert trader_orchestrator_worker._source_config_allows_new_entries(disabled_alias) is False

    enabled = dict(base_config)
    enabled["strategy_params"] = {"allow_new_entries": True}
    assert trader_orchestrator_worker._source_config_allows_new_entries(enabled) is True


def test_source_config_allows_new_entries_infers_exit_only_strategy(monkeypatch):
    class _ExitOnlyStrategy:
        def should_exit(self, position, market_state):
            return None

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda _source_config: _ExitOnlyStrategy(),
    )

    assert trader_orchestrator_worker._source_config_allows_new_entries(
        {
            "source_key": "crypto",
            "strategy_key": "manual_wallet_position",
            "strategy_params": {},
        }
    ) is False


def test_source_config_allows_new_entries_keeps_entry_enabled_when_evaluate_defined(monkeypatch):
    class _EvaluateAndExitStrategy:
        def evaluate(self, signal, context):
            return None

        def should_exit(self, position, market_state):
            return None

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda _source_config: _EvaluateAndExitStrategy(),
    )

    assert trader_orchestrator_worker._source_config_allows_new_entries(
        {
            "source_key": "crypto",
            "strategy_key": "manual_wallet_position",
            "strategy_params": {},
        }
    ) is True


def test_merged_strategy_params_use_loaded_strategy_config_defaults(monkeypatch):
    strategy = SimpleNamespace(
        config={
            "opening_directional_buy_yes_enabled": True,
            "max_signal_age_seconds": 30.0,
        }
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda source_config: strategy,
    )

    merged = trader_orchestrator_worker._merged_strategy_params_for_source_config(
        {
            "source_key": "crypto",
            "strategy_key": "btc_eth_maker_quote",
            "strategy_params": {
                "max_signal_age_seconds": 7.5,
            },
        }
    )

    assert merged["opening_directional_buy_yes_enabled"] is True
    assert merged["max_signal_age_seconds"] == 7.5


@pytest.mark.asyncio
async def test_build_triggered_trade_signals_prefers_runtime_signal_snapshots(monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        AsyncMock(return_value=[SimpleNamespace(id="signal-1", status="pending", runtime_sequence=7)]),
    )

    rows = await trader_orchestrator_worker._build_triggered_trade_signals(
        None,
        trader_id="trader-1",
        signal_ids_by_source={"crypto": ["signal-1"]},
        signal_snapshots_by_source={
            "crypto": {
                "signal-1": {
                    "id": "signal-1",
                    "source": "crypto",
                    "strategy_type": "crypto_strategy",
                    "market_id": "market-1",
                    "direction": "buy_no",
                    "entry_price": 0.2,
                    "edge_percent": 5.0,
                    "confidence": 0.7,
                    "liquidity": 5000.0,
                    "status": "pending",
                    "payload_json": {"asset": "BTC", "timeframe": "5m"},
                    "strategy_context_json": {"asset": "BTC", "timeframe": "5m"},
                    "created_at": "2026-03-10T02:39:50Z",
                    "updated_at": "2026-03-10T02:39:51Z",
                }
            }
        },
        sources=["crypto"],
        strategy_types_by_source={"crypto": ["crypto_strategy"]},
        cursor_runtime_sequence=None,
        cursor_created_at=None,
        cursor_signal_id=None,
        statuses=["pending", "selected"],
        limit=10,
    )

    assert [row.id for row in rows] == ["signal-1"]
    assert rows[0].payload_json["asset"] == "BTC"


@pytest.mark.asyncio
async def test_build_triggered_trade_signals_excludes_consumed_or_stale_ids(monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        AsyncMock(return_value=[SimpleNamespace(id="signal-2", status="pending", runtime_sequence=12)]),
    )

    rows = await trader_orchestrator_worker._build_triggered_trade_signals(
        None,
        trader_id="trader-1",
        signal_ids_by_source={"scanner": ["signal-1", "signal-2"]},
        signal_snapshots_by_source={
            "scanner": {
                "signal-1": {
                    "id": "signal-1",
                    "source": "scanner",
                    "strategy_type": "tail_end_carry",
                    "market_id": "market-1",
                    "direction": "buy_no",
                    "entry_price": 0.08,
                    "edge_percent": 6.0,
                    "confidence": 0.75,
                    "liquidity": 2000.0,
                    "status": "pending",
                    "payload_json": {"question": "stale"},
                    "strategy_context_json": {},
                    "created_at": "2026-03-10T02:39:50Z",
                    "updated_at": "2026-03-10T02:39:51Z",
                },
                "signal-2": {
                    "id": "signal-2",
                    "source": "scanner",
                    "strategy_type": "custom_scanner_strategy",
                    "market_id": "market-2",
                    "direction": "buy_no",
                    "entry_price": 0.09,
                    "edge_percent": 7.0,
                    "confidence": 0.78,
                    "liquidity": 2500.0,
                    "status": "pending",
                    "payload_json": {
                        "question": "fresh",
                        "live_market": {
                            "available": True,
                            "market_data_source": "ws_strict",
                            "market_data_age_ms": 120.0,
                        },
                    },
                    "strategy_context_json": {},
                    "created_at": "2026-03-10T02:39:52Z",
                    "updated_at": "2026-03-10T02:39:53Z",
                },
            }
        },
        sources=["scanner"],
        strategy_types_by_source={"scanner": ["custom_scanner_strategy"]},
        cursor_runtime_sequence=None,
        cursor_created_at=None,
        cursor_signal_id=None,
        statuses=["pending", "selected"],
        limit=10,
    )

    assert [row.id for row in rows] == ["signal-2"]
    assert rows[0].payload_json["question"] == "fresh"
    assert rows[0].payload_json["live_market"]["market_data_source"] == "ws_strict"


@pytest.mark.asyncio
async def test_build_triggered_trade_signals_uses_runtime_snapshot_when_projection_lags(monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        AsyncMock(return_value=[]),
    )

    rows = await trader_orchestrator_worker._build_triggered_trade_signals(
        None,
        trader_id="trader-1",
        signal_ids_by_source={"crypto": ["signal-1"]},
        signal_snapshots_by_source={
            "crypto": {
                "signal-1": {
                    "id": "signal-1",
                    "source": "crypto",
                    "strategy_type": "crypto_strategy",
                    "market_id": "market-1",
                    "direction": "buy_no",
                    "entry_price": 0.2,
                    "edge_percent": 5.0,
                    "confidence": 0.7,
                    "liquidity": 5000.0,
                    "status": "pending",
                    "runtime_sequence": 21,
                    "payload_json": {"asset": "BTC", "timeframe": "5m"},
                    "strategy_context_json": {"asset": "BTC", "timeframe": "5m"},
                    "created_at": "2026-03-10T02:39:50Z",
                    "updated_at": "2026-03-10T02:39:51Z",
                }
            }
        },
        sources=["crypto"],
        strategy_types_by_source={"crypto": ["crypto_strategy"]},
        cursor_runtime_sequence=20,
        cursor_created_at=None,
        cursor_signal_id=None,
        statuses=["pending", "selected"],
        limit=10,
    )

    assert [row.id for row in rows] == ["signal-1"]
    assert rows[0].runtime_sequence == 21
    assert rows[0].payload_json["asset"] == "BTC"


@pytest.mark.asyncio
async def test_upsert_trader_signal_cursor_prefers_persisted_runtime_sequence(monkeypatch):
    class _Session:
        def __init__(self):
            self.cursor_row = SimpleNamespace(last_runtime_sequence=None)

        async def get(self, model, key):
            model_name = getattr(model, "__name__", "")
            if model_name == "TradeSignal" and key == "signal-1":
                return SimpleNamespace(runtime_sequence=42)
            if model_name == "TraderSignalCursor" and key == "trader-1":
                return self.cursor_row
            return None

        async def flush(self):
            return None

    session = _Session()
    hot_state_mock = SimpleNamespace(update_signal_cursor=Mock(return_value=None))
    persist_mock = AsyncMock(return_value=None)
    runtime_mock = SimpleNamespace(get_runtime_sequence=lambda _signal_id: 7)

    monkeypatch.setattr(trader_orchestrator_worker, "hot_state", hot_state_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_signal_cursor", persist_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_intent_runtime", lambda: runtime_mock)

    await trader_orchestrator_worker.upsert_trader_signal_cursor(
        session,
        trader_id="trader-1",
        last_signal_created_at=datetime.now(timezone.utc),
        last_signal_id="signal-1",
        commit=False,
    )

    hot_state_mock.update_signal_cursor.assert_called_once()
    assert hot_state_mock.update_signal_cursor.call_args.args[3] == "signal-1"
    assert hot_state_mock.update_signal_cursor.call_args.args[4] == 42
    assert session.cursor_row.last_runtime_sequence == 42
    persist_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_trader_once_fetches_signals_without_market_exclusion(monkeypatch):
    list_calls: list[dict[str, object]] = []

    async def _list_unconsumed(*args, **kwargs):
        list_calls.append(kwargs)
        return []

    heartbeat_mock = AsyncMock(return_value=None)
    persist_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value={"market-open", "market-cooldown"}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_reentry_cooldown_market_ids_for_trader",
        AsyncMock(return_value={"market-cooldown"}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_unrealized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_live_provider_failure_snapshot",
        AsyncMock(return_value={"count": 0, "errors": []}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_live_risk_clamp_event_should_emit", AsyncMock(return_value=False))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        AsyncMock(return_value={"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_enforce_source_open_order_timeouts",
        AsyncMock(
            return_value={
                "configured": 0,
                "updated": 0,
                "suppressed": 0,
                "taker_rescue_attempted": 0,
                "taker_rescue_succeeded": 0,
                "taker_rescue_failed": 0,
                "sources": [],
                "errors": [],
                "provider_reconcile": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", heartbeat_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", persist_mock)

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=False),
        _base_control_payload(),
    )

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    assert list_calls
    assert all("exclude_market_ids" not in call for call in list_calls)


def test_trigger_signal_snapshots_for_trader_filters_by_source():
    trader = {
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "crypto_strategy",
                "strategy_params": {},
            }
        ]
    }
    trigger = {
        "source_signal_snapshots": {
            "crypto": {"signal-1": {"id": "signal-1", "source": "crypto"}},
            "scanner": {"signal-2": {"id": "signal-2", "source": "scanner"}},
        }
    }

    filtered = trader_orchestrator_worker._trigger_signal_snapshots_for_trader(trader, trigger)

    assert filtered == {"crypto": {"signal-1": {"id": "signal-1", "source": "crypto"}}}


def test_trigger_signal_snapshots_for_trader_filters_by_strategy_type():
    trader = {
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "tail_end_carry",
                "strategy_params": {},
            }
        ]
    }
    trigger = {
        "source_signal_snapshots": {
            "scanner": {
                "signal-1": {"id": "signal-1", "source": "scanner", "strategy_type": "tail_end_carry"},
                "signal-2": {"id": "signal-2", "source": "scanner", "strategy_type": "negrisk"},
            }
        }
    }

    filtered = trader_orchestrator_worker._trigger_signal_snapshots_for_trader(trader, trigger)

    assert filtered == {
        "scanner": {
            "signal-1": {"id": "signal-1", "source": "scanner", "strategy_type": "tail_end_carry"},
        }
    }


def test_runtime_trigger_matches_trader_rejects_mismatched_strategy_snapshots():
    trader = {
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "tail_end_carry",
                "strategy_params": {},
            }
        ]
    }
    trigger = {
        "source_signal_ids": {
            "scanner": ["signal-1"],
        },
        "source_signal_snapshots": {
            "scanner": {
                "signal-1": {"id": "signal-1", "source": "scanner", "strategy_type": "negrisk"},
            }
        },
    }

    assert trader_orchestrator_worker._runtime_trigger_matches_trader(trader, trigger) is False
    assert trader_orchestrator_worker._trigger_signal_ids_for_trader(trader, trigger) is None


def test_is_crypto_source_trader_uses_source_key_not_strategy_slug():
    trader = {
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "some_future_crypto_strategy",
                "strategy_params": {},
            }
        ]
    }

    assert trader_orchestrator_worker._is_crypto_source_trader(trader) is True


def test_normalize_source_configs_merges_strategy_defaults(monkeypatch):
    strategy = SimpleNamespace(
        config={
            "opening_directional_buy_yes_enabled": True,
            "reentry_cooldown_seconds_per_market": 15.0,
        }
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda source_config: strategy,
    )

    normalized = trader_orchestrator_worker._normalize_source_configs(
        {
            "source_configs": [
                {
                    "source_key": "crypto",
                    "strategy_key": "btc_eth_maker_quote",
                    "strategy_params": {"reentry_cooldown_seconds_per_market": 8.0},
                }
            ]
        }
    )

    crypto = normalized["crypto"]["strategy_params"]
    assert crypto["opening_directional_buy_yes_enabled"] is True
    assert crypto["reentry_cooldown_seconds_per_market"] == 8.0


def test_normalize_source_configs_preserves_explicit_params_without_strategy_defaults(monkeypatch):
    strategy = SimpleNamespace(
        config={
            "take_profit_pct": 15.0,
            "opening_directional_buy_yes_enabled": True,
        }
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda source_config: strategy,
    )

    normalized = trader_orchestrator_worker._normalize_source_configs(
        {
            "source_configs": [
                {
                    "source_key": "crypto",
                    "strategy_key": "btc_eth_maker_quote",
                    "strategy_params": {"max_signal_age_seconds": 7.5},
                }
            ]
        }
    )

    crypto = normalized["crypto"]
    assert crypto["strategy_params"]["take_profit_pct"] == 15.0
    assert crypto["strategy_params"]["opening_directional_buy_yes_enabled"] is True
    assert crypto["explicit_strategy_params"] == {"max_signal_age_seconds": 7.5}


def test_merged_strategy_params_for_traders_copy_trade_uses_copy_validator(monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_strategy_instance_for_source_config",
        lambda source_config: None,
    )

    merged = trader_orchestrator_worker._merged_strategy_params_for_source_config(
        {
            "source_key": "traders",
            "strategy_key": "traders_copy_trade",
            "strategy_params": {
                "min_confidence": "0.73",
                "copy_buys": "false",
                "copy_sells": "true",
                "max_signal_age_seconds": "45",
                "firehose_require_active_signal": False,
            },
        }
    )

    assert merged["min_confidence"] == 0.73
    assert merged["copy_buys"] is False
    assert merged["copy_sells"] is True
    assert merged["max_signal_age_seconds"] == 45
    assert "firehose_require_active_signal" not in merged


def test_resume_policy_normalizes_to_supported_values():
    assert trader_orchestrator_worker._normalize_resume_policy("manage_only") == "manage_only"
    assert trader_orchestrator_worker._normalize_resume_policy("flatten_then_start") == "flatten_then_start"
    assert trader_orchestrator_worker._normalize_resume_policy("unexpected") == "resume_full"


def test_live_risk_clamps_tighten_aggressive_limits():
    limits = {
        "allow_averaging": True,
        "cooldown_seconds": 0,
        "max_consecutive_losses": 8,
        "max_open_orders": 20,
        "max_open_positions": 12,
        "max_trade_notional_usd": 500.0,
        "max_per_market_exposure_usd": 500.0,
        "max_orders_per_cycle": 50,
    }
    changes = trader_orchestrator_worker._apply_live_risk_clamps(
        limits,
        {
            "enforce_allow_averaging_off": True,
            "min_cooldown_seconds": 90,
            "max_consecutive_losses_cap": 3,
            "max_open_orders_cap": 6,
            "max_open_positions_cap": 4,
            "max_trade_notional_usd_cap": 200.0,
            "max_per_market_exposure_usd_cap": 150.0,
            "max_orders_per_cycle_cap": 4,
            "enforce_halt_on_consecutive_losses": True,
        },
    )

    assert limits["allow_averaging"] is False
    assert limits["cooldown_seconds"] == 90
    assert limits["max_consecutive_losses"] == 3
    assert limits["max_open_orders"] == 6
    assert limits["max_open_positions"] == 4
    assert limits["max_trade_notional_usd"] == 200.0
    assert limits["max_per_market_exposure_usd"] == 150.0
    assert limits["max_orders_per_cycle"] == 4
    assert limits["halt_on_consecutive_losses"] is True
    assert "allow_averaging" in changes
    assert "max_trade_notional_usd" in changes


def test_live_risk_clamps_honor_configured_caps():
    limits = {
        "allow_averaging": False,
        "cooldown_seconds": 0,
        "max_consecutive_losses": 8,
        "max_open_orders": 20,
        "max_open_positions": 12,
        "max_trade_notional_usd": 500.0,
        "max_per_market_exposure_usd": 500.0,
        "max_orders_per_cycle": 50,
        "halt_on_consecutive_losses": False,
    }
    trader_orchestrator_worker._apply_live_risk_clamps(
        limits,
        {
            "enforce_allow_averaging_off": False,
            "min_cooldown_seconds": 0,
            "max_consecutive_losses_cap": 7,
            "max_open_orders_cap": 9,
            "max_open_positions_cap": 8,
            "max_trade_notional_usd_cap": 420.0,
            "max_per_market_exposure_usd_cap": 300.0,
            "max_orders_per_cycle_cap": 11,
            "enforce_halt_on_consecutive_losses": False,
        },
    )

    assert limits["cooldown_seconds"] == 0
    assert limits["max_consecutive_losses"] == 7
    assert limits["max_open_orders"] == 9
    assert limits["max_open_positions"] == 8
    assert limits["max_trade_notional_usd"] == 420.0
    assert limits["max_per_market_exposure_usd"] == 300.0
    assert limits["max_orders_per_cycle"] == 11
    assert limits["halt_on_consecutive_losses"] is False


def test_live_risk_clamps_empty_does_not_override_trader_limits():
    limits = {
        "allow_averaging": True,
        "cooldown_seconds": 0,
        "max_consecutive_losses": 100,
        "max_open_orders": 50,
        "max_open_positions": 25,
        "max_trade_notional_usd": 5000.0,
        "max_per_market_exposure_usd": 5000.0,
        "max_orders_per_cycle": 80,
        "halt_on_consecutive_losses": False,
    }
    changes = trader_orchestrator_worker._apply_live_risk_clamps(limits, {})

    assert limits["allow_averaging"] is True
    assert limits["cooldown_seconds"] == 0
    assert limits["max_consecutive_losses"] == 100
    assert limits["max_open_orders"] == 50
    assert limits["max_open_positions"] == 25
    assert limits["max_trade_notional_usd"] == 5000.0
    assert limits["max_per_market_exposure_usd"] == 5000.0
    assert limits["max_orders_per_cycle"] == 80
    assert limits["halt_on_consecutive_losses"] is False
    assert changes == {}


def test_live_risk_clamps_explicit_fields_only_clamp_configured_limits():
    limits = {
        "max_open_positions": 25,
        "max_open_orders": 50,
        "max_trade_notional_usd": 5000.0,
        "max_per_market_exposure_usd": 5000.0,
    }
    changes = trader_orchestrator_worker._apply_live_risk_clamps(
        limits, {
            "max_open_positions_cap": 10,
            "max_trade_notional_usd_cap": 25.0,
            "max_per_market_exposure_usd_cap": 30.0,
        }
    )

    assert limits["max_open_positions"] == 10
    assert limits["max_open_orders"] == 50
    assert limits["max_trade_notional_usd"] == 25.0
    assert limits["max_per_market_exposure_usd"] == 30.0
    assert "max_open_positions" in changes
    assert "max_open_orders" not in changes


def test_live_risk_clamps_normalization_drops_legacy_implicit_defaults():
    assert trader_orchestrator_state._normalize_live_risk_clamps(
        trader_orchestrator_state.LEGACY_IMPLICIT_LIVE_RISK_CLAMPS
    ) == {}


def test_live_risk_clamps_normalization_preserves_explicit_legacy_values():
    assert trader_orchestrator_state._normalize_live_risk_clamps(
        trader_orchestrator_state.LEGACY_IMPLICIT_LIVE_RISK_CLAMPS,
        explicit=True,
    ) == trader_orchestrator_state.LEGACY_IMPLICIT_LIVE_RISK_CLAMPS


def test_live_provider_infra_error_detection_excludes_allowance_rejections():
    assert trader_orchestrator_worker._is_live_provider_infra_error(
        "dial tcp 127.0.0.1:5432: connect: connection refused"
    )
    assert trader_orchestrator_worker._is_live_provider_infra_error("database system is not yet accepting connections")
    assert not trader_orchestrator_worker._is_live_provider_infra_error("not enough balance / allowance")


def test_provider_reconcile_material_change_gate_ignores_non_material_heartbeat_updates():
    assert trader_orchestrator_worker._provider_reconcile_has_material_changes(
        {
            "active_seen": 5,
            "updated_orders": 5,
            "status_changes": 0,
            "session_status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
            "terminal_session_cancels": 0,
        }
    ) is False
    assert trader_orchestrator_worker._provider_reconcile_has_material_changes(
        {
            "active_seen": 5,
            "updated_orders": 5,
            "status_changes": 1,
            "session_status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
            "terminal_session_cancels": 0,
        }
    ) is True


@pytest.mark.asyncio
async def test_live_provider_failure_snapshot_handles_non_sql_session():
    snapshot = await trader_orchestrator_worker._live_provider_failure_snapshot(
        object(),
        trader_id="trader-1",
        window_seconds=120,
    )
    assert snapshot["count"] == 0
    assert snapshot["errors"] == []


def test_query_strategy_types_for_configs_maps_each_source():
    source_configs = {
        "scanner": {
            "source_key": "scanner",
            "strategy_key": "tail_end_carry",
            "strategy_params": {},
        },
        "weather": {
            "source_key": "weather",
            "strategy_key": "weather_distribution",
            "strategy_params": {},
        },
        "news": {
            "source_key": "news",
            "strategy_key": "",
            "strategy_params": {},
        },
    }
    assert trader_orchestrator_worker._query_strategy_types_for_configs(source_configs) == {
        "scanner": ["tail_end_carry"],
        "weather": ["weather_distribution"],
    }


def test_source_open_order_timeout_seconds_prefers_strategy_level_param():
    timeout = trader_orchestrator_worker._source_open_order_timeout_seconds(
        {
            "strategy_params": {
                "max_open_order_seconds": 20,
                "order_ttl_seconds": 1200,
            }
        }
    )
    assert timeout == 20.0


def test_source_open_order_timeout_seconds_honors_explicit_crypto_timeout():
    timeout = trader_orchestrator_worker._source_open_order_timeout_seconds(
        {
            "source_key": "crypto",
            "strategy_key": "btc_eth_maker_quote",
            "strategy_params": {
                "max_open_order_seconds": 8,
            },
        }
    )
    assert timeout == 8.0


@pytest.mark.asyncio
async def test_enforce_source_open_order_timeouts_calls_cleanup_with_source_scoped_filters(monkeypatch):
    cleanup_mock = AsyncMock(
        side_effect=[
            {"matched": 2, "updated": 1},
            {"matched": 1, "updated": 0},
        ]
    )
    monkeypatch.setattr(trader_orchestrator_worker, "cleanup_trader_open_orders", cleanup_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_live_provider_orders",
        AsyncMock(return_value={"provider_ready": True, "updated_orders": 0, "status_changes": 0, "active_seen": 0}),
    )
    trader_orchestrator_worker._live_provider_reconcile_cache.clear()

    try:
        summary = await trader_orchestrator_worker._enforce_source_open_order_timeouts(
            object(),
            trader_id="trader-1",
            run_mode="live",
            source_configs={
                "crypto": {
                    "source_key": "crypto",
                    "strategy_key": "btc_eth_maker_quote",
                    "strategy_params": {"max_open_order_seconds": 20},
                },
                "news": {
                    "source_key": "news",
                    "strategy_key": "news_edge",
                    "strategy_params": {},
                },
                "scanner": {
                    "source_key": "scanner",
                    "strategy_key": "basic",
                    "strategy_params": {"open_order_timeout_seconds": 45},
                },
            },
        )
    finally:
        trader_orchestrator_worker._live_provider_reconcile_cache.clear()

    assert cleanup_mock.await_count == 2
    first_call = cleanup_mock.await_args_list[0].kwargs
    second_call = cleanup_mock.await_args_list[1].kwargs
    assert first_call["scope"] == "live"
    assert first_call["source"] == "crypto"
    assert first_call["max_age_seconds"] == 20.0
    assert first_call["require_unfilled"] is True
    assert first_call["attempt_live_taker_rescue"] is True
    assert first_call["live_taker_rescue_time_in_force"] == "IOC"
    assert second_call["source"] == "scanner"
    assert second_call["max_age_seconds"] == 45.0
    assert second_call["attempt_live_taker_rescue"] is False
    assert summary["configured"] == 2
    assert summary["updated"] == 1
    assert summary["suppressed"] == 0
    assert summary["errors"] == []


@pytest.mark.asyncio
async def test_enforce_source_open_order_timeouts_suppresses_repeated_failures(monkeypatch):
    cleanup_mock = AsyncMock(side_effect=RuntimeError("cleanup_failed"))
    monkeypatch.setattr(trader_orchestrator_worker, "cleanup_trader_open_orders", cleanup_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_live_provider_orders",
        AsyncMock(return_value={"provider_ready": True, "updated_orders": 0, "status_changes": 0, "active_seen": 0}),
    )

    fixed_now = datetime(2026, 2, 23, 19, 26, 51, tzinfo=timezone.utc)
    monkeypatch.setattr(trader_orchestrator_worker, "utcnow", lambda: fixed_now)
    trader_orchestrator_worker._open_order_timeout_cleanup_failure_cooldown_until.clear()
    trader_orchestrator_worker._live_provider_reconcile_cache.clear()
    try:
        first = await trader_orchestrator_worker._enforce_source_open_order_timeouts(
            object(),
            trader_id="trader-1",
            run_mode="live",
            source_configs={
                "crypto": {
                    "source_key": "crypto",
                    "strategy_key": "btc_eth_maker_quote",
                    "strategy_params": {"max_open_order_seconds": 20},
                }
            },
        )
        second = await trader_orchestrator_worker._enforce_source_open_order_timeouts(
            object(),
            trader_id="trader-1",
            run_mode="live",
            source_configs={
                "crypto": {
                    "source_key": "crypto",
                    "strategy_key": "btc_eth_maker_quote",
                    "strategy_params": {"max_open_order_seconds": 20},
                }
            },
        )
    finally:
        trader_orchestrator_worker._open_order_timeout_cleanup_failure_cooldown_until.clear()
        trader_orchestrator_worker._live_provider_reconcile_cache.clear()

    assert cleanup_mock.await_count == 1
    assert first["configured"] == 1
    assert len(first["errors"]) == 1
    assert first["suppressed"] == 0
    assert second["configured"] == 1
    assert second["errors"] == []
    assert second["suppressed"] == 1


@pytest.mark.asyncio
async def test_enforce_source_open_order_timeouts_reuses_recent_provider_reconcile(monkeypatch):
    cleanup_mock = AsyncMock(return_value={"matched": 0, "updated": 0})
    reconcile_mock = AsyncMock(
        return_value={
            "provider_ready": True,
            "active_seen": 5,
            "updated_orders": 5,
            "status_changes": 0,
            "session_status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
            "terminal_session_cancels": 0,
        }
    )
    monkeypatch.setattr(trader_orchestrator_worker, "cleanup_trader_open_orders", cleanup_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_live_provider_orders", reconcile_mock)
    fixed_now = datetime(2026, 2, 23, 19, 26, 51, tzinfo=timezone.utc)
    monkeypatch.setattr(trader_orchestrator_worker, "utcnow", lambda: fixed_now)
    trader_orchestrator_worker._live_provider_reconcile_cache.clear()
    try:
        first = await trader_orchestrator_worker._enforce_source_open_order_timeouts(
            object(),
            trader_id="trader-cache",
            run_mode="live",
            source_configs={
                "scanner": {
                    "source_key": "scanner",
                    "strategy_key": "basic",
                    "strategy_params": {"open_order_timeout_seconds": 45},
                }
            },
        )
        second = await trader_orchestrator_worker._enforce_source_open_order_timeouts(
            object(),
            trader_id="trader-cache",
            run_mode="live",
            source_configs={
                "scanner": {
                    "source_key": "scanner",
                    "strategy_key": "basic",
                    "strategy_params": {"open_order_timeout_seconds": 45},
                }
            },
        )
    finally:
        trader_orchestrator_worker._live_provider_reconcile_cache.clear()

    assert reconcile_mock.await_count == 1
    assert cleanup_mock.await_count == 2
    assert first["provider_reconcile"].get("cache_hit") is not True
    assert second["provider_reconcile"].get("cache_hit") is True


def test_signal_wallets_reads_strategy_context_firehose_wallets():
    signal = SimpleNamespace(
        payload_json={
            "strategy_context": {
                "firehose": {
                    "wallets": [
                        "0x1111111111111111111111111111111111111111",
                        "0x2222222222222222222222222222222222222222",
                    ],
                    "top_wallets": [
                        {"address": "0x3333333333333333333333333333333333333333"},
                    ],
                }
            }
        }
    )

    wallets = trader_orchestrator_worker._signal_wallets(signal)
    assert "0x1111111111111111111111111111111111111111" in wallets
    assert "0x2222222222222222222222222222222222222222" in wallets
    assert "0x3333333333333333333333333333333333333333" in wallets


def test_signal_matches_traders_scope_with_firehose_wallets():
    signal = SimpleNamespace(
        payload_json={
            "strategy_context": {
                "firehose": {
                    "wallets": [
                        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    ]
                }
            }
        }
    )
    scope = {
        "modes": {"pool"},
        "pool_wallets": {"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        "tracked_wallets": set(),
        "individual_wallets": set(),
        "group_wallets": set(),
    }

    matched, payload = trader_orchestrator_worker._signal_matches_traders_scope(signal, scope)
    assert matched is True
    assert payload["matched_modes"] == ["pool"]


@pytest.mark.asyncio
async def test_main_initializes_database_before_worker_loop(monkeypatch):
    call_order: list[str] = []

    async def _fake_init_database() -> None:
        call_order.append("init_database")

    async def _fake_run_loop() -> None:
        call_order.append("run_worker_loop")
        raise asyncio.CancelledError()

    async def _fake_runtime_loop(*, lane: str = "general") -> None:
        del lane
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "init_database",
        _fake_init_database,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "run_runtime_trigger_loop",
        _fake_runtime_loop,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "run_worker_loop",
        _fake_run_loop,
    )

    await trader_orchestrator_worker.main()

    assert call_order == ["init_database", "run_worker_loop"]


@pytest.mark.asyncio
async def test_run_runtime_trigger_loop_dispatches_general_runtime_signals(monkeypatch):
    trigger = {
        "event_type": "runtime_signal_batch",
        "source": "scanner",
        "source_signal_ids": {"scanner": ["signal-1"]},
        "source_signal_snapshots": {"scanner": {"signal-1": {"id": "signal-1"}}},
    }
    wait_mock = AsyncMock(
        side_effect=[
            (trigger, "cursor-1", None),
            asyncio.CancelledError(),
        ]
    )
    build_specs_mock = AsyncMock(
        return_value=(
            {"is_enabled": True, "is_paused": False, "mode": "live"},
            [
                {
                    "trader": {"id": "trader-1"},
                    "process_signals": True,
                    "trigger_signal_ids_by_source": {"scanner": ["signal-1"]},
                    "trigger_signal_snapshots_by_source": {"scanner": {"signal-1": {"id": "signal-1"}}},
                }
            ],
            9.0,
        )
    )
    run_once_mock = AsyncMock(return_value=(1, 1, 1))
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", wait_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_build_runtime_trigger_specs", build_specs_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_runtime_trigger_loop()
    await asyncio.sleep(0)

    assert build_specs_mock.await_count == 1
    assert run_once_mock.await_count == 1
    assert run_once_mock.await_args.kwargs["process_signals"] is True
    assert run_once_mock.await_args.kwargs["trigger_signal_ids_by_source"] == {"scanner": ["signal-1"]}
    assert run_once_mock.await_args.kwargs["timeout_seconds"] == 9.0


@pytest.mark.asyncio
async def test_run_runtime_trigger_loop_continues_draining_batches_while_prior_dispatch_runs(monkeypatch):
    trigger_one = {
        "event_type": "runtime_signal_batch",
        "source": "scanner",
        "source_signal_ids": {"scanner": ["signal-1"]},
        "source_signal_snapshots": {"scanner": {"signal-1": {"id": "signal-1"}}},
    }
    trigger_two = {
        "event_type": "runtime_signal_batch",
        "source": "scanner",
        "source_signal_ids": {"scanner": ["signal-2"]},
        "source_signal_snapshots": {"scanner": {"signal-2": {"id": "signal-2"}}},
    }
    wait_mock = AsyncMock(
        side_effect=[
            (trigger_one, "cursor-1", None),
            (trigger_two, "cursor-2", None),
            asyncio.CancelledError(),
        ]
    )
    build_specs_mock = AsyncMock(
        side_effect=[
            (
                {"is_enabled": True, "is_paused": False, "mode": "live"},
                [
                    {
                        "trader": {"id": "trader-1"},
                        "process_signals": True,
                        "trigger_signal_ids_by_source": {"scanner": ["signal-1"]},
                        "trigger_signal_snapshots_by_source": {"scanner": {"signal-1": {"id": "signal-1"}}},
                    }
                ],
                9.0,
            ),
            (
                {"is_enabled": True, "is_paused": False, "mode": "live"},
                [
                    {
                        "trader": {"id": "trader-1"},
                        "process_signals": True,
                        "trigger_signal_ids_by_source": {"scanner": ["signal-2"]},
                        "trigger_signal_snapshots_by_source": {"scanner": {"signal-2": {"id": "signal-2"}}},
                    }
                ],
                9.0,
            ),
        ]
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def _run_once(*_args, **kwargs):
        signal_ids = kwargs["trigger_signal_ids_by_source"]["scanner"]
        if signal_ids == ["signal-1"]:
            first_started.set()
            await release_first.wait()
        return (1, 0, 1)

    run_once_mock = AsyncMock(side_effect=_run_once)
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", wait_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_build_runtime_trigger_specs", build_specs_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_runtime_trigger_loop()
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.sleep(0)

    assert build_specs_mock.await_count == 2
    assert run_once_mock.await_count == 2


@pytest.mark.asyncio
async def test_pause_until_next_cycle_sleeps_without_runtime_trigger_wait(monkeypatch):
    sleep_mock = AsyncMock(return_value=None)
    wait_mock = AsyncMock(return_value=({"event_type": "runtime_signal_batch"}, "cursor-2", datetime.now(timezone.utc)))
    monkeypatch.setattr(trader_orchestrator_worker, "_worker_sleep", sleep_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", wait_mock)

    result = await trader_orchestrator_worker._pause_until_next_cycle(
        process_runtime_triggers=False,
        sleep_seconds=1.5,
        stream_consumer_name="consumer-1",
        stream_group="orchestrator:general",
        stream_claim_cursor="cursor-1",
        stream_last_claim_run_at=None,
    )

    assert result == (None, "cursor-1", None)
    assert sleep_mock.await_count == 1
    assert wait_mock.await_count == 0


@pytest.mark.asyncio
async def test_run_trader_once_with_timeout_queues_pending_runtime_trigger_when_at_capacity():
    """At capacity (N signal cycles already running), the next runtime
    trigger must queue rather than dispatch."""
    trader_id = "trader-queue"
    existing_tasks: list[asyncio.Task] = []
    bucket: set[asyncio.Task] = set()
    trader_orchestrator_worker._inflight_trader_cycle_tasks[trader_id] = bucket
    capacity = trader_orchestrator_worker._RUNTIME_TRIGGER_MAX_CONCURRENT_PER_TRADER

    async def _hold_open() -> None:
        await asyncio.Future()

    for _ in range(capacity):
        task = asyncio.create_task(_hold_open())
        existing_tasks.append(task)
        bucket.add(task)
        trader_orchestrator_worker._inflight_trader_cycle_process_signals[task] = True
        trader_orchestrator_worker._inflight_trader_cycle_start[task] = 0.0
    trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)

    try:
        result = await trader_orchestrator_worker._run_trader_once_with_timeout(
            {"id": trader_id},
            {"mode": "live"},
            process_signals=True,
            trigger_signal_ids_by_source={"scanner": ["signal-1"]},
            trigger_signal_snapshots_by_source={"scanner": {"signal-1": {"id": "signal-1"}}},
            timeout_seconds=7.0,
        )
    finally:
        trader_orchestrator_worker._inflight_trader_cycle_tasks.pop(trader_id, None)
        for task in existing_tasks:
            trader_orchestrator_worker._inflight_trader_cycle_process_signals.pop(task, None)
            trader_orchestrator_worker._inflight_trader_cycle_start.pop(task, None)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    pending = trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)

    assert result == (0, 0, 0)
    assert pending is not None
    assert pending["trigger_signal_ids_by_source"] == {"scanner": ["signal-1"]}
    assert pending["trigger_signal_snapshots_by_source"]["scanner"]["signal-1"]["id"] == "signal-1"
    assert pending["process_signals"] is True
    assert pending["timeout_seconds"] == 7.0


@pytest.mark.asyncio
async def test_run_trader_once_with_timeout_queues_signal_cycle_when_trader_already_running(monkeypatch):
    """With one signal cycle already running, the next runtime trigger is coalesced."""
    trader_id = "trader-concurrent"
    bucket: set[asyncio.Task] = set()
    trader_orchestrator_worker._inflight_trader_cycle_tasks[trader_id] = bucket

    async def _hold_open() -> None:
        await asyncio.Future()

    incumbent = asyncio.create_task(_hold_open())
    bucket.add(incumbent)
    trader_orchestrator_worker._inflight_trader_cycle_process_signals[incumbent] = True
    trader_orchestrator_worker._inflight_trader_cycle_start[incumbent] = 0.0
    trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)

    run_once_mock = AsyncMock(return_value=(1, 2, 3))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once", run_once_mock)

    try:
        result = await trader_orchestrator_worker._run_trader_once_with_timeout(
            {"id": trader_id},
            {"mode": "shadow"},
            process_signals=True,
            trigger_signal_ids_by_source={"scanner": ["signal-1"]},
            trigger_signal_snapshots_by_source={"scanner": {"signal-1": {"id": "signal-1"}}},
            timeout_seconds=7.0,
        )
    finally:
        trader_orchestrator_worker._inflight_trader_cycle_tasks.pop(trader_id, None)
        trader_orchestrator_worker._inflight_trader_cycle_process_signals.pop(incumbent, None)
        trader_orchestrator_worker._inflight_trader_cycle_start.pop(incumbent, None)
        incumbent.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await incumbent

    pending = trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)

    assert result == (0, 0, 0)
    assert pending is not None
    assert pending["trigger_signal_ids_by_source"] == {"scanner": ["signal-1"]}
    assert pending["process_signals"] is True
    run_once_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_with_timeout_preempts_inflight_maintenance_for_runtime_trigger(monkeypatch):
    trader_id = "trader-preempt"
    cancelled = asyncio.Event()

    async def _maintenance_only_cycle():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    existing = asyncio.create_task(_maintenance_only_cycle())
    await asyncio.sleep(0)
    trader_orchestrator_worker._inflight_trader_cycle_tasks[trader_id] = {existing}
    trader_orchestrator_worker._inflight_trader_cycle_process_signals[existing] = False
    trader_orchestrator_worker._inflight_trader_cycle_start[existing] = 0.0
    trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)
    run_once_mock = AsyncMock(return_value=(2, 3, 4))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once", run_once_mock)

    try:
        result = await trader_orchestrator_worker._run_trader_once_with_timeout(
            {"id": trader_id},
            {"mode": "live"},
            process_signals=True,
            trigger_signal_ids_by_source={"scanner": ["signal-1"]},
            trigger_signal_snapshots_by_source={"scanner": {"signal-1": {"id": "signal-1"}}},
            timeout_seconds=7.0,
        )
    finally:
        trader_orchestrator_worker._inflight_trader_cycle_tasks.pop(trader_id, None)
        trader_orchestrator_worker._inflight_trader_cycle_process_signals.pop(existing, None)
        trader_orchestrator_worker._inflight_trader_cycle_start.pop(existing, None)
        trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)
        if not existing.done():
            existing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await existing

    assert cancelled.is_set()
    assert result == (2, 3, 4)
    assert trader_orchestrator_worker._pending_runtime_cycle_specs.get(trader_id) is None
    run_once_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_trader_once_with_timeout_cancels_live_signal_cycle_after_soft_timeout(monkeypatch):
    trader_id = "trader-soft-timeout"
    created_events: list[dict[str, object]] = []

    async def _slow_run_once(*args, **kwargs):
        await asyncio.Future()
        return 5, 6, 7

    async def _buffer_event(**kwargs):
        created_events.append(kwargs)
        return "event-1"

    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once", _slow_run_once)
    monkeypatch.setattr(trader_orchestrator_worker, "_MIN_LIVE_PROCESS_SIGNAL_CYCLE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_trader_event", _buffer_event)

    trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)
    trader_orchestrator_worker._inflight_trader_cycle_tasks.pop(trader_id, None)

    try:
        result = await trader_orchestrator_worker._run_trader_once_with_timeout(
            {"id": trader_id},
            {"mode": "live"},
            process_signals=True,
            trigger_signal_ids_by_source={"scanner": ["signal-1"]},
            trigger_signal_snapshots_by_source={"scanner": {"signal-1": {"id": "signal-1"}}},
            timeout_seconds=0.01,
        )
        await asyncio.sleep(0)

        assert result == (0, 0, 0)
        assert trader_orchestrator_worker._inflight_trader_cycle_tasks.get(trader_id) is None
        assert created_events
        assert created_events[-1]["event_type"] == "cycle_timeout"
    finally:
        trader_orchestrator_worker._pending_runtime_cycle_specs.pop(trader_id, None)
        bucket = trader_orchestrator_worker._inflight_trader_cycle_tasks.pop(trader_id, None)
        live_task = next(iter(bucket)) if bucket else None
        if live_task is not None:
            trader_orchestrator_worker._inflight_trader_cycle_process_signals.pop(live_task, None)
            trader_orchestrator_worker._inflight_trader_cycle_start.pop(live_task, None)
        if live_task is not None and not live_task.done():
            live_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await live_task

    assert trader_orchestrator_worker._inflight_trader_cycle_tasks.get(trader_id) is None


@pytest.mark.asyncio
async def test_launch_pending_runtime_cycle_dispatches_queued_trigger(monkeypatch):
    trader_id = "trader-dispatch"
    run_once_mock = AsyncMock(return_value=(1, 1, 1))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)
    trader_orchestrator_worker._pending_runtime_cycle_specs[trader_id] = {
        "trader": {"id": trader_id},
        "control": {"mode": "live"},
        "process_signals": True,
        "trigger_signal_ids_by_source": {"scanner": ["signal-2"]},
        "trigger_signal_snapshots_by_source": {"scanner": {"signal-2": {"id": "signal-2"}}},
        "timeout_seconds": 4.0,
    }

    await trader_orchestrator_worker._launch_pending_runtime_cycle_if_any(trader_id)

    assert run_once_mock.await_count == 1
    assert run_once_mock.await_args.args[0] == {"id": trader_id}
    assert run_once_mock.await_args.kwargs["process_signals"] is True
    assert run_once_mock.await_args.kwargs["trigger_signal_ids_by_source"] == {"scanner": ["signal-2"]}


class _DummySession:
    class _NoAutoflush:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _ExecuteResult:
        rowcount = 1

        def scalar_one_or_none(self):
            return None

        def scalar_one(self):
            return 0

        def scalars(self):
            return self

        def first(self):
            return None

        def all(self):
            return []

    def __init__(self):
        self.new = []
        self.no_autoflush = self._NoAutoflush()

    def add(self, row):
        self.new.append(row)

    async def get(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        del args, kwargs
        return self._ExecuteResult()

    async def flush(self):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None

    def in_transaction(self):
        return False


class _DummySessionContext:
    async def __aenter__(self):
        return _DummySession()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SelectedStrategy:
    key = "btc_eth_maker_quote"

    def evaluate(self, signal, context):
        return StrategyDecision(
            decision="selected",
            reason="selected",
            score=10.0,
            size_usd=25.0,
            checks=[],
            payload={},
        )


class _SkippedStrategy:
    key = "btc_eth_maker_quote"

    def evaluate(self, signal, context):
        return StrategyDecision(
            decision="skipped",
            reason="strategy veto",
            score=1.0,
            size_usd=0.0,
            checks=[],
            payload={},
        )


def _mock_resolve_strategy_version(strategy_key):
    """Return a SimpleNamespace matching ResolvedStrategyVersion shape."""
    return SimpleNamespace(
        strategy=SimpleNamespace(slug=strategy_key, id="strat-id-" + strategy_key),
        version_row=SimpleNamespace(version=1, source_code="", config={}),
        latest_version=1,
        requested_version=None,
    )


def _base_trader_payload(*, allow_averaging: bool) -> dict:
    return {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "btc_eth_maker_quote",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
                    "enforce_directional_timeframe": False,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": False,
                },
            }
        ],
        "risk_limits": {"allow_averaging": allow_averaging},
        "metadata": {"resume_policy": "resume_full"},
    }


def _base_control_payload() -> dict:
    return {
        "mode": "live",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": False, "strict_ws_pricing_only": False},
            },
        },
    }


def _base_signal() -> SimpleNamespace:
    return SimpleNamespace(
        id="signal-1",
        created_at=datetime.utcnow(),
        source="crypto",
        signal_type="crypto_worker_multistrat",
        strategy_type="btc_eth_maker_quote",
        market_id="market-1",
        market_question="Will BTC close higher?",
        direction="buy_yes",
        entry_price=0.4,
        edge_percent=8.0,
        confidence=0.72,
        payload_json={},
    )


@pytest.mark.asyncio
async def test_run_trader_once_prefilters_mismatched_source_strategy_type(monkeypatch):
    signal = SimpleNamespace(
        id="signal-mismatch",
        created_at=datetime.utcnow(),
        source="scanner",
        signal_type="scanner_opportunity",
        strategy_type="stat_arb",
        market_id="scanner-market-1",
        market_question="Will event happen?",
        direction="buy_no",
        entry_price=0.88,
        edge_percent=4.0,
        confidence=0.5,
        payload_json={},
    )
    list_calls = {"count": 0}
    list_kwargs: list[dict] = []
    consumptions: list[dict] = []

    async def _list_unconsumed(*args, **kwargs):
        list_kwargs.append(dict(kwargs))
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _record_consumption(_session, **kwargs):
        consumptions.append(kwargs)
        return None

    async def _buffer_consumption(**kwargs):
        consumptions.append(kwargs)

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    create_decision_mock = AsyncMock(return_value=SimpleNamespace(id="decision-1"))

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "tail_end_carry",
                "strategy_params": {
                    "max_signals_per_cycle": 2,
                    "scan_batch_size": 2,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": False, "strict_ws_pricing_only": False},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "cleanup_trader_open_orders",
        AsyncMock(return_value={"matched": 0, "updated": 0}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", create_decision_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_signal_consumption", _buffer_consumption)

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 1
    create_decision_mock.assert_not_awaited()
    assert any(c.get("signal_id") == "signal-mismatch" and c.get("outcome") == "skipped" for c in consumptions)
    assert any("source strategy filter" in str(c.get("reason", "")) for c in consumptions)
    assert all(
        call_kwargs.get("strategy_types_by_source") == {"scanner": ["tail_end_carry"]}
        for call_kwargs in list_kwargs
        if "strategy_types_by_source" in call_kwargs
    )


@pytest.mark.asyncio
async def test_run_trader_once_emits_filtered_heartbeat_for_crypto_scope_prefilter(monkeypatch):
    signal = SimpleNamespace(
        id="signal-scope-filter",
        created_at=datetime.utcnow(),
        source="crypto",
        signal_type="crypto_opportunity",
        strategy_type="btc_eth_maker_quote",
        market_id="crypto-market-1",
        market_question="Solana Up or Down",
        direction="buy_yes",
        entry_price=0.62,
        edge_percent=3.1,
        confidence=0.6,
        payload_json={"asset": "SOL", "timeframe": "5m"},
        strategy_context_json={"asset": "SOL", "timeframe": "5m"},
    )
    list_calls = {"count": 0}
    consumptions: list[dict] = []

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _record_consumption(_session, **kwargs):
        consumptions.append(kwargs)
        return None

    async def _buffer_consumption(**kwargs):
        consumptions.append(kwargs)

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    create_decision_mock = AsyncMock(return_value=SimpleNamespace(id="decision-1"))
    create_event_mock = AsyncMock(return_value=None)

    trader_payload = {
        "id": "trader-crypto-scope",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "btc_eth_maker_quote",
                "strategy_params": {
                    "max_signals_per_cycle": 2,
                    "scan_batch_size": 2,
                    "include_assets": ["BTC", "ETH"],
                    "exclude_assets": ["SOL", "XRP"],
                    "include_timeframes": ["5m", "15m", "1h", "4h"],
                },
            }
        ],
        "risk_limits": {"allow_averaging": False},
        "metadata": {"resume_policy": "resume_full"},
    }
    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": False, "strict_ws_pricing_only": False},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", create_decision_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_signal_consumption", _buffer_consumption)

    trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()
    try:
        decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
            trader_payload,
            control_payload,
        )
    finally:
        trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 1
    create_decision_mock.assert_not_awaited()
    assert any(c.get("signal_id") == "signal-scope-filter" and c.get("outcome") == "skipped" for c in consumptions)
    heartbeat_calls = [
        call
        for call in create_event_mock.await_args_list
        if call.kwargs.get("event_type") == "cycle_heartbeat"
    ]
    assert len(heartbeat_calls) == 1
    heartbeat_call = heartbeat_calls[0]
    assert heartbeat_call.kwargs["message"] == "Idle cycle: pending signals filtered before strategy evaluation."
    payload = heartbeat_call.kwargs["payload"]
    assert payload["prefiltered_signals"] == 1
    assert payload["prefiltered_by_reason"] == {"crypto_scope_filter": 1}
    assert payload["crypto_scope_prefiltered_dimensions"] == {"SOL:5m": 1}


@pytest.mark.asyncio
async def test_persist_trader_cycle_heartbeat_updates_last_run_and_clears_request(monkeypatch):
    class _Result:
        rowcount = 1

    class _Session:
        async def execute(self, *_args, **_kwargs):
            return _Result()

    commit_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(trader_orchestrator_worker, "_commit_with_retry", commit_mock)

    await trader_orchestrator_worker._persist_trader_cycle_heartbeat(_Session(), "trader-1")

    commit_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_trader_once_persists_heartbeat_when_idle_gate_short_circuits(monkeypatch):
    trader_id = "trader-idle"

    commit_mock = AsyncMock(return_value=None)
    create_event_mock = AsyncMock(return_value=None)
    backfill_mock = AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []})
    reconcile_mock = AsyncMock(
        return_value={
            "matched": 0,
            "closed": 0,
            "held": 0,
            "skipped": 0,
            "total_realized_pnl": 0.0,
            "by_status": {},
        }
    )
    sync_mock = AsyncMock(return_value={})
    open_positions_mock = AsyncMock(return_value=0)
    open_markets_mock = AsyncMock(return_value=set())
    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_commit_with_retry", commit_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        backfill_mock,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_shadow_positions", reconcile_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", sync_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", open_positions_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", open_markets_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", AsyncMock(return_value=[]))

    trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()
    trader_orchestrator_worker._trader_idle_maintenance_last_run[trader_id] = datetime.now(timezone.utc)
    try:
        decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
            {
                "id": trader_id,
                "source_configs": [
                    {
                        "source_key": "weather",
                        "strategy_key": "weather_primary",
                        "strategy_params": {},
                    }
                ],
                "risk_limits": {},
                "metadata": {"resume_policy": "resume_full"},
            },
            {"mode": "shadow", "settings": {}},
            process_signals=True,
        )
    finally:
        trader_orchestrator_worker._trader_idle_maintenance_last_run.pop(trader_id, None)
        trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()

    assert decisions_written == 0
    assert orders_written == 0
    create_event_mock.assert_awaited_once()
    assert create_event_mock.await_args.kwargs["event_type"] == "cycle_heartbeat"
    backfill_mock.assert_awaited_once()
    reconcile_mock.assert_awaited_once()
    sync_mock.assert_awaited_once()
    open_positions_mock.assert_not_awaited()
    open_markets_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_runs_live_execution_session_maintenance(monkeypatch):
    trader_id = "trader-live-maintenance"

    commit_mock = AsyncMock(return_value=None)
    create_event_mock = AsyncMock(return_value=None)
    backfill_mock = AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []})
    shadow_reconcile_mock = AsyncMock(
        return_value={
            "matched": 0,
            "closed": 0,
            "held": 0,
            "skipped": 0,
            "total_realized_pnl": 0.0,
            "by_status": {},
        }
    )
    sync_mock = AsyncMock(return_value={})
    open_positions_mock = AsyncMock(return_value=0)
    open_markets_mock = AsyncMock(return_value=set())
    reconcile_calls: list[dict[str, object]] = []

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        del self
        reconcile_calls.append({"mode": mode, "trader_id": trader_id})
        return {"active_seen": 2, "expired": 1, "completed": 0, "failed": 1}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_commit_with_retry", commit_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        backfill_mock,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_shadow_positions", shadow_reconcile_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", sync_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", open_positions_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", open_markets_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )

    trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()
    trader_orchestrator_worker._trader_idle_maintenance_last_run[trader_id] = datetime.now(timezone.utc)
    try:
        decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
            {
                "id": trader_id,
                "source_configs": [
                    {
                        "source_key": "scanner",
                        "strategy_key": "basic",
                        "strategy_params": {},
                    }
                ],
                "risk_limits": {},
                "metadata": {"resume_policy": "resume_full"},
            },
            {"mode": "live", "settings": {}},
            process_signals=True,
        )
    finally:
        trader_orchestrator_worker._trader_idle_maintenance_last_run.pop(trader_id, None)
        trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()

    assert decisions_written == 0
    assert orders_written == 0
    assert reconcile_calls == [{"mode": "live", "trader_id": trader_id}]
    backfill_mock.assert_not_awaited()
    shadow_reconcile_mock.assert_not_awaited()
    sync_mock.assert_not_awaited()
    open_positions_mock.assert_not_awaited()
    open_markets_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_persists_heartbeat_when_signal_queue_is_empty(monkeypatch):
    heartbeat_mock = AsyncMock(return_value=None)
    persist_mock = AsyncMock(return_value=None)
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return []

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_trader_idle_maintenance_last_run", {})
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        "services.trader_orchestrator.position_lifecycle.reconcile_live_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_sequence_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_pending_live_exit_summary_for_trader", AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_unrealized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_live_provider_failure_snapshot", AsyncMock(return_value={"count": 0, "errors": []}))
    monkeypatch.setattr(trader_orchestrator_worker, "_live_risk_clamp_event_should_emit", AsyncMock(return_value=False))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        AsyncMock(return_value={"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_enforce_source_open_order_timeouts",
        AsyncMock(
            return_value={
                "configured": 0,
                "updated": 0,
                "suppressed": 0,
                "taker_rescue_attempted": 0,
                "taker_rescue_succeeded": 0,
                "taker_rescue_failed": 0,
                "sources": [],
                "errors": [],
                "provider_reconcile": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", heartbeat_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", persist_mock)

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=True),
        _base_control_payload(),
    )

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    assert list_calls["count"] >= 2
    heartbeat_mock.assert_awaited()
    assert heartbeat_mock.await_args.kwargs["message"] == "Idle cycle: no pending signals."
    persist_mock.assert_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_reconciles_positions_when_source_configs_missing(monkeypatch):
    trader_id = "trader-no-config"

    commit_mock = AsyncMock(return_value=None)
    create_event_mock = AsyncMock(return_value=None)
    list_signals_mock = AsyncMock(return_value=[])
    backfill_mock = AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []})
    reconcile_mock = AsyncMock(
        return_value={
            "matched": 0,
            "closed": 0,
            "held": 0,
            "skipped": 0,
            "total_realized_pnl": 0.0,
            "by_status": {},
        }
    )
    sync_mock = AsyncMock(return_value={})
    open_positions_mock = AsyncMock(return_value=0)
    open_markets_mock = AsyncMock(return_value=set())

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_commit_with_retry", commit_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        backfill_mock,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_shadow_positions", reconcile_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", sync_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", open_positions_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", open_markets_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", list_signals_mock)

    trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()
    try:
        decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
            {
                "id": trader_id,
                "source_configs": [],
                "risk_limits": {},
                "metadata": {"resume_policy": "resume_full"},
            },
            {"mode": "shadow", "settings": {}},
            process_signals=True,
        )
    finally:
        trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()

    assert decisions_written == 0
    assert orders_written == 0
    create_event_mock.assert_awaited_once()
    assert create_event_mock.await_args.kwargs["event_type"] == "cycle_heartbeat"
    backfill_mock.assert_awaited_once()
    reconcile_mock.assert_awaited_once()
    assert reconcile_mock.await_args.kwargs["trader_params"] == {}
    sync_mock.assert_awaited_once()
    open_positions_mock.assert_not_awaited()
    open_markets_mock.assert_not_awaited()
    list_signals_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_skips_heavy_maintenance_when_manage_only_cycle_is_not_due(monkeypatch):
    trader_id = "trader-manage-only"

    commit_mock = AsyncMock(return_value=None)
    create_event_mock = AsyncMock(return_value=None)
    backfill_mock = AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []})
    reconcile_mock = AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}})
    sync_mock = AsyncMock(return_value={})
    cursor_mock = AsyncMock(return_value=(None, None))
    list_signals_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_commit_with_retry", commit_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        backfill_mock,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_shadow_positions", reconcile_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", sync_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", cursor_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", list_signals_mock)

    trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()
    trader_orchestrator_worker._trader_maintenance_last_run[trader_id] = datetime.now(timezone.utc)
    try:
        decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
            {
                "id": trader_id,
                "source_configs": [
                    {
                        "source_key": "weather",
                        "strategy_key": "weather_primary",
                        "strategy_params": {},
                    }
                ],
                "risk_limits": {},
                "metadata": {"resume_policy": "resume_full"},
            },
            {"mode": "shadow", "settings": {}},
            process_signals=False,
        )
    finally:
        trader_orchestrator_worker._trader_maintenance_last_run.pop(trader_id, None)
        trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    create_event_mock.assert_not_awaited()
    backfill_mock.assert_not_awaited()
    reconcile_mock.assert_not_awaited()
    sync_mock.assert_not_awaited()
    cursor_mock.assert_not_awaited()
    list_signals_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_reconciles_live_sessions_on_runtime_trigger_cycle(monkeypatch):
    trader_id = "trader-runtime-hot"

    class _Session:
        async def get(self, *_args, **_kwargs):
            return None

        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def execute(self, *_args, **_kwargs):
            return None

        def in_transaction(self):
            return False

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    reconcile_sessions_mock = AsyncMock(return_value={"active_seen": 0, "expired": 0, "completed": 0, "failed": 0})
    timeout_cleanup_mock = AsyncMock(
        return_value={
            "configured": 0,
            "updated": 0,
            "suppressed": 0,
            "taker_rescue_attempted": 0,
            "taker_rescue_succeeded": 0,
            "taker_rescue_failed": 0,
            "sources": [],
            "errors": [],
            "provider_reconcile": {},
        }
    )

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_build_triggered_trade_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_sequence_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_unrealized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_live_provider_failure_snapshot", AsyncMock(return_value={"count": 0, "errors": []}))
    monkeypatch.setattr(trader_orchestrator_worker, "_live_risk_clamp_event_should_emit", AsyncMock(return_value=False))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_enforce_source_open_order_timeouts", timeout_cleanup_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        reconcile_sessions_mock,
    )

    trader_payload = dict(_base_trader_payload(allow_averaging=True))
    trader_payload["id"] = trader_id

    try:
        decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
            trader_payload,
            _base_control_payload(),
            trigger_signal_ids_by_source={"crypto": ["signal-1"]},
            trigger_signal_snapshots_by_source={"crypto": {}},
        )
    finally:
        trader_orchestrator_worker._execution_session_maintenance_last_run.pop(trader_id, None)

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    reconcile_sessions_mock.assert_awaited_once()
    timeout_cleanup_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_skips_live_maintenance_on_scheduled_cycle(monkeypatch):
    trader_id = "trader-live-scheduled"

    class _Session:
        async def get(self, *_args, **_kwargs):
            return None

        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def execute(self, *_args, **_kwargs):
            return None

        def in_transaction(self):
            return False

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    reconcile_sessions_mock = AsyncMock(return_value={"active_seen": 0, "expired": 0, "completed": 0, "failed": 0})
    timeout_cleanup_mock = AsyncMock(
        return_value={
            "configured": 0,
            "updated": 0,
            "suppressed": 0,
            "taker_rescue_attempted": 0,
            "taker_rescue_succeeded": 0,
            "taker_rescue_failed": 0,
            "sources": [],
            "errors": [],
            "provider_reconcile": {},
        }
    )

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_unrealized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_live_provider_failure_snapshot", AsyncMock(return_value={"count": 0, "errors": []}))
    monkeypatch.setattr(trader_orchestrator_worker, "_live_risk_clamp_event_should_emit", AsyncMock(return_value=False))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_enforce_source_open_order_timeouts", timeout_cleanup_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        reconcile_sessions_mock,
    )

    trader_payload = dict(_base_trader_payload(allow_averaging=True))
    trader_payload["id"] = trader_id
    trader_orchestrator_worker._trader_maintenance_last_run[trader_id] = datetime.now(timezone.utc)
    trader_orchestrator_worker._execution_session_maintenance_last_run[trader_id] = datetime.now(timezone.utc)

    try:
        decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
            trader_payload,
            _base_control_payload(),
            process_signals=False,
        )
    finally:
        trader_orchestrator_worker._trader_maintenance_last_run.pop(trader_id, None)
        trader_orchestrator_worker._execution_session_maintenance_last_run.pop(trader_id, None)

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    reconcile_sessions_mock.assert_not_awaited()
    timeout_cleanup_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_emit_cycle_heartbeat_is_throttled(monkeypatch):
    create_event_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)

    trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()
    try:
        await trader_orchestrator_worker._emit_cycle_heartbeat_if_due(
            object(),
            trader_id="trader-1",
            message="Idle cycle",
            payload={},
        )
        await trader_orchestrator_worker._emit_cycle_heartbeat_if_due(
            object(),
            trader_id="trader-1",
            message="Idle cycle",
            payload={},
        )
    finally:
        trader_orchestrator_worker._trader_cycle_heartbeat_last_emitted.clear()

    create_event_mock.assert_awaited_once()
    assert create_event_mock.await_args.kwargs["event_type"] == "cycle_heartbeat"


def test_is_terminal_market_state_detects_closed_winner_and_settled_prices(monkeypatch):
    now = datetime.now(timezone.utc)
    assert trader_orchestrator_worker._is_terminal_market_state({"closed": True}, now=now) is True
    assert trader_orchestrator_worker._is_terminal_market_state({"winner": "Yes"}, now=now) is True

    monkeypatch.setattr(
        trader_orchestrator_worker.polymarket_client, "is_market_tradable", lambda *_args, **_kwargs: False
    )
    assert (
        trader_orchestrator_worker._is_terminal_market_state(
            {"closed": False, "outcome_prices": [1.0, 0.0]},
            now=now,
        )
        is True
    )
    assert (
        trader_orchestrator_worker._is_terminal_market_state(
            {"closed": False, "outcome_prices": [0.61, 0.39]},
            now=now,
        )
        is False
    )


@pytest.mark.asyncio
async def test_terminal_stale_order_watchdog_respects_alert_cooldown(monkeypatch):
    now = datetime.now(timezone.utc)
    stale_order = SimpleNamespace(
        id="order-1",
        trader_id="trader-1",
        mode="shadow",
        status="executed",
        market_id="market-1",
        executed_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        created_at=now - timedelta(minutes=12),
    )

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return [stale_order]

    class _Session:
        async def execute(self, *_args, **_kwargs):
            return _Result()

        async def commit(self):
            return None

    create_event_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", create_event_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "load_market_info_for_orders",
        AsyncMock(return_value={"market-1": {"closed": True, "outcome_prices": [1.0, 0.0]}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.polymarket_client, "is_market_tradable", lambda *_args, **_kwargs: False
    )

    trader_orchestrator_worker._terminal_stale_order_last_checked_at = None
    trader_orchestrator_worker._terminal_stale_order_alert_last_emitted.clear()
    try:
        first = await trader_orchestrator_worker._run_terminal_stale_order_watchdog(_Session(), now=now)
        second = await trader_orchestrator_worker._run_terminal_stale_order_watchdog(
            _Session(),
            now=now + timedelta(seconds=31),
        )
    finally:
        trader_orchestrator_worker._terminal_stale_order_last_checked_at = None
        trader_orchestrator_worker._terminal_stale_order_alert_last_emitted.clear()

    assert first["checked"] is True
    assert first["stale"] == 1
    assert first["alerted"] == 1
    assert second["checked"] is True
    assert second["stale"] == 1
    assert second["alerted"] == 0
    create_event_mock.assert_awaited_once()
    assert create_event_mock.await_args.kwargs["event_type"] == "terminal_stale_orders"


@pytest.mark.asyncio
async def test_run_worker_loop_skips_terminal_stale_watchdog_when_globally_disabled(monkeypatch):
    class _Session:
        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    stale_watchdog_mock = AsyncMock(return_value={})
    snapshot_mock = AsyncMock(return_value=None)

    async def _cancel_wait(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", stale_watchdog_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": False,
                "is_paused": True,
                "kill_switch": False,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "list_traders", AsyncMock(return_value=[]))
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", snapshot_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", _cancel_wait)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop()

    stale_watchdog_mock.assert_not_awaited()
    snapshot_mock.assert_awaited_once()
    assert snapshot_mock.await_args.kwargs["current_activity"] == "Disabled"


@pytest.mark.asyncio
async def test_run_worker_loop_releases_session_before_hot_state_maintenance(monkeypatch):
    """Hot-state maintenance (flush + reseed) must run outside any DB session."""

    class _SessionContext:
        async def __aenter__(self):
            return _DummySession()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    maintenance_events: list[str] = []
    snapshot_mock = AsyncMock(return_value=None)

    async def _flush_audit_buffer():
        maintenance_events.append("flush")
        return 0

    async def _reseed_if_stale():
        maintenance_events.append("reseed")
        return False

    async def _cancel_wait(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": True,
                "kill_switch": False,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {},
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_traders",
        AsyncMock(return_value=[{"id": "t-1", "is_enabled": True, "is_paused": False, "metadata": {}}]),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", AsyncMock(return_value=(0, 0, 0)))
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", snapshot_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", _cancel_wait)
    monkeypatch.setattr(
        trader_orchestrator_worker.hot_state,
        "flush_audit_buffer",
        AsyncMock(side_effect=_flush_audit_buffer),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.hot_state,
        "reseed_if_stale",
        AsyncMock(side_effect=_reseed_if_stale),
    )

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop()

    assert maintenance_events == ["flush", "reseed"]


@pytest.mark.asyncio
async def test_run_worker_loop_runs_manage_only_cycle_when_globally_paused(monkeypatch):
    class _Session:
        async def get(self, model, key):
            if getattr(model, "__name__", "") == "SimulationAccount" and key == "shadow-1":
                return object()
            return None

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    run_once_mock = AsyncMock(return_value=(0, 0, 0))
    snapshot_mock = AsyncMock(return_value=None)

    async def _cancel_sleep(_interval: float):
        raise asyncio.CancelledError()

    async def _cancel_wait(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": True,
                "kill_switch": False,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {"shadow_account_id": "shadow-1"},
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_traders",
        AsyncMock(return_value=[{"id": "trader-1", "is_enabled": True, "is_paused": False, "metadata": {}}]),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", snapshot_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "update_orchestrator_control", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_worker_sleep", _cancel_sleep)
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", _cancel_wait)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop()

    run_once_mock.assert_awaited_once()
    assert run_once_mock.await_args.kwargs["process_signals"] is False
    assert snapshot_mock.await_args.kwargs["current_activity"].startswith("Manage-only[")
    assert "(global_pause)" in snapshot_mock.await_args.kwargs["current_activity"]


@pytest.mark.asyncio
async def test_run_worker_loop_dispatches_crypto_traders_concurrently_with_non_crypto(monkeypatch):
    class _Session:
        async def get(self, model, key):
            if getattr(model, "__name__", "") == "SimulationAccount" and key == "shadow-1":
                return object()
            return None

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    call_order: list[str] = []

    async def _run_once(trader, control, **kwargs):
        del control, kwargs
        trader_id = str(trader.get("id"))
        call_order.append(trader_id)
        return 1, 0, 1 if trader_id == "crypto-1" else (0, 0, 0)

    async def _cancel_wait(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner_for_lane", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner_for_lane", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": False,
                "kill_switch": False,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {"shadow_account_id": "shadow-1"},
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_traders",
        AsyncMock(
            return_value=[
                {
                    "id": "crypto-1",
                    "is_enabled": True,
                    "is_paused": False,
                    "mode": "shadow",
                    "metadata": {},
                    "source_configs": [{"source_key": "crypto", "strategy_key": "generic_crypto", "strategy_params": {}}],
                },
                {
                    "id": "weather-1",
                    "is_enabled": True,
                    "is_paused": False,
                    "mode": "shadow",
                    "metadata": {},
                    "source_configs": [{"source_key": "weather", "strategy_key": "weather_edge", "strategy_params": {}}],
                },
            ]
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", AsyncMock(side_effect=_run_once))
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "update_orchestrator_control", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", _cancel_wait)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop(lane="crypto", write_snapshot=False)

    assert call_order == ["crypto-1"]


@pytest.mark.asyncio
async def test_run_worker_loop_processes_scanner_signals_on_scheduled_general_lane(monkeypatch):
    class _Session:
        async def get(self, model, key):
            if getattr(model, "__name__", "") == "SimulationAccount" and key == "shadow-1":
                return object()
            return None

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    run_once_mock = AsyncMock(return_value=(1, 0, 1))

    async def _cancel_sleep(_interval: float):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": False,
                "kill_switch": False,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {"shadow_account_id": "shadow-1"},
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_traders",
        AsyncMock(
            return_value=[
                {
                    "id": "scanner-1",
                    "is_enabled": True,
                    "is_paused": False,
                    "mode": "shadow",
                    "metadata": {},
                    "source_configs": [{"source_key": "scanner", "strategy_key": "custom_scanner_strategy"}],
                }
            ]
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "update_orchestrator_control", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_worker_sleep", _cancel_sleep)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop(process_runtime_triggers=False)

    run_once_mock.assert_awaited_once()
    assert run_once_mock.await_args.kwargs["process_signals"] is True
    assert run_once_mock.await_args.kwargs["trigger_signal_ids_by_source"] is None


@pytest.mark.asyncio
async def test_run_worker_loop_skips_scheduled_signals_when_disabled(monkeypatch):
    class _Session:
        async def get(self, model, key):
            if getattr(model, "__name__", "") == "SimulationAccount" and key == "shadow-1":
                return object()
            return None

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    run_once_mock = AsyncMock(return_value=(0, 0, 0))

    async def _cancel_sleep(_interval: float):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": False,
                "kill_switch": False,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {"shadow_account_id": "shadow-1"},
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_traders",
        AsyncMock(
            return_value=[
                {
                    "id": "scanner-1",
                    "is_enabled": True,
                    "is_paused": False,
                    "mode": "shadow",
                    "metadata": {},
                    "source_configs": [{"source_key": "scanner", "strategy_key": "custom_scanner_strategy"}],
                }
            ]
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "update_orchestrator_control", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_worker_sleep", _cancel_sleep)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop(
            process_runtime_triggers=False,
            process_scheduled_signals=False,
        )

    run_once_mock.assert_awaited_once()
    assert run_once_mock.await_args.kwargs["process_signals"] is False
    assert run_once_mock.await_args.kwargs["trigger_signal_ids_by_source"] is None


@pytest.mark.asyncio
async def test_run_worker_loop_runs_manage_only_cycle_when_kill_switch_enabled(monkeypatch):
    class _Session:
        async def get(self, model, key):
            if getattr(model, "__name__", "") == "SimulationAccount" and key == "shadow-1":
                return object()
            return None

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    run_once_mock = AsyncMock(return_value=(0, 0, 0))
    snapshot_mock = AsyncMock(return_value=None)

    async def _cancel_sleep(_interval: float):
        raise asyncio.CancelledError()

    async def _cancel_wait(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        trader_orchestrator_worker, "_ensure_orchestrator_cycle_lock_owner", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        trader_orchestrator_worker, "_release_orchestrator_cycle_lock_owner", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_reconcile_orphan_open_orders", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_run_terminal_stale_order_watchdog", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": False,
                "kill_switch": True,
                "run_interval_seconds": 1,
                "mode": "shadow",
                "settings": {"shadow_account_id": "shadow-1"},
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_traders",
        AsyncMock(return_value=[{"id": "trader-1", "is_enabled": True, "is_paused": False, "metadata": {}}]),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "_run_trader_once_with_timeout", run_once_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_build_orchestrator_snapshot_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_write_orchestrator_snapshot_best_effort", snapshot_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "update_orchestrator_control", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "_worker_sleep", _cancel_sleep)
    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", _cancel_wait)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop()

    run_once_mock.assert_awaited_once()
    assert run_once_mock.await_args.kwargs["process_signals"] is False
    assert snapshot_mock.await_args.kwargs["current_activity"].startswith("Manage-only[")
    assert "(kill_switch)" in snapshot_mock.await_args.kwargs["current_activity"]


@pytest.mark.asyncio
async def test_reconcile_orphan_open_orders_routes_shadow_and_non_shadow(monkeypatch):
    rows = [
        SimpleNamespace(trader_id="orphan-shadow", mode_key="shadow", count=2),
        SimpleNamespace(trader_id="orphan-live", mode_key="live", count=1),
    ]

    class _Result:
        def all(self):
            return rows

    class _Session:
        async def execute(self, *_args, **_kwargs):
            return _Result()

    reconcile_mock = AsyncMock(return_value={"closed": 2})
    cleanup_mock = AsyncMock(return_value={"updated": 1})
    sync_mock = AsyncMock(return_value={})
    event_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_shadow_positions", reconcile_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "cleanup_trader_open_orders", cleanup_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", sync_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", event_mock)

    summary = await trader_orchestrator_worker._reconcile_orphan_open_orders(_Session())

    assert summary["traders_seen"] == 2
    assert summary["rows_seen"] == 2
    assert summary["shadow_closed"] == 2
    assert summary["non_shadow_cancelled"] == 1
    reconcile_mock.assert_awaited_once()
    cleanup_mock.assert_awaited_once()
    sync_mock.assert_awaited_once()
    event_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_trader_once_blocks_stacking_when_allow_averaging_false(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    decision_checks: list[list[dict]] = []
    submit_calls = {"count": 0}
    list_calls = {"count": 0}
    evaluate_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _create_decision_checks(session, *, checks, **kwargs):
        decision_checks.append(checks)

    async def _execute_signal(self, **kwargs):
        submit_calls["count"] += 1
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.4,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=1))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=1))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value={"market-1"}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", _create_decision_checks)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "execute_signal",
        _execute_signal,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=False),
        _base_control_payload(),
    )

    # stacking-guard pre-filter writes 2 DB ops (status + consumption), not a trader_decision row
    assert decisions_written == 0
    assert orders_written == 0
    assert submit_calls["count"] == 0
    assert evaluate_calls["count"] == 0


@pytest.mark.asyncio
async def test_run_trader_once_claims_live_signal_before_submit(monkeypatch):
    signal = _base_signal()
    call_log: list[str] = []
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        return SimpleNamespace(id="decision-live-claim")

    async def _execute_signal(self, **kwargs):
        raise AssertionError("submit_order should be mocked directly in this test")

    async def _submit_order(**kwargs):
        assert kwargs["signal"].id == "signal-1"
        assert "status:selected" in call_log
        assert "consumption:claiming" in call_log
        assert "cursor:signal-1" in call_log
        assert "commit" in call_log
        assert call_log.index("status:selected") < call_log.index("submit")
        assert call_log.index("consumption:claiming") < call_log.index("submit")
        assert call_log.index("cursor:signal-1") < call_log.index("submit")
        assert call_log.index("commit") < call_log.index("submit")
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.4,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _record_consumption(session, *, outcome, signal_id, **kwargs):
        del session, signal_id, kwargs
        call_log.append(f"consumption:{outcome}")

    async def _set_status(session, *, signal_id, status, **kwargs):
        del session, signal_id, kwargs
        call_log.append(f"status:{status}")
        return True

    async def _cursor(session, *, last_signal_id, **kwargs):
        del session, kwargs
        call_log.append(f"cursor:{last_signal_id}")

    async def _commit(session):
        del session
        call_log.append("commit")

    async def _create_event(session, **kwargs):
        del session, kwargs
        return None

    async def _update_decision(session, **kwargs):
        del session, kwargs
        return None

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(
            return_value={
                "attempted": 0,
                "backfilled": 0,
                "skipped": 0,
                "errors": [],
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "execute_signal",
        _execute_signal,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", _set_status)
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", _create_event)
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", _cursor)
    monkeypatch.setattr(trader_orchestrator_worker, "_commit_with_retry", _commit)
    monkeypatch.setattr(trader_orchestrator_worker, "submit_order", _submit_order)
    monkeypatch.setattr(trader_orchestrator_worker, "update_trader_decision", _update_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "execution_latency_metrics",
        SimpleNamespace(record=AsyncMock(return_value=None)),
    )

    async def _submit_wrapper(**kwargs):
        call_log.append("submit")
        return await _submit_order(**kwargs)

    monkeypatch.setattr(trader_orchestrator_worker, "submit_order", _submit_wrapper)

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=True),
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 1
    assert processed_signals == 1
    assert call_log.count("status:selected") == 1
    assert "consumption:claiming" in call_log
    assert "submit" in call_log
    submit_index = call_log.index("submit")
    commit_indexes = [index for index, value in enumerate(call_log) if value == "commit"]
    assert len(commit_indexes) == 2
    assert sum(1 for index in commit_indexes if index < submit_index) == 1
    assert sum(1 for index in commit_indexes if index > submit_index) == 1


@pytest.mark.asyncio
async def test_run_trader_once_persists_blocked_decision_when_db_stacking_verification_hits(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    call_log: list[str] = []
    list_calls = {"count": 0}

    class _StackingSession(_DummySession):
        class _ExecuteResult(_DummySession._ExecuteResult):
            def scalar_one(self):
                return 1

        async def execute(self, *args, **kwargs):
            del args, kwargs
            return self._ExecuteResult()

    class _StackingSessionContext:
        async def __aenter__(self):
            return _StackingSession()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        del session
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-db-stacking")

    async def _submit_order(**kwargs):
        del kwargs
        raise AssertionError("submit_order must not run when DB stacking verification blocks the signal")

    async def _record_consumption(session, *, outcome, signal_id, **kwargs):
        del session, signal_id, kwargs
        call_log.append(f"consumption:{outcome}")

    async def _buffer_consumption(*, outcome, **kwargs):
        del kwargs
        call_log.append(f"consumption:{outcome}")

    async def _set_status(session, *, signal_id, status, **kwargs):
        del session, signal_id, kwargs
        call_log.append(f"status:{status}")
        return True

    async def _cursor(session, *, last_signal_id, **kwargs):
        del session, last_signal_id, kwargs
        return None

    async def _create_event(session, **kwargs):
        del session, kwargs
        return None

    async def _update_decision(session, **kwargs):
        del session, kwargs
        return None

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _StackingSessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(
            return_value={
                "attempted": 0,
                "backfilled": 0,
                "skipped": 0,
                "errors": [],
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", _set_status)
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", _create_event)
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", _cursor)
    monkeypatch.setattr(trader_orchestrator_worker, "submit_order", _submit_order)
    monkeypatch.setattr(trader_orchestrator_worker, "update_trader_decision", _update_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "execution_latency_metrics",
        SimpleNamespace(record=AsyncMock(return_value=None)),
    )
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_signal_consumption", _buffer_consumption)

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=False),
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 0
    assert processed_signals == 1
    assert decisions[0]["decision"] == "blocked"
    assert "db verification" in decisions[0]["reason"].lower()
    assert "status:selected" not in call_log
    assert "consumption:claiming" not in call_log
    assert "status:skipped" not in call_log
    assert "consumption:blocked" in call_log


@pytest.mark.asyncio
async def test_run_trader_once_prefilters_open_live_markets_before_live_context_refresh(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    live_context_calls = {"count": 0}
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-open-market-prefilter")

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    async def _live_contexts(*args, **kwargs):
        live_context_calls["count"] += 1
        return {"unexpected": True}

    trader = _base_trader_payload(allow_averaging=False)
    control = _base_control_payload()
    control["settings"]["global_runtime"]["live_market_context"] = {
        "enabled": True,
        "strict_ws_pricing_only": True,
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(trader_orchestrator_worker, "_ensure_runtime_signal_persisted", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_ensure_prefetched_source_runtime_state", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value={"market-1"}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "build_cached_live_signal_contexts",
        _live_contexts,
    )

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader,
        control,
    )

    # stacking-guard pre-filter writes 2 DB ops (status + consumption), not a trader_decision row
    assert decisions_written == 0
    assert orders_written == 0
    assert live_context_calls["count"] == 0


@pytest.mark.asyncio
async def test_run_trader_once_handles_aware_loss_cooldown_without_datetime_type_error(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    risk_calls: list[dict] = []
    submit_calls = {"count": 0}
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-cooldown")

    def _evaluate_risk(**kwargs):
        risk_calls.append(kwargs)
        if kwargs.get("cooldown_active"):
            return SimpleNamespace(allowed=False, reason="cooldown active", checks=[])
        return SimpleNamespace(allowed=True, reason="ok", checks=[])

    async def _execute_signal(self, **kwargs):
        submit_calls["count"] += 1
        return SimpleNamespace(
            session_id="session-cooldown",
            status="completed",
            effective_price=0.4,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "evaluate_risk", _evaluate_risk)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_enforce_source_open_order_timeouts",
        AsyncMock(return_value={"configured": 0, "updated": 0, "suppressed": 0, "errors": []}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=1))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_last_resolved_loss_at",
        AsyncMock(return_value=datetime.now(timezone.utc) - timedelta(seconds=20)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "execute_signal",
        _execute_signal,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    trader_payload = _base_trader_payload(allow_averaging=True)
    trader_payload["risk_limits"]["cooldown_seconds"] = 120

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 0
    assert submit_calls["count"] == 0
    assert risk_calls
    assert risk_calls[0]["cooldown_active"] is True
    assert decisions[0]["decision"] == "blocked"
    assert "cooldown active" in str(decisions[0]["reason"]).lower()


@pytest.mark.asyncio
async def test_run_trader_once_allows_reentry_when_allow_averaging_true(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    submit_calls = {"count": 0}
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _execute_signal(self, **kwargs):
        submit_calls["count"] += 1
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.4,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=1))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=1))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value={"market-1"}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "execute_signal",
        _execute_signal,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    control_payload = _base_control_payload()
    control_payload["mode"] = "shadow"
    control_payload["settings"]["shadow_account_id"] = "shadow-1"

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=True),
        control_payload,
    )

    assert decisions_written == 1
    assert orders_written == 1
    assert submit_calls["count"] == 1
    assert decisions[0]["decision"] == "selected"


@pytest.mark.asyncio
async def test_run_trader_once_records_buy_gate_skip_on_selected_live_decision(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    decision_check_batches: list[list[dict]] = []
    consumptions: list[dict] = []
    list_calls = {"count": 0}
    update_decision_mock = AsyncMock(return_value=None)

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-buy-gate")

    async def _create_decision_checks(session, *, checks, **kwargs):
        decision_check_batches.append(list(checks))

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-buy-gate",
            status="skipped",
            effective_price=None,
            error_message=(
                "BUY pre-submit gate failed: not enough collateral balance/allowance. "
                "required_total_usdc=304.425 available_usdc=300.332299"
            ),
            orders_written=0,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    async def _record_consumption(_session, **kwargs):
        consumptions.append(kwargs)
        return None

    async def _buffer_consumption(**kwargs):
        consumptions.append(kwargs)

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", _create_decision_checks)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "execute_signal",
        _execute_signal,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_persist_trader_decision_update",
        update_decision_mock,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_signal_consumption", _buffer_consumption)

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=True),
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 0
    assert decisions[0]["decision"] == "selected"
    update_decision_mock.assert_awaited_once()
    update_call = update_decision_mock.await_args.kwargs
    assert update_call["decision_id"] == "decision-buy-gate"
    assert update_call["decision"] == "skipped"
    assert "BUY pre-submit gate failed" in str(update_call["reason"])
    assert update_call["checks_summary_patch"]["count"] > decisions[0]["checks_summary"]["count"]

    flattened_checks = [check for batch in decision_check_batches for check in batch]
    buy_gate_check = next(check for check in flattened_checks if check["check_key"] == "buy_pre_submit_gate")
    assert buy_gate_check["passed"] is False
    assert "not enough collateral balance/allowance" in buy_gate_check["detail"]
    assert any(c.get("signal_id") == "signal-1" and c.get("outcome") == "skipped" for c in consumptions)


@pytest.mark.asyncio
async def test_run_trader_once_marks_signal_skipped_when_strategy_skips(monkeypatch):
    signal = _base_signal()
    decisions: list[dict] = []
    statuses: list[tuple[str, str]] = []
    consumptions: list[dict] = []
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    async def _set_status(_session, *, signal_id, status, **_kwargs):
        statuses.append((str(signal_id), str(status)))
        return True

    async def _buffer_status(*, signal_id, status, **_kwargs):
        statuses.append((str(signal_id), str(status)))

    async def _record_consumption(_session, **kwargs):
        consumptions.append(kwargs)
        return None

    async def _buffer_consumption(**kwargs):
        consumptions.append(kwargs)

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SkippedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", _set_status)
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_signal_consumption", _buffer_consumption)
    monkeypatch.setattr(trader_orchestrator_worker.hot_state, "buffer_signal_status", _buffer_status)

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=True),
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 0
    assert decisions[0]["decision"] == "skipped"
    assert any(entry[0] == "signal-1" and entry[1] == "skipped" for entry in statuses)
    assert any(c.get("signal_id") == "signal-1" and c.get("outcome") == "skipped" for c in consumptions)


@pytest.mark.asyncio
async def test_run_trader_once_blocks_unavailable_strategy_only(monkeypatch):
    crypto_signal = _base_signal()
    news_signal = SimpleNamespace(
        id="signal-2",
        created_at=datetime.utcnow(),
        source="news",
        signal_type="news_intent",
        strategy_type="news_reaction",
        market_id="news-market-1",
        market_question="Will event happen?",
        direction="buy_yes",
        entry_price=0.35,
        edge_percent=9.0,
        confidence=0.75,
        payload_json={},
    )
    decisions: list[dict] = []
    consumptions: list[dict] = []
    statuses: list[tuple[str, str]] = []
    submit_calls = {"count": 0}
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [crypto_signal, news_signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id=f"decision-{len(decisions)}")

    async def _execute_signal(self, **kwargs):
        submit_calls["count"] += 1
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.4,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    async def _set_status(_session, *, signal_id, status, **_kwargs):
        statuses.append((str(signal_id), str(status)))
        return True

    async def _record_consumption(_session, **kwargs):
        consumptions.append(kwargs)
        return None

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "btc_eth_maker_quote",
                "strategy_params": {
                    "max_signals_per_cycle": 2,
                    "scan_batch_size": 2,
                },
            },
            {
                "source_key": "news",
                "strategy_key": "news_reaction",
                "strategy_params": {},
            },
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": False, "strict_ws_pricing_only": False},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto", "news"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(
            side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(
                strategy_key
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_active_strategy_experiment",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: (
            SimpleNamespace(
                available=False,
                strategy_key=strategy_key,
                resolved_key=strategy_key,
                reason=f"strategy_unavailable:{strategy_key}",
            )
            if strategy_key == "btc_eth_maker_quote"
            else SimpleNamespace(
                available=True,
                strategy_key=strategy_key,
                resolved_key=strategy_key,
                reason=None,
            )
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda strategy_key: _SelectedStrategy() if strategy_key in {"news_reaction", "news"} else None,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "execute_signal",
        _execute_signal,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", _set_status)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    decisions_written, orders_written, _processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    blocked = [d for d in decisions if d.get("decision") == "blocked"]
    selected = [d for d in decisions if d.get("decision") == "selected"]

    assert decisions_written == 2
    assert orders_written == 1
    assert submit_calls["count"] == 1
    assert len(blocked) == 1
    assert blocked[0]["strategy_key"] == "btc_eth_maker_quote"
    assert blocked[0]["reason"].startswith("strategy_unavailable:btc_eth_maker_quote")
    assert len(selected) == 1
    assert any(entry[0] == "signal-1" and entry[1] == "skipped" for entry in statuses)
    assert any(c.get("signal_id") == "signal-1" and c.get("outcome") == "blocked" for c in consumptions)


@pytest.mark.asyncio
async def test_run_trader_once_uses_cached_live_context_builder_for_trigger_cycles(monkeypatch):
    trigger_snapshot = {
        "id": "signal-1",
        "source": "crypto",
        "source_item_id": "signal-1",
        "signal_type": "crypto_worker_multistrat",
        "strategy_type": "btc_eth_maker_quote",
        "market_id": "market-1",
        "market_question": "Will BTC close higher?",
        "direction": "buy_yes",
        "entry_price": 0.4,
        "edge_percent": 8.0,
        "confidence": 0.72,
        "liquidity": 5000.0,
        "status": "pending",
        "payload_json": {
            "markets": [
                {
                    "id": "market-1",
                    "question": "Will BTC close higher?",
                    "token_ids": ["123456789012345678", "987654321098765432"],
                    "outcomes": ["Yes", "No"],
                }
            ]
        },
        "strategy_context_json": {},
        "created_at": "2026-03-10T02:39:50Z",
        "updated_at": "2026-03-10T02:39:51Z",
    }
    cached_context_mock = AsyncMock(
        return_value={
            "signal-1": {
                "available": True,
                "live_selected_price": 0.39,
                "live_edge_percent": 9.0,
                "market_data_source": "ws_strict",
                "market_data_age_ms": 12.0,
            }
        }
    )
    _full_context_mock = AsyncMock(side_effect=AssertionError("trigger cycles should not use full live builder"))

    async def _create_decision(session, **kwargs):
        return SimpleNamespace(id="decision-trigger")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-trigger",
            status="completed",
            effective_price=0.39,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = _base_trader_payload(allow_averaging=True)
    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": False},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(trader_orchestrator_worker, "build_cached_live_signal_contexts", cached_context_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        AsyncMock(return_value=[SimpleNamespace(id="signal-1", status="pending", runtime_sequence=None)]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_unconsumed_trade_signals",
        AsyncMock(side_effect=AssertionError("trigger cycles should use prefetched signal snapshots")),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
        trigger_signal_ids_by_source={"crypto": ["signal-1"]},
        trigger_signal_snapshots_by_source={"crypto": {"signal-1": trigger_snapshot}},
    )

    assert decisions_written == 1
    assert orders_written == 1
    assert processed_signals == 1
    cached_context_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_trader_once_trigger_cycle_fetches_full_live_context_when_strict_ws_required(monkeypatch):
    trigger_snapshot = {
        "id": "signal-1",
        "source": "crypto",
        "source_item_id": "signal-1",
        "signal_type": "crypto_worker_multistrat",
        "strategy_type": "btc_eth_maker_quote",
        "market_id": "market-1",
        "market_question": "Will BTC close higher?",
        "direction": "buy_yes",
        "entry_price": 0.4,
        "edge_percent": 8.0,
        "confidence": 0.72,
        "liquidity": 5000.0,
        "status": "pending",
        "payload_json": {
            "markets": [
                {
                    "id": "market-1",
                    "question": "Will BTC close higher?",
                    "token_ids": ["123456789012345678", "987654321098765432"],
                    "outcomes": ["Yes", "No"],
                }
            ]
        },
        "strategy_context_json": {},
        "created_at": "2026-03-10T02:39:50Z",
        "updated_at": "2026-03-10T02:39:51Z",
    }
    strict_live_context = {
        "signal-1": {
            "available": True,
            "live_selected_price": 0.39,
            "live_edge_percent": 9.0,
            "market_data_source": "ws_strict",
            "market_data_age_ms": 12.0,
        }
    }
    cached_context_mock = AsyncMock(return_value=strict_live_context)
    _full_context_mock = AsyncMock(return_value=strict_live_context)

    async def _create_decision(session, **kwargs):
        return SimpleNamespace(id="decision-trigger-strict")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-trigger-strict",
            status="completed",
            effective_price=0.39,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = _base_trader_payload(allow_averaging=True)
    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(trader_orchestrator_worker, "build_cached_live_signal_contexts", cached_context_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_list_unconsumed_trade_signals_authoritative",
        AsyncMock(return_value=[SimpleNamespace(id="signal-1", status="pending", runtime_sequence=None)]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "list_unconsumed_trade_signals",
        AsyncMock(side_effect=AssertionError("trigger cycles should use prefetched signal snapshots")),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
        trigger_signal_ids_by_source={"crypto": ["signal-1"]},
        trigger_signal_snapshots_by_source={"crypto": {"signal-1": trigger_snapshot}},
    )

    assert decisions_written == 1
    assert processed_signals == 1
    cached_context_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_trader_once_defers_signals_when_strict_ws_context_unavailable(monkeypatch):
    signal = _base_signal()
    heartbeat_calls: list[dict] = []
    record_consumption_mock = AsyncMock(return_value=None)
    cursor_mock = AsyncMock(return_value=None)

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _heartbeat(session, trader_id, message, payload=None):
        heartbeat_calls.append({"message": message, "payload": payload or {}})

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = _base_trader_payload(allow_averaging=True)
    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(trader_orchestrator_worker, "build_cached_live_signal_contexts", AsyncMock(return_value={}))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", record_consumption_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", cursor_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", _heartbeat)
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    record_consumption_mock.assert_not_awaited()
    cursor_mock.assert_not_awaited()
    assert heartbeat_calls == [] or heartbeat_calls[-1]["message"] == "Idle cycle: pending signals deferred awaiting live market context."


@pytest.mark.asyncio
async def test_run_trader_once_uses_strategy_configured_strict_sources_for_live_context(monkeypatch):
    signal = _base_signal()
    signal.source = "scanner"
    signal.signal_type = "scanner_opportunity"
    signal.strategy_type = "generic_strategy"
    signal.direction = "buy_no"
    signal.entry_price = 0.91
    signal.market_id = "scanner-market-1"
    signal.market_question = "Will event resolve no?"
    signal.payload_json = {
        "signal_emitted_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "ingested_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
    }
    decisions: list[dict] = []
    defer_signal_mock = AsyncMock(return_value=True)

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.905,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "generic_strategy",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": False,
                    "require_strict_ws_pricing": True,
                    "strict_ws_price_sources": ["market_snapshot"],
                    "max_market_data_age_ms": 15000,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["scanner"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "build_cached_live_signal_contexts",
        AsyncMock(
            return_value={
                signal.id: {
                    "available": True,
                    "live_selected_price": 0.905,
                    "market_data_source": "market_snapshot",
                    "market_data_age_ms": 25.0,
                }
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_intent_runtime",
        lambda: SimpleNamespace(defer_signal=defer_signal_mock),
    )

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 1
    assert processed_signals == 1
    assert len(decisions) == 1
    defer_signal_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_loads_strict_scanner_live_context_from_cache(monkeypatch):
    signal = _base_signal()
    signal.source = "scanner"
    signal.signal_type = "scanner_opportunity"
    signal.strategy_type = "generic_strategy"
    signal.direction = "buy_no"
    signal.entry_price = 0.905
    signal.market_id = "scanner-market-current"
    signal.market_question = "Will event resolve no?"
    signal.payload_json = {
        "signal_emitted_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "ingested_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
    }
    decisions: list[dict] = []
    defer_signal_mock = AsyncMock(return_value=True)
    cached_context_mock = AsyncMock(
        return_value={
            signal.id: {
                "available": True,
                "live_selected_price": 0.908,
                "market_data_source": "ws_strict",
                "market_data_age_ms": 16000.0,
                "ws_subscription_current": True,
            }
        }
    )

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-scanner-strict")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-scanner-strict",
            status="completed",
            effective_price=0.908,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "generic_strategy",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": True,
                    "require_strict_ws_pricing": True,
                    "strict_ws_price_sources": ["ws_strict"],
                    "max_market_data_age_ms": 15000,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["scanner"])
    monkeypatch.setattr(trader_orchestrator_worker, "build_cached_live_signal_contexts", cached_context_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_intent_runtime",
        lambda: SimpleNamespace(defer_signal=defer_signal_mock),
    )

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 1
    assert processed_signals == 1
    assert len(decisions) == 1
    cached_context_mock.assert_awaited_once()
    await_args = cached_context_mock.await_args.args[0]
    assert len(await_args) == 1
    assert getattr(await_args[0], "id", None) == signal.id
    defer_signal_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_uses_scanner_signal_market_snapshot_when_live_context_cache_is_empty(monkeypatch):
    signal = _base_signal()
    signal.source = "scanner"
    signal.signal_type = "scanner_opportunity"
    signal.strategy_type = "generic_strategy"
    signal.direction = "buy_no"
    signal.entry_price = 0.885
    signal.market_id = "scanner-market-1"
    signal.market_question = "Will XRP finish lower?"
    now = datetime.now(timezone.utc)
    price_updated_at = (now - timedelta(milliseconds=250)).isoformat().replace("+00:00", "Z")
    signal.payload_json = {
        "signal_emitted_at": now.isoformat().replace("+00:00", "Z"),
        "ingested_at": now.isoformat().replace("+00:00", "Z"),
        "markets": [
            {
                "id": "scanner-market-1",
                "question": "Will XRP finish lower?",
                "condition_id": "condition-1",
                "clob_token_ids": ["token-yes", "token-no"],
                "current_yes_price": 0.115,
                "current_no_price": 0.885,
                "price_updated_at": price_updated_at,
                "liquidity": 10000.0,
                "volume": 5000.0,
            }
        ],
        "positions_to_take": [
            {
                "action": "BUY",
                "outcome": "NO",
                "price": 0.885,
                "token_id": "token-no",
            }
        ],
    }
    decisions: list[dict] = []
    defer_signal_mock = AsyncMock(return_value=True)

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.884,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "generic_strategy",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": False,
                    "require_strict_ws_pricing": True,
                    "strict_ws_price_sources": ["market_snapshot"],
                    "max_market_data_age_ms": 15000,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["scanner"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "build_cached_live_signal_contexts",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_intent_runtime",
        lambda: SimpleNamespace(defer_signal=defer_signal_mock),
    )

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 1
    assert processed_signals == 1
    assert len(decisions) == 1
    defer_signal_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_uses_scanner_signal_created_at_when_payload_timestamp_is_missing(monkeypatch):
    signal = _base_signal()
    signal.source = "scanner"
    signal.signal_type = "scanner_opportunity"
    signal.strategy_type = "generic_strategy"
    signal.direction = "buy_no"
    signal.entry_price = 0.885
    signal.market_id = "scanner-market-1"
    signal.market_question = "Will XRP finish lower?"
    now = datetime.now(timezone.utc)
    signal.created_at = now - timedelta(milliseconds=250)
    signal.payload_json = {
        "markets": [
            {
                "id": "scanner-market-1",
                "question": "Will XRP finish lower?",
                "condition_id": "condition-1",
                "clob_token_ids": ["token-yes", "token-no"],
                "current_yes_price": 0.115,
                "current_no_price": 0.885,
                "liquidity": 10000.0,
                "volume": 5000.0,
            }
        ],
        "positions_to_take": [
            {
                "action": "BUY",
                "outcome": "NO",
                "price": 0.885,
                "token_id": "token-no",
            }
        ],
    }
    decisions: list[dict] = []
    defer_signal_mock = AsyncMock(return_value=True)

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.884,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "generic_strategy",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": False,
                    "require_strict_ws_pricing": True,
                    "strict_ws_price_sources": ["market_snapshot"],
                    "max_market_data_age_ms": 15000,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["scanner"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "build_cached_live_signal_contexts",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_intent_runtime",
        lambda: SimpleNamespace(defer_signal=defer_signal_mock),
    )

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 1
    assert processed_signals == 1
    assert len(decisions) == 1
    defer_signal_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_defers_signals_when_strict_ws_release_is_stale(monkeypatch):
    signal = _base_signal()
    signal.payload_json = {
        "signal_emitted_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "ingested_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
    }
    heartbeat_calls: list[dict] = []
    record_consumption_mock = AsyncMock(return_value=None)
    cursor_mock = AsyncMock(return_value=None)
    defer_signal_mock = AsyncMock(return_value=True)

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _heartbeat(session, trader_id, message, payload=None):
        heartbeat_calls.append({"message": message, "payload": payload or {}})

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = _base_trader_payload(allow_averaging=True)
    trader_payload["source_configs"][0]["strategy_params"]["max_market_data_age_ms"] = 15000
    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {
                    "enabled": True,
                    "strict_ws_pricing_only": True,
                    "max_market_data_age_ms": 10000,
                },
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "build_cached_live_signal_contexts",
        AsyncMock(
            return_value={
                signal.id: {
                    "market_data_source": "ws_strict",
                    "live_selected_price": 0.41,
                    "market_data_age_ms": 5,
                }
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", record_consumption_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", cursor_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", _heartbeat)
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_intent_runtime",
        lambda: SimpleNamespace(defer_signal=defer_signal_mock),
    )

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 0
    assert orders_written == 0
    assert processed_signals == 0
    record_consumption_mock.assert_not_awaited()
    cursor_mock.assert_not_awaited()
    defer_signal_mock.assert_awaited_once()
    assert defer_signal_mock.await_args.kwargs["reason"] == "strict_ws_pricing_signal_release_stale"
    assert heartbeat_calls == [] or heartbeat_calls[-1]["message"] == "Idle cycle: pending signals deferred awaiting live market context."


@pytest.mark.asyncio
async def test_run_trader_once_uses_fresh_scanner_row_timestamp_for_strict_ws_release_age(monkeypatch):
    signal = _base_signal()
    signal.source = "scanner"
    signal.signal_type = "scanner_opportunity"
    signal.strategy_type = "generic_strategy"
    signal.direction = "buy_no"
    signal.entry_price = 0.885
    signal.market_id = "scanner-market-1"
    signal.market_question = "Will XRP finish lower?"
    signal.created_at = datetime.now(timezone.utc) - timedelta(seconds=20)
    signal.updated_at = datetime.now(timezone.utc)
    signal.payload_json = {
        "signal_emitted_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "ingested_at": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "markets": [
            {
                "id": "scanner-market-1",
                "question": "Will XRP finish lower?",
                "condition_id": "condition-1",
                "clob_token_ids": ["token-yes", "token-no"],
                "current_yes_price": 0.115,
                "current_no_price": 0.885,
                "liquidity": 10000.0,
                "volume": 5000.0,
            }
        ],
        "positions_to_take": [
            {
                "action": "BUY",
                "outcome": "NO",
                "price": 0.885,
                "token_id": "token-no",
            }
        ],
    }
    decisions: list[dict] = []
    defer_signal_mock = AsyncMock(return_value=True)

    async def _list_unconsumed(*args, **kwargs):
        return [signal]

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id="session-1",
            status="completed",
            effective_price=0.884,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "generic_strategy",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": False,
                    "require_strict_ws_pricing": True,
                    "strict_ws_price_sources": ["redis_strict"],
                    "max_market_data_age_ms": 15000,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }

    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": True, "strict_ws_pricing_only": True},
            },
            "shadow_account_id": "shadow-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["scanner"])
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "build_cached_live_signal_contexts",
        AsyncMock(
            return_value={
                signal.id: {
                    "market_data_source": "redis_strict",
                    "live_selected_price": 0.885,
                    "market_data_age_ms": 5,
                }
            }
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(return_value={"matched": 0, "closed": 0, "held": 0, "skipped": 0, "total_realized_pnl": 0.0, "by_status": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_intent_runtime",
        lambda: SimpleNamespace(defer_signal=defer_signal_mock),
    )

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 1
    assert processed_signals == 1
    assert orders_written == 0
    assert len(decisions) == 1
    defer_signal_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_trader_once_prefetches_strategy_metadata_once_per_source(monkeypatch):
    signal_one = _base_signal()
    signal_two = SimpleNamespace(**signal_one.__dict__)
    signal_two.id = "signal-2"
    signal_two.market_id = "market-2"
    signal_two.market_question = "Will ETH close higher?"
    signal_two.created_at = datetime.utcnow() + timedelta(seconds=1)

    decisions: list[dict] = []
    list_calls = {"count": 0}

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal_one, signal_two] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id=f"decision-{len(decisions)}")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id=f"session-{len(decisions)}",
            status="completed",
            effective_price=0.4,
            error_message=None,
            orders_written=1,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = _base_trader_payload(allow_averaging=True)
    trader_payload["source_configs"][0]["strategy_params"]["max_signals_per_cycle"] = 2
    trader_payload["source_configs"][0]["strategy_params"]["scan_batch_size"] = 2
    control_payload = {
        "mode": "shadow",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "global_runtime": {
                "live_market_context": {"enabled": False, "strict_ws_pricing_only": False},
            },
            "shadow_account_id": "shadow-1",
        },
    }
    resolve_mock = AsyncMock(
        side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)
    )
    experiment_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
    monkeypatch.setattr(trader_orchestrator_worker, "resolve_strategy_version", resolve_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", experiment_mock)
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_shadow_positions",
        AsyncMock(
            return_value={
                "matched": 0,
                "closed": 0,
                "held": 0,
                "skipped": 0,
                "total_realized_pnl": 0.0,
                "by_status": {},
            }
        ),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_occupied_market_ids_for_trader",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "list_unconsumed_trade_signals", _list_unconsumed)
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    assert decisions_written == 2
    assert orders_written == 2
    assert processed_signals == 2
    assert resolve_mock.await_count == 1
    assert experiment_mock.await_count == 1


@pytest.mark.asyncio
async def test_run_trader_once_live_runtime_trigger_processes_runtime_trigger_batch(monkeypatch):
    signals: list[SimpleNamespace] = []
    for idx in range(5):
        signal = _base_signal()
        signal.id = f"signal-{idx + 1}"
        signal.source = "scanner"
        signal.signal_type = "scanner_opportunity"
        signal.strategy_type = "generic_strategy"
        signal.market_id = f"market-{idx + 1}"
        signal.market_question = f"Will market {idx + 1} resolve?"
        signal.direction = "buy_no"
        signal.entry_price = 0.88
        signal.created_at = datetime.utcnow() + timedelta(seconds=idx)
        signals.append(signal)

    decisions: list[dict] = []

    async def _build_triggered(*args, **kwargs):
        return list(signals)

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id=f"decision-{len(decisions)}")

    async def _execute_signal(self, **kwargs):
        return SimpleNamespace(
            session_id=f"session-{len(decisions)}",
            status="completed",
            effective_price=0.88,
            error_message=None,
            orders_written=0,
            payload={},
        )

    async def _reconcile_active_sessions(self, *, mode, trader_id=None):
        return {"active_seen": 0, "expired": 0, "completed": 0, "failed": 0}

    trader_payload = {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "generic_strategy",
                "strategy_params": {
                    "max_signals_per_cycle": 10,
                    "scan_batch_size": 10,
                    "require_live_market_revalidation": False,
                    "require_live_revalidation_for_sources": [],
                    "enforce_market_data_freshness": False,
                },
            }
        ],
        "risk_limits": {"allow_averaging": True},
        "metadata": {"resume_policy": "resume_full"},
    }
    control_payload = _base_control_payload()

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["scanner"])
    monkeypatch.setattr(trader_orchestrator_worker, "_build_triggered_trade_signals", _build_triggered)
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "resolve_strategy_version",
        AsyncMock(side_effect=lambda _session, *, strategy_key, requested_version: _mock_resolve_strategy_version(strategy_key)),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_active_strategy_experiment", AsyncMock(return_value=None))
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_availability",
        lambda strategy_key: SimpleNamespace(
            available=True,
            strategy_key=strategy_key,
            resolved_key=strategy_key,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker.strategy_db_loader,
        "get_strategy",
        lambda *_: _SelectedStrategy(),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "RuntimeTradeSignalView",
        lambda sig, live_context=None: SimpleNamespace(**sig.__dict__, live_context=(live_context or {})),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "evaluate_risk",
        lambda **_: SimpleNamespace(allowed=True, reason="ok", checks=[]),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_backfill_simulation_ledger_for_active_shadow_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "sync_trader_position_inventory",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_position_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_open_order_count_for_trader", AsyncMock(return_value=0))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_pending_live_exit_summary_for_trader",
        AsyncMock(return_value={"count": 0, "order_ids": [], "market_ids": [], "statuses": {}}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_occupied_market_ids_for_trader", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "cleanup_trader_open_orders",
        AsyncMock(return_value={"matched": 0, "updated": 0}),
    )
    monkeypatch.setattr(trader_orchestrator_worker, "get_daily_realized_pnl", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_consecutive_loss_count", AsyncMock(return_value=0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_last_resolved_loss_at", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_cursor", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(trader_orchestrator_worker, "get_trader_signal_sequence_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "get_gross_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "get_market_exposure", AsyncMock(return_value=0.0))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision", _create_decision)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_decision_checks", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker.ExecutionSessionEngine, "execute_signal", _execute_signal)
    monkeypatch.setattr(
        trader_orchestrator_worker.ExecutionSessionEngine,
        "reconcile_active_sessions",
        _reconcile_active_sessions,
    )
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_emit_cycle_heartbeat_if_due", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "_persist_trader_cycle_heartbeat", AsyncMock(return_value=None))
    publish_signal_batch_mock = AsyncMock(return_value="batch-1")
    monkeypatch.setattr(trader_orchestrator_worker, "publish_signal_batch", publish_signal_batch_mock)

    decisions_written, orders_written, processed_signals = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
        trigger_signal_ids_by_source={"scanner": [signal.id for signal in signals]},
        trigger_signal_snapshots_by_source={"scanner": {signal.id: {"id": signal.id} for signal in signals}},
    )

    # Standard lane per-cycle signal limit is 4; with 5 signals queued,
    # the cycle processes the first 4 and re-publishes signal-5 back to
    # the runtime-trigger queue for the next cycle.
    assert decisions_written == 4
    assert orders_written == 0
    assert processed_signals == 4
    assert len(decisions) == 4
    publish_signal_batch_mock.assert_awaited_once()
    assert publish_signal_batch_mock.await_args.kwargs["signal_ids"] == [
        "signal-5",
    ]


