import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import trader_orchestrator_worker
from services.trader_orchestrator.strategies.base import StrategyDecision
from services.trader_orchestrator.strategies.registry import get_strategy


def test_supports_live_market_context_excludes_crypto_source():
    assert trader_orchestrator_worker._supports_live_market_context(SimpleNamespace(source="crypto")) is False
    assert trader_orchestrator_worker._supports_live_market_context(SimpleNamespace(source="weather")) is True


def test_strategy_registry_supports_legacy_default_alias():
    from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy
    strategy = get_strategy("strategy.default")
    assert isinstance(strategy, BtcEthHighFreqStrategy)


def test_resume_policy_normalizes_to_supported_values():
    assert trader_orchestrator_worker._normalize_resume_policy("manage_only") == "manage_only"
    assert trader_orchestrator_worker._normalize_resume_policy("flatten_then_start") == "flatten_then_start"
    assert trader_orchestrator_worker._normalize_resume_policy("unexpected") == "resume_full"


@pytest.mark.asyncio
async def test_main_initializes_database_before_worker_loop(monkeypatch):
    call_order: list[str] = []

    async def _fake_init_database() -> None:
        call_order.append("init_database")

    async def _fake_run_loop() -> None:
        call_order.append("run_worker_loop")
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        trader_orchestrator_worker,
        "init_database",
        _fake_init_database,
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "run_worker_loop",
        _fake_run_loop,
    )

    await trader_orchestrator_worker.main()

    assert call_order == ["init_database", "run_worker_loop"]


class _DummySession:
    async def get(self, *args, **kwargs):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None


class _DummySessionContext:
    async def __aenter__(self):
        return _DummySession()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SelectedStrategy:
    key = "crypto_15m"

    def evaluate(self, signal, context):
        return StrategyDecision(
            decision="selected",
            reason="selected",
            score=10.0,
            size_usd=25.0,
            checks=[],
            payload={},
        )


def _base_trader_payload(*, allow_averaging: bool) -> dict:
    return {
        "id": "trader-1",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "crypto_15m",
                "strategy_params": {
                    "max_signals_per_cycle": 1,
                    "scan_batch_size": 1,
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
            "enable_live_market_context": False,
        },
    }


def _base_signal() -> SimpleNamespace:
    return SimpleNamespace(
        id="signal-1",
        created_at=datetime.utcnow(),
        source="crypto",
        signal_type="crypto_worker_multistrat",
        market_id="market-1",
        market_question="Will BTC close higher?",
        direction="buy_yes",
        entry_price=0.4,
        edge_percent=8.0,
        confidence=0.72,
        payload_json={},
    )


@pytest.mark.asyncio
async def test_reconcile_orphan_open_orders_routes_paper_and_non_paper(monkeypatch):
    rows = [
        SimpleNamespace(trader_id="orphan-paper", mode_key="paper", count=2),
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

    monkeypatch.setattr(trader_orchestrator_worker, "reconcile_paper_positions", reconcile_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "cleanup_trader_open_orders", cleanup_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "sync_trader_position_inventory", sync_mock)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", event_mock)

    summary = await trader_orchestrator_worker._reconcile_orphan_open_orders(_Session())

    assert summary["traders_seen"] == 2
    assert summary["rows_seen"] == 2
    assert summary["paper_closed"] == 2
    assert summary["non_paper_cancelled"] == 1
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

    async def _list_unconsumed(*args, **kwargs):
        list_calls["count"] += 1
        return [signal] if list_calls["count"] == 1 else []

    async def _create_decision(session, **kwargs):
        decisions.append(kwargs)
        return SimpleNamespace(id="decision-1")

    async def _create_decision_checks(session, *, checks, **kwargs):
        decision_checks.append(checks)

    async def _submit_order(**kwargs):
        submit_calls["count"] += 1
        return "executed", 0.4, None, {}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
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
        "_backfill_simulation_ledger_for_active_paper_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_paper_positions",
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
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_live_positions",
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
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_open_market_ids_for_trader",
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
    monkeypatch.setattr(trader_orchestrator_worker, "submit_order", _submit_order)
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    decisions_written, orders_written = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=False),
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 0
    assert submit_calls["count"] == 0
    assert decisions[0]["decision"] == "blocked"
    assert "allow_averaging=false" in decisions[0]["reason"]
    stacking_check = next(check for check in decision_checks[0] if check["check_key"] == "stacking_guard")
    assert stacking_check["passed"] is False


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

    async def _submit_order(**kwargs):
        submit_calls["count"] += 1
        return "executed", 0.4, None, {}

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto"])
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
        "_backfill_simulation_ledger_for_active_paper_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_paper_positions",
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
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_live_positions",
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
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_open_market_ids_for_trader",
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
    monkeypatch.setattr(trader_orchestrator_worker, "submit_order", _submit_order)
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    decisions_written, orders_written = await trader_orchestrator_worker._run_trader_once(
        _base_trader_payload(allow_averaging=True),
        _base_control_payload(),
    )

    assert decisions_written == 1
    assert orders_written == 1
    assert submit_calls["count"] == 1
    assert decisions[0]["decision"] == "selected"


@pytest.mark.asyncio
async def test_run_trader_once_blocks_unavailable_strategy_only(monkeypatch):
    crypto_signal = _base_signal()
    news_signal = SimpleNamespace(
        id="signal-2",
        created_at=datetime.utcnow(),
        source="news",
        signal_type="news_intent",
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

    async def _submit_order(**kwargs):
        submit_calls["count"] += 1
        return "executed", 0.4, None, {}

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
                "strategy_key": "crypto_15m",
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
        "mode": "paper",
        "settings": {
            "global_risk": {"max_orders_per_cycle": 50, "max_daily_loss_usd": 5000.0},
            "enable_live_market_context": False,
            "paper_account_id": "paper-1",
        },
    }

    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", lambda: _DummySessionContext())
    monkeypatch.setattr(trader_orchestrator_worker, "_query_sources_for_configs", lambda *_: ["crypto", "news"])
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
            if strategy_key == "crypto_15m"
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
        lambda strategy_key: None if strategy_key == "crypto_15m" else _SelectedStrategy(),
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
        "_backfill_simulation_ledger_for_active_paper_orders",
        AsyncMock(return_value={"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}),
    )
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "reconcile_paper_positions",
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
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "get_open_market_ids_for_trader",
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
    monkeypatch.setattr(trader_orchestrator_worker, "submit_order", _submit_order)
    monkeypatch.setattr(trader_orchestrator_worker, "set_trade_signal_status", _set_status)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_order", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "record_signal_consumption", _record_consumption)
    monkeypatch.setattr(trader_orchestrator_worker, "create_trader_event", AsyncMock(return_value=None))
    monkeypatch.setattr(trader_orchestrator_worker, "upsert_trader_signal_cursor", AsyncMock(return_value=None))

    decisions_written, orders_written = await trader_orchestrator_worker._run_trader_once(
        trader_payload,
        control_payload,
    )

    blocked = [d for d in decisions if d.get("decision") == "blocked"]
    selected = [d for d in decisions if d.get("decision") == "selected"]

    assert decisions_written == 2
    assert orders_written == 1
    assert submit_calls["count"] == 1
    assert len(blocked) == 1
    assert blocked[0]["strategy_key"] == "crypto_15m"
    assert blocked[0]["reason"] == "strategy_unavailable:crypto_15m"
    assert len(selected) == 1
    assert any(entry[0] == "signal-1" and entry[1] == "skipped" for entry in statuses)
    assert any(c.get("signal_id") == "signal-1" and c.get("outcome") == "blocked" for c in consumptions)
