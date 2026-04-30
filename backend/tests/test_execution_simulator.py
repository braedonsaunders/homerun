from __future__ import annotations

import sys
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.simulation.execution_simulator import ExecutionSimulator

execution_simulator_module = importlib.import_module("services.simulation.execution_simulator")


class _FakeScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, emissions):
        self._emissions = emissions
        self.added = []

    async def execute(self, _query):
        return _FakeScalarResult(self._emissions)

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_execution_simulator_uses_shadow_order_manager_and_persists_manifest(monkeypatch):
    simulator = ExecutionSimulator()
    now = datetime.now(timezone.utc)
    emission_created_at = (now - timedelta(minutes=20)).replace(tzinfo=None)
    replay_points = [
        {"t": int((now - timedelta(minutes=25)).timestamp() * 1000), "p": 0.48, "v": 50.0},
        {"t": int((now - timedelta(minutes=20)).timestamp() * 1000), "p": 0.49, "v": 70.0},
        {"t": int((now - timedelta(minutes=10)).timestamp() * 1000), "p": 0.52, "v": 80.0},
    ]

    emission = SimpleNamespace(
        id="em-1",
        signal_id="sig-1",
        source="scanner",
        signal_type="basic",
        strategy_type="basic",
        market_id="market-1",
        direction="buy_yes",
        entry_price=0.50,
        edge_percent=4.0,
        confidence=0.75,
        liquidity=3500.0,
        status="pending",
        payload_json={"market_question": "Will this execution test pass?"},
        created_at=emission_created_at,
    )
    fake_session = _FakeSession([emission])
    run_row = SimpleNamespace(
        id="run-1",
        strategy_key="unit_strategy",
        source_key="scanner",
        status="queued",
        run_seed=None,
        dataset_hash=None,
        config_hash=None,
        code_sha=None,
        requested_start_at=None,
        requested_end_at=None,
        started_at=None,
        finished_at=None,
        market_scope_json={},
        params_json={},
        summary_json={},
        error_message=None,
    )

    class _FakeStrategy:
        def evaluate(self, _signal, _context):
            return SimpleNamespace(
                decision="selected",
                reason="selected_for_test",
                score=0.9,
                size_usd=40.0,
                checks=[],
            )

    async def _reload_from_db(_strategy_key, session):
        return {"status": "loaded", "source_hash": "runtime-hash-123"}

    async def _get_points(**_kwargs):
        return replay_points

    async def _submit_execution_leg(**_kwargs):
        return SimpleNamespace(
            status="executed",
            effective_price=0.505,
            error_message=None,
            payload={
                "mode": "shadow",
                "submission": "simulated",
                "shadow_simulation": {
                    "filled": True,
                    "fill_ratio": 1.0,
                    "slippage_bps": 11.25,
                },
            },
            shares=78.0,
            notional_usd=39.39,
        )

    monkeypatch.setattr(execution_simulator_module.strategy_db_loader, "reload_from_db", _reload_from_db)
    monkeypatch.setattr(execution_simulator_module.strategy_db_loader, "get_instance", lambda _key: _FakeStrategy())
    monkeypatch.setattr(
        execution_simulator_module.strategy_db_loader,
        "get_runtime_status",
        lambda _key: {"source_hash": "runtime-hash-123"},
    )
    monkeypatch.setattr(simulator._provider, "get_polymarket_points", _get_points)
    monkeypatch.setattr(execution_simulator_module, "submit_execution_leg", _submit_execution_leg)

    summary = await simulator.run(
        fake_session,
        run_row=run_row,
        payload={
            "strategy_key": "unit_strategy",
            "source_key": "scanner",
            "market_provider": "polymarket",
            "timeframe": "15m",
            "run_seed": "seed-777",
            "default_notional_usd": 50.0,
            "fee_bps": 200.0,
            "slippage_bps": 8.0,
        },
    )

    assert summary["status"] == "completed"
    assert summary["orders_filled"] == 1
    assert summary["replay_granularity"] == "points"
    assert summary["run_manifest"]["run_seed"] == "seed-777"
    assert len(summary["run_manifest"]["dataset_hash"]) == 64
    assert len(summary["run_manifest"]["config_hash"]) == 64
    assert summary["run_manifest"]["code_sha"] == "runtime-hash-123"
    assert run_row.status == "completed"
    assert run_row.run_seed == "seed-777"
    assert run_row.dataset_hash == summary["run_manifest"]["dataset_hash"]
    assert run_row.config_hash == summary["run_manifest"]["config_hash"]
    assert run_row.code_sha == "runtime-hash-123"

    event_types = [getattr(item, "event_type", None) for item in fake_session.added if hasattr(item, "event_type")]
    assert "signal_evaluated" in event_types
    assert "order_filled" in event_types
    assert "position_mtm_close" in event_types
