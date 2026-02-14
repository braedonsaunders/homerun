import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import trader_orchestrator_worker
from services.trader_orchestrator.strategies.registry import get_strategy


def test_supports_live_market_context_excludes_crypto_source():
    assert trader_orchestrator_worker._supports_live_market_context(SimpleNamespace(source="crypto")) is False
    assert trader_orchestrator_worker._supports_live_market_context(SimpleNamespace(source="weather")) is True


def test_strategy_registry_supports_legacy_default_alias():
    strategy = get_strategy("strategy.default")
    assert strategy.key == "crypto_15m"


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
