import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_ml
from services import market_runtime


class _FakeMachineLearningSdk:
    def __init__(self, *, recording_enabled: bool, deployment_active: bool) -> None:
        self.get_runtime_state = AsyncMock(
            return_value={
                "task_key": "crypto_directional",
                "recording_enabled": recording_enabled,
                "deployment_active": deployment_active,
                "engaged": recording_enabled or deployment_active,
            }
        )
        self.annotate_market_batch = AsyncMock(return_value={"annotated": 1})
        self.record_market_batch = AsyncMock(return_value={"recorded": 1})
        self.prune_data = AsyncMock(return_value={"status": "pruned", "deleted_rows": 0})
        self.preview_prune_data = AsyncMock(return_value={"status": "preview", "would_delete": 0})


@pytest.mark.asyncio
async def test_prune_route_previews_without_confirm(monkeypatch):
    sdk = _FakeMachineLearningSdk(recording_enabled=False, deployment_active=False)
    monkeypatch.setattr(routes_ml, "get_machine_learning_sdk", lambda: sdk)

    result = await routes_ml.prune_data(routes_ml.PruneDataRequest(older_than_days=12, confirm=False))

    assert result == {"status": "preview", "would_delete": 0}
    sdk.preview_prune_data.assert_awaited_once_with(12)
    sdk.prune_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_prune_route_executes_with_confirm(monkeypatch):
    sdk = _FakeMachineLearningSdk(recording_enabled=False, deployment_active=False)
    monkeypatch.setattr(routes_ml, "get_machine_learning_sdk", lambda: sdk)

    result = await routes_ml.prune_data(routes_ml.PruneDataRequest(older_than_days=21, confirm=True))

    assert result == {"status": "pruned", "deleted_rows": 0}
    sdk.prune_data.assert_awaited_once_with(21)
    sdk.preview_prune_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_market_runtime_skips_ml_when_idle(monkeypatch):
    sdk = _FakeMachineLearningSdk(recording_enabled=False, deployment_active=False)
    monkeypatch.setattr(market_runtime, "get_machine_learning_sdk", lambda: sdk)

    runtime = market_runtime.MarketRuntime()
    runtime._last_ml_gate_check_mono = 0.0

    await runtime._refresh_ml_pipeline([{"id": "btc-15m"}], allow_record=True)
    await runtime._ml_runtime_refresh_task

    sdk.get_runtime_state.assert_awaited_once_with("crypto_directional")
    sdk.annotate_market_batch.assert_not_awaited()
    sdk.record_market_batch.assert_not_awaited()
    sdk.prune_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_market_runtime_skips_sdk_when_deployment_inactive(monkeypatch):
    # Recording is now the recorded_event_bus's responsibility; market_runtime
    # only runs inference (annotate) when a deployment is active.
    sdk = _FakeMachineLearningSdk(recording_enabled=True, deployment_active=False)
    monkeypatch.setattr(market_runtime, "get_machine_learning_sdk", lambda: sdk)

    runtime = market_runtime.MarketRuntime()
    runtime._last_ml_gate_check_mono = 0.0
    runtime._last_ml_prune_mono = time.monotonic()

    payload = [{"id": "btc-15m"}]
    await runtime._refresh_ml_pipeline(payload, allow_record=True)
    await runtime._ml_runtime_refresh_task
    await runtime._refresh_ml_pipeline(payload, allow_record=True)

    sdk.annotate_market_batch.assert_not_awaited()
    sdk.record_market_batch.assert_not_awaited()
    sdk.prune_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_market_runtime_annotates_without_recording(monkeypatch):
    sdk = _FakeMachineLearningSdk(recording_enabled=False, deployment_active=True)
    monkeypatch.setattr(market_runtime, "get_machine_learning_sdk", lambda: sdk)

    runtime = market_runtime.MarketRuntime()
    runtime._last_ml_gate_check_mono = 0.0

    payload = [{"id": "btc-15m"}]
    await runtime._refresh_ml_pipeline(payload, allow_record=True)
    await runtime._ml_runtime_refresh_task
    await runtime._refresh_ml_pipeline(payload, allow_record=True)

    sdk.annotate_market_batch.assert_awaited_once_with(task_key="crypto_directional", markets=payload)
    sdk.record_market_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_market_runtime_reactive_updates_never_record(monkeypatch):
    sdk = _FakeMachineLearningSdk(recording_enabled=True, deployment_active=True)
    monkeypatch.setattr(market_runtime, "get_machine_learning_sdk", lambda: sdk)

    runtime = market_runtime.MarketRuntime()
    runtime._last_ml_gate_check_mono = 0.0

    payload = [{"id": "btc-15m"}]
    await runtime._refresh_ml_pipeline(payload, allow_record=False)
    await runtime._ml_runtime_refresh_task
    await runtime._refresh_ml_pipeline(payload, allow_record=False)

    sdk.annotate_market_batch.assert_awaited_once_with(task_key="crypto_directional", markets=payload)
    sdk.record_market_batch.assert_not_awaited()
    sdk.prune_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_market_runtime_resolves_ml_runtime_state_with_timeout_fallback(monkeypatch):
    async def _slow_runtime_state(_task_key):
        await asyncio.sleep(0.02)
        return {
            "task_key": "crypto_directional",
            "recording_enabled": True,
            "deployment_active": True,
            "engaged": True,
        }

    sdk = _FakeMachineLearningSdk(recording_enabled=True, deployment_active=True)
    sdk.get_runtime_state = AsyncMock(side_effect=_slow_runtime_state)
    monkeypatch.setattr(market_runtime, "get_machine_learning_sdk", lambda: sdk)
    monkeypatch.setattr(market_runtime, "_ML_RUNTIME_STATE_TIMEOUT_SECONDS", 0.001)

    runtime = market_runtime.MarketRuntime()
    runtime._last_ml_gate_check_mono = 0.0

    result = await runtime._resolve_ml_runtime_state(allow_record=True)

    assert result is None
    assert runtime._ml_runtime_recording_enabled is False
    assert runtime._ml_runtime_deployment_active is False
