"""Crypto fast-binary lane toggle behaviour in ``MarketRuntime``."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import market_runtime


def _make_runtime() -> market_runtime.MarketRuntime:
    runtime = market_runtime.MarketRuntime()
    runtime._reference_runtime = SimpleNamespace(
        start=AsyncMock(),
        on_update=lambda *_a, **_k: None,
    )
    return runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [
        {
            "is_enabled": False,
            "is_paused": False,
            "interval_seconds": 1,
            "requested_run_at": None,
        },
        {
            "is_enabled": True,
            "is_paused": True,
            "interval_seconds": 1,
            "requested_run_at": None,
        },
    ],
    ids=["disabled", "paused"],
)
async def test_start_skips_crypto_refresh_when_lane_off(monkeypatch, control):
    runtime = _make_runtime()
    fake_feed_manager = SimpleNamespace(
        _started=False,
        start=AsyncMock(),
        cache=SimpleNamespace(add_on_update_callback=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(market_runtime, "get_feed_manager", lambda: fake_feed_manager)
    monkeypatch.setattr(
        runtime, "_schedule_event_catalog_refresh", lambda *, force=False: None
    )

    refresh_crypto = AsyncMock()
    monkeypatch.setattr(runtime, "_refresh_crypto_markets", refresh_crypto)

    async def _read_control() -> dict[str, object]:
        return control

    monkeypatch.setattr(runtime, "_read_crypto_control", _read_control)

    runtime._crypto_markets = [{"id": "stale-row"}]
    runtime._crypto_markets_by_lookup = {"stale": {}}
    runtime._crypto_token_to_market_ids = {"stale": {"stale-row"}}
    runtime._crypto_asset_to_market_ids = {"BTC": {"stale-row"}}

    await runtime.start()

    refresh_crypto.assert_not_awaited()
    assert runtime._crypto_markets == []
    assert runtime._crypto_markets_by_lookup == {}
    assert runtime._crypto_token_to_market_ids == {}
    assert runtime._crypto_asset_to_market_ids == {}
    assert runtime._crypto_lane_was_enabled is False


@pytest.mark.asyncio
async def test_start_runs_crypto_refresh_when_lane_enabled(monkeypatch):
    runtime = _make_runtime()
    fake_feed_manager = SimpleNamespace(
        _started=False,
        start=AsyncMock(),
        cache=SimpleNamespace(add_on_update_callback=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(market_runtime, "get_feed_manager", lambda: fake_feed_manager)
    monkeypatch.setattr(
        runtime, "_schedule_event_catalog_refresh", lambda *, force=False: None
    )

    refresh_crypto = AsyncMock()
    monkeypatch.setattr(runtime, "_refresh_crypto_markets", refresh_crypto)

    async def _read_control() -> dict[str, object]:
        return {
            "is_enabled": True,
            "is_paused": False,
            "interval_seconds": 1,
            "requested_run_at": None,
        }

    monkeypatch.setattr(runtime, "_read_crypto_control", _read_control)

    await runtime.start()

    refresh_crypto.assert_awaited_once_with(trigger="startup", full_source_sweep=True)
    assert runtime._crypto_lane_was_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [
        {"is_enabled": False, "is_paused": False, "interval_seconds": 1},
        {"is_enabled": True, "is_paused": True, "interval_seconds": 1},
    ],
    ids=["disabled", "paused"],
)
async def test_drain_reactive_updates_returns_early_when_lane_off(
    monkeypatch, control
):
    runtime = _make_runtime()
    monkeypatch.setattr(market_runtime, "_WS_REACTIVE_DEBOUNCE_SECONDS", 0.0)

    runtime._pending_tokens = {"btc-up", "btc-down"}
    runtime._pending_assets = {"BTC"}
    runtime._crypto_markets = [
        {"id": "btc-15m", "slug": "btc-15m", "asset": "BTC", "clob_token_ids": ["btc-up"]}
    ]
    runtime._crypto_markets_by_lookup = {}
    runtime._crypto_token_to_market_ids = {"btc-up": {"btc-15m"}}
    runtime._crypto_asset_to_market_ids = {"BTC": {"btc-15m"}}

    async def _read_cached(*, ttl_seconds: float = 5.0) -> dict[str, object]:
        return control

    monkeypatch.setattr(runtime, "_read_crypto_control_cached", _read_cached)

    rebuild_called = False

    def _rebuild(rows):
        nonlocal rebuild_called
        rebuild_called = True
        return rows

    monkeypatch.setattr(runtime, "_rebuild_crypto_rows_from_cache", _rebuild)

    publish_calls: list[tuple] = []

    async def _publish(payload, *, trigger):
        publish_calls.append((payload, trigger))

    monkeypatch.setattr(runtime, "_publish_crypto_snapshot", _publish)

    await runtime._drain_reactive_updates()

    assert rebuild_called is False
    assert publish_calls == []
    assert runtime._pending_tokens == set()
    assert runtime._pending_assets == set()


@pytest.mark.asyncio
async def test_drain_reactive_updates_runs_when_lane_enabled(monkeypatch):
    runtime = _make_runtime()
    monkeypatch.setattr(market_runtime, "_WS_REACTIVE_DEBOUNCE_SECONDS", 0.0)

    runtime._pending_tokens = set()
    runtime._pending_assets = {"BTC"}
    runtime._crypto_markets = [
        {
            "id": "btc-15m",
            "slug": "btc-15m",
            "asset": "BTC",
            "clob_token_ids": ["btc-up"],
            "oracle_price": 70000.0,
            "oracle_history": [],
        }
    ]
    runtime._crypto_markets_by_lookup = {}
    runtime._crypto_token_to_market_ids = {"btc-up": {"btc-15m"}}
    runtime._crypto_asset_to_market_ids = {"BTC": {"btc-15m"}}

    async def _read_cached(*, ttl_seconds: float = 5.0) -> dict[str, object]:
        return {"is_enabled": True, "is_paused": False, "interval_seconds": 1}

    monkeypatch.setattr(runtime, "_read_crypto_control_cached", _read_cached)

    def _rebuild(rows):
        return [dict(row) for row in rows]

    monkeypatch.setattr(runtime, "_rebuild_crypto_rows_from_cache", _rebuild)

    async def _queue_ml(*_args, **_kwargs):
        return None

    monkeypatch.setattr(runtime, "_queue_ml_pipeline_refresh", _queue_ml)

    publish_calls: list[tuple] = []

    async def _publish(payload, *, trigger):
        publish_calls.append((payload, trigger))

    monkeypatch.setattr(runtime, "_publish_crypto_snapshot", _publish)

    async def _queue_dispatch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(runtime, "_queue_opportunity_dispatch", _queue_dispatch)

    await runtime._drain_reactive_updates()

    assert len(publish_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [
        {"is_enabled": False, "is_paused": False, "interval_seconds": 1},
        {"is_enabled": True, "is_paused": True, "interval_seconds": 1},
    ],
    ids=["disabled", "paused"],
)
async def test_run_loop_iteration_clears_cache_on_active_to_off_transition(
    monkeypatch, control
):
    runtime = _make_runtime()
    runtime._last_catalog_refresh_mono = float("inf")
    runtime._crypto_lane_was_enabled = True
    runtime._crypto_markets = [{"id": "btc-15m"}]
    runtime._crypto_markets_by_lookup = {"btc-15m": {}}
    runtime._crypto_token_to_market_ids = {"btc-up": {"btc-15m"}}
    runtime._crypto_asset_to_market_ids = {"BTC": {"btc-15m"}}

    monkeypatch.setattr(market_runtime, "_FULL_REFRESH_FLOOR_SECONDS", 0.0)

    async def _read_control() -> dict[str, object]:
        return control

    monkeypatch.setattr(runtime, "_read_crypto_control", _read_control)

    refresh_crypto = AsyncMock()
    monkeypatch.setattr(runtime, "_refresh_crypto_markets", refresh_crypto)

    async def _persist(*, control, force):
        return None

    monkeypatch.setattr(runtime, "_persist_crypto_worker_snapshot", _persist)

    await runtime._run_loop_iteration()

    refresh_crypto.assert_not_awaited()
    assert runtime._crypto_markets == []
    assert runtime._crypto_markets_by_lookup == {}
    assert runtime._crypto_token_to_market_ids == {}
    assert runtime._crypto_asset_to_market_ids == {}
    assert runtime._crypto_lane_was_enabled is False


@pytest.mark.asyncio
async def test_run_loop_iteration_triggers_refresh_on_disabled_to_enabled_transition(
    monkeypatch,
):
    runtime = _make_runtime()
    runtime._last_catalog_refresh_mono = float("inf")
    runtime._crypto_lane_was_enabled = False
    runtime._crypto_markets = []

    monkeypatch.setattr(market_runtime, "_FULL_REFRESH_FLOOR_SECONDS", 0.0)
    monkeypatch.setattr(market_runtime, "_near_market_boundary", lambda: False)

    async def _read_control() -> dict[str, object]:
        return {"is_enabled": True, "is_paused": False, "interval_seconds": 1}

    monkeypatch.setattr(runtime, "_read_crypto_control", _read_control)

    refresh_crypto = AsyncMock()
    monkeypatch.setattr(runtime, "_refresh_crypto_markets", refresh_crypto)

    await runtime._run_loop_iteration()

    refresh_crypto.assert_awaited_once()
    kwargs = refresh_crypto.await_args.kwargs
    assert kwargs["trigger"] == "lane_re_enabled"
    assert kwargs["full_source_sweep"] is True
    assert kwargs["force_refresh"] is True
    assert runtime._crypto_lane_was_enabled is True
    assert runtime._crypto_lane_pending_refresh is False


@pytest.mark.asyncio
async def test_read_crypto_control_cached_uses_ttl(monkeypatch):
    runtime = _make_runtime()
    call_count = 0

    async def _read_control() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"is_enabled": True, "is_paused": False, "interval_seconds": 1}

    monkeypatch.setattr(runtime, "_read_crypto_control", _read_control)

    first = await runtime._read_crypto_control_cached(ttl_seconds=5.0)
    second = await runtime._read_crypto_control_cached(ttl_seconds=5.0)

    assert first is second
    assert call_count == 1

    runtime._crypto_control_cache_at -= 100.0
    third = await runtime._read_crypto_control_cached(ttl_seconds=5.0)
    assert call_count == 2
    assert third["is_enabled"] is True
