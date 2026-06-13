import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import market_runtime, reference_runtime


class _FakeBinanceFeed:
    def __init__(self) -> None:
        self._callback = None

    def on_update(self, callback) -> None:
        self._callback = callback

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def emit(self, asset: str, mid: float, bid: float, ask: float, timestamp_ms: int) -> None:
        assert self._callback is not None
        self._callback(asset, mid, bid, ask, timestamp_ms)


class _FakeChainlinkFeed:
    def __init__(self) -> None:
        self._callback = None
        self.binance_updates: list[tuple[str, float, float, float, int]] = []

    def on_update(self, callback) -> None:
        self._callback = callback

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def update_from_binance_direct(
        self,
        asset: str,
        mid: float,
        bid: float,
        ask: float,
        timestamp_ms: int,
    ) -> None:
        self.binance_updates.append((asset, mid, bid, ask, timestamp_ms))

    def emit(self, *, asset: str, source: str) -> None:
        assert self._callback is not None
        self._callback(SimpleNamespace(asset=asset, source=source))


class _FakeReferenceRuntime:
    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = {asset.upper(): price for asset, price in prices.items()}
        self._callbacks = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def on_update(self, callback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_on_update(self, callback) -> None:
        self._callbacks = [registered for registered in self._callbacks if registered != callback]

    def get_oracle_price(self, asset: str):
        normalized = str(asset or "").upper()
        price = self._prices.get(normalized)
        if price is None:
            return None
        updated_at_ms = 1_700_000_000_000
        return {
            "asset": normalized,
            "price": price,
            "updated_at_ms": updated_at_ms,
            "age_seconds": 0.05,
            "source": "chainlink",
        }

    def get_oracle_prices_by_source(self, asset: str):
        oracle = self.get_oracle_price(asset)
        if oracle is None:
            return {}
        return {
            "chainlink": {
                "source": "chainlink",
                "price": oracle["price"],
                "updated_at_ms": oracle["updated_at_ms"],
                "age_seconds": oracle["age_seconds"],
            }
        }

    def get_oracle_history(self, asset: str, *, points: int = 80, max_age_seconds: float | None = None):
        del max_age_seconds  # fake doesn't filter
        oracle = self.get_oracle_price(asset)
        if oracle is None:
            return []
        return [{"t": oracle["updated_at_ms"], "p": oracle["price"]}]

    def get_oracle_motion_summary(self, asset: str):
        return {}

    def get_status(self):
        return {}


class _FakeAsyncSessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_reference_runtime_notifies_on_binance_and_chainlink_updates(monkeypatch):
    fake_binance = _FakeBinanceFeed()
    fake_chainlink = _FakeChainlinkFeed()
    monkeypatch.setattr(reference_runtime, "get_binance_feed", lambda: fake_binance)
    monkeypatch.setattr(reference_runtime, "get_chainlink_feed", lambda: fake_chainlink)

    runtime = reference_runtime.ReferenceRuntime()
    seen: list[str] = []
    runtime.on_update(seen.append)

    fake_binance.emit("btc", 70001.25, 70001.0, 70001.5, 1_700_000_000_001)
    fake_chainlink.emit(asset="ETH", source="chainlink")
    fake_chainlink.emit(asset="BTC", source="binance_direct")

    assert fake_chainlink.binance_updates == [
        ("btc", 70001.25, 70001.0, 70001.5, 1_700_000_000_001)
    ]
    assert seen == ["BTC", "ETH"]


@pytest.mark.asyncio
async def test_market_runtime_reacts_to_reference_updates_without_waiting_for_periodic_scan(monkeypatch):
    fake_reference = _FakeReferenceRuntime({"BTC": 70000.0, "ETH": 3500.0})
    monkeypatch.setattr(market_runtime, "get_reference_runtime", lambda: fake_reference)
    monkeypatch.setattr(market_runtime, "_WS_REACTIVE_DEBOUNCE_SECONDS", 0.0)

    runtime = market_runtime.MarketRuntime()
    runtime._reference_runtime = fake_reference
    runtime._feed_manager = SimpleNamespace(_started=False)
    runtime._started = True
    runtime._crypto_markets = [
        {
            "id": "btc-15m",
            "slug": "btc-15m",
            "asset": "BTC",
            "clob_token_ids": ["btc-up", "btc-down"],
            "oracle_price": 70000.0,
            "oracle_source": "chainlink",
            "oracle_updated_at_ms": 1_700_000_000_000,
            "oracle_age_seconds": 0.05,
            "oracle_prices_by_source": {},
            "oracle_history": [],
        },
        {
            "id": "eth-15m",
            "slug": "eth-15m",
            "asset": "ETH",
            "clob_token_ids": ["eth-up", "eth-down"],
            "oracle_price": 3500.0,
            "oracle_source": "chainlink",
            "oracle_updated_at_ms": 1_700_000_000_000,
            "oracle_age_seconds": 0.05,
            "oracle_prices_by_source": {},
            "oracle_history": [],
        },
    ]
    runtime._crypto_markets_by_lookup = {}
    runtime._crypto_token_to_market_ids = {}
    runtime._crypto_asset_to_market_ids = {}
    for row in runtime._crypto_markets:
        runtime._index_crypto_market_row(row)

    published: list[tuple[list[dict], str]] = []

    async def _capture_publish(payload, *, trigger):
        published.append((payload, trigger))

    async def _skip_opportunity_dispatch(payload, *, trigger, full_source_sweep):
        return None

    monkeypatch.setattr(runtime, "_publish_crypto_snapshot", _capture_publish)
    monkeypatch.setattr(runtime, "_queue_opportunity_dispatch", _skip_opportunity_dispatch)

    fake_reference._prices["BTC"] = 70123.45
    runtime._on_reference_update("BTC")
    await asyncio.sleep(0)
    assert runtime._reactive_task is not None
    await runtime._reactive_task

    assert len(published) == 1
    payload, trigger = published[0]
    assert trigger == "reference_ws"
    assert [row["id"] for row in payload] == ["btc-15m"]
    assert payload[0]["oracle_price"] == 70123.45
    assert runtime._crypto_markets[0]["oracle_price"] == 70123.45
    assert runtime._crypto_markets[1]["oracle_price"] == 3500.0
    assert runtime._last_crypto_trigger == "reference_ws"
    assert runtime._last_crypto_refresh_at is not None


def test_rebuild_crypto_rows_from_cache_refreshes_time_derived_fields():
    fake_reference = _FakeReferenceRuntime({"BTC": 70000.0})
    runtime = market_runtime.MarketRuntime()
    runtime._reference_runtime = fake_reference
    runtime._feed_manager = SimpleNamespace(_started=False)

    now = market_runtime.utcnow()
    rebuilt = runtime._rebuild_crypto_rows_from_cache(
        [
            {
                "id": "btc-5m",
                "slug": "btc-5m",
                "asset": "BTC",
                "timeframe": "5min",
                "start_time": (now - timedelta(seconds=15)).isoformat().replace("+00:00", "Z"),
                "end_time": (now + timedelta(seconds=9)).isoformat().replace("+00:00", "Z"),
                "seconds_left": 120,
                "is_live": False,
                "is_current": False,
                "up_price": 0.52,
                "down_price": 0.48,
                "combined": 0.10,
                "clob_token_ids": ["btc-up", "btc-down"],
                "oracle_price": None,
                "oracle_source": None,
                "oracle_updated_at_ms": None,
                "oracle_age_seconds": None,
                "oracle_prices_by_source": {},
                "oracle_history": [],
            }
        ]
    )

    assert len(rebuilt) == 1
    row = rebuilt[0]
    assert row["is_live"] is True
    assert row["is_current"] is True
    assert isinstance(row["seconds_left"], int)
    assert 0 <= row["seconds_left"] <= 9
    assert row["combined"] == pytest.approx(1.0)
    assert row["oracle_price"] == 70000.0
    assert row["oracle_source"] == "chainlink"


@pytest.mark.asyncio
async def test_refresh_event_catalog_skips_full_reload_when_catalog_unchanged(monkeypatch):
    runtime = market_runtime.MarketRuntime()
    updated_at = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
    metadata_only = ([], [], {"updated_at": updated_at})
    full_catalog = (
        [],
        [
            {
                "id": "market-1",
                "condition_id": "condition-1",
                "clob_token_ids": ["yes-1", "no-1"],
            }
        ],
        {"updated_at": updated_at},
    )
    read_catalog = AsyncMock(side_effect=[metadata_only, full_catalog, metadata_only])

    monkeypatch.setattr(market_runtime, "AsyncSessionLocal", lambda: _FakeAsyncSessionContext())
    monkeypatch.setattr(market_runtime.shared_state, "read_market_catalog", read_catalog)
    monkeypatch.setattr(market_runtime, "_CATALOG_REFRESH_SECONDS", 0.0)

    await runtime._refresh_event_catalog(force=True)
    assert runtime._event_catalog_updated_at == updated_at.isoformat()
    assert runtime._event_catalog_markets["market-1"]["id"] == "market-1"

    runtime._last_catalog_refresh_mono = 0.0
    await runtime._refresh_event_catalog()

    assert read_catalog.await_count == 3
    assert runtime._event_catalog_markets["condition-1"]["id"] == "market-1"


@pytest.mark.asyncio
async def test_run_loop_iteration_schedules_catalog_refresh_without_waiting(monkeypatch):
    runtime = market_runtime.MarketRuntime()
    runtime._last_catalog_refresh_mono = 0.0

    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    refresh_calls: list[bool] = []
    crypto_calls: list[str] = []

    async def _slow_refresh(*, force: bool = False):
        refresh_calls.append(bool(force))
        refresh_started.set()
        await release_refresh.wait()

    async def _read_control():
        return {"is_enabled": True, "is_paused": False, "interval_seconds": 0.01}

    async def _refresh_crypto_markets(*, trigger: str, full_source_sweep: bool, force_refresh: bool = False):
        crypto_calls.append(str(trigger))

    monkeypatch.setattr(runtime, "_refresh_event_catalog", _slow_refresh)
    monkeypatch.setattr(runtime, "_read_crypto_control", _read_control)
    monkeypatch.setattr(runtime, "_refresh_crypto_markets", _refresh_crypto_markets)
    monkeypatch.setattr(market_runtime, "_near_market_boundary", lambda: False)
    monkeypatch.setattr(market_runtime, "_FULL_REFRESH_FLOOR_SECONDS", 0.0)
    monkeypatch.setattr(market_runtime, "_CATALOG_REFRESH_SECONDS", 0.0)

    started_at = time.perf_counter()
    await runtime._run_loop_iteration()
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.15
    assert crypto_calls == ["periodic_scan"]
    assert runtime._event_catalog_refresh_task is not None
    await asyncio.wait_for(refresh_started.wait(), timeout=1.0)

    release_refresh.set()
    await asyncio.wait_for(runtime._event_catalog_refresh_task, timeout=1.0)
    assert refresh_calls == [False]


@pytest.mark.asyncio
async def test_start_schedules_event_catalog_refresh_without_blocking_startup(monkeypatch):
    runtime = market_runtime.MarketRuntime()
    runtime._reference_runtime = SimpleNamespace(start=AsyncMock(), on_update=lambda *_args, **_kwargs: None)
    fake_feed_manager = SimpleNamespace(
        _started=False,
        start=AsyncMock(),
        cache=SimpleNamespace(add_on_update_callback=lambda *_args, **_kwargs: None),
    )
    refresh_calls: list[bool] = []
    refresh_crypto = AsyncMock()

    monkeypatch.setattr(market_runtime, "get_feed_manager", lambda: fake_feed_manager)
    monkeypatch.setattr(runtime, "_schedule_event_catalog_refresh", lambda *, force=False: refresh_calls.append(bool(force)))
    monkeypatch.setattr(runtime, "_refresh_crypto_markets", refresh_crypto)

    await runtime.start()

    runtime._reference_runtime.start.assert_awaited_once()
    fake_feed_manager.start.assert_awaited_once()
    refresh_crypto.assert_awaited_once_with(trigger="startup", full_source_sweep=True)
    assert refresh_calls == [True]
    assert runtime.started is True


def test_get_market_snapshot_schedules_forced_catalog_refresh_on_event_market_miss(monkeypatch):
    runtime = market_runtime.MarketRuntime()
    runtime._started = True
    runtime._last_catalog_refresh_mono = 0.0
    refresh_calls: list[bool] = []

    monkeypatch.setattr(market_runtime, "_CATALOG_MISS_REFRESH_SECONDS", 0.0)
    monkeypatch.setattr(runtime, "_schedule_event_catalog_refresh", lambda *, force=False: refresh_calls.append(bool(force)))

    snapshot = runtime.get_market_snapshot(
        "missing-market",
        hint={"condition_id": "missing-market", "clob_token_ids": ["yes-miss", "no-miss"]},
    )

    assert snapshot == {"condition_id": "missing-market", "clob_token_ids": ["yes-miss", "no-miss"]}
    assert refresh_calls == [True]


def test_get_market_snapshot_does_not_hydrate_event_catalog_when_inactive(monkeypatch):
    runtime = market_runtime.MarketRuntime()
    runtime._last_catalog_refresh_mono = 0.0
    refresh_calls: list[bool] = []

    monkeypatch.setattr(market_runtime, "_CATALOG_MISS_REFRESH_SECONDS", 0.0)
    monkeypatch.setattr(runtime, "_schedule_event_catalog_refresh", lambda *, force=False: refresh_calls.append(bool(force)))

    snapshot = runtime.get_market_snapshot(
        "missing-market",
        hint={"condition_id": "missing-market", "clob_token_ids": ["yes-miss", "no-miss"]},
    )

    assert snapshot == {"condition_id": "missing-market", "clob_token_ids": ["yes-miss", "no-miss"]}
    assert refresh_calls == []
    assert runtime._event_catalog_markets == {}


@pytest.mark.asyncio
async def test_refresh_crypto_markets_publishes_snapshot_without_waiting_for_opportunity_dispatch(monkeypatch):
    fake_reference = _FakeReferenceRuntime({"BTC": 70000.0})
    monkeypatch.setattr(market_runtime, "get_reference_runtime", lambda: fake_reference)

    runtime = market_runtime.MarketRuntime()
    runtime._reference_runtime = fake_reference
    runtime._feed_manager = SimpleNamespace(_started=False)

    published: list[tuple[list[dict], str]] = []

    async def _capture_publish(payload, *, trigger):
        published.append((payload, trigger))

    async def _noop_sync_subscriptions():
        return None

    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    dispatch_payloads: list[tuple[list[dict], str]] = []

    async def _slow_dispatch(event):
        dispatch_payloads.append((event.payload["markets"], event.payload["trigger"]))
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    class _FakeIntentRuntime:
        async def publish_opportunities(self, opportunities, **kwargs):
            return 0

    class _FakeCryptoService:
        def get_live_markets(self, force_refresh):
            return ["ignored"]

    async def _noop_attach_history(_payload):
        return None

    monkeypatch.setattr(runtime, "_publish_crypto_snapshot", _capture_publish)
    monkeypatch.setattr(runtime, "_sync_crypto_subscriptions", _noop_sync_subscriptions)
    monkeypatch.setattr(runtime, "_attach_polymarket_price_history", _noop_attach_history)
    monkeypatch.setattr(
        runtime,
        "_build_crypto_market_payload",
        lambda markets: [
            {
                "id": "btc-15m",
                "slug": "btc-15m",
                "asset": "BTC",
                "clob_token_ids": ["btc-up", "btc-down"],
                "oracle_price": 70000.0,
                "oracle_source": "chainlink",
                "oracle_updated_at_ms": 1_700_000_000_000,
                "oracle_age_seconds": 0.05,
                "oracle_prices_by_source": {},
                "oracle_history": [],
            }
        ],
    )
    monkeypatch.setattr(market_runtime, "get_crypto_service", lambda: _FakeCryptoService())
    monkeypatch.setattr(market_runtime, "_dispatch_with_per_trader_fanout", _slow_dispatch)
    monkeypatch.setattr(market_runtime, "get_intent_runtime", lambda: _FakeIntentRuntime())

    started_at = time.perf_counter()
    await runtime._refresh_crypto_markets(trigger="startup", full_source_sweep=True)
    elapsed = time.perf_counter() - started_at

    assert len(published) == 1
    assert published[0][1] == "startup"
    assert elapsed < 0.5

    await asyncio.wait_for(dispatch_started.wait(), timeout=1.0)
    release_dispatch.set()
    assert runtime._opportunity_dispatch_task is not None
    await asyncio.wait_for(runtime._opportunity_dispatch_task, timeout=1.0)

    assert len(dispatch_payloads) == 1
    assert dispatch_payloads[0][1] == "startup"
    assert dispatch_payloads[0][0][0]["id"] == "btc-15m"


@pytest.mark.asyncio
async def test_opportunity_dispatch_coalesces_to_latest_pending_payload(monkeypatch):
    fake_reference = _FakeReferenceRuntime({"BTC": 70000.0})
    monkeypatch.setattr(market_runtime, "get_reference_runtime", lambda: fake_reference)

    runtime = market_runtime.MarketRuntime()
    runtime._reference_runtime = fake_reference

    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    seen_prices: list[float] = []

    async def _slow_dispatch(event):
        seen_prices.append(event.payload["markets"][0]["oracle_price"])
        if len(seen_prices) == 1:
            dispatch_started.set()
            await release_dispatch.wait()
        return []

    class _FakeIntentRuntime:
        async def publish_opportunities(self, opportunities, **kwargs):
            return 0

    monkeypatch.setattr(market_runtime, "_dispatch_with_per_trader_fanout", _slow_dispatch)
    monkeypatch.setattr(market_runtime, "get_intent_runtime", lambda: _FakeIntentRuntime())

    await runtime._queue_opportunity_dispatch(
        [{"id": "btc-15m", "oracle_price": 70000.0}],
        trigger="first",
        full_source_sweep=False,
    )
    await asyncio.wait_for(dispatch_started.wait(), timeout=1.0)

    await runtime._queue_opportunity_dispatch(
        [{"id": "btc-15m", "oracle_price": 70010.0}],
        trigger="second",
        full_source_sweep=False,
    )
    await runtime._queue_opportunity_dispatch(
        [{"id": "btc-15m", "oracle_price": 70020.0}],
        trigger="third",
        full_source_sweep=False,
    )

    release_dispatch.set()
    assert runtime._opportunity_dispatch_task is not None
    await asyncio.wait_for(runtime._opportunity_dispatch_task, timeout=1.0)

    assert seen_prices == [70000.0, 70020.0]


def test_build_crypto_filter_diagnostics_aggregates_per_strategy():
    class _FakeStrategy:
        def __init__(self, diag):
            self._diag = diag

        def get_filter_diagnostics(self):
            return self._diag

    diagnostics = market_runtime._build_crypto_filter_diagnostics(
        [
            (
                "alpha",
                _FakeStrategy(
                    {
                        "markets_scanned": 15,
                        "signals_emitted": 0,
                        "rejections": [{"gate": "oracle_move"}, {"gate": "oracle_move"}],
                        "message": "alpha filtered",
                        "summary": {"oracle_move": 2},
                    }
                ),
            ),
            (
                "beta",
                _FakeStrategy(
                    {
                        "markets_scanned": 15,
                        "signals_emitted": 0,
                        "rejections": {"low_liquidity": 4},
                        "message": "beta filtered",
                        "summary": {"rejected_low_liquidity": 4},
                    }
                ),
            ),
            ("gamma", _FakeStrategy(None)),
        ],
        [
            SimpleNamespace(strategy="alpha"),
            SimpleNamespace(strategy="beta"),
            SimpleNamespace(strategy="beta"),
        ],
    )

    assert diagnostics["signals_emitted"] == 3
    assert diagnostics["markets_scanned"] == 15
    assert diagnostics["strategies"]["alpha"]["signals_emitted"] == 1
    assert diagnostics["strategies"]["beta"]["signals_emitted"] == 2
    assert diagnostics["dispatch_summary"]["strategies_missing_diagnostics"] == ["gamma"]
    assert diagnostics["dispatch_summary"]["rejection_counts_by_strategy"]["alpha"]["oracle_move"] == 2
    assert diagnostics["dispatch_summary"]["opportunities_by_strategy"] == {"alpha": 1, "beta": 2}
