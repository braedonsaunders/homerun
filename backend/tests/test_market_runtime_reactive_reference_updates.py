import asyncio
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

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

    def get_oracle_history(self, asset: str, *, points: int = 80):
        oracle = self.get_oracle_price(asset)
        if oracle is None:
            return []
        return [{"t": oracle["updated_at_ms"], "p": oracle["price"]}]

    def get_status(self):
        return {}


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

    monkeypatch.setattr(runtime, "_publish_crypto_snapshot", _capture_publish)
    monkeypatch.setattr(runtime, "_sync_crypto_subscriptions", _noop_sync_subscriptions)
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
    monkeypatch.setattr(market_runtime.event_dispatcher, "dispatch", _slow_dispatch)
    monkeypatch.setattr(market_runtime, "get_intent_runtime", lambda: _FakeIntentRuntime())

    started_at = time.perf_counter()
    await runtime._refresh_crypto_markets(trigger="startup", full_source_sweep=True)
    elapsed = time.perf_counter() - started_at

    assert len(published) == 1
    assert published[0][1] == "startup"
    assert elapsed < 0.15

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

    monkeypatch.setattr(market_runtime.event_dispatcher, "dispatch", _slow_dispatch)
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
