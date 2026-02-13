import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_crypto


@pytest.mark.asyncio
async def test_get_crypto_markets_uses_fresh_worker_snapshot(monkeypatch):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    near_future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat().replace(
        "+00:00", "Z"
    )
    snapshot_markets = [
        {"id": "m1", "asset": "BTC", "timeframe": "15min", "end_time": near_future}
    ]

    monkeypatch.setattr(
        routes_crypto,
        "read_worker_snapshot",
        AsyncMock(
            return_value={
                "updated_at": now,
                "stats": {"markets": snapshot_markets},
            }
        ),
    )
    fallback_mock = AsyncMock(return_value=[{"id": "live"}])
    monkeypatch.setattr(routes_crypto, "_build_live_markets_from_source", fallback_mock)
    monkeypatch.setattr(routes_crypto, "_configured_timeframes", lambda: {"15m"})

    out = await routes_crypto.get_crypto_markets(object())
    assert out == snapshot_markets
    fallback_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_crypto_markets_falls_back_when_snapshot_stale(monkeypatch):
    stale = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat().replace(
        "+00:00", "Z"
    )
    snapshot_markets = [{"id": "stale-market", "asset": "BTC"}]
    live_markets = [{"id": "live-market", "asset": "BTC"}]

    monkeypatch.setattr(
        routes_crypto,
        "read_worker_snapshot",
        AsyncMock(
            return_value={
                "updated_at": stale,
                "stats": {"markets": snapshot_markets},
            }
        ),
    )
    fallback_mock = AsyncMock(return_value=live_markets)
    monkeypatch.setattr(routes_crypto, "_build_live_markets_from_source", fallback_mock)

    out = await routes_crypto.get_crypto_markets(object())
    assert out == live_markets
    fallback_mock.assert_awaited_once_with(snapshot_markets)


@pytest.mark.asyncio
async def test_get_crypto_markets_falls_back_when_snapshot_only_far_future(monkeypatch):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    # Nearest expiry ~22h out should fail the snapshot sanity check.
    far_future = (datetime.now(timezone.utc) + timedelta(hours=22)).isoformat().replace(
        "+00:00", "Z"
    )
    snapshot_markets = [
        {"id": "future-market", "asset": "BTC", "timeframe": "15min", "end_time": far_future}
    ]
    live_markets = [{"id": "live-market", "asset": "BTC", "timeframe": "15min"}]

    monkeypatch.setattr(
        routes_crypto,
        "read_worker_snapshot",
        AsyncMock(
            return_value={
                "updated_at": now,
                "stats": {"markets": snapshot_markets},
            }
        ),
    )
    monkeypatch.setattr(routes_crypto, "_configured_timeframes", lambda: {"15m"})
    fallback_mock = AsyncMock(return_value=live_markets)
    monkeypatch.setattr(routes_crypto, "_build_live_markets_from_source", fallback_mock)

    out = await routes_crypto.get_crypto_markets(object())
    assert out == live_markets
    fallback_mock.assert_awaited_once_with(snapshot_markets)


@pytest.mark.asyncio
async def test_source_fallback_hydrates_price_to_beat_from_snapshot(monkeypatch):
    class FakeMarket:
        slug = "btc-updown-123"
        asset = "BTC"

        def to_dict(self) -> dict:
            return {
                "id": "m1",
                "slug": self.slug,
                "asset": self.asset,
                "oracle_price": None,
                "oracle_updated_at_ms": None,
                "oracle_age_seconds": None,
                "price_to_beat": None,
                "oracle_history": [],
            }

    svc = SimpleNamespace(
        _price_to_beat={},
        get_live_markets=lambda force_refresh=False: [FakeMarket()],
        _update_price_to_beat=lambda markets: None,
    )
    feed = SimpleNamespace(
        started=True,
        get_price=lambda asset: None,
        _history={},
    )

    monkeypatch.setattr(routes_crypto, "get_crypto_service", lambda: svc)
    monkeypatch.setattr(routes_crypto, "get_chainlink_feed", lambda: feed)

    out = await routes_crypto._build_live_markets_from_source(
        [{"slug": "btc-updown-123", "price_to_beat": 67890.12}]
    )
    assert out and out[0]["price_to_beat"] == 67890.12
    assert svc._price_to_beat["btc-updown-123"] == 67890.12
