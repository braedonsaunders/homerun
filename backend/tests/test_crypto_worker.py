import sys
import asyncio
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import crypto_worker


class _FakePriceCache:
    def __init__(
        self,
        mids: dict[str, float],
        fresh: set[str],
        trade_totals: dict[str, float] | None = None,
    ):
        self._mids = dict(mids)
        self._fresh = set(fresh)
        self._trade_totals = dict(trade_totals or {})

    def is_fresh(self, token_id: str) -> bool:
        return token_id in self._fresh

    def get_mid(self, token_id: str):
        return self._mids.get(token_id)

    def get_trade_volume(self, token_id: str, lookback_seconds: float = 300.0):
        total = float(self._trade_totals.get(token_id, 0.0) or 0.0)
        return {"buy_volume": total, "sell_volume": 0.0, "total": total, "trade_count": 1 if total > 0 else 0}


def test_collect_ws_prices_for_markets_uses_only_fresh_valid_tokens():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    stale_token = "stale_token_12345678901234567890"

    markets = [
        SimpleNamespace(clob_token_ids=[up_token, down_token, stale_token, "short_token"]),
    ]
    feed_manager = SimpleNamespace(
        _started=True,
        price_cache=_FakePriceCache(
            mids={
                up_token: 0.623,
                down_token: 1.009,
                stale_token: 0.42,
            },
            fresh={up_token, down_token},
        ),
    )

    out = crypto_worker._collect_ws_prices_for_markets(feed_manager, markets)

    assert out == {
        up_token: 0.623,
        down_token: 1.0,
    }


def test_overlay_ws_prices_on_market_row_prefers_ws_for_both_legs():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    market = SimpleNamespace(
        clob_token_ids=[up_token, down_token],
        up_token_index=0,
        down_token_index=1,
    )
    row = {"up_price": 0.41, "down_price": 0.59, "combined": 1.0}

    crypto_worker._overlay_ws_prices_on_market_row(
        row,
        market,
        {up_token: 0.71, down_token: 0.29},
    )

    assert row["up_price"] == 0.71
    assert row["down_price"] == 0.29
    assert row["combined"] == 1.0


def test_overlay_ws_prices_on_market_row_derives_missing_opposite_leg():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    market = SimpleNamespace(
        clob_token_ids=[up_token, down_token],
        up_token_index=0,
        down_token_index=1,
    )
    row = {"up_price": 0.41, "down_price": 0.59, "combined": 1.0}

    crypto_worker._overlay_ws_prices_on_market_row(
        row,
        market,
        {up_token: 0.82},
    )

    assert row["up_price"] == 0.82
    assert row["down_price"] == pytest.approx(0.18)
    assert row["combined"] == 1.0


def test_overlay_ws_trade_volume_on_market_row_uses_ws_cache_notional():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    market = SimpleNamespace(
        clob_token_ids=[up_token, down_token],
        up_token_index=0,
        down_token_index=1,
        timeframe="15m",
    )
    row = {"volume": 0.0}
    feed_manager = SimpleNamespace(
        _started=True,
        cache=_FakePriceCache(
            mids={},
            fresh=set(),
            trade_totals={up_token: 1250.5, down_token: 980.25},
        ),
    )

    crypto_worker._overlay_ws_trade_volume_on_market_row(row, market, feed_manager)

    assert row["volume_usd"] == pytest.approx(2230.75)
    assert row["volume_usd_source"] == "ws_trade_cache"
    assert row["volume_usd_lookback_seconds"] == 900.0


def test_overlay_ws_trade_volume_on_market_row_keeps_higher_existing_volume_usd():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    market = SimpleNamespace(
        clob_token_ids=[up_token, down_token],
        up_token_index=0,
        down_token_index=1,
        timeframe="5m",
    )
    row = {"volume_usd": 9000.0}
    feed_manager = SimpleNamespace(
        _started=True,
        cache=_FakePriceCache(
            mids={},
            fresh=set(),
            trade_totals={up_token: 1200.0, down_token: 800.0},
        ),
    )

    crypto_worker._overlay_ws_trade_volume_on_market_row(row, market, feed_manager)

    assert row["volume_usd"] == 9000.0
    assert "volume_usd_source" not in row
    assert "volume_usd_lookback_seconds" not in row


def test_oracle_history_payload_uses_stable_tail_window(monkeypatch):
    asset = "BTC"
    history = deque(
        ((idx, float(idx)) for idx in range(1, 181)),
        maxlen=crypto_worker._MAX_ORACLE_HISTORY_POINTS,
    )
    monkeypatch.setattr(crypto_worker, "_oracle_history_by_asset", {asset: history})

    first_payload = crypto_worker._oracle_history_payload(asset)
    assert len(first_payload) == crypto_worker._ORACLE_HISTORY_PAYLOAD_POINTS
    assert first_payload[0] == {"t": 101, "p": 101.0}
    assert first_payload[-1] == {"t": 180, "p": 180.0}

    history.append((181, 181.0))
    second_payload = crypto_worker._oracle_history_payload(asset)
    assert len(second_payload) == crypto_worker._ORACLE_HISTORY_PAYLOAD_POINTS
    assert second_payload[0] == {"t": 102, "p": 102.0}
    assert second_payload[-1] == {"t": 181, "p": 181.0}


def test_build_crypto_market_payload_includes_fetched_at(monkeypatch):
    class _FakeMarket:
        asset = "BTC"
        slug = "btc-window"

        def to_dict(self):
            return {"asset": "BTC", "slug": "btc-window", "clob_token_ids": []}

    fake_service = SimpleNamespace(
        _price_to_beat={},
        _update_price_to_beat=lambda _markets: None,
    )
    fake_feed = SimpleNamespace(
        get_price=lambda _asset: None,
        get_prices_by_source=lambda _asset: {},
    )

    monkeypatch.setattr(crypto_worker, "get_crypto_service", lambda: fake_service)
    monkeypatch.setattr(crypto_worker, "get_chainlink_feed", lambda: fake_feed)

    payload = crypto_worker._build_crypto_market_payload([_FakeMarket()])

    assert len(payload) == 1
    fetched_at = payload[0].get("fetched_at")
    assert isinstance(fetched_at, str)
    assert fetched_at.endswith("Z")


def test_chainlink_feed_staleness_reports_stale_when_age_exceeds_threshold():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    stale_price = SimpleNamespace(updated_at_ms=now_ms - 70_000)
    fake_feed = SimpleNamespace(get_all_prices=lambda: {"BTC": stale_price})

    stale, age_seconds, samples = crypto_worker._chainlink_feed_staleness(fake_feed, max_age_seconds=45.0)

    assert stale is True
    assert age_seconds is not None
    assert age_seconds >= 70.0
    assert samples == 1


def test_chainlink_feed_staleness_reports_fresh_when_age_is_within_threshold():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    fresh_price = SimpleNamespace(updated_at_ms=now_ms - 5_000)
    fake_feed = SimpleNamespace(get_all_prices=lambda: {"BTC": fresh_price})

    stale, age_seconds, samples = crypto_worker._chainlink_feed_staleness(fake_feed, max_age_seconds=45.0)

    assert stale is False
    assert age_seconds is not None
    assert age_seconds < 45.0
    assert samples == 1


class _DispatchSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "expected_sweep_missing"),
    [
        ("crypto_ws", False),
        ("periodic_scan", True),
    ],
)
async def test_dispatch_crypto_opportunities_scopes_sweep_to_periodic_scan(
    monkeypatch,
    trigger: str,
    expected_sweep_missing: bool,
):
    bridge_mock = AsyncMock(return_value=3)
    monkeypatch.setattr(crypto_worker.event_dispatcher, "dispatch", AsyncMock(return_value=[]))
    monkeypatch.setattr(crypto_worker, "AsyncSessionLocal", lambda: _DispatchSession())
    monkeypatch.setattr(crypto_worker, "bridge_opportunities_to_signals", bridge_mock)
    monkeypatch.setattr(crypto_worker.event_bus, "publish", AsyncMock())

    emitted, elapsed = await crypto_worker._dispatch_crypto_opportunities(
        [{"id": "m1"}],
        trigger=trigger,
        run_at=datetime.now(timezone.utc),
        emit_lock=asyncio.Lock(),
    )

    assert emitted == 3
    assert elapsed >= 0.0
    assert bridge_mock.await_args.kwargs["sweep_missing"] is expected_sweep_missing
