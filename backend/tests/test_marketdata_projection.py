"""Tests for the crypto_update projection (Phase 2b).

Covers the pure reconstruction logic (clamp/parse, market liveness, market-dict
shape + binary-complement derivation) and an end-to-end projection over a
hand-built MarketDataView (no DB) asserting event shape + gap-fill exclusion.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.projection import (
    ProjectedMarket,
    _clamp01,
    _market_dict,
    _parse_iso_us,
    project_crypto_update_events,
)
from services.marketdata.view import MarketDataView


def _us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def test_clamp01():
    assert _clamp01(0.5) == 0.5
    assert _clamp01(0.0) is None
    assert _clamp01(1.0) is None
    assert _clamp01(None) is None


def test_parse_iso_us():
    dt = datetime(2026, 5, 28, 19, 13, tzinfo=timezone.utc)
    assert _parse_iso_us(dt.isoformat().replace("+00:00", "Z")) == _us(dt)
    assert _parse_iso_us(None) is None
    assert _parse_iso_us("nope") is None


def _market():
    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    return ProjectedMarket(
        market_id="111", condition_id="0xabc", slug="btc-updown-5m-1",
        title="BTC Up or Down", coin="btc", timeframe="5m",
        start_us=_us(base), end_us=_us(base) + 300_000_000,
        up_token="UP", down_token="DOWN", price_to_beat=80000.0,
    )


def test_market_alive_at_and_key():
    m = _market()
    base = _us(datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc))
    assert m.market_key() == "0xabc"
    assert m.alive_at(base) is True
    assert m.alive_at(base - 1) is False
    assert m.alive_at(base + 300_000_000) is False  # end exclusive
    assert set(m.tokens) == {"UP", "DOWN"}


def test_market_dict_two_sided():
    m = _market()
    tick = datetime(2026, 5, 28, 21, 1, tzinfo=timezone.utc)
    md = _market_dict(m, tick=tick, up_bid=0.60, up_ask=0.62, down_bid=0.38, down_ask=0.40)
    assert md is not None
    assert md["clob_token_ids"] == ["UP", "DOWN"]
    assert md["up_price"] == pytest.approx(0.61)
    assert md["down_price"] == pytest.approx(0.39)
    assert md["seconds_left"] == pytest.approx(240, abs=2)
    assert md["is_live"] is True
    assert md["condition_id"] == "0xabc"


def test_market_dict_binary_complement_for_missing_side():
    m = _market()
    tick = datetime(2026, 5, 28, 21, 1, tzinfo=timezone.utc)
    # only UP book present -> DOWN derived as 1 - up
    md = _market_dict(m, tick=tick, up_bid=0.60, up_ask=0.62, down_bid=None, down_ask=None)
    assert md is not None
    assert md["down_price"] == pytest.approx(0.39)  # mid of (1-0.62, 1-0.60)=(0.38,0.40)


def test_market_dict_none_without_book():
    m = _market()
    tick = datetime(2026, 5, 28, 21, 1, tzinfo=timezone.utc)
    assert _market_dict(m, tick=tick, up_bid=None, up_ask=None, down_bid=None, down_ask=None) is None


# ── end-to-end projection over a hand-built view (no DB) ────────────────
def _write_snapshot_file(path, token_id, ticks):
    n = len(ticks)
    table = pa.table(
        {
            "token_id": pa.array([token_id] * n, pa.string()),
            "observed_at_us": pa.array([t[0] for t in ticks], pa.int64()),
            "sequence": pa.array(list(range(n)), pa.int64()),
            "best_bid": pa.array([t[1] for t in ticks], pa.float64()),
            "best_ask": pa.array([t[2] for t in ticks], pa.float64()),
            "spread_bps": pa.array([None] * n, pa.float64()),
            "bids_price": pa.array([[t[1]] for t in ticks], pa.list_(pa.float64())),
            "bids_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "asks_price": pa.array([[t[2]] for t in ticks], pa.list_(pa.float64())),
            "asks_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "trade_price": pa.array([None] * n, pa.float64()),
            "trade_size": pa.array([None] * n, pa.float64()),
            "trade_side": pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )
    pq.write_table(table, str(path))


@pytest.mark.asyncio
async def test_project_crypto_update_events_end_to_end(tmp_path):
    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    base_us = _us(base)
    # UP/DOWN book files, 1s cadence for 10s
    up_ticks = [(base_us + i * 1_000_000, 0.60, 0.62) for i in range(10)]
    dn_ticks = [(base_us + i * 1_000_000, 0.38, 0.40) for i in range(10)]
    fu = tmp_path / "snapshots__UP.parquet"
    fd = tmp_path / "snapshots__DOWN.parquet"
    _write_snapshot_file(fu, "UP", up_ticks)
    _write_snapshot_file(fd, "DOWN", dn_ticks)

    start = base
    end = datetime.fromtimestamp((base_us + 10_000_000) / 1e6, tz=timezone.utc)
    cov = CoverageMap(
        by_token={
            "UP": TokenCoverage("UP", files=(str(fu),), start_us=base_us, end_us=base_us + 10_000_000),
            "DOWN": TokenCoverage("DOWN", files=(str(fd),), start_us=base_us, end_us=base_us + 10_000_000),
        },
        requested=("UP", "DOWN"), window_start_us=base_us, window_end_us=base_us + 10_000_000,
    )
    view = MarketDataView(coverage=cov, start=start, end=end)

    m = ProjectedMarket(
        market_id="111", condition_id="0xabc", slug="btc-updown-5m-1",
        title="BTC Up or Down", coin="btc", timeframe="5m",
        start_us=base_us, end_us=base_us + 300_000_000,
        up_token="UP", down_token="DOWN", price_to_beat=80000.0,
    )

    events, stats = await project_crypto_update_events(
        start=start, end=end, cadence_seconds=2.0, view=view, markets=[m],
    )
    assert stats["events"] > 0
    ev0 = events[0]
    md = ev0.payload["markets"][0]
    assert md["condition_id"] == "0xabc"
    assert md["up_price"] == pytest.approx(0.61)
    assert md["down_price"] == pytest.approx(0.39)
    # cadence 2s over a 10s window -> ~5-6 events
    assert 4 <= stats["events"] <= 6


@pytest.mark.asyncio
async def test_projection_excludes_recorded_markets(tmp_path):
    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    base_us = _us(base)
    fu = tmp_path / "snapshots__UP.parquet"
    _write_snapshot_file(fu, "UP", [(base_us, 0.6, 0.62)])
    cov = CoverageMap(
        by_token={"UP": TokenCoverage("UP", files=(str(fu),), start_us=base_us, end_us=base_us + 5_000_000)},
        requested=("UP",), window_start_us=base_us, window_end_us=base_us + 5_000_000,
    )
    view = MarketDataView(coverage=cov, start=base, end=datetime.fromtimestamp((base_us + 5_000_000) / 1e6, tz=timezone.utc))
    m = ProjectedMarket(
        market_id="111", condition_id="0xabc", slug="s", title="t", coin="btc", timeframe="5m",
        start_us=base_us, end_us=base_us + 300_000_000, up_token="UP", down_token="DOWN", price_to_beat=None,
    )
    events, stats = await project_crypto_update_events(
        start=base, end=datetime.fromtimestamp((base_us + 5_000_000) / 1e6, tz=timezone.utc),
        cadence_seconds=1.0, view=view, markets=[m], exclude_market_keys={"0xabc"},
    )
    assert stats["markets_active"] == 0
    assert stats["events"] == 0
