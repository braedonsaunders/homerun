"""Tests for MarketDataViewSource — the _BookSource adapter over MarketDataView.

The adapter is what the matching engine + discovery grid consume. It must
present the engine's _BookSource protocol (snapshot_at + iter_snapshots) and
faithfully delegate to the view's point-in-time book access.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.book_source import MarketDataViewSource
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.view import MarketDataView


def _write(path, token_id, ticks):
    n = len(ticks)
    table = pa.table(
        {
            "token_id": pa.array([token_id] * n, pa.string()),
            "observed_at_us": pa.array([t[0] for t in ticks], pa.int64()),
            "sequence": pa.array(list(range(n)), pa.int64()),
            "best_bid": pa.array([t[1] for t in ticks], pa.float64()),
            "best_ask": pa.array([t[2] for t in ticks], pa.float64()),
            "spread_bps": pa.array([None] * n, pa.float64()),
            "bids_price": pa.array([[t[1], t[1] - 0.01] for t in ticks], pa.list_(pa.float64())),
            "bids_size": pa.array([[10.0, 5.0]] * n, pa.list_(pa.float64())),
            "asks_price": pa.array([[t[2], t[2] + 0.01] for t in ticks], pa.list_(pa.float64())),
            "asks_size": pa.array([[10.0, 5.0]] * n, pa.list_(pa.float64())),
            "trade_price": pa.array([None] * n, pa.float64()),
            "trade_size": pa.array([None] * n, pa.float64()),
            "trade_side": pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )
    pq.write_table(table, str(path))


def _view(tmp_path, token_ticks, base, span_us):
    by_token = {}
    for tok, ticks in token_ticks.items():
        f = tmp_path / f"snapshots__{tok}.parquet"
        _write(f, tok, ticks)
        by_token[tok] = TokenCoverage(tok, files=(str(f),), start_us=base, end_us=base + span_us)
    cov = CoverageMap(by_token=by_token, requested=tuple(token_ticks), window_start_us=base, window_end_us=base + span_us)
    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + span_us) / 1e6, tz=timezone.utc)
    return MarketDataView(coverage=cov, start=start, end=end)


@pytest.mark.asyncio
async def test_snapshot_at_point_in_time(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {"up": [(base, 0.40, 0.42), (base + 10_000_000, 0.55, 0.57)]}, base, 60_000_000)
    src = MarketDataViewSource(view)

    def at(sec):
        return datetime.fromtimestamp((base + sec * 1_000_000) / 1e6, tz=timezone.utc)

    assert await src.snapshot_at(token_id="up", ts=at(-1)) is None      # before first
    s0 = await src.snapshot_at(token_id="up", ts=at(5))
    assert s0 is not None and s0.bids[0].price == 0.40                  # full ladder preserved
    assert len(s0.bids) == 2 and len(s0.asks) == 2
    s1 = await src.snapshot_at(token_id="up", ts=at(15))
    assert s1 is not None and s1.bids[0].price == 0.55


@pytest.mark.asyncio
async def test_iter_snapshots_global_order(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {
        "up": [(base + 0, 0.40, 0.42), (base + 20_000_000, 0.45, 0.47)],
        "down": [(base + 10_000_000, 0.58, 0.60), (base + 30_000_000, 0.55, 0.57)],
    }, base, 60_000_000)
    src = MarketDataViewSource(view)
    times = [int(s.observed_at.timestamp() * 1e6) async for s in src.iter_snapshots()]
    assert times == sorted(times)
    assert len(times) == 4


@pytest.mark.asyncio
async def test_truncation_surface_present(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {"up": [(base, 0.4, 0.42)]}, base, 60_000_000)
    src = MarketDataViewSource(view)
    # The engine reads .truncated on the book source; the adapter exposes it.
    assert src.truncated is False
    assert src.truncation_reason is None
