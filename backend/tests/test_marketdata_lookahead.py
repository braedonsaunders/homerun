"""Property-based anti-lookahead guarantees for the unified market-data layer.

The cardinal rule of institutional backtesting: a decision at time tau may
only see data observed at-or-before tau. The unified layer enforces this in
ONE place — the as-of primitive (services.marketdata.asof.AsOfSeries) — so
every reader (book_at, the discovery grid, the matcher, the crypto_update
projection) inherits the guarantee.

These tests hammer that primitive with seeded-random inputs and assert two
invariants across thousands of cases:

  1. ANTI-LOOKAHEAD: as_of(tau) NEVER returns an entry observed strictly after
     tau (no future leak).
  2. CORRECTNESS: when an entry IS returned, it is the most-recent one at-or-
     before tau.

Plus the staleness bound and an end-to-end check through MarketDataView.book_at
over real parquet.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.asof import AsOfSeries, bisect_as_of
from services.marketdata.book import us_from_dt
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.view import MarketDataView


def test_bisect_as_of_property():
    rng = random.Random(20260528)
    for _ in range(3000):
        n = rng.randint(0, 40)
        ts = sorted(rng.randint(0, 1_000_000) for _ in range(n))
        q = rng.randint(-50, 1_000_050)
        idx = bisect_as_of(ts, q)
        if idx < 0:
            assert all(t > q for t in ts)
        else:
            assert ts[idx] <= q                       # anti-lookahead
            assert idx == len(ts) - 1 or ts[idx + 1] > q  # most-recent
            assert ts[idx] == max(t for t in ts if t <= q)


def test_asof_series_never_returns_future():
    rng = random.Random(987654)
    for _ in range(2000):
        n = rng.randint(0, 30)
        ts = [rng.randint(0, 5_000_000) for _ in range(n)]
        s: AsOfSeries[int] = AsOfSeries()
        # add in random order — finalize must sort, ties resolve to last-added
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            s.add(ts[i], i)
        q = rng.randint(-100, 5_000_100)
        entry = s.as_of_entry(q)
        if entry is None:
            assert not ts or min(ts) > q
        else:
            obs, _val = entry
            assert obs <= q                              # ANTI-LOOKAHEAD
            assert obs == max(t for t in ts if t <= q)   # correct as-of


def test_asof_series_staleness_property():
    rng = random.Random(42)
    for _ in range(1000):
        n = rng.randint(1, 20)
        ts = sorted(rng.randint(0, 1_000_000) for _ in range(n))
        s: AsOfSeries[int] = AsOfSeries()
        for i, t in enumerate(ts):
            s.add(t, i)
        q = rng.randint(0, 1_100_000)
        max_stale = rng.randint(0, 200_000)
        entry = s.as_of_entry(q, max_staleness_us=max_stale)
        plain = s.as_of_entry(q)
        if plain is None:
            assert entry is None
        else:
            obs, _ = plain
            if q - obs > max_stale:
                assert entry is None          # too stale -> absent
            else:
                assert entry == plain          # fresh enough -> same


def _write_snapshot_file(path, token_id, ticks):
    n = len(ticks)
    pq.write_table(
        pa.table(
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
        ),
        str(path),
    )


@pytest.mark.asyncio
async def test_view_book_at_never_returns_future(tmp_path):
    base = 1_700_000_000_000_000
    rng = random.Random(7)
    # 200 snapshots over 200s, strictly increasing observed times
    ticks = [(base + i * 1_000_000, round(0.4 + 0.001 * i, 4), round(0.41 + 0.001 * i, 4)) for i in range(200)]
    f = tmp_path / "snapshots__T.parquet"
    _write_snapshot_file(f, "T", ticks)
    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + 200_000_000) / 1e6, tz=timezone.utc)
    cov = CoverageMap(
        by_token={"T": TokenCoverage("T", files=(str(f),), start_us=base, end_us=base + 200_000_000)},
        requested=("T",), window_start_us=base, window_end_us=base + 200_000_000,
    )
    view = MarketDataView(coverage=cov, start=start, end=end)

    for _ in range(500):
        q_us = base + rng.randint(-2_000_000, 202_000_000)
        q = datetime.fromtimestamp(q_us / 1e6, tz=timezone.utc)
        snap = await view.book_at("T", q)
        if snap is not None:
            # ANTI-LOOKAHEAD: the returned book was observed at-or-before q
            assert us_from_dt(snap.observed_at) <= q_us
