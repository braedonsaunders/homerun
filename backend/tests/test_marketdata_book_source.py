"""Behavior-equivalence: MarketDataViewSource vs legacy ParquetBookReplay.

Phase 3 swaps the matcher's parquet book source from ParquetBookReplay to the
unified MarketDataView (via MarketDataViewSource). This test pins that the swap
is behavior-preserving: identical snapshot_at results and identical
iter_snapshots streams over the same canonical parquet.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.backtest.parquet_replay import ParquetBookReplay
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


def _snap_key(s):
    if s is None:
        return None
    return (
        s.token_id,
        int(s.observed_at.timestamp() * 1e6),
        tuple((round(b.price, 9), round(b.size, 9)) for b in s.bids),
        tuple((round(a.price, 9), round(a.size, 9)) for a in s.asks),
    )


@pytest.mark.asyncio
async def test_snapshot_at_equivalent_to_parquet_replay(tmp_path):
    base = 1_700_000_000_000_000
    toks = {
        "up": [(base + i * 1_000_000, 0.40 + i * 0.01, 0.42 + i * 0.01) for i in range(20)],
        "down": [(base + i * 1_500_000, 0.58 - i * 0.01, 0.60 - i * 0.01) for i in range(20)],
    }
    files = {}
    by_token = {}
    for t, ticks in toks.items():
        f = tmp_path / f"snapshots__{t}.parquet"
        _write(f, t, ticks)
        files[t] = str(f)
        by_token[t] = TokenCoverage(t, files=(str(f),), start_us=base, end_us=base + 60_000_000)

    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + 60_000_000) / 1e6, tz=timezone.utc)

    legacy = ParquetBookReplay(per_token_files=files, start=start, end=end)
    view = MarketDataView(
        coverage=CoverageMap(by_token=by_token, requested=tuple(by_token), window_start_us=base, window_end_us=base + 60_000_000),
        start=start, end=end,
    )
    adapter = MarketDataViewSource(view)

    # Probe snapshot_at across many timestamps for both tokens.
    for tok in ("up", "down"):
        for sec in range(0, 30):
            ts = datetime.fromtimestamp((base + sec * 1_000_000) / 1e6, tz=timezone.utc)
            a = await legacy.snapshot_at(token_id=tok, ts=ts)
            b = await adapter.snapshot_at(token_id=tok, ts=ts)
            assert _snap_key(a) == _snap_key(b), f"mismatch tok={tok} sec={sec}"


@pytest.mark.asyncio
async def test_iter_snapshots_equivalent_to_parquet_replay(tmp_path):
    base = 1_700_000_000_000_000
    toks = {
        "up": [(base + i * 1_000_000, 0.40, 0.42) for i in range(10)],
        "down": [(base + i * 1_000_000 + 500_000, 0.58, 0.60) for i in range(10)],
    }
    files = {}
    by_token = {}
    for t, ticks in toks.items():
        f = tmp_path / f"snapshots__{t}.parquet"
        _write(f, t, ticks)
        files[t] = str(f)
        by_token[t] = TokenCoverage(t, files=(str(f),), start_us=base, end_us=base + 60_000_000)

    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + 60_000_000) / 1e6, tz=timezone.utc)

    legacy = ParquetBookReplay(per_token_files=files, start=start, end=end)
    view = MarketDataView(
        coverage=CoverageMap(by_token=by_token, requested=tuple(by_token), window_start_us=base, window_end_us=base + 60_000_000),
        start=start, end=end,
    )
    adapter = MarketDataViewSource(view)

    legacy_stream = [_snap_key(s) async for s in legacy.iter_snapshots()]
    adapter_stream = [_snap_key(s) async for s in adapter.iter_snapshots()]
    # Same multiset of snapshots; both globally ordered by observed_at.
    assert sorted(legacy_stream) == sorted(adapter_stream)
    assert [k[1] for k in adapter_stream] == sorted(k[1] for k in adapter_stream)
    assert len(adapter_stream) == 20
