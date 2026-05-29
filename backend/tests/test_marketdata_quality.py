"""Tests for parquet-native data-quality assessment (Phase 8)."""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.view import MarketDataView  # noqa: F401 (ensures pkg import path)


def _write(path, token_id, ticks):
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
async def test_assess_book_quality_flags_crossed_and_gap(tmp_path, monkeypatch):
    base = 1_700_000_000_000_000
    # 5 normal ticks 1s apart, then a 30s gap, then a crossed book (bid>=ask)
    ticks = [(base + i * 1_000_000, 0.40, 0.42) for i in range(5)]
    ticks.append((base + 35_000_000, 0.60, 0.55))  # crossed + 30s gap
    f = tmp_path / "snapshots__T.parquet"
    _write(f, "T", ticks)

    # Patch resolve_coverage to point at our temp file (no DB).
    import services.marketdata.quality as q

    async def _fake_cov(*, token_ids, start, end, providers=None, ensure_scan=True):
        return CoverageMap(
            by_token={"T": TokenCoverage("T", files=(str(f),), start_us=base, end_us=base + 60_000_000)},
            requested=("T",), window_start_us=base, window_end_us=base + 60_000_000,
        )

    monkeypatch.setattr("services.marketdata.coverage.resolve_coverage", _fake_cov)

    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + 60_000_000) / 1e6, tz=timezone.utc)
    rep = await q.assess_book_quality(token_id="T", start=start, end=end, gap_threshold_seconds=5.0)

    assert rep["covered"] is True
    assert rep["rows"] == 6
    assert rep["crossed_book_count"] == 1
    assert rep["has_large_gap"] is True
    assert rep["max_gap_seconds"] >= 30.0
    assert rep["staleness_at_window_end_seconds"] >= 0


@pytest.mark.asyncio
async def test_assess_book_quality_uncovered(tmp_path, monkeypatch):
    import services.marketdata.quality as q

    async def _empty_cov(*, token_ids, start, end, providers=None, ensure_scan=True):
        return CoverageMap(by_token={}, requested=tuple(token_ids), window_start_us=0, window_end_us=0)

    monkeypatch.setattr("services.marketdata.coverage.resolve_coverage", _empty_cov)
    rep = await q.assess_book_quality(
        token_id="NOPE",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert rep["covered"] is False
