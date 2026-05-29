"""Tests for the Phase-2 keystone: MarketDataView (book side) + coverage.

The book-read path is validated against real temp parquet via a hand-built
CoverageMap (no DB). resolve_coverage is exercised against the real test
Postgres (marked ``db``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.book import load_book_series, row_to_book_snapshot
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.view import MarketDataView


def _write_snapshot_file(path, token_id, ticks):
    """ticks: list of (observed_at_us, best_bid, best_ask)."""
    n = len(ticks)
    table = pa.table(
        {
            "token_id": pa.array([token_id] * n, pa.string()),
            "observed_at_us": pa.array([t[0] for t in ticks], pa.int64()),
            "sequence": pa.array([i for i in range(n)], pa.int64()),
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


# ── row conversion / series loader ──────────────────────────────────────
def test_row_to_book_snapshot_filters_invalid_levels():
    row = {
        "observed_at_us": 1_000_000,
        "sequence": 1,
        "spread_bps": None,
        "bids_price": [0.40, 0.0, 1.5],   # 0.0 and 1.5 invalid
        "bids_size": [10.0, 5.0, 5.0],
        "asks_price": [0.60, 0.62],
        "asks_size": [10.0, 0.0],          # second has size 0 -> dropped
    }
    snap = row_to_book_snapshot("tok", row)
    assert [lvl.price for lvl in snap.bids] == [0.40]
    assert [lvl.price for lvl in snap.asks] == [0.60]
    assert snap.token_id == "tok"


def test_load_book_series_window_filter(tmp_path):
    f = tmp_path / "snapshots__tok.parquet"
    base = 1_700_000_000_000_000  # us
    _write_snapshot_file(f, "tok", [
        (base + 0, 0.40, 0.42),
        (base + 10_000_000, 0.50, 0.52),
        (base + 20_000_000, 0.60, 0.62),
    ])
    # window excludes the first tick
    series, rows = load_book_series("tok", [f], start_us=base + 5_000_000, end_us=base + 30_000_000)
    assert rows == 2
    assert len(series) == 2
    snap = series.as_of(base + 15_000_000)
    assert snap is not None and snap.bids[0].price == 0.50


# ── CoverageMap logic ───────────────────────────────────────────────────
def test_coverage_map_covered_uncovered_fraction():
    cov = CoverageMap(
        by_token={
            "a": TokenCoverage("a", files=("f1.parquet",)),
            "b": TokenCoverage("b", files=()),
        },
        requested=("a", "b"),
        window_start_us=0, window_end_us=100,
    )
    assert cov.covered_tokens == ("a",)
    assert cov.uncovered_tokens == ("b",)
    assert cov.coverage_fraction == 0.5
    assert cov.files_for("a") == ("f1.parquet",)
    assert cov.as_per_token_files() == {"a": ["f1.parquet"]}


# ── MarketDataView book side (no DB) ────────────────────────────────────
def _view_over(tmp_path, token_ticks: dict[str, list], start_us, end_us):
    by_token = {}
    for tok, ticks in token_ticks.items():
        f = tmp_path / f"snapshots__{tok}.parquet"
        _write_snapshot_file(f, tok, ticks)
        by_token[tok] = TokenCoverage(tok, files=(str(f),), start_us=start_us, end_us=end_us)
    cov = CoverageMap(by_token=by_token, requested=tuple(token_ticks), window_start_us=start_us, window_end_us=end_us)
    start = datetime.fromtimestamp(start_us / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp(end_us / 1e6, tz=timezone.utc)
    return MarketDataView(coverage=cov, start=start, end=end)


@pytest.mark.asyncio
async def test_view_book_at_point_in_time(tmp_path):
    base = 1_700_000_000_000_000
    view = _view_over(tmp_path, {
        "up": [(base, 0.40, 0.42), (base + 10_000_000, 0.55, 0.57)],
    }, base, base + 60_000_000)

    def at(sec):
        return datetime.fromtimestamp((base + sec * 1_000_000) / 1e6, tz=timezone.utc)

    assert await view.book_at("up", at(-5)) is None      # before first
    s0 = await view.book_at("up", at(5))
    assert s0 is not None and s0.bids[0].price == 0.40    # first tick visible
    s1 = await view.book_at("up", at(15))
    assert s1 is not None and s1.bids[0].price == 0.55    # second tick


@pytest.mark.asyncio
async def test_view_book_at_staleness(tmp_path):
    base = 1_700_000_000_000_000
    view = _view_over(tmp_path, {"up": [(base, 0.40, 0.42)]}, base, base + 600_000_000)
    near = datetime.fromtimestamp((base + 20_000_000) / 1e6, tz=timezone.utc)
    far = datetime.fromtimestamp((base + 120_000_000) / 1e6, tz=timezone.utc)
    assert await view.book_at("up", near, max_staleness_seconds=30) is not None
    assert await view.book_at("up", far, max_staleness_seconds=30) is None


@pytest.mark.asyncio
async def test_view_iter_books_global_order(tmp_path):
    base = 1_700_000_000_000_000
    view = _view_over(tmp_path, {
        "up":   [(base + 0, 0.40, 0.42), (base + 20_000_000, 0.45, 0.47)],
        "down": [(base + 10_000_000, 0.58, 0.60), (base + 30_000_000, 0.55, 0.57)],
    }, base, base + 60_000_000)
    seen = [(s.token_id, int(s.observed_at.timestamp() * 1e6)) async for s in view.iter_books()]
    # globally ascending by observed_at across the two tokens
    times = [t for _, t in seen]
    assert times == sorted(times)
    assert {tok for tok, _ in seen} == {"up", "down"}
    assert len(seen) == 4


@pytest.mark.asyncio
async def test_view_dataset_snapshot_pin(tmp_path):
    base = 1_700_000_000_000_000
    view = _view_over(tmp_path, {"up": [(base, 0.40, 0.42)]}, base, base + 60_000_000)
    _ = await view.book_at("up", datetime.fromtimestamp((base + 1_000_000) / 1e6, tz=timezone.utc))
    snap = view.dataset_snapshot()
    assert snap.content_hash.startswith("sha256:")
    assert len(snap.entries) == 1
    assert snap.entries[0].rows == 1


# ── resolve_coverage against the real test DB ───────────────────────────
@pytest.mark.asyncio
@pytest.mark.db
async def test_resolve_coverage_finds_registered_dataset(tmp_path):
    import uuid
    from sqlalchemy import delete
    from models.database import AsyncSessionLocal, ProviderDataset
    from services.marketdata.coverage import resolve_coverage

    tok = "covtok_" + uuid.uuid4().hex[:12]
    win = tmp_path / "20260101T000000__20260101T001500"
    win.mkdir(parents=True)
    f = win / f"snapshots__{tok}.parquet"
    _write_snapshot_file(f, tok, [(1_767_225_600_000_000, 0.4, 0.42)])  # 2026-01-01T00:00Z

    ds_id = "test:" + uuid.uuid4().hex[:12]
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    async with AsyncSessionLocal() as session:
        session.add(ProviderDataset(
            id=ds_id, provider="polybacktest", coin="btc",
            external_id=ds_id, external_slug=win.name, title="t",
            asset_class="prediction", token_ids_json=[tok],
            storage_type="parquet", storage_uri=win.resolve().as_uri(),
            start_ts=start.replace(tzinfo=None), end_ts=end.replace(tzinfo=None),
            snapshot_count=1, trade_count=0,
        ))
        await session.commit()
    try:
        cov = await resolve_coverage(
            token_ids=[tok], start=start, end=end,
            providers=["polybacktest"], ensure_scan=False,
        )
        assert cov.by_token[tok].covered
        assert str(f) in cov.files_for(tok) or any(str(f) == p for p in cov.files_for(tok))
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(delete(ProviderDataset).where(ProviderDataset.id == ds_id))
            await session.commit()
