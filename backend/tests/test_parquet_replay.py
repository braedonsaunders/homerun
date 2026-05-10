"""Unit tests for the parquet-backed book replay path.

Covers the read/write surface end-to-end without touching Postgres so
the matcher's "swap any BookSource impl" contract is verifiable in
isolation.  The auto-discovery scanner (which DOES touch Postgres) is
exercised in ``test_parquet_scanner.py``.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from services.backtest.book_replay import (
    BookSnapshot,
    HybridBookSource,
    InMemoryBookReplay,
    PriceLevel,
)
from services.backtest.parquet_replay import (
    ParquetBookReplay,
    write_snapshots,
)
from services.external_data.parquet_schema import (
    DELTA_SCHEMA,
    SNAPSHOT_SCHEMA,
    parquet_path_for,
    parquet_root,
)


def _utc(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _make_snapshots(token_id: str, anchor: datetime, n: int) -> list[BookSnapshot]:
    return [
        BookSnapshot(
            token_id=token_id,
            observed_at=anchor + timedelta(minutes=i),
            bids=(PriceLevel(price=0.5 + i * 0.001, size=100.0),),
            asks=(PriceLevel(price=0.51 + i * 0.001, size=80.0),),
            spread_bps=10.0 + i,
            sequence=i,
        )
        for i in range(n)
    ]


# ── Schema + path helpers ────────────────────────────────────────────


def test_snapshot_schema_required_columns_present():
    """Snapshot schema must carry every column ParquetBookReplay reads
    when materialising a BookSnapshot — drift here breaks production
    backtests in subtle ways (silent zero levels)."""
    required = {
        "token_id", "observed_at_us", "sequence", "best_bid", "best_ask",
        "spread_bps", "bids_price", "bids_size", "asks_price", "asks_size",
        "trade_price", "trade_size", "trade_side",
    }
    assert required <= set(SNAPSHOT_SCHEMA.names), (
        f"missing columns: {required - set(SNAPSHOT_SCHEMA.names)}"
    )


def test_delta_schema_required_columns_present():
    required = {
        "token_id", "observed_at_us", "sequence", "event_type", "side",
        "price", "queue_depth_before", "queue_depth_after",
    }
    assert required <= set(DELTA_SCHEMA.names)


def test_parquet_path_for_normalises_long_token_ids(tmp_path):
    """78-char Polymarket token ids must survive the path helper
    (no truncation, no collisions for distinct tokens)."""
    t1 = "1" * 78
    t2 = "2" * 78
    p1 = parquet_path_for(
        provider="vendor", coin="btc", token_id=t1,
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
        kind="snapshots", root=tmp_path,
    )
    p2 = parquet_path_for(
        provider="vendor", coin="btc", token_id=t2,
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
        kind="snapshots", root=tmp_path,
    )
    assert p1.file_path != p2.file_path, "distinct tokens must have distinct paths"
    # Same window slug
    assert p1.window_dir == p2.window_dir


def test_parquet_root_env_override(monkeypatch, tmp_path):
    """HOMERUN_PARQUET_ROOT must override the default location."""
    monkeypatch.setenv("HOMERUN_PARQUET_ROOT", str(tmp_path))
    assert parquet_root() == tmp_path.resolve()


# ── Writer round-trip ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_then_iter_round_trip(tmp_path):
    """Snapshots written via ``write_snapshots`` come back through
    ``iter_snapshots`` in the same order with identical fields."""
    f = tmp_path / "snapshots__T1.parquet"
    snaps = _make_snapshots("T1", _utc(2026, 5, 1, 12, 0), 25)
    n = write_snapshots(file_path=f, snapshots=snaps)
    assert n == 25

    replay = ParquetBookReplay(
        per_token_files={"T1": str(f)},
        start=_utc(2026, 5, 1, 12, 0), end=_utc(2026, 5, 1, 14, 0),
    )
    out = []
    async for s in replay.iter_snapshots():
        out.append(s)
    assert len(out) == 25
    assert all(out[i].observed_at <= out[i + 1].observed_at for i in range(len(out) - 1))
    assert out[0].best_bid == pytest.approx(0.500)
    assert out[-1].best_bid == pytest.approx(0.524)


# ── snapshot_at ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_at_at_or_before(tmp_path):
    f = tmp_path / "snapshots__T1.parquet"
    snaps = _make_snapshots("T1", _utc(2026, 5, 1, 12, 0), 20)
    write_snapshots(file_path=f, snapshots=snaps)

    replay = ParquetBookReplay(
        per_token_files={"T1": str(f)},
        start=_utc(2026, 5, 1, 12, 0), end=_utc(2026, 5, 1, 14, 0),
    )
    snap = await replay.snapshot_at(
        token_id="T1", ts=_utc(2026, 5, 1, 12, 12, 30),
    )
    assert snap is not None
    assert snap.best_bid == pytest.approx(0.512)  # snapshot from minute 12


@pytest.mark.asyncio
async def test_snapshot_at_before_window_returns_none(tmp_path):
    f = tmp_path / "snapshots__T1.parquet"
    snaps = _make_snapshots("T1", _utc(2026, 5, 1, 12, 0), 5)
    write_snapshots(file_path=f, snapshots=snaps)
    replay = ParquetBookReplay(
        per_token_files={"T1": str(f)},
        start=_utc(2026, 5, 1, 12, 0), end=_utc(2026, 5, 1, 14, 0),
    )
    snap = await replay.snapshot_at(
        token_id="T1", ts=_utc(2026, 5, 1, 11, 0),
    )
    assert snap is None


@pytest.mark.asyncio
async def test_snapshot_at_unknown_token_returns_none(tmp_path):
    f = tmp_path / "snapshots__T1.parquet"
    write_snapshots(file_path=f, snapshots=_make_snapshots("T1", _utc(2026, 5, 1), 3))
    replay = ParquetBookReplay(
        per_token_files={"T1": str(f)},
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
    )
    assert await replay.snapshot_at(token_id="UNKNOWN", ts=_utc(2026, 5, 1, 1)) is None


# ── Heap merge ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_iter_snapshots_heap_merges_multiple_tokens(tmp_path):
    """Two tokens at staggered timestamps must come out interleaved
    in chronological order, not grouped by token."""
    f1 = tmp_path / "snapshots__T1.parquet"
    f2 = tmp_path / "snapshots__T2.parquet"
    write_snapshots(
        file_path=f1,
        snapshots=[
            BookSnapshot(token_id="T1",
                         observed_at=_utc(2026, 5, 1, 12, i * 2),
                         bids=(PriceLevel(0.5, 100.0),),
                         asks=(PriceLevel(0.51, 80.0),),
                         spread_bps=10.0)
            for i in range(5)
        ],
    )
    write_snapshots(
        file_path=f2,
        snapshots=[
            BookSnapshot(token_id="T2",
                         observed_at=_utc(2026, 5, 1, 12, i * 2 + 1),
                         bids=(PriceLevel(0.4, 100.0),),
                         asks=(PriceLevel(0.42, 80.0),),
                         spread_bps=20.0)
            for i in range(5)
        ],
    )
    replay = ParquetBookReplay(
        per_token_files={"T1": str(f1), "T2": str(f2)},
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
    )
    order = []
    async for s in replay.iter_snapshots():
        order.append((s.token_id, s.observed_at.minute))
    # Strict chronological interleave
    expected = [("T1", 0), ("T2", 1), ("T1", 2), ("T2", 3),
                ("T1", 4), ("T2", 5), ("T1", 6), ("T2", 7),
                ("T1", 8), ("T2", 9)]
    assert order == expected


# ── HybridBookSource dispatch ───────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_routes_per_token_correctly(tmp_path):
    """Hybrid source: T1 served by parquet, T2 by an in-memory replay
    (proxy for the SQL replays in tests).  ``snapshot_at`` must hit
    the assigned backend; ``iter_snapshots`` must merge cleanly."""
    f = tmp_path / "snapshots__T1.parquet"
    write_snapshots(
        file_path=f,
        snapshots=[BookSnapshot(token_id="T1",
                                 observed_at=_utc(2026, 5, 1, 12, i * 2),
                                 bids=(PriceLevel(0.5 + i * 0.01, 100.0),),
                                 asks=(PriceLevel(0.51 + i * 0.01, 80.0),),
                                 spread_bps=10.0)
                    for i in range(5)],
    )
    parquet = ParquetBookReplay(
        per_token_files={"T1": str(f)},
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
    )
    in_mem = InMemoryBookReplay(
        snapshots=[BookSnapshot(token_id="T2",
                                 observed_at=_utc(2026, 5, 1, 12, i * 2 + 1),
                                 bids=(PriceLevel(0.4, 100.0),),
                                 asks=(PriceLevel(0.42, 80.0),),
                                 spread_bps=20.0)
                    for i in range(5)],
    )
    hybrid = HybridBookSource(
        backends={"parquet": parquet, "snapshots": in_mem},
        routing={"T1": "parquet", "T2": "snapshots"},
    )
    # snapshot_at routes correctly
    s1 = await hybrid.snapshot_at(token_id="T1", ts=_utc(2026, 5, 1, 12, 5))
    s2 = await hybrid.snapshot_at(token_id="T2", ts=_utc(2026, 5, 1, 12, 5))
    assert s1 is not None and s1.best_bid == pytest.approx(0.520)  # parquet T1 minute 4
    assert s2 is not None and s2.best_bid == pytest.approx(0.4)     # in-mem T2

    # iter_snapshots merges chronologically
    order = []
    async for s in hybrid.iter_snapshots():
        order.append((s.token_id, s.observed_at.minute))
    assert len(order) == 10
    minutes = [m for _, m in order]
    assert minutes == sorted(minutes)


@pytest.mark.asyncio
async def test_hybrid_empty_backends_yields_nothing():
    hybrid = HybridBookSource(backends={}, routing={})
    out = [s async for s in hybrid.iter_snapshots()]
    assert out == []
    assert await hybrid.snapshot_at(token_id="X", ts=_utc(2026, 5, 1)) is None


# ── Truncation safety ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_truncates_on_corrupt_file(tmp_path):
    """A corrupt parquet file must NOT crash the matcher — ParquetBookReplay
    flips its ``truncated`` flag and the iter loop yields what it can."""
    bad = tmp_path / "snapshots__T1.parquet"
    bad.write_bytes(b"not a parquet file")
    replay = ParquetBookReplay(
        per_token_files={"T1": str(bad)},
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
    )
    out = [s async for s in replay.iter_snapshots()]
    assert out == []
    assert replay.truncated
    assert replay.truncation_reason


# ── LRU cache ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_at_cache_is_hit_on_second_call(tmp_path):
    """Second snapshot_at on the same file must reuse the cached
    table (verified by overwriting the file mid-run — second read
    sees the cached, pre-overwrite content)."""
    f = tmp_path / "snapshots__T1.parquet"
    write_snapshots(
        file_path=f,
        snapshots=[BookSnapshot(token_id="T1",
                                 observed_at=_utc(2026, 5, 1, 12, i),
                                 bids=(PriceLevel(0.5, 100.0),),
                                 asks=(PriceLevel(0.51, 80.0),),
                                 spread_bps=10.0)
                    for i in range(5)],
    )
    replay = ParquetBookReplay(
        per_token_files={"T1": str(f)},
        start=_utc(2026, 5, 1), end=_utc(2026, 5, 2),
    )
    s_first = await replay.snapshot_at(token_id="T1", ts=_utc(2026, 5, 1, 12, 3))
    assert s_first is not None and s_first.best_bid == pytest.approx(0.5)

    # Overwrite the file with totally different prices.  If the cache
    # works, the second snapshot_at returns the OLD bid (0.5).  If the
    # cache is broken it would return the NEW bid (0.99).
    write_snapshots(
        file_path=f,
        snapshots=[BookSnapshot(token_id="T1",
                                 observed_at=_utc(2026, 5, 1, 12, i),
                                 bids=(PriceLevel(0.99, 100.0),),
                                 asks=(PriceLevel(0.999, 80.0),),
                                 spread_bps=10.0)
                    for i in range(5)],
    )
    s_second = await replay.snapshot_at(token_id="T1", ts=_utc(2026, 5, 1, 12, 3))
    assert s_second is not None
    assert s_second.best_bid == pytest.approx(0.5), "cache must be hit"
