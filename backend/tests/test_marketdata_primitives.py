"""Unit tests for the Phase-1 unified market-data primitives.

Covers the four foundations that replace the scattered reimplementations:
as-of lookup, ordered merge (sync+async), the dataset-snapshot manifest, and
the schema-validation authority.
"""
from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from services.marketdata.asof import AsOfSeries, bisect_as_of
from services.marketdata.manifest import (
    DatasetSnapshot,
    SnapshotEntry,
    build_snapshot,
    compute_content_hash,
)
from services.marketdata.merge import aordered_merge, ordered_merge
from services.marketdata.schema import (
    BOOK_SCHEMA_VERSION,
    SchemaValidationError,
    schema_for,
    validate_arrow_schema,
    validate_file,
    validate_table,
)


# ── as-of ────────────────────────────────────────────────────────────
def test_bisect_as_of_basic():
    ts = [10, 20, 30]
    assert bisect_as_of(ts, 9) == -1     # before first
    assert bisect_as_of(ts, 10) == 0     # exact
    assert bisect_as_of(ts, 25) == 1     # between
    assert bisect_as_of(ts, 100) == 2    # after last


def test_asof_series_point_in_time_barrier():
    s = AsOfSeries[str]()
    s.add(30, "c")
    s.add(10, "a")
    s.add(20, "b")
    # query before any data -> None (the anti-lookahead guarantee)
    assert s.as_of(5) is None
    # never returns a value observed strictly after tau
    assert s.as_of(10) == "a"
    assert s.as_of(15) == "a"
    assert s.as_of(29) == "b"
    assert s.as_of(30) == "c"
    assert s.as_of(999) == "c"


def test_asof_series_staleness_bound():
    s = AsOfSeries[str]()
    s.add(100, "x")
    # within staleness
    assert s.as_of(120, max_staleness_us=50) == "x"
    # beyond staleness -> treated as absent
    assert s.as_of(200, max_staleness_us=50) is None


def test_asof_series_stable_latest_wins_on_tie():
    s = AsOfSeries[str]()
    s.add(10, "first")
    s.add(10, "second")  # same ts, added later -> wins
    assert s.as_of(10) == "second"


def test_asof_series_add_after_read_raises():
    s = AsOfSeries[int]()
    s.add(1, 1)
    _ = s.as_of(1)  # finalizes
    with pytest.raises(RuntimeError):
        s.add(2, 2)


def test_asof_series_iter_range_and_bounds():
    s = AsOfSeries[int]()
    for t in (10, 20, 30, 40):
        s.add(t, t)
    assert s.first_ts_us == 10
    assert s.last_ts_us == 40
    assert list(s.iter_range(15, 35)) == [(20, 20), (30, 30)]
    assert len(s) == 4
    assert bool(s) is True


def test_asof_series_empty():
    s = AsOfSeries[int]()
    assert s.as_of(123) is None
    assert s.first_ts_us is None
    assert not s


# ── ordered merge ──────────────────────────────────────────────────────
def test_ordered_merge_sync():
    a = [(10, "a1"), (30, "a2")]
    b = [(20, "b1"), (40, "b2")]
    out = list(ordered_merge([a, b], key=lambda x: x[0]))
    assert [t for t, _ in out] == [10, 20, 30, 40]


def test_ordered_merge_stable_on_tie():
    a = [(10, "from_a")]
    b = [(10, "from_b")]
    # first source wins the tie
    out = list(ordered_merge([a, b], key=lambda x: x[0]))
    assert out[0][1] == "from_a"
    assert out[1][1] == "from_b"


def test_aordered_merge_async():
    async def gen(items):
        for it in items:
            yield it

    async def run():
        a = gen([(10, "a"), (30, "a")])
        b = gen([(20, "b"), (40, "b")])
        return [t async for t in aordered_merge([a, b], key=lambda x: x[0])]

    out = asyncio.run(run())
    assert [t for t, _ in out] == [10, 20, 30, 40]


def test_aordered_merge_handles_empty_source():
    async def gen(items):
        for it in items:
            yield it

    async def run():
        a = gen([])
        b = gen([(5, "b")])
        return [t async for t in aordered_merge([a, b], key=lambda x: x[0])]

    assert asyncio.run(run()) == [(5, "b")]


# ── manifest ────────────────────────────────────────────────────────────
def test_content_hash_order_independent():
    e1 = SnapshotEntry(path="a.parquet", size_bytes=1, mtime_us=1, rows=5)
    e2 = SnapshotEntry(path="b.parquet", size_bytes=2, mtime_us=2, rows=7)
    assert compute_content_hash([e1, e2]) == compute_content_hash([e2, e1])


def test_content_hash_changes_on_rewrite():
    e1 = SnapshotEntry(path="a.parquet", size_bytes=1, mtime_us=1, rows=5)
    e1b = SnapshotEntry(path="a.parquet", size_bytes=1, mtime_us=999, rows=5)
    assert compute_content_hash([e1]) != compute_content_hash([e1b])


def test_content_hash_path_separator_normalized():
    win = SnapshotEntry(path="dir\\a.parquet", size_bytes=1, mtime_us=1, rows=1)
    nix = SnapshotEntry(path="dir/a.parquet", size_bytes=1, mtime_us=1, rows=1)
    assert compute_content_hash([win]) == compute_content_hash([nix])


def test_build_snapshot_from_real_files(tmp_path):
    f1 = tmp_path / "snapshots__tokA.parquet"
    f2 = tmp_path / "snapshots__tokB.parquet"
    f1.write_bytes(b"x" * 10)
    f2.write_bytes(b"y" * 20)
    snap = build_snapshot([f1, f2, tmp_path / "missing.parquet"])
    assert len(snap.entries) == 2  # missing file skipped
    assert snap.content_hash.startswith("sha256:")
    assert snap.schema_version == BOOK_SCHEMA_VERSION
    assert snap.contains_path(f1)
    assert not snap.contains_path(tmp_path / "missing.parquet")


def test_snapshot_roundtrip_and_aggregates():
    entries = (
        SnapshotEntry(path="a", size_bytes=1, mtime_us=1, rows=3, token_ids=("t1",), start_us=10, end_us=20),
        SnapshotEntry(path="b", size_bytes=2, mtime_us=2, rows=4, token_ids=("t2",), start_us=15, end_us=30),
    )
    snap = DatasetSnapshot(
        entries=entries,
        schema_version="1",
        created_at_us=12345,
        content_hash=compute_content_hash(entries),
    )
    assert snap.total_rows == 7
    assert set(snap.token_ids) == {"t1", "t2"}
    assert snap.span_us == (10, 30)
    back = DatasetSnapshot.from_dict(snap.to_dict())
    assert back.content_hash == snap.content_hash
    assert back.total_rows == 7
    assert back.span_us == (10, 30)


# ── schema authority ────────────────────────────────────────────────────
def test_schema_for_unknown_kind_raises():
    with pytest.raises(SchemaValidationError):
        schema_for("bogus")


def test_validate_table_accepts_canonical():
    schema = schema_for("snapshots")
    # build an empty table with the canonical schema -> valid
    table = pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)
    validate_table(table, "snapshots")  # no raise


def test_validate_arrow_schema_reports_missing_column():
    schema = schema_for("snapshots")
    # drop one canonical column
    reduced = pa.schema([f for f in schema if f.name != "best_bid"])
    problems = validate_arrow_schema(reduced, "snapshots")
    assert any("best_bid" in p for p in problems)


def test_validate_arrow_schema_allows_int_width_mismatch():
    # observed_at_us int64 in canonical; int32 should be accepted (compatible)
    fields = []
    for f in schema_for("snapshots"):
        if f.name == "observed_at_us":
            fields.append(pa.field("observed_at_us", pa.int32()))
        else:
            fields.append(f)
    problems = [p for p in validate_arrow_schema(pa.schema(fields), "snapshots") if not p.startswith("note:")]
    assert problems == []


def test_validate_table_raises_on_bad_schema():
    bad = pa.Table.from_pydict({"only_col": [1, 2, 3]})
    with pytest.raises(SchemaValidationError):
        validate_table(bad, "snapshots")


def test_validate_file_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    schema = schema_for("snapshots")
    table = pa.Table.from_pydict(
        {f.name: ([None] if not pa.types.is_string(f.type) and not pa.types.is_list(f.type) else [None]) for f in schema},
        schema=schema,
    )
    # one row of valid-ish data
    row = {
        "token_id": ["tok"], "observed_at_us": [1], "sequence": [1],
        "best_bid": [0.4], "best_ask": [0.6], "spread_bps": [None],
        "bids_price": [[0.4]], "bids_size": [[1.0]],
        "asks_price": [[0.6]], "asks_size": [[1.0]],
        "trade_price": [None], "trade_size": [None], "trade_side": [None],
    }
    table = pa.Table.from_pydict(row, schema=schema)
    path = tmp_path / "snapshots__tok.parquet"
    pq.write_table(table, str(path))
    ok, reason = validate_file(path, "snapshots")
    assert ok, reason
    # missing file
    ok2, reason2 = validate_file(tmp_path / "nope.parquet", "snapshots")
    assert not ok2 and "unreadable" in (reason2 or "")
