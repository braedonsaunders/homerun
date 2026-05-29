"""Tests for the single canonical book/delta parquet writer."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.schema import BOOK_SCHEMA_VERSION, SchemaValidationError
from services.marketdata.writer import write_canonical_table


def _valid_snapshot_table(n=2):
    return pa.table(
        {
            "token_id": pa.array(["T"] * n, pa.string()),
            "observed_at_us": pa.array([1, 2][:n], pa.int64()),
            "sequence": pa.array([1, 2][:n], pa.int64()),
            "best_bid": pa.array([0.4, 0.41][:n], pa.float64()),
            "best_ask": pa.array([0.42, 0.43][:n], pa.float64()),
            "spread_bps": pa.array([None] * n, pa.float64()),
            "bids_price": pa.array([[0.4]] * n, pa.list_(pa.float64())),
            "bids_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "asks_price": pa.array([[0.42]] * n, pa.list_(pa.float64())),
            "asks_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "trade_price": pa.array([None] * n, pa.float64()),
            "trade_size": pa.array([None] * n, pa.float64()),
            "trade_side": pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )


def test_write_canonical_table_writes_and_stamps_lineage(tmp_path):
    dest = tmp_path / "snapshots__T.parquet"
    rows = write_canonical_table(
        _valid_snapshot_table(2), dest_path=dest, kind="snapshots",
        provider="polybacktest", job_id="job-123",
    )
    assert rows == 2
    assert dest.exists()
    # no temp file left behind (atomic)
    assert not (tmp_path / "snapshots__T.parquet.tmp").exists()
    # lineage stamped into footer metadata
    meta = pq.read_schema(str(dest)).metadata or {}
    assert meta.get(b"schema_version") == str(BOOK_SCHEMA_VERSION).encode()
    assert meta.get(b"canonical_kind") == b"snapshots"
    assert meta.get(b"provider") == b"polybacktest"
    assert meta.get(b"import_job_id") == b"job-123"
    # round-trips through the canonical reader
    back = pq.read_table(str(dest))
    assert back.num_rows == 2


def test_write_canonical_table_rejects_bad_schema(tmp_path):
    bad = pa.table({"only_col": [1, 2, 3]})
    dest = tmp_path / "snapshots__bad.parquet"
    with pytest.raises(SchemaValidationError):
        write_canonical_table(bad, dest_path=dest, kind="snapshots")
    # fail-closed: nothing written, no temp file
    assert not dest.exists()
    assert not (tmp_path / "snapshots__bad.parquet.tmp").exists()


def test_provider_protocol_is_structural():
    from services.marketdata.provider import MarketDataProvider

    class _Stub:
        provider_name = "stub"

        async def fetch_to_canonical(self, *, coin, market_ids, start, end):
            return {}

    assert isinstance(_Stub(), MarketDataProvider)
