from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from services.external_data.parquet_compactor import compact_window_dir, optimize_bundle_row_groups
from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.book import load_book_series


def _write_snapshot_file(path, token_id: str, observed: list[int]) -> None:
    n = len(observed)
    table = pa.table(
        {
            "token_id": pa.array([token_id] * n, pa.string()),
            "observed_at_us": pa.array(observed, pa.int64()),
            "sequence": pa.array(list(range(n)), pa.int64()),
            "best_bid": pa.array([0.40] * n, pa.float64()),
            "best_ask": pa.array([0.42] * n, pa.float64()),
            "spread_bps": pa.array([None] * n, pa.float64()),
            "bids_price": pa.array([[0.40]] * n, pa.list_(pa.float64())),
            "bids_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "asks_price": pa.array([[0.42]] * n, pa.list_(pa.float64())),
            "asks_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "trade_price": pa.array([None] * n, pa.float64()),
            "trade_size": pa.array([None] * n, pa.float64()),
            "trade_side": pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )
    pq.write_table(table, str(path))


def test_compact_window_dir_writes_verified_bundle_and_manifest(tmp_path):
    window_dir = tmp_path / "live_ingestor" / "_" / "20260101T000000__20260101T001500"
    window_dir.mkdir(parents=True)
    base = 1_767_225_600_000_000
    _write_snapshot_file(window_dir / "snapshots__tok_a.parquet", "tok_a", [base, base + 1_000_000])
    _write_snapshot_file(window_dir / "snapshots__tok_b.parquet", "tok_b", [base + 2_000_000])

    result = compact_window_dir(window_dir)

    bundle = window_dir / "snapshots.parquet"
    assert result.error is None
    assert result.kinds_compacted == ["snapshots"]
    assert result.rows == 3
    assert result.source_files_removed == 2
    assert bundle.exists()
    assert not (window_dir / "snapshots__tok_a.parquet").exists()
    manifest = json.loads((window_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshots"]["rows"] == 3
    assert manifest["snapshots"]["tokens"] == ["tok_a", "tok_b"]

    series, rows = load_book_series("tok_a", [bundle], start_us=base, end_us=base + 3_000_000)
    assert rows == 2
    assert len(series) == 2


def test_optimize_bundle_row_groups_rewrites_fragmented_bundle(tmp_path):
    window_dir = tmp_path / "live_ingestor" / "_" / "20260101T001500__20260101T003000"
    window_dir.mkdir(parents=True)
    bundle = window_dir / "snapshots.parquet"
    base = 1_767_226_500_000_000
    writer = pq.ParquetWriter(str(bundle), SNAPSHOT_SCHEMA, compression="zstd")
    for idx in range(4):
        table = pa.table(
            {
                "token_id": pa.array(["tok_a"], pa.string()),
                "observed_at_us": pa.array([base + idx], pa.int64()),
                "sequence": pa.array([idx], pa.int64()),
                "best_bid": pa.array([0.40], pa.float64()),
                "best_ask": pa.array([0.42], pa.float64()),
                "spread_bps": pa.array([None], pa.float64()),
                "bids_price": pa.array([[0.40]], pa.list_(pa.float64())),
                "bids_size": pa.array([[10.0]], pa.list_(pa.float64())),
                "asks_price": pa.array([[0.42]], pa.list_(pa.float64())),
                "asks_size": pa.array([[10.0]], pa.list_(pa.float64())),
                "trade_price": pa.array([None], pa.float64()),
                "trade_size": pa.array([None], pa.float64()),
                "trade_side": pa.array([None], pa.string()),
            },
            schema=SNAPSHOT_SCHEMA,
        )
        writer.write_table(table)
    writer.close()
    (window_dir / "manifest.json").write_text(
        json.dumps({"snapshots": {"rows": 4, "tokens": ["tok_a"], "row_groups": 4}}),
        encoding="utf-8",
    )
    assert pq.read_metadata(str(bundle)).num_row_groups == 4

    result = optimize_bundle_row_groups(tmp_path, workers=1)

    assert result.errors == []
    assert result.files_rewritten == 1
    assert result.rows_rewritten == 4
    assert pq.read_metadata(str(bundle)).num_row_groups == 1
    manifest = json.loads((window_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshots"]["row_groups"] == 1
    assert manifest["snapshots"]["rows"] == 4
