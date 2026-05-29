"""The single canonical writer for book/delta parquet.

Three call sites used to each ``pq.write_table`` the canonical SNAPSHOT/DELTA
schemas with their own copy of the write logic — the live ingestor sink, the
polybacktest importer, and the telonex importer — and NONE validated the table
against the schema before persisting. This is the one place canonical book
parquet is written: it validates against the schema authority
(``services.marketdata.schema``) at the write boundary, stamps footer
lineage metadata (schema version, kind, provider, optional job id), and writes
atomically (``.tmp`` + ``os.replace``) so a reader never sees a partial file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from services.marketdata.schema import BOOK_SCHEMA_VERSION, validate_table


def write_canonical_table(
    table: pa.Table,
    *,
    dest_path: str | Path,
    kind: str,
    provider: Optional[str] = None,
    compression: str = "zstd",
    job_id: Optional[str] = None,
) -> int:
    """Validate, lineage-stamp, and atomically write a canonical book/delta table.

    ``kind`` is 'snapshots' | 'deltas'.  Raises
    :class:`~services.marketdata.schema.SchemaValidationError` if the table does
    not conform to the canonical schema (fail-closed at the write boundary).
    Returns the number of rows written.
    """
    validate_table(table, kind)  # fail-closed before any bytes hit disk

    metadata: dict[bytes, bytes] = {
        b"schema_version": str(BOOK_SCHEMA_VERSION).encode(),
        b"canonical_kind": str(kind).encode(),
    }
    if provider:
        metadata[b"provider"] = str(provider).encode()
    if job_id:
        metadata[b"import_job_id"] = str(job_id).encode()
    # Preserve any existing footer metadata, then overlay ours.
    existing = table.schema.metadata or {}
    merged = {**existing, **metadata}
    table = table.replace_schema_metadata(merged)

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    pq.write_table(table, str(tmp), compression=compression)
    os.replace(tmp, dest)
    return int(table.num_rows)


__all__ = ["write_canonical_table"]
