"""Unified market-data layer.

This package is the single, point-in-time-correct foundation for reading
recorded/imported market data — used by both the backtest engine and (at
``as_of=now``) live strategies, so live and replay share one access path.

Layering (built across phases):

  * ``asof``     — the one point-in-time (<= tau) lookup primitive.
  * ``merge``    — the one ordered k-way merge (sync + async).
  * ``manifest`` — content-hashed DatasetSnapshot for reproducible runs.
  * ``schema``   — the single schema authority (validate on read + write).
  * ``view``     — (Phase 2) ``MarketDataView``: book_at / iter_books /
                   events / coverage, resolving across canonical parquet +
                   the event bus, hiding source selection from callers.

Phase 1 ships the bottom four as pure, independently-tested primitives with
no wiring into the engine yet (no behavior change).
"""
from __future__ import annotations

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

__all__ = [
    "AsOfSeries",
    "bisect_as_of",
    "ordered_merge",
    "aordered_merge",
    "DatasetSnapshot",
    "SnapshotEntry",
    "build_snapshot",
    "compute_content_hash",
    "BOOK_SCHEMA_VERSION",
    "SchemaValidationError",
    "schema_for",
    "validate_arrow_schema",
    "validate_file",
    "validate_table",
]
