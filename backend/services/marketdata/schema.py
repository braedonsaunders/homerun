"""The single schema authority for canonical market-data parquet.

Today schema knowledge is scattered: the pyarrow ``SNAPSHOT_SCHEMA`` /
``DELTA_SCHEMA`` live in ``external_data.parquet_schema``; the parquet-scanner
re-implements a column-presence check; the live sink, the telonex importer and
``parquet_replay.write_snapshots`` each write with their own copy of the
schema and none validate on read. This module is the one place that owns:

  * ``schema_for(kind)``        — the canonical pyarrow schema per kind,
  * ``validate_arrow_schema``   — structural validation (names + types),
  * ``validate_table`` /
    ``validate_file``           — the read/write validation primitive every
                                  writer and reader should call,
  * ``BOOK_SCHEMA_VERSION``     — the single column-layout version constant.

It deliberately *wraps* (does not re-declare) the arrow schemas so there is
still exactly one definition of the columns; this module adds the validation
+ versioning authority on top.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pyarrow as pa

from services.external_data.parquet_schema import (
    DELTA_SCHEMA,
    SNAPSHOT_SCHEMA,
    SCHEMA_VERSION as _PARQUET_SCHEMA_VERSION,
)

# Single source of truth for the canonical book/delta column-layout version.
# (Mirrors the value stamped into parquet footers by the writers.)
BOOK_SCHEMA_VERSION: str = str(_PARQUET_SCHEMA_VERSION)

Kind = Literal["snapshots", "deltas"]

_SCHEMAS: dict[str, pa.Schema] = {
    "snapshots": SNAPSHOT_SCHEMA,
    "deltas": DELTA_SCHEMA,
}


class SchemaValidationError(ValueError):
    """Raised when a table/file does not conform to the canonical schema."""


def schema_for(kind: str) -> pa.Schema:
    """Return the canonical pyarrow schema for ``kind`` ('snapshots'|'deltas')."""
    try:
        return _SCHEMAS[kind]
    except KeyError:
        raise SchemaValidationError(
            f"unknown canonical schema kind {kind!r}; expected one of {sorted(_SCHEMAS)}"
        ) from None


def validate_arrow_schema(schema: pa.Schema, kind: str) -> list[str]:
    """Structurally validate a pyarrow ``schema`` against the canonical one.

    Returns a list of human-readable problems (empty when valid). Checks that
    every canonical column is present with a compatible type. Extra columns are
    permitted (forward-compatible) but reported as an informational note so the
    writers can be tightened over time.
    """
    expected = schema_for(kind)
    problems: list[str] = []
    present = {f.name: f.type for f in schema}
    for field in expected:
        if field.name not in present:
            problems.append(f"missing column {field.name!r} ({field.type})")
            continue
        actual = present[field.name]
        if not _types_compatible(actual, field.type):
            problems.append(
                f"column {field.name!r} type {actual} incompatible with canonical {field.type}"
            )
    extra = [n for n in present if n not in set(expected.names)]
    if extra:
        problems.append(f"note: extra columns not in canonical schema: {sorted(extra)}")
    return problems


def _types_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    """Lenient type check: exact match, or both integer / both floating, or
    both list-of-floating (the bid/ask ladders). Keeps validation from being
    brittle to int32-vs-int64 / float32-vs-float64 differences between writers.
    """
    if actual.equals(expected):
        return True
    if pa.types.is_integer(actual) and pa.types.is_integer(expected):
        return True
    if pa.types.is_floating(actual) and pa.types.is_floating(expected):
        return True
    if pa.types.is_list(actual) and pa.types.is_list(expected):
        return _types_compatible(actual.value_type, expected.value_type)
    # string vs large_string
    if pa.types.is_string(actual) and pa.types.is_string(expected):
        return True
    if pa.types.is_large_string(actual) and pa.types.is_string(expected):
        return True
    return False


def validate_table(table: pa.Table, kind: str) -> None:
    """Raise :class:`SchemaValidationError` if ``table`` violates the canonical
    schema (ignoring the informational extra-columns note). Writers should call
    this before persisting; cheap (schema-only, no data scan)."""
    problems = [p for p in validate_arrow_schema(table.schema, kind) if not p.startswith("note:")]
    if problems:
        raise SchemaValidationError(
            f"{kind} table fails canonical schema: " + "; ".join(problems)
        )


def validate_file(path: str | Path, kind: str) -> tuple[bool, str | None]:
    """Footer-only validation of an on-disk parquet file (no data scan).

    Returns ``(ok, reason)``. ``ok`` is False with a reason when the file is
    unreadable, has zero rows, or violates the canonical schema. This is the
    single primitive the parquet scanner / coverage layer should use.
    """
    import pyarrow.parquet as pq

    p = str(path)
    try:
        meta = pq.read_metadata(p)
        file_schema = pq.read_schema(p)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {str(exc)[:160]}"
    problems = [x for x in validate_arrow_schema(file_schema, kind) if not x.startswith("note:")]
    if problems:
        return False, "; ".join(problems)
    if meta.num_rows <= 0:
        return False, "empty file (0 rows)"
    return True, None


__all__ = [
    "BOOK_SCHEMA_VERSION",
    "Kind",
    "SchemaValidationError",
    "schema_for",
    "validate_arrow_schema",
    "validate_table",
    "validate_file",
]
