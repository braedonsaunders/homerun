"""Parquet schema + path helpers for the bring-your-own-data backtest path.

Operators drop parquet files under one of the directories configured in
Data Lab → Providers → Parquet.  Multiple roots can be configured (e.g.
one per vendor, one for shared data, etc.); the scanner walks all of
them in order.  When no roots are configured the built-in default
``<repo>/data/parquet`` is used.

The auto-discovery scanner (``parquet_scanner.py``) walks each root,
validates schema, and inserts matching rows into ``provider_datasets``
with ``storage_type='parquet'``.  The unified ``MarketDataView`` then
reads the file directly — no Postgres round-trip — when a backtest's
opp tokens fall inside a covered window.

Two schema variants are supported:

  * ``snapshots`` — point-in-time L2 book snapshots.  The dominant case;
    carries best bid/ask + full L2 ladder columns.
  * ``deltas``    — book-delta events (per-level changes).  Optional;
    feeds fill-model calibration (``marketdata.aggregate_delta_events``).

The ``token_id`` is stored as ``string`` rather than ``int64`` because
Polymarket CLOB asset IDs are 256-bit decimals that don't fit native
integer types — converting at write time would lose precision and
break round-tripping.

File layout on disk:

    {root}/
      {provider}/                    # e.g. polybacktest, vendor_acme
        {coin_or_asset_class}/       # e.g. btc, eth, prediction
          {start_iso}__{end_iso}/    # window slug, ``YYYYMMDDTHHMMSS``
            {kind}__{token_id}.parquet

Files are immutable per ``(provider, coin, window, kind, token_id)``.
A re-import for the same key atomically replaces (write to ``.tmp``
then ``os.rename``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa


# ── Schemas ──────────────────────────────────────────────────────────

# Snapshot kind: one row per book observation (top-of-book + N-deep
# bids/asks).  Sorted by ``observed_at_us`` within each token's file.
SNAPSHOT_SCHEMA: pa.Schema = pa.schema(
    [
        ("token_id", pa.string()),
        ("observed_at_us", pa.int64()),  # epoch microseconds (sortable)
        ("sequence", pa.int64()),  # nullable
        ("best_bid", pa.float64()),
        ("best_ask", pa.float64()),
        ("spread_bps", pa.float64()),  # nullable
        # Per-side level arrays (parallel; index i is one level).
        # ``list<float64>`` keeps row-group compression efficient and
        # avoids the JSON-blob overhead of mms.bids_json/asks_json.
        ("bids_price", pa.list_(pa.float64())),
        ("bids_size", pa.list_(pa.float64())),
        ("asks_price", pa.list_(pa.float64())),
        ("asks_size", pa.list_(pa.float64())),
        # Trade tape — populated only for the snapshot type recording
        # the matched print; null otherwise.
        ("trade_price", pa.float64()),
        ("trade_size", pa.float64()),
        ("trade_side", pa.string()),  # 'BUY' | 'SELL' | null
    ]
)

# Delta kind: one row per book-level change (event_type, side, price,
# trade/cancel size) — kept compact for high-frequency events.
DELTA_SCHEMA: pa.Schema = pa.schema(
    [
        ("token_id", pa.string()),
        ("observed_at_us", pa.int64()),
        ("sequence", pa.int64()),
        ("event_type", pa.string()),  # 'trade' | 'cancel'
        ("side", pa.string()),  # 'bid' | 'ask'
        ("price", pa.float64()),
        ("trade_size", pa.float64()),
        ("cancel_size", pa.float64()),
        ("queue_depth_before", pa.float64()),
        ("queue_depth_after", pa.float64()),
        ("spread_bps_at_event", pa.float64()),
    ]
)

# Schema version embedded in file footer metadata so future readers
# can route forward-compatibly.  Bumped whenever the schema gains a
# required column.
SCHEMA_VERSION = "1"


# ── Path helpers ─────────────────────────────────────────────────────


# Process-level cache of the UI-set roots.  Populated by API handlers
# (GET/PUT /providers/parquet/root) so callers of ``parquet_roots()``
# (sync, called from every parquet path-builder + the backtester hot
# path) don't have to await a DB round-trip.  Empty list means "no
# UI overrides set — use the built-in default location".
_PARQUET_ROOT_OVERRIDES: list[str] = []


def _builtin_default_root() -> Path:
    """The fall-back root used when the operator hasn't configured any.
    Lives at ``<repo>/data/parquet`` so a fresh install just works.
    """
    # backend/services/external_data/parquet_schema.py → ../../../../data/parquet
    here = Path(__file__).resolve()
    return (here.parents[3] / "data" / "parquet").resolve()


def set_parquet_root_overrides(values: list[str] | None) -> None:
    """Update the in-memory list of UI-configured roots.  Called by
    the API handlers (PUT /providers/parquet/root + GET hydrate path)
    after persisting to ``app_settings.parquet_root_overrides`` so
    subsequent ``parquet_roots()`` calls see the new state without a
    DB round-trip.

    Pass ``None`` or ``[]`` to clear all overrides — the resolver
    will fall back to the built-in default location.

    Empty/whitespace-only entries are dropped.  Duplicates are
    de-duped while preserving insertion order.
    """
    global _PARQUET_ROOT_OVERRIDES
    if not values:
        _PARQUET_ROOT_OVERRIDES = []
        return
    cleaned: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        cleaned.append(s)
        seen.add(s)
    _PARQUET_ROOT_OVERRIDES = cleaned


def parquet_roots() -> list[Path]:
    """All configured parquet ingest roots.  Returns the UI-set list
    when populated; otherwise a single-element list containing the
    built-in default ``<repo>/data/parquet``.

    The scanner walks every entry in order; the backtester's
    per-token coverage lookup queries against ``provider_datasets``
    rows whose paths fall under any of these roots.  Sync function
    — no DB I/O, reads from the in-process cache.
    """
    if _PARQUET_ROOT_OVERRIDES:
        return [Path(p).expanduser().resolve() for p in _PARQUET_ROOT_OVERRIDES]
    return [_builtin_default_root()]


def parquet_root() -> Path:
    """Primary write root — the destination ``parquet_path_for`` uses
    when a caller doesn't explicitly pick one.  Returns the FIRST
    configured root, or the built-in default when none configured.

    Reads happen against ALL configured roots via ``parquet_roots()``;
    this helper exists only because path-construction needs a single
    canonical destination.  Tests + the synthetic-import writer use it.
    """
    return parquet_roots()[0]


def parquet_root_source() -> str:
    """Which layer of the resolution chain is currently providing
    roots.  Returned by ``GET /providers/parquet/root`` so the UI
    can show provenance.  No env-var path anymore — operators
    configure roots exclusively from Data Lab → Providers → Parquet.
    """
    if _PARQUET_ROOT_OVERRIDES:
        return "configured"
    return "default"


def _iso_slug(dt: datetime) -> str:
    """Compact ISO timestamp suitable for a directory name.  Strips
    timezone (assumed UTC), seconds-precision, no separators that
    fight Windows or shells.  Example: ``20260429T020832``.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _safe_segment(value: str) -> str:
    """Trim a path segment to characters that survive every common FS.
    Removes path separators and trims to 64 chars.  Polymarket token
    ids are 78-char decimals; we hash-collapse longer names rather
    than truncating (truncation would alias different tokens).
    """
    cleaned = "".join(c for c in str(value or "") if c.isalnum() or c in {"-", "_"})
    if len(cleaned) <= 78:
        return cleaned
    # Token-id-ish: keep the first 16 chars + last 8 + a 6-char hash
    # of the middle to disambiguate.  Total fits in 32 chars.
    import hashlib
    middle_hash = hashlib.sha256(cleaned.encode()).hexdigest()[:6]
    return f"{cleaned[:16]}_{middle_hash}_{cleaned[-8:]}"


@dataclass(frozen=True)
class ParquetPath:
    """Resolved file path + the directory components used to build it.

    ``window_dir`` is exposed because the auto-discovery scanner uses
    it to group sibling files into a single ``provider_datasets`` row.
    """

    root: Path
    provider: str
    coin: str
    window_dir: Path
    file_path: Path
    kind: str
    token_id: str
    start: datetime
    end: datetime


def parquet_path_for(
    *,
    provider: str,
    coin: str,
    token_id: str,
    start: datetime,
    end: datetime,
    kind: str = "snapshots",
    root: Path | None = None,
) -> ParquetPath:
    """Build the canonical on-disk location for a (provider, coin,
    token, window, kind) tuple.  Pure function — does not create
    directories or touch the filesystem.
    """
    if kind not in {"snapshots", "deltas"}:
        raise ValueError(f"unknown parquet kind: {kind!r} (expected 'snapshots' or 'deltas')")
    base = (root or parquet_root()).resolve()
    window_slug = f"{_iso_slug(start)}__{_iso_slug(end)}"
    window_dir = base / _safe_segment(provider) / _safe_segment(coin or "_") / window_slug
    file_path = window_dir / f"{kind}__{_safe_segment(token_id)}.parquet"
    return ParquetPath(
        root=base,
        provider=provider,
        coin=coin,
        window_dir=window_dir,
        file_path=file_path,
        kind=kind,
        token_id=token_id,
        start=start,
        end=end,
    )


def schema_for(kind: str) -> pa.Schema:
    if kind == "snapshots":
        return SNAPSHOT_SCHEMA
    if kind == "deltas":
        return DELTA_SCHEMA
    raise ValueError(f"unknown parquet kind: {kind!r}")


# ── Bundle layout (v2) ───────────────────────────────────────────────
#
# The legacy layout wrote ONE FILE PER (kind, token) per window:
# ``{kind}__{token_id}.parquet``.  At live-recording scale (~17k tokens per
# 15-min window) that is >1M files/day — NTFS, every tree walker, the
# pruner and the catalog scanner all degrade with file count (a single
# storage-summary walk wedged the API event loop for minutes).
#
# The v2 layout stores ONE BUNDLE PER (kind, window):
#
#     {window_dir}/{kind}.parquet          # all tokens; token_id is a column,
#                                          # rows sorted (token_id, observed_at_us)
#                                          # so row-group stats give per-token
#                                          # predicate pushdown on read
#     {window_dir}/manifest.json           # {kind: {"tokens": [...], "rows": N}}
#                                          # discovery without column scans
#
# The CURRENT (still-open) window accumulates small immutable increments in
# a ``_parts/`` subdirectory (invisible to every legacy reader — they only
# glob files directly inside the window dir):
#
#     {window_dir}/_parts/{kind}__part-{seq:06d}.parquet
#
# A compactor merges parts -> bundle when the window closes, verifies row
# counts, then deletes the parts.  Readers prefer the bundle and fall back
# to legacy per-token files, so both layouts stay readable indefinitely
# (operator-imported per-token datasets are small and stay as-is).

PARTS_DIRNAME = "_parts"
MANIFEST_FILENAME = "manifest.json"


def bundle_path_for(window_dir: Path, kind: str) -> Path:
    """The single-file bundle for *kind* inside *window_dir* (v2 layout)."""
    if kind not in {"snapshots", "deltas"}:
        raise ValueError(f"unknown parquet kind: {kind!r} (expected 'snapshots' or 'deltas')")
    return Path(window_dir) / f"{kind}.parquet"


def parts_dir_for(window_dir: Path) -> Path:
    return Path(window_dir) / PARTS_DIRNAME


def manifest_path_for(window_dir: Path) -> Path:
    return Path(window_dir) / MANIFEST_FILENAME


__all__ = [
    "SNAPSHOT_SCHEMA",
    "DELTA_SCHEMA",
    "SCHEMA_VERSION",
    "PARTS_DIRNAME",
    "MANIFEST_FILENAME",
    "ParquetPath",
    "parquet_root",
    "parquet_roots",
    "parquet_root_source",
    "set_parquet_root_overrides",
    "parquet_path_for",
    "bundle_path_for",
    "parts_dir_for",
    "manifest_path_for",
    "schema_for",
]
