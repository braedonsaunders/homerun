"""Parquet schema + path helpers for the bring-your-own-data backtest path.

Operators drop parquet files under ``HOMERUN_PARQUET_ROOT`` (default
``<repo>/data/parquet``).  The auto-discovery scanner
(``parquet_scanner.py``) walks the root, validates schema, and inserts
matching rows into ``provider_datasets`` with ``storage_type='parquet'``.
The backtester's ``ParquetBookReplay`` then reads the file directly —
no Postgres round-trip — when a backtest's opp tokens fall inside a
covered window.

Two schema variants are supported:

  * ``snapshots`` — point-in-time L2 book snapshots.  The dominant case;
    matches the columns ``MarketMicrostructureSnapshot`` (book) carries.
  * ``deltas``    — book-delta events (per-level changes).  Optional;
    consumed by the matcher when delta coverage is materially denser
    than snapshot coverage.

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

import os
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

# Delta kind: one row per book-level change.  Same shape as
# ``BookDeltaEvent`` — kept compact for high-frequency events.
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


def parquet_root() -> Path:
    """Storage root.  Defaults to ``<repo>/data/parquet`` so a fresh
    install just works; operators can override via env to point at a
    larger volume or shared mount.
    """
    raw = os.environ.get("HOMERUN_PARQUET_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    # backend/services/external_data/parquet_schema.py → ../../../../data/parquet
    here = Path(__file__).resolve()
    return (here.parents[3] / "data" / "parquet").resolve()


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


__all__ = [
    "SNAPSHOT_SCHEMA",
    "DELTA_SCHEMA",
    "SCHEMA_VERSION",
    "ParquetPath",
    "parquet_root",
    "parquet_path_for",
    "schema_for",
]
