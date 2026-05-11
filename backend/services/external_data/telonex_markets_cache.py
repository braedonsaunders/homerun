"""Local cache of Telonex's public markets dataset.

The ``GET /v1/datasets/polymarket/markets`` endpoint returns a single
~660 MB Parquet file (1.28M rows as of 2026-05) with the full Polymarket
market catalog *plus* per-channel availability windows
(``trades_from`` / ``trades_to`` etc.) baked right into each row.  That
makes it the single source of truth for "what can I import from
Telonex" — no need to call the separate availability endpoint for
Polymarket discovery.

We cache the file locally under the parquet ingest root and serve
paginated, filtered queries to the UI via PyArrow's column-projection
+ row-group filter pushdown so the 660 MB file never has to be loaded
fully into memory.

Refresh strategy:
  * On-demand only (explicit operator click), because re-downloading
    660 MB on every backend restart is wasteful and the catalog updates
    daily after midnight UTC.
  * A cache stamp file records the download timestamp so the UI can
    show "catalog last refreshed: 6h ago" and warn when stale.

Binance has no markets dataset — operators type the symbol directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Columns we project for the UI market browser.  Don't load the long
# `description` / `rules_url` fields — they bloat the result and the
# UI doesn't need them for the list view.
_LIST_COLUMNS: tuple[str, ...] = (
    "market_id",
    "slug",
    "event_id",
    "event_slug",
    "event_title",
    "question",
    "category",
    "outcome_0",
    "outcome_1",
    "asset_id_0",
    "asset_id_1",
    "status",
    "start_date_us",
    "end_date_us",
    "settled_at_us",
    "tags",
    "trades_from",
    "trades_to",
    "quotes_from",
    "quotes_to",
    "book_snapshot_5_from",
    "book_snapshot_5_to",
    "book_snapshot_25_from",
    "book_snapshot_25_to",
    "book_snapshot_full_from",
    "book_snapshot_full_to",
    "onchain_fills_from",
    "onchain_fills_to",
)

# Channel-availability column pairs.  Used both to filter ("only show
# markets that have <channel> data") and to emit the per-channel windows
# in the response payload.
_CHANNEL_DATE_COLS: dict[str, tuple[str, str]] = {
    "trades": ("trades_from", "trades_to"),
    "quotes": ("quotes_from", "quotes_to"),
    "book_snapshot_5": ("book_snapshot_5_from", "book_snapshot_5_to"),
    "book_snapshot_25": ("book_snapshot_25_from", "book_snapshot_25_to"),
    "book_snapshot_full": ("book_snapshot_full_from", "book_snapshot_full_to"),
    "onchain_fills": ("onchain_fills_from", "onchain_fills_to"),
}


# ---------------------------------------------------------------------------
# Catalog filesystem layout
# ---------------------------------------------------------------------------


def catalog_dir() -> Path:
    """Telonex's local catalog/data root, beneath the parquet ingest root.

    Lives in its own directory so the parquet_scanner (which expects
    Homerun's snapshot/delta schema) doesn't accidentally try to index
    raw Telonex parquets.
    """
    from services.external_data.parquet_schema import parquet_root

    return parquet_root() / "_telonex"


def markets_catalog_path(exchange: str) -> Path:
    return catalog_dir() / "catalog" / f"{exchange.lower()}_markets.parquet"


def markets_catalog_stamp_path(exchange: str) -> Path:
    return catalog_dir() / "catalog" / f"{exchange.lower()}_markets.stamp.json"


@dataclass(frozen=True)
class CatalogStatus:
    exchange: str
    exists: bool
    size_bytes: int
    downloaded_at_epoch: Optional[float]
    rows: Optional[int]
    path: str


def catalog_status(exchange: str) -> CatalogStatus:
    p = markets_catalog_path(exchange)
    stamp = markets_catalog_stamp_path(exchange)
    if not p.exists():
        return CatalogStatus(exchange=exchange, exists=False, size_bytes=0,
                             downloaded_at_epoch=None, rows=None, path=str(p))
    downloaded_at: Optional[float] = None
    rows: Optional[int] = None
    if stamp.exists():
        try:
            data = json.loads(stamp.read_text(encoding="utf-8"))
            downloaded_at = data.get("downloaded_at_epoch")
            rows = data.get("rows")
        except Exception:
            pass
    return CatalogStatus(
        exchange=exchange,
        exists=True,
        size_bytes=p.stat().st_size,
        downloaded_at_epoch=downloaded_at,
        rows=rows,
        path=str(p),
    )


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


async def refresh_markets_catalog(exchange: str = "polymarket") -> dict[str, Any]:
    """Download a fresh copy of the public markets dataset.

    Public endpoint — costs zero downloads against the operator's quota.
    Atomically replaces the existing cache (writes to .tmp, renames).
    """
    from services.external_data.telonex_client import build_client_from_settings

    target = markets_catalog_path(exchange)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = markets_catalog_stamp_path(exchange)

    client = await build_client_from_settings(require_api_key=False)
    try:
        url = client.dataset_url(exchange, "markets")
        started = time.time()
        bytes_written = await client.stream_to_path(url, target)
    finally:
        await client.close()

    # Stamp + row count.  Reading the row count requires opening the
    # file's metadata footer; cheap (no full scan).
    rows: Optional[int] = None
    try:
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(str(target)).metadata.num_rows
    except Exception:
        logger.debug("markets catalog row count read failed", exc_info=True)

    stamp.write_text(
        json.dumps(
            {
                "exchange": exchange,
                "downloaded_at_epoch": time.time(),
                "elapsed_seconds": round(time.time() - started, 2),
                "bytes": int(bytes_written),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "exchange": exchange,
        "path": str(target),
        "bytes": int(bytes_written),
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    return s if s else None


def _us_to_iso(value: Any) -> Optional[str]:
    """Convert microseconds-epoch to ISO-8601 UTC.  Returns None for
    null/zero/invalid inputs.  All Telonex timestamps use this format.
    """
    try:
        if value is None:
            return None
        ms_or_us = int(value)
        if ms_or_us <= 0:
            return None
        # Some columns are us, others may be ns or ms — heuristically
        # detect by magnitude.  Telonex uses µs for *_at_us cols.
        from datetime import datetime, timezone

        if ms_or_us > 10**18:  # ns
            ts = ms_or_us / 1_000_000_000
        elif ms_or_us > 10**14:  # us
            ts = ms_or_us / 1_000_000
        elif ms_or_us > 10**11:  # ms
            ts = ms_or_us / 1_000
        else:  # seconds
            ts = float(ms_or_us)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


async def list_markets(
    exchange: str = "polymarket",
    *,
    search: Optional[str] = None,
    status: Optional[str] = None,
    channel: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated, filtered query against the cached markets catalog.

    Returns the same shape as the polybacktest market browser so the
    UI table component can render either provider with one code path.

    ``channel`` restricts results to markets that have data for the
    given channel (i.e. the ``<channel>_from`` column is non-empty).
    """
    p = markets_catalog_path(exchange)
    if not p.exists():
        return {
            "exchange": exchange,
            "total": 0,
            "limit": int(limit),
            "offset": int(offset),
            "markets": [],
            "catalog_missing": True,
        }

    # Run the pyarrow scan off the event loop — the 660 MB file with
    # filters takes a few hundred ms on a warm disk cache.
    return await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: _list_markets_sync(
            p, exchange=exchange, search=search, status=status,
            channel=channel, limit=int(limit), offset=int(offset),
        ),
    )


def _list_markets_sync(
    catalog_path: Path,
    *,
    exchange: str,
    search: Optional[str],
    status: Optional[str],
    channel: Optional[str],
    limit: int,
    offset: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(str(catalog_path), columns=list(_LIST_COLUMNS))

    # Filters — composed as a boolean mask we can &-combine.
    mask: Optional[pa.Array] = None

    def _and(m: pa.Array) -> None:
        nonlocal mask
        mask = m if mask is None else pc.and_(mask, m)

    if status:
        _and(pc.equal(table["status"], status))
    if channel:
        pair = _CHANNEL_DATE_COLS.get(channel.lower())
        if pair is not None:
            from_col = table[pair[0]]
            # Has data when the "from" column is non-null AND not the empty string.
            non_null = pc.is_valid(from_col)
            non_empty = pc.not_equal(pc.cast(from_col, pa.string()), pa.scalar(""))
            _and(pc.and_(non_null, non_empty))
    if search:
        needle = search.strip().lower()
        if needle:
            # Case-insensitive contains across slug, question, event_title.
            def _hay(col: str) -> pa.Array:
                arr = pc.utf8_lower(pc.cast(table[col], pa.string()))
                return pc.match_substring(arr, needle)

            _and(pc.or_(pc.or_(_hay("slug"), _hay("question")), _hay("event_title")))

    if mask is not None:
        table = table.filter(mask)

    total = int(table.num_rows)
    # Slice for the page window.
    sliced = table.slice(max(0, offset), max(0, limit))

    rows_dict = sliced.to_pydict()
    markets: list[dict[str, Any]] = []
    n = sliced.num_rows
    for i in range(n):
        market_id = _safe_str(rows_dict["market_id"][i])
        slug = _safe_str(rows_dict["slug"][i])
        question = _safe_str(rows_dict["question"][i])
        if not (market_id or slug):
            continue
        markets.append(_serialize_row(rows_dict, i))

    return {
        "exchange": exchange,
        "total": total,
        "limit": limit,
        "offset": offset,
        "markets": markets,
        "catalog_missing": False,
    }


def _serialize_row(cols: dict[str, list[Any]], i: int) -> dict[str, Any]:
    def g(name: str) -> Any:
        return cols.get(name, [None])[i] if i < len(cols.get(name, [])) else None

    channels: dict[str, dict[str, Optional[str]]] = {}
    for ch, (f, t) in _CHANNEL_DATE_COLS.items():
        fv = _safe_str(g(f))
        tv = _safe_str(g(t))
        if fv or tv:
            channels[ch] = {"from_date": fv, "to_date": tv}

    tags_raw = g("tags") or []
    tags = [str(x) for x in tags_raw if x is not None] if isinstance(tags_raw, list) else []

    return {
        "market_id": _safe_str(g("market_id")),
        "slug": _safe_str(g("slug")),
        "event_id": _safe_str(g("event_id")),
        "event_slug": _safe_str(g("event_slug")),
        "event_title": _safe_str(g("event_title")),
        "question": _safe_str(g("question")),
        "category": _safe_str(g("category")),
        "outcomes": [
            {"label": _safe_str(g("outcome_0")), "asset_id": _safe_str(g("asset_id_0"))},
            {"label": _safe_str(g("outcome_1")), "asset_id": _safe_str(g("asset_id_1"))},
        ],
        "status": _safe_str(g("status")),
        "start_date": _us_to_iso(g("start_date_us")),
        "end_date": _us_to_iso(g("end_date_us")),
        "settled_at": _us_to_iso(g("settled_at_us")),
        "tags": tags,
        "channels": channels,
    }


__all__ = [
    "catalog_dir",
    "markets_catalog_path",
    "markets_catalog_stamp_path",
    "catalog_status",
    "CatalogStatus",
    "refresh_markets_catalog",
    "list_markets",
]
