"""Columnar parquet sink for the live market-data ingestor.

Purpose: get the L2 book/delta recording OFF Postgres so it imposes ZERO
DB write pressure on the trading process's hot path.  The ingestor's hot
path (``record_book`` → sync enqueue) is untouched; only the background
flush task's PERSISTENCE target changes from ``session.add_all`` to this
sink, which writes the canonical ``snapshots__/deltas__`` columnar layout
that :class:`ParquetBookReplay` reads natively (no new backtest reader).

Design (mirrors the ingestor's own discipline):
  * Writes happen on a dedicated background flush loop, and the parquet
    encode/IO runs in ``asyncio.to_thread`` so it never blocks the
    worker's event loop (and thus never the trader's decision loop).
  * Rolling 15-min windows, one file per (token, window), atomic
    replace (``.tmp`` → ``os.replace``).
  * Self-bounded: prunes window dirs older than ``retention_days`` and
    trims oldest when total exceeds ``max_bytes`` — so it can NEVER fill
    the disk (the failure mode that crashed the host before).
  * Bounded in-memory buffer with drop-oldest on overflow — same
    back-pressure discipline as the SQL path.

Layout (see ``parquet_schema.parquet_path_for``)::

    {root}/live_ingestor/_/{window}/snapshots__{token}.parquet
    {root}/live_ingestor/_/{window}/deltas__{token}.parquet
"""
from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa

from services.external_data.parquet_schema import (
    DELTA_SCHEMA,
    SNAPSHOT_SCHEMA,
    parquet_path_for,
    parquet_root,
)
from services.marketdata.writer import write_canonical_table
from utils.logger import get_logger

logger = get_logger("book_parquet_sink")

PROVIDER = "live_ingestor"
_WINDOW_SECONDS = 900
_FLUSH_INTERVAL_SECONDS = 2.0
_CATALOG_INTERVAL_SECONDS = 30.0
_PRUNE_INTERVAL_SECONDS = 300.0
_MAX_BUFFERED_ROWS = 200_000  # drop-oldest backstop
_DEFAULT_RETENTION_DAYS = 7
_DEFAULT_MAX_BYTES = 6 * 1024 * 1024 * 1024  # 6 GB


# ── In-memory row carriers ────────────────────────────────────────────
#
# The live ingestor builds these lightweight rows and hands them to the
# sink, which encodes them to parquet.  They replace the old practice of
# constructing throwaway ``MarketMicrostructureSnapshot`` / ``BookDeltaEvent``
# ORM objects purely as attribute bags — book data never touches SQL now.
# Field names match the canonical SNAPSHOT_SCHEMA / DELTA_SCHEMA columns the
# converters below read; all default so callers set only what they have.


@dataclass
class BookSnapshotRow:
    token_id: str = ""
    observed_at: Optional[datetime] = None
    snapshot_type: str = "book"  # 'book' | 'trade'
    provider: str = "polymarket"
    sequence: Optional[int] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread_bps: Optional[float] = None
    bids_json: Optional[list] = None
    asks_json: Optional[list] = None
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    trade_side: Optional[str] = None
    exchange_ts_ms: Optional[int] = None
    payload_json: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    id: Optional[str] = None


@dataclass
class BookDeltaRow:
    token_id: str = ""
    observed_at: Optional[datetime] = None
    provider: str = "polymarket"
    sequence: Optional[int] = None
    event_type: Optional[str] = None  # 'trade' | 'cancel'
    side: Optional[str] = None
    price: Optional[float] = None
    trade_size: Optional[float] = None
    cancel_size: Optional[float] = None
    queue_depth_before: Optional[float] = None
    queue_depth_after: Optional[float] = None
    spread_bps_at_event: Optional[float] = None
    exchange_ts_ms: Optional[int] = None
    payload_json: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    id: Optional[str] = None


def _levels(side_json: Any) -> tuple[list[float], list[float]]:
    prices: list[float] = []
    sizes: list[float] = []
    if isinstance(side_json, list):
        for lvl in side_json:
            if isinstance(lvl, dict):
                try:
                    prices.append(float(lvl.get("price")))
                    sizes.append(float(lvl.get("size")))
                except (TypeError, ValueError):
                    continue
    return prices, sizes


def _snapshot_row(r: Any) -> Optional[dict]:
    observed_at = getattr(r, "observed_at", None)
    tok = str(getattr(r, "token_id", "") or "")
    if not tok or observed_at is None:
        return None
    bp, bs = _levels(getattr(r, "bids_json", None))
    ap, as_ = _levels(getattr(r, "asks_json", None))
    return {
        "token_id": tok,
        "observed_at_us": int(observed_at.timestamp() * 1_000_000),
        "sequence": getattr(r, "sequence", None),
        "best_bid": getattr(r, "best_bid", None),
        "best_ask": getattr(r, "best_ask", None),
        "spread_bps": getattr(r, "spread_bps", None),
        "bids_price": bp, "bids_size": bs,
        "asks_price": ap, "asks_size": as_,
        "trade_price": getattr(r, "trade_price", None),
        "trade_size": getattr(r, "trade_size", None),
        "trade_side": getattr(r, "trade_side", None),
    }


def _delta_row(r: Any) -> Optional[dict]:
    observed_at = getattr(r, "observed_at", None)
    tok = str(getattr(r, "token_id", "") or "")
    if not tok or observed_at is None:
        return None
    return {
        "token_id": tok,
        "observed_at_us": int(observed_at.timestamp() * 1_000_000),
        "sequence": getattr(r, "sequence", None),
        "event_type": getattr(r, "event_type", None),
        "side": getattr(r, "side", None),
        "price": getattr(r, "price", None),
        "trade_size": getattr(r, "trade_size", None),
        "cancel_size": getattr(r, "cancel_size", None),
        "queue_depth_before": getattr(r, "queue_depth_before", None),
        "queue_depth_after": getattr(r, "queue_depth_after", None),
        "spread_bps_at_event": getattr(r, "spread_bps_at_event", None),
    }


class BookParquetSink:
    def __init__(self, *, retention_days: int = _DEFAULT_RETENTION_DAYS,
                 max_bytes: int = _DEFAULT_MAX_BYTES, root: Path | None = None):
        self._retention_days = retention_days
        self._max_bytes = max_bytes
        self._root = root
        # buffer: (kind, token, bucket) -> list[row dict]
        self._buf: dict[tuple[str, str, int], list[dict]] = {}
        self._dirty: set[tuple[str, str, int]] = set()
        self._buffered_rows = 0
        self._dropped = 0
        self._written = 0
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._last_catalog = 0.0
        self._last_prune = 0.0

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    def stats(self) -> dict[str, Any]:
        return {"running": self.started, "buffered_rows": self._buffered_rows,
                "rows_written": self._written, "rows_dropped": self._dropped,
                "active_buffers": len(self._buf)}

    # ── ingest (called from the ingestor's background flush task) ──────
    def write(self, rows: list[Any], *, kind: str) -> None:
        """Buffer a flushed batch (sync, fast).  kind: 'snapshot' | 'delta'.

        snapshot rows go to the ``snapshots`` parquet kind; delta rows to
        ``deltas``.  Drop-oldest if the buffer is saturated."""
        conv = _snapshot_row if kind != "delta" else _delta_row
        pkind = "deltas" if kind == "delta" else "snapshots"
        for r in rows:
            row = conv(r)
            if row is None:
                continue
            bucket = int(row["observed_at_us"] / 1_000_000 // _WINDOW_SECONDS) * _WINDOW_SECONDS
            key = (pkind, row["token_id"], bucket)
            self._buf.setdefault(key, []).append(row)
            self._dirty.add(key)
            self._buffered_rows += 1
        if self._buffered_rows > _MAX_BUFFERED_ROWS:
            self._drop_oldest()

    def _drop_oldest(self) -> None:
        # Drop the oldest bucket entirely (back-pressure backstop).
        if not self._buf:
            return
        oldest = min(self._buf.keys(), key=lambda k: k[2])
        n = len(self._buf.pop(oldest, []))
        self._dirty.discard(oldest)
        self._buffered_rows -= n
        self._dropped += n

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self.started:
            return
        self._stopped = False
        self._task = asyncio.create_task(self._loop(), name="book-parquet-sink")

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._flush(force=True)

    async def _loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
                await self._flush()
                now = time.monotonic()
                if now - self._last_catalog >= _CATALOG_INTERVAL_SECONDS:
                    await self._catalog()
                    self._last_catalog = now
                if now - self._last_prune >= _PRUNE_INTERVAL_SECONDS:
                    await asyncio.to_thread(self._prune)
                    self._last_prune = now
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("book_parquet_sink: loop error")

    async def _flush(self, *, force: bool = False) -> None:
        current_bucket = int(time.time() // _WINDOW_SECONDS) * _WINDOW_SECONDS
        keys = list(self._buf.keys()) if force else list(self._dirty)
        for key in keys:
            rows = self._buf.get(key)
            if not rows:
                self._dirty.discard(key)
                continue
            snapshot = list(rows)
            try:
                await asyncio.to_thread(self._write_file, key, snapshot)
                self._written += len(snapshot)
            except Exception:
                logger.exception("book_parquet_sink: write failed %s", key)
                continue
            self._dirty.discard(key)
            # Free closed windows from memory once written.
            if key[2] < current_bucket and not force:
                self._buffered_rows -= len(self._buf.get(key, []))
                self._buf.pop(key, None)

    def _write_file(self, key: tuple[str, str, int], rows: list[dict]) -> None:
        pkind, token, bucket = key
        schema = SNAPSHOT_SCHEMA if pkind == "snapshots" else DELTA_SCHEMA
        start = datetime.fromtimestamp(bucket, tz=timezone.utc)
        end = datetime.fromtimestamp(bucket + _WINDOW_SECONDS, tz=timezone.utc)
        pp = parquet_path_for(provider=PROVIDER, coin="_", token_id=token,
                              start=start, end=end, kind=pkind, root=self._root)
        rows_sorted = sorted(rows, key=lambda r: r["observed_at_us"])
        cols = {name: [r.get(name) for r in rows_sorted] for name in schema.names}
        table = pa.table(cols, schema=schema)
        # Single canonical writer: schema-validates + lineage-stamps + atomic.
        write_canonical_table(table, dest_path=pp.file_path, kind=pkind, provider=PROVIDER)

    async def _catalog(self) -> None:
        try:
            from services.live_pressure import is_db_pressure_active
            if is_db_pressure_active():
                return
            from services.external_data.parquet_scanner import rescan_parquet_root
            await (rescan_parquet_root(root=self._root) if self._root else rescan_parquet_root())
        except Exception:
            logger.debug("book_parquet_sink: catalog rescan skipped", exc_info=True)

    def _prune(self) -> None:
        """Bound the on-disk footprint: drop window dirs older than
        retention_days, then trim oldest until under max_bytes."""
        base = (self._root or parquet_root()) / PROVIDER / "_"
        if not base.exists():
            return
        import re as _re
        win_re = _re.compile(r"^(\d{8}T\d{6})__(\d{8}T\d{6})$")
        dirs: list[tuple[datetime, Path, int]] = []
        for d in base.iterdir():
            if not d.is_dir():
                continue
            m = win_re.match(d.name)
            if not m:
                continue
            try:
                end = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            sz = sum(f.stat().st_size for f in d.rglob("*.parquet"))
            dirs.append((end, d, sz))
        now = datetime.now(timezone.utc)
        kept: list[tuple[datetime, Path, int]] = []
        for end, d, sz in dirs:
            if (now - end).total_seconds() > self._retention_days * 86400:
                shutil.rmtree(d, ignore_errors=True)
            else:
                kept.append((end, d, sz))
        total = sum(s for _, _, s in kept)
        if total > self._max_bytes:
            kept.sort(key=lambda x: x[0])  # oldest first
            for end, d, sz in kept:
                if total <= self._max_bytes:
                    break
                shutil.rmtree(d, ignore_errors=True)
                total -= sz
