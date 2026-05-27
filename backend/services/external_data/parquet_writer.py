"""Reusable parquet time-series writer — the single write path for ALL
recorded/imported market data.

Postgres is the control plane (configs, jobs, catalog rows, runtime
state).  Parquet is the data plane (every market time-series: books,
deltas, reference prices, ML features).  This writer is how producers
put rows on the data plane WITHOUT touching the orchestrator's Postgres:

  * rows buffer in memory (sync, lock-free append from the producer),
  * a background task flushes them to atomic rolling-window parquet
    files on the canonical provider layout, and
  * the ONLY Postgres touch is a small, infrequent ``ProviderDataset``
    catalog upsert — gated on ``is_db_pressure_active()`` so it defers
    when the orchestrator's DB is under load.

Producers: ``LiveMarketDataIngestor`` (books/deltas), ``crypto_ohlc_recorder``
(reference), REST backfill, provider import.  Each constructs a writer
with the right ``provider`` / ``kind`` / ``schema`` and calls ``append``.

File layout (see ``parquet_schema.parquet_path_for``)::

    {root}/{provider}/{coin}/{window_start__window_end}/{kind}__{series}.parquet

The active window's file is rewritten atomically (``.tmp`` → ``os.replace``)
on each flush; closed windows finalise once then free memory.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from services.external_data.parquet_schema import SCHEMA_VERSION, parquet_path_for
from utils.logger import get_logger

logger = get_logger("parquet_writer")


class ParquetTimeSeriesWriter:
    """Buffer rows and flush to atomic rolling-window parquet files.

    ``schema`` is a pyarrow Schema; ``append`` rows are dicts keyed by the
    schema's column names.  Rows MUST carry ``observed_at_us`` (int micros)
    — it's the sort/window key.
    """

    def __init__(
        self,
        *,
        provider: str,
        kind: str,
        schema: pa.Schema,
        window_seconds: int = 900,
        flush_interval_seconds: float = 10.0,
        catalog_interval_seconds: float = 30.0,
        compression: str = "zstd",
        root: Path | None = None,
    ):
        self._provider = provider
        self._kind = kind
        self._schema = schema
        self._cols = list(schema.names)
        self._window_seconds = max(1, int(window_seconds))
        self._flush_interval = max(0.5, float(flush_interval_seconds))
        self._catalog_interval = max(5.0, float(catalog_interval_seconds))
        self._compression = compression
        self._root = root

        # buffer: (coin, series_id, bucket_start_s) -> list[dict]
        self._buffers: dict[tuple[str, str, int], list[dict]] = {}
        self._dirty: set[tuple[str, str, int]] = set()
        self._flush_task: Optional[asyncio.Task] = None
        self._stopped = False
        self._last_catalog_mono = 0.0
        self._rows_appended = 0
        self._rows_flushed = 0
        self._flush_count = 0
        self._catalog_deferred = 0

    # ── stats ─────────────────────────────────────────────────────────
    @property
    def started(self) -> bool:
        return self._flush_task is not None and not self._flush_task.done()

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self._provider,
            "kind": self._kind,
            "running": self.started,
            "rows_appended": self._rows_appended,
            "rows_flushed": self._rows_flushed,
            "flush_count": self._flush_count,
            "active_buffers": len(self._buffers),
            "catalog_deferred": self._catalog_deferred,
        }

    def _bucket_start(self, ts_s: float) -> int:
        return int(ts_s // self._window_seconds) * self._window_seconds

    # ── producer API (sync, lock-free) ────────────────────────────────
    def append(self, *, coin: str, series_id: str, row: dict) -> None:
        """Buffer one row.  ``row`` must contain ``observed_at_us`` and the
        schema's columns.  Called from the producer (WS callback / loop)."""
        obs_us = row.get("observed_at_us")
        if obs_us is None:
            return
        bucket = self._bucket_start(int(obs_us) / 1_000_000.0)
        key = (str(coin or "_").lower(), str(series_id), bucket)
        buf = self._buffers.get(key)
        if buf is None:
            buf = []
            self._buffers[key] = buf
        buf.append(row)
        self._dirty.add(key)
        self._rows_appended += 1

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self.started:
            return
        self._stopped = False
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush(force_all=True)
        await self._catalog(force=True)

    async def _flush_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
                if (time.monotonic() - self._last_catalog_mono) >= self._catalog_interval:
                    await self._catalog()
                    self._last_catalog_mono = time.monotonic()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("parquet_writer[%s/%s]: flush loop error", self._provider, self._kind)

    async def flush(self, *, force_all: bool = False) -> None:
        current_bucket = self._bucket_start(time.time())
        keys = list(self._buffers.keys()) if force_all else list(self._dirty)
        for key in keys:
            buf = self._buffers.get(key)
            if not buf:
                self._dirty.discard(key)
                continue
            rows = list(buf)  # snapshot on the loop thread (append is loop-only)
            coin, series_id, bucket = key
            try:
                await asyncio.to_thread(self._write_window, coin, series_id, bucket, rows)
                self._rows_flushed += len(rows) - 0  # rewrite semantics: full window each flush
                self._flush_count += 1
            except Exception:
                logger.exception("parquet_writer[%s/%s]: write failed %s", self._provider, self._kind, key)
                continue
            self._dirty.discard(key)
            # Finalise + free memory for closed windows.
            if bucket < current_bucket and not force_all:
                self._buffers.pop(key, None)

    def _write_window(self, coin: str, series_id: str, bucket: int, rows: list[dict]) -> None:
        start = datetime.fromtimestamp(bucket, tz=timezone.utc)
        end = datetime.fromtimestamp(bucket + self._window_seconds, tz=timezone.utc)
        pp = parquet_path_for(
            provider=self._provider, coin=coin, token_id=series_id,
            start=start, end=end, kind=self._kind, root=self._root,
        )
        pp.window_dir.mkdir(parents=True, exist_ok=True)
        # Build columns in schema order; rows sorted by observed_at_us.
        rows_sorted = sorted(rows, key=lambda r: r.get("observed_at_us", 0))
        arrays = {c: [r.get(c) for r in rows_sorted] for c in self._cols}
        table = pa.table(arrays, schema=self._schema)
        table = table.replace_schema_metadata({"schema_version": SCHEMA_VERSION})
        tmp = pp.file_path.with_suffix(".parquet.tmp")
        pq.write_table(table, str(tmp), compression=self._compression)
        os.replace(tmp, pp.file_path)

    async def _catalog(self, *, force: bool = False) -> None:
        """Upsert ProviderDataset catalog rows for written windows.  This is
        the ONLY Postgres touch — deferred when the orchestrator DB is under
        pressure so recording never competes with live trading."""
        if not force:
            try:
                from services.live_pressure import is_db_pressure_active
                if is_db_pressure_active():
                    self._catalog_deferred += 1
                    return
            except Exception:
                pass
        try:
            from services.external_data import parquet_scanner as _scan
            if self._root is not None:
                await _scan.rescan_parquet_root(root=self._root)
            else:
                await _scan.rescan_parquet_root()
        except Exception:
            logger.exception("parquet_writer[%s/%s]: catalog rescan failed", self._provider, self._kind)
