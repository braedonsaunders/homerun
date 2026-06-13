"""Columnar parquet sink for the live market-data ingestor.

Purpose: get the L2 book/delta recording OFF Postgres so it imposes ZERO
DB write pressure on the trading process's hot path.  The ingestor's hot
path (``record_book`` → sync enqueue) is untouched; only the background
flush task's PERSISTENCE target changes from ``session.add_all`` to this
sink, which writes the canonical ``snapshots__/deltas__`` columnar layout
that the unified ``MarketDataView`` reads natively (no new backtest reader).

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
    parts_dir_for,
    parquet_path_for,
    parquet_root,
)
from services.marketdata.writer import write_canonical_table
from utils.logger import get_logger

logger = get_logger("book_parquet_sink")

PROVIDER = "live_ingestor"
_WINDOW_SECONDS = 900
_FLUSH_INTERVAL_SECONDS = 2.0
_CATALOG_INTERVAL_SECONDS = 60.0
_COMPACT_INTERVAL_SECONDS = 30.0
_PRUNE_INTERVAL_SECONDS = 300.0
_MAX_BUFFERED_ROWS = 200_000  # drop-oldest backstop
_ROW_GROUP_SIZE = 64_000
_DEFAULT_RETENTION_DAYS = 7
_DEFAULT_MAX_BYTES = 40 * 1024 * 1024 * 1024  # 40 GB — the denser REST-baseline
# recording (every active market every ~10 min) needs a larger budget to retain
# a full backtest window; operator-tunable live via the recording config.


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
        # v2 incremental layout: buffer keyed by (kind, bucket) — token_id is
        # a COLUMN, not a key.  Each flush drains a key's rows into an
        # immutable part file ({window}/_parts/{kind}__part-NNNNNN.parquet)
        # and frees them, so sink memory is bounded by one flush interval of
        # inflow (the old (kind, token, bucket) design rewrote each token's
        # FULL window every 2s and leaked every closed window's rows — the
        # 2.7GB recording plane).  The compactor merges parts -> one bundle
        # per (kind, window) when the window closes.
        self._buf: dict[tuple[str, int], list[dict]] = {}
        self._part_seq: dict[tuple[str, int], int] = {}
        self._buffered_rows = 0
        self._dropped = 0
        self._written = 0
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._last_catalog = 0.0
        self._last_prune = 0.0
        self._last_compact = 0.0
        self._compact_inflight = False
        # Window dirs whose BUNDLES changed since the last catalog pass
        # (compaction output) — registered incrementally.  Part files are
        # never registered: they're invisible to readers by design.
        self._pending_catalog: set[Path] = set()

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
            key = (pkind, bucket)
            self._buf.setdefault(key, []).append(row)
            self._buffered_rows += 1
        if self._buffered_rows > _MAX_BUFFERED_ROWS:
            self._drop_oldest()

    def _drop_oldest(self) -> None:
        # Drop the oldest bucket entirely (back-pressure backstop).
        if not self._buf:
            return
        oldest = min(self._buf.keys(), key=lambda k: k[1])
        n = len(self._buf.pop(oldest, []))
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
                if now - self._last_compact >= _COMPACT_INTERVAL_SECONDS:
                    self._maybe_start_compaction()
                    self._last_compact = now
                if now - self._last_prune >= _PRUNE_INTERVAL_SECONDS:
                    await asyncio.to_thread(self._prune)
                    self._last_prune = now
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("book_parquet_sink: loop error")

    async def _flush(self, *, force: bool = False) -> None:
        # Free-DISK guard: if total free disk has dropped below the configured
        # headroom, never write more — that's exactly how the recorder zeroed
        # the drive and crashed the host.  Drop the in-memory batch (back-
        # pressure) and force-prune oldest windows to recover space.  This is
        # independent of the size caps (which only bound the app's OWN
        # footprint, not the whole disk).  Fail-open on any guard error.
        try:
            from services import disk_guard
            blocked, _free_now, _min_free = await disk_guard.evaluate(self._root or parquet_root())
        except Exception:
            blocked, _min_free = False, 0.0
        if blocked:
            self._dropped += self._buffered_rows
            self._buf.clear()
            self._part_seq.clear()
            self._buffered_rows = 0
            await asyncio.to_thread(self._prune, _min_free)
            return
        # Incremental drain: write each key's buffered rows as ONE immutable
        # part file and free them immediately.  No key is ever rewritten, no
        # rows outlive their flush — sink memory is bounded by one interval of
        # inflow, and the per-flush write cost is O(new rows), not O(window).
        for key in list(self._buf.keys()):
            rows = self._buf.get(key)
            if not rows:
                self._buf.pop(key, None)
                self._part_seq.pop(key, None)
                continue
            # Atomic drain on the event loop: no await between read and reset,
            # so rows appended during the threaded write land in the fresh
            # list and ride the next part.
            drained = rows
            self._buf[key] = []
            seq = self._part_seq.get(key, 0)
            try:
                _dest, next_seq = await asyncio.to_thread(self._write_part, key, seq, drained)
                self._part_seq[key] = next_seq
                self._written += len(drained)
                self._buffered_rows -= len(drained)
            except Exception:
                # Put the rows back at the FRONT so ordering is preserved for
                # the retry; the part file write is atomic so a failure left
                # no partial artifact.
                self._buf[key] = drained + self._buf[key]
                logger.exception("book_parquet_sink: part write failed %s", key)
                continue
            # Drop fully-drained closed-window keys; their parts await the
            # compactor.
            current_bucket = int(time.time() // _WINDOW_SECONDS) * _WINDOW_SECONDS
            if key[1] < current_bucket and not self._buf.get(key):
                self._buf.pop(key, None)
                self._part_seq.pop(key, None)
        if self._buffered_rows < 0:
            self._buffered_rows = 0

    def _write_part(self, key: tuple[str, int], seq: int, rows: list[dict]) -> tuple[Path, int]:
        pkind, bucket = key
        schema = SNAPSHOT_SCHEMA if pkind == "snapshots" else DELTA_SCHEMA
        start = datetime.fromtimestamp(bucket, tz=timezone.utc)
        end = datetime.fromtimestamp(bucket + _WINDOW_SECONDS, tz=timezone.utc)
        pp = parquet_path_for(provider=PROVIDER, coin="_", token_id="_",
                              start=start, end=end, kind=pkind, root=self._root)
        pdir = parts_dir_for(pp.window_dir)
        if seq == 0:
            seq = self._next_part_sequence(pdir, pkind)
        rows_sorted = sorted(rows, key=lambda r: (str(r.get("token_id") or ""), int(r["observed_at_us"])))
        cols = {name: [r.get(name) for r in rows_sorted] for name in schema.names}
        table = pa.table(cols, schema=schema)
        dest = pdir / f"{pkind}__part-{seq:06d}.parquet"
        write_canonical_table(
            table,
            dest_path=dest,
            kind=pkind,
            provider=PROVIDER,
            row_group_size=_ROW_GROUP_SIZE,
        )
        return dest, seq + 1

    @staticmethod
    def _next_part_sequence(parts_dir: Path, kind: str) -> int:
        try:
            entries = list(parts_dir.iterdir())
        except OSError:
            return 0
        prefix = f"{kind}__part-"
        next_seq = 0
        for fp in entries:
            name = fp.name
            if not name.startswith(prefix) or not name.endswith(".parquet"):
                continue
            raw = name[len(prefix):-8]
            if not raw.isdigit():
                continue
            next_seq = max(next_seq, int(raw) + 1)
        return next_seq

    def _maybe_start_compaction(self) -> None:
        if self._compact_inflight:
            return
        self._compact_inflight = True
        asyncio.create_task(self._compact_closed_windows(), name="book-parquet-compactor")

    async def _compact_closed_windows(self) -> None:
        try:
            from services.external_data.parquet_compactor import compact_closed_windows

            changed = await asyncio.to_thread(
                compact_closed_windows,
                self._root or parquet_root(),
                provider=PROVIDER,
                max_windows=1,
            )
            self._pending_catalog.update(changed)
        except Exception:
            logger.exception("book_parquet_sink: compaction pass failed")
        finally:
            self._compact_inflight = False

    async def _catalog(self) -> None:
        try:
            from services.live_pressure import is_db_pressure_active
            if is_db_pressure_active():
                return
            if not self._pending_catalog:
                return
            # Register ONLY the windows written since the last pass — never a
            # full-tree rescan (that re-validated + re-UPSERTed every historical
            # window every cycle, making a parquet-only recorder the busiest
            # Postgres writer).  Snapshot + remove exactly those, so writes that
            # land during the await stay queued and a failure leaves them to retry.
            pending = list(self._pending_catalog)
            from services.external_data.parquet_scanner import register_window_dirs
            await register_window_dirs(pending)
            self._pending_catalog.difference_update(pending)
        except Exception:
            logger.debug("book_parquet_sink: incremental catalog skipped", exc_info=True)

    def _prune(self, emergency_target_free_gb: float | None = None) -> None:
        """Bound the on-disk footprint: drop window dirs older than
        retention_days, then trim oldest until under max_bytes.

        When ``emergency_target_free_gb`` is set (the disk guard tripped), also
        keep shedding the oldest non-protected windows until total FREE disk
        recovers above that headroom — the size cap alone can't help when the
        drive is full of unrelated data.

        Both are read LIVE from the recording config (``book_retention_days`` /
        ``book_max_bytes``) each pass so an operator can size the disk budget for
        the denser REST-baseline recording from the UI without a restart; they
        fall back to the construction-time defaults when unset."""
        retention_days = self._retention_days
        max_bytes = self._max_bytes
        try:
            from services.recording_control import get_recorder_config_cached

            _cfg = get_recorder_config_cached()
            _rd = _cfg.get("book_retention_days")
            _mb = _cfg.get("book_max_bytes")
            if isinstance(_rd, (int, float)) and _rd > 0:
                retention_days = int(_rd)
            if isinstance(_mb, (int, float)) and _mb > 0:
                max_bytes = int(_mb)
        except Exception:
            pass
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
        # Reproducibility guard: never delete a window dir a running backtest
        # has pinned (cross-process via services.marketdata.pins).
        try:
            from services.marketdata.pins import active_pinned_paths, is_path_pinned
            _pinned = active_pinned_paths()
        except Exception:
            _pinned, is_path_pinned = set(), None  # type: ignore[assignment]

        def _protected(path: Path) -> bool:
            if not _pinned or is_path_pinned is None:
                return False
            return is_path_pinned(path, _pinned)

        now = datetime.now(timezone.utc)
        kept: list[tuple[datetime, Path, int]] = []
        for end, d, sz in dirs:
            if (now - end).total_seconds() > retention_days * 86400 and not _protected(d):
                shutil.rmtree(d, ignore_errors=True)
            else:
                kept.append((end, d, sz))
        total = sum(s for _, _, s in kept)
        if total > max_bytes:
            kept.sort(key=lambda x: x[0])  # oldest first
            for end, d, sz in kept:
                if total <= max_bytes:
                    break
                if _protected(d):
                    continue
                shutil.rmtree(d, ignore_errors=True)
                total -= sz
        # Emergency shed (disk guard tripped): the size cap alone isn't enough
        # because the DISK is low from other data too.  Keep deleting oldest
        # non-protected windows until total free disk recovers above the guard
        # threshold (or nothing prunable remains — protected recordings stay).
        if emergency_target_free_gb and emergency_target_free_gb > 0:
            try:
                free_gb = shutil.disk_usage(str(base)).free / (1024.0 ** 3)
            except Exception:
                free_gb = float(emergency_target_free_gb)
            if free_gb < emergency_target_free_gb:
                for _end, d, sz in sorted(kept, key=lambda x: x[0]):
                    if free_gb >= emergency_target_free_gb:
                        break
                    if not d.exists() or _protected(d):
                        continue
                    shutil.rmtree(d, ignore_errors=True)
                    try:
                        free_gb = shutil.disk_usage(str(base)).free / (1024.0 ** 3)
                    except Exception:
                        free_gb += sz / (1024.0 ** 3)
                logger.warning(
                    "book_parquet_sink: emergency prune for disk headroom "
                    "(target %.1fGB free, now %.1fGB)",
                    float(emergency_target_free_gb), free_gb,
                )
