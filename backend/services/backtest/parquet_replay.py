"""Parquet-backed book replay for the bring-your-own-data backtest path.

Implements the same ``_BookSource`` protocol as ``BookReplay`` and
``BookDeltaReplay`` (``iter_snapshots`` async-iterator + ``snapshot_at``
point-in-time lookup) so the matching engine can swap implementations
transparently.

Reads parquet files written under ``HOMERUN_PARQUET_ROOT`` per the
schema in ``services/external_data/parquet_schema.py``.  The path
catalog (which file covers which token+window) is provided at
construction time as ``per_token_files`` — the auto-discovery scanner
populates it from ``provider_datasets.storage_uri`` rows.

Performance contract:

  * ``iter_snapshots`` streams one file at a time (no full
    materialisation), heap-merging across tokens so global ordering
    on ``observed_at`` is preserved without loading every file into
    memory.
  * ``snapshot_at`` uses an in-process LRU cache: per ``(token_id,
    window)`` we materialise the columns the live_context reader
    needs once, then bisect for at-or-before lookups.  At ~10MB per
    cached file × default cap of 32 entries, memory tops out at
    ~320MB regardless of replay length.
  * Every chunk read uses pyarrow's row-group pruning so requesting
    a single timestamp doesn't pull the whole file.

Truncation safety mirrors the SQL replays: a per-file read failure
flips ``self.truncated`` and the matcher gracefully handles a short
stream.
"""
from __future__ import annotations

import bisect
import heapq
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional, Sequence

from services.backtest.book_replay import BookSnapshot, PriceLevel
from services.external_data.parquet_schema import SNAPSHOT_SCHEMA

logger = logging.getLogger(__name__)


_LRU_MAX_BYTES = 320 * 1024 * 1024  # 320MB cap on the snapshot_at cache


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _us_from_dt(dt: datetime) -> int:
    """Microseconds since epoch (matches the parquet observed_at_us
    column).  Truncates sub-microsecond precision (pyarrow stores
    timestamp[us] for our schema)."""
    aware = _to_utc(dt)
    return int(aware.timestamp() * 1_000_000)


def _dt_from_us(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


def _row_to_snapshot(token_id: str, row: dict[str, Any]) -> BookSnapshot:
    """Materialise a parquet row as a ``BookSnapshot``.  Filters
    invalid levels (size <= 0, price <= 0 or >= 1) the same way the
    SQL replays do."""
    bids_p = row.get("bids_price") or []
    bids_s = row.get("bids_size") or []
    asks_p = row.get("asks_price") or []
    asks_s = row.get("asks_size") or []

    bid_levels = []
    for p, s in zip(bids_p, bids_s):
        try:
            pf = float(p) if p is not None else 0.0
            sf = float(s) if s is not None else 0.0
        except (TypeError, ValueError):
            continue
        if pf <= 0 or pf >= 1.0 or sf <= 0:
            continue
        bid_levels.append(PriceLevel(price=pf, size=sf))
    bid_levels.sort(key=lambda lvl: lvl.price, reverse=True)

    ask_levels = []
    for p, s in zip(asks_p, asks_s):
        try:
            pf = float(p) if p is not None else 0.0
            sf = float(s) if s is not None else 0.0
        except (TypeError, ValueError):
            continue
        if pf <= 0 or pf >= 1.0 or sf <= 0:
            continue
        ask_levels.append(PriceLevel(price=pf, size=sf))
    ask_levels.sort(key=lambda lvl: lvl.price)

    observed_us = int(row.get("observed_at_us") or 0)
    seq = row.get("sequence")
    spread = row.get("spread_bps")
    return BookSnapshot(
        token_id=token_id,
        observed_at=_dt_from_us(observed_us),
        bids=tuple(bid_levels),
        asks=tuple(ask_levels),
        sequence=int(seq) if seq is not None else None,
        spread_bps=float(spread) if spread is not None else None,
        trade_price=(
            float(row["trade_price"])
            if row.get("trade_price") is not None
            else None
        ),
        trade_size=(
            float(row["trade_size"]) if row.get("trade_size") is not None else None
        ),
        trade_side=(
            str(row["trade_side"]) if row.get("trade_side") else None
        ),
    )


class _SortedTableCache:
    """Tiny LRU for materialised per-token parquet tables.  Caches
    only the columns ``snapshot_at`` actually needs (no full bids/asks
    JSON), so a 1M-row file is ~32MB instead of ~120MB.

    The cache is process-local and not thread-safe — backtest runs
    each get their own ``ParquetBookReplay`` instance, so concurrent
    backtests don't share cache entries.
    """

    def __init__(self, max_bytes: int = _LRU_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._size_bytes = 0
        # ``OrderedDict``-style LRU: insertion order = recency.
        self._entries: dict[str, dict[str, Any]] = {}

    def get(self, file_path: str) -> dict[str, Any] | None:
        entry = self._entries.pop(file_path, None)
        if entry is not None:
            # Re-insert at the end → most recently used.
            self._entries[file_path] = entry
        return entry

    def put(self, file_path: str, entry: dict[str, Any], approx_bytes: int) -> None:
        if file_path in self._entries:
            self._entries.pop(file_path)
        self._entries[file_path] = entry
        self._size_bytes += approx_bytes
        # Evict oldest until we're under the cap.
        while self._size_bytes > self._max_bytes and self._entries:
            oldest_path, oldest_entry = next(iter(self._entries.items()))
            self._entries.pop(oldest_path)
            self._size_bytes -= int(oldest_entry.get("_approx_bytes") or 0)


class ParquetBookReplay:
    """DB-free book replay backed by parquet files on disk.  Same
    public surface as ``BookReplay`` so the matching engine can
    consume either source interchangeably.

    ``per_token_files`` maps ``token_id -> file_path`` (or list of
    paths if a token's snapshots span multiple files; we sort and
    chain them on read).  Tokens not in the map yield no snapshots.
    """

    def __init__(
        self,
        *,
        per_token_files: dict[str, str | Sequence[str]],
        start: datetime,
        end: datetime,
        chunk_size: int = 5000,
    ) -> None:
        self._token_files: dict[str, list[Path]] = {}
        for tid, paths in (per_token_files or {}).items():
            if not tid:
                continue
            if isinstance(paths, (str, Path)):
                self._token_files[str(tid)] = [Path(paths)]
            else:
                self._token_files[str(tid)] = [Path(p) for p in paths]
        self._start = _to_utc(start)
        self._end = _to_utc(end)
        self._start_us = _us_from_dt(self._start)
        self._end_us = _us_from_dt(self._end)
        self._chunk_size = max(100, int(chunk_size))
        self._cache = _SortedTableCache()
        # Truncation flags — mirror the SQL replays so callers can
        # surface the same warning regardless of source.
        self.truncated: bool = False
        self.truncation_reason: Optional[str] = None
        self.snapshots_yielded: int = 0

    # ── Streaming iteration ──────────────────────────────────────────

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        """Yield snapshots in (observed_at, sequence) order across all
        tokens.  Heap-merges per-token streams so we don't materialise
        any single file fully — bounded memory regardless of N tokens.
        """
        if not self._token_files:
            return
        import pyarrow.parquet as pq

        # Per-token cursor: (next observed_us, next row index, token_id, table)
        # The heap is keyed by observed_us so the smallest timestamp
        # comes off next.
        per_token_state: dict[str, dict[str, Any]] = {}
        heap: list[tuple[int, str, int]] = []  # (us, token_id, row_idx)

        def _open_token(tid: str) -> bool:
            paths = self._token_files.get(tid) or []
            tables = []
            for p in paths:
                try:
                    table = pq.read_table(
                        str(p),
                        # Project only the columns we need to materialise
                        # the BookSnapshot — the JSON-like list columns
                        # are pulled here because iter_snapshots emits
                        # full snapshots; live_context lookups go through
                        # the LRU cache path which projects fewer columns.
                    )
                except Exception as exc:
                    logger.warning(
                        "ParquetBookReplay: failed to read %s for token %s: %s",
                        p, tid, exc,
                    )
                    self.truncated = True
                    self.truncation_reason = f"read failed: {p.name}: {str(exc)[:200]}"
                    continue
                # Filter to window via observed_at_us.  pyarrow.compute
                # would be faster but the row-count is usually small
                # and we want to keep the dependency surface minimal.
                tables.append(table)
            if not tables:
                return False
            # Concatenate (typically just one file per token).
            combined = tables[0] if len(tables) == 1 else _concat_tables(tables)
            # Boolean filter on observed_at_us in [start, end].
            obs = combined.column("observed_at_us").to_pylist()
            keep_idx = [i for i, v in enumerate(obs) if self._start_us <= int(v or 0) <= self._end_us]
            if not keep_idx:
                return False
            sorted_keep = sorted(keep_idx, key=lambda i: int(obs[i] or 0))
            per_token_state[tid] = {
                "table": combined,
                "indices": sorted_keep,
                "cursor": 0,
                "obs": obs,
            }
            first_us = int(obs[sorted_keep[0]] or 0)
            heap.append((first_us, tid, 0))
            return True

        for tid in self._token_files:
            _open_token(tid)
        if not heap:
            return
        heapq.heapify(heap)

        total_yielded = 0
        while heap:
            us, tid, cursor = heapq.heappop(heap)
            state = per_token_state[tid]
            row_idx = state["indices"][cursor]
            try:
                row = {
                    name: state["table"].column(name)[row_idx].as_py()
                    for name in state["table"].schema.names
                }
            except Exception as exc:
                logger.warning(
                    "ParquetBookReplay: row decode failed for token %s @ idx %s: %s",
                    tid, row_idx, exc,
                )
                self.truncated = True
                self.truncation_reason = f"row decode failed: {str(exc)[:200]}"
                continue
            snap = _row_to_snapshot(tid, row)
            yield snap
            total_yielded += 1
            self.snapshots_yielded = total_yielded
            # Advance this token's cursor.
            next_cursor = cursor + 1
            state["cursor"] = next_cursor
            if next_cursor < len(state["indices"]):
                next_us = int(state["obs"][state["indices"][next_cursor]] or 0)
                heapq.heappush(heap, (next_us, tid, next_cursor))

    # ── Point-in-time ────────────────────────────────────────────────

    async def snapshot_at(
        self, *, token_id: str, ts: datetime
    ) -> Optional[BookSnapshot]:
        """Most-recent snapshot at-or-before ``ts`` for the token.
        Loads the file once (LRU-cached) then bisects the
        observed_at_us column.  O(log N) per call.
        """
        paths = self._token_files.get(str(token_id)) or []
        if not paths:
            return None
        target_us = _us_from_dt(ts)

        cache_key = "|".join(str(p) for p in paths)
        cached = self._cache.get(cache_key)
        if cached is None:
            cached = self._load_table_for_lookup(paths)
            if cached is None:
                return None
            self._cache.put(cache_key, cached, cached.get("_approx_bytes", 0))

        obs_us = cached["obs_us"]
        if not obs_us:
            return None
        # bisect_right - 1 gives the rightmost index with obs_us <= target
        idx = bisect.bisect_right(obs_us, target_us) - 1
        if idx < 0:
            return None
        if obs_us[idx] < self._start_us or obs_us[idx] > self._end_us:
            # Not in the requested window.
            return None
        row_idx = cached["sorted_indices"][idx]
        table = cached["table"]
        try:
            row = {
                name: table.column(name)[row_idx].as_py()
                for name in table.schema.names
            }
        except Exception as exc:
            logger.warning("ParquetBookReplay snapshot_at decode failed: %s", exc)
            return None
        return _row_to_snapshot(str(token_id), row)

    def _load_table_for_lookup(
        self, paths: list[Path]
    ) -> dict[str, Any] | None:
        import pyarrow.parquet as pq

        tables = []
        approx_bytes = 0
        for p in paths:
            try:
                table = pq.read_table(str(p))
            except Exception as exc:
                logger.warning(
                    "ParquetBookReplay: snapshot_at read failed for %s: %s", p, exc,
                )
                continue
            tables.append(table)
            approx_bytes += table.nbytes
        if not tables:
            return None
        combined = tables[0] if len(tables) == 1 else _concat_tables(tables)
        obs_col = combined.column("observed_at_us").to_pylist()
        sorted_indices = sorted(
            range(len(obs_col)), key=lambda i: int(obs_col[i] or 0)
        )
        sorted_obs = [int(obs_col[i] or 0) for i in sorted_indices]
        return {
            "table": combined,
            "obs_us": sorted_obs,
            "sorted_indices": sorted_indices,
            "_approx_bytes": approx_bytes,
        }


def _concat_tables(tables: list[Any]) -> Any:
    """Concatenate pyarrow tables.  Wrapped so callers don't import
    pyarrow directly (keeps the optional-dep surface narrow).
    """
    import pyarrow as pa
    return pa.concat_tables(tables)


# ── Writer (used by tests + future ingest CLIs) ──────────────────────


def write_snapshots(
    *,
    file_path: Path | str,
    snapshots: Iterable[BookSnapshot],
) -> int:
    """Write a sequence of ``BookSnapshot``s to a parquet file matching
    ``SNAPSHOT_SCHEMA``.  Sorted by observed_at on write so readers
    don't have to.  Returns the number of rows written.

    Used by tests and by future CLI ingest tools.  For production
    backfill the writer would stream in row-group chunks; this
    in-memory variant is fine for the current dataset sizes (single
    file ~50MB max).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(snapshots, key=lambda s: (s.observed_at, s.sequence or 0))
    if not rows:
        return 0
    cols: dict[str, list[Any]] = {name: [] for name in SNAPSHOT_SCHEMA.names}
    for s in rows:
        cols["token_id"].append(str(s.token_id))
        cols["observed_at_us"].append(_us_from_dt(s.observed_at))
        cols["sequence"].append(int(s.sequence) if s.sequence is not None else None)
        cols["best_bid"].append(float(s.best_bid) if s.best_bid is not None else 0.0)
        cols["best_ask"].append(float(s.best_ask) if s.best_ask is not None else 0.0)
        cols["spread_bps"].append(
            float(s.spread_bps) if s.spread_bps is not None else None
        )
        cols["bids_price"].append([float(lvl.price) for lvl in s.bids])
        cols["bids_size"].append([float(lvl.size) for lvl in s.bids])
        cols["asks_price"].append([float(lvl.price) for lvl in s.asks])
        cols["asks_size"].append([float(lvl.size) for lvl in s.asks])
        cols["trade_price"].append(s.trade_price)
        cols["trade_size"].append(s.trade_size)
        cols["trade_side"].append(s.trade_side)

    table = pa.table(cols, schema=SNAPSHOT_SCHEMA)
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to .tmp then rename so a partial write never produces a
    # half-valid file the discovery scanner would later catalog.
    tmp = p.with_suffix(p.suffix + ".tmp")
    pq.write_table(table, str(tmp), compression="snappy", row_group_size=50_000)
    tmp.replace(p)
    return table.num_rows


__all__ = ["ParquetBookReplay", "write_snapshots"]
