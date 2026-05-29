"""Content-hashed dataset snapshot manifest for reproducible backtests.

A backtest today reads "whatever parquet is on disk now" — and the live sink
+ bus pruners delete data on their own schedules, so two runs of the same
window days apart can read different bytes. That makes results irreproducible
and lets pruning silently change history mid-analysis.

A :class:`DatasetSnapshot` pins the EXACT set of files a run depends on:
each file's path, byte size, mtime, row count, and token/time span, plus a
single ``content_hash`` over that set. Phase 7 wires this in two ways:

  * persist the snapshot (and its hash) on the ``BacktestRun`` so a re-run can
    detect drift and so identical (snapshot, strategy, config, seed) yields an
    identical result hash;
  * teach the pruners to refuse deleting any file referenced by a live pin.

This module is pure (no DB, no engine wiring) so it can be unit-tested and
reused by both the access layer and the reproducibility layer.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from services.marketdata.schema import BOOK_SCHEMA_VERSION


@dataclass(frozen=True)
class SnapshotEntry:
    """One pinned parquet file. ``size_bytes``/``mtime_us`` make the content
    hash sensitive to any rewrite; ``rows`` + span aid drift diagnostics."""

    path: str
    size_bytes: int
    mtime_us: int
    rows: int = 0
    token_ids: tuple[str, ...] = ()
    start_us: Optional[int] = None
    end_us: Optional[int] = None

    def fingerprint(self) -> str:
        """Stable per-file fingerprint string fed into the content hash."""
        # Normalize path separators so the hash is stable across OSes.
        norm = self.path.replace("\\", "/")
        return f"{norm}|{self.size_bytes}|{self.mtime_us}|{self.rows}"


def compute_content_hash(entries: Iterable[SnapshotEntry]) -> str:
    """Deterministic sha256 over the sorted per-file fingerprints.

    Sorting by path makes the hash independent of discovery order, so the same
    set of files always hashes identically.
    """
    fps = sorted(e.fingerprint() for e in entries)
    h = hashlib.sha256()
    for fp in fps:
        h.update(fp.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class DatasetSnapshot:
    """An immutable, content-hashed manifest of the files a run pinned."""

    entries: tuple[SnapshotEntry, ...]
    schema_version: str
    created_at_us: int
    content_hash: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(e.rows for e in self.entries)

    @property
    def token_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for e in self.entries:
            for t in e.token_ids:
                seen.setdefault(t, None)
        return tuple(seen.keys())

    @property
    def span_us(self) -> tuple[Optional[int], Optional[int]]:
        starts = [e.start_us for e in self.entries if e.start_us is not None]
        ends = [e.end_us for e in self.entries if e.end_us is not None]
        return (min(starts) if starts else None, max(ends) if ends else None)

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(e.path.replace("\\", "/") for e in self.entries)

    def contains_path(self, path: str | Path) -> bool:
        """True if ``path`` is pinned by this snapshot (pruner guard, Phase 7)."""
        return str(path).replace("\\", "/") in self.paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_us": self.created_at_us,
            "content_hash": self.content_hash,
            "entries": [
                {
                    "path": e.path,
                    "size_bytes": e.size_bytes,
                    "mtime_us": e.mtime_us,
                    "rows": e.rows,
                    "token_ids": list(e.token_ids),
                    "start_us": e.start_us,
                    "end_us": e.end_us,
                }
                for e in self.entries
            ],
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetSnapshot":
        entries = tuple(
            SnapshotEntry(
                path=e["path"],
                size_bytes=int(e.get("size_bytes", 0)),
                mtime_us=int(e.get("mtime_us", 0)),
                rows=int(e.get("rows", 0)),
                token_ids=tuple(e.get("token_ids", ()) or ()),
                start_us=e.get("start_us"),
                end_us=e.get("end_us"),
            )
            for e in d.get("entries", [])
        )
        return cls(
            entries=entries,
            schema_version=str(d.get("schema_version", BOOK_SCHEMA_VERSION)),
            created_at_us=int(d.get("created_at_us", 0)),
            content_hash=str(d.get("content_hash", "")),
            extra=dict(d.get("extra", {}) or {}),
        )


def _stat_entry(
    path: str | Path,
    *,
    rows: int = 0,
    token_ids: Iterable[str] = (),
    start_us: Optional[int] = None,
    end_us: Optional[int] = None,
) -> SnapshotEntry:
    p = Path(path)
    st = p.stat()
    return SnapshotEntry(
        path=str(p),
        size_bytes=int(st.st_size),
        mtime_us=int(st.st_mtime * 1_000_000),
        rows=int(rows),
        token_ids=tuple(token_ids),
        start_us=start_us,
        end_us=end_us,
    )


def build_snapshot(
    files: Iterable[str | Path | SnapshotEntry],
    *,
    schema_version: str = BOOK_SCHEMA_VERSION,
    created_at_us: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> DatasetSnapshot:
    """Build a :class:`DatasetSnapshot` from paths (``stat``-ed automatically)
    or pre-built :class:`SnapshotEntry` objects (when row/span metadata is
    already known from the coverage layer). Missing files are skipped.
    """
    entries: list[SnapshotEntry] = []
    for f in files:
        if isinstance(f, SnapshotEntry):
            entries.append(f)
            continue
        try:
            entries.append(_stat_entry(f))
        except (OSError, ValueError):
            continue
    # Dedup by normalized path (keep first).
    seen: set[str] = set()
    deduped: list[SnapshotEntry] = []
    for e in entries:
        norm = e.path.replace("\\", "/")
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(e)
    deduped.sort(key=lambda e: e.path.replace("\\", "/"))
    return DatasetSnapshot(
        entries=tuple(deduped),
        schema_version=str(schema_version),
        created_at_us=int(created_at_us if created_at_us is not None else time.time() * 1_000_000),
        content_hash=compute_content_hash(deduped),
        extra=dict(extra or {}),
    )


__all__ = [
    "SnapshotEntry",
    "DatasetSnapshot",
    "compute_content_hash",
    "build_snapshot",
]
