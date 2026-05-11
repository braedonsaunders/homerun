"""Recorded-event-bus pruner.

The bus's parquet-backed topics accumulate without bound by default —
``crypto.update.dispatch`` alone can be ~20 MB / 5 minutes during peak
crypto activity, which is ~6 GB/day if you let it run forever.  This
module enforces three rotation rules:

  1. **Per-topic age cap (``retention_days``).**  Any whole partition
     file (one day, one entity) older than the cap gets deleted.
     Day-precision so the pruner deletes whole directories rather than
     opening files to check row-level timestamps.

  2. **Per-topic size cap (``max_bytes``).**  When a topic's total
     on-disk size exceeds this, oldest partition files are deleted
     (oldest first, day-major ordering) until under cap.

  3. **Global size cap (``recorded_event_bus_global_max_bytes``).**
     Sum across all parquet topics.  When exceeded, oldest files
     across the LARGEST topic are pruned first; iterates until under
     cap or no more files to delete.

The pruner runs as a single asyncio task on the live process,
firing every ``_PRUNE_INTERVAL_SECONDS`` (default 5 min).  The
master kill switch is ``app_settings.recorded_event_bus_pruner_enabled``
— flipping it off pauses pruning without restarting the app.

Per-topic ``enabled=false`` does NOT pause pruning for that topic —
disabled topics still need their disk reclaimed.  ``enabled=false``
gates whether the bus accepts new publishes (see
``catalog.require_topic``); the pruner's job is independent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal, TopicCatalog

logger = logging.getLogger(__name__)


_PRUNE_INTERVAL_SECONDS = 300.0  # 5 minutes — matches our flush cadence


# ── Disk usage helpers ──────────────────────────────────────────────


def _topic_dir_size_bytes(uri: str) -> int:
    """Total bytes of every .parquet file under a topic root.  Returns
    0 if the directory doesn't exist (operator may have cleared it
    manually)."""
    p = Path(uri)
    if not p.exists():
        return 0
    total = 0
    for fp in p.rglob("*.parquet"):
        try:
            total += fp.stat().st_size
        except OSError:
            continue
    return total


def _list_partition_files(uri: str) -> list[tuple[Path, int, float]]:
    """Returns (path, size_bytes, mtime) for every parquet file under
    a topic, sorted oldest-first by mtime.  Used by the size-cap
    pruner to decide which files to delete first."""
    p = Path(uri)
    if not p.exists():
        return []
    out: list[tuple[Path, int, float]] = []
    for fp in p.rglob("*.parquet"):
        try:
            st = fp.stat()
        except OSError:
            continue
        out.append((fp, st.st_size, st.st_mtime))
    out.sort(key=lambda x: x[2])  # oldest mtime first
    return out


def _list_partition_dirs_older_than(uri: str, cutoff_dt: datetime) -> list[Path]:
    """Whole partition directories (``{uri}/{entity}/{YYYY-MM-DD}``)
    whose date is before the cutoff.  Used for age-based pruning —
    deleting a whole day-dir is faster than file-by-file and matches
    how the writer organises data."""
    p = Path(uri)
    if not p.exists():
        return []
    cutoff_date = cutoff_dt.date()
    out: list[Path] = []
    for entity_dir in p.iterdir():
        if not entity_dir.is_dir():
            continue
        for date_dir in entity_dir.iterdir():
            if not date_dir.is_dir():
                continue
            try:
                d = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < cutoff_date:
                out.append(date_dir)
    return out


def _delete_dir_safely(p: Path) -> int:
    """Delete a directory + return bytes freed.  Logs failures but
    doesn't raise (one bad permission shouldn't break the prune loop)."""
    if not p.exists():
        return 0
    freed = 0
    try:
        for fp in p.rglob("*.parquet"):
            try:
                freed += fp.stat().st_size
                fp.unlink()
            except OSError:
                continue
        # Empty parent dirs (entity_dir might still hold other dates)
        # Walk bottom-up + remove empty dirs.
        for sub in sorted(p.rglob("*"), reverse=True):
            if sub.is_dir():
                try:
                    sub.rmdir()
                except OSError:
                    pass
        try:
            p.rmdir()
        except OSError:
            pass
    except Exception:
        logger.exception("pruner: failed to delete %s", p)
    return freed


# ── Age cap ─────────────────────────────────────────────────────────


def _prune_topic_age(spec_slug: str, uri: str, retention_days: int) -> int:
    """Delete partition dirs older than retention_days.  Returns bytes
    freed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    dirs = _list_partition_dirs_older_than(uri, cutoff)
    if not dirs:
        return 0
    freed = 0
    for d in dirs:
        n = _delete_dir_safely(d)
        freed += n
    if freed > 0:
        logger.info(
            "pruner age cap: %s freed %d MB across %d partitions (older than %d days)",
            spec_slug, freed // (1024 * 1024), len(dirs), retention_days,
        )
    return freed


# ── Per-topic size cap ──────────────────────────────────────────────


def _prune_topic_size(spec_slug: str, uri: str, max_bytes: int) -> int:
    """Delete oldest files until the topic is under max_bytes.  Returns
    bytes freed."""
    files = _list_partition_files(uri)
    total = sum(s for _f, s, _m in files)
    if total <= max_bytes:
        return 0
    target = max_bytes
    freed = 0
    n_deleted = 0
    for fp, size, _mtime in files:
        if total - freed <= target:
            break
        try:
            fp.unlink()
            freed += size
            n_deleted += 1
        except OSError:
            continue
    if freed > 0:
        logger.info(
            "pruner size cap: %s freed %d MB across %d files (cap %d MB)",
            spec_slug, freed // (1024 * 1024), n_deleted, max_bytes // (1024 * 1024),
        )
    return freed


# ── Global size cap ─────────────────────────────────────────────────


def _prune_global_size(specs_with_uri: list[tuple[str, str]], global_max_bytes: int) -> int:
    """Across-all-parquet-topics cap.  Repeatedly identifies the
    largest topic + deletes its oldest file until total fits under
    cap.  Returns bytes freed."""
    sizes = {slug: _topic_dir_size_bytes(uri) for slug, uri in specs_with_uri}
    total = sum(sizes.values())
    if total <= global_max_bytes:
        return 0
    uri_by_slug = dict(specs_with_uri)
    freed = 0
    iters = 0
    max_iters = 1000  # belt-and-brace
    while total - freed > global_max_bytes and iters < max_iters:
        iters += 1
        # Pick largest current topic.
        cur = {s: sizes[s] - 0 for s in sizes}  # already integers
        largest = max(cur, key=lambda s: cur[s])
        if cur[largest] <= 0:
            break
        files = _list_partition_files(uri_by_slug[largest])
        if not files:
            sizes[largest] = 0
            continue
        # Delete oldest file from the largest topic.
        fp, size, _mtime = files[0]
        try:
            fp.unlink()
            freed += size
            sizes[largest] -= size
        except OSError:
            sizes[largest] -= size  # treat as freed-from-our-perspective
    if freed > 0:
        logger.info(
            "pruner global cap: freed %d MB to fit under %d MB",
            freed // (1024 * 1024), global_max_bytes // (1024 * 1024),
        )
    return freed


# ── Top-level pass ──────────────────────────────────────────────────


async def prune_once() -> dict[str, Any]:
    """One pruning pass.  Returns a small report dict (handy for
    operator UI to surface "what did the last run do")."""
    async with AsyncSessionLocal() as session:
        settings_row = (
            await session.execute(select(AppSettings).limit(1))
        ).scalar_one_or_none()
        if settings_row is not None and settings_row.recorded_event_bus_pruner_enabled is False:
            return {"skipped": "pruner disabled in app_settings"}
        global_cap = settings_row.recorded_event_bus_global_max_bytes if settings_row else None
        topics = (
            await session.execute(
                select(TopicCatalog).where(
                    TopicCatalog.storage_kind.in_(("parquet", "external_parquet"))
                )
            )
        ).scalars().all()

    report: dict[str, Any] = {
        "scanned_topics": len(topics),
        "freed_bytes_age": 0,
        "freed_bytes_per_topic": 0,
        "freed_bytes_global": 0,
        "global_cap": global_cap,
    }
    parquet_specs: list[tuple[str, str]] = []  # for global pass — only bus-native parquet (we don't touch external_parquet on global)

    for t in topics:
        if not t.storage_uri:
            continue
        # Age cap (both kinds — operator might set retention on a
        # Telonex topic too).
        if t.retention_days is not None and t.retention_days > 0:
            try:
                report["freed_bytes_age"] += _prune_topic_age(
                    t.slug, t.storage_uri, int(t.retention_days),
                )
            except Exception:
                logger.exception("pruner: age cap failed for %s", t.slug)
        # Per-topic size cap (only meaningful for bus-native parquet,
        # since external_parquet has its own layout where deleting a
        # file would corrupt an import the operator uploaded).
        if t.storage_kind == "parquet":
            parquet_specs.append((t.slug, t.storage_uri))
            if t.max_bytes is not None and t.max_bytes > 0:
                try:
                    report["freed_bytes_per_topic"] += _prune_topic_size(
                        t.slug, t.storage_uri, int(t.max_bytes),
                    )
                except Exception:
                    logger.exception("pruner: per-topic size cap failed for %s", t.slug)

    # Global cap — bus-native parquet only.
    if global_cap is not None and global_cap > 0 and parquet_specs:
        try:
            report["freed_bytes_global"] = _prune_global_size(parquet_specs, int(global_cap))
        except Exception:
            logger.exception("pruner: global size cap failed")

    # Refresh bytes_on_disk in catalog after pruning so UI doesn't
    # show stale totals.
    try:
        async with AsyncSessionLocal() as session:
            for t in topics:
                if not t.storage_uri:
                    continue
                actual = _topic_dir_size_bytes(t.storage_uri)
                row = (await session.execute(
                    select(TopicCatalog).where(TopicCatalog.slug == t.slug)
                )).scalar_one_or_none()
                if row is not None:
                    row.bytes_on_disk = actual
            await session.commit()
    except Exception:
        logger.exception("pruner: failed to refresh bytes_on_disk")

    return report


# ── Background loop ─────────────────────────────────────────────────


_pruner_task: Optional[asyncio.Task] = None
_pruner_lock: Optional[asyncio.Lock] = None


def _get_pruner_lock() -> asyncio.Lock:
    global _pruner_lock
    if _pruner_lock is None:
        _pruner_lock = asyncio.Lock()
    return _pruner_lock


async def start_pruner() -> None:
    """Lifespan-managed task.  Idempotent; safe to call from app
    startup hook."""
    global _pruner_task
    async with _get_pruner_lock():
        if _pruner_task is not None and not _pruner_task.done():
            return
        _pruner_task = asyncio.create_task(_prune_loop(), name="rec-event-bus-pruner")
        logger.info("recorded-event-bus pruner started (interval=%ds)", int(_PRUNE_INTERVAL_SECONDS))


async def stop_pruner() -> None:
    """Lifespan shutdown hook."""
    global _pruner_task
    if _pruner_task is None:
        return
    _pruner_task.cancel()
    try:
        await _pruner_task
    except (asyncio.CancelledError, BaseException):
        pass
    _pruner_task = None


async def _prune_loop() -> None:
    """Wakeup → prune → sleep.  First run fires after the interval to
    avoid pinging the disk during app boot."""
    while True:
        try:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            t0 = time.monotonic()
            report = await prune_once()
            elapsed = time.monotonic() - t0
            if any(v for k, v in report.items() if k.startswith("freed_") and isinstance(v, int) and v > 0):
                logger.info(
                    "pruner pass complete in %.1fs: %s",
                    elapsed, {k: v for k, v in report.items() if k != "global_cap"},
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pruner: pass failed")
