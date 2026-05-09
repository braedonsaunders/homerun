"""Shared DB-backed worker control/snapshot primitives."""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

from utils.utcnow import utcnow
from typing import Any, Iterable, Optional

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import WorkerControl, WorkerSnapshot
from services.event_bus import event_bus
from services.live_pressure import is_db_pressure_active, maybe_mark_db_pressure
from utils.converters import to_iso
from utils.logger import get_logger
from utils.retry import db_retry_delay as _shared_db_retry_delay

logger = get_logger("worker_state")
from utils.retry import is_retryable_db_error as _shared_is_retryable_db_error


_COMMIT_RETRYABLE_MARKERS = (
    "deadlock detected",
    "serialization failure",
    "could not serialize access",
    "lock not available",
    "lock timeout",
    "canceling statement due to lock timeout",
)


def _is_retryable_commit_error(exc: Exception) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    if any(marker in message for marker in _COMMIT_RETRYABLE_MARKERS):
        return True

    sqlstate = str(
        getattr(getattr(exc, "orig", None), "sqlstate", "")
        or getattr(exc, "sqlstate", "")
        or ""
    ).strip()
    return sqlstate in {"40P01", "40001", "55P03"}


DEFAULT_WORKER_INTERVALS: dict[str, int] = {
    "scanner": 60,
    "scanner_slo": 5,
    "market_universe": 120,
    "news": 120,
    "weather": 14400,
    "crypto": 2,
    "tracked_traders": 60,
    "trader_orchestrator": 5,
    "trader_reconciliation": 1,
    "redeemer": 120,
    "discovery": 3600,
    "events": 300,
}
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BASE_DELAY_SECONDS = 0.05
DB_RETRY_MAX_DELAY_SECONDS = 0.3
_SCALAR_STATUS_TYPES = (str, int, float, bool, type(None))
_WORKER_SNAPSHOT_STATEMENT_TIMEOUT_MS = 3000
_WORKER_SNAPSHOT_LOCK_TIMEOUT_MS = 500


def _is_retryable_db_error(exc: Exception) -> bool:
    return bool(_shared_is_retryable_db_error(exc))


def _db_retry_delay(
    attempt: int,
    *,
    base_delay: float = DB_RETRY_BASE_DELAY_SECONDS,
    max_delay: float = DB_RETRY_MAX_DELAY_SECONDS,
) -> float:
    return float(_shared_db_retry_delay(attempt, base_delay=base_delay, max_delay=max_delay))


def _is_status_scalar(value: Any) -> bool:
    return isinstance(value, _SCALAR_STATUS_TYPES)


async def _apply_snapshot_write_timeouts(session: AsyncSession) -> None:
    try:
        await session.execute(text(f"SET LOCAL statement_timeout = '{_WORKER_SNAPSHOT_STATEMENT_TIMEOUT_MS}ms'"))
    except Exception:
        pass
    try:
        await session.execute(text(f"SET LOCAL lock_timeout = '{_WORKER_SNAPSHOT_LOCK_TIMEOUT_MS}ms'"))
    except Exception:
        pass


def _summarize_execution_latency(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("internal_sla_definition", "internal_sla_target_ms", "rolling_window_seconds", "sample_count"):
        value = payload.get(key)
        if _is_status_scalar(value):
            summary[key] = value
    overall = payload.get("overall")
    if isinstance(overall, dict):
        overall_summary: dict[str, Any] = {}
        for stage_key, stage_value in overall.items():
            if _is_status_scalar(stage_value):
                overall_summary[stage_key] = stage_value
                continue
            if not isinstance(stage_value, dict):
                continue
            metric_summary: dict[str, Any] = {}
            for metric_key in ("count", "p50", "p95", "p99"):
                metric_value = stage_value.get(metric_key)
                if _is_status_scalar(metric_value):
                    metric_summary[metric_key] = metric_value
            if metric_summary:
                overall_summary[stage_key] = metric_summary
        if overall_summary:
            summary["overall"] = overall_summary
    for key in ("by_source", "by_strategy", "by_trader"):
        value = payload.get(key)
        if isinstance(value, dict):
            summary[key] = value
            summary[f"{key}_count"] = len(value)
    return summary


def summarize_worker_stats(stats: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    summary: dict[str, Any] = {}
    for key, value in stats.items():
        if key == "execution_latency" and isinstance(value, dict):
            latency_summary = _summarize_execution_latency(value)
            if latency_summary:
                summary[key] = latency_summary
            continue
        if _is_status_scalar(value):
            summary[key] = value
            continue
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
            continue
        if isinstance(value, dict):
            scalar_children = {
                child_key: child_value
                for child_key, child_value in value.items()
                if _is_status_scalar(child_value)
            }
            if scalar_children:
                summary[key] = scalar_children
            summary[f"{key}_count"] = len(value)
    return summary


def summarize_worker_snapshot(snapshot: Optional[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(snapshot or {})
    payload["stats"] = summarize_worker_stats(payload.get("stats"))
    return payload


def _capture_pending_session_state(session: AsyncSession) -> dict[str, Any]:
    """Capture pending SQLAlchemy unit-of-work state so lock retries can replay it."""
    if not all(hasattr(session, attr) for attr in ("new", "dirty", "deleted")):
        return {"new": [], "dirty": [], "deleted": []}

    try:
        pending_new = list(session.new)
        pending_dirty_candidates = list(session.dirty)
        pending_deleted_candidates = list(session.deleted)
    except Exception:
        return {"new": [], "dirty": [], "deleted": []}

    pending_deleted: list[tuple[type[Any], dict[str, Any]]] = []
    pending_dirty: list[tuple[type[Any], dict[str, Any], dict[str, Any]]] = []

    for obj in pending_dirty_candidates:
        if obj in pending_new or obj in pending_deleted_candidates:
            continue
        try:
            state = sa_inspect(obj)
        except Exception:
            continue
        primary_keys = {str(column.key): deepcopy(getattr(obj, str(column.key))) for column in state.mapper.primary_key}
        if not primary_keys:
            continue
        values: dict[str, Any] = {}
        for column_attr in state.mapper.column_attrs:
            key = column_attr.key
            if key in primary_keys:
                continue
            values[key] = deepcopy(getattr(obj, key))
        pending_dirty.append((type(obj), primary_keys, values))

    for obj in pending_deleted_candidates:
        if obj in pending_new:
            continue
        try:
            state = sa_inspect(obj)
        except Exception:
            continue
        primary_keys = {str(column.key): deepcopy(getattr(obj, str(column.key))) for column in state.mapper.primary_key}
        if not primary_keys:
            continue
        pending_deleted.append((type(obj), primary_keys))

    return {
        "new": pending_new,
        "dirty": pending_dirty,
        "deleted": pending_deleted,
    }


async def _restore_pending_session_state(session: AsyncSession, snapshot: dict[str, Any]) -> None:
    """Re-apply captured unit-of-work objects after rollback."""
    for obj in snapshot.get("new", []):
        session.add(obj)

    for model_cls, primary_keys, values in snapshot.get("dirty", []):
        if not values:
            continue
        where_clauses = [getattr(model_cls, key) == value for key, value in primary_keys.items()]
        await session.execute(sa_update(model_cls).where(*where_clauses).values(**deepcopy(values)))

    for model_cls, primary_keys in snapshot.get("deleted", []):
        where_clauses = [getattr(model_cls, key) == value for key, value in primary_keys.items()]
        await session.execute(sa_delete(model_cls).where(*where_clauses))


@dataclass(frozen=True)
class CommitMetrics:
    """R12-C: fine-grained timing/row metrics for a ``_commit_with_retry`` call.

    Used by the trader orchestrator (and anyone else who wants to
    attribute slow commits) to split a 3s ``ps_db_commit`` budget into:

    * ``commit_call_ms`` — wall time of the final successful
      ``await session.commit()`` itself.  Excludes rollback, sleep, and
      replay time inside the retry loop.  If this is high, the commit
      itself is slow (big transaction, lock wait at commit).
    * ``total_ms`` — wall time of the entire ``_commit_with_retry``
      call, including any retries.  ``total_ms - commit_call_ms`` is
      the overhead from rollback + sleep + replay.
    * ``retry_count`` — number of retries before success.  ``0`` means
      the first attempt succeeded.
    * ``dirty_rows`` — ``len(session.new) + len(session.dirty) +
      len(session.deleted)`` captured before the first commit attempt.
      Lets us tell "one big transaction" apart from "small transaction
      waiting on a lock".

    Returned from every ``_commit_with_retry`` call.  Callers that
    don't care can safely ignore the return value — existing semantics
    (return ``None`` on success, raise on final failure) are preserved
    because failures still raise and successful returns now hand back
    metrics instead of ``None``.
    """

    commit_call_ms: int
    total_ms: int
    retry_count: int
    dirty_rows: int


def _count_pending_rows(session: AsyncSession) -> int:
    """Best-effort pending-unit-of-work row count for R12-C metrics."""
    try:
        new = len(session.new) if hasattr(session, "new") else 0
        dirty = len(session.dirty) if hasattr(session, "dirty") else 0
        deleted = len(session.deleted) if hasattr(session, "deleted") else 0
        return int(new + dirty + deleted)
    except Exception:
        return 0


_SLOW_COMMIT_THRESHOLD_MS = 2000
_SLOW_COMMIT_DIAG_LAST_AT_MONO: float = 0.0
_SLOW_COMMIT_DIAG_INTERVAL_S = 30.0


async def _capture_slow_commit_diagnostic(commit_ms: int, dirty_rows: int) -> None:
    """Capture pg-side state when a commit takes >_SLOW_COMMIT_THRESHOLD_MS.

    The 2026-05-09 soak showed steady-state ``ps_db_commit`` at 1500 ms/row
    vs a baseline of 90 ms/row at startup — a 16× regression with
    ``ps_db_commit_retries=0`` (no LockNotAvailable, no row-lock
    contention). The slowness is below the application layer: index
    update overhead, TOAST rewrites, WAL fsync latency, or autovacuum
    interaction. To diagnose without expensive always-on observation,
    capture pg_stat_user_tables (n_dead_tup, n_live_tup, last_vacuum,
    last_autovacuum), pg_stat_user_indexes (idx_scan), and the current
    pg_stat_activity for the session's PID — only when a slow commit
    fires AND we haven't dumped recently (rate-limited to once per
    30s to keep the log readable).

    Uses a fresh asyncpg connection so it never competes with the work
    it's trying to observe.
    """
    global _SLOW_COMMIT_DIAG_LAST_AT_MONO
    now = time.monotonic()
    if now - _SLOW_COMMIT_DIAG_LAST_AT_MONO < _SLOW_COMMIT_DIAG_INTERVAL_S:
        return
    _SLOW_COMMIT_DIAG_LAST_AT_MONO = now

    try:
        import asyncpg
        from config import settings as _settings

        dsn = str(_settings.DATABASE_URL).replace("+asyncpg", "")
        probe = await asyncio.wait_for(asyncpg.connect(dsn=dsn, timeout=3), timeout=4)
    except Exception as exc:
        logger.debug("slow_commit diag connect failed: %s", exc)
        return

    try:
        # Per-table bloat + autovacuum lag for the hot tables that the
        # orchestrator commits typically touch. Add others if the
        # diagnostic surfaces gaps.
        table_rows = await asyncio.wait_for(
            probe.fetch(
                """
                SELECT relname,
                       n_live_tup,
                       n_dead_tup,
                       n_mod_since_analyze,
                       EXTRACT(EPOCH FROM now() - last_vacuum)::int AS last_vacuum_age_s,
                       EXTRACT(EPOCH FROM now() - last_autovacuum)::int AS last_autovacuum_age_s,
                       EXTRACT(EPOCH FROM now() - last_analyze)::int AS last_analyze_age_s,
                       EXTRACT(EPOCH FROM now() - last_autoanalyze)::int AS last_autoanalyze_age_s
                FROM pg_stat_user_tables
                WHERE relname IN (
                    'trade_signals',
                    'trader_decisions',
                    'trader_decision_checks',
                    'trade_signal_emissions',
                    'trader_signal_consumptions',
                    'trader_events'
                )
                ORDER BY n_dead_tup DESC NULLS LAST
                """,
            ),
            timeout=3,
        )
        # Top WAL writers: who's been writing the most since pg_stat_reset?
        # Combined with the table-level dead_tup, this points at the
        # producer driving the load.
        wal_rows = await asyncio.wait_for(
            probe.fetch(
                """
                SELECT relname,
                       n_tup_ins,
                       n_tup_upd,
                       n_tup_del,
                       n_tup_hot_upd
                FROM pg_stat_user_tables
                WHERE relname IN (
                    'trade_signals',
                    'trade_signal_emissions',
                    'trader_decisions'
                )
                """,
            ),
            timeout=3,
        )
    except Exception as exc:
        logger.debug("slow_commit diag query failed: %s", exc)
        try:
            await probe.close()
        except Exception:
            pass
        return
    finally:
        try:
            await probe.close()
        except Exception:
            pass

    tables = {
        r["relname"]: {
            "live": int(r["n_live_tup"] or 0),
            "dead": int(r["n_dead_tup"] or 0),
            "dead_pct": (
                round(100.0 * (r["n_dead_tup"] or 0) / max(1, (r["n_live_tup"] or 0) + (r["n_dead_tup"] or 0)), 1)
            ),
            "mod_since_analyze": int(r["n_mod_since_analyze"] or 0),
            "vacuum_age_s": int(r["last_vacuum_age_s"] or 0),
            "autovacuum_age_s": int(r["last_autovacuum_age_s"] or 0),
            "analyze_age_s": int(r["last_analyze_age_s"] or 0),
            "autoanalyze_age_s": int(r["last_autoanalyze_age_s"] or 0),
        }
        for r in table_rows
    }
    wal = {
        r["relname"]: {
            "ins": int(r["n_tup_ins"] or 0),
            "upd": int(r["n_tup_upd"] or 0),
            "del": int(r["n_tup_del"] or 0),
            "hot_upd": int(r["n_tup_hot_upd"] or 0),
            "hot_pct": (
                round(100.0 * (r["n_tup_hot_upd"] or 0) / max(1, r["n_tup_upd"] or 0), 1)
            ),
        }
        for r in wal_rows
    }
    logger.warning(
        "SLOW COMMIT DIAGNOSTIC commit_ms=%d dirty_rows=%d tables=%r wal=%r",
        commit_ms,
        dirty_rows,
        tables,
        wal,
    )


async def _commit_with_retry(
    session: AsyncSession,
    *,
    retry_attempts: int = DB_RETRY_ATTEMPTS,
    base_delay_seconds: float = DB_RETRY_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DB_RETRY_MAX_DELAY_SECONDS,
) -> "CommitMetrics | None":
    if not hasattr(session, "commit"):
        return None

    attempts = max(1, int(retry_attempts))
    base_delay = max(0.0, float(base_delay_seconds))
    max_delay = max(base_delay, float(max_delay_seconds))

    # R12-C: capture row count + start the total-wall timer before we
    # snapshot pending state.  ``dirty_rows`` is sampled once, pre-
    # commit; on retry the pending snapshot is replayed so the count
    # stays meaningful across attempts.
    dirty_rows = _count_pending_rows(session)
    total_started = time.monotonic()

    pending_snapshot = _capture_pending_session_state(session)
    retry_count = 0
    for attempt in range(attempts):
        try:
            # R12-C: time only the commit call itself, not the
            # surrounding retry loop.  On the final successful attempt
            # this is what we emit as ``ps_db_commit_call``.
            commit_started = time.monotonic()
            await session.commit()
            commit_call_ms = int((time.monotonic() - commit_started) * 1000)
            total_ms = int((time.monotonic() - total_started) * 1000)
            # 2026-05-09 slow-commit diagnostic: when a single commit
            # exceeds the threshold AND nothing visible at the
            # application layer (retry_count=0, no LockNotAvailable)
            # explains it, the cause is DB-side. Capture pg_stat
            # state inline so the next slow-commit log line carries
            # the autovacuum / dead-tup / WAL-writer evidence.
            # Rate-limited to one dump per _SLOW_COMMIT_DIAG_INTERVAL_S
            # to keep the log readable.
            if commit_call_ms >= _SLOW_COMMIT_THRESHOLD_MS and retry_count == 0:
                try:
                    await _capture_slow_commit_diagnostic(commit_call_ms, dirty_rows)
                except Exception as diag_exc:
                    logger.debug("slow_commit diag suppressed: %s", diag_exc)
            return CommitMetrics(
                commit_call_ms=commit_call_ms,
                total_ms=total_ms,
                retry_count=retry_count,
                dirty_rows=dirty_rows,
            )
        except DBAPIError as exc:
            if hasattr(session, "rollback"):
                await session.rollback()
            is_locked = _is_retryable_commit_error(exc)
            is_last = attempt >= attempts - 1
            if not is_locked or is_last:
                raise
            retry_count += 1
            if pending_snapshot.get("new") or pending_snapshot.get("dirty") or pending_snapshot.get("deleted"):
                await _restore_pending_session_state(session, pending_snapshot)
            delay = min(base_delay * (2**attempt), max_delay)
            if delay > 0:
                await asyncio.sleep(delay)
    return None


def _now() -> datetime:
    return utcnow()


def _default_interval(worker_name: str) -> int:
    return int(DEFAULT_WORKER_INTERVALS.get(worker_name, 60))


def _read_process_rss_bytes(pid: int) -> Optional[int]:
    if pid <= 0:
        return None
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        try:
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
        except Exception:
            return None

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            rss_bytes = int(counters.WorkingSetSize or 0)
            return rss_bytes if rss_bytes > 0 else None
        finally:
            kernel32.CloseHandle(handle)

    statm_path = f"/proc/{int(pid)}/statm"
    try:
        with open(statm_path, "r", encoding="utf-8") as handle:
            fields = handle.read().strip().split()
    except Exception:
        return None
    if len(fields) < 2:
        return None
    try:
        resident_pages = int(fields[1])
    except (TypeError, ValueError):
        return None
    if resident_pages <= 0:
        return None
    page_size = os.sysconf("SC_PAGE_SIZE")
    rss_bytes = resident_pages * int(page_size)
    return rss_bytes if rss_bytes > 0 else None


def _read_peak_rss_bytes() -> Optional[int]:
    if sys.platform == "win32":
        return None
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    except Exception:
        return None
    if peak <= 0:
        return None
    if sys.platform == "darwin":
        return peak
    return peak * 1024


def _with_runtime_process_stats(base_stats: Optional[dict[str, Any]]) -> dict[str, Any]:
    stats_payload = dict(base_stats or {})
    pid = os.getpid()
    stats_payload["pid"] = pid

    rss_bytes = _read_process_rss_bytes(pid)
    if rss_bytes is None:
        rss_bytes = _read_peak_rss_bytes()
    if rss_bytes is not None and rss_bytes > 0:
        stats_payload["rss_bytes"] = int(rss_bytes)
        stats_payload["memory_mb"] = round(float(rss_bytes) / (1024 * 1024), 1)
    return stats_payload


async def ensure_worker_control(
    session: AsyncSession,
    worker_name: str,
    *,
    default_interval: Optional[int] = None,
) -> WorkerControl:
    result = await session.execute(select(WorkerControl).where(WorkerControl.worker_name == worker_name))
    row = result.scalar_one_or_none()
    if row is None:
        row = WorkerControl(
            worker_name=worker_name,
            is_enabled=True,
            is_paused=False,
            interval_seconds=int(default_interval or _default_interval(worker_name)),
            requested_run_at=None,
            updated_at=_now(),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_worker_control(
    session: AsyncSession,
    worker_name: str,
    *,
    default_interval: Optional[int] = None,
) -> dict[str, Any]:
    result = await session.execute(select(WorkerControl).where(WorkerControl.worker_name == worker_name))
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "worker_name": worker_name,
            "is_enabled": True,
            "is_paused": False,
            "interval_seconds": int(default_interval or _default_interval(worker_name)),
            "requested_run_at": None,
            "updated_at": None,
        }
    return {
        "worker_name": row.worker_name,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int(row.interval_seconds or default_interval or _default_interval(worker_name)),
        "requested_run_at": row.requested_run_at,
        "updated_at": row.updated_at,
    }


async def set_worker_paused(
    session: AsyncSession,
    worker_name: str,
    paused: bool,
) -> None:
    row = await ensure_worker_control(session, worker_name)
    row.is_paused = bool(paused)
    row.updated_at = _now()
    await session.commit()


async def set_worker_interval(
    session: AsyncSession,
    worker_name: str,
    interval_seconds: int,
) -> None:
    row = await ensure_worker_control(session, worker_name)
    row.interval_seconds = max(1, min(86400, int(interval_seconds)))
    row.updated_at = _now()
    await session.commit()


async def request_worker_run(session: AsyncSession, worker_name: str) -> None:
    row = await ensure_worker_control(session, worker_name)
    row.requested_run_at = _now()
    row.updated_at = _now()
    await session.commit()


async def clear_worker_run_request(session: AsyncSession, worker_name: str) -> None:
    row = await ensure_worker_control(session, worker_name)
    row.requested_run_at = None
    row.updated_at = _now()
    await session.commit()


async def write_worker_snapshot(
    session: AsyncSession,
    worker_name: str,
    *,
    running: bool,
    enabled: bool,
    current_activity: Optional[str],
    interval_seconds: Optional[int],
    last_run_at: Optional[datetime] = None,
    lag_seconds: Optional[float] = None,
    last_error: Optional[str] = None,
    stats: Optional[dict[str, Any]] = None,
    publish_event: bool = True,
) -> None:
    await _apply_snapshot_write_timeouts(session)
    updated_at = _now()
    resolved_interval = (
        max(1, int(interval_seconds))
        if interval_seconds is not None
        else _default_interval(worker_name)
    )
    base_stats = summarize_worker_stats(stats) if is_db_pressure_active() else stats
    stats_payload = _with_runtime_process_stats(base_stats)
    values = {
        "worker_name": worker_name,
        "updated_at": updated_at,
        "last_run_at": last_run_at,
        "running": bool(running),
        "enabled": bool(enabled),
        "current_activity": current_activity,
        "interval_seconds": resolved_interval,
        "lag_seconds": lag_seconds,
        "last_error": last_error,
        "stats_json": stats_payload,
    }
    insert_stmt = pg_insert(WorkerSnapshot).values(**values)
    try:
        await session.execute(
            insert_stmt.on_conflict_do_update(
                index_elements=[WorkerSnapshot.worker_name],
                set_={
                    "updated_at": insert_stmt.excluded.updated_at,
                    "last_run_at": insert_stmt.excluded.last_run_at,
                    "running": insert_stmt.excluded.running,
                    "enabled": insert_stmt.excluded.enabled,
                    "current_activity": insert_stmt.excluded.current_activity,
                    "interval_seconds": insert_stmt.excluded.interval_seconds,
                    "lag_seconds": insert_stmt.excluded.lag_seconds,
                    "last_error": insert_stmt.excluded.last_error,
                    "stats_json": insert_stmt.excluded.stats_json,
                },
            )
        )
        await _commit_with_retry(session)
    except Exception as exc:
        maybe_mark_db_pressure(exc, component=f"worker_snapshot:{worker_name}")
        raise

    # Publish worker status update event (fire-and-forget, don't hold DB conn).
    if publish_event:
        _updated_at_iso = to_iso(updated_at)
        _interval = int(resolved_interval)

        async def _publish_event() -> None:
            try:
                await event_bus.publish(
                    "worker_status_update",
                    {
                        "workers": [
                            {
                                "worker_name": worker_name,
                                "running": bool(running),
                                "enabled": bool(enabled),
                                "current_activity": current_activity,
                                "interval_seconds": _interval,
                                "last_run_at": to_iso(last_run_at),
                                "lag_seconds": lag_seconds,
                                "last_error": last_error,
                                "updated_at": _updated_at_iso,
                            }
                        ],
                    },
                )
            except Exception:
                pass

        asyncio.ensure_future(_publish_event())


async def read_worker_snapshot(
    session: AsyncSession,
    worker_name: str,
) -> dict[str, Any]:
    normalized_worker_name = str(worker_name or "").strip().lower()
    if normalized_worker_name == "trader_orchestrator":
        from services.trader_orchestrator_state import (
            ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
            read_orchestrator_control,
            read_orchestrator_snapshot,
        )

        control = await read_orchestrator_control(session)
        snapshot = await read_orchestrator_snapshot(session)
        interval_seconds = int(
            control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
        )
        return {
            "worker_name": "trader_orchestrator",
            "running": bool(snapshot.get("running", False)),
            "enabled": bool(control.get("is_enabled", True)) and not bool(control.get("is_paused", False)),
            "current_activity": snapshot.get("current_activity"),
            "interval_seconds": interval_seconds,
            "last_run_at": snapshot.get("last_run_at"),
            "lag_seconds": None,
            "last_error": snapshot.get("last_error"),
            "stats": snapshot.get("stats", {}) if isinstance(snapshot, dict) else {},
            "updated_at": snapshot.get("updated_at"),
            "control": {
                "is_enabled": bool(control.get("is_enabled", True)),
                "is_paused": bool(control.get("is_paused", False)),
                "interval_seconds": interval_seconds,
                "requested_run_at": to_iso(control.get("requested_run_at")),
            },
        }

    result = await session.execute(select(WorkerSnapshot).where(WorkerSnapshot.worker_name == worker_name))
    row = result.scalar_one_or_none()
    if row is None:
        control = await read_worker_control(session, worker_name)
        return {
            "worker_name": worker_name,
            "running": False,
            "enabled": bool(control.get("is_enabled", True)) and not bool(control.get("is_paused", False)),
            "current_activity": "Waiting for worker startup.",
            "interval_seconds": int(control.get("interval_seconds") or _default_interval(worker_name)),
            "last_run_at": None,
            "lag_seconds": None,
            "last_error": None,
            "stats": {},
            "updated_at": None,
        }

    return {
        "worker_name": row.worker_name,
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "current_activity": row.current_activity,
        "interval_seconds": int(row.interval_seconds or _default_interval(worker_name)),
        "last_run_at": to_iso(row.last_run_at),
        "lag_seconds": row.lag_seconds,
        "last_error": row.last_error,
        "stats": row.stats_json or {},
        "updated_at": to_iso(row.updated_at),
    }


async def list_worker_snapshots(
    session: AsyncSession,
    *,
    include_stats: bool = True,
    stats_mode: str = "full",
    worker_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    normalized_stats_mode = str(stats_mode or "full").strip().lower()
    include_stats = bool(include_stats and normalized_stats_mode != "none")
    normalized_worker_names = {
        str(worker_name or "").strip().lower()
        for worker_name in (worker_names or ())
        if str(worker_name or "").strip()
    }
    worker_filter = normalized_worker_names or None
    db_worker_names = sorted(
        worker_name
        for worker_name in (worker_filter or DEFAULT_WORKER_INTERVALS.keys())
        if worker_name != "trader_orchestrator"
    )
    rows: list[WorkerSnapshot] = []
    if db_worker_names:
        result = await session.execute(
            select(WorkerSnapshot)
            .where(WorkerSnapshot.worker_name.in_(db_worker_names))
            .order_by(WorkerSnapshot.worker_name.asc())
        )
        rows = list(result.scalars().all())

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if worker_filter is None or "trader_orchestrator" in worker_filter:
        orchestrator_snapshot = await read_worker_snapshot(session, "trader_orchestrator")
        if not include_stats:
            orchestrator_snapshot.pop("stats", None)
        elif normalized_stats_mode == "summary":
            orchestrator_snapshot["stats"] = summarize_worker_stats(orchestrator_snapshot.get("stats"))
        out.append(orchestrator_snapshot)
        seen.add("trader_orchestrator")
    for row in rows:
        seen.add(row.worker_name)
        snapshot = {
            "worker_name": row.worker_name,
            "running": bool(row.running),
            "enabled": bool(row.enabled),
            "current_activity": row.current_activity,
            "interval_seconds": int(row.interval_seconds or _default_interval(row.worker_name)),
            "last_run_at": to_iso(row.last_run_at),
            "lag_seconds": row.lag_seconds,
            "last_error": row.last_error,
            "updated_at": to_iso(row.updated_at),
        }
        if include_stats:
            stats_payload = row.stats_json or {}
            snapshot["stats"] = (
                summarize_worker_stats(stats_payload)
                if normalized_stats_mode == "summary"
                else stats_payload
            )
        out.append(snapshot)

    for worker_name in sorted(worker_filter or DEFAULT_WORKER_INTERVALS.keys()):
        if worker_name in seen:
            continue
        out.append(await read_worker_snapshot(session, worker_name))

    out.sort(key=lambda r: r.get("worker_name") or "")
    return out
