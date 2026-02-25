"""Shared DB-backed worker control/snapshot primitives."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime

from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import WorkerControl, WorkerSnapshot
from services.event_bus import event_bus
from utils.converters import to_iso


DEFAULT_WORKER_INTERVALS: dict[str, int] = {
    "scanner": 60,
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


def _is_retryable_db_error(exc: Exception) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    return any(
        marker in message
        for marker in (
            "deadlock detected",
            "serialization failure",
            "could not serialize access",
            "lock not available",
            "too many clients already",
            "sorry, too many clients already",
            "remaining connection slots are reserved",
            "cannot connect now",
            "connection is closed",
            "underlying connection is closed",
            "connection has been closed",
            "closed the connection unexpectedly",
            "terminating connection",
            "connection reset by peer",
            "broken pipe",
        )
    )


def _db_retry_delay(attempt: int) -> float:
    return min(DB_RETRY_BASE_DELAY_SECONDS * (2**attempt), DB_RETRY_MAX_DELAY_SECONDS)


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


async def _commit_with_retry(
    session: AsyncSession,
    *,
    retry_attempts: int = DB_RETRY_ATTEMPTS,
    base_delay_seconds: float = DB_RETRY_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DB_RETRY_MAX_DELAY_SECONDS,
) -> None:
    if not hasattr(session, "commit"):
        return

    attempts = max(1, int(retry_attempts))
    base_delay = max(0.0, float(base_delay_seconds))
    max_delay = max(base_delay, float(max_delay_seconds))

    pending_snapshot = _capture_pending_session_state(session)
    for attempt in range(attempts):
        try:
            await session.commit()
            return
        except (OperationalError, InterfaceError) as exc:
            if hasattr(session, "rollback"):
                await session.rollback()
            is_locked = _is_retryable_db_error(exc)
            is_last = attempt >= attempts - 1
            if not is_locked or is_last:
                raise
            if pending_snapshot.get("new") or pending_snapshot.get("dirty") or pending_snapshot.get("deleted"):
                await _restore_pending_session_state(session, pending_snapshot)
            delay = min(base_delay * (2**attempt), max_delay)
            if delay > 0:
                await asyncio.sleep(delay)


def _now() -> datetime:
    return utcnow()


def _default_interval(worker_name: str) -> int:
    return int(DEFAULT_WORKER_INTERVALS.get(worker_name, 60))


def _read_process_rss_bytes(pid: int) -> Optional[int]:
    if pid <= 0:
        return None
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=0.25,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    output = str(proc.stdout or "").strip()
    if not output:
        return None

    line = output.splitlines()[-1].strip()
    if not line:
        return None

    try:
        rss_kib = int(line.split()[0])
    except (TypeError, ValueError):
        return None

    if rss_kib <= 0:
        return None
    return rss_kib * 1024


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
) -> None:
    result = await session.execute(select(WorkerSnapshot).where(WorkerSnapshot.worker_name == worker_name))
    row = result.scalar_one_or_none()
    if row is None:
        row = WorkerSnapshot(worker_name=worker_name)
        session.add(row)

    row.updated_at = _now()
    if last_run_at is not None:
        row.last_run_at = last_run_at
    row.running = bool(running)
    row.enabled = bool(enabled)
    row.current_activity = current_activity
    if interval_seconds is not None:
        row.interval_seconds = max(1, int(interval_seconds))
    row.lag_seconds = lag_seconds
    row.last_error = last_error
    stats_payload = _with_runtime_process_stats(stats)
    row.stats_json = stats_payload
    await _commit_with_retry(session)

    # Publish worker status update event.
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
                        "interval_seconds": int(row.interval_seconds or _default_interval(worker_name)),
                        "last_run_at": to_iso(last_run_at),
                        "lag_seconds": lag_seconds,
                        "last_error": last_error,
                        "updated_at": to_iso(row.updated_at),
                    }
                ],
            },
        )
    except Exception:
        pass  # fire-and-forget


async def read_worker_snapshot(
    session: AsyncSession,
    worker_name: str,
) -> dict[str, Any]:
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
) -> list[dict[str, Any]]:
    result = await session.execute(select(WorkerSnapshot).order_by(WorkerSnapshot.worker_name.asc()))
    rows = list(result.scalars().all())

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
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
            snapshot["stats"] = row.stats_json or {}
        out.append(snapshot)

    for worker_name in DEFAULT_WORKER_INTERVALS:
        if worker_name in seen:
            continue
        out.append(await read_worker_snapshot(session, worker_name))

    out.sort(key=lambda r: r.get("worker_name") or "")
    return out
