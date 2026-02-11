"""Shared DB-backed worker control/snapshot primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import WorkerControl, WorkerSnapshot


DEFAULT_WORKER_INTERVALS: dict[str, int] = {
    "scanner": 60,
    "news": 120,
    "weather": 14400,
    "crypto": 2,
    "tracked_traders": 60,
    "autotrader": 2,
    "discovery": 3600,
}


def _now() -> datetime:
    return utcnow()


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + "Z"


def _default_interval(worker_name: str) -> int:
    return int(DEFAULT_WORKER_INTERVALS.get(worker_name, 60))


async def ensure_worker_control(
    session: AsyncSession,
    worker_name: str,
    *,
    default_interval: Optional[int] = None,
) -> WorkerControl:
    result = await session.execute(
        select(WorkerControl).where(WorkerControl.worker_name == worker_name)
    )
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
    result = await session.execute(
        select(WorkerControl).where(WorkerControl.worker_name == worker_name)
    )
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
    result = await session.execute(
        select(WorkerSnapshot).where(WorkerSnapshot.worker_name == worker_name)
    )
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
    row.stats_json = stats or {}
    await session.commit()


async def read_worker_snapshot(
    session: AsyncSession,
    worker_name: str,
) -> dict[str, Any]:
    result = await session.execute(
        select(WorkerSnapshot).where(WorkerSnapshot.worker_name == worker_name)
    )
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
        "last_run_at": _to_iso(row.last_run_at),
        "lag_seconds": row.lag_seconds,
        "last_error": row.last_error,
        "stats": row.stats_json or {},
        "updated_at": _to_iso(row.updated_at),
    }


async def list_worker_snapshots(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.execute(
        select(WorkerSnapshot).order_by(WorkerSnapshot.worker_name.asc())
    )
    rows = list(result.scalars().all())

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        seen.add(row.worker_name)
        out.append(
            {
                "worker_name": row.worker_name,
                "running": bool(row.running),
                "enabled": bool(row.enabled),
                "current_activity": row.current_activity,
                "interval_seconds": int(row.interval_seconds or _default_interval(row.worker_name)),
                "last_run_at": _to_iso(row.last_run_at),
                "lag_seconds": row.lag_seconds,
                "last_error": row.last_error,
                "stats": row.stats_json or {},
                "updated_at": _to_iso(row.updated_at),
            }
        )

    for worker_name in DEFAULT_WORKER_INTERVALS:
        if worker_name in seen:
            continue
        out.append(await read_worker_snapshot(session, worker_name))

    out.sort(key=lambda r: r.get("worker_name") or "")
    return out
