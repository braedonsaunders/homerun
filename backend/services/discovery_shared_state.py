"""Shared DB state for trader discovery worker/API."""

from __future__ import annotations

from datetime import datetime, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import DiscoveryControl, DiscoverySnapshot

DISCOVERY_SNAPSHOT_ID = "latest"
DISCOVERY_CONTROL_ID = "default"


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("+00:00+00:00"):
        text = text[:-6]
    if text.endswith("Z"):
        text = text[:-1]
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _format_iso_utc_z(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + "Z"


def _default_status() -> dict[str, Any]:
    return {
        "running": False,
        "enabled": True,
        "run_interval_minutes": settings.DISCOVERY_RUN_INTERVAL_MINUTES,
        "priority_backlog_mode": True,
        "last_run_at": None,
        "current_activity": "Waiting for discovery worker.",
        "wallets_discovered_last_run": 0,
        "wallets_analyzed_last_run": 0,
    }


async def write_discovery_snapshot(
    session: AsyncSession,
    status: dict[str, Any],
) -> None:
    raw_last_run = status.get("last_run_at")
    if isinstance(raw_last_run, str):
        try:
            last_run = _parse_iso_datetime(raw_last_run)
        except Exception:
            last_run = None
    elif isinstance(raw_last_run, datetime):
        last_run = raw_last_run
    else:
        last_run = None

    result = await session.execute(select(DiscoverySnapshot).where(DiscoverySnapshot.id == DISCOVERY_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = DiscoverySnapshot(id=DISCOVERY_SNAPSHOT_ID)
        session.add(row)

    row.updated_at = utcnow()
    row.last_run_at = last_run or row.last_run_at
    row.running = bool(status.get("running", row.running))
    row.enabled = bool(status.get("enabled", row.enabled))
    row.current_activity = status.get("current_activity", row.current_activity)
    row.run_interval_minutes = int(
        status.get("run_interval_minutes", row.run_interval_minutes or settings.DISCOVERY_RUN_INTERVAL_MINUTES)
    )
    row.wallets_discovered_last_run = int(
        status.get("wallets_discovered_last_run", row.wallets_discovered_last_run or 0)
    )
    row.wallets_analyzed_last_run = int(status.get("wallets_analyzed_last_run", row.wallets_analyzed_last_run or 0))
    await session.commit()


async def read_discovery_snapshot(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(select(DiscoverySnapshot).where(DiscoverySnapshot.id == DISCOVERY_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return _default_status()

    return {
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "run_interval_minutes": int(row.run_interval_minutes or settings.DISCOVERY_RUN_INTERVAL_MINUTES),
        "last_run_at": _format_iso_utc_z(row.last_run_at),
        "current_activity": row.current_activity,
        "wallets_discovered_last_run": int(row.wallets_discovered_last_run or 0),
        "wallets_analyzed_last_run": int(row.wallets_analyzed_last_run or 0),
    }


async def get_discovery_status_from_db(session: AsyncSession) -> dict[str, Any]:
    status = await read_discovery_snapshot(session)
    control = await read_discovery_control(session)
    status["paused"] = bool(control.get("is_paused", False))
    status["priority_backlog_mode"] = bool(control.get("priority_backlog_mode", True))
    status["requested_run_at"] = (
        _format_iso_utc_z(control.get("requested_run_at")) if control.get("requested_run_at") else None
    )
    return status


async def ensure_discovery_control(session: AsyncSession) -> DiscoveryControl:
    result = await session.execute(select(DiscoveryControl).where(DiscoveryControl.id == DISCOVERY_CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = DiscoveryControl(id=DISCOVERY_CONTROL_ID)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_discovery_control(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(select(DiscoveryControl).where(DiscoveryControl.id == DISCOVERY_CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "run_interval_minutes": settings.DISCOVERY_RUN_INTERVAL_MINUTES,
            "priority_backlog_mode": True,
            "requested_run_at": None,
        }
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "run_interval_minutes": int(row.run_interval_minutes or settings.DISCOVERY_RUN_INTERVAL_MINUTES),
        "priority_backlog_mode": bool(row.priority_backlog_mode if row.priority_backlog_mode is not None else True),
        "requested_run_at": row.requested_run_at,
    }


async def set_discovery_paused(session: AsyncSession, paused: bool) -> None:
    row = await ensure_discovery_control(session)
    row.is_paused = paused
    row.updated_at = utcnow()
    await session.commit()


async def set_discovery_interval(session: AsyncSession, interval_minutes: int) -> None:
    row = await ensure_discovery_control(session)
    row.run_interval_minutes = max(5, min(1440, int(interval_minutes)))
    row.updated_at = utcnow()
    await session.commit()


async def set_discovery_priority_backlog_mode(
    session: AsyncSession,
    enabled: bool,
) -> None:
    row = await ensure_discovery_control(session)
    row.priority_backlog_mode = bool(enabled)
    row.updated_at = utcnow()
    await session.commit()


async def request_one_discovery_run(session: AsyncSession) -> None:
    row = await ensure_discovery_control(session)
    row.requested_run_at = utcnow()
    row.updated_at = utcnow()
    await session.commit()


async def clear_discovery_run_request(session: AsyncSession) -> None:
    row = await ensure_discovery_control(session)
    row.requested_run_at = None
    row.updated_at = utcnow()
    await session.commit()
