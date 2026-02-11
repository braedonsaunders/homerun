"""
Shared state: DB as single source of truth.
Scanner worker writes snapshot; API and other workers read from DB.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import ScannerControl, ScannerSnapshot
from models.opportunity import ArbitrageOpportunity, OpportunityFilter

logger = logging.getLogger(__name__)

SNAPSHOT_ID = "latest"
CONTROL_ID = "default"


async def write_scanner_snapshot(
    session: AsyncSession,
    opportunities: list[ArbitrageOpportunity],
    status: dict[str, Any],
) -> None:
    """Write current opportunities and status to scanner_snapshot (worker calls this)."""
    last_scan = status.get("last_scan")
    if isinstance(last_scan, str):
        last_scan = datetime.fromisoformat(last_scan.replace("Z", "+00:00"))
    elif last_scan is None:
        last_scan = datetime.utcnow()

    payload = [
        o.model_dump(mode="json") if hasattr(o, "model_dump") else o
        for o in opportunities
    ]
    result = await session.execute(
        select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(id=SNAPSHOT_ID)
        session.add(row)
    row.updated_at = datetime.utcnow()
    row.last_scan_at = last_scan
    row.opportunities_json = payload
    row.running = status.get("running", True)
    row.enabled = status.get("enabled", True)
    row.current_activity = status.get("current_activity")
    row.interval_seconds = status.get("interval_seconds", 60)
    row.strategies_json = status.get("strategies", [])
    row.tiered_scanning_json = status.get("tiered_scanning")
    row.ws_feeds_json = status.get("ws_feeds")
    await session.commit()


async def update_scanner_activity(session: AsyncSession, activity: str) -> None:
    """Update only current_activity in the snapshot (worker calls during scan for live status)."""
    result = await session.execute(
        select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(
            id=SNAPSHOT_ID,
            current_activity=activity,
            running=True,
            enabled=True,
            interval_seconds=60,
            opportunities_json=[],
        )
        session.add(row)
    else:
        row.current_activity = activity
        row.updated_at = datetime.utcnow()
    await session.commit()


async def read_scanner_snapshot(
    session: AsyncSession,
) -> tuple[list[ArbitrageOpportunity], dict[str, Any]]:
    """Read latest opportunities and status from DB. Returns (opportunities, status_dict)."""
    result = await session.execute(
        select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return [], _default_status()

    opportunities: list[ArbitrageOpportunity] = []
    for d in row.opportunities_json or []:
        try:
            opportunities.append(ArbitrageOpportunity.model_validate(d))
        except Exception as e:
            logger.debug("Skip invalid opportunity row: %s", e)

    status = {
        "running": row.running,
        "enabled": row.enabled,
        "interval_seconds": row.interval_seconds,
        "last_scan": row.last_scan_at.isoformat() + "Z" if row.last_scan_at else None,
        "opportunities_count": len(opportunities),
        "current_activity": row.current_activity,
        "strategies": row.strategies_json or [],
        "tiered_scanning": row.tiered_scanning_json,
        "ws_feeds": row.ws_feeds_json,
    }
    return opportunities, status


def _default_status() -> dict[str, Any]:
    return {
        "running": False,
        "enabled": True,
        "interval_seconds": 60,
        "last_scan": None,
        "opportunities_count": 0,
        "current_activity": "Waiting for scanner worker.",
        "strategies": [],
        "tiered_scanning": None,
        "ws_feeds": None,
    }


async def get_opportunities_from_db(
    session: AsyncSession,
    filter: Optional[OpportunityFilter] = None,
) -> list[ArbitrageOpportunity]:
    """Get current opportunities from DB with optional filter (API use)."""
    opportunities, _ = await read_scanner_snapshot(session)
    if not filter:
        return opportunities
    if filter.min_profit > 0:
        opportunities = [o for o in opportunities if o.roi_percent >= filter.min_profit * 100]
    if filter.max_risk < 1.0:
        opportunities = [o for o in opportunities if o.risk_score <= filter.max_risk]
    if filter.strategies:
        opportunities = [o for o in opportunities if o.strategy in filter.strategies]
    if filter.min_liquidity > 0:
        opportunities = [o for o in opportunities if o.min_liquidity >= filter.min_liquidity]
    if filter.category:
        cl = filter.category.lower()
        opportunities = [
            o for o in opportunities
            if o.category and o.category.lower() == cl
        ]
    return opportunities


async def get_scanner_status_from_db(session: AsyncSession) -> dict[str, Any]:
    """Get scanner status from DB (API use)."""
    _, status = await read_scanner_snapshot(session)
    return status


# ---------- Scanner control (API writes, worker reads) ----------


async def read_scanner_control(session: AsyncSession) -> dict[str, Any]:
    """Read scanner control row. Returns dict with is_enabled, is_paused, scan_interval_seconds, requested_scan_at."""
    result = await session.execute(
        select(ScannerControl).where(ScannerControl.id == CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "scan_interval_seconds": 60,
            "requested_scan_at": None,
        }
    return {
        "is_enabled": row.is_enabled,
        "is_paused": row.is_paused,
        "scan_interval_seconds": row.scan_interval_seconds,
        "requested_scan_at": row.requested_scan_at,
    }


async def ensure_scanner_control(session: AsyncSession) -> ScannerControl:
    """Ensure scanner_control row exists; return it."""
    result = await session.execute(
        select(ScannerControl).where(ScannerControl.id == CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerControl(id=CONTROL_ID)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def set_scanner_paused(session: AsyncSession, paused: bool) -> None:
    """Set scanner pause state (API: pause/resume)."""
    row = await ensure_scanner_control(session)
    row.is_paused = paused
    row.updated_at = datetime.utcnow()
    await session.commit()


async def set_scanner_interval(session: AsyncSession, interval_seconds: int) -> None:
    """Set scan interval (API)."""
    row = await ensure_scanner_control(session)
    row.scan_interval_seconds = max(10, min(3600, interval_seconds))
    row.updated_at = datetime.utcnow()
    await session.commit()


async def request_one_scan(session: AsyncSession) -> None:
    """Set requested_scan_at so worker runs one scan on next loop (API: scan now)."""
    row = await ensure_scanner_control(session)
    row.requested_scan_at = datetime.utcnow()
    await session.commit()


async def clear_scan_request(session: AsyncSession) -> None:
    """Clear requested_scan_at after worker has run (worker calls this)."""
    result = await session.execute(
        select(ScannerControl).where(ScannerControl.id == CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row and row.requested_scan_at is not None:
        row.requested_scan_at = None
        await session.commit()


async def clear_opportunities_in_snapshot(session: AsyncSession) -> int:
    """Clear opportunities in snapshot (API: clear all). Returns count cleared."""
    opportunities, status = await read_scanner_snapshot(session)
    count = len(opportunities)
    await write_scanner_snapshot(session, [], {**status, "opportunities_count": 0})
    return count


def _remove_expired_opportunities(
    opportunities: list[ArbitrageOpportunity],
) -> list[ArbitrageOpportunity]:
    """Drop opportunities whose resolution date has passed."""
    from datetime import timezone
    now = datetime.now(timezone.utc)
    out = []
    for o in opportunities:
        if o.resolution_date is None:
            out.append(o)
            continue
        rd = o.resolution_date if o.resolution_date.tzinfo else o.resolution_date.replace(tzinfo=timezone.utc)
        if rd > now:
            out.append(o)
    return out


def _remove_old_opportunities(
    opportunities: list[ArbitrageOpportunity],
    max_age_minutes: int,
) -> list[ArbitrageOpportunity]:
    """Drop opportunities older than max_age_minutes."""
    from datetime import timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    def ok(o: ArbitrageOpportunity) -> bool:
        d = o.detected_at
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d >= cutoff
    return [o for o in opportunities if ok(o)]


async def cleanup_snapshot_opportunities(
    session: AsyncSession,
    remove_expired: bool = True,
    max_age_minutes: Optional[int] = None,
) -> dict[str, int]:
    """Remove expired/old opportunities from snapshot; return counts."""
    opportunities, status = await read_scanner_snapshot(session)
    expired_removed, old_removed = 0, 0
    if remove_expired:
        before = len(opportunities)
        opportunities = _remove_expired_opportunities(opportunities)
        expired_removed = before - len(opportunities)
    if max_age_minutes:
        before = len(opportunities)
        opportunities = _remove_old_opportunities(opportunities, max_age_minutes)
        old_removed = before - len(opportunities)
    status["opportunities_count"] = len(opportunities)
    await write_scanner_snapshot(session, opportunities, status)
    return {"expired_removed": expired_removed, "old_removed": old_removed, "remaining_count": len(opportunities)}
