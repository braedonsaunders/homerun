"""
Shared state: DB as single source of truth.
Scanner worker writes snapshot; API and other workers read from DB.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    ScannerControl,
    ScannerSnapshot,
    ScannerRun,
    OpportunityState,
    OpportunityEvent,
)
from models.opportunity import ArbitrageOpportunity, OpportunityFilter

logger = logging.getLogger(__name__)

SNAPSHOT_ID = "latest"
CONTROL_ID = "default"


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO datetime strings defensively, including legacy malformed UTC suffixes."""
    text = value.strip()

    # Handle legacy malformed values like "...+00:00+00:00".
    if text.endswith("+00:00+00:00"):
        text = text[:-6]

    # Normalize trailing Z to an explicit offset for fromisoformat().
    if text.endswith("Z"):
        text = text[:-1]

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _format_iso_utc_z(dt: Optional[datetime]) -> Optional[str]:
    """Format datetimes as canonical UTC ISO strings with a trailing Z."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + "Z"


async def write_scanner_snapshot(
    session: AsyncSession,
    opportunities: list[ArbitrageOpportunity],
    status: dict[str, Any],
    market_history: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> None:
    """Write current opportunities and status to scanner_snapshot (worker calls this)."""
    last_scan = status.get("last_scan")
    if isinstance(last_scan, str):
        try:
            last_scan = _parse_iso_datetime(last_scan)
        except Exception as e:
            logger.warning("Invalid last_scan timestamp in snapshot status: %s", e)
            last_scan = utcnow()
    elif last_scan is None:
        last_scan = utcnow()

    payload: list[dict[str, Any]] = []
    skipped = 0
    for o in opportunities:
        try:
            if hasattr(o, "model_dump"):
                payload.append(o.model_dump(mode="json"))
            else:
                payload.append(
                    ArbitrageOpportunity.model_validate(o).model_dump(mode="json")
                )
        except Exception as e:
            skipped += 1
            logger.debug("Skip unserializable opportunity in snapshot write: %s", e)
    if skipped:
        logger.warning(
            "Skipped %d/%d opportunities while writing scanner snapshot",
            skipped,
            len(opportunities),
        )
    result = await session.execute(
        select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(id=SNAPSHOT_ID)
        session.add(row)
    row.updated_at = utcnow()
    row.last_scan_at = last_scan
    row.opportunities_json = payload
    row.running = status.get("running", True)
    row.enabled = status.get("enabled", True)
    row.current_activity = status.get("current_activity")
    row.interval_seconds = status.get("interval_seconds", 60)
    row.strategies_json = status.get("strategies", [])
    row.tiered_scanning_json = status.get("tiered_scanning")
    row.ws_feeds_json = status.get("ws_feeds")
    if market_history is not None:
        row.market_history_json = market_history
    await _persist_incremental_state(session, payload, status, last_scan)
    await session.commit()


async def _persist_incremental_state(
    session: AsyncSession,
    payload: list[dict[str, Any]],
    status: dict[str, Any],
    completed_at: datetime,
) -> None:
    """Persist per-run + per-opportunity incremental state/event records."""
    # Determine scan mode for observability.
    scan_mode = "full"
    activity = (status.get("current_activity") or "").lower()
    if "fast scan" in activity:
        scan_mode = "fast"
    elif "requested" in activity:
        scan_mode = "manual"

    run = ScannerRun(
        id=uuid.uuid4().hex[:16],
        scan_mode=scan_mode,
        success=not str(status.get("current_activity", "")).lower().startswith(
            "last scan error"
        ),
        opportunity_count=len(payload),
        started_at=completed_at,
        completed_at=completed_at,
    )
    session.add(run)

    # Build stable_id -> payload map; stable_id is the lifecycle key.
    current_map: dict[str, dict[str, Any]] = {}
    for item in payload:
        stable_id = str(item.get("stable_id") or item.get("id") or "").strip()
        if stable_id:
            current_map[stable_id] = item

    current_ids = set(current_map.keys())

    # Pull active rows and any rows that appear in current payload.
    if current_ids:
        existing_rows = (
            (
                await session.execute(
                    select(OpportunityState).where(
                        or_(
                            OpportunityState.is_active == True,  # noqa: E712
                            OpportunityState.stable_id.in_(current_ids),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
    else:
        existing_rows = (
            (
                await session.execute(
                    select(OpportunityState).where(OpportunityState.is_active == True)  # noqa: E712
                )
            )
            .scalars()
            .all()
        )

    existing_by_id = {row.stable_id: row for row in existing_rows}

    # Upsert current opportunities and emit detected/updated/reactivated events.
    for stable_id, item in current_map.items():
        row = existing_by_id.get(stable_id)
        if row is None:
            row = OpportunityState(
                stable_id=stable_id,
                opportunity_json=item,
                first_seen_at=completed_at,
                last_seen_at=completed_at,
                is_active=True,
                last_run_id=run.id,
            )
            session.add(row)
            session.add(
                OpportunityEvent(
                    id=uuid.uuid4().hex[:16],
                    stable_id=stable_id,
                    run_id=run.id,
                    event_type="detected",
                    opportunity_json=item,
                    created_at=completed_at,
                )
            )
            continue

        was_active = bool(row.is_active)
        changed = row.opportunity_json != item
        row.opportunity_json = item
        row.last_seen_at = completed_at
        row.last_run_id = run.id
        row.is_active = True

        if not was_active:
            event_type = "reactivated"
        elif changed:
            event_type = "updated"
        else:
            event_type = None

        if event_type:
            session.add(
                OpportunityEvent(
                    id=uuid.uuid4().hex[:16],
                    stable_id=stable_id,
                    run_id=run.id,
                    event_type=event_type,
                    opportunity_json=item,
                    created_at=completed_at,
                )
            )

    # Any previously active row missing from current payload is now expired.
    for stable_id, row in existing_by_id.items():
        if not row.is_active:
            continue
        if stable_id in current_ids:
            continue
        row.is_active = False
        row.last_seen_at = completed_at
        row.last_run_id = run.id
        session.add(
            OpportunityEvent(
                id=uuid.uuid4().hex[:16],
                stable_id=stable_id,
                run_id=run.id,
                event_type="expired",
                opportunity_json=row.opportunity_json,
                created_at=completed_at,
            )
        )


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
        if row.current_activity == activity:
            return
        row.current_activity = activity
        row.updated_at = utcnow()
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
    market_history = (
        row.market_history_json if isinstance(row.market_history_json, dict) else {}
    )
    for d in row.opportunities_json or []:
        try:
            opp = ArbitrageOpportunity.model_validate(d)
            for market in opp.markets:
                mid = market.get("id", "")
                if mid and mid in market_history:
                    market["price_history"] = market_history.get(mid, [])
            opportunities.append(opp)
        except Exception as e:
            logger.debug("Skip invalid opportunity row: %s", e)

    status = {
        "running": row.running,
        "enabled": row.enabled,
        "interval_seconds": row.interval_seconds,
        "last_scan": _format_iso_utc_z(row.last_scan_at),
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


def _stable_id_from_opportunity_id(opportunity_id: Optional[str]) -> Optional[str]:
    """Best-effort stable_id extraction from <stable_id>_<timestamp> IDs."""
    if not opportunity_id:
        return None
    text = str(opportunity_id).strip()
    if not text:
        return None
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return text


async def update_opportunity_ai_analysis_in_snapshot(
    session: AsyncSession,
    opportunity_id: str,
    stable_id: Optional[str],
    ai_analysis: dict[str, Any],
) -> bool:
    """Persist ai_analysis into scanner snapshot + opportunity_state for one opportunity."""
    sid = (stable_id or "").strip() or _stable_id_from_opportunity_id(opportunity_id)
    oid = (opportunity_id or "").strip()
    if not oid and not sid:
        return False

    updated = False

    result = await session.execute(
        select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is not None and isinstance(row.opportunities_json, list):
        patched_payload: list[dict[str, Any]] = []
        for item in row.opportunities_json:
            if not isinstance(item, dict):
                patched_payload.append(item)
                continue
            item_id = str(item.get("id") or "").strip()
            item_sid = str(item.get("stable_id") or "").strip()
            if (oid and item_id == oid) or (sid and item_sid == sid):
                patched = dict(item)
                patched["ai_analysis"] = ai_analysis
                patched_payload.append(patched)
                updated = True
            else:
                patched_payload.append(item)
        if updated:
            row.opportunities_json = patched_payload
            row.updated_at = utcnow()

    if sid:
        state_row = await session.get(OpportunityState, sid)
        if state_row is not None and isinstance(state_row.opportunity_json, dict):
            patched_state = dict(state_row.opportunity_json)
            patched_state["ai_analysis"] = ai_analysis
            state_row.opportunity_json = patched_state
            state_row.last_seen_at = utcnow()
            updated = True

    if updated:
        await session.commit()
    return updated


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
    row.updated_at = utcnow()
    await session.commit()


async def set_scanner_interval(session: AsyncSession, interval_seconds: int) -> None:
    """Set scan interval (API)."""
    row = await ensure_scanner_control(session)
    row.scan_interval_seconds = max(10, min(3600, interval_seconds))
    row.updated_at = utcnow()
    await session.commit()


async def request_one_scan(session: AsyncSession) -> None:
    """Set requested_scan_at so worker runs one scan on next loop (API: scan now)."""
    row = await ensure_scanner_control(session)
    row.requested_scan_at = utcnow()
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
    from utils.utcnow import utcnow
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
