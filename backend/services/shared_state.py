"""
Shared state: DB as single source of truth.
Scanner worker writes snapshot; API and other workers read from DB.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    ScannerControl,
    ScannerSnapshot,
    ScannerRun,
    OpportunityState,
    OpportunityEvent,
    ScannerBatchQueue,
)
from models.opportunity import Opportunity, OpportunityFilter
from services.event_bus import event_bus
from services.market_tradability import get_market_tradability_map
from utils.utcnow import utcnow

SQL_IN_CLAUSE_CHUNK_SIZE = 900


def _chunked_in(column, values, chunk_size: int = SQL_IN_CLAUSE_CHUNK_SIZE):
    values = list(values)
    if len(values) <= chunk_size:
        return column.in_(values)
    clauses = []
    for i in range(0, len(values), chunk_size):
        clauses.append(column.in_(values[i : i + chunk_size]))
    return or_(*clauses)


logger = logging.getLogger(__name__)

SNAPSHOT_ID = "latest"
TRADERS_SNAPSHOT_ID = "traders_latest"
CONTROL_ID = "default"
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BASE_DELAY_SECONDS = 0.05
DB_RETRY_MAX_DELAY_SECONDS = 0.3
SCANNER_BATCH_LEASE_MIN_SECONDS = 5
SCANNER_BATCH_LEASE_MAX_SECONDS = 300

# In-memory targeted condition IDs for the next scan request.
# Set by the evaluate endpoint, consumed and cleared by the scanner worker.
_pending_targeted_condition_ids: list[str] = []


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


def _is_retryable_db_error(exc: Exception) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    return any(
        marker in message
        for marker in (
            "deadlock detected",
            "serialization failure",
            "could not serialize access",
            "lock not available",
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


async def _commit_with_retry(session: AsyncSession) -> None:
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            await session.commit()
            return
        except (OperationalError, InterfaceError) as exc:
            await session.rollback()
            is_locked = _is_retryable_db_error(exc)
            is_last = attempt >= DB_RETRY_ATTEMPTS - 1
            if not is_locked or is_last:
                raise
            await asyncio.sleep(_db_retry_delay(attempt))


def _normalize_weather_edge_title(title: str) -> str:
    prefix = "weather edge:"
    return title[len(prefix) :].lstrip() if title.lower().startswith(prefix) else title


def _serialize_opportunity_payload(opportunities: list[Opportunity]) -> tuple[list[dict[str, Any]], int]:
    payload: list[dict[str, Any]] = []
    skipped = 0
    for opportunity in opportunities:
        try:
            if hasattr(opportunity, "model_dump"):
                payload.append(opportunity.model_dump(mode="json"))
            else:
                payload.append(Opportunity.model_validate(opportunity).model_dump(mode="json"))
        except Exception as exc:
            skipped += 1
            logger.debug("Skip unserializable opportunity payload row: %s", exc)
    return payload, skipped


async def write_scanner_snapshot(
    session: AsyncSession,
    opportunities: list[Opportunity],
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

    payload, skipped = _serialize_opportunity_payload(opportunities)
    if skipped:
        logger.warning(
            "Skipped %d/%d opportunities while writing scanner snapshot",
            skipped,
            len(opportunities),
        )
    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
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
    await _commit_with_retry(session)

    # Publish events so the broadcaster can relay immediately.
    try:
        scanner_status = {
            "running": status.get("running", True),
            "enabled": status.get("enabled", True),
            "interval_seconds": status.get("interval_seconds", 60),
            "last_scan": status.get("last_scan"),
            "last_fast_scan": status.get("last_fast_scan"),
            "last_heavy_scan": status.get("last_heavy_scan"),
            "opportunities_count": len(opportunities),
            "current_activity": status.get("current_activity"),
            "lane_watchdogs": status.get("lane_watchdogs"),
            "strategies": status.get("strategies", []),
            "tiered_scanning": status.get("tiered_scanning"),
            "ws_feeds": status.get("ws_feeds"),
        }
        await event_bus.publish("scanner_status", scanner_status)
        await event_bus.publish(
            "scanner_activity",
            {"activity": status.get("current_activity") or "Idle"},
        )
        await event_bus.publish(
            "opportunities_update",
            {
                "count": len(opportunities),
                "source": "scanner_snapshot_write",
            },
        )
    except Exception:
        pass  # fire-and-forget


# ---------------------------------------------------------------------------
# Market catalog persistence (events + markets from upstream APIs)
# ---------------------------------------------------------------------------

CATALOG_ID = "latest"


async def write_market_catalog(
    session: AsyncSession,
    events: list,
    markets: list,
    duration_seconds: float = 0.0,
    error: str | None = None,
) -> None:
    """Persist the upstream market catalog (events + markets) to DB."""
    from models.database import MarketCatalog

    events_payload = []
    for e in events:
        try:
            events_payload.append(e.model_dump(mode="json") if hasattr(e, "model_dump") else e)
        except Exception:
            pass

    markets_payload = []
    for m in markets:
        try:
            markets_payload.append(m.model_dump(mode="json") if hasattr(m, "model_dump") else m)
        except Exception:
            pass

    result = await session.execute(select(MarketCatalog).where(MarketCatalog.id == CATALOG_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = MarketCatalog(id=CATALOG_ID)
        session.add(row)

    row.updated_at = utcnow()
    row.events_json = events_payload
    row.markets_json = markets_payload
    row.event_count = len(events_payload)
    row.market_count = len(markets_payload)
    row.fetch_duration_seconds = duration_seconds
    row.error = error
    await _commit_with_retry(session)


async def read_market_catalog(
    session: AsyncSession,
) -> tuple[list, list, dict[str, Any]]:
    """Read persisted market catalog. Returns (events, markets, metadata)."""
    from models.database import MarketCatalog
    from models.market import Event, Market

    result = await session.execute(select(MarketCatalog).where(MarketCatalog.id == CATALOG_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return [], [], {"updated_at": None, "error": None}

    events = []
    for d in row.events_json or []:
        try:
            events.append(Event.model_validate(d))
        except Exception:
            pass

    markets = []
    for d in row.markets_json or []:
        try:
            markets.append(Market.model_validate(d))
        except Exception:
            pass

    metadata: dict[str, Any] = {
        "updated_at": row.updated_at,
        "event_count": row.event_count,
        "market_count": row.market_count,
        "fetch_duration_seconds": row.fetch_duration_seconds,
        "error": row.error,
    }
    return events, markets, metadata


async def write_traders_snapshot(
    session: AsyncSession,
    opportunities: list[Opportunity],
    status: dict[str, Any],
) -> None:
    """Write trader opportunities into dedicated snapshot storage."""
    last_scan = status.get("last_scan")
    if isinstance(last_scan, str):
        try:
            last_scan = _parse_iso_datetime(last_scan)
        except Exception:
            last_scan = utcnow()
    elif last_scan is None:
        last_scan = utcnow()

    payload: list[dict[str, Any]] = []
    skipped = 0
    for o in opportunities:
        try:
            item = (
                o.model_dump(mode="json")
                if hasattr(o, "model_dump")
                else Opportunity.model_validate(o).model_dump(mode="json")
            )
            if isinstance(item.get("strategy_context"), dict):
                item["strategy_context"]["source_key"] = "traders"
            else:
                item["strategy_context"] = {"source_key": "traders"}
            payload.append(item)
        except Exception:
            skipped += 1

    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == TRADERS_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(id=TRADERS_SNAPSHOT_ID)
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
    await _commit_with_retry(session)

    logger.info(
        "Wrote traders snapshot: opportunities=%s skipped=%s running=%s enabled=%s",
        len(payload),
        skipped,
        bool(row.running),
        bool(row.enabled),
    )
    try:
        await event_bus.publish(
            "opportunities_update",
            {
                "count": len(payload),
                "source": "traders_snapshot_write",
            },
        )
    except Exception:
        pass


async def enqueue_scanner_batch(
    session: AsyncSession,
    opportunities: list[Opportunity],
    status: dict[str, Any],
    *,
    source: str = "scanner",
    batch_kind: str = "scan_cycle",
) -> str | None:
    """Enqueue a scanner detection batch for aggregation worker processing."""
    payload, skipped = _serialize_opportunity_payload(opportunities)
    if skipped:
        logger.warning(
            "Skipped %d/%d opportunities while enqueueing scanner batch",
            skipped,
            len(opportunities),
        )
    if not payload and opportunities:
        return None

    batch_id = uuid.uuid4().hex[:16]
    row = ScannerBatchQueue(
        id=batch_id,
        source=str(source or "scanner"),
        batch_kind=str(batch_kind or "scan_cycle"),
        opportunities_json=payload,
        status_json=dict(status or {}),
        emitted_at=utcnow(),
        lease_owner=None,
        lease_expires_at=None,
        attempt_count=0,
        processed_at=None,
        error=None,
    )
    session.add(row)
    await _commit_with_retry(session)
    return batch_id


async def claim_next_scanner_batch(
    session: AsyncSession,
    *,
    owner: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    """Claim the next unprocessed scanner batch and acquire a short lease."""
    now = utcnow()
    lease_ttl = max(
        SCANNER_BATCH_LEASE_MIN_SECONDS,
        min(SCANNER_BATCH_LEASE_MAX_SECONDS, int(lease_seconds or SCANNER_BATCH_LEASE_MIN_SECONDS)),
    )
    lease_until = now + timedelta(seconds=lease_ttl)

    stmt = (
        select(ScannerBatchQueue)
        .where(ScannerBatchQueue.processed_at.is_(None))
        .where(
            (ScannerBatchQueue.lease_expires_at.is_(None))
            | (ScannerBatchQueue.lease_expires_at < now)
        )
        .order_by(ScannerBatchQueue.emitted_at.asc(), ScannerBatchQueue.id.asc())
        .limit(1)
    )
    bind = session.get_bind()
    if bind is not None and str(getattr(bind.dialect, "name", "") or "").lower() == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    result = await session.execute(stmt)
    row = result.scalars().first()
    if row is None:
        return None

    row.lease_owner = str(owner or "")
    row.lease_expires_at = lease_until
    row.attempt_count = int(row.attempt_count or 0) + 1
    row.error = None
    await _commit_with_retry(session)
    return {
        "id": row.id,
        "source": row.source,
        "batch_kind": row.batch_kind,
        "opportunities_json": list(row.opportunities_json or []),
        "status_json": dict(row.status_json or {}),
        "emitted_at": row.emitted_at,
        "attempt_count": int(row.attempt_count or 0),
    }


async def mark_scanner_batch_processed(
    session: AsyncSession,
    batch_id: str,
) -> None:
    """Mark a claimed scanner batch as fully processed."""
    row = await session.get(ScannerBatchQueue, str(batch_id or "").strip())
    if row is None:
        return
    row.processed_at = utcnow()
    row.lease_owner = None
    row.lease_expires_at = None
    row.error = None
    await _commit_with_retry(session)


async def mark_scanner_batch_failed(
    session: AsyncSession,
    batch_id: str,
    *,
    error: str,
    requeue: bool,
) -> None:
    """Record scanner batch failure and optionally requeue for retry."""
    row = await session.get(ScannerBatchQueue, str(batch_id or "").strip())
    if row is None:
        return
    row.error = str(error or "")[:2000]
    if requeue:
        row.lease_owner = None
        row.lease_expires_at = None
        row.processed_at = None
    else:
        row.lease_owner = None
        row.lease_expires_at = None
        row.processed_at = utcnow()
    await _commit_with_retry(session)


async def count_pending_scanner_batches(session: AsyncSession) -> int:
    value = await session.execute(
        select(func.count())
        .select_from(ScannerBatchQueue)
        .where(ScannerBatchQueue.processed_at.is_(None))
    )
    return int(value.scalar_one() or 0)


async def cleanup_processed_scanner_batches(
    session: AsyncSession,
    *,
    retain_hours: int = 24,
) -> int:
    """Delete old processed scanner batches to keep queue table bounded."""
    cutoff = utcnow() - timedelta(hours=max(1, int(retain_hours or 24)))
    result = await session.execute(
        delete(ScannerBatchQueue).where(
            ScannerBatchQueue.processed_at.is_not(None),
            ScannerBatchQueue.processed_at < cutoff,
        )
    )
    await _commit_with_retry(session)
    return int(result.rowcount or 0)


async def drop_oldest_pending_scanner_batch(session: AsyncSession) -> str | None:
    """Drop the oldest unprocessed scanner batch (backpressure safety valve)."""
    now = utcnow()
    stmt = (
        select(ScannerBatchQueue)
        .where(ScannerBatchQueue.processed_at.is_(None))
        .where(
            (ScannerBatchQueue.lease_expires_at.is_(None))
            | (ScannerBatchQueue.lease_expires_at < now)
        )
        .order_by(ScannerBatchQueue.emitted_at.asc(), ScannerBatchQueue.id.asc())
        .limit(1)
    )
    bind = session.get_bind()
    if bind is not None and str(getattr(bind.dialect, "name", "") or "").lower() == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    row = (await session.execute(stmt)).scalars().first()
    if row is None:
        return None
    dropped_id = str(row.id)
    await session.delete(row)
    await _commit_with_retry(session)
    return dropped_id


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
        success=not str(status.get("current_activity", "")).lower().startswith("last scan error"),
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
                            _chunked_in(OpportunityState.stable_id, current_ids),
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
        incoming_item = item if isinstance(item, dict) else dict(item)
        row = existing_by_id.get(stable_id)
        if row is None:
            incoming_item["revision"] = 1
            row = OpportunityState(
                stable_id=stable_id,
                opportunity_json=incoming_item,
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
                    opportunity_json=incoming_item,
                    created_at=completed_at,
                )
            )
            continue

        was_active = bool(row.is_active)
        previous_payload = row.opportunity_json if isinstance(row.opportunity_json, dict) else {}
        previous_revision = int(previous_payload.get("revision") or 0)
        changed = row.opportunity_json != incoming_item
        if changed:
            incoming_item["revision"] = max(1, previous_revision + 1)
        else:
            incoming_item["revision"] = max(1, previous_revision)
        row.opportunity_json = incoming_item
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
                    opportunity_json=incoming_item,
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
    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
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
    await _commit_with_retry(session)

    # Publish activity change event.
    try:
        await event_bus.publish("scanner_activity", {"activity": activity})
    except Exception:
        pass  # fire-and-forget


async def read_scanner_snapshot(
    session: AsyncSession,
) -> tuple[list[Opportunity], dict[str, Any]]:
    """Read latest opportunities and status from DB. Returns (opportunities, status_dict)."""
    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return [], _default_status()

    opportunities: list[Opportunity] = []
    market_history = row.market_history_json if isinstance(row.market_history_json, dict) else {}
    for d in row.opportunities_json or []:
        try:
            opp = Opportunity.model_validate(d)
            for market in opp.markets:
                candidates = (
                    str(market.get("id", "") or "").strip(),
                    str(market.get("condition_id", "") or "").strip(),
                    str(market.get("conditionId", "") or "").strip(),
                )
                for candidate in candidates:
                    if not candidate:
                        continue
                    history = market_history.get(candidate)
                    if isinstance(history, list) and len(history) >= 2:
                        market["price_history"] = history
                        break
            opportunities.append(opp)
        except Exception as e:
            logger.debug("Skip invalid opportunity row: %s", e)

    tiered = row.tiered_scanning_json if isinstance(row.tiered_scanning_json, dict) else {}
    status = {
        "running": row.running,
        "enabled": row.enabled,
        "interval_seconds": row.interval_seconds,
        "last_scan": _format_iso_utc_z(row.last_scan_at),
        "last_fast_scan": tiered.get("last_fast_scan"),
        "last_heavy_scan": tiered.get("last_heavy_scan") or tiered.get("last_full_snapshot_strategy_scan"),
        "opportunities_count": len(opportunities),
        "current_activity": row.current_activity,
        "lane_watchdogs": tiered.get("lane_watchdogs"),
        "strategies": row.strategies_json or [],
        "tiered_scanning": row.tiered_scanning_json,
        "ws_feeds": row.ws_feeds_json,
    }
    return opportunities, status


async def read_scanner_status(
    session: AsyncSession,
    *,
    include_opportunity_count: bool = True,
) -> dict[str, Any]:
    """Read scanner status without deserializing opportunity payloads."""
    result = await session.execute(
        select(
            ScannerSnapshot.running,
            ScannerSnapshot.enabled,
            ScannerSnapshot.interval_seconds,
            ScannerSnapshot.last_scan_at,
            ScannerSnapshot.current_activity,
            ScannerSnapshot.strategies_json,
            ScannerSnapshot.tiered_scanning_json,
            ScannerSnapshot.ws_feeds_json,
        ).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.one_or_none()
    if row is None:
        return _default_status()

    opportunities_count = 0
    if include_opportunity_count:
        opportunities_count = int(
            (
                await session.execute(
                    select(func.count()).select_from(OpportunityState).where(OpportunityState.is_active == True)  # noqa: E712
                )
            ).scalar_one()
            or 0
        )

    tiered = row.tiered_scanning_json if isinstance(row.tiered_scanning_json, dict) else {}
    return {
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "interval_seconds": int(row.interval_seconds or 60),
        "last_scan": _format_iso_utc_z(row.last_scan_at),
        "last_fast_scan": tiered.get("last_fast_scan"),
        "last_heavy_scan": tiered.get("last_heavy_scan") or tiered.get("last_full_snapshot_strategy_scan"),
        "opportunities_count": opportunities_count,
        "current_activity": row.current_activity,
        "lane_watchdogs": tiered.get("lane_watchdogs"),
        "strategies": row.strategies_json or [],
        "tiered_scanning": row.tiered_scanning_json,
        "ws_feeds": row.ws_feeds_json,
    }


async def read_traders_snapshot(
    session: AsyncSession,
) -> tuple[list[Opportunity], dict[str, Any]]:
    """Read latest trader opportunities and status from dedicated snapshot row."""
    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == TRADERS_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return [], _default_status()

    market_history_row = (
        await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
    ).scalar_one_or_none()
    market_history = (
        market_history_row.market_history_json
        if market_history_row is not None and isinstance(market_history_row.market_history_json, dict)
        else {}
    )

    opportunities: list[Opportunity] = []
    for d in row.opportunities_json or []:
        try:
            opp = Opportunity.model_validate(d)
            if isinstance(market_history, dict):
                for market in opp.markets:
                    if not isinstance(market, dict):
                        continue
                    candidates = {
                        str(market.get("id", "") or "").strip(),
                        str(market.get("condition_id", "") or "").strip(),
                    }
                    for candidate in candidates:
                        if not candidate:
                            continue
                        history = market_history.get(candidate)
                        if isinstance(history, list):
                            market["price_history"] = history
                            break
            if isinstance(opp.strategy_context, dict):
                opp.strategy_context["source_key"] = "traders"
            else:
                opp.strategy_context = {"source_key": "traders"}
            opportunities.append(opp)
        except Exception as e:
            logger.debug("Skip invalid trader opportunity row: %s", e)

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
        "last_fast_scan": None,
        "last_heavy_scan": None,
        "opportunities_count": 0,
        "current_activity": "Waiting for scanner worker.",
        "lane_watchdogs": None,
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


def _is_traders_opportunity(opp: Opportunity) -> bool:
    context = opp.strategy_context if isinstance(opp.strategy_context, dict) else {}
    return str(context.get("source_key") or "").strip().lower() == "traders"


def _market_ids_from_opportunity(opp: Opportunity) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    for market in opp.markets or []:
        if not isinstance(market, dict):
            continue
        mid = str(market.get("id") or market.get("condition_id") or "").strip().lower()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)

    for position in opp.positions_to_take or []:
        if not isinstance(position, dict):
            continue
        mid = str(position.get("market_id") or position.get("market") or position.get("id") or "").strip().lower()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)

    return ids


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

    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
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
        await _commit_with_retry(session)
    return updated


async def get_opportunities_from_db(
    session: AsyncSession,
    filter: Optional[OpportunityFilter] = None,
    source: str = "markets",
) -> list[Opportunity]:
    """Get current opportunities from DB with optional filter (API use)."""
    source_key = str(source or "markets").strip().lower()
    if source_key == "traders":
        opportunities, _ = await read_traders_snapshot(session)
    elif source_key == "all":
        market_opps, _ = await read_scanner_snapshot(session)
        trader_opps, _ = await read_traders_snapshot(session)
        opportunities = list(market_opps) + list(trader_opps)
    else:
        opportunities, _ = await read_scanner_snapshot(session)

    # Release the DB connection before price overlays/tradability checks,
    # which may perform network or cache I/O.
    try:
        if session.in_transaction():
            await session.rollback()
    except Exception:
        pass

    for opp in opportunities:
        opp.title = _normalize_weather_edge_title(opp.title)

    if opportunities:
        try:
            from services.scanner import scanner as market_scanner

            opportunities = await market_scanner.refresh_opportunity_prices(
                opportunities,
                drop_stale=False,
            )
        except Exception:
            pass

    if opportunities:
        history_candidates: list[Opportunity] = []
        seen_ids: set[str] = set()
        for opp in opportunities:
            has_missing_market_history = False
            for market in opp.markets:
                history = market.get("price_history")
                if not isinstance(history, list) or len(history) < 2:
                    has_missing_market_history = True
                    break
            if not has_missing_market_history:
                continue
            candidate_id = str(getattr(opp, "stable_id", "") or getattr(opp, "id", "") or "").strip()
            if candidate_id and candidate_id in seen_ids:
                continue
            if candidate_id:
                seen_ids.add(candidate_id)
            history_candidates.append(opp)

        if history_candidates:
            try:
                from services.scanner import scanner as market_scanner

                await market_scanner.attach_price_history_to_opportunities(
                    history_candidates,
                    timeout_seconds=0.0,
                )
            except Exception:
                pass

    apply_tradability_filter = source_key in {"markets", "all"}
    if opportunities and apply_tradability_filter:
        by_index: dict[int, list[str]] = {}
        all_market_ids: set[str] = set()
        for idx, opp in enumerate(opportunities):
            if _is_traders_opportunity(opp):
                by_index[idx] = []
                continue
            market_ids = _market_ids_from_opportunity(opp)
            by_index[idx] = market_ids
            all_market_ids.update(market_ids)

        if all_market_ids:
            tradability = await get_market_tradability_map(all_market_ids)
            opportunities = [
                opp
                for idx, opp in enumerate(opportunities)
                if _is_traders_opportunity(opp) or all(tradability.get(mid, True) for mid in by_index.get(idx, []))
            ]

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
        opportunities = [o for o in opportunities if o.category and o.category.lower() == cl]
    return opportunities


async def get_scanner_status_from_db(session: AsyncSession) -> dict[str, Any]:
    """Get scanner status from DB (API use)."""
    return await read_scanner_status(session)


# ---------- Scanner control (API writes, worker reads) ----------


async def read_scanner_control(session: AsyncSession) -> dict[str, Any]:
    """Read scanner control row. Returns dict with is_enabled, is_paused, scan_interval_seconds, requested_scan_at."""
    result = await session.execute(select(ScannerControl).where(ScannerControl.id == CONTROL_ID))
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
    result = await session.execute(select(ScannerControl).where(ScannerControl.id == CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerControl(id=CONTROL_ID)
        session.add(row)
        await _commit_with_retry(session)
        await session.refresh(row)
    return row


async def set_scanner_paused(session: AsyncSession, paused: bool) -> None:
    """Set scanner pause state (API: pause/resume)."""
    row = await ensure_scanner_control(session)
    row.is_paused = paused
    row.updated_at = utcnow()
    await _commit_with_retry(session)


async def set_scanner_interval(session: AsyncSession, interval_seconds: int) -> None:
    """Set scan interval (API)."""
    row = await ensure_scanner_control(session)
    row.scan_interval_seconds = max(10, min(3600, interval_seconds))
    row.updated_at = utcnow()
    await _commit_with_retry(session)


async def request_one_scan(
    session: AsyncSession,
    condition_ids: list[str] | None = None,
) -> None:
    """Set requested_scan_at so worker runs one scan on next loop (API: scan now).

    If *condition_ids* is provided, the scan will prioritise those markets
    instead of running a full untargeted scan.
    """
    global _pending_targeted_condition_ids
    row = await ensure_scanner_control(session)
    row.requested_scan_at = utcnow()
    if condition_ids:
        _pending_targeted_condition_ids = list(condition_ids)
    await _commit_with_retry(session)


def pop_targeted_condition_ids() -> list[str]:
    """Return and clear any pending targeted condition IDs for the next scan."""
    global _pending_targeted_condition_ids
    ids = _pending_targeted_condition_ids
    _pending_targeted_condition_ids = []
    return ids


async def clear_scan_request(session: AsyncSession) -> None:
    """Clear requested_scan_at after worker has run (worker calls this)."""
    result = await session.execute(select(ScannerControl).where(ScannerControl.id == CONTROL_ID))
    row = result.scalar_one_or_none()
    if row and row.requested_scan_at is not None:
        row.requested_scan_at = None
        await _commit_with_retry(session)


async def clear_opportunities_in_snapshot(session: AsyncSession) -> int:
    """Clear opportunities in snapshot (API: clear all). Returns count cleared."""
    opportunities, status = await read_scanner_snapshot(session)
    count = len(opportunities)
    await write_scanner_snapshot(session, [], {**status, "opportunities_count": 0})
    return count


def _remove_expired_opportunities(
    opportunities: list[Opportunity],
) -> list[Opportunity]:
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
    opportunities: list[Opportunity],
    max_age_minutes: int,
) -> list[Opportunity]:
    """Drop opportunities older than max_age_minutes."""
    from datetime import timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    def ok(o: Opportunity) -> bool:
        d = o.last_detected_at or o.last_seen_at or o.detected_at
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
