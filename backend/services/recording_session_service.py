"""Recording-session service.

A recording session is a user-defined, scoped market-data capture.
The session row carries the spec (markets, capture types, tick
interval, time window).  The actual rows still land in
``MarketMicrostructureSnapshot`` / ``BookDeltaEvent`` — those tables
are populated by the always-on WebSocket recorder regardless.  The
session pins the rows it "owns" by ``(target_token_ids, started_at,
ended_at)`` so the unified backtester can replay just that slice
without a separate datastore.

What the service does:

  * Resolve operator targets (token_id / condition_id / event slug)
    into a concrete ``target_token_ids`` list using OutcomeResolver.
  * Subscribe the orchestrator's WebSocket feed to the target tokens
    if they're not already live (best-effort; if subscribe is
    unavailable, the session still records whatever rows happen to
    land for those tokens).
  * Track ``started_at`` / ``ended_at`` and ``rows_captured`` (read
    from the DB on a refresh tick).
  * Auto-stop the session at ``scheduled_end_at`` or after
    ``max_duration_seconds`` since started_at, whichever comes first.
  * Mark sessions ``completed`` / ``failed`` accordingly.

The service runs as part of the existing host worker loop — a single
async task ticks every ``_LOOP_INTERVAL_SECONDS`` and progresses any
running / scheduled sessions.  See ``host.py`` registration.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    BookDeltaEvent,
    MarketMicrostructureSnapshot,
    RecordingSession,
)

logger = logging.getLogger("recording_session_service")


_LOOP_INTERVAL_SECONDS = 5.0


# ── Helpers ─────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionSpec:
    name: str
    description: str | None = None
    platform: str = "polymarket"
    target_kind: str = "token"  # token | condition | event
    target_values: list[str] | None = None
    capture_types: list[str] | None = None  # subset of book/trade/delta
    tick_interval_ms: int = 500
    retention_days: int | None = None
    scheduled_start_at: datetime | None = None
    scheduled_end_at: datetime | None = None
    max_duration_seconds: int | None = None
    config: dict[str, Any] | None = None


# ── Public surface ─────────────────────────────────────────────────────


async def create_session(spec: SessionSpec) -> RecordingSession:
    """Create a new recording session in ``pending`` (or ``scheduled``)
    state.  Resolves target tokens up-front when possible.
    """
    target_tokens = await _resolve_target_tokens(
        kind=spec.target_kind, values=spec.target_values or []
    )

    status = "scheduled" if spec.scheduled_start_at else "pending"
    row = RecordingSession(
        id=uuid.uuid4().hex,
        name=spec.name,
        description=spec.description,
        status=status,
        platform=spec.platform,
        target_kind=spec.target_kind,
        target_values_json=list(spec.target_values or []),
        target_token_ids_json=target_tokens,
        capture_types_json=list(spec.capture_types or ["book", "trade"]),
        tick_interval_ms=int(spec.tick_interval_ms),
        retention_days=spec.retention_days,
        scheduled_start_at=spec.scheduled_start_at,
        scheduled_end_at=spec.scheduled_end_at,
        max_duration_seconds=spec.max_duration_seconds,
        config_json=spec.config or None,
    )
    async with AsyncSessionLocal() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    logger.info(
        "Created recording session id=%s name=%s status=%s tokens=%d",
        row.id, row.name, row.status, len(target_tokens or []),
    )
    return row


async def start_session(session_id: str) -> RecordingSession:
    """Manually transition a session to ``running``."""
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(RecordingSession).where(RecordingSession.id == session_id)
        )).scalar_one_or_none()
        if row is None:
            raise ValueError(f"Recording session {session_id} not found")
        if row.status in {"completed", "cancelled", "failed"}:
            raise ValueError(f"Session {session_id} is terminal ({row.status})")
        await _activate_session(session, row)
        await session.commit()
        await session.refresh(row)
    return row


async def stop_session(
    session_id: str, *, status: str = "completed", error: str | None = None,
) -> RecordingSession:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(RecordingSession).where(RecordingSession.id == session_id)
        )).scalar_one_or_none()
        if row is None:
            raise ValueError(f"Recording session {session_id} not found")
        await _terminate_session(session, row, status=status, error=error)
        await session.commit()
        await session.refresh(row)
    return row


async def list_sessions(
    *, statuses: list[str] | None = None, limit: int = 100,
) -> list[RecordingSession]:
    async with AsyncSessionLocal() as session:
        stmt = select(RecordingSession).order_by(RecordingSession.created_at.desc())
        if statuses:
            stmt = stmt.where(RecordingSession.status.in_(statuses))
        stmt = stmt.limit(int(limit))
        rows = (await session.execute(stmt)).scalars().all()
    return list(rows)


async def get_session(session_id: str) -> RecordingSession | None:
    async with AsyncSessionLocal() as session:
        return (await session.execute(
            select(RecordingSession).where(RecordingSession.id == session_id)
        )).scalar_one_or_none()


async def delete_session(session_id: str) -> bool:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(RecordingSession).where(RecordingSession.id == session_id)
        )).scalar_one_or_none()
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
    return True


# ── Loop ────────────────────────────────────────────────────────────────


async def session_loop_tick() -> None:
    """One iteration of the session manager.

    1. Promote ``scheduled`` sessions whose ``scheduled_start_at`` has
       passed into ``running``.
    2. For ``running`` sessions, refresh ``rows_captured`` + check
       termination conditions (``scheduled_end_at`` /
       ``max_duration_seconds``).
    """
    now = _utcnow()
    async with AsyncSessionLocal() as session:
        # 1. Activate scheduled sessions whose start time has passed.
        scheduled = (await session.execute(
            select(RecordingSession).where(
                and_(
                    RecordingSession.status == "scheduled",
                    RecordingSession.scheduled_start_at <= now,
                )
            )
        )).scalars().all()
        for row in scheduled:
            await _activate_session(session, row)

        # 2. Tick running sessions.
        running = (await session.execute(
            select(RecordingSession).where(RecordingSession.status == "running")
        )).scalars().all()
        for row in running:
            await _tick_running(session, row, now=now)

        await session.commit()


async def _activate_session(session: AsyncSession, row: RecordingSession) -> None:
    if row.target_token_ids_json is None:
        try:
            row.target_token_ids_json = await _resolve_target_tokens(
                kind=row.target_kind, values=list(row.target_values_json or []),
            )
        except Exception as exc:
            logger.warning("Token resolution failed for session %s: %s", row.id, exc)
    row.status = "running"
    row.started_at = _utcnow()
    row.error = None
    # Best-effort: subscribe to the target tokens.  No-op if the
    # orchestrator already has them (the most common case for
    # passively-recorded prediction markets).
    try:
        from services.ws_feeds import get_ws_feeds  # type: ignore

        feeds = get_ws_feeds()
        if feeds is not None and hasattr(feeds, "subscribe_tokens"):
            tokens = list(row.target_token_ids_json or [])
            if tokens:
                await feeds.subscribe_tokens(tokens)
    except Exception as exc:
        logger.debug("ws subscribe skipped for session %s: %s", row.id, exc)


async def _tick_running(
    session: AsyncSession, row: RecordingSession, *, now: datetime,
) -> None:
    # Refresh rows_captured by counting rows in the snapshot/delta tables
    # within the session's window for its target tokens.
    rows = await _count_session_rows(session=session, row=row, until=now)
    row.rows_captured = rows.get("total", 0)
    if rows.get("last_at") is not None:
        row.last_capture_at = rows["last_at"]

    # Termination conditions
    end_by_schedule = (
        row.scheduled_end_at is not None and now >= row.scheduled_end_at
    )
    end_by_duration = (
        row.max_duration_seconds is not None
        and row.started_at is not None
        and (now - row.started_at).total_seconds() >= row.max_duration_seconds
    )
    if end_by_schedule or end_by_duration:
        await _terminate_session(session, row, status="completed")


async def _terminate_session(
    session: AsyncSession,
    row: RecordingSession,
    *,
    status: str,
    error: str | None = None,
) -> None:
    row.status = status
    row.ended_at = _utcnow()
    if error is not None:
        row.error = error


async def _count_session_rows(
    *, session: AsyncSession, row: RecordingSession, until: datetime,
) -> dict[str, Any]:
    tokens = list(row.target_token_ids_json or [])
    if not tokens or row.started_at is None:
        return {"total": 0, "last_at": None}
    total = 0
    last_at: datetime | None = None
    capture_types = set(row.capture_types_json or [])
    if "book" in capture_types or "trade" in capture_types:
        types = []
        if "book" in capture_types:
            types.append("book")
        if "trade" in capture_types:
            types.append("trade")
        stmt = select(
            func.count(MarketMicrostructureSnapshot.id),
            func.max(MarketMicrostructureSnapshot.observed_at),
        ).where(
            and_(
                MarketMicrostructureSnapshot.token_id.in_(tokens),
                MarketMicrostructureSnapshot.observed_at >= row.started_at,
                MarketMicrostructureSnapshot.observed_at <= until,
                MarketMicrostructureSnapshot.snapshot_type.in_(types),
            )
        )
        n, t = (await session.execute(stmt)).one()
        total += int(n or 0)
        if t is not None and (last_at is None or t > last_at):
            last_at = t
    if "delta" in capture_types:
        stmt = select(
            func.count(BookDeltaEvent.id),
            func.max(BookDeltaEvent.observed_at),
        ).where(
            and_(
                BookDeltaEvent.token_id.in_(tokens),
                BookDeltaEvent.observed_at >= row.started_at,
                BookDeltaEvent.observed_at <= until,
            )
        )
        n, t = (await session.execute(stmt)).one()
        total += int(n or 0)
        if t is not None and (last_at is None or t > last_at):
            last_at = t
    return {"total": total, "last_at": last_at}


async def _resolve_target_tokens(*, kind: str, values: list[str]) -> list[str]:
    """Resolve operator-supplied targets into concrete token_ids.

    ``token`` — passthrough (already token_ids)
    ``condition`` — look up market_id in MarketCatalog and expand to
                    its outcome token_ids
    ``event`` — match by event_slug across MarketCatalog

    Falls back to passthrough on any error so the session still has a
    list to record against (rather than crashing the create).
    """
    cleaned = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not cleaned:
        return []
    if kind == "token":
        return cleaned
    try:
        from services.backtest.outcome_resolver import get_outcome_resolver

        resolver = get_outcome_resolver()
        await resolver._ensure_index()  # noqa: SLF001 — same package
        out: list[str] = []
        if kind == "condition":
            for value in cleaned:
                rec = await resolver.market_record(value)
                if rec is not None:
                    out.extend(rec.token_ids)
        elif kind == "event":
            # Walk the index for every market with this event slug.
            idx = resolver._index  # noqa: SLF001
            for rec in idx.by_market.values():
                if rec.event_slug and rec.event_slug in cleaned:
                    out.extend(rec.token_ids)
        # Dedupe + return
        return sorted({t for t in out if t})
    except Exception as exc:
        logger.warning("Target token resolution failed (kind=%s): %s", kind, exc)
        return cleaned


# ── Backtester integration ─────────────────────────────────────────────


async def get_protected_token_windows() -> list[dict[str, Any]]:
    """Return (token_ids, started_at, ended_at) tuples for every
    session whose captured rows must NOT be auto-pruned.

    Protection contract: any maintenance job that bulk-deletes from
    ``MarketMicrostructureSnapshot`` / ``BookDeltaEvent`` MUST exclude
    rows whose ``(token_id, observed_at)`` falls inside any returned
    window.  Sessions explicitly stand in for "the operator (or a
    scheduled job) wanted these specific rows kept" — losing them to
    a default retention policy would silently corrupt every
    backtest, ML training run, and replay scoped to that session.

    Cancelled / failed sessions are NOT protected (the rows weren't
    a deliberate capture in the end).  Pending / scheduled sessions
    have no captured rows yet so there's nothing to protect — they
    return an empty token list and the caller can skip them.

    Cheap query — single SELECT with a status filter.  Cache for
    a few minutes if you call it on a hot loop.
    """
    PROTECTED_STATUSES = ("running", "paused", "completed")
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(RecordingSession).where(
                RecordingSession.status.in_(PROTECTED_STATUSES)
            )
        )).scalars().all()
    out: list[dict[str, Any]] = []
    for r in rows:
        tokens = list(r.target_token_ids_json or [])
        if not tokens or r.started_at is None:
            continue
        out.append({
            "session_id": r.id,
            "session_name": r.name,
            "token_ids": tokens,
            "started_at": r.started_at,
            "ended_at": r.ended_at,  # may be None for running sessions — protect to "now"
            "status": r.status,
        })
    return out


async def session_training_scope(session_id: str) -> dict[str, Any] | None:
    """Translate a session into ML adapter training filters.

    Adapters consume ``training_rows`` filtered by
    ``(token_ids, started_at..ended_at)``; the SDK's
    ``_load_training_rows`` accepts these as additional filters when
    a ``recording_session_id`` is set in the training payload.

    Returns None if the session is missing, has no captured rows, or
    hasn't started yet.
    """
    row = await get_session(session_id)
    if row is None or row.started_at is None:
        return None
    tokens = list(row.target_token_ids_json or [])
    if not tokens:
        return None
    end = row.ended_at or _utcnow()
    return {
        "session_id": row.id,
        "session_name": row.name,
        "token_ids": tokens,
        "start": row.started_at,
        "end": end,
        "rows_captured": int(row.rows_captured or 0),
    }


async def session_backtest_scope(session_id: str) -> dict[str, Any] | None:
    """Translate a session into ``run_unified_backtest`` kwargs.

    Returns ``None`` when the session is missing or has no captured
    rows.  Otherwise: ``{token_ids, start, end}`` ready to pass through
    to the backtest runner.
    """
    row = await get_session(session_id)
    if row is None:
        return None
    tokens = list(row.target_token_ids_json or [])
    if not tokens:
        return None
    start = row.started_at
    end = row.ended_at or _utcnow()
    if start is None:
        return None
    return {
        "token_ids": tokens,
        "start": start,
        "end": end,
        "session_id": row.id,
        "session_name": row.name,
    }
