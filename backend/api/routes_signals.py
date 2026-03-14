"""API routes for normalized trade signals."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, TradeSignal, get_db_session
from services.signal_bus import list_trade_signals, read_trade_signal_source_stats
from utils.retry import is_retryable_db_error

router = APIRouter(prefix="/signals", tags=["Signals"])

_SIGNALS_STATS_CACHE_TTL_SECONDS = 5.0
_signals_stats_cache: dict | None = None
_signals_stats_cache_updated_at: float = 0.0
_signals_stats_refresh_task: asyncio.Task | None = None


async def _build_signal_stats_payload(session: AsyncSession) -> dict:
    snapshots = await read_trade_signal_source_stats(session)

    totals = {
        "pending": 0,
        "selected": 0,
        "submitted": 0,
        "executed": 0,
        "skipped": 0,
        "expired": 0,
        "failed": 0,
    }
    for row in snapshots:
        totals["pending"] += int(row.get("pending_count", 0) or 0)
        totals["selected"] += int(row.get("selected_count", 0) or 0)
        totals["submitted"] += int(row.get("submitted_count", 0) or 0)
        totals["executed"] += int(row.get("executed_count", 0) or 0)
        totals["skipped"] += int(row.get("skipped_count", 0) or 0)
        totals["expired"] += int(row.get("expired_count", 0) or 0)
        totals["failed"] += int(row.get("failed_count", 0) or 0)

    return {
        "totals": totals,
        "sources": snapshots,
    }


async def _refresh_signal_stats_cache() -> None:
    global _signals_stats_cache
    global _signals_stats_cache_updated_at
    async with AsyncSessionLocal() as session:
        payload = await _build_signal_stats_payload(session)
    _signals_stats_cache = payload
    _signals_stats_cache_updated_at = time.monotonic()


@router.get("")
async def get_signals(
    source: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
):
    rows = await list_trade_signals(
        session,
        source=source,
        status=status,
        limit=limit,
        offset=offset,
    )

    total_query = select(func.count(TradeSignal.id))
    if source:
        total_query = total_query.where(TradeSignal.source == source)
    if status:
        total_query = total_query.where(TradeSignal.status == status)
    total = int((await session.execute(total_query)).scalar() or 0)

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "signals": [
            {
                "id": row.id,
                "source": row.source,
                "source_item_id": row.source_item_id,
                "signal_type": row.signal_type,
                "strategy_type": row.strategy_type,
                "market_id": row.market_id,
                "market_question": row.market_question,
                "direction": row.direction,
                "entry_price": row.entry_price,
                "effective_price": row.effective_price,
                "edge_percent": row.edge_percent,
                "confidence": row.confidence,
                "liquidity": row.liquidity,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                "status": row.status,
                "payload": row.payload_json,
                "dedupe_key": row.dedupe_key,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ],
    }


@router.get("/stats")
async def get_signal_stats():
    global _signals_stats_cache
    global _signals_stats_cache_updated_at
    global _signals_stats_refresh_task

    now = time.monotonic()
    cache_fresh = (
        _signals_stats_cache is not None
        and (now - _signals_stats_cache_updated_at) < _SIGNALS_STATS_CACHE_TTL_SECONDS
    )
    if cache_fresh:
        return _signals_stats_cache

    if _signals_stats_refresh_task is not None and not _signals_stats_refresh_task.done():
        if _signals_stats_cache is not None:
            return _signals_stats_cache
        await _signals_stats_refresh_task
        return _signals_stats_cache or {"totals": {}, "sources": []}

    try:
        async with AsyncSessionLocal() as session:
            payload = await _build_signal_stats_payload(session)
        _signals_stats_cache = payload
        _signals_stats_cache_updated_at = now
        return payload
    except Exception as exc:
        if not is_retryable_db_error(exc) or _signals_stats_cache is None:
            raise
        _signals_stats_refresh_task = asyncio.create_task(_refresh_signal_stats_cache())
        return _signals_stats_cache
