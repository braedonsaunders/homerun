"""API routes for normalized trade signals."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import TradeSignal, TradeSignalSnapshot, get_db_session
from services.signal_bus import list_trade_signals, refresh_trade_signal_snapshots

router = APIRouter(prefix="/signals", tags=["Signals"])


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
async def get_signal_stats(session: AsyncSession = Depends(get_db_session)):
    snapshots = await refresh_trade_signal_snapshots(session)

    rows = (
        (await session.execute(select(TradeSignalSnapshot).order_by(TradeSignalSnapshot.source.asc())))
        .scalars()
        .all()
    )

    totals = {
        "pending": 0,
        "selected": 0,
        "submitted": 0,
        "executed": 0,
        "skipped": 0,
        "expired": 0,
        "failed": 0,
    }
    for row in rows:
        totals["pending"] += int(row.pending_count or 0)
        totals["selected"] += int(row.selected_count or 0)
        totals["submitted"] += int(row.submitted_count or 0)
        totals["executed"] += int(row.executed_count or 0)
        totals["skipped"] += int(row.skipped_count or 0)
        totals["expired"] += int(row.expired_count or 0)
        totals["failed"] += int(row.failed_count or 0)

    return {
        "totals": totals,
        "sources": snapshots,
    }
