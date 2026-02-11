"""DB-backed AutoTrader API routes.

The autotrader runtime lives in ``workers.autotrader_worker``.
These endpoints only read/write persisted control, policy, decision, and trade state.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.autotrader_state import (
    get_autotrader_exposure,
    get_autotrader_metrics,
    list_autotrader_decisions,
    list_autotrader_trades,
    read_autotrader_control,
    read_autotrader_policies,
    read_autotrader_snapshot,
    update_autotrader_control,
    upsert_autotrader_policies,
)

router = APIRouter(prefix="/auto-trader", tags=["Auto Trader"])


class AutoTraderControlRequest(BaseModel):
    is_enabled: Optional[bool] = None
    is_paused: Optional[bool] = None
    mode: Optional[str] = Field(None, description="paper | live | shadow | mock")
    run_interval_seconds: Optional[int] = Field(None, ge=1, le=60)
    kill_switch: Optional[bool] = None


class AutoTraderPoliciesRequest(BaseModel):
    global_policy: Optional[dict[str, Any]] = Field(default=None, alias="global")
    sources: Optional[dict[str, dict[str, Any]]] = None

    class Config:
        populate_by_name = True


@router.get("/status")
async def get_auto_trader_status(session: AsyncSession = Depends(get_db_session)):
    control = await read_autotrader_control(session)
    snapshot = await read_autotrader_snapshot(session)
    policies = await read_autotrader_policies(session)
    return {
        "control": control,
        "snapshot": snapshot,
        "policies": policies,
    }


@router.put("/control")
async def update_auto_trader_control(
    request: AutoTraderControlRequest,
    session: AsyncSession = Depends(get_db_session),
):
    payload = request.model_dump(exclude_unset=True)
    control = await update_autotrader_control(session, **payload)
    snapshot = await read_autotrader_snapshot(session)
    return {"status": "updated", "control": control, "snapshot": snapshot}


@router.get("/policies")
async def get_auto_trader_policies(session: AsyncSession = Depends(get_db_session)):
    return await read_autotrader_policies(session)


@router.put("/policies")
async def put_auto_trader_policies(
    request: AutoTraderPoliciesRequest,
    session: AsyncSession = Depends(get_db_session),
):
    payload = {
        "global": request.global_policy,
        "sources": request.sources,
    }
    policies = await upsert_autotrader_policies(session, payload)
    return {"status": "updated", "policies": policies}


@router.get("/trades")
async def get_auto_trader_trades(
    source: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    session: AsyncSession = Depends(get_db_session),
):
    rows = await list_autotrader_trades(
        session,
        source=source,
        status=status,
        limit=limit,
    )

    return {
        "total": len(rows),
        "trades": [
            {
                "id": row.id,
                "signal_id": row.signal_id,
                "decision_id": row.decision_id,
                "source": row.source,
                "market_id": row.market_id,
                "market_question": row.market_question,
                "direction": row.direction,
                "mode": row.mode,
                "status": row.status,
                "notional_usd": row.notional_usd,
                "entry_price": row.entry_price,
                "effective_price": row.effective_price,
                "edge_percent": row.edge_percent,
                "confidence": row.confidence,
                "reason": row.reason,
                "payload": row.payload_json,
                "error_message": row.error_message,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "executed_at": row.executed_at.isoformat() if row.executed_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ],
    }


@router.get("/decisions")
async def get_auto_trader_decisions(
    source: Optional[str] = Query(default=None),
    decision: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
):
    rows = await list_autotrader_decisions(
        session,
        source=source,
        decision=decision,
        limit=limit,
    )

    return {
        "total": len(rows),
        "decisions": [
            {
                "id": row.id,
                "signal_id": row.signal_id,
                "source": row.source,
                "decision": row.decision,
                "reason": row.reason,
                "score": row.score,
                "policy_snapshot": row.policy_snapshot_json,
                "risk_snapshot": row.risk_snapshot_json,
                "payload": row.payload_json,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }


@router.get("/exposure")
async def get_auto_trader_exposure(session: AsyncSession = Depends(get_db_session)):
    return await get_autotrader_exposure(session)


@router.get("/metrics")
async def get_auto_trader_metrics(session: AsyncSession = Depends(get_db_session)):
    return await get_autotrader_metrics(session)


@router.post("/start")
async def start_auto_trader(
    mode: str = Query(default="paper"),
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_autotrader_control(
        session,
        is_enabled=True,
        is_paused=False,
        mode=mode,
        kill_switch=False,
    )
    return {"status": "started", "control": control}


@router.post("/pause")
async def pause_auto_trader(session: AsyncSession = Depends(get_db_session)):
    control = await update_autotrader_control(session, is_paused=True)
    return {"status": "paused", "control": control}


@router.post("/stop")
async def stop_auto_trader(session: AsyncSession = Depends(get_db_session)):
    control = await update_autotrader_control(
        session,
        is_enabled=False,
        is_paused=True,
    )
    return {"status": "stopped", "control": control}


@router.post("/run-once")
async def run_auto_trader_once(session: AsyncSession = Depends(get_db_session)):
    control = await update_autotrader_control(session, requested_run=True)
    return {"status": "queued", "control": control}


@router.post("/kill-switch")
async def set_auto_trader_kill_switch(
    enabled: bool = Query(...),
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_autotrader_control(
        session,
        kill_switch=enabled,
        is_paused=True if enabled else None,
    )
    return {"status": "updated", "control": control}
