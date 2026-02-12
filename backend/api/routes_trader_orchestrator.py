"""API routes for the trader orchestrator control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    arm_live_start,
    compose_trader_orchestrator_config,
    consume_live_arm_token,
    create_live_preflight,
    create_trader_event,
    get_orchestrator_overview,
    read_orchestrator_control,
    read_orchestrator_snapshot,
    update_orchestrator_control,
)

router = APIRouter(prefix="/trader-orchestrator", tags=["Trader Orchestrator"])


class StartRequest(BaseModel):
    mode: Optional[str] = Field(default=None, description="paper | live")
    requested_by: Optional[str] = None


class KillSwitchRequest(BaseModel):
    enabled: bool = True
    requested_by: Optional[str] = None


class LivePreflightRequest(BaseModel):
    mode: str = Field(default="live")
    requested_by: Optional[str] = None


class LiveArmRequest(BaseModel):
    preflight_id: str
    ttl_seconds: int = Field(default=300, ge=30, le=1800)
    requested_by: Optional[str] = None


class LiveStartRequest(BaseModel):
    arm_token: str
    mode: str = Field(default="live")
    requested_by: Optional[str] = None


class LiveStopRequest(BaseModel):
    requested_by: Optional[str] = None


def _assert_not_globally_paused() -> None:
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )


@router.get("/overview")
async def get_overview(session: AsyncSession = Depends(get_db_session)):
    return await get_orchestrator_overview(session)


@router.get("/status")
async def get_status(session: AsyncSession = Depends(get_db_session)):
    control = await read_orchestrator_control(session)
    snapshot = await read_orchestrator_snapshot(session)
    config = await compose_trader_orchestrator_config(session)
    return {
        "control": control,
        "snapshot": snapshot,
        "config": config,
    }


@router.post("/start")
async def start_orchestrator(
    request: StartRequest = StartRequest(),
    session: AsyncSession = Depends(get_db_session),
):
    _assert_not_globally_paused()
    mode = str(request.mode or "paper").strip().lower()
    if mode not in {"paper", "live"}:
        raise HTTPException(status_code=422, detail="mode must be paper or live")

    control = await update_orchestrator_control(
        session,
        is_enabled=True,
        is_paused=False,
        mode=mode,
        requested_run_at=datetime.utcnow(),
    )
    await create_trader_event(
        session,
        event_type="started",
        source="trader_orchestrator",
        operator=request.requested_by,
        message=f"Trader orchestrator started in {mode} mode",
        payload={"mode": mode},
    )
    return {"status": "started", "control": control}


@router.post("/stop")
async def stop_orchestrator(session: AsyncSession = Depends(get_db_session)):
    control = await update_orchestrator_control(
        session,
        is_enabled=False,
        is_paused=True,
        requested_run_at=None,
    )
    await create_trader_event(
        session,
        event_type="stopped",
        source="trader_orchestrator",
        message="Trader orchestrator stopped",
    )
    return {"status": "stopped", "control": control}


@router.post("/kill-switch")
async def set_kill_switch(
    request: KillSwitchRequest,
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_orchestrator_control(
        session,
        kill_switch=bool(request.enabled),
    )
    await create_trader_event(
        session,
        event_type="kill_switch",
        severity="warn" if request.enabled else "info",
        source="trader_orchestrator",
        operator=request.requested_by,
        message="Kill switch updated",
        payload={"enabled": bool(request.enabled)},
    )
    return {
        "status": "updated",
        "kill_switch": bool(request.enabled),
        "control": control,
    }


@router.post("/live/preflight")
async def run_live_preflight(
    request: LivePreflightRequest,
    session: AsyncSession = Depends(get_db_session),
):
    result = await create_live_preflight(
        session,
        requested_mode=request.mode,
        requested_by=request.requested_by,
    )
    return result


@router.post("/live/arm")
async def arm_live(
    request: LiveArmRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        return await arm_live_start(
            session,
            preflight_id=request.preflight_id,
            ttl_seconds=request.ttl_seconds,
            requested_by=request.requested_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/live/start")
async def start_live(
    request: LiveStartRequest,
    session: AsyncSession = Depends(get_db_session),
):
    _assert_not_globally_paused()
    ok = await consume_live_arm_token(session, request.arm_token)
    if not ok:
        raise HTTPException(status_code=409, detail="Invalid or expired arm token")

    control = await update_orchestrator_control(
        session,
        mode="live",
        is_enabled=True,
        is_paused=False,
        requested_run_at=datetime.utcnow(),
    )
    await create_trader_event(
        session,
        event_type="live_started",
        source="trader_orchestrator",
        operator=request.requested_by,
        message="Live trading started",
        payload={"mode": request.mode},
    )
    return {"status": "started", "control": control}


@router.post("/live/stop")
async def stop_live(
    request: LiveStopRequest = LiveStopRequest(),
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_orchestrator_control(
        session,
        mode="paper",
        is_paused=True,
    )
    await create_trader_event(
        session,
        event_type="live_stopped",
        source="trader_orchestrator",
        operator=request.requested_by,
        message="Live trading stopped",
    )
    return {"status": "stopped", "control": control}
