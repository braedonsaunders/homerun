"""API routes for the trader orchestrator control plane."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import SimulationAccount, get_db_session
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
    write_orchestrator_snapshot,
    ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
    update_orchestrator_control,
)
from utils.utcnow import utcnow

router = APIRouter(prefix="/trader-orchestrator", tags=["Trader Orchestrator"])

# ---------------------------------------------------------------------------
# Authentication dependency for state-changing orchestrator endpoints.
# Set ORCHESTRATOR_API_KEY in environment to enable. When set, callers
# must pass the same value in the X-API-Key header.
# ---------------------------------------------------------------------------
_ORCHESTRATOR_API_KEY: str | None = os.environ.get("ORCHESTRATOR_API_KEY") or None


async def _require_orchestrator_auth(x_api_key: Optional[str] = Header(None)) -> None:
    """Validate the X-API-Key header against the configured orchestrator key.

    When ORCHESTRATOR_API_KEY is not configured the guard is a no-op so
    existing development setups are not broken.  In production the env var
    MUST be set.
    """
    if _ORCHESTRATOR_API_KEY is None:
        return  # auth not configured -- allow (dev / local mode)
    if not x_api_key or x_api_key != _ORCHESTRATOR_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header for orchestrator endpoints.",
        )


class StartRequest(BaseModel):
    mode: Optional[str] = Field(default=None, description="paper | live")
    paper_account_id: Optional[str] = None
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


async def _paper_account_exists(session: AsyncSession, account_id: str) -> bool:
    row = (
        await session.execute(select(SimulationAccount.id).where(SimulationAccount.id == account_id))
    ).scalar_one_or_none()
    return row is not None


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
    _auth: None = Depends(_require_orchestrator_auth),
):
    _assert_not_globally_paused()
    mode = str(request.mode or "paper").strip().lower()
    if mode == "live":
        raise HTTPException(
            status_code=422,
            detail=(
                "Live mode cannot be started via /start. "
                "Use the /live/preflight -> /live/arm -> /live/start ceremony instead."
            ),
        )
    if mode != "paper":
        raise HTTPException(status_code=422, detail="mode must be 'paper'. Use /live/start for live mode.")

    control_before = await read_orchestrator_control(session)
    settings_updates = {}
    if mode == "paper":
        requested_paper = str(request.paper_account_id or "").strip() or None
        existing_paper = str((control_before.get("settings") or {}).get("paper_account_id") or "").strip() or None
        paper_account_id = requested_paper or existing_paper
        if not paper_account_id:
            raise HTTPException(
                status_code=422,
                detail="paper_account_id required for paper mode. Select a sandbox account.",
            )
        if not await _paper_account_exists(session, paper_account_id):
            raise HTTPException(
                status_code=422,
                detail="Selected paper_account_id does not exist. Select a valid sandbox account.",
            )
        settings_updates["paper_account_id"] = paper_account_id

    control = await update_orchestrator_control(
        session,
        is_enabled=True,
        is_paused=False,
        mode=mode,
        requested_run_at=utcnow(),
        settings_json=settings_updates,
    )
    await write_orchestrator_snapshot(
        session,
        running=False,
        enabled=True,
        current_activity="Start command queued",
        interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
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
async def stop_orchestrator(
    session: AsyncSession = Depends(get_db_session),
    _auth: None = Depends(_require_orchestrator_auth),
):
    control = await update_orchestrator_control(
        session,
        is_enabled=False,
        is_paused=True,
        requested_run_at=None,
    )
    await write_orchestrator_snapshot(
        session,
        running=False,
        enabled=False,
        current_activity="Manual stop requested",
        interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
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
    _auth: None = Depends(_require_orchestrator_auth),
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
    _auth: None = Depends(_require_orchestrator_auth),
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
    _auth: None = Depends(_require_orchestrator_auth),
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
    _auth: None = Depends(_require_orchestrator_auth),
):
    _assert_not_globally_paused()
    preflight = await create_live_preflight(
        session,
        requested_mode="live",
        requested_by=request.requested_by,
    )
    if preflight.get("status") != "passed":
        raise HTTPException(
            status_code=409, detail="Live preflight failed. Configure trading settings/credentials first."
        )

    ok = await consume_live_arm_token(session, request.arm_token)
    if not ok:
        raise HTTPException(status_code=409, detail="Invalid or expired arm token")

    control = await update_orchestrator_control(
        session,
        mode="live",
        is_enabled=True,
        is_paused=False,
        requested_run_at=utcnow(),
    )
    await write_orchestrator_snapshot(
        session,
        running=False,
        enabled=True,
        current_activity="Live start command queued",
        interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
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
    _auth: None = Depends(_require_orchestrator_auth),
):
    control = await update_orchestrator_control(
        session,
        mode="paper",
        is_enabled=False,
        is_paused=True,
        requested_run_at=None,
    )
    await write_orchestrator_snapshot(
        session,
        running=False,
        enabled=False,
        current_activity="Live stop requested",
        interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
    )
    await create_trader_event(
        session,
        event_type="live_stopped",
        source="trader_orchestrator",
        operator=request.requested_by,
        message="Live trading stopped",
    )
    return {"status": "stopped", "control": control}
