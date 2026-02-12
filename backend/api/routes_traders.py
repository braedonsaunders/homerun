"""API routes for trader CRUD and trader-level runtime surfaces."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    create_config_revision,
    create_trader,
    create_trader_event,
    create_trader_from_template,
    delete_trader,
    get_trader,
    get_trader_decision_detail,
    list_serialized_trader_decisions,
    list_serialized_trader_events,
    list_serialized_trader_orders,
    list_trader_templates,
    list_traders,
    request_trader_run,
    set_trader_paused,
    update_trader,
)

router = APIRouter(prefix="/traders", tags=["Traders"])


class TraderRequest(BaseModel):
    name: str
    description: Optional[str] = None
    strategy_key: str
    sources: list[str] = Field(default_factory=list)
    interval_seconds: int = Field(default=60, ge=1, le=86400)
    params: dict[str, Any] = Field(default_factory=dict)
    risk_limits: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True
    is_paused: bool = False
    requested_by: Optional[str] = None
    reason: Optional[str] = None


class TraderPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_key: Optional[str] = None
    sources: Optional[list[str]] = None
    interval_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    params: Optional[dict[str, Any]] = None
    risk_limits: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    is_enabled: Optional[bool] = None
    is_paused: Optional[bool] = None
    requested_by: Optional[str] = None
    reason: Optional[str] = None


class TraderTemplateCreateRequest(BaseModel):
    template_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    requested_by: Optional[str] = None


def _assert_not_globally_paused() -> None:
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )


@router.get("")
async def get_all_traders(session: AsyncSession = Depends(get_db_session)):
    return {"traders": await list_traders(session)}


@router.get("/templates")
async def get_templates():
    return {"templates": list_trader_templates()}


@router.post("/from-template")
async def create_from_template(
    request: TraderTemplateCreateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        trader = await create_trader_from_template(
            session,
            template_id=request.template_id,
            overrides=request.overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    await create_trader_event(
        session,
        trader_id=trader["id"],
        event_type="trader_created",
        source="operator",
        operator=request.requested_by,
        message="Trader created from template",
        payload={"template_id": request.template_id},
    )
    return trader


@router.post("")
async def create_trader_route(
    request: TraderRequest,
    session: AsyncSession = Depends(get_db_session),
):
    payload = request.model_dump(exclude={"requested_by", "reason"})
    try:
        trader = await create_trader(session, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    await create_config_revision(
        session,
        trader_id=trader["id"],
        operator=request.requested_by,
        reason=request.reason or "trader_create",
        orchestrator_before={},
        orchestrator_after={},
        trader_before={},
        trader_after=trader,
    )
    await create_trader_event(
        session,
        trader_id=trader["id"],
        event_type="trader_created",
        source="operator",
        operator=request.requested_by,
        message="Trader created",
        payload={"trader": trader},
    )
    return trader


@router.get("/{trader_id}")
async def get_trader_route(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    return trader


@router.put("/{trader_id}")
async def update_trader_route(
    trader_id: str,
    request: TraderPatchRequest,
    session: AsyncSession = Depends(get_db_session),
):
    before = await get_trader(session, trader_id)
    if before is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    payload = request.model_dump(exclude_none=True, exclude={"requested_by", "reason"})
    after = await update_trader(session, trader_id, payload)
    if after is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    await create_config_revision(
        session,
        trader_id=trader_id,
        operator=request.requested_by,
        reason=request.reason or "trader_update",
        orchestrator_before={},
        orchestrator_after={},
        trader_before=before,
        trader_after=after,
    )
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_updated",
        source="operator",
        operator=request.requested_by,
        message="Trader updated",
        payload={"changes": payload},
    )
    return after


@router.delete("/{trader_id}")
async def delete_trader_route(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    ok = await delete_trader(session, trader_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_deleted",
        source="operator",
        message="Trader deleted",
    )
    return {"status": "deleted", "trader_id": trader_id}


@router.post("/{trader_id}/start")
async def start_trader(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    _assert_not_globally_paused()
    trader = await set_trader_paused(session, trader_id, False)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_started",
        source="operator",
        message="Trader resumed",
    )
    return trader


@router.post("/{trader_id}/pause")
async def pause_trader(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    trader = await set_trader_paused(session, trader_id, True)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_paused",
        source="operator",
        message="Trader paused",
    )
    return trader


@router.post("/{trader_id}/run-once")
async def run_once(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    _assert_not_globally_paused()
    trader = await request_trader_run(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="run_once_requested",
        source="operator",
        message="Run-once requested",
    )
    return trader


@router.get("/{trader_id}/decisions")
async def get_trader_decisions(
    trader_id: str,
    decision: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
):
    return {
        "decisions": await list_serialized_trader_decisions(
            session,
            trader_id=trader_id,
            decision=decision,
            limit=limit,
        )
    }


@router.get("/{trader_id}/orders")
async def get_trader_orders(
    trader_id: str,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
):
    return {
        "orders": await list_serialized_trader_orders(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    }


@router.get("/{trader_id}/events")
async def get_events(
    trader_id: str,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    types: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    event_types = [item.strip() for item in (types or "").split(",") if item.strip()]
    events, next_cursor = await list_serialized_trader_events(
        session,
        trader_id=trader_id,
        limit=limit,
        cursor=cursor,
        event_types=event_types or None,
    )
    return {"events": events, "next_cursor": next_cursor}


@router.get("/decisions/{decision_id}")
async def get_decision_detail(
    decision_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    detail = await get_trader_decision_detail(session, decision_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return detail
