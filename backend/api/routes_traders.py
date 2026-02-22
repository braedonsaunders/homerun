"""API routes for trader CRUD and trader-level runtime surfaces."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.pause_state import global_pause_state
from services.trader_orchestrator.position_lifecycle import reconcile_live_positions, reconcile_paper_positions
from services.trader_orchestrator.session_engine import ExecutionSessionEngine
from services.trader_orchestrator_state import (
    cleanup_trader_open_orders,
    create_config_revision,
    create_trader,
    create_trader_event,
    create_trader_from_template,
    delete_trader,
    get_trader,
    get_trader_decision_detail,
    get_open_order_summary_for_trader,
    get_open_position_summary_for_trader,
    get_serialized_execution_session_detail,
    list_serialized_trader_decisions,
    list_serialized_trader_events,
    list_serialized_execution_sessions,
    list_serialized_trader_orders,
    list_trader_templates,
    list_traders,
    read_orchestrator_control,
    request_trader_run,
    set_trader_paused,
    sync_trader_position_inventory,
    update_trader,
)

router = APIRouter(prefix="/traders", tags=["Traders"])


class TraderTradersScopeRequest(BaseModel):
    modes: list[str] = Field(default_factory=list)
    individual_wallets: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)


class TraderSourceConfigRequest(BaseModel):
    source_key: str
    strategy_key: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    traders_scope: Optional[TraderTradersScopeRequest] = None


class TraderRequest(BaseModel):
    name: str
    description: Optional[str] = None
    source_configs: list[TraderSourceConfigRequest] = Field(default_factory=list)
    interval_seconds: int = Field(default=60, ge=1, le=86400)
    risk_limits: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True
    is_paused: bool = False
    requested_by: Optional[str] = None
    reason: Optional[str] = None


class TraderPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_configs: Optional[list[TraderSourceConfigRequest]] = None
    interval_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
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


class TraderDeleteAction(str, Enum):
    block = "block"
    disable = "disable"
    force_delete = "force_delete"


class TraderPositionCleanupScope(str, Enum):
    paper = "paper"
    live = "live"
    all = "all"


class TraderPositionCleanupMethod(str, Enum):
    mark_to_market = "mark_to_market"
    cancel = "cancel"


class TraderPositionCleanupRequest(BaseModel):
    scope: TraderPositionCleanupScope = TraderPositionCleanupScope.paper
    method: TraderPositionCleanupMethod = TraderPositionCleanupMethod.mark_to_market
    max_age_hours: Optional[int] = Field(default=None, ge=1, le=24 * 365)
    dry_run: bool = False
    target_status: str = Field(default="cancelled", min_length=1, max_length=64)
    reason: Optional[str] = None
    confirm_live: bool = False


class TraderExecutionSessionControlRequest(BaseModel):
    reason: Optional[str] = None


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
    try:
        after = await update_trader(session, trader_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
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
async def delete_trader_route(
    trader_id: str,
    action: TraderDeleteAction = Query(default=TraderDeleteAction.block),
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    await sync_trader_position_inventory(session, trader_id=trader_id)
    open_summary = await get_open_position_summary_for_trader(session, trader_id)
    open_order_summary = await get_open_order_summary_for_trader(session, trader_id)
    open_live_positions = int(open_summary.get("live", 0))
    open_paper_positions = int(open_summary.get("paper", 0))
    open_other_positions = int(open_summary.get("other", 0))
    open_total_positions = int(open_summary.get("total", 0))
    open_live_orders = int(open_order_summary.get("live", 0))
    open_paper_orders = int(open_order_summary.get("paper", 0))
    open_other_orders = int(open_order_summary.get("other", 0))
    open_total_orders = int(open_order_summary.get("total", 0))

    if action == TraderDeleteAction.disable:
        updated = await update_trader(
            session,
            trader_id,
            {
                "is_enabled": False,
                "is_paused": True,
            },
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Trader not found")
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_delete_requested",
            severity="warn",
            source="operator",
            message="Trader disabled instead of deleted",
            payload={
                "open_live_positions": open_live_positions,
                "open_paper_positions": open_paper_positions,
                "open_other_positions": open_other_positions,
                "open_live_orders": open_live_orders,
                "open_paper_orders": open_paper_orders,
                "open_other_orders": open_other_orders,
            },
        )
        return {
            "status": "disabled",
            "trader_id": trader_id,
            "open_live_positions": open_live_positions,
            "open_paper_positions": open_paper_positions,
            "open_other_positions": open_other_positions,
            "open_live_orders": open_live_orders,
            "open_paper_orders": open_paper_orders,
            "open_other_orders": open_other_orders,
            "message": "Trader disabled and paused. Resolve live exposure before permanent deletion.",
        }

    if (
        open_live_positions > 0 or open_other_positions > 0 or open_live_orders > 0 or open_other_orders > 0
    ) and action != TraderDeleteAction.force_delete:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "open_live_exposure",
                "message": "Trader has live exposure. Disable trader and flatten live positions/orders before deletion.",
                "trader_id": trader_id,
                "open_live_positions": open_live_positions,
                "open_paper_positions": open_paper_positions,
                "open_other_positions": open_other_positions,
                "open_live_orders": open_live_orders,
                "open_paper_orders": open_paper_orders,
                "open_other_orders": open_other_orders,
                "suggested_action": TraderDeleteAction.disable.value,
            },
        )

    try:
        ok = await delete_trader(
            session,
            trader_id,
            force=(action == TraderDeleteAction.force_delete),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=None,
        event_type="trader_deleted",
        severity="warn" if action == TraderDeleteAction.force_delete else "info",
        source="operator",
        message="Trader deleted",
        payload={
            "deleted_trader_id": trader_id,
            "action": action.value,
            "open_live_positions_at_delete": open_live_positions,
            "open_paper_positions_at_delete": open_paper_positions,
            "open_other_positions_at_delete": open_other_positions,
            "open_live_orders_at_delete": open_live_orders,
            "open_paper_orders_at_delete": open_paper_orders,
            "open_other_orders_at_delete": open_other_orders,
        },
    )
    return {
        "status": "deleted",
        "trader_id": trader_id,
        "action": action.value,
        "open_live_positions": open_live_positions,
        "open_paper_positions": open_paper_positions,
        "open_other_positions": open_other_positions,
        "open_live_orders": open_live_orders,
        "open_paper_orders": open_paper_orders,
        "open_other_orders": open_other_orders,
        "open_total_positions": open_total_positions,
        "open_total_orders": open_total_orders,
    }


@router.post("/{trader_id}/start")
async def start_trader(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    _assert_not_globally_paused()
    trader = await set_trader_paused(session, trader_id, False)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    control = await read_orchestrator_control(session)
    mode = str(control.get("mode") or "paper").strip().lower()
    await sync_trader_position_inventory(session, trader_id=trader_id, mode=mode if mode in {"paper", "live"} else None)
    open_summary = await get_open_position_summary_for_trader(session, trader_id)
    open_live = int(open_summary.get("live", 0))
    open_paper = int(open_summary.get("paper", 0))
    if mode == "live" and open_live > 0:
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_start_warning",
            severity="warn",
            source="operator",
            message="Trader started with existing live open positions",
            payload={
                "open_live_positions": open_live,
                "open_paper_positions": open_paper,
            },
        )
    elif mode == "paper" and open_paper > 0:
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_start_notice",
            source="operator",
            message="Trader started with existing paper open positions",
            payload={"open_paper_positions": open_paper},
        )

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


@router.post("/{trader_id}/positions/cleanup")
async def cleanup_trader_positions(
    trader_id: str,
    request: TraderPositionCleanupRequest,
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    if request.scope in {TraderPositionCleanupScope.live, TraderPositionCleanupScope.all} and not request.confirm_live:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "confirm_live_required",
                "message": "Live cleanup requires explicit confirmation. Re-submit with confirm_live=true.",
                "scope": request.scope.value,
            },
        )

    if request.method == TraderPositionCleanupMethod.mark_to_market and request.scope not in {
        TraderPositionCleanupScope.paper,
        TraderPositionCleanupScope.live,
    }:
        raise HTTPException(
            status_code=422,
            detail="mark_to_market cleanup supports paper and live scopes (use scope=paper or scope=live)",
        )

    if request.method == TraderPositionCleanupMethod.cancel:
        try:
            result = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope=request.scope.value,
                max_age_hours=request.max_age_hours,
                dry_run=request.dry_run,
                target_status=request.target_status,
                reason=request.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        if request.scope == TraderPositionCleanupScope.live:
            lifecycle_result = await reconcile_live_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=request.dry_run,
                force_mark_to_market=True,
                max_age_hours=request.max_age_hours,
                reason=str(request.reason or "manual_mark_to_market_cleanup"),
            )
            if not request.dry_run:
                await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
        else:
            lifecycle_result = await reconcile_paper_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=request.dry_run,
                force_mark_to_market=True,
                max_age_hours=request.max_age_hours,
                reason=str(request.reason or "manual_mark_to_market_cleanup"),
            )
            if not request.dry_run:
                await sync_trader_position_inventory(session, trader_id=trader_id, mode="paper")
        result = {
            "trader_id": trader_id,
            "scope": request.scope.value,
            "method": request.method.value,
            "dry_run": request.dry_run,
            "max_age_hours": request.max_age_hours,
            "matched": int(lifecycle_result.get("matched", 0)),
            "would_close": int(lifecycle_result.get("would_close", 0)),
            "updated": int(lifecycle_result.get("closed", 0)),
            "skipped": int(lifecycle_result.get("skipped", 0)),
            "held": int(lifecycle_result.get("held", 0)),
            "total_realized_pnl": float(lifecycle_result.get("total_realized_pnl", 0.0)),
            "by_status": lifecycle_result.get("by_status", {}),
            "skipped_reasons": lifecycle_result.get("skipped_reasons", {}),
            "details": lifecycle_result.get("details", []),
        }

    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_positions_cleanup",
        severity=(
            "warn" if request.scope in {TraderPositionCleanupScope.live, TraderPositionCleanupScope.all} else "info"
        ),
        source="operator",
        message="Trader position cleanup executed" if not request.dry_run else "Trader position cleanup dry-run",
        payload=result,
    )
    return result


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


@router.get("/{trader_id}/execution-sessions")
async def get_execution_sessions(
    trader_id: str,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    return {
        "sessions": await list_serialized_execution_sessions(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    }


@router.get("/execution-sessions/{session_id}")
async def get_execution_session(
    session_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    detail = await get_serialized_execution_session_detail(session, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return detail


@router.post("/execution-sessions/{session_id}/cancel")
async def cancel_execution_session(
    session_id: str,
    request: TraderExecutionSessionControlRequest = TraderExecutionSessionControlRequest(),
    session: AsyncSession = Depends(get_db_session),
):
    engine = ExecutionSessionEngine(session)
    reason = str(request.reason or "manual_cancel").strip()
    ok = await engine.cancel_session(session_id=session_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return {"status": "cancelled", "session_id": session_id}


@router.post("/execution-sessions/{session_id}/pause")
async def pause_execution_session(
    session_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    engine = ExecutionSessionEngine(session)
    ok = await engine.pause_session(session_id=session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return {"status": "paused", "session_id": session_id}


@router.post("/execution-sessions/{session_id}/resume")
async def resume_execution_session(
    session_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    engine = ExecutionSessionEngine(session)
    ok = await engine.resume_session(session_id=session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return {"status": "working", "session_id": session_id}


@router.get("/decisions/{decision_id}")
async def get_decision_detail(
    decision_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    detail = await get_trader_decision_detail(session, decision_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return detail
