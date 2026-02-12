"""DB-backed AutoTrader API routes.

The autotrader runtime lives in ``workers.autotrader_worker``.
These endpoints only read/write persisted control, policy, decision, and trade state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import get_db_session
from services.autotrader_state import (
    arm_preflight_run,
    compose_autotrader_config,
    consume_arm_token,
    create_autotrader_event,
    create_config_revision,
    create_preflight_run,
    get_autotrader_decision_detail,
    get_autotrader_exposure,
    get_autotrader_metrics,
    get_autotrader_overview,
    list_autotrader_decisions,
    list_autotrader_events,
    list_autotrader_trades,
    normalize_trading_domains,
    read_autotrader_control,
    read_autotrader_policies,
    read_autotrader_snapshot,
    update_autotrader_control,
    upsert_autotrader_policies,
)
from services.pause_state import global_pause_state
from services.trading import trading_service

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


class AutoTraderConfigPatchRequest(BaseModel):
    mode: Optional[str] = Field(default=None, description="paper | live | shadow | mock")
    run_interval_seconds: Optional[int] = Field(default=None, ge=1, le=60)
    trading_domains: Optional[list[str]] = None
    enabled_strategies: Optional[list[str]] = None
    llm_verify_trades: Optional[bool] = None
    llm_verify_strategies: Optional[list[str]] = None
    auto_ai_scoring: Optional[bool] = None
    paper_account_id: Optional[str] = None
    paper_enable_spread_exits: Optional[bool] = None
    paper_take_profit_pct: Optional[float] = Field(default=None, ge=0.0)
    paper_stop_loss_pct: Optional[float] = Field(default=None, ge=0.0)
    max_daily_loss_usd: Optional[float] = Field(default=None, ge=0.0)
    max_concurrent_positions: Optional[int] = Field(default=None, ge=0)
    max_per_market_exposure: Optional[float] = Field(default=None, ge=0.0)
    max_per_event_exposure: Optional[float] = Field(default=None, ge=0.0)
    news_workflow_enabled: Optional[bool] = None
    weather_workflow_enabled: Optional[bool] = None
    source_overrides: Optional[dict[str, dict[str, Any]]] = None
    requested_by: Optional[str] = None
    reason: Optional[str] = None


class AutoTraderLivePreflightRequest(BaseModel):
    mode: str = Field(default="live")
    requested_by: Optional[str] = None


class AutoTraderLiveArmRequest(BaseModel):
    preflight_id: str
    ttl_seconds: int = Field(default=300, ge=30, le=1800)
    requested_by: Optional[str] = None


class AutoTraderLiveStartRequest(BaseModel):
    arm_token: str
    mode: str = Field(default="live")
    requested_by: Optional[str] = None


class AutoTraderLiveKillSwitchRequest(BaseModel):
    enabled: bool = True
    requested_by: Optional[str] = None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _assert_not_globally_paused() -> None:
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )


def _live_preflight_checks(
    *,
    control: dict[str, Any],
    snapshot: dict[str, Any],
    policies: dict[str, Any],
) -> list[dict[str, Any]]:
    last_run_at = _parse_iso(snapshot.get("last_run_at"))
    lag_seconds: Optional[float] = None
    if last_run_at is not None:
        lag_seconds = max(
            0.0,
            (
                datetime.now(timezone.utc) - last_run_at.astimezone(timezone.utc)
            ).total_seconds(),
        )

    enabled_sources = [
        source
        for source, cfg in (policies.get("sources") or {}).items()
        if bool((cfg or {}).get("enabled"))
    ]

    checks = [
        {
            "id": "trading_enabled",
            "ok": bool(settings.TRADING_ENABLED),
            "message": "TRADING_ENABLED must be true",
        },
        {
            "id": "trading_service_ready",
            "ok": bool(trading_service.is_ready()),
            "message": "Trading service must be initialized",
        },
        {
            "id": "kill_switch_clear",
            "ok": not bool(control.get("kill_switch")),
            "message": "Kill switch must be disabled",
        },
        {
            "id": "sources_enabled",
            "ok": len(enabled_sources) > 0,
            "message": "At least one source policy must be enabled",
            "enabled_sources": enabled_sources,
        },
        {
            "id": "worker_fresh",
            "ok": lag_seconds is None or lag_seconds <= 30.0,
            "message": "Worker snapshot must be fresh (<=30s)",
            "lag_seconds": lag_seconds,
        },
    ]
    return checks


@router.get("/status")
async def get_auto_trader_status(session: AsyncSession = Depends(get_db_session)):
    control = await read_autotrader_control(session)
    snapshot = await read_autotrader_snapshot(session)
    policies = await read_autotrader_policies(session)
    return {
        "control": control,
        "snapshot": snapshot,
        "policies": policies,
        "config": compose_autotrader_config(control, policies),
    }


@router.get("/overview")
async def get_auto_trader_overview(session: AsyncSession = Depends(get_db_session)):
    return await get_autotrader_overview(session)


@router.get("/events")
async def get_auto_trader_events(
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    types: Optional[str] = Query(default=None, description="Comma-separated event types"),
    session: AsyncSession = Depends(get_db_session),
):
    event_types = [t.strip() for t in (types or "").split(",") if t.strip()]
    rows, next_cursor = await list_autotrader_events(
        session,
        cursor=cursor,
        limit=limit,
        event_types=event_types or None,
    )
    return {
        "events": [
            {
                "id": row.id,
                "event_type": row.event_type,
                "severity": row.severity,
                "source": row.source,
                "operator": row.operator,
                "message": row.message,
                "trace_id": row.trace_id,
                "payload": row.payload_json or {},
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
        "next_cursor": next_cursor,
    }


@router.get("/decisions/{decision_id}")
async def get_auto_trader_decision_by_id(
    decision_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    detail = await get_autotrader_decision_detail(session, decision_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return detail


@router.patch("/config")
async def patch_auto_trader_config(
    request: AutoTraderConfigPatchRequest,
    session: AsyncSession = Depends(get_db_session),
):
    before_control = await read_autotrader_control(session)
    before_policies = await read_autotrader_policies(session)

    control_updates: dict[str, Any] = {}
    merged_settings = dict(before_control.get("settings") or {})
    policy_payload: dict[str, Any] = {"global": {}, "sources": {}}

    if request.mode is not None:
        control_updates["mode"] = request.mode
    if request.run_interval_seconds is not None:
        control_updates["run_interval_seconds"] = request.run_interval_seconds
    if request.trading_domains is not None:
        merged_settings["trading_domains"] = normalize_trading_domains(request.trading_domains)

    if request.enabled_strategies is not None:
        merged_settings["enabled_strategies"] = request.enabled_strategies
    if request.llm_verify_trades is not None:
        merged_settings["llm_verify_trades"] = request.llm_verify_trades
    if request.llm_verify_strategies is not None:
        merged_settings["llm_verify_strategies"] = request.llm_verify_strategies
    if request.auto_ai_scoring is not None:
        merged_settings["auto_ai_scoring"] = request.auto_ai_scoring
    if request.paper_account_id is not None:
        merged_settings["paper_account_id"] = request.paper_account_id or None
    if request.paper_enable_spread_exits is not None:
        merged_settings["paper_enable_spread_exits"] = request.paper_enable_spread_exits
    if request.paper_take_profit_pct is not None:
        merged_settings["paper_take_profit_pct"] = request.paper_take_profit_pct
    if request.paper_stop_loss_pct is not None:
        merged_settings["paper_stop_loss_pct"] = request.paper_stop_loss_pct

    if request.max_daily_loss_usd is not None:
        policy_payload["global"]["max_daily_loss"] = request.max_daily_loss_usd
    if request.max_concurrent_positions is not None:
        policy_payload["global"]["max_total_open_positions"] = request.max_concurrent_positions
    if request.max_per_market_exposure is not None:
        policy_payload["global"]["max_per_market_exposure"] = request.max_per_market_exposure
    if request.max_per_event_exposure is not None:
        policy_payload["global"]["max_per_event_exposure"] = request.max_per_event_exposure
    if request.news_workflow_enabled is not None:
        policy_payload["sources"]["news"] = {"enabled": request.news_workflow_enabled}
    if request.weather_workflow_enabled is not None:
        policy_payload["sources"]["weather"] = {"enabled": request.weather_workflow_enabled}
    if request.source_overrides:
        for source, override in request.source_overrides.items():
            if not isinstance(override, dict):
                continue
            policy_payload["sources"][source] = {
                **(policy_payload["sources"].get(source) or {}),
                **override,
            }

    control_updates["settings"] = merged_settings
    control_after = await update_autotrader_control(session, **control_updates)

    global_updates = policy_payload.get("global") or {}
    source_updates = policy_payload.get("sources") or {}
    if global_updates or source_updates:
        await upsert_autotrader_policies(
            session,
            {
                "global": global_updates or None,
                "sources": source_updates or None,
            },
        )
    policies_after = await read_autotrader_policies(session)

    await create_config_revision(
        session,
        control_before=before_control,
        policies_before=before_policies,
        control_after=control_after,
        policies_after=policies_after,
        operator=request.requested_by,
        reason=request.reason or "config_patch",
    )
    await create_autotrader_event(
        session,
        event_type="config_updated",
        severity="info",
        source="api",
        operator=request.requested_by,
        message="Auto-trader configuration updated",
        payload={
            "reason": request.reason,
            "updated_fields": sorted(
                {
                    *[k for k, v in control_updates.items() if v is not None and k != "settings"],
                    *[
                        k
                        for k in [
                            "enabled_strategies",
                            "trading_domains",
                            "llm_verify_trades",
                            "llm_verify_strategies",
                            "auto_ai_scoring",
                            "paper_account_id",
                            "paper_enable_spread_exits",
                            "paper_take_profit_pct",
                            "paper_stop_loss_pct",
                            "max_daily_loss_usd",
                            "max_concurrent_positions",
                            "max_per_market_exposure",
                            "max_per_event_exposure",
                            "news_workflow_enabled",
                            "weather_workflow_enabled",
                            "source_overrides",
                        ]
                        if getattr(request, k) is not None
                    ],
                }
            ),
        },
    )
    return {
        "status": "updated",
        "control": control_after,
        "policies": policies_after,
        "config": compose_autotrader_config(control_after, policies_after),
    }


@router.put("/control")
async def update_auto_trader_control_route(
    request: AutoTraderControlRequest,
    session: AsyncSession = Depends(get_db_session),
):
    payload = request.model_dump(exclude_unset=True)
    control = await update_autotrader_control(session, **payload)
    snapshot = await read_autotrader_snapshot(session)
    await create_autotrader_event(
        session,
        event_type="control_updated",
        source="api",
        message="Auto-trader control updated",
        payload=payload,
    )
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
    await create_autotrader_event(
        session,
        event_type="policies_updated",
        source="api",
        message="Auto-trader policies updated",
        payload=payload,
    )
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
                "event_id": row.event_id,
                "trace_id": row.trace_id,
                "mode": row.mode,
                "status": row.status,
                "notional_usd": row.notional_usd,
                "entry_price": row.entry_price,
                "effective_price": row.effective_price,
                "edge_percent": row.edge_percent,
                "confidence": row.confidence,
                "actual_profit": row.actual_profit,
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
                "event_id": row.event_id,
                "trace_id": row.trace_id,
                "policy_snapshot": row.policy_snapshot_json,
                "risk_snapshot": row.risk_snapshot_json,
                "payload": row.payload_json,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }


@router.get("/exposure")
async def get_auto_trader_exposure_route(session: AsyncSession = Depends(get_db_session)):
    return await get_autotrader_exposure(session)


@router.get("/metrics")
async def get_auto_trader_metrics_route(session: AsyncSession = Depends(get_db_session)):
    return await get_autotrader_metrics(session)


@router.post("/live/preflight")
async def live_preflight(
    request: AutoTraderLivePreflightRequest,
    session: AsyncSession = Depends(get_db_session),
):
    control = await read_autotrader_control(session)
    snapshot = await read_autotrader_snapshot(session)
    policies = await read_autotrader_policies(session)
    checks = _live_preflight_checks(control=control, snapshot=snapshot, policies=policies)
    row = await create_preflight_run(
        session,
        requested_mode=request.mode,
        checks=checks,
        requested_by=request.requested_by,
    )
    await create_autotrader_event(
        session,
        event_type="live_preflight",
        source="api",
        operator=request.requested_by,
        message=f"Live preflight {row.status}",
        payload={
            "preflight_id": row.id,
            "requested_mode": request.mode,
            "status": row.status,
            "checks": checks,
        },
    )
    return {
        "status": row.status,
        "preflight_id": row.id,
        "requested_mode": request.mode,
        "checks": checks,
        "failed_checks": row.failed_checks_json or [],
    }


@router.post("/live/arm")
async def live_arm(
    request: AutoTraderLiveArmRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        arm_payload = await arm_preflight_run(
            session,
            preflight_id=request.preflight_id,
            ttl_seconds=request.ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await create_autotrader_event(
        session,
        event_type="live_armed",
        source="api",
        operator=request.requested_by,
        message="Live start armed",
        payload=arm_payload,
    )
    return {"status": "armed", **arm_payload}


@router.post("/live/start")
async def live_start(
    request: AutoTraderLiveStartRequest,
    session: AsyncSession = Depends(get_db_session),
):
    _assert_not_globally_paused()

    token_check = await consume_arm_token(session, arm_token=request.arm_token)
    if not token_check.get("ok"):
        raise HTTPException(
            status_code=400,
            detail=f"Live start blocked: {token_check.get('reason', 'invalid_token')}",
        )

    control = await update_autotrader_control(
        session,
        is_enabled=True,
        is_paused=False,
        mode=request.mode,
        kill_switch=False,
    )
    await create_autotrader_event(
        session,
        event_type="live_started",
        source="api",
        operator=request.requested_by,
        message="Live auto-trader started",
        payload={"mode": request.mode, "preflight_id": token_check.get("preflight_id")},
    )
    return {
        "status": "started",
        "control": control,
        "preflight_id": token_check.get("preflight_id"),
    }


@router.post("/live/stop")
async def live_stop(
    requested_by: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_autotrader_control(
        session,
        is_enabled=False,
        is_paused=True,
    )
    await create_autotrader_event(
        session,
        event_type="live_stopped",
        source="api",
        operator=requested_by,
        message="Live auto-trader stopped",
    )
    return {"status": "stopped", "control": control}


@router.post("/live/kill-switch")
async def live_kill_switch(
    request: AutoTraderLiveKillSwitchRequest,
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_autotrader_control(
        session,
        kill_switch=request.enabled,
        is_paused=True if request.enabled else None,
        is_enabled=False if request.enabled else None,
    )
    await create_autotrader_event(
        session,
        event_type="kill_switch",
        severity="warn" if request.enabled else "info",
        source="api",
        operator=request.requested_by,
        message="Kill switch toggled",
        payload={"enabled": request.enabled},
    )
    return {"status": "updated", "control": control}


@router.post("/start")
async def start_auto_trader(
    mode: str = Query(default="paper"),
    account_id: Optional[str] = Query(default=None),
    arm_token: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    _assert_not_globally_paused()

    if mode == "live":
        if not arm_token:
            raise HTTPException(
                status_code=400,
                detail="Live mode requires preflight+arm token via /auto-trader/live/* endpoints",
            )
        token_check = await consume_arm_token(session, arm_token=arm_token)
        if not token_check.get("ok"):
            raise HTTPException(
                status_code=400,
                detail=f"Live start blocked: {token_check.get('reason', 'invalid_token')}",
            )

    control_update_payload: dict[str, Any] = {
        "is_enabled": True,
        "is_paused": False,
        "mode": mode,
        "kill_switch": False,
    }
    if mode != "live" and account_id is not None:
        before_control = await read_autotrader_control(session)
        merged_settings = dict(before_control.get("settings") or {})
        merged_settings["paper_account_id"] = account_id
        control_update_payload["settings"] = merged_settings

    control = await update_autotrader_control(session, **control_update_payload)
    await create_autotrader_event(
        session,
        event_type="started",
        source="api",
        message="Auto-trader started",
        payload={"mode": mode, "account_id": account_id},
    )
    return {"status": "started", "control": control}


@router.post("/pause")
async def pause_auto_trader(session: AsyncSession = Depends(get_db_session)):
    control = await update_autotrader_control(session, is_paused=True)
    await create_autotrader_event(
        session,
        event_type="paused",
        source="api",
        message="Auto-trader paused",
    )
    return {"status": "paused", "control": control}


@router.post("/stop")
async def stop_auto_trader(session: AsyncSession = Depends(get_db_session)):
    control = await update_autotrader_control(
        session,
        is_enabled=False,
        is_paused=True,
    )
    await create_autotrader_event(
        session,
        event_type="stopped",
        source="api",
        message="Auto-trader stopped",
    )
    return {"status": "stopped", "control": control}


@router.post("/run-once")
async def run_auto_trader_once(session: AsyncSession = Depends(get_db_session)):
    _assert_not_globally_paused()

    control = await update_autotrader_control(session, requested_run=True)
    await create_autotrader_event(
        session,
        event_type="run_once_requested",
        source="api",
        message="Manual run requested",
    )
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
        is_enabled=False if enabled else None,
    )
    await create_autotrader_event(
        session,
        event_type="kill_switch",
        severity="warn" if enabled else "info",
        source="api",
        message="Kill switch toggled",
        payload={"enabled": enabled},
    )
    return {"status": "updated", "control": control}
