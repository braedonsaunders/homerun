"""API routes for the trader orchestrator control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    DEFAULT_PENDING_LIVE_EXIT_GUARD,
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
from services.worker_state import summarize_worker_stats
from utils.utcnow import utcnow

router = APIRouter(prefix="/trader-orchestrator", tags=["Trader Orchestrator"])
_ORCHESTRATOR_OVERVIEW_CACHE_TTL_SECONDS = 5.0
_orchestrator_overview_cache: dict | None = None
_orchestrator_overview_cache_updated_at: datetime | None = None


def _clear_orchestrator_caches() -> None:
    global _orchestrator_overview_cache
    global _orchestrator_overview_cache_updated_at
    _orchestrator_overview_cache = None
    _orchestrator_overview_cache_updated_at = None


class StartRequest(BaseModel):
    mode: Optional[str] = Field(default=None, description="shadow | live")
    selected_account_id: str = Field(..., min_length=1)
    requested_by: Optional[str] = None

    @field_validator("selected_account_id")
    @classmethod
    def _validate_selected_account_id(cls, value: str) -> str:
        account_id = str(value or "").strip()
        if not account_id:
            raise ValueError("selected_account_id is required")
        return account_id


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
    selected_account_id: str = Field(..., min_length=1)
    requested_by: Optional[str] = None

    @field_validator("selected_account_id")
    @classmethod
    def _validate_selected_account_id(cls, value: str) -> str:
        account_id = str(value or "").strip()
        if not account_id:
            raise ValueError("selected_account_id is required")
        return account_id


class LiveStopRequest(BaseModel):
    requested_by: Optional[str] = None


class GlobalRiskSettingsRequest(BaseModel):
    max_gross_exposure_usd: float = Field(..., ge=1.0, le=1_000_000.0)
    max_daily_loss_usd: float = Field(..., ge=0.0, le=1_000_000.0)
    max_orders_per_cycle: int = Field(..., ge=1, le=1000)


class PendingLiveExitGuardSettingsRequest(BaseModel):
    max_pending_exits: int = Field(default=0, ge=0, le=1000)
    identity_guard_enabled: bool = True
    terminal_statuses: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
    )

    @field_validator("terminal_statuses")
    @classmethod
    def _validate_terminal_statuses(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            status = str(item or "").strip().lower()
            if not status:
                continue
            if status in seen:
                continue
            seen.add(status)
            normalized.append(status)
        if not normalized:
            return list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
        return normalized


class LiveRiskClampsSettingsRequest(BaseModel):
    enforce_allow_averaging_off: bool | None = None
    min_cooldown_seconds: int | None = Field(default=None, ge=0, le=86400)
    max_consecutive_losses_cap: int | None = Field(default=None, ge=1, le=1000)
    max_open_orders_cap: int | None = Field(default=None, ge=1, le=1000)
    max_open_positions_cap: int | None = Field(default=None, ge=1, le=1000)
    max_trade_notional_usd_cap: float | None = Field(default=None, ge=1.0, le=1_000_000.0)
    max_per_market_exposure_usd_cap: float | None = Field(default=None, ge=1.0, le=1_000_000.0)
    max_orders_per_cycle_cap: int | None = Field(default=None, ge=1, le=1000)
    enforce_halt_on_consecutive_losses: bool | None = None
    # Caps for the freshly-wired risk knobs. Same precedence as the
    # existing clamps: global is a ceiling, per-trader is the effective
    # value if below the ceiling.
    max_daily_spend_usd_cap: float | None = Field(default=None, ge=1.0, le=100_000_000.0)
    max_spread_bps_cap: float | None = Field(default=None, ge=0.0, le=10_000.0)
    slippage_bps_cap: float | None = Field(default=None, ge=0.0, le=10_000.0)
    retry_limit_cap: int | None = Field(default=None, ge=0, le=50)
    retry_backoff_ms_cap: int | None = Field(default=None, ge=0, le=60_000)
    order_ttl_seconds_cap: int | None = Field(default=None, ge=1, le=86_400)


class LiveMarketContextSettingsRequest(BaseModel):
    enabled: bool = True
    history_window_seconds: int = Field(default=7200, ge=300, le=21600)
    history_fidelity_seconds: int = Field(default=300, ge=30, le=1800)
    max_history_points: int = Field(default=120, ge=20, le=240)
    timeout_seconds: float = Field(default=4.0, ge=1.0, le=12.0)
    strict_ws_pricing_only: bool = True
    max_market_data_age_ms: int = Field(default=10000, ge=25, le=30000)


class LiveProviderHealthSettingsRequest(BaseModel):
    window_seconds: int = Field(default=180, ge=30, le=900)
    min_errors: int = Field(default=2, ge=1, le=20)
    block_seconds: int = Field(default=120, ge=15, le=3600)


class GlobalRuntimeSettingsRequest(BaseModel):
    pending_live_exit_guard: PendingLiveExitGuardSettingsRequest = Field(
        default_factory=PendingLiveExitGuardSettingsRequest
    )
    live_risk_clamps: LiveRiskClampsSettingsRequest = Field(default_factory=LiveRiskClampsSettingsRequest)
    live_market_context: LiveMarketContextSettingsRequest = Field(default_factory=LiveMarketContextSettingsRequest)
    live_provider_health: LiveProviderHealthSettingsRequest = Field(default_factory=LiveProviderHealthSettingsRequest)
    trader_cycle_timeout_seconds: float | None = Field(default=None, ge=30.0, le=180.0)
    runtime_trigger_cycle_timeout_seconds: float | None = Field(default=None, ge=3.0, le=60.0)


class UpdateSettingsRequest(BaseModel):
    run_interval_seconds: int | None = Field(default=None, ge=1, le=300)
    global_risk: GlobalRiskSettingsRequest | None = None
    global_runtime: GlobalRuntimeSettingsRequest | None = None
    requested_by: Optional[str] = None


def _assert_not_globally_paused() -> None:
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )


@router.get("/overview")
async def get_overview(session: AsyncSession = Depends(get_db_session)):
    global _orchestrator_overview_cache, _orchestrator_overview_cache_updated_at
    if (
        _orchestrator_overview_cache is not None
        and _orchestrator_overview_cache_updated_at is not None
        and (utcnow() - _orchestrator_overview_cache_updated_at).total_seconds()
        <= _ORCHESTRATOR_OVERVIEW_CACHE_TTL_SECONDS
    ):
        return _orchestrator_overview_cache
    payload = await get_orchestrator_overview(session)
    if session.in_transaction():
        await session.rollback()
    _orchestrator_overview_cache = payload
    _orchestrator_overview_cache_updated_at = utcnow()
    return payload


@router.get("/status")
async def get_status(session: AsyncSession = Depends(get_db_session)):
    control = await read_orchestrator_control(session)
    snapshot = await read_orchestrator_snapshot(session)
    if isinstance(snapshot, dict):
        snapshot = dict(snapshot)
        snapshot["stats"] = summarize_worker_stats(snapshot.get("stats"))
    config = await compose_trader_orchestrator_config(session, control=control)
    if session.in_transaction():
        await session.rollback()
    payload = {
        "control": control,
        "snapshot": snapshot,
        "config": config,
    }
    return payload


@router.put("/settings")
async def update_orchestrator_settings(
    request: UpdateSettingsRequest,
    session: AsyncSession = Depends(get_db_session),
):
    update_kwargs: dict[str, object] = {}
    settings_updates: dict[str, object] = {}

    if request.run_interval_seconds is not None:
        update_kwargs["run_interval_seconds"] = int(request.run_interval_seconds)
    if request.global_risk is not None:
        settings_updates["global_risk"] = request.global_risk.model_dump()
    if request.global_runtime is not None:
        runtime_dump = request.global_runtime.model_dump()
        # Only persist clamp fields that were explicitly set (non-None)
        runtime_dump["live_risk_clamps"] = request.global_runtime.live_risk_clamps.model_dump(exclude_none=True)
        runtime_dump["live_risk_clamps_explicit"] = True
        settings_updates["global_runtime"] = runtime_dump
    if settings_updates:
        update_kwargs["settings_json"] = settings_updates

    control = await update_orchestrator_control(session, **update_kwargs)
    config = await compose_trader_orchestrator_config(session, control=control)

    if request.requested_by or update_kwargs:
        await create_trader_event(
            session,
            event_type="settings_updated",
            source="trader_orchestrator",
            operator=request.requested_by,
            message="Trader orchestrator settings updated",
            payload={
                "updated_fields": sorted(update_kwargs.keys()),
                "run_interval_seconds": config.get("run_interval_seconds"),
                "global_risk": config.get("global_risk"),
                "global_runtime": config.get("global_runtime"),
            },
        )

    _clear_orchestrator_caches()
    return {
        "status": "updated",
        "control": control,
        "config": config,
    }


@router.post("/start")
async def start_orchestrator(
    request: StartRequest,
    session: AsyncSession = Depends(get_db_session),
):
    _assert_not_globally_paused()
    mode = str(request.mode or "shadow").strip().lower()
    if mode == "live":
        raise HTTPException(
            status_code=422,
            detail=(
                "Live mode cannot be started via /start. "
                "Use the /live/preflight -> /live/arm -> /live/start ceremony instead."
            ),
        )
    if mode != "shadow":
        raise HTTPException(
            status_code=422,
            detail="mode must be 'shadow'. Use /live/start for live mode.",
        )
    selected_account_id = str(request.selected_account_id or "").strip()
    if selected_account_id.startswith("live:"):
        raise HTTPException(
            status_code=422,
            detail="Selected global account is live. Use /live/start for live trading.",
        )

    control = await update_orchestrator_control(
        session,
        is_enabled=True,
        is_paused=False,
        mode=mode,
        requested_run_at=utcnow(),
        settings_json={
            "selected_account_id": selected_account_id,
            "shadow_account_id": selected_account_id,
        },
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
        payload={"mode": mode, "selected_account_id": selected_account_id},
    )
    _clear_orchestrator_caches()
    return {"status": "started", "control": control}


@router.post("/stop")
async def stop_orchestrator(
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_orchestrator_control(
        session,
        is_enabled=False,
        is_paused=True,
        requested_run_at=None,
        settings_json={
            "selected_account_id": None,
            "shadow_account_id": None,
        },
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
    _clear_orchestrator_caches()
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
    _clear_orchestrator_caches()
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
    _clear_orchestrator_caches()
    return result


@router.post("/live/arm")
async def arm_live(
    request: LiveArmRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        result = await arm_live_start(
            session,
            preflight_id=request.preflight_id,
            ttl_seconds=request.ttl_seconds,
            requested_by=request.requested_by,
        )
        _clear_orchestrator_caches()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/live/start")
async def start_live(
    request: LiveStartRequest,
    session: AsyncSession = Depends(get_db_session),
):
    _assert_not_globally_paused()
    selected_account_id = str(request.selected_account_id or "").strip()
    if not selected_account_id.startswith("live:"):
        raise HTTPException(
            status_code=422,
            detail="Selected global account is not live. Use /start for shadow trading.",
        )
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
        settings_json={
            "selected_account_id": selected_account_id,
            "shadow_account_id": None,
        },
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
        payload={"mode": request.mode, "selected_account_id": selected_account_id},
    )
    _clear_orchestrator_caches()
    return {"status": "started", "control": control}


@router.post("/live/stop")
async def stop_live(
    request: LiveStopRequest = LiveStopRequest(),
    session: AsyncSession = Depends(get_db_session),
):
    control = await update_orchestrator_control(
        session,
        mode="shadow",
        is_enabled=False,
        is_paused=True,
        requested_run_at=None,
        settings_json={
            "selected_account_id": None,
            "shadow_account_id": None,
        },
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
    _clear_orchestrator_caches()
    return {"status": "stopped", "control": control}
