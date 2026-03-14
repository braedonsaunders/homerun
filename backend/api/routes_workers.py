"""Worker control/status routes for isolated pipeline workers."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    DiscoveryControl,
    NewsWorkflowControl,
    ScannerControl,
    WeatherControl,
    WorkerControl,
    get_db_session,
    recover_pool,
)
from services import discovery_shared_state, shared_state
from services.news import shared_state as news_shared_state
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
    read_orchestrator_snapshot,
    read_orchestrator_control,
    write_orchestrator_snapshot,
    update_orchestrator_control,
)
from services.weather import shared_state as weather_shared_state
from services.worker_state import (
    clear_worker_run_request,
    list_worker_snapshots,
    read_worker_control,
    request_worker_run,
    set_worker_interval,
    set_worker_paused,
    summarize_worker_snapshot,
    summarize_worker_stats,
)
from utils.utcnow import utcnow

router = APIRouter(prefix="/workers", tags=["Workers"])
ALLOWED_WORKERS = {
    "scanner",
    "scanner_slo",
    "news",
    "weather",
    "crypto",
    "tracked_traders",
    "trader_orchestrator",
    "trader_reconciliation",
    "redeemer",
    "discovery",
    "events",
}
WORKER_DISPLAY_ORDER = (
    "scanner",
    "scanner_slo",
    "discovery",
    "weather",
    "news",
    "crypto",
    "tracked_traders",
    "trader_orchestrator",
    "trader_reconciliation",
    "redeemer",
    "events",
)
GENERIC_WORKERS = ("scanner_slo", "crypto", "tracked_traders", "trader_reconciliation", "redeemer", "events")
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BASE_DELAY_SECONDS = 0.2
DB_RETRY_MAX_DELAY_SECONDS = 1.5
_WORKERS_STATUS_CACHE_TTL_SECONDS = 5.0
_workers_status_cache: dict | None = None
_workers_status_cache_updated_at: datetime | None = None
_workers_status_refresh_task: asyncio.Task | None = None


from utils.retry import is_retryable_db_error as _is_retryable_db_error  # noqa: E402
from utils.retry import db_retry_delay as _db_retry_delay  # noqa: E402


def _normalize_worker_name(raw: str) -> str:
    name = (raw or "").strip().lower().replace("-", "_")
    if name.endswith("_worker"):
        name = name[:-7]
    return name


def _assert_supported_worker(name: str) -> None:
    if name not in ALLOWED_WORKERS:
        raise HTTPException(
            status_code=404,
            detail=(f"Unknown worker '{name}'. Supported workers: {sorted(ALLOWED_WORKERS)}"),
        )


async def _generic_worker_controls(session: AsyncSession) -> dict[str, dict]:
    rows = (
        await session.execute(
            select(WorkerControl).where(WorkerControl.worker_name.in_(GENERIC_WORKERS))
        )
    ).scalars().all()
    return {
        str(row.worker_name): {
            "is_enabled": bool(row.is_enabled),
            "is_paused": bool(row.is_paused),
            "interval_seconds": int(row.interval_seconds or 60),
            "requested_run_at": row.requested_run_at.isoformat() if row.requested_run_at else None,
        }
        for row in rows
    }


async def _scanner_control_payload(session: AsyncSession) -> dict:
    row = (
        await session.execute(select(ScannerControl).where(ScannerControl.id == shared_state.CONTROL_ID))
    ).scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "interval_seconds": 60,
            "requested_run_at": None,
            "heavy_lane_forced_degraded": False,
            "heavy_lane_degraded_reason": None,
            "heavy_lane_degraded_until": None,
        }
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int(row.scan_interval_seconds or 60),
        "requested_run_at": row.requested_scan_at.isoformat() if row.requested_scan_at else None,
        "heavy_lane_forced_degraded": bool(row.heavy_lane_forced_degraded),
        "heavy_lane_degraded_reason": row.heavy_lane_degraded_reason,
        "heavy_lane_degraded_until": row.heavy_lane_degraded_until.isoformat() if row.heavy_lane_degraded_until else None,
    }


async def _news_control_payload(session: AsyncSession) -> dict:
    row = (
        await session.execute(select(NewsWorkflowControl).where(NewsWorkflowControl.id == news_shared_state.NEWS_CONTROL_ID))
    ).scalar_one_or_none()
    if row is None:
        return {"is_enabled": True, "is_paused": False, "interval_seconds": 120, "requested_run_at": None}
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int(row.scan_interval_seconds or 120),
        "requested_run_at": row.requested_scan_at.isoformat() if row.requested_scan_at else None,
    }


async def _weather_control_payload(session: AsyncSession) -> dict:
    row = (
        await session.execute(select(WeatherControl).where(WeatherControl.id == weather_shared_state.WEATHER_CONTROL_ID))
    ).scalar_one_or_none()
    if row is None:
        return {"is_enabled": True, "is_paused": False, "interval_seconds": 14400, "requested_run_at": None}
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int(row.scan_interval_seconds or 14400),
        "requested_run_at": row.requested_scan_at.isoformat() if row.requested_scan_at else None,
    }


async def _discovery_control_payload(session: AsyncSession) -> dict:
    row = (
        await session.execute(
            select(DiscoveryControl).where(DiscoveryControl.id == discovery_shared_state.DISCOVERY_CONTROL_ID)
        )
    ).scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "interval_seconds": 3600,
            "priority_backlog_mode": True,
            "requested_run_at": None,
        }
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int((row.run_interval_minutes or 60) * 60),
        "priority_backlog_mode": bool(row.priority_backlog_mode),
        "requested_run_at": row.requested_run_at.isoformat() if row.requested_run_at else None,
    }


async def _collect_workers(session: AsyncSession) -> list[dict]:
    snapshots = {
        str(row.get("worker_name") or ""): dict(row)
        for row in await list_worker_snapshots(
            session,
            include_stats=True,
            stats_mode="summary",
            worker_names=WORKER_DISPLAY_ORDER,
        )
        if isinstance(row, dict) and str(row.get("worker_name") or "")
    }
    generic_controls = await _generic_worker_controls(session)
    scanner_control = await _scanner_control_payload(session)
    news_control = await _news_control_payload(session)
    weather_control = await _weather_control_payload(session)
    discovery_control = await _discovery_control_payload(session)
    orchestrator_control = await read_orchestrator_control(session)
    orchestrator_snapshot = await read_orchestrator_snapshot(session)
    snapshots["trader_orchestrator"] = {
        "worker_name": "trader_orchestrator",
        "running": bool(orchestrator_snapshot.get("running")),
        "enabled": bool(orchestrator_snapshot.get("enabled")),
        "current_activity": orchestrator_snapshot.get("current_activity"),
        "interval_seconds": int(orchestrator_snapshot.get("interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        "last_run_at": orchestrator_snapshot.get("last_run_at"),
        "lag_seconds": None,
        "last_error": orchestrator_snapshot.get("last_error"),
        "stats": summarize_worker_stats(orchestrator_snapshot.get("stats", {})),
        "updated_at": orchestrator_snapshot.get("updated_at"),
        "control": orchestrator_control,
    }

    detail: list[dict] = []
    for worker_name in WORKER_DISPLAY_ORDER:
        snapshot = dict(snapshots.get(worker_name) or {"worker_name": worker_name, "stats": {}})
        if worker_name == "scanner":
            snapshot["control"] = scanner_control
        elif worker_name == "news":
            snapshot["control"] = news_control
        elif worker_name == "weather":
            snapshot["control"] = weather_control
        elif worker_name == "discovery":
            snapshot["control"] = discovery_control
        elif worker_name == "trader_orchestrator":
            snapshot["control"] = orchestrator_control
        else:
            snapshot["control"] = generic_controls.get(
                worker_name,
                {
                    "is_enabled": True,
                    "is_paused": False,
                    "interval_seconds": 60,
                    "requested_run_at": None,
                },
            )
        detail.append(summarize_worker_snapshot(snapshot))
    return detail


async def _refresh_workers_status_payload() -> dict:
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            async with AsyncSessionLocal() as session:
                workers = await _collect_workers(session)
                if session.in_transaction():
                    await session.rollback()
            return {"workers": workers}
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            if not _is_retryable_db_error(exc):
                raise
            if attempt >= DB_RETRY_ATTEMPTS - 1:
                raise HTTPException(status_code=503, detail="Database is busy; please retry.") from exc
            try:
                await recover_pool()
            except Exception:
                pass
            await asyncio.sleep(_db_retry_delay(attempt))
    raise HTTPException(status_code=503, detail="Database is busy; please retry.")


async def _set_all_workers_paused(session: AsyncSession, paused: bool) -> None:
    await shared_state.set_scanner_paused(session, paused)
    await news_shared_state.set_news_paused(session, paused)
    await weather_shared_state.set_weather_paused(session, paused)
    await discovery_shared_state.set_discovery_paused(session, paused)
    await update_orchestrator_control(
        session,
        is_paused=paused,
        requested_run_at=None if paused else None,
    )

    if paused:
        await shared_state.clear_scan_request(session)
        await news_shared_state.clear_news_scan_request(session)
        await weather_shared_state.clear_weather_scan_request(session)
        await discovery_shared_state.clear_discovery_run_request(session)

    for worker_name in GENERIC_WORKERS:
        await set_worker_paused(session, worker_name, paused)
        if paused:
            await clear_worker_run_request(session, worker_name)

    if paused:
        global_pause_state.pause()
    else:
        global_pause_state.resume()


@router.get("/status")
async def get_workers_status():
    global _workers_status_cache, _workers_status_cache_updated_at, _workers_status_refresh_task
    if _workers_status_refresh_task is not None and _workers_status_refresh_task.done():
        try:
            _workers_status_cache = _workers_status_refresh_task.result()
            _workers_status_cache_updated_at = utcnow()
        except Exception:
            pass
        finally:
            _workers_status_refresh_task = None
    if (
        _workers_status_cache is not None
        and _workers_status_cache_updated_at is not None
        and (utcnow() - _workers_status_cache_updated_at).total_seconds() <= _WORKERS_STATUS_CACHE_TTL_SECONDS
    ):
        return _workers_status_cache
    if _workers_status_refresh_task is None:
        _workers_status_refresh_task = asyncio.create_task(
            _refresh_workers_status_payload(),
            name="workers-status-refresh",
        )
        if _workers_status_cache is not None:
            return _workers_status_cache
    try:
        payload = await asyncio.wait_for(asyncio.shield(_workers_status_refresh_task), timeout=2.0)
        _workers_status_cache = payload
        _workers_status_cache_updated_at = utcnow()
        _workers_status_refresh_task = None
        return payload
    except Exception as exc:
        if _workers_status_cache is not None:
            return _workers_status_cache
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=503, detail="Database is busy; please retry.") from exc


@router.post("/pause-all")
async def pause_all_workers(session: AsyncSession = Depends(get_db_session)):
    await _set_all_workers_paused(session, True)
    return {"status": "paused", "workers": await _collect_workers(session)}


@router.post("/resume-all")
async def resume_all_workers(session: AsyncSession = Depends(get_db_session)):
    await _set_all_workers_paused(session, False)
    return {"status": "running", "workers": await _collect_workers(session)}


@router.post("/{worker}/start")
async def start_worker(worker: str, session: AsyncSession = Depends(get_db_session)):
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )

    name = _normalize_worker_name(worker)
    _assert_supported_worker(name)

    if name == "scanner":
        await shared_state.set_scanner_paused(session, False)
    elif name == "news":
        await news_shared_state.set_news_paused(session, False)
    elif name == "weather":
        await weather_shared_state.set_weather_paused(session, False)
    elif name == "discovery":
        await discovery_shared_state.set_discovery_paused(session, False)
    elif name == "trader_orchestrator":
        control = await update_orchestrator_control(
            session,
            is_paused=False,
            is_enabled=True,
            requested_run_at=utcnow(),
        )
        await write_orchestrator_snapshot(
            session,
            running=False,
            enabled=True,
            current_activity="Start command queued",
            interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        )
    else:
        await set_worker_paused(session, name, False)

    return {"status": "started", "worker": await _worker_detail(session, name)}


@router.post("/{worker}/pause")
async def pause_worker(worker: str, session: AsyncSession = Depends(get_db_session)):
    name = _normalize_worker_name(worker)
    _assert_supported_worker(name)

    if name == "scanner":
        await shared_state.set_scanner_paused(session, True)
    elif name == "news":
        await news_shared_state.set_news_paused(session, True)
    elif name == "weather":
        await weather_shared_state.set_weather_paused(session, True)
    elif name == "discovery":
        await discovery_shared_state.set_discovery_paused(session, True)
    elif name == "trader_orchestrator":
        control = await update_orchestrator_control(session, is_paused=True)
        await write_orchestrator_snapshot(
            session,
            running=False,
            enabled=False,
            current_activity="Manual stop requested",
            interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        )
    else:
        await set_worker_paused(session, name, True)

    return {"status": "paused", "worker": await _worker_detail(session, name)}


@router.post("/{worker}/run-once")
async def run_worker_once(worker: str, session: AsyncSession = Depends(get_db_session)):
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Resume execution before queueing runs.",
        )

    name = _normalize_worker_name(worker)
    _assert_supported_worker(name)

    if name == "scanner":
        await shared_state.request_one_scan(session)
    elif name == "news":
        await news_shared_state.request_one_news_scan(session)
    elif name == "weather":
        await weather_shared_state.request_one_weather_scan(session)
    elif name == "discovery":
        await discovery_shared_state.request_one_discovery_run(session)
    elif name == "trader_orchestrator":
        await update_orchestrator_control(session, requested_run_at=utcnow())
    else:
        await request_worker_run(session, name)

    return {"status": "queued", "worker": await _worker_detail(session, name)}


@router.post("/{worker}/interval")
async def set_worker_run_interval(
    worker: str,
    interval_seconds: int = Query(..., ge=1, le=86400),
    session: AsyncSession = Depends(get_db_session),
):
    name = _normalize_worker_name(worker)
    _assert_supported_worker(name)

    if name == "scanner":
        await shared_state.set_scanner_interval(session, interval_seconds)
    elif name == "news":
        await news_shared_state.set_news_interval(session, interval_seconds)
    elif name == "weather":
        await weather_shared_state.set_weather_interval(session, interval_seconds)
    elif name == "discovery":
        await discovery_shared_state.set_discovery_interval(session, max(1, interval_seconds // 60))
    elif name == "trader_orchestrator":
        await update_orchestrator_control(session, run_interval_seconds=interval_seconds)
    else:
        await set_worker_interval(session, name, interval_seconds)

    return {"status": "updated", "worker": await _worker_detail(session, name)}
