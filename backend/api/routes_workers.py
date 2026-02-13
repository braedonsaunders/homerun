"""Worker control/status routes for isolated pipeline workers."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services import discovery_shared_state, shared_state
from services.news import shared_state as news_shared_state
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    read_orchestrator_control,
    update_orchestrator_control,
)
from services.weather import shared_state as weather_shared_state
from services.worker_state import (
    clear_worker_run_request,
    read_worker_control,
    read_worker_snapshot,
    request_worker_run,
    set_worker_interval,
    set_worker_paused,
)

router = APIRouter(prefix="/workers", tags=["Workers"])
ALLOWED_WORKERS = {
    "scanner",
    "news",
    "weather",
    "crypto",
    "tracked_traders",
    "trader_orchestrator",
    "discovery",
    "world_intelligence",
}
WORKER_DISPLAY_ORDER = (
    "scanner",
    "discovery",
    "weather",
    "news",
    "crypto",
    "tracked_traders",
    "trader_orchestrator",
    "world_intelligence",
)
GENERIC_WORKERS = ("crypto", "tracked_traders", "world_intelligence")


def _normalize_worker_name(raw: str) -> str:
    name = (raw or "").strip().lower().replace("-", "_")
    if name.endswith("_worker"):
        name = name[:-7]
    alias = {
        "tracked_traders": "tracked_traders",
        "trackedtraders": "tracked_traders",
        "scanner": "scanner",
        "news": "news",
        "weather": "weather",
        "crypto": "crypto",
        "trader_orchestrator": "trader_orchestrator",
        "discovery": "discovery",
        "world_intelligence": "world_intelligence",
        "worldintelligence": "world_intelligence",
    }
    return alias.get(name, name)


def _assert_supported_worker(name: str) -> None:
    if name not in ALLOWED_WORKERS:
        raise HTTPException(
            status_code=404,
            detail=(f"Unknown worker '{name}'. Supported workers: {sorted(ALLOWED_WORKERS)}"),
        )


async def _worker_detail(session: AsyncSession, worker_name: str) -> dict:
    snapshot = await read_worker_snapshot(session, worker_name)

    if worker_name == "scanner":
        control = await shared_state.read_scanner_control(session)
        snapshot["control"] = {
            "is_enabled": bool(control.get("is_enabled", True)),
            "is_paused": bool(control.get("is_paused", False)),
            "interval_seconds": int(control.get("scan_interval_seconds") or 60),
            "requested_run_at": control.get("requested_scan_at").isoformat()
            if control.get("requested_scan_at")
            else None,
        }
    elif worker_name == "news":
        control = await news_shared_state.read_news_control(session)
        snapshot["control"] = {
            "is_enabled": bool(control.get("is_enabled", True)),
            "is_paused": bool(control.get("is_paused", False)),
            "interval_seconds": int(control.get("scan_interval_seconds") or 120),
            "requested_run_at": control.get("requested_scan_at").isoformat()
            if control.get("requested_scan_at")
            else None,
        }
    elif worker_name == "weather":
        control = await weather_shared_state.read_weather_control(session)
        snapshot["control"] = {
            "is_enabled": bool(control.get("is_enabled", True)),
            "is_paused": bool(control.get("is_paused", False)),
            "interval_seconds": int(control.get("scan_interval_seconds") or 14400),
            "requested_run_at": control.get("requested_scan_at").isoformat()
            if control.get("requested_scan_at")
            else None,
        }
    elif worker_name == "discovery":
        control = await discovery_shared_state.read_discovery_control(session)
        snapshot["control"] = {
            "is_enabled": bool(control.get("is_enabled", True)),
            "is_paused": bool(control.get("is_paused", False)),
            "interval_seconds": int((control.get("run_interval_minutes") or 60) * 60),
            "priority_backlog_mode": bool(control.get("priority_backlog_mode", True)),
            "requested_run_at": control.get("requested_run_at").isoformat()
            if control.get("requested_run_at")
            else None,
        }
    elif worker_name == "trader_orchestrator":
        control = await read_orchestrator_control(session)
        snapshot["control"] = control
    else:
        control = await read_worker_control(session, worker_name)
        snapshot["control"] = {
            "is_enabled": bool(control.get("is_enabled", True)),
            "is_paused": bool(control.get("is_paused", False)),
            "interval_seconds": int(control.get("interval_seconds") or 60),
            "requested_run_at": control.get("requested_run_at").isoformat()
            if control.get("requested_run_at")
            else None,
        }

    return snapshot


async def _collect_workers(session: AsyncSession) -> list[dict]:
    detail: list[dict] = []
    for worker_name in WORKER_DISPLAY_ORDER:
        detail.append(await _worker_detail(session, worker_name))
    return detail


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
async def get_workers_status(session: AsyncSession = Depends(get_db_session)):
    return {"workers": await _collect_workers(session)}


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
        await update_orchestrator_control(session, is_paused=False, is_enabled=True)
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
        await update_orchestrator_control(session, is_paused=True)
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
        await update_orchestrator_control(session, requested_run_at=datetime.utcnow())
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
