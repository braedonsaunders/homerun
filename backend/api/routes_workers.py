"""Worker control/status routes for isolated pipeline workers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services import discovery_shared_state, shared_state
from services.autotrader_state import read_autotrader_control, update_autotrader_control
from services.news import shared_state as news_shared_state
from services.weather import shared_state as weather_shared_state
from services.worker_state import (
    list_worker_snapshots,
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
    "autotrader",
    "discovery",
}


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
        "autotrader": "autotrader",
        "auto_trader": "autotrader",
        "discovery": "discovery",
    }
    return alias.get(name, name)


def _assert_supported_worker(name: str) -> None:
    if name not in ALLOWED_WORKERS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown worker '{name}'. "
                f"Supported workers: {sorted(ALLOWED_WORKERS)}"
            ),
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
            "requested_run_at": control.get("requested_run_at").isoformat()
            if control.get("requested_run_at")
            else None,
        }
    elif worker_name == "autotrader":
        control = await read_autotrader_control(session)
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


@router.get("/status")
async def get_workers_status(session: AsyncSession = Depends(get_db_session)):
    rows = await list_worker_snapshots(session)
    detail = []
    for row in rows:
        name = row.get("worker_name")
        if not name:
            continue
        detail.append(await _worker_detail(session, name))
    return {"workers": detail}


@router.post("/{worker}/start")
async def start_worker(worker: str, session: AsyncSession = Depends(get_db_session)):
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
    elif name == "autotrader":
        await update_autotrader_control(session, is_paused=False, is_enabled=True)
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
    elif name == "autotrader":
        await update_autotrader_control(session, is_paused=True)
    else:
        await set_worker_paused(session, name, True)

    return {"status": "paused", "worker": await _worker_detail(session, name)}


@router.post("/{worker}/run-once")
async def run_worker_once(worker: str, session: AsyncSession = Depends(get_db_session)):
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
    elif name == "autotrader":
        await update_autotrader_control(session, requested_run=True)
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
        await discovery_shared_state.set_discovery_interval(
            session, max(1, interval_seconds // 60)
        )
    elif name == "autotrader":
        await update_autotrader_control(session, run_interval_seconds=interval_seconds)
    else:
        await set_worker_interval(session, name, interval_seconds)

    return {"status": "updated", "worker": await _worker_detail(session, name)}
