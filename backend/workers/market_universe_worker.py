"""Market universe worker: dedicated catalog refresh plane.

Owns upstream market/event fetches and writes canonical ``market_catalog``.
Scanner worker consumes this DB catalog and performs detection only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from config import apply_runtime_settings_overrides, settings
from models.database import AsyncSessionLocal
from services.scanner import scanner
from services.shared_state import read_market_catalog
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.worker_state import clear_worker_run_request, read_worker_control, write_worker_snapshot
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("market_universe_worker")


async def _read_catalog_stats() -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        _, _, metadata = await read_market_catalog(session)
    updated_at = metadata.get("updated_at")
    updated_iso = updated_at.isoformat() if isinstance(updated_at, datetime) else None
    return {
        "catalog_updated_at": updated_iso,
        "event_count": int(metadata.get("event_count") or 0),
        "market_count": int(metadata.get("market_count") or 0),
        "fetch_duration_seconds": (
            float(metadata.get("fetch_duration_seconds"))
            if metadata.get("fetch_duration_seconds") is not None
            else None
        ),
        "error": metadata.get("error"),
    }


async def _run_loop() -> None:
    worker_name = "market_universe"
    heartbeat_interval = max(
        1.0,
        float(getattr(settings, "MARKET_UNIVERSE_HEARTBEAT_INTERVAL_SECONDS", 5.0) or 5.0),
    )
    default_interval = max(
        30,
        int(getattr(settings, "MARKET_UNIVERSE_REFRESH_INTERVAL_SECONDS", 120) or 120),
    )
    refresh_timeout = max(
        30,
        int(getattr(settings, "MARKET_UNIVERSE_REFRESH_TIMEOUT_SECONDS", 300) or 300),
    )

    await scanner.load_settings()
    await scanner.load_plugins(source_keys=["scanner"])

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": default_interval,
        "activity": "Market universe worker started; first refresh pending.",
        "last_error": None,
        "last_run_at": None,
        "last_market_count": 0,
        "last_event_count": 0,
        "catalog_updated_at": None,
        "fetch_duration_seconds": None,
    }
    heartbeat_stop_event = asyncio.Event()
    next_scheduled_run_at: datetime | None = None

    async def _heartbeat_loop() -> None:
        while not heartbeat_stop_event.is_set():
            try:
                await write_worker_snapshot_loop_state(worker_name, state)
            except Exception as exc:
                state["last_error"] = str(exc)
                logger.warning("Market universe heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(heartbeat_stop_event.wait(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                continue

    async def write_worker_snapshot_loop_state(name: str, loop_state: dict[str, Any]) -> None:
        async with AsyncSessionLocal() as session:
            await write_worker_snapshot(
                session,
                name,
                running=True,
                enabled=bool(loop_state.get("enabled", True)),
                current_activity=str(loop_state.get("activity") or "Idle"),
                interval_seconds=int(loop_state.get("interval_seconds") or default_interval),
                last_run_at=loop_state.get("last_run_at"),
                last_error=(str(loop_state["last_error"]) if loop_state.get("last_error") is not None else None),
                stats={
                    "event_count": int(loop_state.get("last_event_count", 0) or 0),
                    "market_count": int(loop_state.get("last_market_count", 0) or 0),
                    "catalog_updated_at": loop_state.get("catalog_updated_at"),
                    "fetch_duration_seconds": loop_state.get("fetch_duration_seconds"),
                },
            )

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="market-universe-heartbeat")
    logger.info("Market universe worker started")

    try:
        while True:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(
                    session,
                    worker_name,
                    default_interval=default_interval,
                )
                try:
                    await apply_runtime_settings_overrides()
                except Exception as exc:
                    logger.warning("Market universe runtime settings refresh failed: %s", exc)
                try:
                    await refresh_strategy_runtime_if_needed(
                        session,
                        source_keys=["scanner"],
                    )
                except Exception as exc:
                    logger.warning("Market universe strategy refresh check failed: %s", exc)

            interval_seconds = max(
                30,
                int(control.get("interval_seconds") or default_interval),
            )
            paused = bool(control.get("is_paused", False))
            requested = control.get("requested_run_at") is not None
            enabled = bool(control.get("is_enabled", True)) and not paused
            now = datetime.now(timezone.utc)
            should_run_scheduled = enabled and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
            should_run = requested or should_run_scheduled

            state["enabled"] = enabled
            state["interval_seconds"] = interval_seconds

            if not should_run:
                state["activity"] = "Paused" if paused else "Idle - waiting for next catalog refresh."
                await asyncio.sleep(min(5.0, float(interval_seconds)))
                continue

            state["activity"] = "Refreshing market universe catalog..."
            try:
                market_count = await asyncio.wait_for(scanner.refresh_catalog(), timeout=float(refresh_timeout))
                catalog_stats = await _read_catalog_stats()
                state["last_market_count"] = int(catalog_stats.get("market_count") or market_count or 0)
                state["last_event_count"] = int(catalog_stats.get("event_count") or 0)
                state["catalog_updated_at"] = catalog_stats.get("catalog_updated_at")
                state["fetch_duration_seconds"] = catalog_stats.get("fetch_duration_seconds")
                state["last_error"] = None
                state["last_run_at"] = utcnow()
                state["activity"] = (
                    f"Market universe refresh complete: {state['last_event_count']} events, "
                    f"{state['last_market_count']} markets."
                )
                async with AsyncSessionLocal() as session:
                    await clear_worker_run_request(session, worker_name)
                next_scheduled_run_at = now + timedelta(seconds=interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state["last_error"] = str(exc)
                state["activity"] = f"Market universe refresh error: {exc}"
                logger.exception("Market universe refresh cycle failed: %s", exc)
                async with AsyncSessionLocal() as session:
                    await clear_worker_run_request(session, worker_name)
                next_scheduled_run_at = now + timedelta(seconds=interval_seconds)

            await asyncio.sleep(0.1)
    finally:
        heartbeat_stop_event.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


async def start_loop() -> None:
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Market universe worker shutting down")
