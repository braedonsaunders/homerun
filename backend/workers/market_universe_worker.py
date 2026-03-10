"""Market universe worker: dedicated catalog refresh plane.

Owns upstream market/event fetches and writes canonical ``market_catalog``.
Scanner worker consumes this DB catalog and performs detection only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from config import apply_runtime_settings_overrides, settings
from models.database import AsyncSessionLocal, recover_pool
from services.scanner import scanner
from services.shared_state import read_market_catalog, read_scanner_status
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.worker_state import (
    _is_retryable_db_error,
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("market_universe_worker")

_IDLE_SLEEP_SECONDS = 5
_MAX_CONSECUTIVE_DB_FAILURES = 3


async def _read_catalog_stats() -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        _, _, metadata = await read_market_catalog(
            session,
            include_events=False,
            include_markets=False,
            validate=False,
        )
        scanner_status = await read_scanner_status(session, include_opportunity_count=False)

    tiered = scanner_status.get("tiered_scanning") or {}
    scanned_markets = int(
        tiered.get("full_snapshot_cycle_processed_markets")
        or tiered.get("last_full_snapshot_chunk_market_count")
        or 0
    )
    total_markets = int(metadata.get("market_count") or 0)
    coverage_ratio = round(float(scanned_markets) / float(total_markets), 6) if total_markets > 0 else None

    updated_at = metadata.get("updated_at")
    updated_iso = updated_at.isoformat() if isinstance(updated_at, datetime) else None
    return {
        "catalog_updated_at": updated_iso,
        "event_count": int(metadata.get("event_count") or 0),
        "market_count": int(metadata.get("market_count") or 0),
        "scanned_markets": scanned_markets,
        "stale_markets": None,
        "coverage_ratio": coverage_ratio,
        "fetch_duration_seconds": (
            float(metadata.get("fetch_duration_seconds"))
            if metadata.get("fetch_duration_seconds") is not None
            else None
        ),
        "error": metadata.get("error"),
    }


async def _run_with_timeout_budget(coro, *, timeout_seconds: float, label: str):
    timeout_budget = max(0.01, float(timeout_seconds))
    started = time.monotonic()
    result = await coro
    elapsed = time.monotonic() - started
    if elapsed > timeout_budget:
        raise asyncio.TimeoutError(f"{label} exceeded timeout budget ({elapsed:.3f}s > {timeout_budget:.3f}s)")
    return result


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
    incremental_timeout = max(
        15,
        int(getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_TIMEOUT_SECONDS", 120) or 120),
    )
    full_reconcile_interval = max(
        60,
        int(getattr(settings, "MARKET_UNIVERSE_FULL_RECONCILE_INTERVAL_SECONDS", 900) or 900),
    )
    incremental_enabled_default = bool(getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_ENABLED", True))

    await scanner.load_settings()
    await scanner.load_plugins(source_keys=["scanner"])

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": default_interval,
        "activity": "Market universe worker started; first refresh pending.",
        "last_error": None,
        "last_run_at": None,
        "run_id": None,
        "phase": "idle",
        "progress": 0.0,
        "last_market_count": 0,
        "last_event_count": 0,
        "last_scanned_markets": 0,
        "last_stale_markets": 0,
        "last_coverage_ratio": None,
        "catalog_updated_at": None,
        "fetch_duration_seconds": None,
        "sync_mode": "incremental" if incremental_enabled_default else "full",
        "last_delta_market_count": 0,
        "last_delta_event_count": 0,
        "next_full_reconcile_at": None,
    }
    heartbeat_stop_event = asyncio.Event()
    next_scheduled_run_at: datetime | None = None
    next_full_reconcile_at: datetime | None = None

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
                    "run_id": loop_state.get("run_id"),
                    "phase": loop_state.get("phase"),
                    "progress": float(loop_state.get("progress", 0.0) or 0.0),
                    "event_count": int(loop_state.get("last_event_count", 0) or 0),
                    "market_count": int(loop_state.get("last_market_count", 0) or 0),
                    "scanned_markets": int(loop_state.get("last_scanned_markets", 0) or 0),
                    "stale_markets": int(loop_state.get("last_stale_markets", 0) or 0),
                    "coverage_ratio": loop_state.get("last_coverage_ratio"),
                    "catalog_updated_at": loop_state.get("catalog_updated_at"),
                    "fetch_duration_seconds": loop_state.get("fetch_duration_seconds"),
                    "sync_mode": loop_state.get("sync_mode"),
                    "delta_market_count": int(loop_state.get("last_delta_market_count", 0) or 0),
                    "delta_event_count": int(loop_state.get("last_delta_event_count", 0) or 0),
                    "next_full_reconcile_at": loop_state.get("next_full_reconcile_at"),
                },
            )

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="market-universe-heartbeat")
    logger.info("Market universe worker started")
    consecutive_db_failures = 0

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
            incremental_enabled = bool(
                getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_ENABLED", incremental_enabled_default)
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
                state["phase"] = "idle"
                state["progress"] = 0.0
                state["activity"] = "Paused" if paused else "Idle - waiting for next catalog refresh."
                await asyncio.sleep(min(5.0, float(interval_seconds)))
                continue

            state["run_id"] = uuid.uuid4().hex[:16]
            force_full = (
                not incremental_enabled
                or next_full_reconcile_at is None
                or now >= next_full_reconcile_at
            )
            state["sync_mode"] = "full" if force_full else "incremental"
            state["phase"] = "refresh_catalog_full" if force_full else "refresh_catalog_incremental"
            state["progress"] = 0.2
            state["activity"] = (
                "Refreshing market universe catalog (full reconcile)..."
                if force_full
                else "Refreshing market universe catalog (incremental deltas)..."
            )
            try:
                sync_timeout = float(refresh_timeout if force_full else incremental_timeout)
                try:
                    sync_result = await _run_with_timeout_budget(
                        scanner.refresh_catalog_incremental(force_full=force_full),
                        timeout_seconds=sync_timeout,
                        label="market_universe_sync",
                    )
                except Exception:
                    if force_full:
                        raise
                    logger.warning("Market universe incremental sync failed, retrying full reconcile", exc_info=True)
                    sync_result = await _run_with_timeout_budget(
                        scanner.refresh_catalog_incremental(force_full=True),
                        timeout_seconds=float(refresh_timeout),
                        label="market_universe_full_reconcile",
                    )
                    force_full = True
                market_count = int(sync_result.get("market_count") or 0)
                state["sync_mode"] = str(sync_result.get("mode") or state["sync_mode"])
                state["last_delta_market_count"] = int(sync_result.get("delta_market_count") or 0)
                state["last_delta_event_count"] = int(sync_result.get("delta_event_count") or 0)
                if str(state["sync_mode"]) == "full":
                    next_full_reconcile_at = now + timedelta(seconds=full_reconcile_interval)
                elif next_full_reconcile_at is None:
                    next_full_reconcile_at = now + timedelta(seconds=full_reconcile_interval)
                state["next_full_reconcile_at"] = (
                    next_full_reconcile_at.isoformat() if next_full_reconcile_at is not None else None
                )
                state["phase"] = "read_coverage"
                state["progress"] = 0.75
                catalog_stats = await _read_catalog_stats()
                state["last_market_count"] = int(catalog_stats.get("market_count") or market_count or 0)
                state["last_event_count"] = int(catalog_stats.get("event_count") or 0)
                state["last_scanned_markets"] = int(catalog_stats.get("scanned_markets") or 0)
                state["last_stale_markets"] = int(catalog_stats.get("stale_markets") or 0)
                state["last_coverage_ratio"] = catalog_stats.get("coverage_ratio")
                state["catalog_updated_at"] = catalog_stats.get("catalog_updated_at")
                state["fetch_duration_seconds"] = catalog_stats.get("fetch_duration_seconds")
                state["last_error"] = None
                state["last_run_at"] = utcnow()
                state["phase"] = "idle"
                state["progress"] = 1.0
                state["activity"] = (
                    f"Market universe {state['sync_mode']} sync complete: {state['last_event_count']} events, "
                    f"{state['last_market_count']} markets ({state['last_stale_markets']} stale, "
                    f"{state['last_delta_market_count']} delta markets)."
                )
                async with AsyncSessionLocal() as session:
                    await clear_worker_run_request(session, worker_name)
                consecutive_db_failures = 0
                next_scheduled_run_at = now + timedelta(seconds=interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                is_db_disconnect = _is_retryable_db_error(exc)
                if is_db_disconnect:
                    consecutive_db_failures += 1
                else:
                    consecutive_db_failures = 0

                state["last_error"] = str(exc)
                state["phase"] = "error"
                state["progress"] = 1.0
                state["activity"] = f"Market universe refresh error: {exc}"
                logger.exception("Market universe refresh cycle failed: %s", exc)

                if is_db_disconnect and consecutive_db_failures >= _MAX_CONSECUTIVE_DB_FAILURES:
                    logger.warning(
                        "DB disconnect streak=%d; disposing connection pool",
                        consecutive_db_failures,
                    )
                    try:
                        await recover_pool()
                    except Exception:
                        pass

                try:
                    async with AsyncSessionLocal() as session:
                        await clear_worker_run_request(session, worker_name)
                except Exception:
                    pass
                next_scheduled_run_at = now + timedelta(seconds=interval_seconds)

            if consecutive_db_failures > 0:
                sleep_seconds = min(
                    _IDLE_SLEEP_SECONDS * (2 ** (consecutive_db_failures - 1)),
                    float(interval_seconds),
                )
            else:
                sleep_seconds = 0.1
            await asyncio.sleep(sleep_seconds)
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
