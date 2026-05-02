"""Search index worker: keeps ``search_index`` continuously fresh.

One pass per tick walks every collector in
``services.search.COLLECTORS`` and reconciles its rows.  Failures in a
single collector don't fail the run — the next collector still runs,
and the failed type's rows simply stay at their previous freshness
until the next tick succeeds.

Like the other workers in this codebase the loop honors the runtime
``worker_control`` row so an operator can pause / resume / change
interval / trigger an immediate run from the UI.  The default
interval is 30 s — short enough that a newly-created strategy or
trader appears in search within seconds, long enough that the
opportunity-snapshot loader (which performs price refreshes) is not a
significant background load.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from config import apply_runtime_settings_overrides, settings
from models.database import AsyncSessionLocal
from services.live_pressure import backpressure_extra_sleep_seconds
from services.search import COLLECTORS, reindex_one
from services.worker_state import (
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("search_index_worker")


_DEFAULT_INTERVAL_SECONDS = 30
_HEARTBEAT_INTERVAL_SECONDS = 5.0


async def _run_loop() -> None:
    worker_name = "search_index"
    default_interval = max(
        5,
        int(getattr(settings, "SEARCH_INDEX_WORKER_INTERVAL_SECONDS", _DEFAULT_INTERVAL_SECONDS) or _DEFAULT_INTERVAL_SECONDS),
    )

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": default_interval,
        "activity": "Search index worker started; awaiting first reindex.",
        "last_error": None,
        "last_run_at": None,
        "run_id": None,
        "phase": "idle",
        "progress": 0.0,
        "total_upserted": 0,
        "total_deleted": 0,
        "total_indexed": 0,
        "last_duration_ms": 0.0,
    }
    heartbeat_stop_event = asyncio.Event()

    async def _heartbeat_loop() -> None:
        import random as _rnd
        await asyncio.sleep(_rnd.uniform(0, _HEARTBEAT_INTERVAL_SECONDS))
        while not heartbeat_stop_event.is_set():
            try:
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        worker_name,
                        running=True,
                        enabled=bool(state.get("enabled", True)),
                        current_activity=str(state.get("activity") or "Idle"),
                        interval_seconds=int(state.get("interval_seconds") or default_interval),
                        last_run_at=state.get("last_run_at"),
                        last_error=(
                            str(state["last_error"])
                            if state.get("last_error") is not None
                            else None
                        ),
                        stats={
                            "run_id": state.get("run_id"),
                            "phase": state.get("phase"),
                            "progress": float(state.get("progress", 0.0) or 0.0),
                            "total_upserted": int(state.get("total_upserted", 0) or 0),
                            "total_deleted": int(state.get("total_deleted", 0) or 0),
                            "total_indexed": int(state.get("total_indexed", 0) or 0),
                            "last_duration_ms": float(state.get("last_duration_ms", 0.0) or 0.0),
                        },
                    )
            except Exception as exc:
                state["last_error"] = str(exc)
                logger.warning("search_index heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(
                    heartbeat_stop_event.wait(), timeout=_HEARTBEAT_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                continue

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="search-index-heartbeat")
    logger.info("Search index worker started (default interval %ss)", default_interval)

    try:
        while True:
            try:
                await apply_runtime_settings_overrides()
            except Exception as exc:
                logger.warning("search_index runtime settings refresh failed: %s", exc)

            async with AsyncSessionLocal() as session:
                control = await read_worker_control(
                    session,
                    worker_name,
                    default_interval=default_interval,
                )

            interval_seconds = max(5, int(control.get("interval_seconds") or default_interval))
            paused = bool(control.get("is_paused", False))
            requested = control.get("requested_run_at") is not None
            enabled = bool(control.get("is_enabled", True)) and not paused

            state["enabled"] = enabled
            state["interval_seconds"] = interval_seconds
            state["run_id"] = uuid.uuid4().hex[:16]

            if not enabled and not requested:
                state["phase"] = "idle"
                state["progress"] = 0.0
                state["activity"] = "Paused"
                await asyncio.sleep(min(_HEARTBEAT_INTERVAL_SECONDS, float(interval_seconds)))
                continue

            run_started = utcnow()
            state["phase"] = "reindex"
            state["progress"] = 0.0
            state["activity"] = "Reindexing search corpus..."

            total_upserted = 0
            total_deleted = 0
            n_types = max(1, len(COLLECTORS))
            try:
                for idx, entity_type in enumerate(COLLECTORS, start=1):
                    state["activity"] = f"Reindexing {entity_type}..."
                    state["progress"] = idx / n_types
                    try:
                        async with AsyncSessionLocal() as session:
                            result = await reindex_one(session, entity_type)
                        if result.get("ok"):
                            total_upserted += int(result.get("upserted") or 0)
                            total_deleted += int(result.get("deleted") or 0)
                        else:
                            logger.warning(
                                "Search reindex returned not-ok for %s: %s",
                                entity_type,
                                result.get("error"),
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.exception(
                            "Search reindex crashed for entity_type=%s: %s",
                            entity_type,
                            exc,
                        )

                state["total_upserted"] = total_upserted
                state["total_deleted"] = total_deleted
                # Read current row count so the UI can show "N entities indexed".
                try:
                    from sqlalchemy import text as _text

                    async with AsyncSessionLocal() as session:
                        rs = await session.execute(_text("SELECT COUNT(*) FROM search_index"))
                        state["total_indexed"] = int(rs.scalar() or 0)
                except Exception:
                    pass

                duration_ms = (utcnow() - run_started).total_seconds() * 1000.0
                state["last_duration_ms"] = round(duration_ms, 1)
                state["last_error"] = None
                state["last_run_at"] = utcnow()
                state["phase"] = "idle"
                state["progress"] = 1.0
                state["activity"] = (
                    f"Reindexed {total_upserted} entities ({total_deleted} pruned) "
                    f"in {state['last_duration_ms']:.0f}ms"
                )

                async with AsyncSessionLocal() as session:
                    await clear_worker_run_request(session, worker_name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state["last_error"] = str(exc)
                state["phase"] = "error"
                state["progress"] = 1.0
                state["activity"] = f"Reindex error: {exc}"
                logger.exception("Search reindex cycle failed")

            base_sleep = float(interval_seconds)
            extra_sleep = backpressure_extra_sleep_seconds(base_sleep)
            await asyncio.sleep(base_sleep + extra_sleep)
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
        logger.info("Search index worker shutting down")
