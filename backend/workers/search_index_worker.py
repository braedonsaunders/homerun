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
from services.live_pressure import backpressure_extra_sleep_seconds, is_db_pressure_active
from services.search import COLLECTORS, reindex_one
from services.worker_state import (
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("search_index_worker")


_DEFAULT_INTERVAL_SECONDS = 120  # was 30s.  A live capture during a
# crypto-backtest run showed search reindex taking 57s and contending
# with the DB pool: 6 worker_snapshot UPSERTs hit
# QueryCanceledError (statement_timeout) and the trader-orchestrator
# cycles slowed to 10s because the connection pool was tied up by
# the search collector queries.  The previous 30s interval meant the
# next reindex started almost immediately after the prior one
# finished — a near-continuous DB load.  120s is well within the
# product freshness budget (newly created strategies/traders appear
# in search within 2 minutes) and gives other workers ample
# uncontended pool capacity between reindex passes.
_HEARTBEAT_INTERVAL_SECONDS = 5.0
# Yield slice between consecutive collectors.  100ms × 11 collectors
# = 1.1s of extra wall-clock per reindex, but it lets the event loop
# tick and other workers checkout connections instead of starving
# behind an 11-collector burst.
_INTER_COLLECTOR_YIELD_SECONDS = 0.1
# Fix UU — chunked reindex.  The 12h soak on 2026-05-05 showed a single
# reindex pass running 79s and the heartbeat upsert hitting
# QueryCanceledError (statement_timeout) on the worker_snapshot row,
# which then flagged DB pressure and cascaded into scanner / trader
# reconciliation deferring.  The contributors are cumulative: 11
# collectors back-to-back, each touching its own large source table,
# adds up under any concurrent DB load.  Cap each cycle to a small
# chunk and persist a cursor so the next cycle resumes — total
# freshness latency rises proportionally (~120s × ceil(11/CHUNK)
# instead of 120s) but the per-cycle DB footprint stays bounded.
_REINDEX_COLLECTORS_PER_CYCLE = 4
# Soft per-cycle wall-clock budget — even within one chunk, if we're
# already this far into the cycle, defer the rest.  Picked below the
# expected statement_timeout so we don't paint ourselves into a corner.
_REINDEX_CYCLE_SOFT_BUDGET_SECONDS = 30.0
# Hard per-collector cap.  The soft budget only checks BETWEEN
# collectors; one collector that runs longer than the budget (observed
# 68.7s vs 30s budget in the 2026-05-07 soak) blows the deferral
# guarantee. Wrap each collector in wait_for so a slow one is
# cancelled cleanly and the next cycle picks it up. Picked at 25s
# (under the soft budget) so we still observe the deferral path
# rather than always running the slowest collector to completion.
_REINDEX_COLLECTOR_HARD_TIMEOUT_SECONDS = 25.0


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
                            "collector_cursor": int(state.get("collector_cursor", 0) or 0),
                        },
                    )
            except Exception as exc:
                # Heartbeat writes use a 3s SET LOCAL statement_timeout
                # (see worker_state._apply_snapshot_write_timeouts). A
                # QueryCanceledError is the timeout firing under DB
                # contention and is expected — don't sticky-set it as
                # last_error or every UI poll will misreport the worker
                # as failed even after the contention clears.
                exc_name = type(exc).__name__
                msg = str(exc).lower()
                is_expected_timeout = (
                    "QueryCanceled" in exc_name
                    or "statement timeout" in msg
                    or "canceling statement" in msg
                )
                if not is_expected_timeout:
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
            collector_list = list(COLLECTORS)
            n_types = max(1, len(collector_list))
            # Fix UU — resume from where the prior cycle left off.  The
            # cursor is process-local, so a worker restart restarts at
            # collector 0; that's intentional, since we want a freshly-
            # started worker to publish a current snapshot of the most
            # important collectors first.
            cursor = int(state.get("collector_cursor", 0) or 0) % n_types
            chunk_started_mono = utcnow()
            try:
                processed_this_cycle = 0
                for chunk_offset in range(min(_REINDEX_COLLECTORS_PER_CYCLE, n_types)):
                    idx = (cursor + chunk_offset) % n_types
                    entity_type = collector_list[idx]
                    # In-loop backpressure check.  If a separate worker
                    # has flagged DB pressure (worker_snapshot writes
                    # cancelled, persist_orders thrashing, etc.)
                    # bail out of the remaining collectors and let the
                    # next 120s interval pick them up.  The freshness
                    # cost of a delayed reindex is acceptable; the
                    # cost of stacking on top of a stressed DB pool
                    # is a cascade of QueryCanceledError across half
                    # a dozen workers (observed in the 02:48:27
                    # cascade during the backtest run).
                    if is_db_pressure_active():
                        state["activity"] = (
                            f"Search reindex paused under DB pressure "
                            f"(cursor={idx}/{n_types}, processed_this_cycle={processed_this_cycle})"
                        )
                        logger.warning(
                            "Search reindex deferring remaining collectors under DB pressure",
                            cursor=idx,
                            total=n_types,
                            processed_this_cycle=processed_this_cycle,
                        )
                        break
                    # Soft wall-clock budget — bail out of this cycle's
                    # chunk if we've already used most of our envelope.
                    elapsed_s = (utcnow() - chunk_started_mono).total_seconds()
                    if elapsed_s >= _REINDEX_CYCLE_SOFT_BUDGET_SECONDS:
                        logger.warning(
                            "Search reindex deferring remaining chunk under soft budget",
                            cursor=idx,
                            total=n_types,
                            elapsed_seconds=round(elapsed_s, 1),
                            budget_seconds=_REINDEX_CYCLE_SOFT_BUDGET_SECONDS,
                        )
                        break
                    state["activity"] = f"Reindexing {entity_type} (cursor={idx}/{n_types})..."
                    state["progress"] = (idx + 1) / n_types
                    try:
                        async def _reindex_collector() -> dict[str, Any]:
                            async with AsyncSessionLocal() as session:
                                return await reindex_one(session, entity_type)
                        try:
                            result = await asyncio.wait_for(
                                _reindex_collector(),
                                timeout=_REINDEX_COLLECTOR_HARD_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Search reindex hard-timeout for %s after %ss; deferring to next cycle",
                                entity_type,
                                _REINDEX_COLLECTOR_HARD_TIMEOUT_SECONDS,
                            )
                            # Don't advance cursor past this collector — same
                            # one re-runs next cycle so it isn't permanently skipped.
                            break
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
                    processed_this_cycle += 1
                    # Yield to the event loop and release the DB pool
                    # slot before grabbing the next collector's session.
                    # Without this, an 11-collector reindex hogs the
                    # pool back-to-back.
                    await asyncio.sleep(_INTER_COLLECTOR_YIELD_SECONDS)
                # Advance the cursor for next cycle.  ``processed_this_cycle``
                # may be smaller than the chunk if we bailed under DB
                # pressure / soft budget — in that case the next cycle
                # picks up exactly where we left off.
                state["collector_cursor"] = (cursor + processed_this_cycle) % n_types

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
