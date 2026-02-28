"""Opportunity aggregator worker: single-writer for scanner snapshot + signals.

Consumes scanner detection batches from ``scanner_batch_queue`` and performs:
1) snapshot/opportunity_state persistence
2) trade signal bridge upsert/sweep
3) per-strategy dead-letter replay pipeline
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings
from models.database import AsyncSessionLocal
from models.opportunity import Opportunity
from services.shared_state import (
    claim_next_scanner_batch,
    claim_next_strategy_dead_letter_batch,
    cleanup_processed_scanner_batches,
    cleanup_processed_strategy_dead_letter_batches,
    count_dead_letter_scanner_batches,
    count_pending_scanner_batches,
    count_pending_strategy_dead_letter_batches,
    count_terminal_strategy_dead_letter_batches,
    enqueue_strategy_dead_letter_batch,
    mark_scanner_batch_failed,
    mark_scanner_batch_processed,
    mark_strategy_dead_letter_failed,
    mark_strategy_dead_letter_processed,
    write_scanner_snapshot,
)
from services.signal_bus import (
    build_signal_contract_from_opportunity,
    expire_source_signals_except,
    list_pending_source_dedupe_keys,
    make_dedupe_key,
)
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.worker_state import (
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("opportunity_aggregator_worker")


def _parse_batch_opportunities(batch_items: list[Any]) -> tuple[list[Opportunity], int]:
    opportunities: list[Opportunity] = []
    skipped = 0
    for item in batch_items:
        try:
            opportunities.append(Opportunity.model_validate(item))
        except Exception:
            skipped += 1
    return opportunities, skipped


def _market_history_from_opportunities(opportunities: list[Opportunity]) -> dict[str, list[dict[str, Any]]]:
    history_by_market: dict[str, list[dict[str, Any]]] = {}
    for opp in opportunities:
        for market in opp.markets:
            if not isinstance(market, dict):
                continue
            points = market.get("price_history")
            if not isinstance(points, list) or len(points) < 2:
                continue
            market_ids = (
                str(market.get("id", "") or "").strip(),
                str(market.get("condition_id") or market.get("conditionId") or "").strip(),
            )
            for market_id in market_ids:
                if market_id and market_id not in history_by_market:
                    history_by_market[market_id] = points
    return history_by_market


def _group_opportunities_by_strategy(opportunities: list[Opportunity]) -> dict[str, list[Opportunity]]:
    grouped: dict[str, list[Opportunity]] = {}
    for opp in opportunities:
        strategy_key = str(getattr(opp, "strategy", "") or "").strip().lower() or "unknown"
        if strategy_key not in grouped:
            grouped[strategy_key] = [opp]
        else:
            grouped[strategy_key].append(opp)
    return grouped


def _strategy_keep_dedupe_keys(opportunities: list[Opportunity]) -> set[str]:
    keep: set[str] = set()
    for opp in opportunities:
        try:
            market_id, _, _, _, _, _ = build_signal_contract_from_opportunity(opp)
        except Exception:
            continue
        if not market_id:
            continue
        keep.add(
            make_dedupe_key(
                opp.stable_id,
                opp.strategy,
                market_id,
            )
        )
    return keep


async def _run_with_deadline(
    coro,
    *,
    step_timeout: float,
    cycle_deadline_monotonic: float,
):
    remaining = cycle_deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError("aggregator global run deadline exceeded")
    timeout_budget = max(0.01, min(float(step_timeout), float(remaining)))
    started = time.monotonic()
    result = await coro
    elapsed = time.monotonic() - started
    if elapsed > timeout_budget:
        raise asyncio.TimeoutError(
            f"aggregator step exceeded timeout budget ({elapsed:.3f}s > {timeout_budget:.3f}s)"
        )
    return result


async def _run_loop() -> None:
    worker_name = "opportunity_aggregator"
    owner = f"{worker_name}:{os.getpid()}"
    heartbeat_interval = max(
        1.0,
        float(getattr(settings, "OPPORTUNITY_AGGREGATOR_HEARTBEAT_INTERVAL_SECONDS", 3.0) or 3.0),
    )
    idle_sleep = max(
        0.05,
        float(getattr(settings, "OPPORTUNITY_AGGREGATOR_IDLE_SLEEP_SECONDS", 0.25) or 0.25),
    )
    cleanup_interval_seconds = max(
        30,
        int(getattr(settings, "SCANNER_BATCH_QUEUE_CLEANUP_INTERVAL_SECONDS", 300) or 300),
    )
    batch_lease_seconds = max(
        5,
        int(getattr(settings, "SCANNER_BATCH_QUEUE_LEASE_SECONDS", 45) or 45),
    )
    max_attempts = max(
        1,
        int(getattr(settings, "SCANNER_BATCH_QUEUE_RETRY_MAX_ATTEMPTS", 5) or 5),
    )
    retain_hours = max(
        1,
        int(getattr(settings, "SCANNER_BATCH_QUEUE_RETAIN_HOURS", 24) or 24),
    )

    batch_timeout_seconds = max(
        5.0,
        float(getattr(settings, "OPPORTUNITY_AGGREGATOR_BATCH_TIMEOUT_SECONDS", 45) or 45),
    )
    strategy_timeout_seconds = max(
        2.0,
        float(getattr(settings, "OPPORTUNITY_AGGREGATOR_STRATEGY_TIMEOUT_SECONDS", 20) or 20),
    )
    global_run_timeout_seconds = max(
        batch_timeout_seconds,
        float(getattr(settings, "OPPORTUNITY_AGGREGATOR_GLOBAL_RUN_TIMEOUT_SECONDS", 120) or 120),
    )
    strategy_dlq_max_attempts = max(
        1,
        int(getattr(settings, "OPPORTUNITY_AGGREGATOR_STRATEGY_DLQ_RETRY_MAX_ATTEMPTS", 10) or 10),
    )
    strategy_dlq_retain_hours = max(
        1,
        int(getattr(settings, "OPPORTUNITY_AGGREGATOR_STRATEGY_DLQ_RETAIN_HOURS", 72) or 72),
    )

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": max(1, int(round(idle_sleep))),
        "activity": "Aggregator worker started; waiting for scanner batches.",
        "last_error": None,
        "last_run_at": None,
        "run_id": None,
        "phase": "idle",
        "progress": 0.0,
        "processed_batches": 0,
        "last_batch_id": None,
        "pending_batches": 0,
        "dead_letter_batches": 0,
        "signals_emitted_last_run": 0,
        "strategy_dead_letter_pending": 0,
        "strategy_dead_letter_terminal": 0,
        "processed_strategy_dead_letter_batches": 0,
        "last_strategy_dead_letter_id": None,
    }
    heartbeat_stop_event = asyncio.Event()

    async def _heartbeat_loop() -> None:
        while not heartbeat_stop_event.is_set():
            try:
                async with AsyncSessionLocal() as session:
                    pending = await count_pending_scanner_batches(session)
                    dead_letter = await count_dead_letter_scanner_batches(session)
                    strategy_pending = await count_pending_strategy_dead_letter_batches(session)
                    strategy_terminal = await count_terminal_strategy_dead_letter_batches(session)
                    state["pending_batches"] = int(pending)
                    state["dead_letter_batches"] = int(dead_letter)
                    state["strategy_dead_letter_pending"] = int(strategy_pending)
                    state["strategy_dead_letter_terminal"] = int(strategy_terminal)
                    await write_worker_snapshot(
                        session,
                        worker_name,
                        running=True,
                        enabled=bool(state.get("enabled", True)),
                        current_activity=str(state.get("activity") or "Idle"),
                        interval_seconds=int(state.get("interval_seconds") or 1),
                        last_run_at=state.get("last_run_at"),
                        last_error=(str(state["last_error"]) if state.get("last_error") is not None else None),
                        stats={
                            "run_id": state.get("run_id"),
                            "phase": state.get("phase"),
                            "progress": float(state.get("progress", 0.0) or 0.0),
                            "processed_batches": int(state.get("processed_batches", 0) or 0),
                            "pending_batches": int(state.get("pending_batches", 0) or 0),
                            "dead_letter_batches": int(state.get("dead_letter_batches", 0) or 0),
                            "last_batch_id": state.get("last_batch_id"),
                            "signals_emitted_last_run": int(state.get("signals_emitted_last_run", 0) or 0),
                            "strategy_dead_letter_pending": int(state.get("strategy_dead_letter_pending", 0) or 0),
                            "strategy_dead_letter_terminal": int(
                                state.get("strategy_dead_letter_terminal", 0) or 0
                            ),
                            "processed_strategy_dead_letter_batches": int(
                                state.get("processed_strategy_dead_letter_batches", 0) or 0
                            ),
                            "last_strategy_dead_letter_id": state.get("last_strategy_dead_letter_id"),
                        },
                    )
            except Exception as exc:
                state["last_error"] = str(exc)
                logger.warning("Aggregator heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(heartbeat_stop_event.wait(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                continue

    async def _bridge_strategy_group(strategy_opps: list[Opportunity]) -> int:
        async with AsyncSessionLocal() as session:
            emitted = await bridge_opportunities_to_signals(
                session,
                strategy_opps,
                source="scanner",
                sweep_missing=False,
                refresh_prices=False,
            )
        return int(emitted)

    async def _enqueue_strategy_dead_letter(
        *,
        batch_id: str,
        strategy_type: str,
        strategy_opps: list[Opportunity],
        status: dict[str, Any],
        error_text: str,
    ) -> str | None:
        async with AsyncSessionLocal() as session:
            return await enqueue_strategy_dead_letter_batch(
                session,
                source="scanner",
                batch_id=batch_id,
                strategy_type=strategy_type,
                opportunities=strategy_opps,
                status=status,
                error=error_text,
            )

    async def _finalize_sweep(
        *,
        keep_dedupe_keys: set[str],
        failed_strategy_types: set[str],
    ) -> None:
        async with AsyncSessionLocal() as session:
            keep = set(keep_dedupe_keys)
            if failed_strategy_types:
                protected = await list_pending_source_dedupe_keys(
                    session,
                    source="scanner",
                    signal_types=["scanner_opportunity"],
                    strategy_types=sorted(failed_strategy_types),
                )
                keep.update(protected)
            await expire_source_signals_except(
                session,
                source="scanner",
                keep_dedupe_keys=keep,
                signal_types=["scanner_opportunity"],
                commit=True,
            )

    async def _mark_batch_processed(batch_id: str) -> None:
        async with AsyncSessionLocal() as session:
            await mark_scanner_batch_processed(session, batch_id)
            await clear_worker_run_request(session, worker_name)

    async def _mark_batch_failed(batch_id: str, *, error: str, requeue: bool) -> None:
        async with AsyncSessionLocal() as session:
            await mark_scanner_batch_failed(
                session,
                batch_id,
                error=error,
                requeue=requeue,
            )

    async def _mark_strategy_dead_letter_as_processed(dead_letter_id: str) -> None:
        async with AsyncSessionLocal() as session:
            await mark_strategy_dead_letter_processed(session, dead_letter_id)

    async def _mark_strategy_dead_letter_as_failed(
        dead_letter_id: str,
        *,
        error: str,
        requeue: bool,
    ) -> None:
        async with AsyncSessionLocal() as session:
            await mark_strategy_dead_letter_failed(
                session,
                dead_letter_id,
                error=error,
                requeue=requeue,
            )

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="opportunity-aggregator-heartbeat")
    next_cleanup_at = datetime.now(timezone.utc)
    logger.info("Opportunity aggregator worker started")

    try:
        while True:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(
                    session,
                    worker_name,
                    default_interval=max(1, int(round(idle_sleep))),
                )

            paused = bool(control.get("is_paused", False))
            enabled = bool(control.get("is_enabled", True)) and not paused
            requested = control.get("requested_run_at") is not None
            state["enabled"] = enabled
            state["interval_seconds"] = max(1, int(round(idle_sleep)))
            state["run_id"] = uuid.uuid4().hex[:16]
            state["signals_emitted_last_run"] = 0

            if not enabled and not requested:
                state["phase"] = "idle"
                state["progress"] = 0.0
                state["activity"] = "Paused"
                await asyncio.sleep(heartbeat_interval)
                continue

            cycle_deadline_monotonic = time.monotonic() + float(global_run_timeout_seconds)
            claimed_scanner_batch: dict[str, Any] | None = None
            claimed_strategy_dlq_batch: dict[str, Any] | None = None

            try:
                # Priority lane: replay one strategy dead-letter batch first.
                async with AsyncSessionLocal() as session:
                    claimed_strategy_dlq_batch = await claim_next_strategy_dead_letter_batch(
                        session,
                        owner=owner,
                        lease_seconds=batch_lease_seconds,
                    )

                if claimed_strategy_dlq_batch is not None:
                    dead_letter_id = str(claimed_strategy_dlq_batch.get("id") or "").strip()
                    strategy_type = str(claimed_strategy_dlq_batch.get("strategy_type") or "unknown")
                    dead_letter_attempt = int(claimed_strategy_dlq_batch.get("attempt_count") or 1)
                    dlq_opportunities, skipped = _parse_batch_opportunities(
                        list(claimed_strategy_dlq_batch.get("opportunities_json") or [])
                    )
                    if skipped:
                        logger.warning(
                            "Aggregator strategy DLQ skipped %d invalid opportunities in dead-letter %s",
                            skipped,
                            dead_letter_id,
                        )

                    state["phase"] = "replay_strategy_dead_letter"
                    state["progress"] = 0.25
                    state["activity"] = (
                        f"Replaying strategy dead-letter {dead_letter_id} ({strategy_type}) "
                        f"with {len(dlq_opportunities)} opportunities..."
                    )

                    try:
                        if dlq_opportunities:
                            emitted = await _run_with_deadline(
                                _bridge_strategy_group(dlq_opportunities),
                                step_timeout=strategy_timeout_seconds,
                                cycle_deadline_monotonic=cycle_deadline_monotonic,
                            )
                        else:
                            emitted = 0

                        await _run_with_deadline(
                            _mark_strategy_dead_letter_as_processed(dead_letter_id),
                            step_timeout=batch_timeout_seconds,
                            cycle_deadline_monotonic=cycle_deadline_monotonic,
                        )
                        state["processed_strategy_dead_letter_batches"] = (
                            int(state.get("processed_strategy_dead_letter_batches", 0) or 0) + 1
                        )
                        state["last_strategy_dead_letter_id"] = dead_letter_id
                        state["signals_emitted_last_run"] = int(emitted)
                        state["last_error"] = None
                        state["last_run_at"] = utcnow()
                        state["phase"] = "idle"
                        state["progress"] = 1.0
                        state["activity"] = (
                            f"Replayed strategy dead-letter {dead_letter_id} ({strategy_type}): {int(emitted)} signals."
                        )
                        async with AsyncSessionLocal() as session:
                            await clear_worker_run_request(session, worker_name)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        requeue = dead_letter_attempt < strategy_dlq_max_attempts
                        await _mark_strategy_dead_letter_as_failed(
                            dead_letter_id,
                            error=str(exc),
                            requeue=requeue,
                        )
                        state["last_error"] = str(exc)
                        state["phase"] = "error"
                        state["progress"] = 1.0
                        state["signals_emitted_last_run"] = 0
                        state["activity"] = (
                            f"Strategy dead-letter replay failed ({dead_letter_id}, {strategy_type}, "
                            f"attempt {dead_letter_attempt}, requeue={requeue})."
                        )
                        logger.exception(
                            "Failed replaying strategy dead-letter %s (strategy=%s attempt=%d requeue=%s)",
                            dead_letter_id,
                            strategy_type,
                            dead_letter_attempt,
                            requeue,
                        )

                    await asyncio.sleep(idle_sleep)
                    continue

                # Main lane: claim and process one scanner batch.
                async with AsyncSessionLocal() as session:
                    claimed_scanner_batch = await claim_next_scanner_batch(
                        session,
                        owner=owner,
                        lease_seconds=batch_lease_seconds,
                    )

                if claimed_scanner_batch is None:
                    state["phase"] = "idle"
                    state["progress"] = 0.0
                    state["activity"] = "Idle - waiting for scanner batches."
                    await asyncio.sleep(idle_sleep)
                    continue

                batch_id = str(claimed_scanner_batch.get("id") or "").strip()
                batch_kind = str(claimed_scanner_batch.get("batch_kind") or "scan_cycle")
                batch_status = dict(claimed_scanner_batch.get("status_json") or {})
                attempt_count = int(claimed_scanner_batch.get("attempt_count") or 1)
                opportunities, skipped = _parse_batch_opportunities(
                    list(claimed_scanner_batch.get("opportunities_json") or [])
                )
                if skipped:
                    logger.warning(
                        "Aggregator skipped %d invalid opportunities in batch %s",
                        skipped,
                        batch_id,
                    )

                state["phase"] = "aggregate_batch"
                state["progress"] = 0.2
                state["activity"] = (
                    f"Aggregating scanner batch {batch_id} ({batch_kind}) with {len(opportunities)} opportunities..."
                )

                try:
                    market_history = _market_history_from_opportunities(opportunities)

                    async def _write_snapshot() -> None:
                        async with AsyncSessionLocal() as session:
                            await write_scanner_snapshot(
                                session,
                                opportunities,
                                batch_status,
                                market_history=market_history if market_history else None,
                            )

                    await _run_with_deadline(
                        _write_snapshot(),
                        step_timeout=batch_timeout_seconds,
                        cycle_deadline_monotonic=cycle_deadline_monotonic,
                    )

                    strategy_groups = _group_opportunities_by_strategy(opportunities)
                    keep_dedupe_keys: set[str] = set()
                    failed_strategy_types: set[str] = set()
                    emitted_total = 0

                    strategy_items = list(strategy_groups.items())
                    for index, (strategy_type, strategy_opps) in enumerate(strategy_items, start=1):
                        state["phase"] = "bridge_strategy_group"
                        state["progress"] = 0.25 + (0.55 * (float(index) / float(max(1, len(strategy_items)))))
                        state["activity"] = (
                            f"Bridging strategy {strategy_type} for batch {batch_id} "
                            f"({index}/{len(strategy_items)})..."
                        )
                        try:
                            emitted = await _run_with_deadline(
                                _bridge_strategy_group(strategy_opps),
                                step_timeout=strategy_timeout_seconds,
                                cycle_deadline_monotonic=cycle_deadline_monotonic,
                            )
                            emitted_total += int(emitted)
                            keep_dedupe_keys.update(_strategy_keep_dedupe_keys(strategy_opps))
                        except asyncio.CancelledError:
                            raise
                        except Exception as strategy_exc:
                            failed_strategy_types.add(strategy_type)
                            dead_letter_id = await _run_with_deadline(
                                _enqueue_strategy_dead_letter(
                                    batch_id=batch_id,
                                    strategy_type=strategy_type,
                                    strategy_opps=strategy_opps,
                                    status=batch_status,
                                    error_text=str(strategy_exc),
                                ),
                                step_timeout=batch_timeout_seconds,
                                cycle_deadline_monotonic=cycle_deadline_monotonic,
                            )
                            state["last_strategy_dead_letter_id"] = dead_letter_id
                            logger.exception(
                                "Failed bridging strategy group %s in batch %s; queued strategy dead-letter %s",
                                strategy_type,
                                batch_id,
                                dead_letter_id,
                            )

                    state["phase"] = "sweep_signals"
                    state["progress"] = 0.88
                    await _run_with_deadline(
                        _finalize_sweep(
                            keep_dedupe_keys=keep_dedupe_keys,
                            failed_strategy_types=failed_strategy_types,
                        ),
                        step_timeout=batch_timeout_seconds,
                        cycle_deadline_monotonic=cycle_deadline_monotonic,
                    )

                    state["phase"] = "mark_processed"
                    state["progress"] = 0.95
                    await _run_with_deadline(
                        _mark_batch_processed(batch_id),
                        step_timeout=batch_timeout_seconds,
                        cycle_deadline_monotonic=cycle_deadline_monotonic,
                    )

                    state["processed_batches"] = int(state.get("processed_batches", 0) or 0) + 1
                    state["last_batch_id"] = batch_id
                    state["signals_emitted_last_run"] = int(emitted_total)
                    state["last_error"] = None
                    state["last_run_at"] = utcnow()
                    state["phase"] = "idle"
                    state["progress"] = 1.0
                    state["activity"] = (
                        f"Aggregated scanner batch {batch_id}: {len(opportunities)} opportunities, "
                        f"{int(emitted_total)} signals, {len(failed_strategy_types)} strategy DLQ entries."
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    requeue = attempt_count < max_attempts
                    await _mark_batch_failed(batch_id, error=str(exc), requeue=requeue)
                    state["last_error"] = str(exc)
                    state["signals_emitted_last_run"] = 0
                    state["phase"] = "error"
                    state["progress"] = 1.0
                    state["activity"] = f"Aggregator batch error: {exc}"
                    logger.exception(
                        "Failed processing scanner batch %s (attempt=%d requeue=%s)",
                        batch_id,
                        attempt_count,
                        requeue,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state["last_error"] = str(exc)
                state["phase"] = "error"
                state["progress"] = 1.0
                state["signals_emitted_last_run"] = 0
                state["activity"] = f"Aggregator cycle error: {exc}"
                logger.exception("Aggregator loop cycle failed")

                if claimed_scanner_batch is not None:
                    batch_id = str(claimed_scanner_batch.get("id") or "").strip()
                    attempt_count = int(claimed_scanner_batch.get("attempt_count") or 1)
                    requeue = attempt_count < max_attempts
                    try:
                        await _mark_batch_failed(
                            batch_id,
                            error=f"cycle_timeout_or_failure: {exc}",
                            requeue=requeue,
                        )
                    except Exception:
                        logger.exception("Failed recording scanner batch failure for %s", batch_id)

                if claimed_strategy_dlq_batch is not None:
                    dead_letter_id = str(claimed_strategy_dlq_batch.get("id") or "").strip()
                    dead_letter_attempt = int(claimed_strategy_dlq_batch.get("attempt_count") or 1)
                    requeue = dead_letter_attempt < strategy_dlq_max_attempts
                    try:
                        await _mark_strategy_dead_letter_as_failed(
                            dead_letter_id,
                            error=f"cycle_timeout_or_failure: {exc}",
                            requeue=requeue,
                        )
                    except Exception:
                        logger.exception("Failed recording strategy dead-letter failure for %s", dead_letter_id)

            now = datetime.now(timezone.utc)
            if now >= next_cleanup_at:
                try:
                    async with AsyncSessionLocal() as session:
                        deleted_batches = await cleanup_processed_scanner_batches(
                            session,
                            retain_hours=retain_hours,
                        )
                        deleted_strategy_dlq = await cleanup_processed_strategy_dead_letter_batches(
                            session,
                            retain_hours=strategy_dlq_retain_hours,
                        )
                    if deleted_batches:
                        logger.info("Aggregator queue cleanup removed %d processed scanner rows", deleted_batches)
                    if deleted_strategy_dlq:
                        logger.info(
                            "Aggregator queue cleanup removed %d processed strategy dead-letter rows",
                            deleted_strategy_dlq,
                        )
                except Exception as exc:
                    logger.warning("Aggregator queue cleanup failed: %s", exc)
                next_cleanup_at = now + timedelta(seconds=cleanup_interval_seconds)
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
        logger.info("Opportunity aggregator worker shutting down")
