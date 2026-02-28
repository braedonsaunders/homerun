"""Opportunity aggregator worker: single-writer for scanner snapshot + signals.

Consumes scanner detection batches from ``scanner_batch_queue`` and performs:
1) snapshot/opportunity_state persistence
2) trade signal bridge upsert/sweep
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings
from models.database import AsyncSessionLocal
from models.opportunity import Opportunity
from services.shared_state import (
    claim_next_scanner_batch,
    cleanup_processed_scanner_batches,
    count_pending_scanner_batches,
    mark_scanner_batch_failed,
    mark_scanner_batch_processed,
    write_scanner_snapshot,
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

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": max(1, int(round(idle_sleep))),
        "activity": "Aggregator worker started; waiting for scanner batches.",
        "last_error": None,
        "last_run_at": None,
        "processed_batches": 0,
        "last_batch_id": None,
        "pending_batches": 0,
        "signals_emitted_last_run": 0,
    }
    heartbeat_stop_event = asyncio.Event()

    async def _heartbeat_loop() -> None:
        while not heartbeat_stop_event.is_set():
            try:
                async with AsyncSessionLocal() as session:
                    pending = await count_pending_scanner_batches(session)
                    state["pending_batches"] = int(pending)
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
                            "processed_batches": int(state.get("processed_batches", 0) or 0),
                            "pending_batches": int(state.get("pending_batches", 0) or 0),
                            "last_batch_id": state.get("last_batch_id"),
                            "signals_emitted_last_run": int(state.get("signals_emitted_last_run", 0) or 0),
                        },
                    )
            except Exception as exc:
                state["last_error"] = str(exc)
                logger.warning("Aggregator heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(heartbeat_stop_event.wait(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                continue

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

            if not enabled and not requested:
                state["activity"] = "Paused"
                await asyncio.sleep(heartbeat_interval)
                continue

            batch: dict[str, Any] | None = None
            async with AsyncSessionLocal() as session:
                batch = await claim_next_scanner_batch(
                    session,
                    owner=owner,
                    lease_seconds=batch_lease_seconds,
                )

            if batch is None:
                state["activity"] = "Idle - waiting for scanner batches."
                state["signals_emitted_last_run"] = 0
                await asyncio.sleep(idle_sleep)
            else:
                batch_id = str(batch.get("id") or "").strip()
                batch_kind = str(batch.get("batch_kind") or "scan_cycle")
                status = dict(batch.get("status_json") or {})
                opportunities, skipped = _parse_batch_opportunities(list(batch.get("opportunities_json") or []))
                if skipped:
                    logger.warning(
                        "Aggregator skipped %d invalid opportunities in batch %s",
                        skipped,
                        batch_id,
                    )
                state["activity"] = (
                    f"Aggregating scanner batch {batch_id} ({batch_kind}) with {len(opportunities)} opportunities..."
                )
                try:
                    market_history = _market_history_from_opportunities(opportunities)
                    async with AsyncSessionLocal() as session:
                        await write_scanner_snapshot(
                            session,
                            opportunities,
                            status,
                            market_history=market_history if market_history else None,
                        )
                        emitted = await bridge_opportunities_to_signals(
                            session,
                            opportunities,
                            source="scanner",
                            sweep_missing=True,
                            refresh_prices=False,
                        )
                        await mark_scanner_batch_processed(session, batch_id)
                        await clear_worker_run_request(session, worker_name)
                    state["processed_batches"] = int(state.get("processed_batches", 0) or 0) + 1
                    state["last_batch_id"] = batch_id
                    state["signals_emitted_last_run"] = int(emitted)
                    state["last_error"] = None
                    state["last_run_at"] = utcnow()
                    state["activity"] = (
                        f"Aggregated scanner batch {batch_id}: {len(opportunities)} opportunities, {int(emitted)} signals."
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    attempt_count = int(batch.get("attempt_count") or 1)
                    requeue = attempt_count < max_attempts
                    state["last_error"] = str(exc)
                    state["signals_emitted_last_run"] = 0
                    state["activity"] = f"Aggregator batch error: {exc}"
                    logger.exception(
                        "Failed processing scanner batch %s (attempt=%d requeue=%s)",
                        batch_id,
                        attempt_count,
                        requeue,
                    )
                    async with AsyncSessionLocal() as session:
                        await mark_scanner_batch_failed(
                            session,
                            batch_id,
                            error=str(exc),
                            requeue=requeue,
                        )

            now = datetime.now(timezone.utc)
            if now >= next_cleanup_at:
                try:
                    async with AsyncSessionLocal() as session:
                        deleted = await cleanup_processed_scanner_batches(
                            session,
                            retain_hours=retain_hours,
                        )
                    if deleted:
                        logger.info("Aggregator queue cleanup removed %d processed rows", deleted)
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
