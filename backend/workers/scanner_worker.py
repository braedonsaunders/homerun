"""
Scanner worker: detection-only plane.

Runs fast/heavy scanner lanes, consumes DB market catalog + Redis prices,
and enqueues normalized detection batches for the opportunity aggregator worker.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

# Native numerical libs (OpenMP/BLAS/FAISS) can segfault under high thread
# contention in long-running workers on macOS; pin conservative defaults.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NEWS_FAISS_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

from config import apply_runtime_settings_overrides, settings
from models.database import AsyncSessionLocal
from services.scanner import scanner
from services.shared_state import (
    clear_scan_request,
    count_pending_scanner_batches,
    drop_oldest_pending_scanner_batch,
    enqueue_scanner_batch,
    pop_targeted_condition_ids,
    read_scanner_control,
    read_scanner_snapshot,
)
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.worker_state import write_worker_snapshot
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("scanner_worker")


def _is_upstream_resolution_error(exc: Exception) -> bool:
    """Classify DNS/name-resolution upstream failures from HTTP client errors."""
    text = str(exc).lower()
    markers = (
        "nodename nor servname provided",
        "name or service not known",
        "temporary failure in name resolution",
    )
    return any(marker in text for marker in markers)


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp into a UTC-aware datetime."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _hydrate_scanner_pool_from_snapshot() -> int:
    """Restore scanner in-memory pool from DB snapshot on worker startup."""
    try:
        async with AsyncSessionLocal() as session:
            existing_opps, existing_status = await read_scanner_snapshot(session)
    except Exception as exc:
        logger.warning("Scanner startup hydration skipped (snapshot read failed): %s", exc)
        return 0

    now = datetime.now(timezone.utc)
    restored = []
    dropped_expired = 0
    for opp in existing_opps:
        resolution_date = opp.resolution_date
        if resolution_date is not None:
            if resolution_date.tzinfo is None:
                resolution_date = resolution_date.replace(tzinfo=timezone.utc)
            else:
                resolution_date = resolution_date.astimezone(timezone.utc)
            if resolution_date <= now:
                dropped_expired += 1
                continue
        restored.append(opp)

    scanner._opportunities = restored
    scanner._market_price_history = {}
    scanner._remember_market_tokens_from_opportunities(restored)
    now_ms = int(now.timestamp() * 1000)
    restored_history_markets = 0
    for opp in restored:
        for market in opp.markets:
            history = market.get("price_history")
            if not isinstance(history, list) or not history:
                continue
            for market_id in (
                str(market.get("id", "") or "").strip(),
                str(market.get("condition_id") or market.get("conditionId") or "").strip(),
            ):
                if not market_id:
                    continue
                merged_len = scanner._merge_market_history_points(market_id, history, now_ms)
                if merged_len >= 2:
                    restored_history_markets += 1

    if isinstance(existing_status, dict):
        parsed_last_scan = _parse_iso_utc(existing_status.get("last_scan"))
        if parsed_last_scan is not None:
            scanner._last_scan = parsed_last_scan
        parsed_last_fast = _parse_iso_utc(existing_status.get("last_fast_scan"))
        if parsed_last_fast is not None:
            scanner._last_fast_scan = parsed_last_fast
        parsed_last_heavy = _parse_iso_utc(existing_status.get("last_heavy_scan"))
        if parsed_last_heavy is not None:
            scanner._last_full_snapshot_strategy_scan = parsed_last_heavy
        tiered = existing_status.get("tiered_scanning")
        if isinstance(tiered, dict):
            parsed_tier_fast = _parse_iso_utc(tiered.get("last_fast_scan"))
            if parsed_tier_fast is not None:
                scanner._last_fast_scan = parsed_tier_fast
            parsed_tier_heavy = _parse_iso_utc(
                tiered.get("last_heavy_scan") or tiered.get("last_full_snapshot_strategy_scan")
            )
            if parsed_tier_heavy is not None:
                scanner._last_full_snapshot_strategy_scan = parsed_tier_heavy

    if restored or dropped_expired:
        logger.info(
            "Hydrated scanner pool from DB snapshot: restored=%d dropped_expired=%d history_markets=%d",
            len(restored),
            dropped_expired,
            restored_history_markets,
        )
    return len(restored)


async def _enqueue_detection_batch(
    opportunities: list,
    status: dict,
    *,
    batch_kind: str,
) -> tuple[str | None, int, int]:
    """Enqueue scanner batch with backpressure trimming."""
    max_pending = max(
        1,
        int(getattr(settings, "SCANNER_BATCH_QUEUE_MAX_PENDING", 200) or 200),
    )
    dropped = 0
    batch_id: str | None = None

    async with AsyncSessionLocal() as session:
        pending = await count_pending_scanner_batches(session)
        while pending >= max_pending:
            dropped_id = await drop_oldest_pending_scanner_batch(session)
            if not dropped_id:
                break
            dropped += 1
            pending = max(0, pending - 1)
            logger.warning(
                "Scanner batch queue backpressure: dropped oldest pending batch %s",
                dropped_id,
            )

        batch_id = await enqueue_scanner_batch(
            session,
            opportunities,
            status,
            source="scanner_worker",
            batch_kind=batch_kind,
        )
        await clear_scan_request(session)
        pending_after = await count_pending_scanner_batches(session)

    return batch_id, pending_after, dropped


async def _run_scan_loop() -> None:
    """Load scanner then run detection loop and enqueue aggregation batches."""
    await scanner.load_settings()
    await scanner.load_plugins(source_keys=["scanner"])
    restored_count = await _hydrate_scanner_pool_from_snapshot()
    scanner._running = True
    scanner._enabled = True

    logger.info("Scanner worker initialized in detection-only mode")

    try:
        from services.opportunity_recorder import opportunity_recorder

        await opportunity_recorder.start()
        logger.info("Opportunity recorder started")
    except Exception as exc:
        logger.warning("Opportunity recorder start failed (non-critical): %s", exc)

    catalog_count = await scanner._hydrate_catalog_from_db()
    if catalog_count:
        logger.info("Hydrated market catalog from DB: %d markets", catalog_count)

    heartbeat_state: dict[str, object] = {
        "enabled": True,
        "interval_seconds": 60,
        "last_error": None,
        "last_run_at": None,
        "queue_pending": 0,
        "dropped_batches": 0,
        "last_batch_id": None,
    }
    heartbeat_interval_seconds = max(
        2.0,
        float(getattr(settings, "SCANNER_HEARTBEAT_INTERVAL_SECONDS", 5.0) or 5.0),
    )
    heartbeat_stop_event = asyncio.Event()

    async def _heartbeat_loop() -> None:
        while not heartbeat_stop_event.is_set():
            try:
                status = scanner.get_status()
                tiered = status.get("tiered_scanning") or {}
                lane_watchdogs = status.get("lane_watchdogs") or {}
                async with AsyncSessionLocal() as session:
                    pending = await count_pending_scanner_batches(session)
                    heartbeat_state["queue_pending"] = int(pending)
                    await write_worker_snapshot(
                        session,
                        "scanner",
                        running=True,
                        enabled=bool(heartbeat_state.get("enabled", True)),
                        current_activity=str(status.get("current_activity") or "Idle"),
                        interval_seconds=int(heartbeat_state.get("interval_seconds") or 60),
                        last_run_at=heartbeat_state.get("last_run_at"),
                        last_error=(
                            str(heartbeat_state["last_error"])
                            if heartbeat_state.get("last_error") is not None
                            else None
                        ),
                        stats={
                            "opportunities_count": int(status.get("opportunities_count", 0) or 0),
                            "signals_emitted_last_run": 0,
                            "full_snapshot_running": bool(tiered.get("full_snapshot_strategy_running", False)),
                            "fast_inflight": bool(tiered.get("fast_inflight", False)),
                            "heavy_inflight": bool(tiered.get("heavy_inflight", False)),
                            "lane_watchdogs": lane_watchdogs,
                            "queue_pending": int(heartbeat_state.get("queue_pending", 0) or 0),
                            "dropped_batches": int(heartbeat_state.get("dropped_batches", 0) or 0),
                            "last_batch_id": heartbeat_state.get("last_batch_id"),
                        },
                    )
            except Exception as exc:
                heartbeat_state["last_error"] = str(exc)
                logger.warning("Scanner heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(heartbeat_stop_event.wait(), timeout=heartbeat_interval_seconds)
            except asyncio.TimeoutError:
                continue

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="scanner-heartbeat-loop")

    async def _run_full_snapshot_cycle(
        *,
        reason: str,
        targeted_ids: list[str] | None,
        force: bool,
        watchdog_seconds: int,
    ) -> None:
        try:
            await asyncio.wait_for(
                scanner.scan_full_snapshot_strategies(
                    reason=reason,
                    targeted_condition_ids=targeted_ids,
                    force=force,
                ),
                timeout=watchdog_seconds,
            )
        except asyncio.TimeoutError:
            scanner.note_heavy_lane_watchdog_timeout(watchdog_seconds)
            logger.warning("Full-snapshot scan timed out after %ss", watchdog_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Full-snapshot scan cycle failed")

    heavy_scan_task: asyncio.Task | None = None
    pending_heavy_targeted_ids: list[str] | None = None
    stale_scan_streak = 0

    # Publish startup status through the aggregation plane.
    try:
        startup_status = scanner.get_status()
        startup_status["running"] = True
        startup_status["enabled"] = True
        startup_status["interval_seconds"] = int(startup_status.get("interval_seconds") or 60)
        startup_status["current_activity"] = (
            f"Scanner resumed with {restored_count} restored opportunities; next scan pending."
            if restored_count > 0
            else "Scanner started; first scan pending."
        )
        startup_status["opportunities_count"] = len(scanner.get_opportunities())
        batch_id, pending, dropped = await _enqueue_detection_batch(
            scanner.get_opportunities(),
            startup_status,
            batch_kind="startup",
        )
        heartbeat_state["queue_pending"] = pending
        heartbeat_state["dropped_batches"] = int(heartbeat_state.get("dropped_batches", 0) or 0) + dropped
        heartbeat_state["last_batch_id"] = batch_id
    except Exception as exc:
        logger.warning("Failed to enqueue startup scanner batch: %s", exc)

    try:
        while True:
            async with AsyncSessionLocal() as session:
                control = await read_scanner_control(session)
                try:
                    await apply_runtime_settings_overrides()
                except Exception as exc:
                    logger.warning("Failed to refresh scanner runtime settings from DB: %s", exc)
                try:
                    await refresh_strategy_runtime_if_needed(
                        session,
                        source_keys=["scanner"],
                    )
                except Exception as exc:
                    logger.warning("Scanner strategy refresh check failed: %s", exc)

            interval = max(10, min(3600, int(control.get("scan_interval_seconds") or 60)))
            paused = bool(control.get("is_paused", False))
            requested = control.get("requested_scan_at")
            scanner._enabled = not paused
            heartbeat_state["enabled"] = not paused
            heartbeat_state["interval_seconds"] = interval

            scan_watchdog_seconds = max(
                30,
                int(settings.SCAN_WATCHDOG_SECONDS),
                int(settings.FAST_SCAN_INTERVAL_SECONDS) * 3,
            )
            full_snapshot_watchdog_seconds = max(
                30,
                int(getattr(settings, "SCANNER_FULL_SNAPSHOT_WATCHDOG_SECONDS", 180) or 180),
            )

            if paused and not requested:
                if heavy_scan_task is not None and not heavy_scan_task.done():
                    heavy_scan_task.cancel()
                    try:
                        await heavy_scan_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    heavy_scan_task = None
                pending_heavy_targeted_ids = None
                await asyncio.sleep(min(5, interval))
                continue

            # Pull latest universe snapshot written by market_universe worker.
            try:
                await scanner._hydrate_catalog_from_db(only_if_newer=True)
            except Exception as exc:
                logger.warning("Scanner catalog sync from DB failed: %s", exc)

            targeted_ids = pop_targeted_condition_ids() if requested else []
            if requested:
                pending_heavy_targeted_ids = list(targeted_ids)

            scan_error: Exception | None = None
            try:
                reactive_tokens = await scanner.consume_reactive_tokens()
                await asyncio.wait_for(
                    scanner.scan_fast(
                        reactive_token_ids=reactive_tokens,
                        targeted_condition_ids=targeted_ids or None,
                    ),
                    timeout=scan_watchdog_seconds,
                )
                stale_scan_streak = 0
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                scanner.note_fast_lane_watchdog_timeout(scan_watchdog_seconds)
                stale_scan_streak += 1
                logger.warning(
                    "Scanner scan cycle timed out after %ss (streak=%d)",
                    scan_watchdog_seconds,
                    stale_scan_streak,
                )
                scan_error = exc
            except Exception as exc:
                stale_scan_streak = 0
                scan_error = exc

            if scan_error is not None:
                heartbeat_state["last_error"] = str(scan_error)
                timeout_error = isinstance(scan_error, asyncio.TimeoutError)
                network_resolution_error = _is_upstream_resolution_error(scan_error)
                if timeout_error:
                    logger.warning("Scanner scan timeout handling after %.1fs: %s", scan_watchdog_seconds, scan_error)
                elif network_resolution_error:
                    logger.warning(
                        "Scanner upstream DNS/network resolution failed; retaining previous snapshot where possible: %s",
                        scan_error,
                    )
                else:
                    logger.exception("Scan failed: %s", scan_error)
            else:
                heartbeat_state["last_error"] = None

            force_heavy_scan = pending_heavy_targeted_ids is not None
            targeted_for_heavy = list(pending_heavy_targeted_ids or [])
            if heavy_scan_task is None or heavy_scan_task.done():
                if heavy_scan_task is not None and heavy_scan_task.done():
                    try:
                        await heavy_scan_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    heavy_scan_task = None
                run_heavy_now = force_heavy_scan or scanner._full_snapshot_strategy_due(datetime.now(timezone.utc))
                if run_heavy_now:
                    pending_heavy_targeted_ids = None
                    heavy_scan_task = asyncio.create_task(
                        _run_full_snapshot_cycle(
                            reason="manual_request" if force_heavy_scan else "scheduled",
                            targeted_ids=targeted_for_heavy if targeted_for_heavy else None,
                            force=force_heavy_scan,
                            watchdog_seconds=full_snapshot_watchdog_seconds,
                        ),
                        name="scanner-full-snapshot-lane",
                    )

            try:
                opportunities = scanner.get_opportunities()
            except Exception as exc:
                logger.exception("Failed to fetch opportunities after scan: %s", exc)
                opportunities = []

            if scan_error is not None and not opportunities:
                try:
                    async with AsyncSessionLocal() as session:
                        previous_opps, _ = await read_scanner_snapshot(session)
                    opportunities = previous_opps
                except Exception:
                    opportunities = []

            if opportunities and any(getattr(opp, "ai_analysis", None) is None for opp in opportunities):
                try:
                    await asyncio.wait_for(scanner._attach_ai_judgments(opportunities), timeout=6)
                except asyncio.TimeoutError:
                    logger.debug("Timed out reattaching AI judgments in scanner worker loop")
                except Exception as exc:
                    logger.debug("AI judgment reattach skipped: %s", exc)

            try:
                status = scanner.get_status()
            except Exception as exc:
                logger.exception("Failed to build scanner status after scan: %s", exc)
                status = {
                    "running": True,
                    "enabled": not paused,
                    "interval_seconds": interval,
                    "current_activity": getattr(scanner, "_current_activity", None),
                    "strategies": [],
                }

            status["running"] = True
            status["enabled"] = not paused
            status["interval_seconds"] = interval

            if scan_error is not None:
                if isinstance(scan_error, asyncio.TimeoutError):
                    activity = "Scanner cycle timeout; retaining last known opportunities."
                elif _is_upstream_resolution_error(scan_error):
                    activity = "Upstream market API DNS/network error; retaining last known opportunities."
                else:
                    activity = f"Last scan error: {scan_error}"
                status["current_activity"] = activity

            batch_kind = "scan_error" if scan_error is not None else "scan_cycle"
            try:
                batch_id, pending, dropped = await _enqueue_detection_batch(
                    opportunities,
                    status,
                    batch_kind=batch_kind,
                )
                heartbeat_state["queue_pending"] = pending
                heartbeat_state["dropped_batches"] = int(heartbeat_state.get("dropped_batches", 0) or 0) + dropped
                heartbeat_state["last_batch_id"] = batch_id
                heartbeat_state["last_run_at"] = utcnow()
                if dropped > 0:
                    logger.warning("Scanner enqueued batch after dropping %d stale pending batches", dropped)
            except Exception as exc:
                heartbeat_state["last_error"] = str(exc)
                logger.exception("Failed enqueueing scanner detection batch: %s", exc)

            sleep_seconds = (
                settings.FAST_SCAN_INTERVAL_SECONDS if settings.TIERED_SCANNING_ENABLED and not requested else interval
            )
            sleep_seconds = max(1, int(sleep_seconds))

            if settings.TIERED_SCANNING_ENABLED and not requested:
                try:
                    scanner._register_reactive_scanning()
                    if scanner._pending_reactive_tokens:
                        scanner._reactive_trigger.set()
                    else:
                        scanner._reactive_trigger.clear()
                    await asyncio.wait_for(
                        scanner._reactive_trigger.wait(),
                        timeout=sleep_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    await asyncio.sleep(sleep_seconds)
            else:
                await asyncio.sleep(sleep_seconds)
    finally:
        heartbeat_stop_event.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Scanner heartbeat task shutdown encountered an error: %s", exc)

        if heavy_scan_task is not None and not heavy_scan_task.done():
            heavy_scan_task.cancel()
            try:
                await heavy_scan_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


async def start_loop() -> None:
    """Run scanner detection worker."""
    try:
        await apply_runtime_settings_overrides()
    except Exception as exc:
        logger.warning("Initial scanner runtime settings refresh failed: %s", exc)
    try:
        await _run_scan_loop()
    except asyncio.CancelledError:
        logger.info("Scanner worker shutting down")
    finally:
        scanner._running = False
