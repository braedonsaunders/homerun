"""
Scanner worker: runs scan loop and writes results to DB (scanner_snapshot).
API and other workers read opportunities/status from DB only.
Run from repo root: cd backend && python -m workers.scanner_worker
Or from backend: python -m workers.scanner_worker
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# Ensure backend is on path when run as python -m workers.scanner_worker from project root
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

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

from utils.utcnow import utcnow
from config import settings
from models.database import AsyncSessionLocal, init_database
from services import scanner
from services.signal_bus import emit_scanner_signals
from services.shared_state import (
    clear_scan_request,
    read_scanner_control,
    read_scanner_snapshot,
    update_scanner_activity,
    write_scanner_snapshot,
)
from services.worker_state import write_worker_snapshot

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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
    except Exception as e:
        logger.warning("Scanner startup hydration skipped (snapshot read failed): %s", e)
        return 0

    now = datetime.now(timezone.utc)
    restored = []
    dropped_expired = 0
    for opp in existing_opps:
        rd = opp.resolution_date
        if rd is not None:
            if rd.tzinfo is None:
                rd = rd.replace(tzinfo=timezone.utc)
            else:
                rd = rd.astimezone(timezone.utc)
            if rd <= now:
                dropped_expired += 1
                continue
        restored.append(opp)

    scanner._opportunities = restored
    if isinstance(existing_status, dict):
        parsed_last_scan = _parse_iso_utc(existing_status.get("last_scan"))
        if parsed_last_scan is not None:
            scanner._last_scan = parsed_last_scan

    if restored or dropped_expired:
        logger.info(
            "Hydrated scanner pool from DB snapshot: restored=%d dropped_expired=%d",
            len(restored),
            dropped_expired,
        )
    return len(restored)


async def _reattach_inline_ai_from_snapshot(opps: list) -> int:
    """Reattach inline ai_analysis from the latest snapshot by stable_id.

    Manual `/ai/judge/opportunity` writes ai_analysis into scanner_snapshot
    immediately. The scanner worker keeps its own in-memory pool and can
    overwrite that snapshot on the next cycle; this restores inline AI fields
    before writing so card-level analysis does not disappear after refetch.
    """
    if not opps:
        return 0

    try:
        async with AsyncSessionLocal() as session:
            snapshot_opps, _ = await read_scanner_snapshot(session)
    except Exception:
        return 0

    ai_by_stable: dict[str, object] = {}
    for snapshot_opp in snapshot_opps:
        stable_id = getattr(snapshot_opp, "stable_id", None)
        analysis = getattr(snapshot_opp, "ai_analysis", None)
        if stable_id and analysis is not None:
            ai_by_stable[stable_id] = analysis

    if not ai_by_stable:
        return 0

    attached = 0
    for opp in opps:
        if getattr(opp, "ai_analysis", None) is not None:
            continue
        stable_id = getattr(opp, "stable_id", None)
        if not stable_id:
            continue
        existing = ai_by_stable.get(stable_id)
        if existing is None:
            continue
        opp.ai_analysis = existing
        attached += 1

    return attached


async def _run_scan_loop() -> None:
    """Load scanner, then loop: read control -> scan -> write snapshot -> sleep."""
    await scanner.load_settings()
    await scanner.load_plugins()
    restored_count = await _hydrate_scanner_pool_from_snapshot()
    scanner._running = True
    scanner._enabled = True

    # Push live activity to DB so API/UI can show "Scanning Polymarket...", etc.
    async def _on_activity(activity: str) -> None:
        try:
            async with AsyncSessionLocal() as session:
                await update_scanner_activity(session, activity)
        except Exception as e:
            logger.debug("Activity update failed: %s", e)

    scanner.add_activity_callback(_on_activity)

    feed_manager = None
    ws_feeds_running = False

    async def _ensure_ws_feeds_running() -> None:
        nonlocal feed_manager, ws_feeds_running
        if ws_feeds_running or not settings.WS_FEED_ENABLED:
            return
        try:
            from services.ws_feeds import get_feed_manager
            from services.polymarket import polymarket_client

            if feed_manager is None:
                feed_manager = get_feed_manager()

                async def _http_book_fallback(token_id: str):
                    try:
                        return await polymarket_client.get_order_book(token_id)
                    except Exception:
                        return None

                feed_manager.set_http_fallback(_http_book_fallback)

            if not feed_manager._started:
                await feed_manager.start()
            ws_feeds_running = True
            logger.info("Scanner worker WebSocket feeds started")
        except Exception as e:
            logger.warning("Worker WS feeds failed to start (non-critical): %s", e)

    async def _stop_ws_feeds() -> None:
        nonlocal ws_feeds_running
        if not ws_feeds_running or feed_manager is None:
            return
        try:
            await feed_manager.stop()
            logger.info("Scanner worker WebSocket feeds paused")
        except Exception as e:
            logger.warning("Worker WS feeds failed to stop cleanly: %s", e)
        finally:
            ws_feeds_running = False

    # Worker-owned WS feed manager enables reactive scanning and fresh
    # order-book overlays in this process (scanner runs here, not in API).
    await _ensure_ws_feeds_running()

    # Opportunity recorder hooks into scanner callbacks in this process.
    try:
        from services.opportunity_recorder import opportunity_recorder

        await opportunity_recorder.start()
        logger.info("Opportunity recorder started")
    except Exception as e:
        logger.warning("Opportunity recorder start failed (non-critical): %s", e)

    logger.info("Scanner worker started (interval from DB)")

    # Write initial status so API doesn't show "Waiting for scanner worker" before first scan
    try:
        current_count = len(scanner.get_opportunities())
        async with AsyncSessionLocal() as session:
            if restored_count > 0:
                activity = f"Scanner resumed with {restored_count} restored opportunities; next scan pending."
            else:
                activity = "Scanner started; first scan pending."
            await update_scanner_activity(session, activity)
            await write_worker_snapshot(
                session,
                "scanner",
                running=True,
                enabled=True,
                current_activity=activity,
                interval_seconds=60,
                last_run_at=None,
                last_error=None,
                stats={
                    "opportunities_count": current_count,
                    "signals_emitted_last_run": 0,
                },
            )
    except Exception:
        pass

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_scanner_control(session)
        interval = max(10, min(3600, control["scan_interval_seconds"] or 60))
        paused = control.get("is_paused", False)
        requested = control.get("requested_scan_at")

        if paused and not requested:
            await _stop_ws_feeds()
            try:
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "scanner",
                        running=True,
                        enabled=False,
                        current_activity="Paused",
                        interval_seconds=interval,
                        last_run_at=None,
                        last_error=None,
                        stats={"opportunities_count": 0, "signals_emitted_last_run": 0},
                    )
            except Exception:
                pass
            await asyncio.sleep(min(10, interval))
            continue

        await _ensure_ws_feeds_running()

        run_full_scan = bool(requested)
        if not run_full_scan and settings.TIERED_SCANNING_ENABLED:
            now = datetime.now(timezone.utc)
            full_interval = max(10, settings.FULL_SCAN_INTERVAL_SECONDS)
            run_full_scan = (
                scanner._last_full_scan is None
                or not scanner._cached_markets
                or (now - scanner._last_full_scan).total_seconds() >= full_interval
            )

        try:
            if settings.TIERED_SCANNING_ENABLED and not run_full_scan and scanner._cached_markets:
                await scanner.scan_fast()
            else:
                await scanner.scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            network_resolution_error = _is_upstream_resolution_error(e)
            if network_resolution_error:
                logger.warning(
                    "Scanner upstream DNS/network resolution failed; retaining previous snapshot where possible: %s",
                    e,
                )
            else:
                logger.exception("Scan failed: %s", e)
            try:
                prev_opps = scanner.get_opportunities()
            except Exception:
                prev_opps = []

            existing_last_scan = None
            async with AsyncSessionLocal() as session:
                try:
                    existing_opps, existing_status = await read_scanner_snapshot(session)
                except Exception:
                    existing_opps, existing_status = [], {}
                if not prev_opps and existing_opps:
                    prev_opps = existing_opps
                if isinstance(existing_status, dict):
                    existing_last_scan = existing_status.get("last_scan")

                if network_resolution_error:
                    activity = "Upstream market API DNS/network error; retaining last known snapshot."
                else:
                    activity = f"Last scan error: {e}"
                await write_scanner_snapshot(
                    session,
                    prev_opps,
                    {
                        "running": True,
                        "enabled": not paused,
                        "interval_seconds": interval,
                        "last_scan": existing_last_scan,
                        "current_activity": activity,
                        "strategies": [],
                    },
                    market_history=scanner.get_market_history_for_opportunities(prev_opps),
                )
                await write_worker_snapshot(
                    session,
                    "scanner",
                    running=True,
                    enabled=not paused,
                    current_activity=activity,
                    interval_seconds=interval,
                    last_run_at=utcnow(),
                    last_error=str(e),
                    stats={
                        "opportunities_count": len(prev_opps),
                        "signals_emitted_last_run": 0,
                    },
                )
            sleep_seconds = settings.FAST_SCAN_INTERVAL_SECONDS if settings.TIERED_SCANNING_ENABLED else interval
            await asyncio.sleep(max(1, sleep_seconds))
            continue

        # Persist snapshot. Post-scan failures must not crash the worker loop.
        try:
            opps = scanner.get_opportunities()
        except Exception as e:
            logger.exception("Failed to fetch opportunities after scan: %s", e)
            opps = []

        # Reattach inline analysis previously persisted by API/manual judging.
        # This prevents worker snapshot writes from clobbering card AI data.
        if opps and any(getattr(o, "ai_analysis", None) is None for o in opps):
            try:
                attached_inline = await _reattach_inline_ai_from_snapshot(opps)
                if attached_inline:
                    logger.debug(
                        "Reattached %d inline AI analyses from snapshot",
                        attached_inline,
                    )
            except Exception as e:
                logger.debug("Inline AI snapshot reattach skipped: %s", e)

        # API-triggered manual AI analysis runs in the API process and writes
        # OpportunityJudgment rows. Re-attach those judgments here so scanner
        # worker memory doesn't wipe them on the next snapshot write.
        if opps and any(getattr(o, "ai_analysis", None) is None for o in opps):
            try:
                await asyncio.wait_for(scanner._attach_ai_judgments(opps), timeout=6)
            except asyncio.TimeoutError:
                logger.debug("Timed out reattaching AI judgments in worker loop")
            except Exception as e:
                logger.debug("AI judgment reattach skipped: %s", e)

        try:
            status = scanner.get_status()
        except Exception as e:
            logger.exception("Failed to build scanner status after scan: %s", e)
            last_scan = None
            try:
                if scanner.last_scan:
                    ls = scanner.last_scan
                    if ls.tzinfo is None:
                        ls = ls.replace(tzinfo=timezone.utc)
                    else:
                        ls = ls.astimezone(timezone.utc)
                    last_scan = ls.replace(tzinfo=None).isoformat() + "Z"
            except Exception:
                pass
            status = {
                "running": True,
                "enabled": not paused,
                "interval_seconds": interval,
                "last_scan": last_scan,
                "current_activity": getattr(scanner, "_current_activity", None),
                "strategies": [],
            }

        try:
            async with AsyncSessionLocal() as session:
                market_history = scanner.get_market_history_for_opportunities(opps)
                await write_scanner_snapshot(session, opps, status, market_history=market_history)
                emitted = await emit_scanner_signals(session, opps)
                await clear_scan_request(session)
                await write_worker_snapshot(
                    session,
                    "scanner",
                    running=True,
                    enabled=not paused,
                    current_activity="Idle - waiting for next scanner cycle.",
                    interval_seconds=interval,
                    last_run_at=utcnow(),
                    last_error=None,
                    stats={
                        "opportunities_count": len(opps),
                        "signals_emitted_last_run": int(emitted),
                    },
                )
            logger.debug("Wrote snapshot: %d opportunities", len(opps))
        except Exception as e:
            logger.exception("Failed to persist scanner snapshot: %s", e)
        sleep_seconds = (
            settings.FAST_SCAN_INTERVAL_SECONDS if settings.TIERED_SCANNING_ENABLED and not requested else interval
        )
        sleep_seconds = max(1, sleep_seconds)

        # Reactive wake-up path: wait for significant WS price changes
        # (debounced in scanner/feed manager) or timeout fallback.
        if settings.TIERED_SCANNING_ENABLED and not requested:
            try:
                scanner._register_reactive_scanning()
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


async def main() -> None:
    """Init DB and run scan loop."""
    await init_database()
    logger.info("Database initialized")
    try:
        await _run_scan_loop()
    except asyncio.CancelledError:
        logger.info("Scanner worker shutting down")
    finally:
        scanner._running = False
        try:
            from services.ws_feeds import get_feed_manager

            await get_feed_manager().stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
