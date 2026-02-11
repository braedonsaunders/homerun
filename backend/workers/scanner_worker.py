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

from config import settings
from models.database import AsyncSessionLocal, init_database
from services import scanner
from services.shared_state import (
    clear_scan_request,
    read_scanner_control,
    update_scanner_activity,
    write_scanner_snapshot,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scanner_worker")


async def _run_scan_loop() -> None:
    """Load scanner, then loop: read control -> scan -> write snapshot -> sleep."""
    await scanner.load_settings()
    await scanner.load_plugins()
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

    # Worker-owned WS feed manager enables reactive scanning and fresh
    # order-book overlays in this process (scanner runs here, not in API).
    if settings.WS_FEED_ENABLED:
        try:
            from services.ws_feeds import get_feed_manager
            from services.polymarket import polymarket_client

            feed_manager = get_feed_manager()

            async def _http_book_fallback(token_id: str):
                try:
                    return await polymarket_client.get_order_book(token_id)
                except Exception:
                    return None

            feed_manager.set_http_fallback(_http_book_fallback)
            if not feed_manager._started:
                await feed_manager.start()
            logger.info("Scanner worker WebSocket feeds started")
        except Exception as e:
            logger.warning("Worker WS feeds failed to start (non-critical): %s", e)

    # Notifier and opportunity_recorder hook into scanner callbacks (same process)
    try:
        from services.notifier import notifier
        from services.opportunity_recorder import opportunity_recorder
        await notifier.start()
        await opportunity_recorder.start()
        logger.info("Notifier and opportunity recorder started")
    except Exception as e:
        logger.warning("Notifier/opportunity_recorder start failed (non-critical): %s", e)

    logger.info("Scanner worker started (interval from DB)")

    # Write initial status so API doesn't show "Waiting for scanner worker" before first scan
    try:
        async with AsyncSessionLocal() as session:
            await update_scanner_activity(session, "Scanner started; first scan pending.")
    except Exception:
        pass

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_scanner_control(session)
        interval = max(10, min(3600, control["scan_interval_seconds"] or 60))
        paused = control.get("is_paused", False)
        requested = control.get("requested_scan_at")

        if paused and not requested:
            await asyncio.sleep(min(10, interval))
            continue

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
            logger.exception("Scan failed: %s", e)
            try:
                prev_opps = scanner.get_opportunities()
            except Exception:
                prev_opps = []
            async with AsyncSessionLocal() as session:
                await write_scanner_snapshot(
                    session,
                    prev_opps,
                    {
                        "running": True,
                        "enabled": not paused,
                        "interval_seconds": interval,
                        "last_scan": None,
                        "current_activity": f"Last scan error: {e}",
                        "strategies": [],
                    },
                    market_history=scanner.get_market_history_for_opportunities(
                        prev_opps, max_points=40
                    ),
                )
            sleep_seconds = (
                settings.FAST_SCAN_INTERVAL_SECONDS
                if settings.TIERED_SCANNING_ENABLED
                else interval
            )
            await asyncio.sleep(max(1, sleep_seconds))
            continue

        # Persist snapshot. Post-scan failures must not crash the worker loop.
        try:
            opps = scanner.get_opportunities()
        except Exception as e:
            logger.exception("Failed to fetch opportunities after scan: %s", e)
            opps = []

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
                market_history = scanner.get_market_history_for_opportunities(
                    opps, max_points=40
                )
                await write_scanner_snapshot(
                    session, opps, status, market_history=market_history
                )
                await clear_scan_request(session)
            logger.debug("Wrote snapshot: %d opportunities", len(opps))
        except Exception as e:
            logger.exception("Failed to persist scanner snapshot: %s", e)
        sleep_seconds = (
            settings.FAST_SCAN_INTERVAL_SECONDS
            if settings.TIERED_SCANNING_ENABLED and not requested
            else interval
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
