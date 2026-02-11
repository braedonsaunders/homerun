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

# Ensure backend is on path when run as python -m workers.scanner_worker from project root
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

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

        try:
            await scanner.scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Scan failed: %s", e)
            async with AsyncSessionLocal() as session:
                await write_scanner_snapshot(session, [], {
                    "running": True,
                    "enabled": not paused,
                    "interval_seconds": interval,
                    "last_scan": None,
                    "current_activity": f"Last scan error: {e}",
                    "strategies": [],
                })
            await asyncio.sleep(interval)
            continue

        opps = scanner.get_opportunities()
        status = scanner.get_status()
        async with AsyncSessionLocal() as session:
            await write_scanner_snapshot(session, opps, status)
            await clear_scan_request(session)
        logger.debug("Wrote snapshot: %d opportunities", len(opps))
        await asyncio.sleep(interval)


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


if __name__ == "__main__":
    asyncio.run(main())
