"""Weather worker: runs independent weather workflow and writes DB snapshot.

Run from backend dir:
  python -m workers.weather_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from config import settings
from models.database import AsyncSessionLocal, init_database
from services.weather.workflow_orchestrator import weather_workflow_orchestrator
from services.weather import shared_state

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("weather_worker")


async def _run_loop() -> None:
    logger.info("Weather worker started")

    # Ensure initial snapshot exists for UI status.
    try:
        async with AsyncSessionLocal() as session:
            await shared_state.write_weather_snapshot(
                session,
                opportunities=[],
                status={
                    "running": True,
                    "enabled": True,
                    "interval_seconds": settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
                    "last_scan": None,
                    "current_activity": "Weather worker started; first scan pending.",
                },
                stats={},
            )
    except Exception:
        pass

    next_scheduled_run_at: datetime | None = None

    while True:
        async with AsyncSessionLocal() as session:
            control = await shared_state.read_weather_control(session)
            wf_settings = await shared_state.get_weather_settings(session)

        interval = int(
            max(
                300,
                min(
                    86400,
                    control.get("scan_interval_seconds")
                    or wf_settings.get("scan_interval_seconds")
                    or settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
                ),
            )
        )
        paused = bool(control.get("is_paused", False))
        requested = control.get("requested_scan_at") is not None
        enabled = bool(wf_settings.get("enabled", True))
        auto_run = bool(wf_settings.get("auto_run", True))
        now = datetime.now(timezone.utc)

        should_run_scheduled = (
            enabled
            and auto_run
            and not paused
            and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
        )
        should_run = requested or should_run_scheduled

        if not should_run:
            await asyncio.sleep(min(10, interval))
            continue

        try:
            async with AsyncSessionLocal() as session:
                result = await weather_workflow_orchestrator.run_cycle(session)
                await shared_state.clear_weather_scan_request(session)
            next_scheduled_run_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ) + timedelta(seconds=interval)
            logger.info(
                "Weather cycle complete",
                extra={
                    "markets": result.get("markets"),
                    "opportunities": result.get("opportunities"),
                    "intents": result.get("intents"),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Weather workflow cycle failed: %s", exc)
            async with AsyncSessionLocal() as session:
                existing, status = await shared_state.read_weather_snapshot(session)
                await shared_state.write_weather_snapshot(
                    session,
                    opportunities=existing,
                    status={
                        "running": True,
                        "enabled": not paused,
                        "interval_seconds": interval,
                        "last_scan": datetime.now(timezone.utc).isoformat(),
                        "current_activity": f"Last weather scan error: {exc}",
                    },
                    stats=status.get("stats") or {},
                )

        await asyncio.sleep(min(10, interval))


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Weather worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
