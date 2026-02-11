"""Discovery worker: runs wallet discovery loop and writes DB status snapshot.

Run from backend dir:
  python -m workers.discovery_worker
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
from services.discovery_shared_state import (
    clear_discovery_run_request,
    read_discovery_control,
    write_discovery_snapshot,
)
from services.pause_state import global_pause_state
from services.wallet_discovery import wallet_discovery

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discovery_worker")


async def _run_loop() -> None:
    logger.info("Discovery worker started")

    # Ensure an initial snapshot exists for UI status.
    try:
        async with AsyncSessionLocal() as session:
            await write_discovery_snapshot(
                session,
                {
                    "running": False,
                    "enabled": True,
                    "run_interval_minutes": settings.DISCOVERY_RUN_INTERVAL_MINUTES,
                    "last_run_at": None,
                    "current_activity": "Discovery worker started; first run pending.",
                    "wallets_discovered_last_run": 0,
                    "wallets_analyzed_last_run": 0,
                },
            )
    except Exception:
        pass

    next_scheduled_run_at: datetime | None = None

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_discovery_control(session)

        interval_minutes = int(
            max(
                5,
                min(
                    1440,
                    control.get("run_interval_minutes")
                    or settings.DISCOVERY_RUN_INTERVAL_MINUTES,
                ),
            )
        )
        paused = bool(control.get("is_paused", False))
        requested = control.get("requested_run_at") is not None
        enabled = bool(control.get("is_enabled", True))
        now = datetime.now(timezone.utc)

        should_run_scheduled = (
            enabled
            and not paused
            and not global_pause_state.is_paused
            and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
        )
        should_run = requested or should_run_scheduled

        if not should_run:
            await asyncio.sleep(10)
            continue

        try:
            async with AsyncSessionLocal() as session:
                await write_discovery_snapshot(
                    session,
                    {
                        "running": True,
                        "enabled": not paused,
                        "run_interval_minutes": interval_minutes,
                        "current_activity": "Scanning trader wallets...",
                    },
                )

            await wallet_discovery.run_discovery(
                max_markets=settings.DISCOVERY_MAX_MARKETS_PER_RUN,
                max_wallets_per_market=settings.DISCOVERY_MAX_WALLETS_PER_MARKET,
            )

            async with AsyncSessionLocal() as session:
                await clear_discovery_run_request(session)
                await write_discovery_snapshot(
                    session,
                    {
                        "running": False,
                        "enabled": not paused,
                        "run_interval_minutes": interval_minutes,
                        "last_run_at": wallet_discovery._last_run_at.isoformat()
                        if wallet_discovery._last_run_at
                        else None,
                        "current_activity": "Idle - waiting for next discovery cycle.",
                        "wallets_discovered_last_run": wallet_discovery._wallets_discovered_last_run,
                        "wallets_analyzed_last_run": wallet_discovery._wallets_analyzed_last_run,
                    },
                )

            next_scheduled_run_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ) + timedelta(minutes=interval_minutes)
            logger.info(
                "Discovery cycle complete",
                wallets_discovered=wallet_discovery._wallets_discovered_last_run,
                wallets_analyzed=wallet_discovery._wallets_analyzed_last_run,
                next_run_at=next_scheduled_run_at.isoformat(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Discovery cycle failed: %s", exc)
            async with AsyncSessionLocal() as session:
                await clear_discovery_run_request(session)
                await write_discovery_snapshot(
                    session,
                    {
                        "running": False,
                        "enabled": not paused,
                        "run_interval_minutes": interval_minutes,
                        "last_run_at": datetime.now(timezone.utc).isoformat(),
                        "current_activity": f"Last discovery run error: {exc}",
                        "wallets_discovered_last_run": wallet_discovery._wallets_discovered_last_run,
                        "wallets_analyzed_last_run": wallet_discovery._wallets_analyzed_last_run,
                    },
                )
            next_scheduled_run_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ) + timedelta(minutes=interval_minutes)

        await asyncio.sleep(10)


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Discovery worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
