"""Global pause state shared across all services.

When paused, all background services (scanner, trader orchestrator, copy trader,
wallet tracker, wallet discovery, wallet intelligence) skip their work cycles.

This module exists as a standalone singleton to avoid circular imports between
services that need to check the pause state.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, List, Optional


class GlobalPauseState:
    """Global pause/resume state for the entire platform."""

    def __init__(self, refresh_interval_seconds: float = 2.0):
        self._paused = False
        self._callbacks: List[Callable] = []
        self._refresh_interval_seconds = max(0.5, float(refresh_interval_seconds))
        self._last_refresh_monotonic = 0.0
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def is_paused(self) -> bool:
        self._schedule_refresh_if_stale()
        return self._paused

    def _schedule_refresh_if_stale(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh_monotonic < self._refresh_interval_seconds:
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refresh_task = loop.create_task(self.refresh_from_db())

    async def refresh_from_db(self, *, force: bool = False) -> bool:
        """Refresh pause state from shared DB controls."""
        now = time.monotonic()
        if (
            not force
            and now - self._last_refresh_monotonic < self._refresh_interval_seconds
        ):
            return self._paused

        try:
            from models.database import AsyncSessionLocal
            from services import discovery_shared_state, shared_state
            from services.news import shared_state as news_shared_state
            from services.trader_orchestrator_state import read_orchestrator_control
            from services.weather import shared_state as weather_shared_state
            from services.worker_state import read_worker_control

            async with AsyncSessionLocal() as session:
                scanner_control = await shared_state.read_scanner_control(session)
                news_control = await news_shared_state.read_news_control(session)
                weather_control = await weather_shared_state.read_weather_control(session)
                discovery_control = await discovery_shared_state.read_discovery_control(
                    session
                )
                orchestrator_control = await read_orchestrator_control(session)
                crypto_control = await read_worker_control(session, "crypto")
                tracked_control = await read_worker_control(session, "tracked_traders")
                world_intel_control = await read_worker_control(
                    session, "world_intelligence"
                )

            self._paused = all(
                bool(control.get("is_paused", False))
                for control in (
                    scanner_control,
                    news_control,
                    weather_control,
                    discovery_control,
                    orchestrator_control,
                    crypto_control,
                    tracked_control,
                    world_intel_control,
                )
            )
        except Exception:
            # Keep last-known state on transient DB/read failures.
            pass
        finally:
            self._last_refresh_monotonic = time.monotonic()
        return self._paused

    def pause(self) -> None:
        """Pause all background services."""
        self._paused = True
        self._last_refresh_monotonic = time.monotonic()

    def resume(self) -> None:
        """Resume all background services."""
        self._paused = False
        self._last_refresh_monotonic = time.monotonic()


# Singleton - import this from any service
global_pause_state = GlobalPauseState()
