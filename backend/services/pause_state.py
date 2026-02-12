"""Global pause state shared across all services.

When paused, all background services (scanner, trader orchestrator, copy trader,
wallet tracker, wallet discovery, wallet intelligence) skip their work cycles.

This module exists as a standalone singleton to avoid circular imports between
services that need to check the pause state.
"""

from typing import Callable, List


class GlobalPauseState:
    """Global pause/resume state for the entire platform."""

    def __init__(self):
        self._paused = False
        self._callbacks: List[Callable] = []

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self):
        """Pause all background services."""
        self._paused = True

    def resume(self):
        """Resume all background services."""
        self._paused = False


# Singleton - import this from any service
global_pause_state = GlobalPauseState()

