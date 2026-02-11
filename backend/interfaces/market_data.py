"""Market data interface contracts.

These protocols define the minimum async API required by scanner/runtime
services, decoupling strategy orchestration from concrete exchange clients.
"""

from __future__ import annotations

from typing import Protocol

from models.market import Event, Market


class MarketDataProvider(Protocol):
    """Read-only market data provider for scanner services."""

    async def get_all_events(self, closed: bool = False) -> list[Event]:
        """Fetch all events used for strategy context."""

    async def get_all_markets(self, active: bool = True) -> list[Market]:
        """Fetch all active markets for opportunity detection."""

    async def get_recent_markets(
        self, since_minutes: int = 10, active: bool = True
    ) -> list[Market]:
        """Fetch recently created markets for incremental scans."""

    async def get_prices_batch(self, token_ids: list[str]) -> dict:
        """Fetch batched token prices keyed by token ID."""

    async def get_cross_platform_events(self, closed: bool = False) -> list[Event]:
        """Fetch cross-platform events (e.g., Kalshi) for merged scans."""

    async def get_cross_platform_markets(self, active: bool = True) -> list[Market]:
        """Fetch cross-platform markets (e.g., Kalshi) for merged scans."""
