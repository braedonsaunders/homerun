"""Concrete market-data adapters.

Provides a single provider implementation that aggregates Polymarket +
Kalshi clients behind the MarketDataProvider interface.
"""

from __future__ import annotations

from interfaces import MarketDataProvider
from models.market import Event, Market
from services.kalshi_client import kalshi_client
from services.polymarket import polymarket_client


class PolymarketKalshiProvider(MarketDataProvider):
    """Default provider used by scanner services."""

    async def get_all_events(self, closed: bool = False) -> list[Event]:
        return await polymarket_client.get_all_events(closed=closed)

    async def get_all_markets(self, active: bool = True) -> list[Market]:
        return await polymarket_client.get_all_markets(active=active)

    async def get_recent_markets(
        self, since_minutes: int = 10, active: bool = True
    ) -> list[Market]:
        return await polymarket_client.get_recent_markets(
            since_minutes=since_minutes, active=active
        )

    async def get_prices_batch(self, token_ids: list[str]) -> dict:
        return await polymarket_client.get_prices_batch(token_ids)

    async def get_cross_platform_events(self, closed: bool = False) -> list[Event]:
        return await kalshi_client.get_all_events(closed=closed)

    async def get_cross_platform_markets(self, active: bool = True) -> list[Market]:
        return await kalshi_client.get_all_markets(active=active)


market_data_provider = PolymarketKalshiProvider()
