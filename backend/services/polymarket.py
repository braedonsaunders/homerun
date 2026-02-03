import httpx
import asyncio
import json
from typing import Optional
from datetime import datetime

from config import settings
from models import Market, Event


class PolymarketClient:
    """Client for interacting with Polymarket APIs"""

    def __init__(self):
        self.gamma_url = settings.GAMMA_API_URL
        self.clob_url = settings.CLOB_API_URL
        self.data_url = settings.DATA_API_URL
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ==================== GAMMA API ====================

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> list[Market]:
        """Fetch markets from Gamma API"""
        client = await self._get_client()
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset
        }

        response = await client.get(f"{self.gamma_url}/markets", params=params)
        response.raise_for_status()
        data = response.json()

        return [Market.from_gamma_response(m) for m in data]

    async def get_all_markets(self, active: bool = True) -> list[Market]:
        """Fetch all markets with pagination"""
        all_markets = []
        offset = 0
        limit = 100

        while True:
            markets = await self.get_markets(
                active=active,
                limit=limit,
                offset=offset
            )
            if not markets:
                break

            all_markets.extend(markets)
            offset += limit

            if len(all_markets) >= settings.MAX_MARKETS_TO_SCAN:
                break

        return all_markets

    async def get_events(
        self,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> list[Event]:
        """Fetch events from Gamma API (events contain grouped markets)"""
        client = await self._get_client()
        params = {
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset
        }

        response = await client.get(f"{self.gamma_url}/events", params=params)
        response.raise_for_status()
        data = response.json()

        return [Event.from_gamma_response(e) for e in data]

    async def get_all_events(self, closed: bool = False) -> list[Event]:
        """Fetch all events with pagination"""
        all_events = []
        offset = 0
        limit = 100

        while True:
            events = await self.get_events(
                closed=closed,
                limit=limit,
                offset=offset
            )
            if not events:
                break

            all_events.extend(events)
            offset += limit

            if offset >= 1000:  # Safety limit
                break

        return all_events

    async def get_event_by_slug(self, slug: str) -> Optional[Event]:
        """Get a specific event by slug"""
        client = await self._get_client()
        response = await client.get(f"{self.gamma_url}/events", params={"slug": slug})
        response.raise_for_status()
        data = response.json()

        if data:
            return Event.from_gamma_response(data[0])
        return None

    # ==================== CLOB API ====================

    async def get_midpoint(self, token_id: str) -> float:
        """Get midpoint price for a token"""
        client = await self._get_client()
        response = await client.get(
            f"{self.clob_url}/midpoint",
            params={"token_id": token_id}
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get("mid", 0))

    async def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Get best price for a token (BUY = best ask, SELL = best bid)"""
        client = await self._get_client()
        response = await client.get(
            f"{self.clob_url}/price",
            params={"token_id": token_id, "side": side}
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get("price", 0))

    async def get_order_book(self, token_id: str) -> dict:
        """Get full order book for a token"""
        client = await self._get_client()
        response = await client.get(
            f"{self.clob_url}/book",
            params={"token_id": token_id}
        )
        response.raise_for_status()
        return response.json()

    async def get_prices_batch(self, token_ids: list[str]) -> dict[str, dict]:
        """Get prices for multiple tokens efficiently"""
        prices = {}

        # Batch requests with concurrency limit
        semaphore = asyncio.Semaphore(10)

        async def fetch_price(token_id: str):
            async with semaphore:
                try:
                    mid = await self.get_midpoint(token_id)
                    prices[token_id] = {"mid": mid}
                except Exception as e:
                    prices[token_id] = {"mid": 0, "error": str(e)}

        await asyncio.gather(*[fetch_price(tid) for tid in token_ids])
        return prices

    # ==================== DATA API ====================

    async def get_wallet_positions(self, address: str) -> list[dict]:
        """Get open positions for a wallet"""
        client = await self._get_client()
        response = await client.get(
            f"{self.data_url}/positions",
            params={"user": address}
        )
        response.raise_for_status()
        return response.json()

    async def get_wallet_trades(
        self,
        address: str,
        limit: int = 100
    ) -> list[dict]:
        """Get recent trades for a wallet"""
        client = await self._get_client()
        response = await client.get(
            f"{self.data_url}/trades",
            params={"user": address, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    async def get_market_trades(
        self,
        condition_id: str,
        limit: int = 100
    ) -> list[dict]:
        """Get recent trades for a market"""
        client = await self._get_client()
        response = await client.get(
            f"{self.data_url}/trades",
            params={"market": condition_id, "limit": limit}
        )
        response.raise_for_status()
        return response.json()


# Singleton instance
polymarket_client = PolymarketClient()
