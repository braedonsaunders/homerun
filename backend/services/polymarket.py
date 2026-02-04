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

    # ==================== LEADERBOARD / DISCOVERY ====================

    async def get_leaderboard(
        self,
        limit: int = 50,
        time_period: str = "ALL",
        order_by: str = "PNL",
        category: str = "OVERALL",
        offset: int = 0
    ) -> list[dict]:
        """
        Fetch top traders from Polymarket leaderboard.
        Returns list of wallets with their profit stats.

        Args:
            limit: Max results (1-50 per request, but we can paginate)
            time_period: DAY, WEEK, MONTH, or ALL
            order_by: PNL (profit/loss) or VOL (volume)
            category: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, WEATHER, ECONOMICS, TECH, FINANCE
            offset: Number of results to skip (for pagination)
        """
        client = await self._get_client()
        try:
            # Polymarket data API leaderboard endpoint
            params = {
                "limit": min(limit, 50),  # API max is 50 per request
                "timePeriod": time_period.upper(),
                "orderBy": order_by.upper(),
            }
            if offset > 0:
                params["offset"] = offset
            # Only include category if not OVERALL
            if category.upper() != "OVERALL":
                params["category"] = category.upper()

            response = await client.get(
                f"{self.data_url}/v1/leaderboard",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Leaderboard fetch error: {e}")
            return []

    async def get_leaderboard_paginated(
        self,
        total_limit: int = 100,
        time_period: str = "ALL",
        order_by: str = "PNL",
        category: str = "OVERALL"
    ) -> list[dict]:
        """
        Fetch traders from Polymarket leaderboard with pagination.
        Fetches multiple pages to get more than 50 traders.

        Args:
            total_limit: Total number of traders to fetch (can exceed 50)
            time_period: DAY, WEEK, MONTH, or ALL
            order_by: PNL (profit/loss) or VOL (volume)
            category: Market category filter
        """
        all_traders = []
        offset = 0
        page_size = 50

        while len(all_traders) < total_limit:
            remaining = total_limit - len(all_traders)
            fetch_count = min(page_size, remaining)

            page = await self.get_leaderboard(
                limit=fetch_count,
                time_period=time_period,
                order_by=order_by,
                category=category,
                offset=offset
            )

            if not page:
                break  # No more results

            all_traders.extend(page)
            offset += len(page)

            # If we got fewer results than requested, we've reached the end
            if len(page) < fetch_count:
                break

        return all_traders[:total_limit]

    async def get_top_traders_from_trades(
        self,
        limit: int = 50,
        min_trades: int = 10,
        time_period: str = "ALL",
        order_by: str = "PNL",
        category: str = "OVERALL"
    ) -> list[dict]:
        """
        Get top traders from Polymarket leaderboard.
        Uses the official leaderboard API.
        """
        # Just use the leaderboard API - it has exactly what we need
        leaderboard = await self.get_leaderboard(
            limit=limit,
            time_period=time_period,
            order_by=order_by,
            category=category
        )

        # Transform to our expected format
        traders = []
        for entry in leaderboard:
            traders.append({
                "address": entry.get("proxyWallet", ""),
                "username": entry.get("userName", ""),
                "trades": 0,  # Not provided by leaderboard
                "volume": float(entry.get("vol", 0)),
                "pnl": float(entry.get("pnl", 0)),
                "rank": entry.get("rank", 0),
                "buys": 0,
                "sells": 0,
            })

        return traders[:limit]

    async def calculate_wallet_win_rate(self, address: str, max_trades: int = 500) -> dict:
        """
        Calculate win rate for a wallet by analyzing trade history.
        Groups trades by market and calculates wins vs losses.

        Returns:
            dict with win_rate, wins, losses, total_markets, trade_count
        """
        try:
            trades = await self.get_wallet_trades(address, limit=max_trades)

            if not trades:
                return {
                    "address": address,
                    "win_rate": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "total_markets": 0,
                    "trade_count": 0
                }

            # Group trades by market
            markets: dict[str, dict] = {}
            for trade in trades:
                market_id = trade.get("market") or trade.get("condition_id") or trade.get("assetId", "unknown")
                if market_id not in markets:
                    markets[market_id] = {"buys": 0.0, "sells": 0.0, "buy_count": 0, "sell_count": 0}

                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                side = trade.get("side", "").upper()

                if side == "BUY":
                    markets[market_id]["buys"] += size * price
                    markets[market_id]["buy_count"] += 1
                elif side == "SELL":
                    markets[market_id]["sells"] += size * price
                    markets[market_id]["sell_count"] += 1

            # Calculate wins/losses (a market is a win if sells > buys)
            wins = 0
            losses = 0
            for market_id, data in markets.items():
                # Only count markets with both buys and sells (closed positions)
                if data["buy_count"] > 0 and data["sell_count"] > 0:
                    if data["sells"] > data["buys"]:
                        wins += 1
                    else:
                        losses += 1

            total_closed = wins + losses
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

            return {
                "address": address,
                "win_rate": win_rate,
                "wins": wins,
                "losses": losses,
                "total_markets": len(markets),
                "trade_count": len(trades)
            }
        except Exception as e:
            print(f"Error calculating win rate for {address}: {e}")
            return {
                "address": address,
                "win_rate": 0.0,
                "wins": 0,
                "losses": 0,
                "total_markets": 0,
                "trade_count": 0,
                "error": str(e)
            }

    async def discover_by_win_rate(
        self,
        min_win_rate: float = 70.0,
        min_trades: int = 10,
        limit: int = 20,
        time_period: str = "ALL",
        category: str = "OVERALL",
        min_volume: float = 0.0,
        max_volume: float = 0.0,
        scan_count: int = 100
    ) -> list[dict]:
        """
        Discover traders with high win rates.
        Fetches leaderboard, calculates win rates, and filters by threshold.

        Args:
            min_win_rate: Minimum win rate percentage (0-100)
            min_trades: Minimum trades required
            limit: Max results to return
            time_period: DAY, WEEK, MONTH, or ALL
            category: Market category filter
            min_volume: Minimum trading volume filter (0 = no minimum)
            max_volume: Maximum trading volume filter (0 = no maximum)
            scan_count: Number of traders to scan from leaderboard (can exceed 50)
        """
        # Fetch traders - use pagination to get more than 50
        if scan_count > 50:
            leaderboard = await self.get_leaderboard_paginated(
                total_limit=scan_count,
                time_period=time_period,
                order_by="PNL",  # Start with profitable traders
                category=category
            )
        else:
            leaderboard = await self.get_leaderboard(
                limit=scan_count,
                time_period=time_period,
                order_by="PNL",
                category=category
            )

        # Pre-filter by volume if specified
        if min_volume > 0 or max_volume > 0:
            filtered_leaderboard = []
            for entry in leaderboard:
                vol = float(entry.get("vol", 0))
                if min_volume > 0 and vol < min_volume:
                    continue
                if max_volume > 0 and vol > max_volume:
                    continue
                filtered_leaderboard.append(entry)
            leaderboard = filtered_leaderboard

        results = []
        semaphore = asyncio.Semaphore(10)  # Increase concurrent requests for speed

        async def analyze_trader(entry: dict):
            async with semaphore:
                address = entry.get("proxyWallet", "")
                if not address:
                    return None

                win_rate_data = await self.calculate_wallet_win_rate(address)

                # Apply filters
                if win_rate_data.get("trade_count", 0) < min_trades:
                    return None
                if win_rate_data.get("win_rate", 0) < min_win_rate:
                    return None

                return {
                    "address": address,
                    "username": entry.get("userName", ""),
                    "volume": float(entry.get("vol", 0)),
                    "pnl": float(entry.get("pnl", 0)),
                    "rank": entry.get("rank", 0),
                    "win_rate": win_rate_data.get("win_rate", 0),
                    "wins": win_rate_data.get("wins", 0),
                    "losses": win_rate_data.get("losses", 0),
                    "total_markets": win_rate_data.get("total_markets", 0),
                    "trade_count": win_rate_data.get("trade_count", 0)
                }

        # Analyze all traders concurrently
        tasks = [analyze_trader(entry) for entry in leaderboard]
        analyzed = await asyncio.gather(*tasks)

        # Filter out None results and sort by win rate
        results = [r for r in analyzed if r is not None]
        results.sort(key=lambda x: x["win_rate"], reverse=True)

        return results[:limit]

    async def get_wallet_pnl(self, address: str) -> dict:
        """
        Calculate PnL for a wallet by analyzing their positions and trades.
        """
        try:
            positions = await self.get_wallet_positions(address)
            trades = await self.get_wallet_trades(address, limit=500)

            total_invested = 0.0
            total_returned = 0.0
            winning_trades = 0
            losing_trades = 0

            for trade in trades:
                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                side = trade.get("side", "").upper()
                outcome = trade.get("outcome", "")

                if side == "BUY":
                    total_invested += size * price
                elif side == "SELL":
                    total_returned += size * price

            # Calculate current position value AND cost basis
            position_value = 0.0
            position_cost_basis = 0.0
            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0)) or 0)
                current_price = float(pos.get("currentPrice", pos.get("price", 0)) or 0)
                position_value += size * current_price
                position_cost_basis += size * avg_price

            realized_pnl = total_returned - total_invested
            unrealized_pnl = position_value - position_cost_basis
            total_pnl = realized_pnl + unrealized_pnl

            return {
                "address": address,
                "total_trades": len(trades),
                "open_positions": len(positions),
                "total_invested": total_invested,
                "total_returned": total_returned,
                "position_value": position_value,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "roi_percent": (total_pnl / total_invested * 100) if total_invested > 0 else 0
            }
        except Exception as e:
            print(f"Error calculating PnL for {address}: {e}")
            return {
                "address": address,
                "error": str(e)
            }


# Singleton instance
polymarket_client = PolymarketClient()
