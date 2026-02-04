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

    async def get_wallet_positions_with_prices(self, address: str) -> list[dict]:
        """
        Get open positions for a wallet with enriched data.

        The Data API already returns:
        - curPrice: current market price
        - cashPnl: realized P&L
        - currentValue: current position value
        - initialValue: cost basis
        - avgPrice: average entry price
        - percentPnl: ROI percentage
        - title: market title
        - outcome: Yes/No
        """
        positions = await self.get_wallet_positions(address)

        if not positions:
            return []

        # Normalize field names and enrich with consistent naming
        enriched = []
        for pos in positions:
            # The API returns curPrice, not currentPrice
            current_price = float(pos.get("curPrice", 0) or 0)
            avg_price = float(pos.get("avgPrice", 0) or 0)
            size = float(pos.get("size", 0) or 0)

            # API provides these directly
            current_value = float(pos.get("currentValue", 0) or 0)
            initial_value = float(pos.get("initialValue", 0) or 0)
            cash_pnl = float(pos.get("cashPnl", 0) or 0)
            percent_pnl = float(pos.get("percentPnl", 0) or 0)

            enriched_pos = {
                **pos,
                # Normalized field names for frontend
                "currentPrice": current_price,
                "avgPrice": avg_price,
                "size": size,
                "currentValue": current_value,
                "initialValue": initial_value,
                "cashPnl": cash_pnl,
                "percentPnl": percent_pnl,
                "title": pos.get("title", ""),
                "outcome": pos.get("outcome", ""),
            }
            enriched.append(enriched_pos)

        return enriched

    async def get_user_profile(self, address: str) -> dict:
        """
        Get user profile info from Polymarket.
        Tries multiple sources: leaderboard API, data API, and website scraping.
        """
        client = await self._get_client()
        address_lower = address.lower()

        # Try the leaderboard API first - it has username for top traders
        try:
            leaderboard = await self.get_leaderboard(limit=50)
            for entry in leaderboard:
                proxy_wallet = entry.get("proxyWallet", "").lower()
                if proxy_wallet == address_lower:
                    username = entry.get("userName", "")
                    if username:
                        return {
                            "username": username,
                            "address": address,
                            "pnl": float(entry.get("pnl", 0)),
                            "volume": float(entry.get("vol", 0)),
                            "rank": entry.get("rank", 0)
                        }
        except Exception as e:
            print(f"Leaderboard lookup failed: {e}")

        # Try the data API profile endpoint
        try:
            response = await client.get(
                f"{self.data_url}/profile",
                params={"address": address}
            )
            if response.status_code == 200:
                data = response.json()
                if data and data.get("username"):
                    return {"username": data.get("username"), "address": address, **data}
        except Exception:
            pass

        # Try the users endpoint which may have username
        try:
            response = await client.get(
                f"{self.data_url}/users",
                params={"proxyAddress": address}
            )
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    user = data[0]
                    username = user.get("name") or user.get("username") or user.get("userName")
                    if username:
                        return {"username": username, "address": address, **user}
        except Exception:
            pass

        # Try fetching from the Polymarket website profile page
        try:
            response = await client.get(
                f"https://polymarket.com/profile/{address}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                follow_redirects=True
            )
            if response.status_code == 200:
                html = response.text
                import re

                # Try title tag
                title_match = re.search(r'<title>([^|<]+)\s*\|?\s*Polymarket', html)
                if title_match:
                    username = title_match.group(1).strip()
                    if username and username.lower() != address_lower[:10]:
                        return {"username": username, "address": address}

                # Try meta og:title
                og_match = re.search(r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"', html)
                if og_match:
                    username = og_match.group(1).strip()
                    if username and "polymarket" not in username.lower():
                        return {"username": username, "address": address}

        except Exception as e:
            print(f"Error fetching profile for {address}: {e}")

        return {"username": None, "address": address}

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
        category: str = "OVERALL"
    ) -> list[dict]:
        """
        Fetch top traders from Polymarket leaderboard.
        Returns list of wallets with their profit stats.

        Args:
            limit: Max results (1-50)
            time_period: DAY, WEEK, MONTH, or ALL
            order_by: PNL (profit/loss) or VOL (volume)
            category: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, WEATHER, ECONOMICS, TECH, FINANCE
        """
        client = await self._get_client()
        try:
            # Polymarket data API leaderboard endpoint
            params = {
                "limit": min(limit, 50),  # API max is 50
                "timePeriod": time_period.upper(),
                "orderBy": order_by.upper(),
            }
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
        Calculate win rate for a wallet by analyzing trade history and open positions.

        Considers:
        - Closed positions: sells > cost basis = win
        - Open positions: current value > cost basis = win (unrealized)

        Returns:
            dict with win_rate, wins, losses, total_markets, trade_count
        """
        try:
            # Fetch both trades and positions with current prices
            trades = await self.get_wallet_trades(address, limit=max_trades)
            positions = await self.get_wallet_positions_with_prices(address)

            if not trades and not positions:
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
                    markets[market_id] = {
                        "buys": 0.0, "sells": 0.0,
                        "buy_count": 0, "sell_count": 0,
                        "buy_size": 0.0, "sell_size": 0.0
                    }

                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                side = trade.get("side", "").upper()

                if side == "BUY":
                    markets[market_id]["buys"] += size * price
                    markets[market_id]["buy_count"] += 1
                    markets[market_id]["buy_size"] += size
                elif side == "SELL":
                    markets[market_id]["sells"] += size * price
                    markets[market_id]["sell_count"] += 1
                    markets[market_id]["sell_size"] += size

            # Process open positions to determine unrealized wins/losses
            # Use the API-provided currentValue and initialValue
            position_wins = 0
            position_losses = 0
            for pos in positions:
                # API provides these directly
                current_value = float(pos.get("currentValue", 0) or 0)
                initial_value = float(pos.get("initialValue", 0) or 0)
                cash_pnl = float(pos.get("cashPnl", 0) or 0)

                # A position is winning if current value + realized > initial value
                # Or simply if the API's cashPnl + (currentValue - initialValue) > 0
                total_position_pnl = cash_pnl + (current_value - initial_value)

                if total_position_pnl > 0:
                    position_wins += 1
                elif total_position_pnl < 0:
                    position_losses += 1

            # Calculate wins/losses from closed positions (positions with sells)
            closed_wins = 0
            closed_losses = 0
            for market_id, data in markets.items():
                # Only count markets with sells (at least partially closed)
                if data["sell_count"] > 0:
                    # If they sold for more than they bought = win
                    if data["sells"] > data["buys"]:
                        closed_wins += 1
                    elif data["sells"] < data["buys"]:
                        closed_losses += 1

            # Combine closed and open position results
            total_wins = closed_wins + position_wins
            total_losses = closed_losses + position_losses
            total_positions = total_wins + total_losses

            win_rate = (total_wins / total_positions * 100) if total_positions > 0 else 0.0

            return {
                "address": address,
                "win_rate": win_rate,
                "wins": total_wins,
                "losses": total_losses,
                "total_markets": len(markets),
                "trade_count": len(trades),
                "open_positions": len(positions),
                "closed_wins": closed_wins,
                "closed_losses": closed_losses,
                "unrealized_wins": position_wins,
                "unrealized_losses": position_losses
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
        category: str = "OVERALL"
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
        """
        # Fetch more traders initially since we'll filter many out
        leaderboard = await self.get_leaderboard(
            limit=50,
            time_period=time_period,
            order_by="PNL",  # Start with profitable traders
            category=category
        )

        results = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

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
        Calculate PnL for a wallet using the Data API's position data.

        The Polymarket Data API returns positions with:
        - cashPnl: realized P&L for the position
        - currentValue: current market value
        - initialValue: cost basis (what was invested)
        - percentPnl: ROI percentage
        """
        try:
            # Get positions with enriched data (already normalized)
            positions = await self.get_wallet_positions_with_prices(address)
            trades = await self.get_wallet_trades(address, limit=500)

            # Sum up position values directly from API data
            total_position_value = 0.0
            total_initial_value = 0.0
            total_cash_pnl = 0.0

            for pos in positions:
                # Use the API-provided values
                current_value = float(pos.get("currentValue", 0) or 0)
                initial_value = float(pos.get("initialValue", 0) or 0)
                cash_pnl = float(pos.get("cashPnl", 0) or 0)

                total_position_value += current_value
                total_initial_value += initial_value
                total_cash_pnl += cash_pnl

            # Unrealized P&L = current value - cost basis
            unrealized_pnl = total_position_value - total_initial_value

            # Total P&L = realized (cash) + unrealized
            total_pnl = total_cash_pnl + unrealized_pnl

            # Calculate ROI based on initial investment
            roi_percent = 0.0
            if total_initial_value > 0:
                roi_percent = (total_pnl / total_initial_value) * 100

            return {
                "address": address,
                "total_trades": len(trades),
                "open_positions": len(positions),
                "total_invested": total_initial_value,
                "total_returned": total_cash_pnl + total_initial_value,  # For compatibility
                "position_value": total_position_value,
                "realized_pnl": total_cash_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "roi_percent": roi_percent
            }
        except Exception as e:
            print(f"Error calculating PnL for {address}: {e}")
            return {
                "address": address,
                "error": str(e)
            }


# Singleton instance
polymarket_client = PolymarketClient()
