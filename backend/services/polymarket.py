import httpx
import asyncio
from typing import Optional
from datetime import datetime, timedelta

from config import settings
from models import Market, Event


class PolymarketClient:
    """Client for interacting with Polymarket APIs"""

    def __init__(self):
        self.gamma_url = settings.GAMMA_API_URL
        self.clob_url = settings.CLOB_API_URL
        self.data_url = settings.DATA_API_URL
        self._client: Optional[httpx.AsyncClient] = None
        self._market_cache: dict[str, dict] = {}  # condition_id -> {question, slug}
        self._username_cache: dict[str, str] = {}  # address (lowercase) -> username
        self._persistent_cache = None  # Lazy-loaded MarketCacheService

    async def _get_persistent_cache(self):
        """Lazy-load the persistent market cache service."""
        if self._persistent_cache is None:
            try:
                from services.market_cache import market_cache_service
                if not market_cache_service._loaded:
                    await market_cache_service.load_from_db()
                self._persistent_cache = market_cache_service
                # Pre-populate in-memory caches from DB
                self._market_cache.update(market_cache_service._market_cache)
                self._username_cache.update(market_cache_service._username_cache)
            except Exception:
                pass  # Graceful degradation: in-memory only
        return self._persistent_cache

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
        offset: int = 0,
    ) -> list[Market]:
        """Fetch markets from Gamma API"""
        client = await self._get_client()
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
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
            markets = await self.get_markets(active=active, limit=limit, offset=offset)
            if not markets:
                break

            all_markets.extend(markets)
            offset += limit

            if len(all_markets) >= settings.MAX_MARKETS_TO_SCAN:
                break

        return all_markets

    async def get_events(
        self, closed: bool = False, limit: int = 100, offset: int = 0
    ) -> list[Event]:
        """Fetch events from Gamma API (events contain grouped markets)"""
        client = await self._get_client()
        params = {"closed": str(closed).lower(), "limit": limit, "offset": offset}

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
            events = await self.get_events(closed=closed, limit=limit, offset=offset)
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

    async def get_market_by_condition_id(self, condition_id: str) -> Optional[dict]:
        """Look up a market by condition_id, using cache when available."""
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.gamma_url}/markets",
                params={"condition_id": condition_id, "limit": 1},
            )
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                market_data = data[0]
                info = {
                    "question": market_data.get("question", ""),
                    "slug": market_data.get("slug", ""),
                    "groupItemTitle": market_data.get("groupItemTitle", ""),
                }
                self._market_cache[condition_id] = info

                # Write-through to persistent SQL cache
                cache = await self._get_persistent_cache()
                if cache:
                    try:
                        await cache.set_market(condition_id, info)
                    except Exception:
                        pass  # Non-critical

                return info
        except Exception as e:
            print(f"Market lookup failed for {condition_id}: {e}")

        return None

    async def enrich_trades_with_market_info(self, trades: list[dict]) -> list[dict]:
        """
        Enrich a list of trades with market question/title and slug.
        Batches lookups and uses cache to minimize API calls.
        """
        # Collect unique condition_ids that need lookup
        unknown_ids = set()
        for trade in trades:
            condition_id = trade.get("market", "")
            if condition_id and condition_id not in self._market_cache:
                unknown_ids.add(condition_id)

        # Batch lookup unknown condition_ids with concurrency limit
        if unknown_ids:
            semaphore = asyncio.Semaphore(5)

            async def lookup(cid: str):
                async with semaphore:
                    await self.get_market_by_condition_id(cid)

            await asyncio.gather(*[lookup(cid) for cid in unknown_ids])

        # Enrich trades
        enriched = []
        for trade in trades:
            condition_id = trade.get("market", "")
            market_info = self._market_cache.get(condition_id)

            enriched_trade = {**trade}
            if market_info:
                enriched_trade["market_title"] = market_info.get("question", "")
                enriched_trade["market_slug"] = market_info.get("slug", "")
                # Use groupItemTitle if available (for multi-outcome markets)
                if market_info.get("groupItemTitle"):
                    enriched_trade["market_title"] = market_info["groupItemTitle"]
            else:
                enriched_trade["market_title"] = ""
                enriched_trade["market_slug"] = ""

            # Normalize timestamp to ISO format
            ts = (
                trade.get("match_time")
                or trade.get("timestamp")
                or trade.get("time")
                or trade.get("created_at")
                or trade.get("createdAt")
            )
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        # Unix timestamp in seconds
                        enriched_trade["timestamp_iso"] = (
                            datetime.utcfromtimestamp(ts).isoformat() + "Z"
                        )
                    elif isinstance(ts, str):
                        if "T" in ts or "-" in ts:
                            # Already ISO format
                            enriched_trade["timestamp_iso"] = ts
                        else:
                            # Numeric string (unix seconds)
                            enriched_trade["timestamp_iso"] = (
                                datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
                            )
                except (ValueError, TypeError, OSError):
                    enriched_trade["timestamp_iso"] = ""
            else:
                enriched_trade["timestamp_iso"] = ""

            # Compute total cost if missing
            if "cost" not in enriched_trade or enriched_trade.get("cost") is None:
                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                enriched_trade["cost"] = size * price

            enriched.append(enriched_trade)

        return enriched

    # ==================== CLOB API ====================

    async def get_midpoint(self, token_id: str) -> float:
        """Get midpoint price for a token"""
        client = await self._get_client()
        response = await client.get(
            f"{self.clob_url}/midpoint", params={"token_id": token_id}
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get("mid", 0))

    async def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Get best price for a token (BUY = best ask, SELL = best bid)"""
        client = await self._get_client()
        response = await client.get(
            f"{self.clob_url}/price", params={"token_id": token_id, "side": side}
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get("price", 0))

    async def get_order_book(self, token_id: str) -> dict:
        """Get full order book for a token"""
        client = await self._get_client()
        response = await client.get(
            f"{self.clob_url}/book", params={"token_id": token_id}
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
            f"{self.data_url}/positions", params={"user": address}
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
        Tries multiple sources: username cache, leaderboard API, data API, and website scraping.
        """
        client = await self._get_client()
        address_lower = address.lower()

        # Check username cache first (populated by discover/leaderboard scans)
        if address_lower in self._username_cache:
            return {
                "username": self._username_cache[address_lower],
                "address": address,
            }

        # Check persistent SQL cache
        cache = await self._get_persistent_cache()
        if cache:
            cached_username = await cache.get_username(address_lower)
            if cached_username:
                self._username_cache[address_lower] = cached_username
                return {"username": cached_username, "address": address}

        # Try the leaderboard API - search both PNL and VOL sorted, multiple pages
        try:
            for order_by in ["PNL", "VOL"]:
                for offset in range(0, 200, 50):
                    leaderboard = await self.get_leaderboard(
                        limit=50, order_by=order_by, offset=offset
                    )
                    if not leaderboard:
                        break
                    for entry in leaderboard:
                        proxy_wallet = entry.get("proxyWallet", "").lower()
                        # Cache all usernames we see
                        uname = entry.get("userName", "")
                        if proxy_wallet and uname:
                            self._username_cache[proxy_wallet] = uname
                        if proxy_wallet == address_lower and uname:
                            return {
                                "username": uname,
                                "address": address,
                                "pnl": float(entry.get("pnl", 0)),
                                "volume": float(entry.get("vol", 0)),
                                "rank": entry.get("rank", 0),
                            }
        except Exception as e:
            print(f"Leaderboard lookup failed: {e}")

        # Try the data API profile endpoint
        try:
            response = await client.get(
                f"{self.data_url}/profile", params={"address": address}
            )
            if response.status_code == 200:
                data = response.json()
                if data and data.get("username"):
                    return {
                        "username": data.get("username"),
                        "address": address,
                        **data,
                    }
        except Exception:
            pass

        # Try the users endpoint which may have username
        try:
            response = await client.get(
                f"{self.data_url}/users", params={"proxyAddress": address}
            )
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    user = data[0]
                    username = (
                        user.get("name") or user.get("username") or user.get("userName")
                    )
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
                follow_redirects=True,
            )
            if response.status_code == 200:
                html = response.text
                import re

                # Try title tag
                title_match = re.search(r"<title>([^|<]+)\s*\|?\s*Polymarket", html)
                if title_match:
                    username = title_match.group(1).strip()
                    if username and username.lower() != address_lower[:10]:
                        return {"username": username, "address": address}

                # Try meta og:title
                og_match = re.search(
                    r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"', html
                )
                if og_match:
                    username = og_match.group(1).strip()
                    if username and "polymarket" not in username.lower():
                        return {"username": username, "address": address}

        except Exception as e:
            print(f"Error fetching profile for {address}: {e}")

        return {"username": None, "address": address}

    async def get_wallet_trades(self, address: str, limit: int = 100) -> list[dict]:
        """Get recent trades for a wallet"""
        client = await self._get_client()
        response = await client.get(
            f"{self.data_url}/trades", params={"user": address, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    async def get_market_trades(
        self, condition_id: str, limit: int = 100
    ) -> list[dict]:
        """Get recent trades for a market"""
        client = await self._get_client()
        response = await client.get(
            f"{self.data_url}/trades", params={"market": condition_id, "limit": limit}
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
        offset: int = 0,
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
                f"{self.data_url}/v1/leaderboard", params=params
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
        category: str = "OVERALL",
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
                offset=offset,
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
        category: str = "OVERALL",
    ) -> list[dict]:
        """
        Get top traders from Polymarket leaderboard with verified trade counts.
        Fetches leaderboard then verifies each trader has real activity.
        """
        # Fetch more candidates than needed so filtering still yields enough
        scan_count = max(limit * 3, 100)
        leaderboard = await self.get_leaderboard_paginated(
            total_limit=scan_count,
            time_period=time_period,
            order_by=order_by,
            category=category,
        )

        # Cache usernames from leaderboard for later profile lookups
        for entry in leaderboard:
            addr = (entry.get("proxyWallet", "") or "").lower()
            uname = entry.get("userName", "")
            if addr and uname:
                self._username_cache[addr] = uname

        # Verify each trader has real activity by fetching win rate data
        semaphore = asyncio.Semaphore(15)
        results = []

        async def verify_trader(entry: dict):
            async with semaphore:
                address = entry.get("proxyWallet", "")
                if not address:
                    return None

                win_rate_data = await self.calculate_wallet_win_rate(address)

                # Use actual closed positions (wins + losses) for min_trades
                closed_positions = win_rate_data.get("wins", 0) + win_rate_data.get(
                    "losses", 0
                )
                if closed_positions < min_trades:
                    return None

                return {
                    "address": address,
                    "username": entry.get("userName", ""),
                    "trades": win_rate_data.get("trade_count", 0),
                    "volume": float(entry.get("vol", 0) or 0),
                    "pnl": float(entry.get("pnl", 0) or 0),
                    "rank": entry.get("rank", 0),
                    "buys": win_rate_data.get("wins", 0),
                    "sells": win_rate_data.get("losses", 0),
                    "win_rate": win_rate_data.get("win_rate", 0),
                    "wins": win_rate_data.get("wins", 0),
                    "losses": win_rate_data.get("losses", 0),
                    "total_markets": win_rate_data.get("total_markets", 0),
                    "trade_count": win_rate_data.get("trade_count", 0),
                }

        tasks = [verify_trader(entry) for entry in leaderboard]
        analyzed = await asyncio.gather(*tasks)
        results = [r for r in analyzed if r is not None]

        # Sort by the requested order
        if order_by.upper() == "VOL":
            results.sort(key=lambda x: x["volume"], reverse=True)
        else:
            results.sort(key=lambda x: x["pnl"], reverse=True)

        return results[:limit]

    def _filter_by_time_period(
        self, trades: list[dict], time_period: str
    ) -> list[dict]:
        """Filter trades by time period (DAY, WEEK, MONTH, ALL)."""
        if not trades or time_period.upper() == "ALL":
            return trades

        now = datetime.utcnow()
        period_map = {
            "DAY": timedelta(days=1),
            "WEEK": timedelta(weeks=1),
            "MONTH": timedelta(days=30),
        }
        delta = period_map.get(time_period.upper())
        if not delta:
            return trades

        cutoff = now - delta
        filtered = []
        for trade in trades:
            ts = (
                trade.get("timestamp")
                or trade.get("created_at")
                or trade.get("createdAt", "")
            )
            if not ts:
                filtered.append(trade)  # Keep trades without timestamps
                continue
            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(
                        tzinfo=None
                    )
                if ts >= cutoff:
                    filtered.append(trade)
            except (ValueError, TypeError):
                filtered.append(trade)  # Keep if we can't parse

        return filtered

    async def calculate_wallet_win_rate(
        self, address: str, max_trades: int = 500, time_period: str = "ALL"
    ) -> dict:
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

            # Apply time period filter to trades
            trades = self._filter_by_time_period(trades, time_period)

            if not trades and not positions:
                return {
                    "address": address,
                    "win_rate": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "total_markets": 0,
                    "trade_count": 0,
                }

            # Group trades by market
            markets: dict[str, dict] = {}
            for trade in trades:
                market_id = (
                    trade.get("market")
                    or trade.get("condition_id")
                    or trade.get("assetId", "unknown")
                )
                if market_id not in markets:
                    markets[market_id] = {
                        "buys": 0.0,
                        "sells": 0.0,
                        "buy_count": 0,
                        "sell_count": 0,
                        "buy_size": 0.0,
                        "sell_size": 0.0,
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

            win_rate = (
                (total_wins / total_positions * 100) if total_positions > 0 else 0.0
            )

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
                "unrealized_losses": position_losses,
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
                "error": str(e),
            }

    async def get_closed_positions(
        self, address: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """Fetch closed positions for a wallet. Much more efficient than analyzing raw trades."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.data_url}/closed-positions",
                params={
                    "user": address,
                    "limit": min(limit, 50),
                    "offset": offset,
                    "sortBy": "TIMESTAMP",
                    "sortDirection": "DESC",
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return []

    async def get_closed_positions_paginated(
        self, address: str, max_positions: int = 200
    ) -> list[dict]:
        """Fetch multiple pages of closed positions."""
        all_positions = []
        offset = 0
        page_size = 50

        while len(all_positions) < max_positions:
            page = await self.get_closed_positions(
                address, limit=page_size, offset=offset
            )
            if not page:
                break
            all_positions.extend(page)
            offset += len(page)
            if len(page) < page_size:
                break

        return all_positions[:max_positions]

    async def calculate_win_rate_fast(
        self, address: str, min_positions: int = 5
    ) -> Optional[dict]:
        """
        Fast win rate calculation using closed-positions endpoint.
        Returns None if trader doesn't meet minimum position threshold.
        Much faster than calculate_wallet_win_rate() since it uses a single
        pre-aggregated endpoint instead of fetching all raw trades.
        """
        try:
            closed = await self.get_closed_positions_paginated(
                address, max_positions=200
            )

            if len(closed) < min_positions:
                return None

            wins = 0
            losses = 0
            for pos in closed:
                realized_pnl = float(pos.get("realizedPnl", 0) or 0)
                if realized_pnl > 0:
                    wins += 1
                elif realized_pnl < 0:
                    losses += 1

            total = wins + losses
            if total < min_positions:
                return None

            win_rate = (wins / total * 100) if total > 0 else 0.0

            return {
                "win_rate": win_rate,
                "wins": wins,
                "losses": losses,
                "closed_positions": total,
            }
        except Exception:
            return None

    async def discover_by_win_rate(
        self,
        min_win_rate: float = 70.0,
        min_trades: int = 10,
        limit: int = 20,
        time_period: str = "ALL",
        category: str = "OVERALL",
        min_volume: float = 0.0,
        max_volume: float = 0.0,
        scan_count: int = 100,
    ) -> list[dict]:
        """
        Discover traders with high win rates. Scans the full leaderboard
        (both PNL and VOL sorts) and uses the fast closed-positions endpoint
        to calculate win rates efficiently.

        Args:
            min_win_rate: Minimum win rate percentage (0-100)
            min_trades: Minimum closed positions (wins + losses) required
            limit: Max results to return
            time_period: DAY, WEEK, MONTH, or ALL
            category: Market category filter
            min_volume: Minimum trading volume filter (0 = no minimum)
            max_volume: Maximum trading volume filter (0 = no maximum)
            scan_count: Number of traders to scan from each leaderboard sort
        """
        # Search both PNL and VOL leaderboards for maximum coverage
        seen_addresses = set()
        all_candidates = []

        for sort_by in ["PNL", "VOL"]:
            batch = await self.get_leaderboard_paginated(
                total_limit=min(scan_count, 1000),
                time_period=time_period,
                order_by=sort_by,
                category=category,
            )

            for entry in batch:
                addr = (entry.get("proxyWallet", "") or "").lower()
                if addr and addr not in seen_addresses:
                    seen_addresses.add(addr)
                    all_candidates.append(entry)
                    # Cache username for later profile lookups
                    uname = entry.get("userName", "")
                    if addr and uname:
                        self._username_cache[addr] = uname

        # Pre-filter by volume if specified
        if min_volume > 0 or max_volume > 0:
            filtered = []
            for entry in all_candidates:
                vol = float(entry.get("vol", 0) or 0)
                if min_volume > 0 and vol < min_volume:
                    continue
                if max_volume > 0 and vol > max_volume:
                    continue
                filtered.append(entry)
            all_candidates = filtered

        # Use higher concurrency since closed-positions is a single lightweight call
        semaphore = asyncio.Semaphore(25)

        async def analyze_trader(entry: dict):
            async with semaphore:
                address = entry.get("proxyWallet", "")
                if not address:
                    return None

                try:
                    result = await self.calculate_win_rate_fast(
                        address, min_positions=min_trades
                    )
                except Exception:
                    return None

                if not result:
                    return None
                if result["win_rate"] < min_win_rate:
                    return None

                return {
                    "address": address,
                    "username": entry.get("userName", ""),
                    "volume": float(entry.get("vol", 0) or 0),
                    "pnl": float(entry.get("pnl", 0) or 0),
                    "rank": entry.get("rank", 0),
                    "win_rate": result["win_rate"],
                    "wins": result["wins"],
                    "losses": result["losses"],
                    "total_markets": result["closed_positions"],
                    "trade_count": result["closed_positions"],
                }

        # Analyze ALL candidates concurrently
        tasks = [analyze_trader(entry) for entry in all_candidates]
        analyzed = await asyncio.gather(*tasks)

        # Filter out None results and sort by win rate
        results = [r for r in analyzed if r is not None]
        results.sort(key=lambda x: (x["win_rate"], x["wins"]), reverse=True)

        return results[:limit]

    async def get_wallet_pnl(self, address: str, time_period: str = "ALL") -> dict:
        """
        Calculate PnL for a wallet using closed-positions, trade history, and open position data.

        Uses closed-positions endpoint for accurate realized P&L (same data source as
        the Discover page), supplemented by open positions for unrealized P&L and
        trade history for buy/sell activity counts.
        """
        try:
            # Fetch all data sources in parallel for speed
            closed_positions, positions, trades = await asyncio.gather(
                self.get_closed_positions_paginated(address, max_positions=1000),
                self.get_wallet_positions_with_prices(address),
                self.get_wallet_trades(address, limit=500),
            )

            # Apply time period filter to trades
            trades = self._filter_by_time_period(trades, time_period)

            # === Realized P&L from closed positions (most accurate) ===
            closed_realized_pnl = 0.0
            closed_invested = 0.0
            closed_returned = 0.0
            for pos in closed_positions:
                rpnl = float(pos.get("realizedPnl", 0) or 0)
                closed_realized_pnl += rpnl
                # Try to get invested amount from closed positions
                init_val = float(pos.get("initialValue", 0) or 0)
                if init_val > 0:
                    closed_invested += init_val
                    closed_returned += init_val + rpnl

            # === Trade-based data (for buy/sell counts and fallback) ===
            total_bought = 0.0
            total_sold = 0.0
            for trade in trades:
                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                side = (trade.get("side", "") or "").upper()
                cost = size * price
                if side == "BUY":
                    total_bought += cost
                elif side == "SELL":
                    total_sold += cost

            trade_realized_pnl = total_sold - total_bought

            # === Open positions (for unrealized P&L) ===
            total_position_value = 0.0
            total_initial_value = 0.0
            total_cash_pnl = 0.0

            for pos in positions:
                current_value = float(
                    pos.get("currentValue", 0) or pos.get("current_value", 0) or 0
                )
                initial_value = float(
                    pos.get("initialValue", 0) or pos.get("initial_value", 0) or 0
                )
                cash_pnl = float(
                    pos.get("cashPnl", 0)
                    or pos.get("cash_pnl", 0)
                    or pos.get("pnl", 0)
                    or 0
                )

                # Fallback: calculate from size * price if API values are 0
                if current_value == 0 and initial_value == 0:
                    size = float(pos.get("size", 0) or 0)
                    avg_price = float(
                        pos.get("avgPrice", 0) or pos.get("avg_price", 0) or 0
                    )
                    current_price = float(
                        pos.get("currentPrice", 0)
                        or pos.get("curPrice", 0)
                        or pos.get("price", 0)
                        or 0
                    )
                    initial_value = size * avg_price
                    current_value = size * current_price

                total_position_value += current_value
                total_initial_value += initial_value
                total_cash_pnl += cash_pnl

            # Unrealized P&L from open positions
            unrealized_pnl = total_position_value - total_initial_value

            # Use closed-positions P&L when available (covers ALL closed positions,
            # not limited to 500 trades like trade history)
            if closed_positions:
                realized_pnl = closed_realized_pnl
            elif abs(trade_realized_pnl) > abs(total_cash_pnl):
                realized_pnl = trade_realized_pnl
            else:
                realized_pnl = total_cash_pnl

            total_pnl = realized_pnl + unrealized_pnl

            # Total invested: prefer closed-positions data if available
            if closed_invested > 0:
                total_invested = closed_invested + total_initial_value
                total_returned = closed_returned
            else:
                total_invested = (
                    total_bought if total_bought > 0 else total_initial_value
                )
                total_returned = (
                    total_sold
                    if total_sold > 0
                    else total_cash_pnl + total_initial_value
                )

            # Calculate ROI
            roi_percent = 0.0
            if total_invested > 0:
                roi_percent = (total_pnl / total_invested) * 100

            # Trade count: use closed positions count when available for accuracy
            total_positions_count = len(closed_positions) + len(positions)
            trade_count = total_positions_count if closed_positions else len(trades)

            return {
                "address": address,
                "total_trades": trade_count,
                "open_positions": len(positions),
                "total_invested": total_invested,
                "total_returned": total_returned,
                "position_value": total_position_value,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "roi_percent": roi_percent,
            }
        except Exception as e:
            print(f"Error calculating PnL for {address}: {e}")
            return {
                "address": address,
                "total_trades": 0,
                "open_positions": 0,
                "total_invested": 0,
                "total_returned": 0,
                "position_value": 0,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "total_pnl": 0,
                "roi_percent": 0,
                "error": str(e),
            }


# Singleton instance
polymarket_client = PolymarketClient()
