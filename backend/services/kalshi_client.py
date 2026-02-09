import httpx
import asyncio
import time
from typing import Optional
from datetime import datetime

from models import Market, Event, Token
from utils.logger import get_logger

logger = get_logger("kalshi")

# Kalshi public API base URL
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    """Client for interacting with Kalshi's prediction market API.

    Mirrors the PolymarketClient interface so that cross-platform scanners
    can work with either data source through a common set of methods.

    Public endpoints (unauthenticated):
        GET /trade-api/v2/events
        GET /trade-api/v2/markets
        GET /trade-api/v2/markets/{ticker}

    Authenticated endpoints (require login or API key):
        POST /trade-api/v2/login
        GET  /trade-api/v2/portfolio/balance
        GET  /trade-api/v2/portfolio/positions
        POST /trade-api/v2/portfolio/orders
    """

    def __init__(self):
        self.base_url: str = KALSHI_API_BASE
        self._client: Optional[httpx.AsyncClient] = None
        self._trading_client: Optional[httpx.AsyncClient] = (
            None  # Proxy-aware for trading
        )

        # Authentication state
        self._auth_token: Optional[str] = None
        self._auth_member_id: Optional[str] = None
        self._api_key: Optional[str] = None

        # Token-bucket rate limiters aligned with Kalshi Basic tier
        # (20 reads/s, 10 writes/s).  We use 10 reads/s with capacity 20
        # to stay safely below the ceiling while still allowing bursts.
        self._read_rate: float = 10.0  # sustained reads per second
        self._read_bucket: float = 20.0
        self._read_capacity: float = 20.0
        self._read_last_refill: float = time.monotonic()
        self._read_lock: asyncio.Lock = asyncio.Lock()

        self._write_rate: float = 5.0  # sustained writes per second
        self._write_bucket: float = 10.0
        self._write_capacity: float = 10.0
        self._write_last_refill: float = time.monotonic()
        self._write_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  HTTP helpers
    # ------------------------------------------------------------------ #

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a long-lived async HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "homerun-arb-scanner/1.0",
                },
            )
        return self._client

    async def _get_trading_client(self) -> httpx.AsyncClient:
        """Get a proxy-aware client for trading-related Kalshi calls.

        Falls back to the standard client if proxy is not configured.
        Reads proxy state from the DB-backed cached config.
        """
        from services.trading_proxy import _get_config, get_async_proxy_client

        if not _get_config().enabled:
            return await self._get_client()

        if self._trading_client is None or self._trading_client.is_closed:
            self._trading_client = get_async_proxy_client()
        return self._trading_client

    async def close(self):
        """Shut down the HTTP client cleanly."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._trading_client and not self._trading_client.is_closed:
            await self._trading_client.aclose()

    async def _rate_limit_wait(self, write: bool = False):
        """Block until a request token is available.

        Uses separate read (10 req/s) and write (5 req/s) buckets to match
        Kalshi's per-category limits.
        """
        if write:
            lock, attr = self._write_lock, "write"
        else:
            lock, attr = self._read_lock, "read"

        async with lock:
            bucket = getattr(self, f"_{attr}_bucket")
            capacity = getattr(self, f"_{attr}_capacity")
            rate = getattr(self, f"_{attr}_rate")
            last = getattr(self, f"_{attr}_last_refill")

            now = time.monotonic()
            elapsed = now - last
            bucket = min(capacity, bucket + elapsed * rate)
            setattr(self, f"_{attr}_last_refill", now)

            if bucket < 1.0:
                wait = (1.0 - bucket) / rate
                logger.debug("Kalshi rate-limit wait", kind=attr, wait_seconds=wait)
                await asyncio.sleep(wait)
                bucket = 0.0
                setattr(self, f"_{attr}_last_refill", time.monotonic())
            else:
                bucket -= 1.0

            setattr(self, f"_{attr}_bucket", bucket)

    def _drain_read_bucket(self):
        """Drain the read bucket after a 429 to prevent burst retries."""
        self._read_bucket = 0.0
        self._read_last_refill = time.monotonic()

    def _auth_headers(self) -> dict:
        """Return authorization headers if authenticated."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        elif self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Perform a rate-limited GET request with 429 retry and return the JSON body."""
        import random

        max_retries = 4
        for attempt in range(max_retries + 1):
            await self._rate_limit_wait()
            client = await self._get_client()
            url = f"{self.base_url}{path}"
            response = await client.get(
                url, params=params, headers=self._auth_headers()
            )
            if response.status_code == 429 and attempt < max_retries:
                # Drain bucket so other concurrent requests also pause
                self._drain_read_bucket()
                backoff = 2**attempt * (0.5 + random.random())  # jittered
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = max(backoff, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "Kalshi 429 rate limited, retrying",
                    path=path,
                    attempt=attempt + 1,
                    backoff_seconds=round(backoff, 2),
                )
                await asyncio.sleep(backoff)
                continue
            response.raise_for_status()
            return response.json()
        # Should not reach here, but satisfy type checker
        response.raise_for_status()
        return response.json()

    async def _post(self, path: str, json_body: Optional[dict] = None) -> dict:
        """Perform a rate-limited POST request with 429 retry and return the JSON body."""
        import random

        max_retries = 4
        for attempt in range(max_retries + 1):
            await self._rate_limit_wait(write=True)
            client = await self._get_client()
            url = f"{self.base_url}{path}"
            response = await client.post(
                url, json=json_body or {}, headers=self._auth_headers()
            )
            if response.status_code == 429 and attempt < max_retries:
                self._write_bucket = 0.0
                self._write_last_refill = time.monotonic()
                backoff = 2**attempt * (0.5 + random.random())
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = max(backoff, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "Kalshi 429 on POST, retrying",
                    path=path,
                    attempt=attempt + 1,
                    backoff_seconds=round(backoff, 2),
                )
                await asyncio.sleep(backoff)
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    #  Mapping helpers: Kalshi JSON -> existing models
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_kalshi_market(data: dict) -> Market:
        """Convert a single Kalshi market dict into the app's Market model."""
        ticker = data.get("ticker", "")
        title = data.get("title", "") or data.get("subtitle", "")

        # Kalshi prices are in cents (0-100); normalise to 0.0-1.0
        yes_bid = (data.get("yes_bid", 0) or 0) / 100.0
        yes_ask = (data.get("yes_ask", 0) or 0) / 100.0
        no_bid = (data.get("no_bid", 0) or 0) / 100.0
        no_ask = (data.get("no_ask", 0) or 0) / 100.0
        last_price = (data.get("last_price", 0) or 0) / 100.0

        # Use midpoint of bid/ask when available; fall back to last_price
        yes_price = (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else last_price
        no_price = (
            (no_bid + no_ask) / 2.0 if (no_bid + no_ask) > 0 else (1.0 - yes_price)
        )

        outcome_prices = [yes_price, no_price]

        # Build Token objects using the ticker as the token id
        yes_token = Token(token_id=f"{ticker}_yes", outcome="Yes", price=yes_price)
        no_token = Token(token_id=f"{ticker}_no", outcome="No", price=no_price)

        # Determine active / closed from status
        status = (data.get("status", "") or "").lower()
        is_active = status in ("open", "active", "")
        is_closed = status in ("closed", "settled", "finalized")

        # Parse close_time / expiration_time as end_date
        end_date = None
        for date_key in ("close_time", "expiration_time", "expected_expiration_time"):
            raw = data.get(date_key)
            if raw:
                try:
                    if isinstance(raw, str):
                        end_date = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    break
                except (ValueError, TypeError):
                    pass

        volume_raw = data.get("volume", 0) or 0
        liquidity_raw = data.get("liquidity", 0) or data.get("open_interest", 0) or 0

        return Market(
            id=ticker,
            condition_id=ticker,
            question=title,
            slug=ticker,
            tokens=[yes_token, no_token],
            clob_token_ids=[f"{ticker}_yes", f"{ticker}_no"],
            outcome_prices=outcome_prices,
            active=is_active,
            closed=is_closed,
            neg_risk=False,
            volume=float(volume_raw),
            liquidity=float(liquidity_raw),
            end_date=end_date,
            platform="kalshi",
        )

    @staticmethod
    def _parse_kalshi_event(data: dict) -> Event:
        """Convert a single Kalshi event dict into the app's Event model."""
        event_ticker = data.get("event_ticker", "") or data.get("ticker", "")
        title = data.get("title", "")
        sub_title = data.get("sub_title", "") or data.get("subtitle", "")
        category = data.get("category", None)

        # Parse nested markets if present
        markets: list[Market] = []
        for m in data.get("markets", []):
            try:
                markets.append(KalshiClient._parse_kalshi_market(m))
            except Exception:
                pass

        return Event(
            id=event_ticker,
            slug=event_ticker,
            title=title,
            description=sub_title,
            category=category,
            markets=markets,
            neg_risk=data.get("mutually_exclusive", False),
            active=True,
            closed=False,
        )

    # ------------------------------------------------------------------ #
    #  Public API: events
    # ------------------------------------------------------------------ #

    async def get_events(
        self,
        closed: bool = False,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[list[Event], Optional[str]]:
        """Fetch one page of events from Kalshi.

        Returns (events, next_cursor).  next_cursor is None when there
        are no more pages.
        """
        params: dict = {"limit": min(limit, 200)}
        if cursor:
            params["cursor"] = cursor
        if not closed:
            params["status"] = "open"

        try:
            data = await self._get("/events", params=params)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Kalshi events request failed",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return [], None
        except Exception as exc:
            logger.error("Kalshi events request error", error=str(exc))
            return [], None

        events_raw = data.get("events", [])
        next_cursor = data.get("cursor", None)
        # An empty cursor string means no more pages
        if not next_cursor:
            next_cursor = None

        events = []
        for e in events_raw:
            try:
                events.append(self._parse_kalshi_event(e))
            except Exception as exc:
                logger.debug("Failed to parse Kalshi event", error=str(exc))

        return events, next_cursor

    async def get_all_events(self, closed: bool = False) -> list[Event]:
        """Fetch all events with automatic cursor-based pagination."""
        all_events: list[Event] = []
        cursor: Optional[str] = None
        page = 0
        max_pages = 20  # safety cap

        while page < max_pages:
            events, next_cursor = await self.get_events(
                closed=closed,
                limit=200,
                cursor=cursor,
            )
            if not events:
                break
            all_events.extend(events)
            cursor = next_cursor
            page += 1
            if cursor is None:
                break

        logger.info("Fetched Kalshi events", count=len(all_events))
        return all_events

    async def search_events(self, query: str, limit: int = 20) -> list[Event]:
        """Search Kalshi events by keyword.

        Kalshi's API does not provide native text search, so we fetch
        events in pages and filter by title match on the client side.
        For each matching event we also fetch its markets.
        """
        query_lower = query.lower()
        matching_events: list[Event] = []
        cursor: Optional[str] = None
        max_pages = 10  # safety cap

        for _ in range(max_pages):
            events, next_cursor = await self.get_events(
                closed=False, limit=200, cursor=cursor
            )
            if not events:
                break

            for event in events:
                if query_lower in event.title.lower():
                    matching_events.append(event)
                    if len(matching_events) >= limit:
                        break

            if len(matching_events) >= limit:
                break

            cursor = next_cursor
            if cursor is None:
                break

        # For events with no nested markets, fetch markets by event_ticker
        for event in matching_events:
            if not event.markets:
                markets, _ = await self.get_markets_page(
                    limit=200, event_ticker=event.id, status="open"
                )
                event.markets = markets

        logger.info(
            "Kalshi search complete",
            query=query,
            events_found=len(matching_events),
        )
        return matching_events[:limit]

    # ------------------------------------------------------------------ #
    #  Public API: markets
    # ------------------------------------------------------------------ #

    async def get_markets_page(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[Market], Optional[str]]:
        """Fetch one page of markets.

        Returns (markets, next_cursor).
        """
        params: dict = {"limit": min(limit, 200)}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status

        try:
            data = await self._get("/markets", params=params)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Kalshi markets request failed",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return [], None
        except Exception as exc:
            logger.error("Kalshi markets request error", error=str(exc))
            return [], None

        markets_raw = data.get("markets", [])
        next_cursor = data.get("cursor", None)
        if not next_cursor:
            next_cursor = None

        markets: list[Market] = []
        for m in markets_raw:
            try:
                markets.append(self._parse_kalshi_market(m))
            except Exception as exc:
                logger.debug("Failed to parse Kalshi market", error=str(exc))

        return markets, next_cursor

    async def get_all_markets(self, active: bool = True) -> list[Market]:
        """Fetch all markets with cursor-based pagination."""
        all_markets: list[Market] = []
        cursor: Optional[str] = None
        page = 0
        max_pages = 30  # safety cap
        status = "open" if active else None

        while page < max_pages:
            markets, next_cursor = await self.get_markets_page(
                limit=200,
                cursor=cursor,
                status=status,
            )
            if not markets:
                break
            all_markets.extend(markets)
            cursor = next_cursor
            page += 1
            if cursor is None:
                break

        logger.info("Fetched Kalshi markets", count=len(all_markets))
        return all_markets

    # ------------------------------------------------------------------ #
    #  Public API: single market
    # ------------------------------------------------------------------ #

    async def get_market(self, market_id: str) -> Optional[Market]:
        """Fetch a single market by ticker.

        Returns None if the market cannot be found or an error occurs.
        """
        try:
            data = await self._get(f"/markets/{market_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Kalshi market not found", ticker=market_id)
                return None
            logger.error(
                "Kalshi single-market request failed",
                ticker=market_id,
                status=exc.response.status_code,
            )
            return None
        except Exception as exc:
            logger.error(
                "Kalshi single-market request error",
                ticker=market_id,
                error=str(exc),
            )
            return None

        market_data = data.get("market", data)
        try:
            return self._parse_kalshi_market(market_data)
        except Exception as exc:
            logger.error(
                "Failed to parse Kalshi market", ticker=market_id, error=str(exc)
            )
            return None

    # ------------------------------------------------------------------ #
    #  Public API: batch prices
    # ------------------------------------------------------------------ #

    async def get_prices_batch(self, token_ids: list[str]) -> dict[str, dict]:
        """Get current prices for a list of Kalshi token IDs.

        Because Kalshi does not have a dedicated batch-price endpoint, we
        fetch each underlying market individually (de-duplicated by ticker)
        and return a mapping of ``token_id -> {mid, yes, no}``.

        Token IDs are expected in the format ``TICKER_yes`` / ``TICKER_no``
        (the same format produced by ``_parse_kalshi_market``).
        """
        # De-duplicate tickers
        ticker_set: set[str] = set()
        for tid in token_ids:
            # Strip the _yes / _no suffix to get the ticker
            base = tid.rsplit("_", 1)[0] if "_" in tid else tid
            ticker_set.add(base)

        # Fetch markets concurrently with concurrency limit
        semaphore = asyncio.Semaphore(10)
        ticker_market_map: dict[str, Market] = {}

        async def fetch_one(ticker: str):
            async with semaphore:
                market = await self.get_market(ticker)
                if market:
                    ticker_market_map[ticker] = market

        await asyncio.gather(*[fetch_one(t) for t in ticker_set])

        # Build result dict keyed by the original token_id
        prices: dict[str, dict] = {}
        for tid in token_ids:
            base = tid.rsplit("_", 1)[0] if "_" in tid else tid
            suffix = tid.rsplit("_", 1)[1] if "_" in tid else "yes"
            market = ticker_market_map.get(base)
            if market:
                yes_p = market.yes_price
                no_p = market.no_price
                mid = yes_p if suffix == "yes" else no_p
                prices[tid] = {"mid": mid, "yes": yes_p, "no": no_p}
            else:
                prices[tid] = {"mid": 0, "error": "market_not_found"}

        return prices

    # ------------------------------------------------------------------ #
    #  Authentication
    # ------------------------------------------------------------------ #

    @property
    def is_authenticated(self) -> bool:
        """Check if the client has valid authentication credentials."""
        return bool(self._auth_token or self._api_key)

    async def login(self, email: str, password: str) -> bool:
        """Authenticate with Kalshi using email/password.

        On success, stores the bearer token for subsequent authenticated requests.
        Returns True if login succeeded, False otherwise.
        """
        try:
            await self._rate_limit_wait(write=True)
            client = await self._get_client()
            url = f"{self.base_url}/login"
            response = await client.post(
                url, json={"email": email, "password": password}
            )
            response.raise_for_status()
            data = response.json()
            self._auth_token = data.get("token")
            self._auth_member_id = data.get("member_id")
            logger.info(
                "Kalshi login successful",
                member_id=self._auth_member_id,
            )
            return True
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Kalshi login failed",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return False
        except Exception as exc:
            logger.error("Kalshi login error", error=str(exc))
            return False

    def set_api_key(self, api_key: str):
        """Set API key for authentication (alternative to email/password login)."""
        self._api_key = api_key
        logger.info("Kalshi API key configured")

    async def initialize_auth(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """Initialize authentication using stored credentials.

        Tries API key first (no network call), then email/password login.
        Returns True if authenticated successfully.
        """
        if api_key:
            self.set_api_key(api_key)
            return True
        if email and password:
            return await self.login(email, password)
        return False

    def logout(self):
        """Clear authentication state."""
        self._auth_token = None
        self._auth_member_id = None
        self._api_key = None

    # ------------------------------------------------------------------ #
    #  Authenticated API: portfolio
    # ------------------------------------------------------------------ #

    async def get_balance(self) -> Optional[dict]:
        """Get account balance.

        Returns dict with 'balance' (cents), 'payout' etc., or None on failure.
        """
        if not self.is_authenticated:
            return None
        try:
            data = await self._get("/portfolio/balance")
            # Kalshi returns balance in cents, convert to dollars
            raw_balance = data.get("balance", 0) or 0
            raw_payout = data.get("payout", 0) or 0
            return {
                "balance": raw_balance / 100.0,
                "payout": raw_payout / 100.0,
                "available": raw_balance / 100.0,
                "reserved": 0.0,
                "currency": "USD",
            }
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Kalshi balance request failed",
                status=exc.response.status_code,
            )
            return None
        except Exception as exc:
            logger.error("Kalshi balance error", error=str(exc))
            return None

    async def get_positions(self) -> list[dict]:
        """Get current portfolio positions.

        Returns a list of position dicts normalised to match the TradingPosition
        format used by the Polymarket live account panel.
        """
        if not self.is_authenticated:
            return []
        try:
            positions = []
            cursor: Optional[str] = None
            max_pages = 10

            for _ in range(max_pages):
                params: dict = {"limit": 200}
                if cursor:
                    params["cursor"] = cursor
                data = await self._get("/portfolio/positions", params=params)

                for pos in data.get("market_positions", []):
                    ticker = pos.get("ticker", "")
                    total_traded = pos.get("total_traded", 0) or 0
                    realized_pnl = pos.get("realized_pnl", 0) or 0

                    # Kalshi positions can have yes/no quantities
                    yes_qty = pos.get("position", 0) or 0  # positive = long YES
                    rest_qty = pos.get("rest_position", 0) or 0

                    if yes_qty == 0 and rest_qty == 0:
                        continue

                    # Determine side and quantity
                    if yes_qty > 0:
                        outcome = "YES"
                        size = yes_qty
                    elif yes_qty < 0:
                        outcome = "NO"
                        size = abs(yes_qty)
                    else:
                        outcome = "YES" if rest_qty > 0 else "NO"
                        size = abs(rest_qty)

                    # Estimate average cost from total_traded / size
                    avg_cost = (total_traded / 100.0 / size) if size > 0 else 0.0

                    positions.append(
                        {
                            "token_id": f"{ticker}_{'yes' if outcome == 'YES' else 'no'}",
                            "market_id": ticker,
                            "market_question": ticker,  # Will be enriched with title if available
                            "outcome": outcome,
                            "size": float(size),
                            "average_cost": avg_cost,
                            "current_price": 0.0,  # Will be enriched with live price
                            "unrealized_pnl": realized_pnl / 100.0,
                            "platform": "kalshi",
                        }
                    )

                cursor = data.get("cursor")
                if not cursor:
                    break

            # Enrich positions with current prices
            if positions:
                tickers = list({p["market_id"] for p in positions})
                for ticker in tickers:
                    market = await self.get_market(ticker)
                    if market:
                        for p in positions:
                            if p["market_id"] == ticker:
                                p["market_question"] = market.question
                                p["current_price"] = (
                                    market.yes_price
                                    if p["outcome"] == "YES"
                                    else market.no_price
                                )
                                p["unrealized_pnl"] = p["size"] * (
                                    p["current_price"] - p["average_cost"]
                                )

            return positions
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Kalshi positions request failed",
                status=exc.response.status_code,
            )
            return []
        except Exception as exc:
            logger.error("Kalshi positions error", error=str(exc))
            return []

    async def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        order_type: str = "limit",
    ) -> Optional[dict]:
        """Place an order on Kalshi.

        Args:
            ticker: Market ticker (e.g., 'KXBTC-24DEC31-T100000')
            side: 'yes' or 'no'
            count: Number of contracts
            price_cents: Price per contract in cents (1-99)
            order_type: 'limit' or 'market'

        Returns order details dict, or None on failure.
        """
        if not self.is_authenticated:
            logger.error("Cannot place order: not authenticated")
            return None
        try:
            body = {
                "ticker": ticker,
                "action": "buy",
                "side": side,
                "count": count,
                "type": order_type,
            }
            if order_type == "limit":
                body["yes_price"] = (
                    price_cents if side == "yes" else (100 - price_cents)
                )

            data = await self._post("/portfolio/orders", json_body=body)
            order = data.get("order", data)
            logger.info(
                "Kalshi order placed",
                ticker=ticker,
                side=side,
                count=count,
                order_id=order.get("order_id"),
            )
            return order
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Kalshi order failed",
                ticker=ticker,
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error("Kalshi order error", ticker=ticker, error=str(exc))
            return None

    async def get_orders(self, status: Optional[str] = None) -> list[dict]:
        """Get order history.

        Args:
            status: Filter by status ('resting', 'canceled', 'executed', 'pending')

        Returns list of order dicts.
        """
        if not self.is_authenticated:
            return []
        try:
            params: dict = {"limit": 200}
            if status:
                params["status"] = status
            data = await self._get("/portfolio/orders", params=params)
            return data.get("orders", [])
        except Exception as exc:
            logger.error("Kalshi get orders error", error=str(exc))
            return []

    async def get_account_status(self) -> dict:
        """Get overall account status including auth state and balance.

        Returns a summary dict suitable for the frontend.
        """
        result = {
            "platform": "kalshi",
            "authenticated": self.is_authenticated,
            "member_id": self._auth_member_id,
            "balance": None,
            "positions_count": 0,
        }
        if self.is_authenticated:
            balance = await self.get_balance()
            if balance:
                result["balance"] = balance
            positions = await self.get_positions()
            result["positions_count"] = len(positions)
        return result


# Singleton instance
kalshi_client = KalshiClient()
