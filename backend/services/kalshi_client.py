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
    """Client for interacting with Kalshi's public prediction market API.

    Mirrors the PolymarketClient interface so that cross-platform scanners
    can work with either data source through a common set of methods.

    Uses only unauthenticated / public endpoints:
        GET /trade-api/v2/events
        GET /trade-api/v2/markets
        GET /trade-api/v2/markets/{ticker}
    """

    def __init__(self):
        self.base_url: str = KALSHI_API_BASE
        self._client: Optional[httpx.AsyncClient] = None
        self._trading_client: Optional[httpx.AsyncClient] = (
            None  # Proxy-aware for trading
        )

        # Simple token-bucket rate limiter: 10 requests / second
        self._rate_limit: float = 10.0  # requests per second
        self._token_bucket: float = 10.0
        self._bucket_capacity: float = 10.0
        self._last_refill: float = time.monotonic()
        self._rate_lock: asyncio.Lock = asyncio.Lock()

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

    async def _rate_limit_wait(self):
        """Block until a request token is available (10 req/s bucket)."""
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._token_bucket = min(
                self._bucket_capacity,
                self._token_bucket + elapsed * self._rate_limit,
            )
            self._last_refill = now

            if self._token_bucket < 1.0:
                wait = (1.0 - self._token_bucket) / self._rate_limit
                logger.debug("Kalshi rate-limit wait", wait_seconds=wait)
                await asyncio.sleep(wait)
                self._token_bucket = 0.0
                self._last_refill = time.monotonic()
            else:
                self._token_bucket -= 1.0

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Perform a rate-limited GET request and return the JSON body."""
        await self._rate_limit_wait()
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        response = await client.get(url, params=params)
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


# Singleton instance
kalshi_client = KalshiClient()
