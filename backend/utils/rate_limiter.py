import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger("rate_limiter")


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an API endpoint"""

    requests_per_window: int
    window_seconds: float = 10.0
    burst_limit: Optional[int] = None  # Max burst if different from rate


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""

    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.monotonic)

    def refill(self):
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens, returns True if successful"""
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate how long to wait for tokens to be available"""
        self.refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """Rate limiter using token bucket algorithm + in-flight concurrency cap.

    The token bucket handles RATE (requests per second).  A separate
    global semaphore caps CONCURRENCY (simultaneous in-flight requests).

    Without the concurrency cap, fan-out callers like
    ``polymarket.get_events_by_slugs`` (per-call Semaphore(12)) and
    ``trader_orchestrator.live_market_context.get_prices_history``
    (called per-signal across N traders) can stack 30+ simultaneous
    HTTP requests on the event loop.  Each httpx request keeps a
    socket + asyncio task + thread-pool slot live; at peak the
    "can't allocate lock" OS error fires (observed in the backtest
    meltdown log at 02:47:30).

    The semaphore is created lazily on first acquire so module
    imports happen before the asyncio loop exists.
    """

    # Polymarket API rate limits (from docs.polymarket.com)
    LIMITS = {
        "gamma_general": RateLimitConfig(requests_per_window=4000, window_seconds=10),
        "gamma_markets": RateLimitConfig(requests_per_window=300, window_seconds=10),
        "gamma_events": RateLimitConfig(requests_per_window=500, window_seconds=10),
        "gamma_search": RateLimitConfig(requests_per_window=350, window_seconds=10),
        "clob_general": RateLimitConfig(requests_per_window=9000, window_seconds=10),
        "clob_market": RateLimitConfig(requests_per_window=1500, window_seconds=10),
        "clob_markets_batch": RateLimitConfig(requests_per_window=500, window_seconds=10),
        "clob_prices_history": RateLimitConfig(requests_per_window=1000, window_seconds=10),
        "data_general": RateLimitConfig(requests_per_window=1000, window_seconds=10),
        "data_trades": RateLimitConfig(requests_per_window=200, window_seconds=10),
        "data_positions": RateLimitConfig(requests_per_window=60, window_seconds=10),
    }

    # Cap on simultaneous in-flight Polymarket HTTP requests across all
    # callers and endpoints.  24 leaves comfortable headroom under
    # every individual endpoint's per-second budget while bounding
    # the asyncio task fan-out the event-loop watchdog observed.
    GLOBAL_INFLIGHT_LIMIT = 24

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._inflight_semaphore: Optional[asyncio.Semaphore] = None

    def _get_bucket(self, endpoint: str) -> TokenBucket:
        """Get or create a token bucket for an endpoint"""
        if endpoint not in self._buckets:
            config = self.LIMITS.get(endpoint, RateLimitConfig(1000, 10))
            capacity = config.burst_limit or config.requests_per_window
            refill_rate = config.requests_per_window / config.window_seconds
            self._buckets[endpoint] = TokenBucket(capacity=capacity, tokens=capacity, refill_rate=refill_rate)
        return self._buckets[endpoint]

    def _get_lock(self, endpoint: str) -> asyncio.Lock:
        """Get or create a lock for an endpoint"""
        if endpoint not in self._locks:
            self._locks[endpoint] = asyncio.Lock()
        return self._locks[endpoint]

    def _get_inflight_semaphore(self) -> asyncio.Semaphore:
        """Lazy global in-flight semaphore (binds to the active loop)."""
        if self._inflight_semaphore is None:
            self._inflight_semaphore = asyncio.Semaphore(self.GLOBAL_INFLIGHT_LIMIT)
        return self._inflight_semaphore

    def inflight_slot(self):
        """Return an awaitable context manager that holds an in-flight
        slot for the duration of the actual HTTP request.  Callers wrap
        the network leg with ``async with rate_limiter.inflight_slot():``
        so the slot is held only across the wire, not across cache
        lookups or post-processing."""
        return self._get_inflight_semaphore()

    async def acquire(self, endpoint: str, tokens: int = 1) -> float:
        """
        Acquire rate limit permission. Returns wait time (0 if immediate).
        Blocks until permission is granted.
        """
        lock = self._get_lock(endpoint)
        async with lock:
            bucket = self._get_bucket(endpoint)
            wait_time = bucket.wait_time(tokens)

            if wait_time > 0:
                logger.debug("Rate limit wait", endpoint=endpoint, wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                bucket.refill()

            bucket.consume(tokens)
            return wait_time

    def check(self, endpoint: str, tokens: int = 1) -> bool:
        """Check if a request would be allowed without consuming"""
        bucket = self._get_bucket(endpoint)
        bucket.refill()
        return bucket.tokens >= tokens

    def get_status(self) -> Dict[str, dict]:
        """Get current rate limit status for all endpoints"""
        status = {}
        for endpoint, bucket in self._buckets.items():
            bucket.refill()
            config = self.LIMITS.get(endpoint)
            status[endpoint] = {
                "available_tokens": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
                "limit": f"{config.requests_per_window}/{config.window_seconds}s" if config else "default",
            }
        return status


# Global rate limiter instance
rate_limiter = RateLimiter()


def endpoint_for_url(url: str) -> str:
    """Determine rate limit endpoint category from URL"""
    if "gamma-api" in url:
        if "/markets" in url:
            return "gamma_markets"
        if "/events" in url:
            return "gamma_events"
        if "/search" in url:
            return "gamma_search"
        return "gamma_general"
    elif "clob" in url:
        if "/prices-history" in url:
            return "clob_prices_history"
        # Batch endpoints (/books, /prices, /midprices) have a lower limit
        if "/books" in url or "/prices" in url or "/midprices" in url:
            return "clob_markets_batch"
        if "/book" in url or "/price" in url or "/midpoint" in url:
            return "clob_market"
        return "clob_general"
    elif "data-api" in url:
        if "/trades" in url:
            return "data_trades"
        if "positions" in url:  # matches /positions AND /closed-positions
            return "data_positions"
        return "data_general"
    return "default"
