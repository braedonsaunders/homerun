import httpx
import asyncio
import random
import json
import re
import time
from typing import Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from utils.utcnow import utcnow, utcfromtimestamp

from config import settings
from models import Market, Event
from utils.converters import coerce_bool as _coerce_bool
from utils.rate_limiter import rate_limiter, endpoint_for_url
from utils.logger import get_logger

_logger = get_logger("polymarket")

# Retry settings for rate-limited requests
_MAX_RETRIES = 4
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0

# 2026-05-05: per-endpoint cooldown applied AFTER the inner retries
# exhausted. The retry loop already handles transient 429s with
# exponential backoff; this is the safety net for sustained throttling
# where multiple consecutive callers each burn _MAX_RETRIES attempts
# against the same already-throttled endpoint. Once cooldown is set,
# subsequent same-endpoint calls short-circuit until expiry. Cooldown
# escalates per consecutive trip to MAX so a heavily-throttled endpoint
# doesn't get hit again every 30s.
_DATA_API_ENDPOINT_COOLDOWN_BASE_SECONDS = 30.0
_DATA_API_ENDPOINT_COOLDOWN_MAX_SECONDS = 300.0
_DATA_POSITIONS_PAGE_SIZE = 50
_CLOSED_POSITIONS_MAX_FETCH = 300
_WALLET_PNL_CLOSED_POSITIONS_MAX_FETCH = 250
_WALLET_POSITIONS_MAX_PAGES = 4
_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")
_ETH_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


# Sentinel for "key not in cache" so legitimate None / empty-list values
# can still be cached.
_CACHE_MISS = object()


class _TTLCache:
    """Single-flight TTL cache.

    Multiple concurrent callers for the same key share ONE in-flight
    fetch (no thundering-herd cache stampede).  Subsequent callers
    within the TTL window get the cached value without re-fetching.

    The cache is intentionally simple — single asyncio loop, no LRU
    eviction.  Callers must keep the keyspace bounded (typically
    ``(wallet_address, ...)`` so it's at most a handful per single-user
    install).

    Use cases:
      * ``get_wallet_trades_paginated``    — wallet-keyed, 60s TTL
      * ``get_closed_positions_paginated`` — wallet-keyed, 60s TTL
      * ``get_midpoint``                   — token-keyed, 3s TTL
    """

    __slots__ = ("_ttl", "_data", "_inflight", "_hits", "_misses")

    def __init__(self, ttl_seconds: float):
        self._ttl = float(ttl_seconds)
        self._data: dict = {}
        self._inflight: dict = {}
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int | float]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "size": len(self._data),
            "inflight": len(self._inflight),
            "ttl_seconds": self._ttl,
        }

    def peek(self, key):
        entry = self._data.get(key)
        if entry is None:
            return _CACHE_MISS
        ts, value = entry
        if time.monotonic() - ts >= self._ttl:
            return _CACHE_MISS
        return value

    def invalidate(self, key=None) -> None:
        if key is None:
            self._data.clear()
        else:
            self._data.pop(key, None)

    async def get_or_fetch(self, key, fetcher):
        # Fast path: live cache entry.
        cached = self.peek(key)
        if cached is not _CACHE_MISS:
            self._hits += 1
            return cached
        # Coalesce concurrent misses for the same key onto a single
        # in-flight fetch.  This is the load-bearing property — without
        # it, N traders waking up together each fire their own
        # ``get_wallet_trades_paginated`` and we lose all caching benefit.
        existing = self._inflight.get(key)
        if existing is not None:
            self._hits += 1
            return await existing
        self._misses += 1
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        # Pre-attach a no-op "consumer" so a cancelled leader with no
        # actual waiters doesn't leak "Future exception was never
        # retrieved" — asyncio considers the exception consumed once
        # any callback (incl. lambda) reads it.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self._inflight[key] = future
        try:
            result = await fetcher()
            self._data[key] = (time.monotonic(), result)
            if not future.done():
                future.set_result(result)
            return result
        except asyncio.CancelledError:
            # Leader was cancelled mid-fetch.  Cancel the future so any
            # coalesced waiters see CancelledError (proper async
            # semantics) rather than getting a set_exception(CancelledError)
            # which leaks as "Future exception was never retrieved".
            if not future.done():
                future.cancel()
            raise
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)


class PolymarketClient:
    """Client for interacting with Polymarket APIs"""

    def __init__(self):
        self.gamma_url = settings.GAMMA_API_URL
        self.clob_url = settings.CLOB_API_URL
        self.data_url = settings.DATA_API_URL
        self._client: Optional[httpx.AsyncClient] = None
        self._trading_client: Optional[httpx.AsyncClient] = None  # Proxy-aware for trading
        self._market_cache: dict[str, dict] = {}  # condition_id -> {question, slug}
        # 35K cap accommodates ~14K active markets × ~2-3 keys
        # (condition_id, normalized condition_id, ``token:<token_id>``)
        # under MARKET_UNIVERSE_TRADABLE_ONLY=True, with headroom for
        # transient lookup churn.  Was 20K but trimming was too lazy
        # in practice; coupled with the duplicate-seeding bug, the
        # dict had grown to 65K entries by the time of memory_dump #2.
        self._market_cache_max_size: int = 35_000
        self._market_lookup_cooldown_until: dict[str, float] = {}
        self._username_cache: dict[str, str] = {}  # address (lowercase) -> username
        self._username_cache_max_size: int = 10_000
        self._persistent_cache = None  # Lazy-loaded MarketCacheService
        self._closed_positions_warning_cooldown_until: float = 0.0
        self._network_error_last_log_at: float = 0.0
        self._network_error_log_interval_seconds: float = 10.0
        # 2026-05-05: per-endpoint hard cooldowns triggered by sustained
        # 429s (the rate-limiter retries exhausted). When the data-api
        # rejects us with 429 after every retry, hammering it again on
        # the next caller's request just deepens the throttle. Stamp a
        # cooldown deadline that ``_rate_limited_get`` honors by short-
        # circuiting the call. Per-endpoint (``data_positions`` etc.) so
        # a 429 on one endpoint doesn't pause unrelated traffic.
        # ``data_positions`` (closed-positions) was the observed offender
        # — fan-out copy-trade signal processor across many wallets
        # produced bursts that exceeded the documented 6 req/s and
        # cascaded into multi-minute throttling. The fan-out stops
        # immediately when the cooldown is set; the in-process
        # _closed_positions_cache (60s TTL) absorbs cache misses during
        # the cooldown.
        self._endpoint_cooldown_until: dict[str, float] = {}
        self._endpoint_cooldown_attempts: dict[str, int] = {}

        # Single-flight TTL caches for the data-API endpoints that the
        # rate limiter throttles hardest (data_trades=20 req/s,
        # data_positions=6 req/s).  In a single-user install the wallet
        # is the same across every trader, so these calls are massively
        # redundant.  60s of staleness on wallet trades / closed
        # positions is acceptable — the verifier runs every 5 minutes
        # and is happy to see data ≤ 60s old.
        self._wallet_trades_cache = _TTLCache(ttl_seconds=60.0)
        self._closed_positions_cache = _TTLCache(ttl_seconds=60.0)
        # Midpoints change continuously, but at the cycle frequency a
        # 3-second cache is plenty for stop-loss / take-profit logic
        # (reconcile cycles run every 30+ seconds).  Cuts ``clob_market``
        # rate-limit pressure dramatically when many traders share the
        # same token universe.
        self._midpoint_cache = _TTLCache(ttl_seconds=3.0)

    def invalidate_wallet_caches(self, address: Optional[str] = None) -> None:
        """Bust wallet_trades + closed_positions caches for *address*
        (or all wallets if None).

        Call this after any write that affects wallet state (e.g. order
        submission with a confirmed fill) so the next verifier cycle
        sees fresh data instead of waiting for the 60s TTL to expire.
        Optional: 60s of staleness is acceptable for the verifier path,
        so callers that don't care can skip this entirely.
        """
        if address is None:
            self._wallet_trades_cache.invalidate()
            self._closed_positions_cache.invalidate()
            return
        normalized = str(address or "").strip().lower()
        if not normalized:
            return
        # Cache keys include max_trades / max_positions, so we must
        # walk and drop every entry whose first key element matches.
        for cache in (self._wallet_trades_cache, self._closed_positions_cache):
            stale_keys = [k for k in cache._data if k and k[0] == normalized]
            for k in stale_keys:
                cache.invalidate(k)

    def cache_stats(self) -> dict[str, dict]:
        """Expose cache hit-rate / size / TTL for diagnostics."""
        return {
            "wallet_trades": self._wallet_trades_cache.stats(),
            "closed_positions": self._closed_positions_cache.stats(),
            "midpoint": self._midpoint_cache.stats(),
        }

    async def _get_persistent_cache(self):
        """Lazy-load the persistent market cache service.

        ``MarketCacheService`` is the canonical store (~14K active
        markets after MARKET_UNIVERSE_TRADABLE_ONLY filter).  Reads
        fall through to it via :meth:`_lookup_persistent_market_info`
        on local miss; we deliberately do NOT bulk-seed our own dict
        with its contents — that was the 65K duplicate-cache leak
        observed in memory_dump #2 (every call to this method was
        reseeding the local map, and the LRU trim never caught up).
        """
        if self._persistent_cache is None:
            try:
                from services.market_cache import market_cache_service

                if not market_cache_service._loaded:
                    market_cache_service.start_background_load()
                self._persistent_cache = market_cache_service
            except Exception:
                pass  # Graceful degradation: in-memory only
        return self._persistent_cache

    async def _lookup_persistent_market_info(self, condition_id: str) -> Optional[dict]:
        """Read-through to MarketCacheService keyed by condition_id."""
        if not condition_id:
            return None
        cache = await self._get_persistent_cache()
        if cache is None:
            return None
        try:
            return await cache.get_market(condition_id)
        except Exception:
            return None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _get_trading_client(self) -> httpx.AsyncClient:
        """Get a proxy-aware client for trading-related CLOB calls.

        Falls back to the standard client if proxy is not configured.
        Reads proxy state from the DB-backed cached config.
        """
        from services.trading_proxy import _get_config, get_async_proxy_client

        if not _get_config().enabled:
            return await self._get_client()

        if self._trading_client is None or self._trading_client.is_closed:
            self._trading_client = get_async_proxy_client()
        return self._trading_client

    async def _rate_limited_get(self, url: str, client: Optional[httpx.AsyncClient] = None, **kwargs) -> httpx.Response:
        """GET request with rate limiting and retry on 429/5xx errors.

        Acquires a token from the global rate limiter before each attempt,
        then retries with exponential backoff when the server returns 429
        or a transient 5xx error.
        """
        client_kind = "external"
        if client is None:
            client = await self._get_client()
            client_kind = "default"
        elif client is self._trading_client:
            client_kind = "trading"
        elif client is self._client:
            client_kind = "default"

        endpoint = endpoint_for_url(url)
        last_response: Optional[httpx.Response] = None

        # 2026-05-05: short-circuit when the endpoint is in 429 cooldown.
        # Returns a synthetic 429 response so callers' status-code
        # branching ("if response.status_code == 429" etc.) keeps
        # working. We avoid the network entirely during cooldown so
        # heavy fan-out callers can't deepen the throttle.
        cooldown_remaining = self._endpoint_cooldown_remaining(endpoint)
        if cooldown_remaining > 0:
            now = time.monotonic()
            # Suppressed-rate logging (every 30s per endpoint) so we don't
            # flood the operator console while cooled down.
            log_key = f"_endpoint_cooldown_log_{endpoint}"
            last_log_at = getattr(self, log_key, 0.0)
            if (now - last_log_at) >= 30.0:
                setattr(self, log_key, now)
                _logger.info(
                    "Polymarket data-api endpoint in 429 cooldown; short-circuiting request",
                    endpoint=endpoint,
                    cooldown_remaining_seconds=round(cooldown_remaining, 1),
                )
            return httpx.Response(
                429,
                headers={"Retry-After": str(int(cooldown_remaining) + 1)},
                request=httpx.Request("GET", url),
            )

        for attempt in range(_MAX_RETRIES):
            await rate_limiter.acquire(endpoint)

            try:
                # Hold a global in-flight slot for the wire portion of the
                # request only.  The token bucket above caps RATE; this
                # caps CONCURRENCY.  Without it, fan-out callers (e.g.
                # get_events_by_slugs's per-call Semaphore(12), or
                # per-trader-cycle prices_history loops) can stack 30+
                # simultaneous HTTP requests, exhausting OS thread/lock
                # resources (the "can't allocate lock" error observed in
                # the backtest meltdown was the smoking gun).
                async with rate_limiter.endpoint_inflight_slot(endpoint):
                    async with rate_limiter.inflight_slot():
                        response = await client.get(url, **kwargs)
                        # Read the body inside the slot so chunked/stream read
                        # failures are retried instead of escaping at
                        # response.json(), AND so the connection isn't held
                        # past the slot release.
                        await response.aread()
            except Exception as exc:
                if not isinstance(exc, httpx.TransportError) and not self._is_retryable_closed_client_error(exc):
                    raise
                closed_client_error = self._is_retryable_closed_client_error(exc)
                await self._reset_failed_client(client)
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                    delay *= 0.5 + random.random()
                    if closed_client_error:
                        _logger.debug(
                            "Closed client reset, retrying",
                            url=url,
                            attempt=attempt + 1,
                        )
                    else:
                        now = time.monotonic()
                        if (now - self._network_error_last_log_at) >= self._network_error_log_interval_seconds:
                            self._network_error_last_log_at = now
                            _logger.info(
                                "Network error, retrying",
                                url=url,
                                attempt=attempt + 1,
                                delay=round(delay, 2),
                                error=str(exc),
                            )
                        else:
                            _logger.debug(
                                "Network error retry suppressed",
                                url=url,
                                attempt=attempt + 1,
                            )
                    if client_kind == "trading":
                        client = await self._get_trading_client()
                    elif client_kind == "default":
                        client = await self._get_client()
                    await asyncio.sleep(delay)
                    continue
                raise

            if response.status_code == 429 or response.status_code >= 500:
                last_response = response
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                    delay *= 0.5 + random.random()
                    # Respect Retry-After header if present
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    _logger.debug(
                        "Rate limited, retrying",
                        url=url,
                        status=response.status_code,
                        attempt=attempt + 1,
                        delay=round(delay, 2),
                    )
                    await asyncio.sleep(delay)
                    continue
                # Final attempt exhausted — return the response as-is so
                # callers that check status_code still work correctly.
                # 2026-05-05: also stamp an endpoint-level cooldown so
                # the next same-endpoint request short-circuits instead
                # of repeating the full retry storm. This is the
                # mitigation for the cascade where multiple wallets'
                # ``get_closed_positions`` calls each burned 4 retries
                # against an already-throttled endpoint.
                if response.status_code == 429:
                    self._stamp_endpoint_cooldown(endpoint, response=response)
                return response

            # 2026-05-05: clear the escalation counter on success — next
            # 429 should restart from the base cooldown, not the escalated
            # window from a long-past throttle event.
            self._clear_endpoint_cooldown(endpoint)
            return response

        # Should not be reached, but just in case
        return last_response  # type: ignore[return-value]

    def _endpoint_cooldown_remaining(self, endpoint: str) -> float:
        """Return seconds remaining on the endpoint's 429 cooldown, or 0
        if no cooldown is active (or it just expired).

        Lazily expires entries: an expired endpoint clears its own state
        on next read so the next 429 starts fresh.
        """
        if not endpoint:
            return 0.0
        deadline = self._endpoint_cooldown_until.get(endpoint)
        if deadline is None:
            return 0.0
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            self._endpoint_cooldown_until.pop(endpoint, None)
            return 0.0
        return remaining

    def _stamp_endpoint_cooldown(
        self,
        endpoint: str,
        *,
        response: Optional[httpx.Response] = None,
    ) -> float:
        """Mark ``endpoint`` cooled-down after sustained 429s. Returns
        the cooldown duration applied (for logging).

        Honors a ``Retry-After`` header on the response as a floor of
        the cooldown window. Escalates exponentially per consecutive
        trip up to ``_DATA_API_ENDPOINT_COOLDOWN_MAX_SECONDS`` so a
        provider in deep throttle isn't pinged again every 30s.
        """
        if not endpoint:
            return 0.0
        attempts = self._endpoint_cooldown_attempts.get(endpoint, 0) + 1
        self._endpoint_cooldown_attempts[endpoint] = attempts
        backoff = min(
            _DATA_API_ENDPOINT_COOLDOWN_BASE_SECONDS * (2 ** max(0, attempts - 1)),
            _DATA_API_ENDPOINT_COOLDOWN_MAX_SECONDS,
        )
        if response is not None:
            try:
                retry_after_header = response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after_seconds = float(retry_after_header)
                        backoff = max(backoff, retry_after_seconds)
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass
        backoff = min(backoff, _DATA_API_ENDPOINT_COOLDOWN_MAX_SECONDS)
        self._endpoint_cooldown_until[endpoint] = time.monotonic() + backoff
        _logger.warning(
            "Polymarket data-api 429 — applying endpoint cooldown",
            endpoint=endpoint,
            cooldown_seconds=round(backoff, 1),
            consecutive_trips=attempts,
        )
        return backoff

    def _clear_endpoint_cooldown(self, endpoint: str) -> None:
        """Reset the cooldown deadline AND escalation counter on success.

        Called after any non-429 successful request through ``endpoint``
        so the next 429 starts from the base window. Without this, an
        endpoint that recovers and is later re-throttled would jump
        straight to a multi-minute cooldown.
        """
        if not endpoint:
            return
        self._endpoint_cooldown_until.pop(endpoint, None)
        self._endpoint_cooldown_attempts.pop(endpoint, None)

    @staticmethod
    def _is_retryable_closed_client_error(exc: Exception) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        return "client has been closed" in message and "cannot send a request" in message

    async def _reset_failed_client(self, client: Optional[httpx.AsyncClient]) -> None:
        if client is None:
            return
        if client is self._client:
            self._client = None
            try:
                await client.aclose()
            except Exception:
                pass
            return
        if client is self._trading_client:
            self._trading_client = None
            try:
                await client.aclose()
            except Exception:
                pass

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._trading_client and not self._trading_client.is_closed:
            await self._trading_client.aclose()

    # ==================== GAMMA API ====================

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 200,
        offset: int = 0,
        order: str = "",
        ascending: bool = False,
    ) -> list[Market]:
        """Fetch markets from Gamma API"""
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        if order:
            params["order"] = order
            params["ascending"] = str(ascending).lower()

        response = await self._rate_limited_get(f"{self.gamma_url}/markets", params=params)
        response.raise_for_status()
        data = response.json()

        return [Market.from_gamma_response(m) for m in data]

    async def _get_markets_keyset_page(
        self,
        *,
        active: bool,
        closed: bool,
        limit: int,
        after_cursor: Optional[str],
        order: str,
        ascending: bool,
    ) -> tuple[list[Market], Optional[str]]:
        """Fetch one page of markets via the cursor-based /markets/keyset endpoint.

        Returns (markets, next_cursor). next_cursor is None when the server
        signals end of pagination.
        """
        params: dict = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
        }
        if after_cursor:
            params["after_cursor"] = after_cursor
        if order:
            params["order"] = order
            params["ascending"] = str(ascending).lower()

        response = await self._rate_limited_get(f"{self.gamma_url}/markets/keyset", params=params)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("markets", []) if isinstance(payload, dict) else []
        next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
        return [Market.from_gamma_response(m) for m in items], next_cursor or None

    async def get_all_markets(self, active: bool = True) -> list[Market]:
        """Fetch all markets with cursor-based pagination.

        Uses the /markets/keyset endpoint (added Apr 10 2026; offset-based
        /markets is deprecated). Sorts by volume descending by default so the
        highest-value markets are fetched first if the cap is reached.
        """
        all_markets: list[Market] = []
        page_size = max(50, min(500, settings.MARKET_FETCH_PAGE_SIZE))
        order = settings.MARKET_FETCH_ORDER or ""
        cap = settings.MAX_MARKETS_TO_SCAN
        after_cursor: Optional[str] = None

        while True:
            markets, next_cursor = await self._get_markets_keyset_page(
                active=active,
                closed=False,
                limit=page_size,
                after_cursor=after_cursor,
                order=order,
                ascending=False,
            )
            if not markets:
                break

            all_markets.extend(markets)

            if cap > 0 and len(all_markets) >= cap:
                break

            if not next_cursor:
                break
            after_cursor = next_cursor

        return all_markets

    async def get_recent_markets(
        self,
        since_minutes: int = 10,
        active: bool = True,
    ) -> list[Market]:
        """Fetch only recently updated markets (incremental delta fetch).

        Queries for markets ordered by updatedAt descending via /markets/keyset
        and stops once it hits markets older than `since_minutes`. Also picks
        up markets that had price changes (not just newly created ones).
        """
        all_markets: list[Market] = []
        page_size = max(50, min(500, settings.MARKET_FETCH_PAGE_SIZE))
        cutoff = utcnow() - timedelta(minutes=since_minutes)
        after_cursor: Optional[str] = None

        while True:
            try:
                params: dict = {
                    "active": str(active).lower(),
                    "closed": "false",
                    "limit": page_size,
                    "order": "updatedAt",
                    "ascending": "false",
                }
                if after_cursor:
                    params["after_cursor"] = after_cursor
                response = await self._rate_limited_get(f"{self.gamma_url}/markets/keyset", params=params)
                response.raise_for_status()
                payload = response.json()
                data = payload.get("markets", []) if isinstance(payload, dict) else []
                next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None

                if not data:
                    break

                page_markets = [Market.from_gamma_response(m) for m in data]
                all_markets.extend(page_markets)

                # Check if the oldest market in this page is older than cutoff.
                oldest_in_page = data[-1]
                updated_at_raw = (
                    oldest_in_page.get("updatedAt")
                    or oldest_in_page.get("updated_at")
                    or oldest_in_page.get("createdAt")
                    or oldest_in_page.get("created_at")
                )
                if updated_at_raw:
                    try:
                        updated_at = PolymarketClient._coerce_datetime(updated_at_raw)
                        if updated_at is not None and updated_at < cutoff:
                            break
                    except (ValueError, TypeError, OSError):
                        pass

                # Safety cap for incremental mode
                if len(all_markets) >= 500:
                    break

                if not next_cursor:
                    break
                after_cursor = next_cursor

            except Exception as e:
                _logger.warning("Incremental market fetch failed", error=str(e))
                break

        return all_markets

    async def get_events(self, closed: bool = False, limit: int = 100, offset: int = 0) -> list[Event]:
        """Fetch events from Gamma API (events contain grouped markets)"""
        params = {"closed": str(closed).lower(), "limit": limit, "offset": offset}

        response = await self._rate_limited_get(f"{self.gamma_url}/events", params=params)
        response.raise_for_status()
        data = response.json()

        return [Event.from_gamma_response(e) for e in data]

    async def _get_events_keyset_page(
        self,
        *,
        closed: bool,
        limit: int,
        after_cursor: Optional[str],
    ) -> tuple[list[Event], Optional[str]]:
        """Fetch one page of events via the cursor-based /events/keyset endpoint."""
        params: dict = {"closed": str(closed).lower(), "limit": limit}
        if after_cursor:
            params["after_cursor"] = after_cursor

        response = await self._rate_limited_get(f"{self.gamma_url}/events/keyset", params=params)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("events", []) if isinstance(payload, dict) else []
        next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
        return [Event.from_gamma_response(e) for e in items], next_cursor or None

    async def get_all_events(self, closed: bool = False) -> list[Event]:
        """Fetch all events with cursor-based pagination.

        Uses the /events/keyset endpoint (added Apr 10 2026; offset-based
        /events is deprecated).
        """
        all_events: list[Event] = []
        page_size = max(50, min(500, settings.MARKET_FETCH_PAGE_SIZE))
        cap = settings.MAX_EVENTS_TO_SCAN
        after_cursor: Optional[str] = None

        while True:
            events, next_cursor = await self._get_events_keyset_page(
                closed=closed,
                limit=page_size,
                after_cursor=after_cursor,
            )
            if not events:
                break

            all_events.extend(events)

            if cap > 0 and len(all_events) >= cap:
                break

            if not next_cursor:
                break
            after_cursor = next_cursor

        return all_events

    async def search_events(self, query: str, limit: int = 20, closed: bool = False) -> list[Event]:
        """Search events on Polymarket by keyword using Gamma API text search.

        Uses the Strapi ``_q`` full-text search parameter for fast, broad
        matching, supplemented by a ``slug_contains`` lookup.  Both requests
        run concurrently and results are merged/deduplicated.
        """
        base_params = {
            "closed": str(closed).lower(),
            "limit": limit,
            "active": "true",
        }

        # Normalise query for slug-based search (lowercase, hyphenated)
        slug_query = query.lower().replace(" ", "-")

        async def _fetch(extra_params: dict) -> list[dict]:
            try:
                resp = await self._rate_limited_get(
                    f"{self.gamma_url}/events",
                    params={**base_params, **extra_params},
                )
                return resp.json() if resp.status_code == 200 else []
            except Exception:
                return []

        # Use Strapi's _q full-text search AND slug_contains concurrently
        text_results, slug_results = await asyncio.gather(
            _fetch({"_q": query}),
            _fetch({"slug_contains": slug_query}),
        )

        # Merge and deduplicate
        seen_ids: set[str] = set()
        combined: list[dict] = []
        for item in text_results + slug_results:
            eid = str(item.get("id", ""))
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                combined.append(item)

        return [Event.from_gamma_response(e) for e in combined[:limit]]

    async def search_markets(self, query: str, limit: int = 50, closed: bool = False) -> list[Market]:
        """Search markets directly by keyword using Gamma API full-text search.

        Faster than searching events when the caller only needs market-level
        data (e.g. for a quick search results page).
        """
        slug_query = query.lower().replace(" ", "-")

        async def _fetch(extra_params: dict) -> list[dict]:
            try:
                resp = await self._rate_limited_get(
                    f"{self.gamma_url}/markets",
                    params={
                        "closed": str(closed).lower(),
                        "active": "true",
                        "limit": limit,
                        **extra_params,
                    },
                )
                return resp.json() if resp.status_code == 200 else []
            except Exception:
                return []

        text_results, slug_results = await asyncio.gather(
            _fetch({"_q": query}),
            _fetch({"slug_contains": slug_query}),
        )

        seen_ids: set[str] = set()
        combined: list[dict] = []
        for item in text_results + slug_results:
            mid = str(item.get("id", "") or item.get("condition_id", ""))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                combined.append(item)

        return [Market.from_gamma_response(m) for m in combined[:limit]]

    async def get_event_by_slug(self, slug: str) -> Optional[Event]:
        """Get a specific event by slug"""
        response = await self._rate_limited_get(f"{self.gamma_url}/events", params={"slug": slug})
        response.raise_for_status()
        data = response.json()

        if data:
            return Event.from_gamma_response(data[0])
        return None

    async def get_events_by_slugs(self, slugs: list[str], closed: bool = False) -> list[Event]:
        """Fetch events by slug with bounded concurrency for incremental catalog sync."""
        seen: set[str] = set()
        normalized_slugs: list[str] = []
        for raw_slug in slugs:
            slug = str(raw_slug or "").strip()
            if not slug or slug in seen:
                continue
            seen.add(slug)
            normalized_slugs.append(slug)
        if not normalized_slugs:
            return []

        events: list[Event] = []
        queue: asyncio.Queue[str] = asyncio.Queue()
        for slug in normalized_slugs:
            queue.put_nowait(slug)

        async def _fetch_worker() -> None:
            while True:
                try:
                    slug = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    response = await self._rate_limited_get(
                        f"{self.gamma_url}/events",
                        params={"slug": slug, "closed": str(bool(closed)).lower(), "limit": 1},
                    )
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, list) and data:
                        events.append(Event.from_gamma_response(data[0]))
                except Exception:
                    pass
                finally:
                    queue.task_done()
                    await asyncio.sleep(0)

        workers = [
            asyncio.create_task(_fetch_worker(), name=f"polymarket-events-by-slug-{i}")
            for i in range(min(4, len(normalized_slugs)))
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
        return events

    async def _evict_market_cache_entry(self, *keys: str):
        """Evict market metadata keys from memory and persistent SQL cache."""
        cache = await self._get_persistent_cache()
        for key in keys:
            norm = self._normalize_identifier(key)
            if not norm:
                continue
            self._market_cache.pop(key, None)
            self._market_cache.pop(norm, None)
            if cache:
                try:
                    await cache.delete_market(norm)
                except Exception:
                    pass

    def _trim_caches(self) -> None:
        """Evict oldest entries when in-memory caches exceed their size caps."""
        if len(self._market_cache) > self._market_cache_max_size:
            # Keep the most recently added half -- dict preserves insertion order
            excess = len(self._market_cache) - self._market_cache_max_size // 2
            keys_to_drop = list(self._market_cache.keys())[:excess]
            for k in keys_to_drop:
                del self._market_cache[k]
        if len(self._username_cache) > self._username_cache_max_size:
            excess = len(self._username_cache) - self._username_cache_max_size // 2
            keys_to_drop = list(self._username_cache.keys())[:excess]
            for k in keys_to_drop:
                del self._username_cache[k]
        # Prune expired cooldown entries to prevent unbounded growth.
        # Each unique condition_id/token_id ever queried gets a cooldown
        # entry that is never removed, leaking memory over long runs.
        _cooldown_max = 5_000
        if len(self._market_lookup_cooldown_until) > _cooldown_max:
            import time as _time_mod
            _now_mono = _time_mod.monotonic()
            expired_keys = [
                k for k, v in self._market_lookup_cooldown_until.items()
                if v < _now_mono
            ]
            for k in expired_keys:
                del self._market_lookup_cooldown_until[k]
            # If still over limit after removing expired, drop oldest half
            if len(self._market_lookup_cooldown_until) > _cooldown_max:
                excess = len(self._market_lookup_cooldown_until) - _cooldown_max // 2
                keys_to_drop = list(self._market_lookup_cooldown_until.keys())[:excess]
                for k in keys_to_drop:
                    del self._market_lookup_cooldown_until[k]

    async def get_market_by_condition_id(self, condition_id: str, **kwargs) -> Optional[dict]:
        """Look up a market by condition_id, using cache when available."""
        self._trim_caches()
        requested = self._normalize_identifier(condition_id)
        force_refresh = bool(kwargs.get("force_refresh", False))
        if not force_refresh:
            cached = self._market_cache.get(condition_id) or self._market_cache.get(requested)
            if cached is None:
                # Fall through to the persistent MarketCacheService rather
                # than re-fetching from the gamma API.  Avoids the
                # duplicate-cache leak fixed when we stopped bulk-seeding
                # this dict from MarketCacheService on every call.
                persistent = await self._lookup_persistent_market_info(requested)
                if persistent is not None:
                    cached = persistent
            if cached:
                cached_cid = self._normalize_identifier(cached.get("condition_id"))
                has_payload = bool(str(cached.get("question") or "").strip() or str(cached.get("slug") or "").strip())
                has_tradability = self._has_tradability_metadata(cached)
                has_outcome_context = self._has_outcome_context(cached)
                if cached_cid == requested and has_payload and has_tradability and has_outcome_context:
                    return cached
                # Drop stale/mismatched cache entries to prevent poisoning.
                await self._evict_market_cache_entry(condition_id, requested)

        try:
            now = time.monotonic()
            cooldown_until = float(self._market_lookup_cooldown_until.get(requested, 0.0) or 0.0)
            if cooldown_until > now:
                raise httpx.HTTPStatusError(
                    "market lookup gamma cooldown active",
                    request=httpx.Request("GET", f"{self.gamma_url}/markets"),
                    response=httpx.Response(429, request=httpx.Request("GET", f"{self.gamma_url}/markets")),
                )
            # Gamma expects plural ``condition_ids`` for direct condition lookups.
            for params in ({"condition_ids": condition_id, "limit": 80},):
                response = await self._rate_limited_get(
                    f"{self.gamma_url}/markets",
                    params=params,
                )
                response.raise_for_status()
                data = self._extract_list_payload(
                    response.json(),
                    preferred_keys=("data", "items"),
                )
                if not data:
                    continue

                market_data = next(
                    (row for row in data if self._market_matches_condition_id(row, requested)),
                    None,
                )
                if market_data is None:
                    continue

                info = self._extract_market_info(market_data)
                resolved_key = self._normalize_identifier(info.get("condition_id", condition_id)) or requested
                self._market_cache[resolved_key] = info
                self._market_cache[condition_id] = info

                # Write-through to persistent SQL cache
                cache = await self._get_persistent_cache()
                if cache:
                    try:
                        await cache.set_market(resolved_key, info)
                    except Exception:
                        pass  # Non-critical
                
                return info
        except Exception as e:
            cooldown_active = bool(
                isinstance(e, httpx.HTTPStatusError)
                and e.response is not None
                and e.response.status_code == 429
                and "cooldown active" in str(e).lower()
            )
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code == 429:
                self._market_lookup_cooldown_until[requested] = time.monotonic() + 60.0
            if cooldown_active:
                _logger.debug("Market lookup gamma cooldown active for %s", condition_id)
            else:
                _logger.warning("Market lookup failed for %s", condition_id, exc_info=e)

        # Fallback path: the Data API reliably accepts ``market=<condition_id>``
        # and returns trade rows with title/slug metadata.
        try:
            trades = await self.get_market_trades(condition_id, limit=20)
            info = self._extract_market_info_from_trades(
                requested_condition_id=requested,
                trades=trades,
            )
            if info:
                resolved_key = self._normalize_identifier(info.get("condition_id", condition_id)) or requested
                self._market_cache[resolved_key] = info
                self._market_cache[condition_id] = info

                for token_id in info.get("token_ids") or []:
                    norm_token = self._normalize_identifier(token_id)
                    if norm_token:
                        self._market_cache[f"token:{norm_token}"] = info

                cache = await self._get_persistent_cache()
                if cache:
                    try:
                        await cache.set_market(resolved_key, info)
                    except Exception:
                        pass

                return info
        except Exception as e:
            _logger.warning("Data API market lookup failed for %s", condition_id, exc_info=e)

        return None

    async def get_market_by_token_id(self, token_id: str, **kwargs) -> Optional[dict]:
        """Look up a market by CLOB token ID (asset_id), using cache when available."""
        self._trim_caches()
        requested = self._normalize_identifier(token_id)
        if not self._looks_like_token_id(requested):
            return None

        cache_key = f"token:{requested}"
        legacy_cache_key = f"token:{token_id}"
        force_refresh = bool(kwargs.get("force_refresh", False))
        if not force_refresh:
            cached = self._market_cache.get(cache_key) or self._market_cache.get(legacy_cache_key)
            if cached:
                token_ids = {
                    self._normalize_identifier(t)
                    for t in (cached.get("token_ids") or [])
                    if self._normalize_identifier(t)
                }
                has_payload = bool(str(cached.get("question") or "").strip() or str(cached.get("slug") or "").strip())
                has_tradability = self._has_tradability_metadata(cached)
                has_outcome_context = self._has_outcome_context(cached)
                if requested in token_ids and has_payload and has_tradability and has_outcome_context:
                    return cached
                cached_cid = self._normalize_identifier(cached.get("condition_id"))
                await self._evict_market_cache_entry(cache_key, legacy_cache_key, token_id, cached_cid)

        try:
            for params in (
                {"clob_token_ids": requested, "limit": 80},
                {"clobTokenIds": requested, "limit": 80},
            ):
                response = await self._rate_limited_get(
                    f"{self.gamma_url}/markets",
                    params=params,
                )
                response.raise_for_status()
                data = self._extract_list_payload(
                    response.json(),
                    preferred_keys=("data", "items"),
                )
                if not data:
                    continue

                market_data = next(
                    (row for row in data if self._market_matches_token_id(row, requested)),
                    None,
                )
                if market_data is None:
                    continue

                info = self._extract_market_info(market_data)
                self._market_cache[cache_key] = info

                # Also cache by condition_id if available.
                cid = self._normalize_identifier(info.get("condition_id"))
                if cid:
                    self._market_cache[cid] = info
                    cache = await self._get_persistent_cache()
                    if cache:
                        try:
                            await cache.set_market(cid, info)
                        except Exception:
                            pass

                return info
        except Exception as e:
            _logger.warning("Market lookup by token_id failed for %s", token_id, exc_info=e)

        return None

    @staticmethod
    def _extract_tags(data: dict) -> list[str]:
        """Extract normalised tag strings from a Gamma API market/event response."""
        tags: list[str] = []
        raw = data.get("tags", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    tags.append(item.strip())
                elif isinstance(item, dict):
                    value = item.get("label") or item.get("name")
                    if value:
                        tags.append(str(value).strip())
        elif isinstance(raw, str) and raw.strip():
            tags.append(raw.strip())
        return tags

    @staticmethod
    def _extract_category(data: dict) -> str:
        category_raw = data.get("category")
        if isinstance(category_raw, dict):
            category = str(
                category_raw.get("label")
                or category_raw.get("name")
                or category_raw.get("value")
                or ""
            ).strip()
        else:
            category = str(category_raw or "").strip()
        if category:
            return category

        events = data.get("events")
        if isinstance(events, list) and events:
            first = events[0]
            if isinstance(first, dict):
                event_category_raw = first.get("category")
                if isinstance(event_category_raw, dict):
                    event_category = str(
                        event_category_raw.get("label")
                        or event_category_raw.get("name")
                        or event_category_raw.get("value")
                        or ""
                    ).strip()
                else:
                    event_category = str(event_category_raw or "").strip()
                if event_category:
                    return event_category

        tags = PolymarketClient._extract_tags(data)
        if tags:
            return tags[0]
        return ""

    @staticmethod
    def _extract_market_info(market_data: dict) -> dict:
        """Extract standardized market info from a Gamma API market response."""
        token_ids = PolymarketClient._extract_token_ids_from_market(market_data)
        category = PolymarketClient._extract_category(market_data)
        outcomes_raw = market_data.get("outcomes")
        if isinstance(outcomes_raw, str):
            text = outcomes_raw.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                outcomes_raw = parsed if isinstance(parsed, list) else []
            elif text:
                outcomes_raw = [part.strip() for part in text.split(",") if part.strip()]
            else:
                outcomes_raw = []
        outcomes: list[str] = []
        if isinstance(outcomes_raw, list):
            for item in outcomes_raw:
                label = str(item or "").strip()
                if label:
                    outcomes.append(label)

        outcome_prices_raw = (
            market_data.get("outcomePrices")
            if market_data.get("outcomePrices") is not None
            else market_data.get("outcome_prices")
        )
        if isinstance(outcome_prices_raw, str):
            text = outcome_prices_raw.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                outcome_prices_raw = parsed if isinstance(parsed, list) else []
            elif text:
                outcome_prices_raw = [part.strip() for part in text.split(",") if part.strip()]
            else:
                outcome_prices_raw = []
        outcome_prices: list[float] = []
        if isinstance(outcome_prices_raw, list):
            for item in outcome_prices_raw:
                try:
                    outcome_prices.append(float(item))
                except Exception:
                    continue

        yes_price = outcome_prices[0] if len(outcome_prices) > 0 else None
        no_price = outcome_prices[1] if len(outcome_prices) > 1 else None

        events = market_data.get("events")
        event_slug = ""
        if isinstance(events, list) and events:
            first = events[0]
            if isinstance(first, dict):
                event_slug = str(first.get("slug") or "").strip()
        if not event_slug:
            event_slug = str(market_data.get("event_slug") or market_data.get("eventSlug") or "").strip()
        uma_resolution_status = (
            market_data.get("umaResolutionStatus")
            if market_data.get("umaResolutionStatus") is not None
            else market_data.get("uma_resolution_status")
        )
        uma_resolution_statuses = PolymarketClient._coerce_text_list(
            market_data.get("umaResolutionStatuses")
            if market_data.get("umaResolutionStatuses") is not None
            else market_data.get("uma_resolution_statuses")
        )

        return {
            "id": market_data.get("id", ""),
            "condition_id": market_data.get("condition_id", "") or market_data.get("conditionId", ""),
            "question": market_data.get("question", ""),
            "slug": market_data.get("slug", ""),
            "groupItemTitle": market_data.get("groupItemTitle", ""),
            "event_slug": event_slug,
            "token_ids": token_ids,
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,
            "yes_price": yes_price,
            "no_price": no_price,
            "active": market_data.get("active"),
            "closed": market_data.get("closed"),
            "archived": market_data.get("archived"),
            "accepting_orders": market_data.get("acceptingOrders")
            if market_data.get("acceptingOrders") is not None
            else market_data.get("accepting_orders"),
            "enable_order_book": market_data.get("enableOrderBook")
            if market_data.get("enableOrderBook") is not None
            else market_data.get("enable_order_book"),
            "resolved": market_data.get("resolved")
            if market_data.get("resolved") is not None
            else market_data.get("isResolved"),
            "end_date": market_data.get("endDate")
            if market_data.get("endDate") is not None
            else market_data.get("end_date"),
            "winner": market_data.get("winner"),
            "winning_outcome": market_data.get("winningOutcome")
            if market_data.get("winningOutcome") is not None
            else market_data.get("winning_outcome"),
            "status": market_data.get("status")
            if market_data.get("status") is not None
            else (
                market_data.get("marketStatus")
                if market_data.get("marketStatus") is not None
                else market_data.get("market_status")
            ),
            "uma_resolution_status": uma_resolution_status,
            "uma_resolution_statuses": uma_resolution_statuses,
            "liquidity": market_data.get("liquidity")
            if market_data.get("liquidity") is not None
            else market_data.get("liquidityNum"),
            "volume": market_data.get("volume")
            if market_data.get("volume") is not None
            else market_data.get("volumeNum"),
            "category": category or None,
            "tags": PolymarketClient._extract_tags(market_data),
        }

    @staticmethod
    def _extract_market_info_from_trades(
        *,
        requested_condition_id: str,
        trades: list[dict],
    ) -> Optional[dict]:
        """Extract market metadata from market trade rows."""
        if not trades:
            return None

        matching: list[dict] = []
        fallback: list[dict] = []

        for row in trades:
            if not isinstance(row, dict):
                continue

            raw_condition = row.get("conditionId") or row.get("condition_id")
            condition_id = PolymarketClient._normalize_identifier(raw_condition)
            if condition_id == requested_condition_id:
                matching.append(row)
            elif not condition_id:
                fallback.append(row)

        candidates = matching if matching else fallback
        if not candidates:
            return None

        token_ids: list[str] = []
        seen_tokens: set[str] = set()
        for row in candidates:
            for raw_token in (
                row.get("asset"),
                row.get("asset_id"),
                row.get("assetId"),
                row.get("token_id"),
                row.get("tokenId"),
            ):
                norm_token = PolymarketClient._normalize_identifier(raw_token)
                if norm_token and norm_token not in seen_tokens:
                    seen_tokens.add(norm_token)
                    token_ids.append(norm_token)

        sample = candidates[0]
        question = str(sample.get("title") or sample.get("market_title") or sample.get("question") or "").strip()
        slug = str(sample.get("slug") or sample.get("market_slug") or sample.get("marketSlug") or "").strip()
        event_slug = str(sample.get("eventSlug") or sample.get("event_slug") or "").strip()

        if not question and not slug:
            return None

        return {
            "id": str(sample.get("market") or sample.get("market_id") or ""),
            "condition_id": requested_condition_id,
            "question": question,
            "slug": slug,
            "groupItemTitle": question,
            "event_slug": event_slug,
            "token_ids": token_ids,
        }

    @staticmethod
    def _normalize_identifier(value: object) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _looks_like_condition_id(value: object) -> bool:
        normalized = PolymarketClient._normalize_identifier(value)
        return bool(_CONDITION_ID_RE.fullmatch(normalized))

    @staticmethod
    def _looks_like_token_id(value: object) -> bool:
        normalized = PolymarketClient._normalize_identifier(value)
        if not normalized or PolymarketClient._looks_like_condition_id(normalized):
            return False
        return bool(_NUMERIC_TOKEN_ID_RE.fullmatch(normalized) or _HEX_TOKEN_ID_RE.fullmatch(normalized))

    @staticmethod
    def _has_tradability_metadata(market_info: Optional[dict]) -> bool:
        if not isinstance(market_info, dict):
            return False
        return any(
            market_info.get(key) is not None
            for key in (
                "closed",
                "active",
                "archived",
                "accepting_orders",
                "enable_order_book",
                "resolved",
                "end_date",
                "winner",
                "winning_outcome",
                "status",
                "uma_resolution_status",
                "uma_resolution_statuses",
            )
        )

    @staticmethod
    def _has_outcome_context(market_info: Optional[dict]) -> bool:
        if not isinstance(market_info, dict):
            return False
        outcomes = market_info.get("outcomes")
        if isinstance(outcomes, list) and len(outcomes) >= 2:
            return True
        outcome_prices = market_info.get("outcome_prices")
        if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
            return True
        yes_price = market_info.get("yes_price")
        no_price = market_info.get("no_price")
        return yes_price is not None and no_price is not None

    @staticmethod
    def _coerce_datetime(value: object) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 10_000_000_000:
                ts /= 1000.0
            try:
                return utcfromtimestamp(ts)
            except (OSError, OverflowError, ValueError):
                return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                numeric = float(text)
                if numeric > 10_000_000_000:
                    numeric /= 1000.0
                return utcfromtimestamp(numeric)
            except (TypeError, ValueError, OSError, OverflowError):
                pass
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return (
                    parsed.astimezone(timezone.utc)
                    if parsed.tzinfo is not None
                    else parsed.replace(tzinfo=timezone.utc)
                )
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_text_list(value: object) -> list[str]:
        if value is None:
            return []

        raw_values: list[object] = []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    raw_values.extend(parsed)
                else:
                    raw_values.append(text)
            else:
                raw_values.extend(part.strip() for part in text.split(",") if part.strip())
        elif isinstance(value, (list, tuple, set)):
            raw_values.extend(list(value))
        else:
            raw_values.append(value)

        out: list[str] = []
        for item in raw_values:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def is_market_tradable(
        market_info: Optional[dict],
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        """Return False when market metadata indicates closed/resolved/non-tradable."""
        if not isinstance(market_info, dict) or not market_info:
            return True

        ref_now = now or utcnow()
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)
        else:
            ref_now = ref_now.astimezone(timezone.utc)

        closed = _coerce_bool(market_info.get("closed"), None)
        if closed is True:
            return False

        active = _coerce_bool(market_info.get("active"), None)
        if active is False:
            return False

        archived = _coerce_bool(market_info.get("archived"), None)
        if archived is True:
            return False

        accepting_orders = _coerce_bool(
            market_info.get("accepting_orders")
            if market_info.get("accepting_orders") is not None
            else market_info.get("acceptingOrders"),
            None,
        )
        if accepting_orders is False:
            return False

        enable_order_book = _coerce_bool(
            market_info.get("enable_order_book")
            if market_info.get("enable_order_book") is not None
            else market_info.get("enableOrderBook"),
            None,
        )
        if enable_order_book is False:
            return False

        resolved = _coerce_bool(
            market_info.get("resolved")
            if market_info.get("resolved") is not None
            else (
                market_info.get("isResolved")
                if market_info.get("isResolved") is not None
                else market_info.get("is_resolved")
            ),
            None,
        )
        if resolved is True:
            return False

        end_dt = PolymarketClient._coerce_datetime(
            market_info.get("end_date") if market_info.get("end_date") is not None else market_info.get("endDate")
        )
        if end_dt:
            if end_dt <= ref_now:
                return False

        winner = market_info.get("winner")
        if winner not in (None, ""):
            return False

        winning_outcome = (
            market_info.get("winning_outcome")
            if market_info.get("winning_outcome") is not None
            else market_info.get("winningOutcome")
        )
        if winning_outcome not in (None, ""):
            return False

        raw_status = (
            market_info.get("status")
            if market_info.get("status") is not None
            else (
                market_info.get("market_status")
                if market_info.get("market_status") is not None
                else market_info.get("marketStatus")
            )
        )
        status_text = str(raw_status or "").strip().lower()
        if status_text:
            normalized = status_text.replace("_", " ").replace("-", " ")
            blocked = (
                "in review",
                "review",
                "in dispute",
                "dispute",
                "final",
                "resolved",
                "settled",
                "closed",
                "expired",
                "cancelled",
                "canceled",
                "suspend",
                "halt",
                "paused",
            )
            if any(term in normalized for term in blocked):
                return False

        uma_status_values: list[str] = []
        uma_status_values.extend(
            PolymarketClient._coerce_text_list(
                market_info.get("uma_resolution_status")
                if market_info.get("uma_resolution_status") is not None
                else market_info.get("umaResolutionStatus")
            )
        )
        uma_status_values.extend(
            PolymarketClient._coerce_text_list(
                market_info.get("uma_resolution_statuses")
                if market_info.get("uma_resolution_statuses") is not None
                else market_info.get("umaResolutionStatuses")
            )
        )
        if uma_status_values:
            blocked = (
                "proposed",
                "review",
                "dispute",
                "final",
                "resolved",
                "settled",
                "closed",
                "expired",
                "cancelled",
                "canceled",
            )
            for raw in uma_status_values:
                normalized = str(raw).strip().lower().replace("_", " ").replace("-", " ")
                if any(term in normalized for term in blocked):
                    return False

        return True

    @staticmethod
    def _extract_token_ids_from_market(market_data: dict) -> list[str]:
        raw_values: list[object] = []
        for key in ("clobTokenIds", "clob_token_ids", "token_ids", "tokenIds"):
            raw = market_data.get(key)
            if raw is None:
                continue
            if isinstance(raw, list):
                raw_values.extend(raw)
            elif isinstance(raw, str):
                text = raw.strip()
                if not text:
                    continue
                if text.startswith("["):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            raw_values.extend(parsed)
                    except Exception:
                        pass
                else:
                    raw_values.extend([part.strip() for part in text.split(",") if part.strip()])

        tokens = market_data.get("tokens")
        if isinstance(tokens, list):
            for token in tokens:
                if isinstance(token, dict):
                    raw_values.extend(
                        [
                            token.get("token_id"),
                            token.get("tokenId"),
                            token.get("asset_id"),
                            token.get("assetId"),
                            token.get("id"),
                        ]
                    )
                else:
                    raw_values.append(token)

        seen: set[str] = set()
        out: list[str] = []
        for value in raw_values:
            normalized = PolymarketClient._normalize_identifier(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    @staticmethod
    def _market_matches_condition_id(market_data: dict, condition_id: str) -> bool:
        return (
            PolymarketClient._normalize_identifier(market_data.get("condition_id") or market_data.get("conditionId"))
            == condition_id
        )

    @staticmethod
    def _market_matches_token_id(market_data: dict, token_id: str) -> bool:
        token_ids = PolymarketClient._extract_token_ids_from_market(market_data)
        return token_id in token_ids

    async def enrich_trades_with_market_info(self, trades: list[dict]) -> list[dict]:
        """
        Enrich a list of trades with market question/title and slug.
        Batches lookups and uses cache to minimize API calls.
        Falls back to asset_id (token_id) lookup when condition_id lookup fails.
        """

        def _get_asset_id(trade: dict) -> str:
            """Get asset/token ID, handling both camelCase and snake_case."""
            return trade.get("asset_id", "") or trade.get("assetId", "") or trade.get("asset", "")

        def _get_condition_id(trade: dict) -> str:
            """Get condition_id from trade, checking multiple field names."""
            return trade.get("condition_id", "") or trade.get("conditionId", "")

        def _is_hex_id(value: str) -> bool:
            """Check if a value looks like a hex condition_id (0x-prefixed)."""
            return value.startswith("0x")

        # Stage 1: Collect hex condition_ids for batch lookup
        unknown_cids = set()
        for trade in trades:
            market_val = trade.get("market", "")
            if market_val and _is_hex_id(market_val) and market_val not in self._market_cache:
                unknown_cids.add(market_val)
            # Also check explicit conditionId / condition_id field
            cid = _get_condition_id(trade)
            if cid and _is_hex_id(cid) and cid not in self._market_cache:
                unknown_cids.add(cid)

        # Batch lookup condition_ids with concurrency limit
        if unknown_cids:
            semaphore = asyncio.Semaphore(15)

            async def lookup(cid: str):
                async with semaphore:
                    await self.get_market_by_condition_id(cid)

            await asyncio.gather(*[lookup(cid) for cid in unknown_cids])

        # Stage 2: Collect token IDs for trades still missing market info
        token_lookups = set()
        for trade in trades:
            market_val = trade.get("market", "")
            cid = _get_condition_id(trade)

            # Check if already resolved via condition_id
            resolved = False
            if market_val and _is_hex_id(market_val) and self._market_cache.get(market_val):
                resolved = True
            if not resolved and cid and _is_hex_id(cid) and self._market_cache.get(cid):
                resolved = True

            if not resolved:
                # Try market value as a token_id if it's not a hex condition_id
                if market_val and not _is_hex_id(market_val):
                    cache_key = f"token:{market_val}"
                    if cache_key not in self._market_cache:
                        token_lookups.add(market_val)

                # Try explicit asset_id / assetId field
                asset_id = _get_asset_id(trade)
                if asset_id and f"token:{asset_id}" not in self._market_cache:
                    token_lookups.add(asset_id)

        # Batch lookup by token_id for trades missing market info
        if token_lookups:
            semaphore = asyncio.Semaphore(15)

            async def lookup_token(tid: str):
                async with semaphore:
                    await self.get_market_by_token_id(tid)

            await asyncio.gather(*[lookup_token(tid) for tid in token_lookups])

        # Enrich trades
        enriched = []
        for trade in trades:
            market_val = trade.get("market", "")
            cid = _get_condition_id(trade)
            market_info = None

            # Try hex condition_id from market field
            if market_val and _is_hex_id(market_val):
                market_info = self._market_cache.get(market_val)

            # Try explicit conditionId / condition_id field
            if not market_info and cid and _is_hex_id(cid):
                market_info = self._market_cache.get(cid)

            # Try market value as token_id (numeric/non-hex)
            if not market_info and market_val and not _is_hex_id(market_val):
                market_info = self._market_cache.get(f"token:{market_val}")

            # Fallback: try asset_id / assetId
            if not market_info:
                asset_id = _get_asset_id(trade)
                if asset_id:
                    market_info = self._market_cache.get(f"token:{asset_id}")

            enriched_trade = {**trade}
            # Prefer Data API native fields (title, slug, eventSlug) - they match the trade.
            # Only use Gamma lookup to fill in gaps; Gamma can return wrong/archived markets.
            api_title = trade.get("title", "")
            api_slug = trade.get("slug", "")
            api_event_slug = trade.get("eventSlug", trade.get("event_slug", ""))
            if market_info:
                enriched_trade["market_title"] = api_title or (
                    market_info.get("groupItemTitle") or market_info.get("question", "")
                )
                enriched_trade["market_slug"] = api_slug or market_info.get("slug", "")
                enriched_trade["event_slug"] = api_event_slug or market_info.get("event_slug", "")
            else:
                enriched_trade["market_title"] = api_title
                enriched_trade["market_slug"] = api_slug
                enriched_trade["event_slug"] = api_event_slug

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
                        enriched_trade["timestamp_iso"] = utcfromtimestamp(ts).isoformat() + "Z"
                    elif isinstance(ts, str):
                        if "T" in ts or "-" in ts:
                            # Already ISO format
                            enriched_trade["timestamp_iso"] = ts
                        else:
                            # Numeric string (unix seconds)
                            enriched_trade["timestamp_iso"] = utcfromtimestamp(float(ts)).isoformat() + "Z"
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

    async def get_midpoint(self, token_id: str, use_trading_proxy: bool = False) -> float:
        """Get midpoint price for a token.

        3-second TTL cache.  Midpoints change continuously, but every
        reconcile cycle for the same token within a 3s window can
        reuse the same value — and N traders sharing the same token
        universe collapse to a single fetch.

        Bypasses the cache when ``use_trading_proxy=True`` (the
        proxied path is used for order-placement context, where
        staleness is unacceptable).
        """
        normalized_token_id = str(token_id or "").strip()
        if not normalized_token_id:
            return 0.0

        async def _fetch() -> float:
            client = await (self._get_trading_client() if use_trading_proxy else self._get_client())
            response = await self._rate_limited_get(
                f"{self.clob_url}/midpoint",
                client=client,
                params={"token_id": normalized_token_id},
            )
            response.raise_for_status()
            data = response.json()
            return float(data.get("mid", 0))

        if use_trading_proxy:
            # Trading-context path — never serve stale.
            return await _fetch()
        return await self._midpoint_cache.get_or_fetch(normalized_token_id, _fetch)

    async def get_midpoints_batch(self, token_ids: list[str]) -> dict[str, float]:
        """Batch midpoint fetch via POST /midpoints — replaces N
        single-token GETs with one HTTP call.

        Polymarket's CLOB ``/midpoints`` endpoint accepts a JSON array
        of ``{"token_id": "..."}`` objects and returns a flat map of
        ``{token_id: price_string}``.  Confirmed live against
        ``https://clob.polymarket.com/midpoints``.

        Cache-aware: tokens already in the 3-second midpoint cache are
        served without network; the batch request is issued only for
        the cache misses.  Result is written back into the cache so a
        subsequent ``get_midpoint(token_id)`` call hits.
        """
        if not token_ids:
            return {}

        normalized: list[str] = []
        seen: set[str] = set()
        for tid in token_ids:
            t = str(tid or "").strip()
            if t and t not in seen:
                seen.add(t)
                normalized.append(t)

        result: dict[str, float] = {}
        misses: list[str] = []
        for t in normalized:
            cached = self._midpoint_cache.peek(t)
            if cached is not _CACHE_MISS:
                result[t] = float(cached)
            else:
                misses.append(t)

        if not misses:
            return result

        # Single-flight per-key: if another caller is already fetching
        # one of our missing tokens, wait for them rather than firing
        # a duplicate batch request.  (Rare in practice — most callers
        # batch all their tokens in one go.)
        client = await self._get_client()
        endpoint = f"{self.clob_url}/midpoints"
        payload = [{"token_id": t} for t in misses]
        await rate_limiter.acquire(endpoint_for_url(endpoint))
        try:
            response = await client.post(endpoint, json=payload, timeout=10.0)
            response.raise_for_status()
            raw = response.json()
        except Exception as exc:
            _logger.warning("Midpoint batch fetch failed; falling back to per-token", error=str(exc))
            # Fall back to single-token cache-aware fetches; each will
            # acquire its own rate-limit token.
            for t in misses:
                try:
                    result[t] = await self.get_midpoint(t)
                except Exception:
                    pass
            return result

        if not isinstance(raw, dict):
            return result
        now = time.monotonic()
        for t, raw_price in raw.items():
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                continue
            result[str(t)] = price
            # Backfill the per-token cache so subsequent get_midpoint
            # calls within the TTL hit instantly.
            self._midpoint_cache._data[str(t)] = (now, price)
        return result

    async def get_price(self, token_id: str, side: str = "BUY", use_trading_proxy: bool = False) -> float:
        """Get best price for a token (BUY = best bid, SELL = best ask)."""
        client = await (self._get_trading_client() if use_trading_proxy else self._get_client())
        response = await self._rate_limited_get(
            f"{self.clob_url}/price",
            client=client,
            params={"token_id": token_id, "side": side},
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get("price", 0))

    async def get_order_book(self, token_id: str, use_trading_proxy: bool = False) -> dict:
        """Get full order book for a token"""
        client = await (self._get_trading_client() if use_trading_proxy else self._get_client())
        response = await self._rate_limited_get(f"{self.clob_url}/book", client=client, params={"token_id": token_id})
        response.raise_for_status()
        return response.json()

    async def get_prices_history(
        self,
        token_id: str,
        interval: Optional[str] = None,
        fidelity: Optional[int] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        use_trading_proxy: bool = False,
    ) -> list[dict[str, float]]:
        """Get historical prices for a token from CLOB ``/prices-history``.

        Returns a normalized list of ``{"t": epoch_ms, "p": price}`` points.
        """
        client = await (self._get_trading_client() if use_trading_proxy else self._get_client())

        params: dict[str, object] = {"market": token_id}
        if interval:
            params["interval"] = interval
        if fidelity is not None:
            params["fidelity"] = int(fidelity)
        if start_ts is not None:
            # CLOB expects unix seconds for startTs/endTs.
            # Accept ms input too and normalize here.
            start_val = int(start_ts)
            if start_val > 10_000_000_000:
                start_val //= 1000
            params["startTs"] = start_val
        if end_ts is not None:
            end_val = int(end_ts)
            if end_val > 10_000_000_000:
                end_val //= 1000
            params["endTs"] = end_val

        response = await self._rate_limited_get(
            f"{self.clob_url}/prices-history",
            client=client,
            params=params,
        )
        response.raise_for_status()
        payload = response.json()

        # CLOB returns {"history": [{"t": ..., "p": ...}, ...]}.
        # Keep parsing defensive in case payload shape drifts.
        raw_history = payload
        if isinstance(payload, dict):
            raw_history = payload.get("history") or payload.get("prices") or payload.get("data") or []
        if not isinstance(raw_history, list):
            return []

        out: list[dict[str, float]] = []
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            t_raw = item.get("t") or item.get("timestamp") or item.get("time") or item.get("ts")
            p_raw = item.get("p")
            if p_raw is None:
                p_raw = item.get("price")

            try:
                t = float(t_raw)
                p = float(p_raw)
            except (TypeError, ValueError):
                continue

            # Handle seconds-vs-milliseconds timestamps.
            if t < 10_000_000_000:
                t *= 1000.0
            out.append({"t": t, "p": p})

        out.sort(key=lambda x: x["t"])
        return out

    async def get_prices_batch(self, token_ids: list[str]) -> dict[str, dict]:
        """Get prices for multiple tokens efficiently.

        Uses the CLOB ``/midpoints`` batch endpoint via
        ``get_midpoints_batch`` — one HTTP call for N tokens instead of
        N parallel single-token calls.  In production this replaced
        ~20 single-token requests fanning out under
        Semaphore(25) with one batch request.
        """
        midpoints = await self.get_midpoints_batch(list(token_ids))
        prices: dict[str, dict] = {}
        for token_id, mid in midpoints.items():
            if 0 <= mid <= 1:
                prices[str(token_id)] = {"mid": float(mid)}
        return prices

    # ==================== DATA API ====================

    async def get_wallet_positions(self, address: str) -> list[dict]:
        """Get open positions for a wallet, paging through all results."""
        normalized_address = str(address or "").strip()
        if not normalized_address:
            return []

        page_size = 500
        offset = 0
        all_rows: list[dict] = []
        seen_position_keys: set[tuple[str, str, str]] = set()

        for _ in range(_WALLET_POSITIONS_MAX_PAGES):
            response = await self._rate_limited_get(
                f"{self.data_url}/positions",
                params={
                    "user": normalized_address,
                    "limit": page_size,
                    "offset": offset,
                },
            )
            response.raise_for_status()

            payload = response.json()
            page_rows = self._extract_list_payload(
                payload,
                preferred_keys=("data", "items", "positions"),
            )
            if not page_rows:
                break

            unique_page_rows: list[dict] = []
            for row in page_rows:
                key = (
                    str(row.get("asset") or row.get("asset_id") or row.get("token_id") or "").strip(),
                    str(row.get("conditionId") or row.get("condition_id") or row.get("market") or "").strip(),
                    str(row.get("outcomeIndex") or row.get("outcome") or "").strip().lower(),
                )
                if key in seen_position_keys:
                    continue
                seen_position_keys.add(key)
                unique_page_rows.append(row)

            if unique_page_rows:
                all_rows.extend(unique_page_rows)

            if len(page_rows) < page_size:
                break

            offset += len(page_rows)

        return all_rows

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

    async def resolve_wallet_identifier(self, identifier: str) -> dict[str, str | None]:
        raw_value = str(identifier or "").strip()
        if not raw_value:
            raise ValueError("Wallet identifier cannot be empty")

        candidate = raw_value
        if "polymarket.com" in candidate.lower():
            url_match = re.search(r"polymarket\.com/(?:@|profile/)?([^/?#]+)", candidate, flags=re.IGNORECASE)
            if url_match:
                candidate = url_match.group(1).strip()

        if candidate.startswith("@"):
            candidate = candidate[1:].strip()

        if not candidate:
            raise ValueError("Wallet identifier cannot be empty")

        if _ETH_ADDRESS_RE.match(candidate):
            return {"address": candidate.lower(), "username": None}

        target_username = candidate.lower()

        for addr, uname in self._username_cache.items():
            if str(uname or "").strip().lower() == target_username and _ETH_ADDRESS_RE.match(addr):
                return {"address": addr.lower(), "username": uname}

        cache = await self._get_persistent_cache()
        if cache:
            # Reverse-lookup walks the persistent username cache by value.
            # Read-only iteration — do NOT copy into self._username_cache,
            # that was the same auto-seeding bug that quadrupled the
            # market cache.  The persistent dict is the source of truth.
            for addr, uname in cache._username_cache.items():
                if str(uname or "").strip().lower() == target_username and _ETH_ADDRESS_RE.match(addr):
                    return {"address": addr.lower(), "username": uname}

        profile_url = f"https://polymarket.com/@{quote(candidate)}"
        try:
            response = await self._rate_limited_get(
                profile_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                follow_redirects=True,
                timeout=20.0,
            )
            if response.status_code == 200:
                html = response.text
                address_match = re.search(
                    r'(?:\\"|")proxyAddress(?:\\"|"):(?:\\"|")(0x[a-fA-F0-9]{40})(?:\\"|")',
                    html,
                )
                if address_match:
                    resolved_address = address_match.group(1).lower()
                    self._username_cache[resolved_address] = candidate
                    if cache:
                        await cache.set_username(resolved_address, candidate)
                    return {"address": resolved_address, "username": candidate}
        except Exception:
            pass

        try:
            for order_by in ("PNL", "VOL"):
                for offset in range(0, 500, 50):
                    leaderboard = await self.get_leaderboard(limit=50, order_by=order_by, offset=offset)
                    if not leaderboard:
                        break
                    for entry in leaderboard:
                        uname = str(entry.get("userName") or "").strip()
                        address = str(entry.get("proxyWallet") or "").strip().lower()
                        if not uname or not _ETH_ADDRESS_RE.match(address):
                            continue
                        self._username_cache[address] = uname
                        if uname.lower() == target_username:
                            if cache:
                                await cache.set_username(address, uname)
                            return {"address": address, "username": uname}
        except Exception:
            pass

        raise ValueError(f"Could not resolve wallet identifier '{raw_value}' to an on-chain address")

    async def get_user_profile(self, address: str) -> dict:
        """
        Get user profile info from Polymarket.
        Tries multiple sources: username cache, leaderboard API, data API, and website scraping.
        """
        address_lower = address.lower()

        # Check username cache first (populated by discover/leaderboard scans)
        if address_lower in self._username_cache:
            return {
                "username": self._username_cache[address_lower],
                "address": address,
            }

        # Check persistent SQL cache (canonical username store).  Don't
        # mirror the result into self._username_cache — every fall-through
        # would otherwise duplicate the entry locally, which is the
        # pattern that produced the 65K shadow market_cache leak.
        cache = await self._get_persistent_cache()
        if cache:
            cached_username = await cache.get_username(address_lower)
            if cached_username:
                return {"username": cached_username, "address": address}

        # Try the leaderboard API - search both PNL and VOL sorted, multiple pages
        try:
            for order_by in ["PNL", "VOL"]:
                for offset in range(0, 200, 50):
                    leaderboard = await self.get_leaderboard(limit=50, order_by=order_by, offset=offset)
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
            _logger.warning(
                "Leaderboard lookup failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )

        # Try the data API profile endpoint
        try:
            response = await self._rate_limited_get(f"{self.data_url}/profile", params={"address": address})
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
            response = await self._rate_limited_get(f"{self.data_url}/users", params={"proxyAddress": address})
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
            response = await self._rate_limited_get(
                f"https://polymarket.com/profile/{address}",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                follow_redirects=True,
                timeout=20.0,
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
                og_match = re.search(r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"', html)
                if og_match:
                    username = og_match.group(1).strip()
                    if username and "polymarket" not in username.lower():
                        return {"username": username, "address": address}

        except httpx.TransportError as e:
            _logger.debug(
                "Profile fetch transport error",
                address=address,
                error=str(e),
                error_type=type(e).__name__,
            )
        except Exception as e:
            _logger.warning(
                "Error fetching profile",
                address=address,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )

        return {"username": None, "address": address}

    async def get_wallet_trades(self, address: str, limit: int = 100, offset: int = 0) -> list[dict]:
        """Get recent trades for a wallet."""
        params = {
            "user": address,
            "limit": min(max(int(limit or 100), 1), 500),
            "offset": max(int(offset or 0), 0),
        }
        response = await self._rate_limited_get(f"{self.data_url}/trades", params=params)
        response.raise_for_status()
        data = response.json()
        return self._extract_list_payload(
            data,
            preferred_keys=("data", "items", "trades", "activity"),
        )

    async def get_wallet_trades_paginated(self, address: str, *, max_trades: int = 1000, page_size: int = 500) -> list[dict]:
        """Pull a wallet's trade history page-by-page from Polymarket.

        Polymarket's ``/trades`` endpoint exposes only offset-based
        pagination and silently caps offsets at ~3500 (returns 400
        Bad Request beyond that).  There is no documented timestamp
        cursor.  We honor that ceiling: on a 400 we stop paging and
        return what we have, so the caller never sees a crash for
        a wallet with > 4000 trades — they just get the most recent
        ~3500.  The cap is logged at INFO so operators know we
        truncated.
        """
        # 60s TTL cache, single-flight.  Single-user install + 5-minute
        # verifier cycle = sub-second cache hit on every cycle except
        # the first.  The cache key includes max_trades because callers
        # ask for different sizes — most are tiny (verifier=10000,
        # lifecycle=300) but the strategy reverse-engineer agent pulls
        # the wallet's full history (capped at 250k as a safety net,
        # though Polymarket's offset ceiling will usually clip first).
        normalized_address = str(address or "").strip().lower()
        capped_page_size = max(1, min(int(page_size or 500), 500))
        capped_max_trades = max(1, min(int(max_trades or 1000), 250_000))
        if not normalized_address:
            return []
        cache_key = (normalized_address, capped_max_trades, capped_page_size)

        async def _fetch() -> list[dict]:
            import httpx as _httpx

            all_trades: list[dict] = []
            offset = 0
            hit_upstream_cap = False
            while len(all_trades) < capped_max_trades:
                try:
                    page = await self.get_wallet_trades(
                        normalized_address, limit=capped_page_size, offset=offset
                    )
                except _httpx.HTTPStatusError as exc:
                    # Polymarket returns 400 once offset exceeds ~3500.
                    # Treat as end-of-data, not as a crash.
                    if exc.response is not None and exc.response.status_code == 400:
                        hit_upstream_cap = True
                        _logger.info(
                            "polymarket /trades hit upstream offset cap at offset=%d for wallet %s — "
                            "returning %d trades captured so far",
                            offset, normalized_address, len(all_trades),
                        )
                        break
                    raise
                if not page:
                    break
                all_trades.extend(page)
                if len(page) < capped_page_size:
                    break
                offset += capped_page_size
                await asyncio.sleep(0.05)
            if hit_upstream_cap and len(all_trades) < capped_max_trades:
                _logger.info(
                    "polymarket /trades returned %d trades for wallet %s "
                    "(requested %d; truncated by upstream offset cap)",
                    len(all_trades), normalized_address, capped_max_trades,
                )
            return all_trades[:capped_max_trades]

        result = await self._wallet_trades_cache.get_or_fetch(cache_key, _fetch)
        # Defensive copy: callers commonly mutate the list (sort, filter,
        # extend); we don't want them poisoning the cache.
        return list(result)

    async def get_market_trades(self, condition_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        """Get recent trades for a market"""
        params = {"market": condition_id, "limit": limit}
        if offset > 0:
            params["offset"] = offset
        response = await self._rate_limited_get(f"{self.data_url}/trades", params=params)
        response.raise_for_status()
        data = response.json()
        return self._extract_list_payload(
            data,
            preferred_keys=("data", "items", "trades", "activity"),
        )

    async def get_activity(
        self,
        limit: int = 100,
        offset: int = 0,
        activity_type: Optional[str] = None,
        user: Optional[str] = None,
    ) -> list[dict]:
        """Fetch recent account activity from the Data API.

        Uses ``/v1/activity`` when available, then falls back to ``/activity``.
        """
        params = {
            "limit": min(limit, 500),
            "offset": min(offset, 1000),
        }
        if user:
            params["user"] = user
        if activity_type:
            params["type"] = activity_type

        for endpoint in (f"{self.data_url}/v1/activity", f"{self.data_url}/activity"):
            try:
                response = await self._rate_limited_get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                return self._extract_list_payload(
                    data,
                    preferred_keys=("data", "items", "activity"),
                )
            except Exception:
                continue
        return []

    async def get_wallet_activity(
        self,
        address: str,
        *,
        limit: int = 100,
        offset: int = 0,
        activity_types: Optional[list[str] | tuple[str, ...] | set[str]] = None,
    ) -> list[dict]:
        """Fetch recent wallet activity rows for a specific wallet."""
        normalized_types = [
            str(activity_type or "").strip()
            for activity_type in (activity_types or [])
            if str(activity_type or "").strip()
        ]
        if not normalized_types:
            return await self.get_activity(limit=limit, offset=offset, user=address)

        collected: list[dict] = []
        seen_fingerprints: set[str] = set()
        for activity_type in normalized_types:
            page = await self.get_activity(
                limit=limit,
                offset=offset,
                activity_type=activity_type,
                user=address,
            )
            for item in page:
                if not isinstance(item, dict):
                    continue
                fingerprint = "|".join(
                    (
                        str(item.get("transactionHash") or item.get("txHash") or "").strip().lower(),
                        str(item.get("type") or activity_type).strip().lower(),
                        str(item.get("asset") or item.get("asset_id") or item.get("token_id") or "").strip(),
                        str(item.get("timestamp") or item.get("createdAt") or item.get("created_at") or "").strip(),
                    )
                )
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                collected.append(item)
        return collected

    async def get_wallet_activity_paginated(
        self,
        address: str,
        *,
        max_items: int = 1000,
        page_size: int = 500,
        activity_types: Optional[list[str] | tuple[str, ...] | set[str]] = None,
    ) -> list[dict]:
        all_items: list[dict] = []
        offset = 0
        capped_page_size = max(1, min(int(page_size or 500), 500))
        capped_max_items = max(1, min(int(max_items or 1000), 10_000))
        while len(all_items) < capped_max_items:
            page = await self.get_wallet_activity(
                address,
                limit=capped_page_size,
                offset=offset,
                activity_types=activity_types,
            )
            if not page:
                break
            all_items.extend(page)
            if len(page) < capped_page_size:
                break
            offset += capped_page_size
            await asyncio.sleep(0.05)
        return all_items[:capped_max_items]

    async def get_market_holders(
        self,
        market_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch holders for a market from the Data API."""
        params = {
            "market": market_id,
            "limit": min(limit, 500),
            "offset": min(offset, 1000),
        }

        for endpoint in (f"{self.data_url}/holders", f"{self.data_url}/v1/holders"):
            try:
                response = await self._rate_limited_get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                return self._extract_list_payload(
                    data,
                    preferred_keys=("data", "items", "holders"),
                )
            except Exception:
                continue
        return []

    @staticmethod
    def _extract_list_payload(
        payload: object,
        preferred_keys: tuple[str, ...] = ("data", "items"),
    ) -> list[dict]:
        """Normalize API payloads that may return either list or wrapped objects."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in preferred_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            # Last resort: find the first list-valued field.
            for value in payload.values():
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

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

            response = await self._rate_limited_get(f"{self.data_url}/v1/leaderboard", params=params)
            response.raise_for_status()
            data = response.json()
            # API may return a list or a wrapper object
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", data.get("leaderboard", data.get("items", [])))
            return []
        except Exception as e:
            _logger.warning("Leaderboard fetch error", error=str(e))
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

        # Verify each trader has real activity using the fast
        # closed-positions endpoint (single call per trader instead of
        # fetching full trade history + open positions).
        # Keep concurrency low to stay within Polymarket rate limits.
        semaphore = asyncio.Semaphore(5)

        async def verify_trader(entry: dict):
            async with semaphore:
                address = entry.get("proxyWallet", "")
                if not address:
                    return None

                result = await self.calculate_win_rate_fast(address, min_positions=min_trades)
                if not result:
                    return None

                return {
                    "address": address,
                    "username": entry.get("userName", ""),
                    "trades": result["closed_positions"],
                    "volume": float(entry.get("vol", 0) or 0),
                    "pnl": float(entry.get("pnl", 0) or 0),
                    "rank": entry.get("rank", 0),
                    "buys": result["wins"],
                    "sells": result["losses"],
                    "win_rate": result["win_rate"],
                    "wins": result["wins"],
                    "losses": result["losses"],
                    "total_markets": result["closed_positions"],
                    "trade_count": result["closed_positions"],
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

    def _filter_by_time_period(self, trades: list[dict], time_period: str) -> list[dict]:
        """Filter trades by time period (DAY, WEEK, MONTH, ALL)."""
        if not trades or time_period.upper() == "ALL":
            return trades

        now = utcnow()
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
            ts = trade.get("timestamp") or trade.get("created_at") or trade.get("createdAt", "")
            if not ts:
                filtered.append(trade)  # Keep trades without timestamps
                continue
            try:
                parsed_ts = self._coerce_datetime(ts)
                if parsed_ts is None:
                    filtered.append(trade)
                    continue
                if parsed_ts >= cutoff:
                    filtered.append(trade)
            except (ValueError, TypeError):
                filtered.append(trade)  # Keep if we can't parse

        return filtered

    async def calculate_wallet_win_rate(self, address: str, max_trades: int = 500, time_period: str = "ALL") -> dict:
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
                market_id = trade.get("market") or trade.get("condition_id") or trade.get("assetId", "unknown")
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
                "unrealized_losses": position_losses,
            }
        except Exception as e:
            _logger.warning(
                "Error calculating win rate",
                address=address,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return {
                "address": address,
                "win_rate": 0.0,
                "wins": 0,
                "losses": 0,
                "total_markets": 0,
                "trade_count": 0,
                "error": str(e),
            }

    async def get_closed_positions(self, address: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Fetch closed positions for a wallet. Much more efficient than analyzing raw trades."""
        try:
            response = await self._rate_limited_get(
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
        except Exception as e:
            now = time.monotonic()
            if now >= self._closed_positions_warning_cooldown_until:
                self._closed_positions_warning_cooldown_until = now + 60.0
                _logger.warning(
                    "closed-positions fetch failed",
                    address=address[:10],
                    error=str(e),
                )
            else:
                _logger.debug(
                    "closed-positions fetch failed (suppressed)",
                    address=address[:10],
                    error_type=type(e).__name__,
                )
            return []

    async def get_closed_positions_paginated(self, address: str, max_positions: int = 200) -> list[dict]:
        """Fetch multiple pages of closed positions.

        60s TTL cache, single-flight.  ``data_positions`` has a 6 req/s
        rate limit; the rate limiter's token bucket already paces page
        requests, so a manual inter-page sleep was double-throttling
        and pushed the verifier past its 12s HTTP budget on a 5-page
        wallet (5 pages × 0.35s = 1.75s of pure idle latency on top
        of the rate-limiter wait).  The single-flight cache collapses
        concurrent same-wallet requests into one fetch chain, and the
        token bucket bounds the cross-wallet aggregate rate.
        """
        normalized_address = str(address or "").strip().lower()
        capped_max_positions = min(
            max(1, int(max_positions or 200)),
            _CLOSED_POSITIONS_MAX_FETCH,
        )
        if not normalized_address:
            return []
        cache_key = (normalized_address, capped_max_positions)

        async def _fetch() -> list[dict]:
            all_positions: list[dict] = []
            offset = 0
            page_size = _DATA_POSITIONS_PAGE_SIZE
            while len(all_positions) < capped_max_positions:
                page = await self.get_closed_positions(
                    normalized_address, limit=page_size, offset=offset
                )
                if not page:
                    break
                all_positions.extend(page)
                offset += len(page)
                if len(page) < page_size:
                    break
            return all_positions[:capped_max_positions]

        result = await self._closed_positions_cache.get_or_fetch(cache_key, _fetch)
        return list(result)

    async def calculate_win_rate_fast(self, address: str, min_positions: int = 5) -> Optional[dict]:
        """
        Fast win rate calculation using closed-positions endpoint.
        Returns None if trader doesn't meet minimum position threshold.
        Much faster than calculate_wallet_win_rate() since it uses a single
        pre-aggregated endpoint instead of fetching all raw trades.
        """
        try:
            # Fetch a single page of 50 closed positions — enough to
            # compute a reliable win rate without hammering the API.
            closed = await self.get_closed_positions(address, limit=50)

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

        semaphore = asyncio.Semaphore(5)

        async def analyze_trader(entry: dict):
            async with semaphore:
                address = entry.get("proxyWallet", "")
                if not address:
                    return None

                try:
                    result = await self.calculate_win_rate_fast(address, min_positions=min_trades)
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
            closed_positions: list[dict] = []
            positions: list[dict] = []
            trades: list[dict] = []

            try:
                closed_positions = await self.get_closed_positions_paginated(
                    address,
                    max_positions=_WALLET_PNL_CLOSED_POSITIONS_MAX_FETCH,
                )
            except Exception as closed_positions_exc:
                _logger.warning(
                    "Closed-positions fetch failed during PnL calculation; using empty fallback",
                    address=address,
                    error=str(closed_positions_exc),
                    error_type=type(closed_positions_exc).__name__,
                    exc_info=closed_positions_exc,
                )

            try:
                positions = await self.get_wallet_positions_with_prices(address)
            except Exception as positions_exc:
                is_rate_limited = (
                    isinstance(positions_exc, httpx.HTTPStatusError)
                    and positions_exc.response is not None
                    and positions_exc.response.status_code == 429
                )
                if is_rate_limited:
                    _logger.debug(
                        "Open-positions fetch rate-limited during PnL calculation; continuing without open positions",
                        address=address,
                    )
                else:
                    _logger.warning(
                        "Open-positions fetch failed during PnL calculation; using empty fallback",
                        address=address,
                        error=str(positions_exc),
                        error_type=type(positions_exc).__name__,
                        exc_info=positions_exc,
                    )

            try:
                trades = await self.get_wallet_trades(address, limit=500)
            except Exception as trades_exc:
                is_rate_limited = (
                    isinstance(trades_exc, httpx.HTTPStatusError)
                    and trades_exc.response is not None
                    and trades_exc.response.status_code == 429
                )
                if is_rate_limited:
                    _logger.debug(
                        "Trade history fetch rate-limited during PnL calculation; continuing without trades",
                        address=address,
                    )
                else:
                    _logger.warning(
                        "Trade history fetch failed during PnL calculation; using empty fallback",
                        address=address,
                        error=str(trades_exc),
                        error_type=type(trades_exc).__name__,
                        exc_info=trades_exc,
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
                current_value = float(pos.get("currentValue", 0) or pos.get("current_value", 0) or 0)
                initial_value = float(pos.get("initialValue", 0) or pos.get("initial_value", 0) or 0)
                cash_pnl = float(pos.get("cashPnl", 0) or pos.get("cash_pnl", 0) or pos.get("pnl", 0) or 0)

                # Fallback: calculate from size * price if API values are 0
                if current_value == 0 and initial_value == 0:
                    size = float(pos.get("size", 0) or 0)
                    avg_price = float(pos.get("avgPrice", 0) or pos.get("avg_price", 0) or 0)
                    current_price = float(
                        pos.get("currentPrice", 0) or pos.get("curPrice", 0) or pos.get("price", 0) or 0
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
                total_invested = total_bought if total_bought > 0 else total_initial_value
                total_returned = total_sold if total_sold > 0 else total_cash_pnl + total_initial_value

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
            _logger.warning(
                "Error calculating PnL",
                address=address,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
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
