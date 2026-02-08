"""
WebSocket real-time data feed service for Polymarket and Kalshi.

Replaces HTTP polling with persistent WebSocket connections for sub-second
order book updates.  Each exchange feed maintains its own connection with
auto-reconnect, heartbeat keep-alive, and exponential backoff.  A unified
FeedManager provides a single entry-point for the rest of the application.

Key classes:
    PriceCache         -- thread-safe in-memory cache of prices and order books
    PolymarketWSFeed   -- WebSocket client for Polymarket CLOB
    KalshiWSFeed       -- WebSocket client for Kalshi
    FeedManager        -- singleton orchestrator with fallback to HTTP
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from config import settings
from services.optimization.vwap import OrderBook, OrderBookLevel
from utils.logger import get_logger

try:
    import websockets
    import websockets.exceptions

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = get_logger("ws_feeds")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLYMARKET_WS_URL = settings.CLOB_WS_URL
KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

DEFAULT_STALE_TTL = 15.0  # seconds before a cached price is considered stale
DEFAULT_HEARTBEAT_INTERVAL = 10.0  # seconds between keep-alive pings
DEFAULT_RECONNECT_BASE_DELAY = 1.0  # initial backoff delay in seconds
DEFAULT_RECONNECT_MAX_DELAY = 60.0  # maximum backoff ceiling
DEFAULT_RECONNECT_MULTIPLIER = 2.0  # exponential multiplier per attempt


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# PriceCache
# ---------------------------------------------------------------------------


@dataclass
class CachedEntry:
    """Single cached order book entry for one token_id."""

    best_bid: float = 0.0
    best_ask: float = 0.0
    order_book: Optional[OrderBook] = None
    updated_at: float = 0.0  # monotonic timestamp


class PriceCache:
    """Thread-safe in-memory cache of latest prices and full order books.

    All public methods acquire a lock so the cache can be read safely from
    non-async threads (e.g. a synchronous health-check endpoint).
    """

    def __init__(self, stale_ttl: float = DEFAULT_STALE_TTL) -> None:
        self._stale_ttl = stale_ttl
        self._lock = Lock()
        self._entries: Dict[str, CachedEntry] = {}

    # -- mutations ----------------------------------------------------------

    def update(
        self,
        token_id: str,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
    ) -> None:
        """Atomically replace the order book for *token_id*."""
        bids_sorted = sorted(bids, key=lambda lvl: lvl.price, reverse=True)
        asks_sorted = sorted(asks, key=lambda lvl: lvl.price)

        book = OrderBook(bids=bids_sorted, asks=asks_sorted)
        best_bid = bids_sorted[0].price if bids_sorted else 0.0
        best_ask = asks_sorted[0].price if asks_sorted else 0.0

        entry = CachedEntry(
            best_bid=best_bid,
            best_ask=best_ask,
            order_book=book,
            updated_at=time.monotonic(),
        )
        with self._lock:
            self._entries[token_id] = entry

    def remove(self, token_id: str) -> None:
        """Remove a token from the cache."""
        with self._lock:
            self._entries.pop(token_id, None)

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._entries.clear()

    # -- queries ------------------------------------------------------------

    def get_mid_price(self, token_id: str) -> Optional[float]:
        """Return mid price or ``None`` if not cached."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None:
            return None
        if entry.best_bid == 0.0 and entry.best_ask == 0.0:
            return None
        if entry.best_bid == 0.0:
            return entry.best_ask
        if entry.best_ask == 0.0:
            return entry.best_bid
        return (entry.best_bid + entry.best_ask) / 2.0

    def get_spread(self, token_id: str) -> Optional[float]:
        """Return absolute bid-ask spread or ``None``."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None or entry.best_bid == 0.0 or entry.best_ask == 0.0:
            return None
        return entry.best_ask - entry.best_bid

    def get_spread_bps(self, token_id: str) -> Optional[float]:
        """Return bid-ask spread in basis points or ``None``."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None or entry.best_bid == 0.0 or entry.best_ask == 0.0:
            return None
        mid = (entry.best_bid + entry.best_ask) / 2.0
        if mid <= 0.0:
            return None
        return ((entry.best_ask - entry.best_bid) / mid) * 10_000

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Return the full ``OrderBook`` or ``None``."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None:
            return None
        return entry.order_book

    def get_best_bid_ask(self, token_id: str) -> Optional[tuple[float, float]]:
        """Return ``(best_bid, best_ask)`` or ``None``."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None:
            return None
        return (entry.best_bid, entry.best_ask)

    def is_fresh(self, token_id: str) -> bool:
        """Return ``True`` if the cached data is within the staleness TTL."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None or entry.updated_at == 0.0:
            return False
        return (time.monotonic() - entry.updated_at) < self._stale_ttl

    def staleness(self, token_id: str) -> Optional[float]:
        """Return age in seconds of the cached data, or ``None``."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None or entry.updated_at == 0.0:
            return None
        return time.monotonic() - entry.updated_at

    def all_token_ids(self) -> List[str]:
        """Return a snapshot of all cached token IDs."""
        with self._lock:
            return list(self._entries.keys())


# ---------------------------------------------------------------------------
# Shared feed statistics
# ---------------------------------------------------------------------------


@dataclass
class FeedStats:
    """Lightweight statistics counters for a single feed."""

    messages_received: int = 0
    messages_parsed: int = 0
    parse_errors: int = 0
    reconnections: int = 0
    last_message_at: float = 0.0  # monotonic
    last_latency_ms: float = 0.0
    connection_uptime_start: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        if self.connection_uptime_start == 0.0:
            return 0.0
        return time.monotonic() - self.connection_uptime_start

    def to_dict(self) -> dict:
        return {
            "messages_received": self.messages_received,
            "messages_parsed": self.messages_parsed,
            "parse_errors": self.parse_errors,
            "reconnections": self.reconnections,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


# ---------------------------------------------------------------------------
# PolymarketWSFeed
# ---------------------------------------------------------------------------


class PolymarketWSFeed:
    """Manages a WebSocket connection to the Polymarket CLOB order book feed.

    Supports dynamic subscription to multiple markets / asset IDs and keeps
    a shared :class:`PriceCache` up to date with the latest order book
    snapshots and deltas.
    """

    def __init__(
        self,
        cache: PriceCache,
        ws_url: str = POLYMARKET_WS_URL,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        reconnect_base_delay: float = DEFAULT_RECONNECT_BASE_DELAY,
        reconnect_max_delay: float = DEFAULT_RECONNECT_MAX_DELAY,
    ) -> None:
        self._cache = cache
        self._ws_url = ws_url
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # Subscription tracking
        self._subscribed_markets: Set[str] = set()  # condition_ids
        self._subscribed_assets: Set[str] = set()  # token_ids
        self._sub_lock = asyncio.Lock()

        # Connection state
        self._ws: Any = None  # websockets connection object
        self._state = ConnectionState.DISCONNECTED
        self._run_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Stats
        self.stats = FeedStats()

    # -- public API ---------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def start(self) -> None:
        """Start the feed in the background.  Idempotent."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error(
                "Cannot start PolymarketWSFeed: websockets library not installed"
            )
            return
        if self._run_task is not None and not self._run_task.done():
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(
            self._run_loop(), name="polymarket-ws-feed"
        )
        logger.info("PolymarketWSFeed started")

    async def stop(self) -> None:
        """Gracefully shut down the connection and background tasks."""
        self._stop_event.set()
        self._state = ConnectionState.CLOSED
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
        self._ws = None
        logger.info("PolymarketWSFeed stopped")

    async def subscribe(
        self,
        condition_ids: Optional[List[str]] = None,
        token_ids: Optional[List[str]] = None,
    ) -> None:
        """Subscribe to additional markets or assets.

        Can be called before or after the feed is started.  If the connection
        is already live the subscription message is sent immediately.
        """
        async with self._sub_lock:
            if condition_ids:
                self._subscribed_markets.update(condition_ids)
            if token_ids:
                self._subscribed_assets.update(token_ids)
            if self._ws and self._state == ConnectionState.CONNECTED:
                await self._send_subscribe(
                    condition_ids or [],
                    token_ids or [],
                )

    async def unsubscribe(
        self,
        condition_ids: Optional[List[str]] = None,
        token_ids: Optional[List[str]] = None,
    ) -> None:
        """Remove subscriptions.  Cached data for removed tokens is cleared."""
        async with self._sub_lock:
            removed_assets: List[str] = []
            if condition_ids:
                self._subscribed_markets.difference_update(condition_ids)
            if token_ids:
                self._subscribed_assets.difference_update(token_ids)
                removed_assets = list(token_ids)

        for tid in removed_assets:
            self._cache.remove(tid)

    # -- internal connection loop -------------------------------------------

    async def _run_loop(self) -> None:
        """Outer loop: connect, listen, and reconnect on failure."""
        attempt = 0
        while not self._stop_event.is_set():
            try:
                self._state = (
                    ConnectionState.CONNECTING
                    if attempt == 0
                    else ConnectionState.RECONNECTING
                )
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                attempt += 1
                self.stats.reconnections += 1
                delay = min(
                    self._reconnect_base_delay
                    * (DEFAULT_RECONNECT_MULTIPLIER ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                logger.warning(
                    f"Polymarket WS disconnected ({exc!r}), reconnecting in {delay:.1f}s "
                    f"(attempt {attempt})"
                )
                self._state = ConnectionState.DISCONNECTED
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                    break  # stop_event was set during wait
                except asyncio.TimeoutError:
                    pass  # delay elapsed, retry
            else:
                # Clean exit from _connect_and_listen (server closed gracefully)
                if not self._stop_event.is_set():
                    attempt += 1
                    self.stats.reconnections += 1
                    continue
                break

        self._state = ConnectionState.CLOSED

    async def _connect_and_listen(self) -> None:
        """Establish connection, subscribe, and process messages."""
        async with websockets.connect(
            self._ws_url,
            ping_interval=None,  # we manage heartbeats ourselves
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._state = ConnectionState.CONNECTED
            self.stats.connection_uptime_start = time.monotonic()
            logger.info("Polymarket WS connected", url=self._ws_url)

            # Re-subscribe to everything tracked
            async with self._sub_lock:
                if self._subscribed_markets or self._subscribed_assets:
                    await self._send_subscribe(
                        list(self._subscribed_markets),
                        list(self._subscribed_assets),
                    )

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ws), name="polymarket-ws-heartbeat"
            )

            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    recv_time = time.monotonic()
                    self.stats.messages_received += 1
                    self.stats.last_message_at = recv_time
                    try:
                        data = json.loads(raw)
                        self._handle_message(data, recv_time)
                        self.stats.messages_parsed += 1
                    except Exception as exc:
                        self.stats.parse_errors += 1
                        logger.debug(f"Polymarket WS parse error: {exc!r}")
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                self._ws = None

    async def _send_subscribe(
        self,
        condition_ids: List[str],
        token_ids: List[str],
    ) -> None:
        """Send a subscribe message over the live WebSocket."""
        if not self._ws:
            return
        msg: dict[str, Any] = {
            "auth": {},
            "type": "subscribe",
            "markets": condition_ids if condition_ids else [],
            "assets_ids": token_ids if token_ids else [],
            "channels": ["book"],
        }
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug(
                "Polymarket WS subscribed",
                markets=len(condition_ids),
                assets=len(token_ids),
            )
        except Exception as exc:
            logger.warning(f"Polymarket WS subscribe send failed: {exc!r}")

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Periodically send a ping frame to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    pong = await ws.ping()
                    start = time.monotonic()
                    await asyncio.wait_for(pong, timeout=5.0)
                    self.stats.last_latency_ms = (time.monotonic() - start) * 1000
                except asyncio.TimeoutError:
                    logger.warning("Polymarket WS heartbeat pong timeout")
                    await ws.close()
                    return
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    # -- message handling ---------------------------------------------------

    def _handle_message(self, data: dict, recv_time: float) -> None:
        """Route an incoming WebSocket message to the right handler."""
        # Polymarket book messages typically carry an "event_type" or can be
        # identified by the presence of bids/asks at the top level or nested
        # inside a "market" key.  We handle the common shapes:
        #   1) {"market": "<condition_id>", "asset_id": "...", "bids": [...], "asks": [...]}
        #   2) {"type": "book", "data": {...}} wrapper with nested payload
        #   3) {"data": [...]} array of updates

        # If bids/asks are directly on the top-level dict, apply immediately
        if "bids" in data or "asks" in data:
            self._apply_book_update(data, recv_time)
            return

        # Nested payload under "data" key (common Polymarket wrapper pattern)
        nested = data.get("data")
        if isinstance(nested, dict):
            self._apply_book_update(nested, recv_time)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    self._apply_book_update(item, recv_time)
        # Silently ignore heartbeat acks, subscribe confirmations, etc.

    def _apply_book_update(self, data: dict, recv_time: float) -> None:
        """Parse bids/asks arrays and push into PriceCache."""
        asset_id = (
            data.get("asset_id") or data.get("token_id") or data.get("market", "")
        )
        if not asset_id:
            return

        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])

        bids = self._parse_levels(raw_bids)
        asks = self._parse_levels(raw_asks)

        if bids or asks:
            self._cache.update(asset_id, bids, asks)

        # Compute server-to-cache latency if a timestamp is present
        ts = data.get("timestamp")
        if ts is not None:
            try:
                server_time = float(ts)
                # Polymarket timestamps are in milliseconds
                if server_time > 1e12:
                    server_time /= 1000.0
                self.stats.last_latency_ms = (recv_time - server_time) * 1000
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _parse_levels(raw: list) -> List[OrderBookLevel]:
        """Convert raw bid/ask entries to OrderBookLevel objects."""
        levels: List[OrderBookLevel] = []
        for entry in raw:
            try:
                price = float(entry.get("price", entry.get("p", 0)))
                size = float(entry.get("size", entry.get("s", 0)))
                if price > 0 and size > 0:
                    levels.append(OrderBookLevel(price=price, size=size))
            except (TypeError, ValueError, AttributeError):
                continue
        return levels


# ---------------------------------------------------------------------------
# KalshiWSFeed
# ---------------------------------------------------------------------------


class KalshiWSFeed:
    """Manages a WebSocket connection to the Kalshi order book feed.

    Protocol: JSON-RPC-style messages with ``cmd`` and ``params``.
    Subscribe:
        {"id": N, "cmd": "subscribe",
         "params": {"channels": ["orderbook_delta"],
                    "market_tickers": ["TICKER"]}}
    """

    def __init__(
        self,
        cache: PriceCache,
        ws_url: str = KALSHI_WS_URL,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        reconnect_base_delay: float = DEFAULT_RECONNECT_BASE_DELAY,
        reconnect_max_delay: float = DEFAULT_RECONNECT_MAX_DELAY,
    ) -> None:
        self._cache = cache
        self._ws_url = ws_url
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # Subscription tracking
        self._subscribed_tickers: Set[str] = set()
        self._sub_lock = asyncio.Lock()
        self._msg_id: int = 0

        # Connection state
        self._ws: Any = None
        self._state = ConnectionState.DISCONNECTED
        self._run_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Stats
        self.stats = FeedStats()

    # -- public API ---------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        return self._state

    def _next_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def start(self) -> None:
        """Start the feed in the background.  Idempotent."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start KalshiWSFeed: websockets library not installed")
            return
        if self._run_task is not None and not self._run_task.done():
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self._run_loop(), name="kalshi-ws-feed")
        logger.info("KalshiWSFeed started")

    async def stop(self) -> None:
        """Gracefully shut down."""
        self._stop_event.set()
        self._state = ConnectionState.CLOSED
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
        self._ws = None
        logger.info("KalshiWSFeed stopped")

    async def subscribe(self, tickers: List[str]) -> None:
        """Subscribe to one or more Kalshi market tickers."""
        async with self._sub_lock:
            self._subscribed_tickers.update(tickers)
            if self._ws and self._state == ConnectionState.CONNECTED:
                await self._send_subscribe(tickers)

    async def unsubscribe(self, tickers: List[str]) -> None:
        """Unsubscribe and clear cached data for given tickers."""
        async with self._sub_lock:
            self._subscribed_tickers.difference_update(tickers)
            if self._ws and self._state == ConnectionState.CONNECTED:
                await self._send_unsubscribe(tickers)
        for t in tickers:
            self._cache.remove(t)

    # -- internal connection loop -------------------------------------------

    async def _run_loop(self) -> None:
        """Outer reconnect loop with exponential backoff."""
        attempt = 0
        while not self._stop_event.is_set():
            try:
                self._state = (
                    ConnectionState.CONNECTING
                    if attempt == 0
                    else ConnectionState.RECONNECTING
                )
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                attempt += 1
                self.stats.reconnections += 1
                delay = min(
                    self._reconnect_base_delay
                    * (DEFAULT_RECONNECT_MULTIPLIER ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                logger.warning(
                    f"Kalshi WS disconnected ({exc!r}), reconnecting in {delay:.1f}s "
                    f"(attempt {attempt})"
                )
                self._state = ConnectionState.DISCONNECTED
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                    break
                except asyncio.TimeoutError:
                    pass
            else:
                if not self._stop_event.is_set():
                    attempt += 1
                    self.stats.reconnections += 1
                    continue
                break

        self._state = ConnectionState.CLOSED

    async def _connect_and_listen(self) -> None:
        """Open connection, subscribe, and consume messages."""
        extra_headers = {
            "User-Agent": "homerun-arb-scanner/1.0",
        }
        async with websockets.connect(
            self._ws_url,
            ping_interval=None,
            close_timeout=5,
            additional_headers=extra_headers,
        ) as ws:
            self._ws = ws
            self._state = ConnectionState.CONNECTED
            self.stats.connection_uptime_start = time.monotonic()
            logger.info("Kalshi WS connected", url=self._ws_url)

            # Re-subscribe
            async with self._sub_lock:
                if self._subscribed_tickers:
                    await self._send_subscribe(list(self._subscribed_tickers))

            # Heartbeat
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ws), name="kalshi-ws-heartbeat"
            )

            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    recv_time = time.monotonic()
                    self.stats.messages_received += 1
                    self.stats.last_message_at = recv_time
                    try:
                        data = json.loads(raw)
                        self._handle_message(data, recv_time)
                        self.stats.messages_parsed += 1
                    except Exception as exc:
                        self.stats.parse_errors += 1
                        logger.debug(f"Kalshi WS parse error: {exc!r}")
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                self._ws = None

    async def _send_subscribe(self, tickers: List[str]) -> None:
        if not self._ws:
            return
        msg = {
            "id": self._next_msg_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug("Kalshi WS subscribed", tickers=len(tickers))
        except Exception as exc:
            logger.warning(f"Kalshi WS subscribe send failed: {exc!r}")

    async def _send_unsubscribe(self, tickers: List[str]) -> None:
        if not self._ws:
            return
        msg = {
            "id": self._next_msg_id(),
            "cmd": "unsubscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug("Kalshi WS unsubscribed", tickers=len(tickers))
        except Exception as exc:
            logger.warning(f"Kalshi WS unsubscribe send failed: {exc!r}")

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Ping/pong keep-alive."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    pong = await ws.ping()
                    start = time.monotonic()
                    await asyncio.wait_for(pong, timeout=5.0)
                    self.stats.last_latency_ms = (time.monotonic() - start) * 1000
                except asyncio.TimeoutError:
                    logger.warning("Kalshi WS heartbeat pong timeout")
                    await ws.close()
                    return
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    # -- message handling ---------------------------------------------------

    def _handle_message(self, data: dict, recv_time: float) -> None:
        """Route incoming Kalshi messages."""
        # Kalshi pushes channel data under a "msg" or "data" key, and can
        # also have a "type" field.  Common shapes:
        #   {"id": ..., "type": "orderbook_delta", "msg": {"market_ticker": ..., "yes": [...], "no": [...]}}
        #   {"type": "orderbook_snapshot", "msg": {...}}
        #   {"sid": ..., "type": "orderbook_delta", "msg": {...}}

        msg_type = data.get("type", "")
        payload = data.get("msg") or data.get("data") or data

        if "orderbook" in msg_type:
            self._apply_orderbook(payload, recv_time)
        elif isinstance(payload, dict) and "market_ticker" in payload:
            self._apply_orderbook(payload, recv_time)

    def _apply_orderbook(self, payload: dict, recv_time: float) -> None:
        """Parse a Kalshi orderbook snapshot/delta into PriceCache."""
        ticker = payload.get("market_ticker", "")
        if not ticker:
            return

        # Kalshi books can come as:
        #   {"yes": [[price, size], ...], "no": [[price, size], ...]}
        # or:
        #   {"orderbook": {"yes": [...], "no": [...]}}
        book_data = payload.get("orderbook", payload)

        yes_raw = book_data.get("yes", [])
        no_raw = book_data.get("no", [])

        # For the unified cache we store Kalshi books keyed by ticker.
        # "yes" side asks are levels you can buy YES at, bids are implied
        # from the "no" side (buy NO at price p => implied YES bid at 1-p).
        yes_asks = self._parse_kalshi_levels(yes_raw)
        no_asks = self._parse_kalshi_levels(no_raw)

        # Build the YES order book:
        #   asks = yes_raw (prices you pay to buy YES)
        #   bids = derived from no_raw: if you can buy NO at p, YES implied bid = 1 - p
        implied_bids = [
            OrderBookLevel(price=round(1.0 - lvl.price, 4), size=lvl.size)
            for lvl in no_asks
            if lvl.price < 1.0
        ]

        self._cache.update(ticker, implied_bids, yes_asks)

    @staticmethod
    def _parse_kalshi_levels(raw: list) -> List[OrderBookLevel]:
        """Convert Kalshi price/qty pairs into OrderBookLevel objects.

        Kalshi levels can be ``[price_cents, size]`` or
        ``{"price": ..., "size": ...}``.  Price is in cents (1-99).
        """
        levels: List[OrderBookLevel] = []
        for entry in raw:
            try:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    price_raw = float(entry[0])
                    size = float(entry[1])
                elif isinstance(entry, dict):
                    price_raw = float(entry.get("price", 0))
                    size = float(entry.get("quantity", entry.get("size", 0)))
                else:
                    continue

                # Kalshi prices are in cents (1-99); normalise to 0-1
                price = price_raw / 100.0 if price_raw > 1.0 else price_raw

                if 0 < price < 1.0 and size > 0:
                    levels.append(OrderBookLevel(price=price, size=size))
            except (TypeError, ValueError):
                continue
        return levels


# ---------------------------------------------------------------------------
# FeedManager -- unified singleton interface
# ---------------------------------------------------------------------------


class FeedManager:
    """Singleton orchestrator for all WebSocket feeds.

    Provides a unified API for querying real-time prices across Polymarket
    and Kalshi, with automatic fallback to HTTP when WebSocket data is stale.

    Usage::

        mgr = FeedManager.get_instance()
        await mgr.start()
        await mgr.polymarket_feed.subscribe(condition_ids=["0x..."], token_ids=["123..."])
        price = await mgr.get_price("123...")
        book  = await mgr.get_order_book("123...")
    """

    _instance: Optional["FeedManager"] = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._cache = PriceCache()
        self._polymarket_feed = PolymarketWSFeed(cache=self._cache)
        self._kalshi_feed = KalshiWSFeed(cache=self._cache)
        self._http_fallback_fn: Optional[
            Callable[[str], Coroutine[Any, Any, Optional[OrderBook]]]
        ] = None
        self._started = False

    @classmethod
    def get_instance(cls) -> "FeedManager":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tear down the singleton (useful in tests)."""
        with cls._instance_lock:
            cls._instance = None

    # -- properties ---------------------------------------------------------

    @property
    def cache(self) -> PriceCache:
        return self._cache

    @property
    def polymarket_feed(self) -> PolymarketWSFeed:
        return self._polymarket_feed

    @property
    def kalshi_feed(self) -> KalshiWSFeed:
        return self._kalshi_feed

    # -- lifecycle ----------------------------------------------------------

    def set_http_fallback(
        self,
        fn: Callable[[str], Coroutine[Any, Any, Optional[OrderBook]]],
    ) -> None:
        """Register an async callback ``fn(token_id) -> OrderBook | None``
        to be used when WebSocket data is stale."""
        self._http_fallback_fn = fn

    async def start(self) -> None:
        """Start both feeds.  Idempotent."""
        if self._started:
            return
        await self._polymarket_feed.start()
        await self._kalshi_feed.start()
        self._started = True
        logger.info("FeedManager started (Polymarket + Kalshi)")

    async def stop(self) -> None:
        """Stop both feeds and clear cache."""
        await self._polymarket_feed.stop()
        await self._kalshi_feed.stop()
        self._cache.clear()
        self._started = False
        logger.info("FeedManager stopped")

    # -- unified queries ----------------------------------------------------

    async def get_price(self, token_id: str) -> Optional[float]:
        """Return the mid price for *token_id*, falling back to HTTP if stale.

        Returns ``None`` if neither WebSocket nor HTTP can provide a price.
        """
        if self._cache.is_fresh(token_id):
            return self._cache.get_mid_price(token_id)

        # Attempt HTTP fallback
        book = await self._http_fallback(token_id)
        if book is not None:
            mid = self._mid_from_book(book)
            if mid is not None:
                return mid

        # Even stale WS data is better than nothing
        return self._cache.get_mid_price(token_id)

    async def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Return the full order book, falling back to HTTP if stale."""
        if self._cache.is_fresh(token_id):
            return self._cache.get_order_book(token_id)

        book = await self._http_fallback(token_id)
        if book is not None:
            return book

        return self._cache.get_order_book(token_id)

    def is_fresh(self, token_id: str) -> bool:
        """Check whether cached data for *token_id* is within TTL."""
        return self._cache.is_fresh(token_id)

    async def get_best_bid_ask(self, token_id: str) -> Optional[tuple[float, float]]:
        """Return ``(best_bid, best_ask)`` with HTTP fallback."""
        if self._cache.is_fresh(token_id):
            return self._cache.get_best_bid_ask(token_id)

        book = await self._http_fallback(token_id)
        if book is not None:
            bid = book.bids[0].price if book.bids else 0.0
            ask = book.asks[0].price if book.asks else 0.0
            return (bid, ask)

        return self._cache.get_best_bid_ask(token_id)

    # -- health & stats -----------------------------------------------------

    def health_check(self) -> dict:
        """Return a health summary for monitoring endpoints."""
        poly_ok = self._polymarket_feed.state == ConnectionState.CONNECTED
        kalshi_ok = self._kalshi_feed.state == ConnectionState.CONNECTED

        return {
            "healthy": poly_ok or kalshi_ok,
            "websockets_available": WEBSOCKETS_AVAILABLE,
            "started": self._started,
            "polymarket": {
                "state": self._polymarket_feed.state.value,
                "connected": poly_ok,
                "stats": self._polymarket_feed.stats.to_dict(),
            },
            "kalshi": {
                "state": self._kalshi_feed.state.value,
                "connected": kalshi_ok,
                "stats": self._kalshi_feed.stats.to_dict(),
            },
            "cache": {
                "token_count": len(self._cache.all_token_ids()),
            },
        }

    def get_statistics(self) -> dict:
        """Detailed statistics for both feeds."""
        return {
            "polymarket": self._polymarket_feed.stats.to_dict(),
            "kalshi": self._kalshi_feed.stats.to_dict(),
            "cache_size": len(self._cache.all_token_ids()),
        }

    # -- internal helpers ---------------------------------------------------

    async def _http_fallback(self, token_id: str) -> Optional[OrderBook]:
        """Attempt to fetch an order book via HTTP.

        Returns ``None`` if no fallback function is registered or the fetch
        fails.  On success the cache is also refreshed so subsequent reads
        can use the fresh data.
        """
        if self._http_fallback_fn is None:
            return None
        try:
            book = await self._http_fallback_fn(token_id)
            if book is not None:
                # Refresh cache so subsequent synchronous reads are fresh
                self._cache.update(token_id, book.bids, book.asks)
            return book
        except Exception as exc:
            logger.debug(f"HTTP fallback failed for {token_id}: {exc!r}")
            return None

    @staticmethod
    def _mid_from_book(book: OrderBook) -> Optional[float]:
        """Compute mid price from an OrderBook."""
        if book.bids and book.asks:
            return (book.bids[0].price + book.asks[0].price) / 2.0
        if book.asks:
            return book.asks[0].price
        if book.bids:
            return book.bids[0].price
        return None


# ---------------------------------------------------------------------------
# Module-level convenience accessor
# ---------------------------------------------------------------------------


def get_feed_manager() -> FeedManager:
    """Shorthand for ``FeedManager.get_instance()``."""
    return FeedManager.get_instance()
