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
from collections import deque
from datetime import datetime, timezone
import json
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from sqlalchemy import select

from config import settings
from models.database import AppSettings, AsyncSessionLocal
from services.optimization.vwap import OrderBook, OrderBookLevel
from services.redis_price_cache import redis_price_cache
from utils.logger import get_logger
from utils.secrets import decrypt_secret

try:
    import websockets
    import websockets.exceptions

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = get_logger("ws_feeds")


def _is_expected_close(exc: Exception) -> bool:
    """Return True for expected lifecycle closures from remote peers."""

    if not WEBSOCKETS_AVAILABLE:
        return False

    if isinstance(exc, (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError)):
        close_code = getattr(exc, "code", None)
        if close_code is None:
            received = getattr(exc, "rcvd", None)
            sent = getattr(exc, "sent", None)
            close_code = getattr(received, "code", None) or getattr(sent, "code", None)
        return close_code in {1000, 1001, 1005}
    return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLYMARKET_WS_URL = settings.CLOB_WS_URL
KALSHI_WS_URL = settings.KALSHI_WS_URL

DEFAULT_STALE_TTL = float(settings.WS_PRICE_STALE_SECONDS)
DEFAULT_HEARTBEAT_INTERVAL = max(1.0, float(settings.WS_HEARTBEAT_INTERVAL))  # seconds between keep-alive pings
DEFAULT_HEARTBEAT_PONG_TIMEOUT = 12.0
DEFAULT_HEARTBEAT_MAX_MISSES = 2
DEFAULT_RECONNECT_BASE_DELAY = 1.0  # initial backoff delay in seconds
DEFAULT_RECONNECT_MAX_DELAY = 60.0  # maximum backoff ceiling
DEFAULT_RECONNECT_MULTIPLIER = 2.0  # exponential multiplier per attempt
MAX_RECONNECT_ATTEMPTS = 30  # after this many consecutive failures, extend delay
MAX_RECONNECT_EXTENDED_DELAY = 300.0  # 5 minutes between attempts after exhaustion


# ---------------------------------------------------------------------------
# Kalshi auth helpers
# ---------------------------------------------------------------------------


def _clean_secret(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


async def _load_kalshi_stored_credentials() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Load Kalshi credentials from DB-backed app settings."""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            app_settings = result.scalar_one_or_none()
    except Exception as exc:
        logger.warning(
            "Failed to read Kalshi settings for WS auth",
            error=repr(exc),
        )
        return None, None, None

    if app_settings is None:
        return None, None, None

    api_key = _clean_secret(decrypt_secret(app_settings.kalshi_api_key))
    email = _clean_secret(app_settings.kalshi_email)
    password = _clean_secret(decrypt_secret(app_settings.kalshi_password))
    return api_key, email, password


async def _resolve_kalshi_ws_auth_headers() -> dict[str, str]:
    """Return auth headers for Kalshi WS, or {} when not configured."""
    from services.kalshi_client import kalshi_client

    if kalshi_client.is_authenticated:
        return kalshi_client.get_auth_headers()

    api_key, email, password = await _load_kalshi_stored_credentials()
    if not api_key and not (email and password):
        return {}

    initialized = await kalshi_client.initialize_auth(
        email=email,
        password=password,
        api_key=api_key,
    )
    if not initialized:
        logger.warning("Kalshi WS auth initialization failed; feed will remain stopped")
        return {}

    return kalshi_client.get_auth_headers()


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
    updated_at_epoch: float = 0.0  # wall-clock ingest timestamp (seconds)
    exchange_ts: float = 0.0  # exchange-reported timestamp (seconds)
    sequence: int = 0  # local monotonic cache sequence


@dataclass
class TradeRecord:
    """A single executed trade from the CLOB."""

    price: float
    size: float  # in shares
    side: str  # "BUY" or "SELL"
    timestamp: float  # epoch seconds


class PriceCache:
    """Thread-safe in-memory cache of latest prices and full order books.

    All public methods acquire a lock so the cache can be read safely from
    non-async threads (e.g. a synchronous health-check endpoint).

    Supports registering price change callbacks that fire when a token's
    mid-price moves by more than a configurable threshold (default 0.5%).
    """

    def __init__(
        self,
        stale_ttl: float = DEFAULT_STALE_TTL,
        change_threshold_pct: float = 0.5,
    ) -> None:
        self._stale_ttl = stale_ttl
        self._change_threshold_pct = change_threshold_pct
        self._history_max_snapshots = max(50, int(settings.WS_PRICE_HISTORY_MAX_SNAPSHOTS or 1500))
        self._lock = Lock()
        self._entries: Dict[str, CachedEntry] = {}
        self._history: Dict[str, deque[tuple[float, float, float, float]]] = {}
        self._sequence: int = 0
        self._on_change_callbacks: List[Callable[[str, float, float], None]] = []
        self._on_update_callbacks: List[Callable[[str, float, float, float, float, float, int], None]] = []
        self._trades: Dict[str, deque] = {}
        self._trades_max = 500
        self._on_trade_callbacks: List[Callable] = []

    def add_on_change_callback(self, callback: Callable[[str, float, float], None]) -> None:
        """Register a callback invoked on significant price changes.

        The callback receives ``(token_id, old_mid, new_mid)`` and is called
        outside the lock so it must be thread-safe itself.
        """
        self._on_change_callbacks.append(callback)

    def add_on_update_callback(self, callback: Callable[[str, float, float, float, float, float, int], None]) -> None:
        """Register a callback invoked on every cache replacement.

        The callback receives
        ``(token_id, mid, bid, ask, exchange_ts, ingest_ts, sequence)`` and is called
        outside the lock so it must be thread-safe itself.
        """
        self._on_update_callbacks.append(callback)

    # -- mutations ----------------------------------------------------------

    def update(
        self,
        token_id: str,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        *,
        exchange_ts: float | None = None,
    ) -> None:
        """Atomically replace the order book for *token_id*."""
        bids_sorted = sorted(bids, key=lambda lvl: lvl.price, reverse=True)
        asks_sorted = sorted(asks, key=lambda lvl: lvl.price)

        book = OrderBook(bids=bids_sorted, asks=asks_sorted)
        best_bid = bids_sorted[0].price if bids_sorted else 0.0
        best_ask = asks_sorted[0].price if asks_sorted else 0.0

        new_mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else best_bid or best_ask

        now_mono = time.monotonic()
        now_epoch = time.time()
        old_mid = 0.0
        with self._lock:
            prev = self._entries.get(token_id)
            if prev:
                old_mid = (
                    (prev.best_bid + prev.best_ask) / 2.0
                    if prev.best_bid > 0 and prev.best_ask > 0
                    else prev.best_bid or prev.best_ask
                )
            self._sequence += 1
            sequence = self._sequence
            parsed_exchange_ts = float(exchange_ts) if exchange_ts is not None and exchange_ts > 0 else now_epoch
            entry = CachedEntry(
                best_bid=best_bid,
                best_ask=best_ask,
                order_book=book,
                updated_at=now_mono,
                updated_at_epoch=now_epoch,
                exchange_ts=parsed_exchange_ts,
                sequence=sequence,
            )
            self._entries[token_id] = entry
            if new_mid > 0:
                history = self._history.get(token_id)
                if history is None:
                    history = deque(maxlen=self._history_max_snapshots)
                    self._history[token_id] = history
                history.append((now_epoch, new_mid, best_bid, best_ask))

        # Fire change callbacks outside the lock
        if (
            self._on_change_callbacks
            and old_mid > 0
            and new_mid > 0
            and abs(new_mid - old_mid) / old_mid * 100 >= self._change_threshold_pct
        ):
            for cb in self._on_change_callbacks:
                try:
                    cb(token_id, old_mid, new_mid)
                except Exception:
                    pass

        if self._on_update_callbacks:
            for cb in self._on_update_callbacks:
                try:
                    cb(
                        token_id,
                        new_mid,
                        best_bid,
                        best_ask,
                        entry.exchange_ts,
                        entry.updated_at_epoch,
                        entry.sequence,
                    )
                except Exception:
                    pass

    def remove(self, token_id: str) -> None:
        """Remove a token from the cache."""
        with self._lock:
            self._entries.pop(token_id, None)
            self._history.pop(token_id, None)
            self._trades.pop(token_id, None)

    def add_on_trade_callback(self, callback: Callable) -> None:
        """Register a callback invoked on each trade. Receives (token_id, TradeRecord)."""
        self._on_trade_callbacks.append(callback)

    def record_trade(self, token_id: str, price: float, size: float, side: str, timestamp: float) -> None:
        """Record a trade execution."""
        trade = TradeRecord(price=price, size=size, side=side.upper(), timestamp=timestamp)
        with self._lock:
            if token_id not in self._trades:
                self._trades[token_id] = deque(maxlen=self._trades_max)
            self._trades[token_id].append(trade)
        # Fire callbacks outside the lock
        for cb in self._on_trade_callbacks:
            try:
                cb(token_id, trade)
            except Exception:
                pass

    def get_recent_trades(self, token_id: str, max_trades: int = 100) -> List["TradeRecord"]:
        """Return up to max_trades most recent trades for a token."""
        with self._lock:
            trades = self._trades.get(token_id)
            if not trades:
                return []
            return list(trades)[-max_trades:]

    def get_trade_volume(self, token_id: str, lookback_seconds: float = 300.0) -> Dict[str, float]:
        """Return buy/sell/total volume over the lookback window."""
        cutoff = time.time() - lookback_seconds
        buy_vol = 0.0
        sell_vol = 0.0
        count = 0
        with self._lock:
            trades = self._trades.get(token_id)
            if trades:
                for t in trades:
                    if t.timestamp >= cutoff:
                        if t.side == "BUY":
                            buy_vol += t.size * t.price
                        else:
                            sell_vol += t.size * t.price
                        count += 1
        return {"buy_volume": buy_vol, "sell_volume": sell_vol, "total": buy_vol + sell_vol, "trade_count": count}

    def get_buy_sell_imbalance(self, token_id: str, lookback_seconds: float = 300.0) -> float:
        """Return buy/sell imbalance in [-1, 1]. +1 = all buys, -1 = all sells."""
        vol = self.get_trade_volume(token_id, lookback_seconds)
        total = vol["total"]
        if total <= 0:
            return 0.0
        return (vol["buy_volume"] - vol["sell_volume"]) / total

    def evict_stale_ids(self, max_age_seconds: float = 600.0) -> list[str]:
        """Remove entries not updated within *max_age_seconds*.

        Returns the list of evicted token_ids.  Intended to be called
        periodically (e.g. every 60 s) to prevent unbounded growth when
        markets are subscribed but later expire without an explicit
        ``remove()`` call.
        """
        cutoff = time.monotonic() - max_age_seconds
        to_remove: list[str] = []
        with self._lock:
            for token_id, entry in self._entries.items():
                if entry.updated_at < cutoff:
                    to_remove.append(token_id)
            for token_id in to_remove:
                self._entries.pop(token_id, None)
                self._history.pop(token_id, None)
                self._trades.pop(token_id, None)
        return to_remove

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._entries.clear()
            self._history.clear()

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

    def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
        """Return ``True`` if the cached data is within the staleness TTL."""
        with self._lock:
            entry = self._entries.get(token_id)
        if entry is None or entry.updated_at == 0.0:
            return False
        ttl = self._stale_ttl if max_age_seconds is None else max(0.0, float(max_age_seconds))
        return (time.monotonic() - entry.updated_at) < ttl

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

    def get_price_history(self, token_id: str, max_snapshots: int = 60) -> List[dict]:
        """Return recent price snapshots for a token (newest first)."""
        limit = max(1, int(max_snapshots))
        with self._lock:
            raw = list(self._history.get(token_id, []))
        if not raw:
            return []
        out = [
            {
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "mid": mid,
                "bid": bid,
                "ask": ask,
            }
            for ts, mid, bid, ask in raw[-limit:]
        ]
        out.reverse()
        return out

    def get_price_change(self, token_id: str, lookback_seconds: int = 300) -> Optional[dict]:
        """Return absolute/percent price move over a lookback window."""
        lookback = max(1, int(lookback_seconds))
        cutoff = time.time() - lookback
        with self._lock:
            history = list(self._history.get(token_id, []))
        if len(history) < 2:
            return None
        current_ts, current_mid, _, _ = history[-1]
        if current_mid <= 0:
            return None

        prior = None
        for snap in reversed(history[:-1]):
            if snap[0] <= cutoff and snap[1] > 0:
                prior = snap
                break
        if prior is None:
            for snap in history:
                if snap[1] > 0:
                    prior = snap
                    break
        if prior is None:
            return None

        prior_ts, prior_mid, _, _ = prior
        if prior_mid <= 0:
            return None
        change_abs = current_mid - prior_mid
        change_pct = (change_abs / prior_mid) * 100.0
        return {
            "current_mid": current_mid,
            "prior_mid": prior_mid,
            "change_abs": change_abs,
            "change_pct": change_pct,
            "snapshots_in_window": sum(1 for snap in history if snap[0] >= cutoff),
            "current_timestamp": datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat(),
            "prior_timestamp": datetime.fromtimestamp(prior_ts, tz=timezone.utc).isoformat(),
        }


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
    consecutive_failures: int = 0
    last_failure_at: float = 0.0  # monotonic

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
            "consecutive_failures": self.consecutive_failures,
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
        self._heartbeat_pong_timeout = DEFAULT_HEARTBEAT_PONG_TIMEOUT
        self._heartbeat_max_misses = DEFAULT_HEARTBEAT_MAX_MISSES
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # Callbacks
        self.on_persistent_failure: Optional[Callable[[int], Coroutine]] = None
        self.on_recovery: Optional[Callable[[], Coroutine]] = None

        # Subscription tracking
        self._subscribed_assets: Set[str] = set()  # token_ids
        self._sub_lock = asyncio.Lock()

        # Connection state
        self._ws: Any = None  # websockets connection object
        self._state = ConnectionState.DISCONNECTED
        self._run_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._reconnect_attempt = 0

        # Stats
        self.stats = FeedStats()

    # -- public API ---------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def start(self) -> None:
        """Start the feed in the background.  Idempotent."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start PolymarketWSFeed: websockets library not installed")
            return
        if self._run_task is not None and not self._run_task.done():
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self._run_loop(), name="polymarket-ws-feed")
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

    async def subscribe(self, token_ids: List[str]) -> None:
        """Subscribe to additional assets by token ID.

        Can be called before or after the feed is started.  If the connection
        is already live the subscription message is sent immediately.
        """
        if not token_ids:
            return
        async with self._sub_lock:
            self._subscribed_assets.update(token_ids)
            if self._ws and self._state == ConnectionState.CONNECTED:
                await self._send_subscribe(token_ids)

    async def unsubscribe(self, token_ids: List[str]) -> None:
        """Remove subscriptions.  Cached data for removed tokens is cleared."""
        if not token_ids:
            return
        async with self._sub_lock:
            self._subscribed_assets.difference_update(token_ids)

        for tid in token_ids:
            self._cache.remove(tid)

    # -- internal connection loop -------------------------------------------

    async def _run_loop(self) -> None:
        """Outer loop: connect, listen, and reconnect on failure."""
        self._reconnect_attempt = 0
        while not self._stop_event.is_set():
            try:
                self._state = ConnectionState.CONNECTING if self._reconnect_attempt == 0 else ConnectionState.RECONNECTING
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._reconnect_attempt += 1
                self.stats.reconnections += 1
                self.stats.consecutive_failures = self._reconnect_attempt
                self.stats.last_failure_at = time.monotonic()

                # After max attempts, switch to extended delay
                if self._reconnect_attempt >= MAX_RECONNECT_ATTEMPTS:
                    delay = MAX_RECONNECT_EXTENDED_DELAY
                    if self._reconnect_attempt == MAX_RECONNECT_ATTEMPTS:
                        logger.error(
                            f"Polymarket WS: {MAX_RECONNECT_ATTEMPTS} consecutive failures, "
                            f"extending retry delay to {delay}s"
                        )
                        if self.on_persistent_failure:
                            try:
                                await self.on_persistent_failure(self._reconnect_attempt)
                            except Exception:
                                logger.exception("on_persistent_failure callback error")
                else:
                    delay = min(
                        self._reconnect_base_delay * (DEFAULT_RECONNECT_MULTIPLIER ** (self._reconnect_attempt - 1)),
                        self._reconnect_max_delay,
                    )

                if _is_expected_close(exc):
                    logger.info(
                        "Polymarket WS disconnected cleanly; reconnecting",
                        delay=round(delay, 1),
                        attempt=self._reconnect_attempt,
                        close_code=getattr(exc, "code", None),
                    )
                else:
                    logger.warning(
                        f"Polymarket WS disconnected ({exc!r}), reconnecting in {delay:.1f}s (attempt {self._reconnect_attempt})"
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
                    self._reconnect_attempt += 1
                    self.stats.reconnections += 1
                    delay = min(
                        self._reconnect_base_delay * (DEFAULT_RECONNECT_MULTIPLIER ** (self._reconnect_attempt - 1)),
                        self._reconnect_max_delay,
                    )
                    logger.info(
                        "Polymarket WS server closed connection; reconnecting",
                        delay=round(delay, 1),
                        attempt=self._reconnect_attempt,
                    )
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                        break
                    except asyncio.TimeoutError:
                        pass
                    continue
                break

        self._state = ConnectionState.CLOSED

    async def _connect_and_listen(self) -> None:
        """Establish connection, subscribe, and process messages."""
        async with websockets.connect(
            self._ws_url,
            ping_interval=None,  # we manage heartbeats ourselves
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._state = ConnectionState.CONNECTED
            connect_time = time.monotonic()
            self.stats.connection_uptime_start = connect_time
            reconnect_at_connect = self._reconnect_attempt

            logger.info("Polymarket WS connected", url=self._ws_url, attempt=reconnect_at_connect)

            # Re-subscribe to everything tracked
            async with self._sub_lock:
                if self._subscribed_assets:
                    await self._send_subscribe(list(self._subscribed_assets))

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="polymarket-ws-heartbeat")

            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    recv_time = time.monotonic()
                    self.stats.messages_received += 1
                    self.stats.last_message_at = recv_time

                    # Reset failure counters only after the connection has
                    # been stable for 30s.  Resetting immediately on connect
                    # (as the old code did) prevents backoff from escalating
                    # when the server drops connections every 10-30 seconds.
                    if reconnect_at_connect > 0 and (recv_time - connect_time) >= 30.0:
                        was_failing = self.stats.consecutive_failures >= MAX_RECONNECT_ATTEMPTS
                        self._reconnect_attempt = 0
                        self.stats.consecutive_failures = 0
                        reconnect_at_connect = 0
                        if was_failing and self.on_recovery:
                            try:
                                await self.on_recovery()
                            except Exception:
                                logger.exception("on_recovery callback error")

                    try:
                        data = json.loads(raw)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    self._handle_message(item, recv_time)
                        elif isinstance(data, dict):
                            self._handle_message(data, recv_time)
                        self.stats.messages_parsed += 1
                    except Exception:
                        self.stats.parse_errors += 1
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                self._ws = None

    async def _send_subscribe(self, token_ids: List[str]) -> None:
        """Send a subscribe message over the live WebSocket."""
        if not self._ws or not token_ids:
            return
        msg: dict[str, Any] = {
            "type": "market",
            "assets_ids": token_ids,
        }
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug("Polymarket WS subscribed", assets=len(token_ids))
        except Exception as exc:
            if _is_expected_close(exc):
                logger.info("Polymarket WS subscribe interrupted by clean close")
            else:
                logger.warning(f"Polymarket WS subscribe send failed: {exc!r}")

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Periodically send a ping frame to keep the connection alive."""
        consecutive_misses = 0
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    pong = await ws.ping()
                    start = time.monotonic()
                    await asyncio.wait_for(pong, timeout=self._heartbeat_pong_timeout)
                    self.stats.last_latency_ms = (time.monotonic() - start) * 1000
                    consecutive_misses = 0
                except asyncio.TimeoutError:
                    consecutive_misses += 1
                    logger.warning(
                        "Polymarket WS heartbeat pong timeout",
                        misses=consecutive_misses,
                        threshold=self._heartbeat_max_misses,
                    )
                    if consecutive_misses >= self._heartbeat_max_misses:
                        await ws.close()
                        return
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    # -- message handling ---------------------------------------------------

    def _handle_message(self, data: dict, recv_time: float) -> None:
        """Route an incoming WebSocket message to the right handler."""
        # Trade execution messages (from "trades" channel)
        event_type = data.get("event_type", "")
        if event_type == "trade" or ("price" in data and "size" in data and "side" in data and "bids" not in data):
            self._apply_trade(data, recv_time)
            return

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
        asset_id = data.get("asset_id") or data.get("token_id") or data.get("market", "")
        if not asset_id:
            return

        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])

        bids = self._parse_levels(raw_bids)
        asks = self._parse_levels(raw_asks)

        exchange_ts: float | None = None
        ts = data.get("timestamp")
        if ts is not None:
            try:
                exchange_ts = float(ts)
                if exchange_ts > 1e12:
                    exchange_ts /= 1000.0
            except (TypeError, ValueError):
                exchange_ts = None

        if bids or asks:
            self._cache.update(asset_id, bids, asks, exchange_ts=exchange_ts)

        # Compute server-to-cache latency if a timestamp is present
        if exchange_ts is not None:
            try:
                self.stats.last_latency_ms = (recv_time - exchange_ts) * 1000
            except Exception:
                pass

    def _apply_trade(self, data: dict, recv_time: float) -> None:
        """Parse a trade execution message and record it."""
        asset_id = data.get("asset_id") or data.get("token_id") or data.get("market", "")
        if not asset_id:
            return
        try:
            price = float(data.get("price", 0))
            size = float(data.get("size", 0))
            side = str(data.get("side", "")).upper()
            if side not in ("BUY", "SELL"):
                side = "BUY"  # default
            ts = data.get("timestamp")
            if ts is not None:
                timestamp = float(ts)
                if timestamp > 1e12:
                    timestamp /= 1000.0  # ms -> s
            else:
                timestamp = recv_time
            if price > 0 and size > 0:
                self._cache.record_trade(asset_id, price, size, side, timestamp)
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
        self._auth_headers: dict[str, str] = {}

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
        self._auth_headers = await self._load_auth_headers()
        if not self._auth_headers:
            self._state = ConnectionState.CLOSED
            logger.info("KalshiWSFeed not started: credentials are not configured in app settings")
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self._run_loop(), name="kalshi-ws-feed")
        logger.info("KalshiWSFeed started")

    async def _load_auth_headers(self) -> dict[str, str]:
        """Resolve auth headers for Kalshi WebSocket connection."""
        return await _resolve_kalshi_ws_auth_headers()

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
                self._state = ConnectionState.CONNECTING if attempt == 0 else ConnectionState.RECONNECTING
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break

                # Stop retrying on auth failures — they won't resolve
                # without new credentials.
                exc_str = repr(exc)
                if "401" in exc_str or "403" in exc_str:
                    logger.warning(
                        f"Kalshi WS auth failure ({exc!r}), stopping reconnect (credentials missing or expired)"
                    )
                    break

                attempt += 1
                self.stats.reconnections += 1
                delay = min(
                    self._reconnect_base_delay * (DEFAULT_RECONNECT_MULTIPLIER ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                if _is_expected_close(exc):
                    logger.info(
                        "Kalshi WS disconnected cleanly; reconnecting",
                        delay=round(delay, 1),
                        attempt=attempt,
                        close_code=getattr(exc, "code", None),
                    )
                else:
                    logger.warning(
                        f"Kalshi WS disconnected ({exc!r}), reconnecting in {delay:.1f}s (attempt {attempt})"
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
        extra_headers.update(self._auth_headers)
        async with websockets.connect(
            self._ws_url,
            ping_interval=None,
            open_timeout=10,
            close_timeout=5,
            extra_headers=extra_headers,
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
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="kalshi-ws-heartbeat")

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
                    except Exception:
                        self.stats.parse_errors += 1
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
            OrderBookLevel(price=round(1.0 - lvl.price, 4), size=lvl.size) for lvl in no_asks if lvl.price < 1.0
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
        await mgr.polymarket_feed.subscribe(["123..."])
        price = await mgr.get_price("123...")
        book  = await mgr.get_order_book("123...")
    """

    _instance: Optional["FeedManager"] = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._cache = PriceCache()
        self._polymarket_feed = PolymarketWSFeed(cache=self._cache)
        self._kalshi_feed = KalshiWSFeed(cache=self._cache)
        self._http_fallback_fn: Optional[Callable[[str], Coroutine[Any, Any, Optional[OrderBook]]]] = None
        self._started = False
        # Reactive scan: accumulate changed tokens and debounce trigger
        self._changed_tokens: Set[str] = set()
        self._reactive_scan_callback: Optional[Callable[[Set[str]], Coroutine[Any, Any, None]]] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._debounce_seconds: float = 2.0  # coalesce changes over 2s window
        self._redis_price_updates: Dict[
            str,
            tuple[float | None, float | None, float | None, float | None, float | None, int | None],
        ] = {}
        self._redis_update_lock = asyncio.Lock()
        self._redis_flush_task: Optional[asyncio.Task] = None
        self._redis_flush_interval_seconds: float = max(0.01, float(settings.WS_REDIS_FLUSH_INTERVAL_SECONDS))
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._eviction_task: Optional[asyncio.Task] = None
        self._cache.add_on_change_callback(self._on_price_change)
        self._cache.add_on_trade_callback(self._on_trade)
        self._cache.add_on_update_callback(self._on_price_update)

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
        loop = asyncio.get_running_loop()
        self._eviction_task = loop.create_task(self._cache_eviction_loop())
        logger.info(
            "FeedManager started",
            polymarket_ws=True,
            kalshi_state=self._kalshi_feed.state.value,
        )

    async def stop(self) -> None:
        """Stop both feeds and clear cache."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._redis_flush_task and not self._redis_flush_task.done():
            self._redis_flush_task.cancel()
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
        await self._polymarket_feed.stop()
        await self._kalshi_feed.stop()
        async with self._redis_update_lock:
            self._redis_price_updates = {}
        self._cache.clear()
        self._started = False
        logger.info("FeedManager stopped")

    # -- reactive scanning --------------------------------------------------

    def set_reactive_scan_callback(
        self,
        callback: Callable[[Set[str]], Coroutine[Any, Any, None]],
        debounce_seconds: float = 2.0,
    ) -> None:
        """Register an async callback to trigger when prices change significantly.

        The callback is debounced: rapid price changes within *debounce_seconds*
        are coalesced into a single invocation.
        """
        self._reactive_scan_callback = callback
        self._debounce_seconds = debounce_seconds

    def _on_price_change(self, token_id: str, old_mid: float, new_mid: float) -> None:
        """Called by PriceCache when a significant price change occurs."""
        if self._reactive_scan_callback:
            self._changed_tokens.add(token_id)
            # Schedule a debounced trigger on the event loop
            try:
                loop = asyncio.get_running_loop()
                if self._debounce_task is None or self._debounce_task.done():
                    self._debounce_task = loop.create_task(self._debounced_trigger())
            except RuntimeError:
                pass  # no running event loop (e.g. during shutdown)

        # Dispatch DataEvent for event-driven strategies
        try:
            from services.data_events import DataEvent, EventType
            from services.event_dispatcher import event_dispatcher
            from utils.utcnow import utcnow

            event = DataEvent(
                event_type=EventType.PRICE_CHANGE,
                source="polymarket_ws",
                timestamp=utcnow(),
                token_id=token_id,
                old_price=old_mid,
                new_price=new_mid,
            )
            loop = asyncio.get_running_loop()
            task = loop.create_task(event_dispatcher.dispatch(event))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)
        except RuntimeError:
            pass  # No running event loop

    def _on_trade(self, token_id: str, trade: "TradeRecord") -> None:
        """Dispatch a TRADE_EXECUTION DataEvent for each trade."""
        try:
            from services.data_events import DataEvent, EventType
            from services.event_dispatcher import event_dispatcher
            from utils.utcnow import utcnow

            event = DataEvent(
                event_type=EventType.TRADE_EXECUTION,
                source="polymarket_ws",
                timestamp=utcnow(),
                token_id=token_id,
                payload={
                    "price": trade.price,
                    "size": trade.size,
                    "side": trade.side,
                    "timestamp": trade.timestamp,
                },
            )
            loop = asyncio.get_running_loop()
            task = loop.create_task(event_dispatcher.dispatch(event))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)
        except RuntimeError:
            pass  # No running event loop

    def _on_price_update(
        self,
        token_id: str,
        mid: float,
        bid: float,
        ask: float,
        exchange_ts: float | None = None,
        ingest_ts: float | None = None,
        sequence: int | None = None,
    ) -> None:
        """Buffer all price updates for periodic Redis persistence."""
        if mid <= 0 or bid < 0 or ask < 0:
            return
        ingest = float(ingest_ts) if ingest_ts is not None and ingest_ts > 0 else float(time.time())
        exchange = float(exchange_ts) if exchange_ts is not None and exchange_ts > 0 else ingest
        seq = int(sequence) if sequence is not None and int(sequence) > 0 else 0
        self._redis_price_updates[token_id] = (
            float(mid),
            float(bid),
            float(ask),
            exchange,
            ingest,
            seq,
        )
        if self._redis_flush_task is None or self._redis_flush_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._redis_flush_task = loop.create_task(self._flush_redis_price_updates())
            except RuntimeError:
                pass

    async def _flush_redis_price_updates(self) -> None:
        """Batch in-memory WS updates into Redis snapshots."""
        try:
            while True:
                await asyncio.sleep(self._redis_flush_interval_seconds)
                updates: dict[
                    str,
                    tuple[float | None, float | None, float | None, float | None, float | None, int | None],
                ] = {}
                async with self._redis_update_lock:
                    if not self._redis_price_updates:
                        self._redis_flush_task = None
                        return
                    updates = dict(self._redis_price_updates)
                    self._redis_price_updates = {}
                try:
                    await redis_price_cache.write_prices(updates)
                except Exception as exc:
                    async with self._redis_update_lock:
                        self._redis_price_updates = {**updates, **self._redis_price_updates}
                    logger.warning(
                        "Redis live price flush failed",
                        error=str(exc),
                        token_count=len(updates),
                    )
        except asyncio.CancelledError:
            return

    async def _cache_eviction_loop(self) -> None:
        """Periodically evict stale entries from the price cache and prune
        feed subscriptions for tokens that were evicted."""
        try:
            while True:
                await asyncio.sleep(60.0)
                try:
                    stale_ids = self._cache.evict_stale_ids(max_age_seconds=600.0)
                    if stale_ids:
                        # Prune feed subscription sets so they don't grow
                        # unbounded and the reconnect re-subscribe message
                        # stays lean.
                        async with self._polymarket_feed._sub_lock:
                            self._polymarket_feed._subscribed_assets.difference_update(stale_ids)
                        logger.info("PriceCache evicted stale entries", evicted=len(stale_ids))
                except Exception as exc:
                    logger.warning("PriceCache eviction failed", exc_info=exc)
        except asyncio.CancelledError:
            return

    async def _debounced_trigger(self) -> None:
        """Wait for the debounce window, then fire the reactive scan callback."""
        await asyncio.sleep(self._debounce_seconds)
        if not self._reactive_scan_callback or not self._changed_tokens:
            return
        changed_tokens = set(self._changed_tokens)
        changed_count = len(changed_tokens)
        self._changed_tokens.clear()
        try:
            logger.debug(f"Reactive scan triggered by {changed_count} price changes")
            await self._reactive_scan_callback(changed_tokens)
        except Exception as exc:
            logger.warning(f"Reactive scan callback failed: {exc!r}")

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

    def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
        """Check whether cached data for *token_id* is within TTL."""
        return self._cache.is_fresh(token_id, max_age_seconds=max_age_seconds)

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
                "pending_reactive_tokens": len(self._changed_tokens),
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
