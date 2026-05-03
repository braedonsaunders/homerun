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
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from sqlalchemy import select

from config import settings
from models.database import AppSettings, AsyncSessionLocal
from services.optimization.vwap import OrderBook, OrderBookLevel
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
        # None means no close frame was received (raw TCP drop) — this is
        # normal for WS servers that disconnect idle/overloaded clients.
        return close_code is None or close_code in {1000, 1001, 1005}
    return False


def _ws_state_is_closed(ws: Any) -> bool:
    """Best-effort check that a websockets connection is no longer open.

    websockets >= 11 exposes ``state`` (an enum of OPEN/CLOSING/CLOSED);
    older versions expose ``closed``.  Either signal that ``send()``
    will raise, so callers can skip the attempt entirely.
    """
    state = getattr(ws, "state", None)
    if state is not None:
        name = getattr(state, "name", str(state)).upper()
        return name in {"CLOSING", "CLOSED"}
    return bool(getattr(ws, "closed", False))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLYMARKET_WS_URL = settings.CLOB_WS_URL
KALSHI_WS_URL = settings.KALSHI_WS_URL

DEFAULT_STALE_TTL = float(settings.WS_PRICE_STALE_SECONDS)
DEFAULT_HEARTBEAT_INTERVAL = max(1.0, float(settings.WS_HEARTBEAT_INTERVAL))  # seconds between keep-alive pings
DEFAULT_HEARTBEAT_PONG_TIMEOUT = max(5.0, float(settings.WS_HEARTBEAT_PONG_TIMEOUT))
DEFAULT_HEARTBEAT_MAX_MISSES = max(1, int(settings.WS_HEARTBEAT_MAX_MISSES))
DEFAULT_RECONNECT_BASE_DELAY = 1.0  # initial backoff delay in seconds
DEFAULT_RECONNECT_MAX_DELAY = 60.0  # maximum backoff ceiling
DEFAULT_RECONNECT_MULTIPLIER = 2.0  # exponential multiplier per attempt
MAX_RECONNECT_ATTEMPTS = 30  # after this many consecutive failures, extend delay
MAX_RECONNECT_EXTENDED_DELAY = 60.0  # 1 minute max between attempts (was 5min — too slow for network recovery)


# ---------------------------------------------------------------------------
# Kalshi auth helpers
# ---------------------------------------------------------------------------


def _normalize_epoch_timestamp(value: Any) -> float | None:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0.0:
        return None
    if timestamp > 1e12:
        timestamp /= 1000.0
    return timestamp


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
# Mid-price sanity guards for binary 0-1 contracts (Polymarket / Kalshi YES/NO)
# ---------------------------------------------------------------------------

# Prices outside this band are degenerate prints (sub-penny ladders, near-edge
# crosses). Common during cycle resets on up/down markets and on illiquid books.
_BINARY_PRICE_LO = 0.005
_BINARY_PRICE_HI = 0.995
# One-sided fallback only fires inside this narrower band.  See
# ``_safe_binary_mid`` for the rationale; in practice this prevents a
# lone-ask near 0.99 (the F1 Ocon failure mode on 2026-04-28) or a
# lone-bid near 0.01 from being treated as the mid.
_ONE_SIDED_NON_EXTREME_LO = 0.10
_ONE_SIDED_NON_EXTREME_HI = 0.90

# A spread wider than 50¢ on a 0-1 binary contract means there is no real
# two-sided market — quoting a mid from it produces noise that fires spurious
# exit-eval callbacks downstream.
_BINARY_MAX_SPREAD = 0.50


def _safe_binary_mid(best_bid: float, best_ask: float) -> Optional[float]:
    """Compute a guarded mid for a 0-1 binary contract.

    Returns ``None`` for degenerate books.  When BOTH sides are
    in-range, the spread is bounded, and the average is returned as
    the mid.

    When only ONE side is in-range, fallback to that side is allowed
    ONLY when it is in the non-extreme middle band
    ``(_ONE_SIDED_NON_EXTREME_LO, _ONE_SIDED_NON_EXTREME_HI)``.  The
    rationale is that a binary book with a single quote near 0.99 (or
    near 0.01) is overwhelmingly likely to be a stale limit / thin
    liquidity provider quote rather than a real market signal.  The
    2026-04-28 incident on the F1 Ocon market saw the YES token's
    orderbook collapse to a single ask at 0.99 while the real
    midpoint was ~0.495; falling back to that ask as the mark
    produced a phantom +1314% unrealized P&L that would have
    triggered take-profit immediately had the SELL retry path been
    healthy.

    Returning ``None`` here forces upstream callers to either skip the
    mark update or fall back to a different source (last trade price,
    cached outcome_prices from the catalog, etc.) — the institutional
    correct behaviour for "I cannot derive a trustworthy mark from
    this orderbook".
    """
    valid_bid = best_bid > _BINARY_PRICE_LO and best_bid < _BINARY_PRICE_HI
    valid_ask = best_ask > _BINARY_PRICE_LO and best_ask < _BINARY_PRICE_HI
    if valid_bid and valid_ask:
        if best_ask - best_bid > _BINARY_MAX_SPREAD:
            return None
        return (best_bid + best_ask) / 2.0
    # One-sided fallback: only trust the surviving side when it is in
    # the non-extreme band.  Outside that band the lone quote is more
    # likely a stale order than a meaningful price.
    if valid_bid and _ONE_SIDED_NON_EXTREME_LO < best_bid < _ONE_SIDED_NON_EXTREME_HI:
        return best_bid
    if valid_ask and _ONE_SIDED_NON_EXTREME_LO < best_ask < _ONE_SIDED_NON_EXTREME_HI:
        return best_ask
    return None


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
        # No lock needed — all access is from the single-threaded asyncio
        # event loop.  A threading.Lock was previously here but it added
        # unnecessary overhead and is semantically incorrect for async code.
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

        # Guarded mid: returns None for degenerate books (extreme prints,
        # >50¢ spread). When None, we still cache best_bid/best_ask but skip
        # the history append and downstream change/update callbacks so
        # exit-eval and other consumers don't act on noise.
        safe_new_mid = _safe_binary_mid(best_bid, best_ask)

        now_mono = time.monotonic()
        now_epoch = time.time()
        prev = self._entries.get(token_id)
        safe_old_mid: Optional[float] = (
            _safe_binary_mid(prev.best_bid, prev.best_ask) if prev else None
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
        if safe_new_mid is not None:
            history = self._history.get(token_id)
            if history is None:
                history = deque(maxlen=self._history_max_snapshots)
                self._history[token_id] = history
            history.append((now_epoch, safe_new_mid, best_bid, best_ask))

        # Fire change callbacks outside the lock. Only fires on guarded mids
        # so a degenerate print can't trigger a spurious exit-eval.
        if (
            self._on_change_callbacks
            and safe_old_mid is not None
            and safe_old_mid > 0
            and safe_new_mid is not None
            and abs(safe_new_mid - safe_old_mid) / safe_old_mid * 100 >= self._change_threshold_pct
        ):
            for cb in self._on_change_callbacks:
                try:
                    cb(token_id, safe_old_mid, safe_new_mid)
                except Exception:
                    pass

        # Update callbacks fire only with a non-degenerate mid for the same
        # reason. Consumers that need raw bid/ask without the guard should
        # read them off the cache directly.
        if self._on_update_callbacks and safe_new_mid is not None:
            for cb in self._on_update_callbacks:
                try:
                    cb(
                        token_id,
                        safe_new_mid,
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
        self._entries.pop(token_id, None)
        self._history.pop(token_id, None)
        self._trades.pop(token_id, None)

    def add_on_trade_callback(self, callback: Callable) -> None:
        """Register a callback invoked on each trade. Receives (token_id, TradeRecord)."""
        self._on_trade_callbacks.append(callback)

    def record_trade(self, token_id: str, price: float, size: float, side: str, timestamp: float) -> None:
        """Record a trade execution."""
        trade = TradeRecord(price=price, size=size, side=side.upper(), timestamp=timestamp)
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
        self._entries.clear()
        self._history.clear()

    # -- queries ------------------------------------------------------------

    def get_mid_price(self, token_id: str) -> Optional[float]:
        """Return mid price or ``None`` if not cached or the book is degenerate.

        A degenerate book is one where the spread exceeds 50¢ on a 0-1 binary
        contract or both sides are at the extreme edges (sub-half-cent /
        99.5¢+). See ``_safe_binary_mid``.
        """
        entry = self._entries.get(token_id)
        if entry is None:
            return None
        return _safe_binary_mid(entry.best_bid, entry.best_ask)

    def get_spread(self, token_id: str) -> Optional[float]:
        """Return absolute bid-ask spread or ``None``."""
        entry = self._entries.get(token_id)
        if entry is None or entry.best_bid == 0.0 or entry.best_ask == 0.0:
            return None
        return entry.best_ask - entry.best_bid

    def get_spread_bps(self, token_id: str) -> Optional[float]:
        """Return bid-ask spread in basis points or ``None``."""
        entry = self._entries.get(token_id)
        if entry is None or entry.best_bid == 0.0 or entry.best_ask == 0.0:
            return None
        mid = (entry.best_bid + entry.best_ask) / 2.0
        if mid <= 0.0:
            return None
        return ((entry.best_ask - entry.best_bid) / mid) * 10_000

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Return the full ``OrderBook`` or ``None``."""
        entry = self._entries.get(token_id)
        if entry is None:
            return None
        return entry.order_book

    def get_best_bid_ask(self, token_id: str) -> Optional[tuple[float, float]]:
        """Return ``(best_bid, best_ask)`` or ``None``."""
        entry = self._entries.get(token_id)
        if entry is None:
            return None
        return (entry.best_bid, entry.best_ask)

    def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
        """Return ``True`` if the cached data is within the staleness TTL."""
        entry = self._entries.get(token_id)
        if entry is None or entry.updated_at == 0.0:
            return False
        ttl = self._stale_ttl if max_age_seconds is None else max(0.0, float(max_age_seconds))
        return (time.monotonic() - entry.updated_at) < ttl

    def get_observed_at_epoch(self, token_id: str) -> Optional[float]:
        """Return wall-clock ingest timestamp in epoch seconds, or ``None``."""
        entry = self._entries.get(token_id)
        if entry is None or entry.updated_at_epoch <= 0.0:
            return None
        return float(entry.updated_at_epoch)

    def get_sequence(self, token_id: str) -> Optional[int]:
        """Return local cache update sequence, or ``None`` when unavailable."""
        entry = self._entries.get(token_id)
        if entry is None or entry.sequence <= 0:
            return None
        return int(entry.sequence)

    def staleness(self, token_id: str) -> Optional[float]:
        """Return age in seconds of the cached data, or ``None``."""
        entry = self._entries.get(token_id)
        if entry is None or entry.updated_at == 0.0:
            return None
        return time.monotonic() - entry.updated_at

    def touch_all(self) -> None:
        """Refresh ``updated_at`` on every cached entry without changing prices.

        Called when the WS connection is confirmed alive (heartbeat pong
        received).  For quiet markets the order book doesn't change so no WS
        messages arrive, but the cached price is still valid — it just hasn't
        moved.  Touching resets the staleness timer so the execution gate
        doesn't reject unchanged but still-accurate prices.
        """
        now_mono = time.monotonic()
        for entry in self._entries.values():
            entry.updated_at = now_mono

    def all_token_ids(self) -> List[str]:
        """Return a snapshot of all cached token IDs."""
        return list(self._entries.keys())

    def get_price_history(self, token_id: str, max_snapshots: int = 60) -> List[dict]:
        """Return recent price snapshots for a token (newest first)."""
        limit = max(1, int(max_snapshots))
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

    def is_subscribed(self, token_id: str) -> bool:
        normalized = str(token_id or "").strip().lower()
        if not normalized:
            return False
        return normalized in self._subscribed_assets or str(token_id or "").strip() in self._subscribed_assets

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
            new_ids = [tid for tid in token_ids if tid not in self._subscribed_assets]
            self._subscribed_assets.update(token_ids)
            should_send = bool(new_ids) and self._ws and self._state == ConnectionState.CONNECTED
        if should_send:
            _SUBSCRIBE_CHUNK = 100
            for i in range(0, len(new_ids), _SUBSCRIBE_CHUNK):
                chunk = new_ids[i : i + _SUBSCRIBE_CHUNK]
                await self._send_subscribe(chunk)
                if i + _SUBSCRIBE_CHUNK < len(new_ids):
                    await asyncio.sleep(0.05)

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
            ping_timeout=None,
            open_timeout=10,
            close_timeout=5,
            max_size=2**22,  # 4 MB — large book snapshots can exceed 1 MB default
        ) as ws:
            self._ws = ws
            self._state = ConnectionState.CONNECTED
            connect_time = time.monotonic()
            self.stats.connection_uptime_start = connect_time
            reconnect_at_connect = self._reconnect_attempt

            logger.info("Polymarket WS connected", url=self._ws_url, attempt=reconnect_at_connect)
            if reconnect_at_connect > 0:
                self._cache.clear()

            # Re-subscribe to everything tracked, in chunks to avoid
            # overwhelming the server with a single massive subscribe message.
            async with self._sub_lock:
                if self._subscribed_assets:
                    all_ids = list(self._subscribed_assets)
                    _SUBSCRIBE_CHUNK = 100
                    for i in range(0, len(all_ids), _SUBSCRIBE_CHUNK):
                        chunk = all_ids[i : i + _SUBSCRIBE_CHUNK]
                        await self._send_subscribe(chunk)
                        if i + _SUBSCRIBE_CHUNK < len(all_ids):
                            await asyncio.sleep(0.05)

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="polymarket-ws-heartbeat")

            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    recv_time_mono = time.monotonic()
                    recv_time_epoch = time.time()
                    self.stats.messages_received += 1
                    self.stats.last_message_at = recv_time_mono

                    # Reset failure counters after the connection has been
                    # stable for 10s.  This is long enough to confirm the
                    # connection is healthy, but short enough that backoff
                    # resets reasonably fast after transient outages (the
                    # previous 30s threshold meant backoff never reset when
                    # the server dropped connections every 20-30 seconds).
                    if reconnect_at_connect > 0 and (recv_time_mono - connect_time) >= 10.0:
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
                                    self._handle_message(item, recv_time_epoch)
                        elif isinstance(data, dict):
                            self._handle_message(data, recv_time_epoch)
                        self.stats.messages_parsed += 1
                    except Exception:
                        self.stats.parse_errors += 1
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                self._ws = None

    async def _send_subscribe(self, token_ids: List[str]) -> None:
        """Send a subscribe message over the live WebSocket.

        Captures a local reference to ``self._ws`` so a concurrent
        reconnect that swaps ``self._ws`` mid-call cannot turn the send
        into an attribute-on-None crash.  Also checks ``ws.state`` /
        ``ws.closed`` first so a NORMAL_CLOSURE that arrived between
        scheduling this coroutine and running it is treated as an
        expected lifecycle event, not a warning.
        """
        if not token_ids:
            return
        ws = self._ws
        if ws is None:
            return
        if getattr(ws, "closed", False) or _ws_state_is_closed(ws):
            logger.debug("Polymarket WS subscribe skipped — socket already closed")
            return
        msg: dict[str, Any] = {
            "type": "market",
            "assets_ids": token_ids,
        }
        try:
            await ws.send(json.dumps(msg))
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
                    # Connection confirmed alive — touch cached prices so quiet
                    # markets don't go stale while the feed is healthy.
                    self._cache.touch_all()
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

    def _handle_message(self, data: dict, recv_time_epoch: float) -> None:
        """Route an incoming WebSocket message to the right handler."""
        # Trade execution messages (from "trades" channel)
        event_type = data.get("event_type", "")
        if event_type == "trade" or ("price" in data and "size" in data and "side" in data and "bids" not in data):
            self._apply_trade(data, recv_time_epoch)
            return

        # Polymarket book messages typically carry an "event_type" or can be
        # identified by the presence of bids/asks at the top level or nested
        # inside a "market" key.  We handle the common shapes:
        #   1) {"market": "<condition_id>", "asset_id": "...", "bids": [...], "asks": [...]}
        #   2) {"type": "book", "data": {...}} wrapper with nested payload
        #   3) {"data": [...]} array of updates

        # If bids/asks are directly on the top-level dict, apply immediately
        if "bids" in data or "asks" in data:
            self._apply_book_update(data, recv_time_epoch)
            return

        # Nested payload under "data" key (common Polymarket wrapper pattern)
        nested = data.get("data")
        if isinstance(nested, dict):
            self._apply_book_update(nested, recv_time_epoch)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    self._apply_book_update(item, recv_time_epoch)
        # Silently ignore heartbeat acks, subscribe confirmations, etc.

    def _apply_book_update(self, data: dict, recv_time_epoch: float) -> None:
        """Parse bids/asks arrays and push into PriceCache."""
        asset_id = data.get("asset_id") or data.get("token_id") or data.get("market", "")
        if not asset_id:
            return

        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])

        bids = self._parse_levels(raw_bids)
        asks = self._parse_levels(raw_asks)

        exchange_ts = _normalize_epoch_timestamp(data.get("timestamp"))

        if bids or asks:
            self._cache.update(asset_id, bids, asks, exchange_ts=exchange_ts)

        # Crypto latency harness: record book-update wire arrival per
        # token.  Polymarket's CLOB book is one of the inputs the
        # crypto strategies consume (entry-price VWAP); tracking it
        # alongside the Binance/Chainlink feeds gives the harness a
        # full picture of which input was freshest at decision time.
        try:
            from services.crypto_latency_trace import record_wire_event

            record_wire_event(
                "polymarket",
                str(asset_id),
                int(recv_time_epoch * 1000.0),
            )
        except Exception:
            pass

        # Compute server-to-cache latency if a timestamp is present
        if exchange_ts is not None:
            try:
                self.stats.last_latency_ms = max(0.0, (recv_time_epoch - exchange_ts) * 1000.0)
            except Exception:
                pass

    def _apply_trade(self, data: dict, recv_time_epoch: float) -> None:
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
            timestamp = _normalize_epoch_timestamp(data.get("timestamp"))
            if timestamp is None:
                timestamp = recv_time_epoch
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
            ping_timeout=None,
            open_timeout=10,
            close_timeout=5,
            max_size=2**22,
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
                    recv_time_mono = time.monotonic()
                    recv_time_epoch = time.time()
                    self.stats.messages_received += 1
                    self.stats.last_message_at = recv_time_mono
                    try:
                        data = json.loads(raw)
                        self._handle_message(data, recv_time_epoch)
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

    def _handle_message(self, data: dict, recv_time_epoch: float) -> None:
        """Route incoming Kalshi messages."""
        # Kalshi pushes channel data under a "msg" or "data" key, and can
        # also have a "type" field.  Common shapes:
        #   {"id": ..., "type": "orderbook_delta", "msg": {"market_ticker": ..., "yes": [...], "no": [...]}}
        #   {"type": "orderbook_snapshot", "msg": {...}}
        #   {"sid": ..., "type": "orderbook_delta", "msg": {...}}

        msg_type = data.get("type", "")
        payload = data.get("msg") or data.get("data") or data

        if "orderbook" in msg_type:
            self._apply_orderbook(payload, recv_time_epoch)
        elif isinstance(payload, dict) and "market_ticker" in payload:
            self._apply_orderbook(payload, recv_time_epoch)

    def _apply_orderbook(self, payload: dict, recv_time_epoch: float) -> None:
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

        exchange_ts = _normalize_epoch_timestamp(payload.get("timestamp") or payload.get("ts"))
        self._cache.update(ticker, implied_bids, yes_asks, exchange_ts=exchange_ts)
        if exchange_ts is not None:
            self.stats.last_latency_ms = max(0.0, (recv_time_epoch - exchange_ts) * 1000.0)

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


async def _build_polymarket_http_fallback_order_book(token_id: str) -> Optional[OrderBook]:
    normalized = str(token_id or "").strip()
    if not normalized:
        return None
    try:
        from services.polymarket import polymarket_client
    except Exception:
        return None
    try:
        payload = await polymarket_client.get_order_book(normalized)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    def _parse_levels(raw_levels: Any, *, reverse: bool) -> List[OrderBookLevel]:
        levels: List[OrderBookLevel] = []
        if not isinstance(raw_levels, list):
            return levels
        for raw_level in raw_levels:
            if not isinstance(raw_level, dict):
                continue
            try:
                price = float(raw_level.get("price", 0.0))
                size = float(raw_level.get("size", 0.0))
            except (TypeError, ValueError):
                continue
            if price <= 0.0 or size <= 0.0:
                continue
            levels.append(OrderBookLevel(price=price, size=size))
        levels.sort(key=lambda level: level.price, reverse=reverse)
        return levels

    bids = _parse_levels(payload.get("bids"), reverse=True)
    asks = _parse_levels(payload.get("asks"), reverse=False)
    if not bids and not asks:
        return None
    return OrderBook(bids=bids, asks=asks)


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

    def __init__(self) -> None:
        self._cache = PriceCache()
        self._polymarket_feed = PolymarketWSFeed(cache=self._cache)
        self._kalshi_feed = KalshiWSFeed(cache=self._cache)
        # User-channel feed (wallet's own orders/fills).  Only starts
        # when ``configure_user_feed_credentials`` has been called with
        # valid Polymarket API credentials — see ``start()`` below.
        # The feed feeds ``WalletStateCache`` so the orchestrator can
        # read wallet state without REST polls on the hot path.
        from services.polymarket_user_feed import get_polymarket_user_feed
        self._polymarket_user_feed = get_polymarket_user_feed()
        self._user_feed_credentials_set: bool = False
        self._http_fallback_fn: Optional[Callable[[str], Coroutine[Any, Any, Optional[OrderBook]]]] = None
        self._started = False
        self._start_lock = asyncio.Lock()
        # Reactive scan: accumulate changed tokens and debounce trigger
        self._changed_tokens: Set[str] = set()
        self._reactive_scan_callback: Optional[Callable[[Set[str]], Coroutine[Any, Any, None]]] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._debounce_seconds: float = 2.0  # coalesce changes over 2s window
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._eviction_task: Optional[asyncio.Task] = None
        # -- WS price push state -----------------------------------------------
        # Registered frontends: ws_id -> {token_ids: set, callback: async callable}
        # No lock needed — all access from single-threaded asyncio event loop.
        self._ws_push_clients: Dict[int, Dict[str, Any]] = {}
        # token_id -> set of ws_ids wanting that token's prices
        self._token_to_ws_clients: Dict[str, Set[int]] = {}
        # Coalesced price updates pending push: token_id -> price dict
        self._pending_price_updates: Dict[str, Dict[str, Any]] = {}
        self._price_push_task: Optional[asyncio.Task] = None
        self._coalesce_ms: float = max(10, int(settings.WS_PRICE_PUSH_COALESCE_MS or 100))
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # captured in start()
        self._cache.add_on_change_callback(self._on_price_change)
        self._cache.add_on_trade_callback(self._on_trade)
        self._cache.add_on_update_callback(self._on_price_update)

    @classmethod
    def get_instance(cls) -> "FeedManager":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tear down the singleton (useful in tests)."""
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

    @property
    def polymarket_user_feed(self):
        """The wallet's own-order WS feed (``services.polymarket_user_feed``).

        May be inactive if credentials weren't configured at startup.
        """
        return self._polymarket_user_feed

    def configure_user_feed_credentials(
        self,
        *,
        api_key: str | None,
        api_secret: str | None,
        api_passphrase: str | None,
        wallet_address: str | None,
    ) -> bool:
        """Set the credentials the user-channel WS will use.

        Called by ``live_execution_service`` once it has resolved the
        wallet's API key trio.  Returns True if credentials are
        complete; False if anything is missing (in which case the
        user feed will refuse to start, the trading plane will be
        unable to read wallet state from WS, and the freshness gate
        will keep the orchestrator from trading on stale REST state).

        If ``FeedManager`` has already started (race: live_execution_service
        finishes initializing AFTER feed_manager.start), kicks off the
        user feed in the background here.
        """
        if not (api_key and api_secret and api_passphrase):
            self._user_feed_credentials_set = False
            return False
        self._polymarket_user_feed.configure_credentials(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            wallet_address=wallet_address,
        )
        self._user_feed_credentials_set = True
        # Late-arriving credentials: start the user feed now if the
        # rest of FeedManager is already up.
        if self._started and not self._polymarket_user_feed.is_running:
            try:
                loop = self._loop or asyncio.get_event_loop()
                loop.create_task(
                    self._polymarket_user_feed.start(),
                    name="polymarket-user-feed-late-start",
                )
            except Exception as exc:
                logger.warning("Polymarket user feed late start failed", exc_info=exc)
        return True

    async def ensure_user_subscribed(self, condition_ids: list[str]) -> None:
        """Subscribe the user feed to additional markets.

        **Architectural mandate**: strategies call this BEFORE emitting
        signals into the runtime — it guarantees that by the time the
        orchestrator picks up a signal, the wallet's WS subscription
        is live for that market and any subsequent fills will flow
        into ``WalletStateCache``.
        """
        if not condition_ids:
            return
        await self._polymarket_user_feed.subscribe_conditions(condition_ids)

    # -- lifecycle ----------------------------------------------------------

    def set_http_fallback(
        self,
        fn: Callable[[str], Coroutine[Any, Any, Optional[OrderBook]]],
    ) -> None:
        """Register an async callback ``fn(token_id) -> OrderBook | None``
        to be used when WebSocket data is stale."""
        self._http_fallback_fn = fn

    async def start(self) -> None:
        """Start all feeds (market + kalshi + user channel).  Idempotent."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._http_fallback_fn is None:
                self._http_fallback_fn = _build_polymarket_http_fallback_order_book
            await self._polymarket_feed.start()
            await self._kalshi_feed.start()
            # Start the user-channel feed only if credentials were
            # configured.  Missing creds is the dev/shadow-mode case
            # — the user feed is skipped and the freshness gate will
            # block live trading until credentials arrive.
            if self._user_feed_credentials_set:
                try:
                    await self._polymarket_user_feed.start()
                except Exception as exc:
                    logger.warning("Polymarket user feed start failed (non-fatal)", exc_info=exc)
            else:
                logger.info(
                    "Polymarket user feed not started: credentials not configured "
                    "(call configure_user_feed_credentials() first)"
                )
            self._started = True
            loop = asyncio.get_running_loop()
            self._loop = loop  # store for thread-safe scheduling from callbacks
            from services.microstructure_recorder import get_microstructure_recorder
            from services.book_delta_decomposer import get_book_delta_decomposer

            get_microstructure_recorder().start()
            get_book_delta_decomposer().start()
            self._eviction_task = loop.create_task(self._cache_eviction_loop())
            # Wire PositionMarkState to receive every price tick
            from services.position_mark_state import get_position_mark_state

            pms = get_position_mark_state()
            self._cache.add_on_update_callback(pms.on_price_update)
            logger.info(
                "FeedManager started: loop=%s, on_update_callbacks=%d, pms_callback=%s, pms_on_marks_changed=%s",
                id(loop), len(self._cache._on_update_callbacks),
                "wired" if pms.on_price_update in self._cache._on_update_callbacks else "MISSING",
                "set" if pms._on_marks_changed else "NOT SET",
            )

    async def stop(self) -> None:
        """Stop both feeds and clear cache."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
        if self._price_push_task and not self._price_push_task.done():
            self._price_push_task.cancel()
        await self._polymarket_feed.stop()
        await self._kalshi_feed.stop()
        try:
            await self._polymarket_user_feed.stop()
        except Exception as exc:
            logger.debug("Polymarket user feed stop failed (non-fatal)", exc_info=exc)
        from services.microstructure_recorder import get_microstructure_recorder
        from services.book_delta_decomposer import get_book_delta_decomposer

        await get_microstructure_recorder().stop()
        await get_book_delta_decomposer().stop()
        self._cache.clear()
        self._ws_push_clients.clear()
        self._token_to_ws_clients.clear()
        self._pending_price_updates.clear()
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
        """Called by PriceCache when a significant price change occurs.

        NOTE: Fires from WS receive threads, not the asyncio loop.
        Must use call_soon_threadsafe for all async scheduling.
        """
        if self._reactive_scan_callback:
            self._changed_tokens.add(token_id)
            if self._loop is not None:
                try:
                    self._loop.call_soon_threadsafe(self._schedule_debounced_trigger)
                except RuntimeError:
                    pass  # loop closed

        # Event-driven exit evaluation for open positions on significant moves
        try:
            from services.trader_orchestrator.position_lifecycle import (
                get_registered_token_ids,
            )

            if token_id in get_registered_token_ids() and self._loop is not None:
                bid_ask = self._cache.get_best_bid_ask(token_id)
                bid = bid_ask[0] if bid_ask else new_mid
                ask = bid_ask[1] if bid_ask else new_mid
                self._loop.call_soon_threadsafe(
                    self._schedule_exit_eval, token_id, new_mid, bid, ask
                )
        except RuntimeError:
            pass
        except Exception:
            pass

        # Dispatch DataEvent for event-driven strategies
        if self._loop is not None:
            try:
                from services.data_events import DataEvent, EventType
                from utils.utcnow import utcnow

                event = DataEvent(
                    event_type=EventType.PRICE_CHANGE,
                    source="polymarket_ws",
                    timestamp=utcnow(),
                    token_id=token_id,
                    old_price=old_mid,
                    new_price=new_mid,
                )
                self._loop.call_soon_threadsafe(
                    self._schedule_event_dispatch, event
                )
            except RuntimeError:
                pass
            except Exception:
                pass

    def _on_trade(self, token_id: str, trade: "TradeRecord") -> None:
        """Dispatch a TRADE_EXECUTION DataEvent for each trade."""
        try:
            from services.microstructure_recorder import get_microstructure_recorder
            from services.book_delta_decomposer import get_book_delta_decomposer

            get_microstructure_recorder().record_trade(token_id=token_id, trade=trade)
            # Order matters — feed the trade BEFORE the post-trade book
            # update arrives so the decomposer can match the depth delta
            # against this print.
            get_book_delta_decomposer().record_trade(token_id=token_id, trade=trade)
        except Exception as exc:
            logger.debug("Microstructure trade record failed", exc_info=exc)
        if self._loop is None:
            return
        try:
            from services.data_events import DataEvent, EventType
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
            self._loop.call_soon_threadsafe(
                self._schedule_event_dispatch, event
            )
        except RuntimeError:
            pass
        except Exception:
            pass

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
        """Called by PriceCache on every cache replacement.

        Accumulates updates for the coalescing window, then a background task
        flushes them to registered WS clients.
        """
        try:
            from services.microstructure_recorder import get_microstructure_recorder
            from services.book_delta_decomposer import get_book_delta_decomposer

            order_book_snapshot = self._cache.get_order_book(token_id)
            get_microstructure_recorder().record_book(
                token_id=token_id,
                order_book=order_book_snapshot,
                best_bid=bid,
                best_ask=ask,
                exchange_ts=exchange_ts,
                ingest_ts=ingest_ts,
                sequence=sequence,
            )
            get_book_delta_decomposer().record_book(
                token_id=token_id,
                order_book=order_book_snapshot,
                best_bid=bid,
                best_ask=ask,
                observed_ts=ingest_ts,
            )
        except Exception as exc:
            logger.debug("Microstructure book record failed", exc_info=exc)
        should_schedule = False
        ws_ids = self._token_to_ws_clients.get(token_id)
        if not ws_ids:
            return
        self._pending_price_updates[token_id] = {
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "exchange_ts": exchange_ts or 0.0,
            "ingest_ts": ingest_ts or 0.0,
            "sequence": sequence or 0,
            "is_fresh": True,
        }
        # Check under the same lock to prevent duplicate task creation
        if self._price_push_task is None or self._price_push_task.done():
            should_schedule = True
        # Schedule outside the lock to avoid holding it during asyncio calls.
        # NOTE: This callback fires from WS receive threads, NOT the asyncio
        # event loop, so asyncio.get_running_loop() would raise RuntimeError.
        # Use call_soon_threadsafe to schedule the task on the captured loop.
        if should_schedule and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._schedule_price_push)
            except RuntimeError:
                pass  # loop closed

    def _schedule_price_push(self) -> None:
        """Create the coalesced price push task from the event loop thread."""
        if self._price_push_task is None or self._price_push_task.done():
            self._price_push_task = self._loop.create_task(self._coalesced_price_push())

    def _schedule_debounced_trigger(self) -> None:
        """Schedule the debounced reactive scan trigger from the event loop thread."""
        if self._debounce_task is None or self._debounce_task.done():
            self._debounce_task = self._loop.create_task(self._debounced_trigger())

    def _schedule_exit_eval(self, token_id: str, mid: float, bid: float, ask: float) -> None:
        """Schedule exit evaluation from the event loop thread."""
        from services.trader_orchestrator.position_lifecycle import evaluate_exit_for_token
        task = self._loop.create_task(evaluate_exit_for_token(token_id, mid, bid, ask))
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    def _schedule_event_dispatch(self, event) -> None:
        """Schedule event dispatcher from the event loop thread."""
        from services.event_dispatcher import event_dispatcher
        task = self._loop.create_task(event_dispatcher.dispatch(event))
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    # -- WS price push registration -----------------------------------------

    def register_ws_client(
        self,
        ws_id: int,
        token_ids: Set[str],
        push_callback: Callable[..., Any],
    ) -> None:
        """Register a frontend WS connection to receive price pushes.

        *ws_id* must be a unique identifier for the websocket (``id(ws)``).
        *token_ids* is the initial set of token_ids to receive.
        *push_callback* is an async callable ``(list[dict]) -> None`` that
        sends the batched price messages to the frontend.
        """
        self._ws_push_clients[ws_id] = {
            "token_ids": set(token_ids),
            "callback": push_callback,
        }
        for tid in token_ids:
            if tid not in self._token_to_ws_clients:
                self._token_to_ws_clients[tid] = set()
            self._token_to_ws_clients[tid].add(ws_id)

    def unregister_ws_client(self, ws_id: int) -> None:
        """Remove a frontend WS connection from price push."""
        client = self._ws_push_clients.pop(ws_id, None)
        if client:
            for tid in client.get("token_ids", set()):
                ws_set = self._token_to_ws_clients.get(tid)
                if ws_set:
                    ws_set.discard(ws_id)
                    if not ws_set:
                        del self._token_to_ws_clients[tid]

    def update_ws_client_tokens(self, ws_id: int, token_ids: Set[str]) -> None:
        """Update the set of token_ids a frontend WS client is watching."""
        client = self._ws_push_clients.get(ws_id)
        if client is None:
            return
        old_tokens = client["token_ids"]
        removed = old_tokens - token_ids
        added = token_ids - old_tokens
        client["token_ids"] = set(token_ids)
        for tid in removed:
            ws_set = self._token_to_ws_clients.get(tid)
            if ws_set:
                ws_set.discard(ws_id)
                if not ws_set:
                    del self._token_to_ws_clients[tid]
        for tid in added:
            if tid not in self._token_to_ws_clients:
                self._token_to_ws_clients[tid] = set()
            self._token_to_ws_clients[tid].add(ws_id)

    async def _coalesced_price_push(self) -> None:
        """Wait for the coalescing window, then push accumulated updates."""
        await asyncio.sleep(self._coalesce_ms / 1000.0)
        pending = dict(self._pending_price_updates)
        self._pending_price_updates.clear()
        if not pending:
            return
        # Build per-client batch: for each client, collect updates for tokens they care about
        client_batches: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for token_id, price_data in pending.items():
            ws_ids = self._token_to_ws_clients.get(token_id)
            if not ws_ids:
                continue
            for ws_id in ws_ids:
                if ws_id not in client_batches:
                    client_batches[ws_id] = {}
                client_batches[ws_id][token_id] = price_data
        callbacks = {
            ws_id: self._ws_push_clients[ws_id]["callback"]
            for ws_id in client_batches
            if ws_id in self._ws_push_clients
        }
        for ws_id, token_updates in client_batches.items():
            cb = callbacks.get(ws_id)
            if cb is None:
                continue
            try:
                await cb(token_updates)
            except Exception:
                pass

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

    def has_current_subscription_price(
        self,
        token_id: str,
        *,
        max_age_seconds: float | None = None,
        allow_stale_subscribed: bool = False,
    ) -> bool:
        normalized = str(token_id or "").strip().lower()
        if not normalized:
            return False
        if self._cache.get_mid_price(normalized) is None:
            return False
        if self._cache.is_fresh(normalized, max_age_seconds=max_age_seconds):
            return True
        if not allow_stale_subscribed:
            return False
        return (
            self._polymarket_feed.state == ConnectionState.CONNECTED
            and self._polymarket_feed.is_subscribed(normalized)
        )

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
