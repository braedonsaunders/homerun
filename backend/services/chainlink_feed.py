"""Chainlink oracle price feed via Polymarket's Real Time Data Stream (RTDS).

Subscribes to ``wss://ws-live-data.polymarket.com`` topics:
  - ``crypto_prices_chainlink`` — Chainlink oracle prices (resolution source)
  - ``crypto_prices`` — Binance exchange prices (more frequent updates)

These are the **exact prices** Polymarket uses to resolve 15-minute
crypto markets.  Maintains a rolling history buffer so the "price to beat"
can be looked up for any recent timestamp, even if the app started mid-window.

See: https://docs.polymarket.com/developers/RTDS/RTDS-crypto-prices
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Optional, Callable

import websockets

from utils.logger import get_logger

logger = get_logger(__name__)

WS_URL = "wss://ws-live-data.polymarket.com"
CHAINLINK_TOPIC = "crypto_prices_chainlink"
BINANCE_TOPIC = "crypto_prices"
RECONNECT_BASE_MS = 500
RECONNECT_MAX_MS = 10_000

# Rolling history: keep 25 minutes of prices (covers any 15-min window start)
HISTORY_MAX_AGE_MS = 25 * 60 * 1000
HISTORY_MAX_ENTRIES = 3000  # ~2 updates/sec * 25 min * 4 assets

# Maps RTDS symbols to our canonical asset names
# Chainlink uses "btc/usd", Binance uses "btcusdt"
_SYMBOL_MAP = {
    "btc/usd": "BTC",
    "eth/usd": "ETH",
    "sol/usd": "SOL",
    "xrp/usd": "XRP",
    "btcusdt": "BTC",
    "ethusdt": "ETH",
    "solusdt": "SOL",
    "xrpusdt": "XRP",
    # Fallback substring matches
    "btc": "BTC",
    "bitcoin": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
    "sol": "SOL",
    "solana": "SOL",
    "xrp": "XRP",
    "ripple": "XRP",
}


class OraclePrice:
    """A single oracle price snapshot."""

    __slots__ = ("asset", "price", "updated_at_ms", "source")

    def __init__(self, asset: str, price: float, updated_at_ms: Optional[int] = None):
        self.asset = asset
        self.price = price
        self.updated_at_ms = updated_at_ms or int(time.time() * 1000)
        self.source = "chainlink_polymarket_ws"

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "price": self.price,
            "updated_at_ms": self.updated_at_ms,
            "age_seconds": (time.time() * 1000 - self.updated_at_ms) / 1000
            if self.updated_at_ms
            else None,
            "source": self.source,
        }


class ChainlinkFeed:
    """WebSocket client for Polymarket's Chainlink oracle price feed.

    Usage::

        feed = ChainlinkFeed()
        await feed.start()

        # Get latest prices
        btc = feed.get_price("BTC")  # -> OraclePrice or None
        all_prices = feed.get_all_prices()  # -> dict[str, OraclePrice]

        await feed.stop()
    """

    def __init__(self, ws_url: str = WS_URL):
        self._ws_url = ws_url
        self._prices: dict[str, OraclePrice] = {}
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._on_update: Optional[Callable[[OraclePrice], None]] = None
        # Rolling price history per asset: deque of (timestamp_ms, price)
        self._history: dict[str, deque] = {}

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    def get_price(self, asset: str) -> Optional[OraclePrice]:
        """Get the latest oracle price for an asset (e.g. 'BTC')."""
        return self._prices.get(asset.upper())

    def get_all_prices(self) -> dict[str, OraclePrice]:
        """Get all latest oracle prices."""
        return dict(self._prices)

    def get_price_at_time(self, asset: str, timestamp_s: float) -> Optional[float]:
        """Get the oracle price closest to a given Unix timestamp (seconds).

        Searches the rolling history buffer for the Chainlink price
        recorded closest to ``timestamp_s``.  Returns None if no history
        is available within 60 seconds of the target time.

        This is used to determine the "price to beat" for a market whose
        ``eventStartTime`` is known.
        """
        asset = asset.upper()
        history = self._history.get(asset)
        if not history:
            return None

        target_ms = timestamp_s * 1000
        best_price = None
        best_dist = float("inf")

        for ts_ms, price in history:
            dist = abs(ts_ms - target_ms)
            if dist < best_dist:
                best_dist = dist
                best_price = price

        # Only return if within 60 seconds of target
        if best_dist <= 60_000:
            return best_price
        return None

    def on_update(self, callback: Callable[[OraclePrice], None]) -> None:
        """Register a callback for price updates."""
        self._on_update = callback

    async def start(self) -> None:
        """Start the WebSocket connection in the background."""
        if self._task and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"ChainlinkFeed: starting connection to {self._ws_url}")

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("ChainlinkFeed: stopped")

    async def _run_loop(self) -> None:
        """Reconnecting WebSocket loop."""
        reconnect_ms = RECONNECT_BASE_MS

        while not self._stopped:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    reconnect_ms = RECONNECT_BASE_MS
                    logger.info("ChainlinkFeed: connected")

                    # Subscribe to both Chainlink (resolution source) and
                    # Binance (more frequent updates) price feeds
                    sub_msg = json.dumps(
                        {
                            "action": "subscribe",
                            "subscriptions": [
                                {
                                    "topic": CHAINLINK_TOPIC,
                                    "type": "*",
                                    "filters": "",
                                },
                                {
                                    "topic": BINANCE_TOPIC,
                                    "type": "update",
                                },
                            ],
                        }
                    )
                    await ws.send(sub_msg)
                    logger.info(
                        f"ChainlinkFeed: subscribed to {CHAINLINK_TOPIC} + {BINANCE_TOPIC}"
                    )

                    async for raw in ws:
                        if self._stopped:
                            break
                        self._handle_message(raw)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._stopped:
                    break
                logger.debug(f"ChainlinkFeed: connection error: {e}")
                await asyncio.sleep(reconnect_ms / 1000.0)
                reconnect_ms = min(
                    int(reconnect_ms * 1.5), RECONNECT_MAX_MS
                )

    def _handle_message(self, raw: str | bytes) -> None:
        """Parse a WebSocket message and update the price cache + history."""
        msg_str = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if not msg_str.strip():
            return

        try:
            data = json.loads(msg_str)
        except json.JSONDecodeError:
            return

        topic = data.get("topic")
        if topic not in (CHAINLINK_TOPIC, BINANCE_TOPIC):
            return

        payload = data.get("payload")
        if payload is None:
            return
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return
        if not isinstance(payload, dict):
            return

        # Extract symbol
        symbol = str(
            payload.get("symbol")
            or payload.get("pair")
            or payload.get("ticker")
            or ""
        ).lower()

        # Map to canonical asset -- try exact match first, then substring
        asset = _SYMBOL_MAP.get(symbol)
        if not asset:
            for keyword, canonical in _SYMBOL_MAP.items():
                if keyword in symbol:
                    asset = canonical
                    break
        if not asset:
            return

        # Extract price (field is "value" per RTDS docs)
        price_val = (
            payload.get("value")
            or payload.get("price")
            or payload.get("current")
        )
        try:
            price = float(price_val)
        except (TypeError, ValueError):
            return
        if not (price > 0):
            return

        # Extract timestamp
        ts_val = payload.get("timestamp") or payload.get("updatedAt")
        updated_at_ms = None
        if ts_val is not None:
            try:
                ts_float = float(ts_val)
                updated_at_ms = (
                    int(ts_float * 1000) if ts_float < 1e12 else int(ts_float)
                )
            except (TypeError, ValueError):
                pass
        if updated_at_ms is None:
            updated_at_ms = int(time.time() * 1000)

        # Determine source priority: Chainlink is authoritative (resolution source)
        is_chainlink = topic == CHAINLINK_TOPIC
        source = "chainlink" if is_chainlink else "binance"

        # Update latest price (Chainlink takes priority over Binance)
        existing = self._prices.get(asset)
        if existing is None or is_chainlink or source == existing.source:
            oracle = OraclePrice(
                asset=asset,
                price=price,
                updated_at_ms=updated_at_ms,
            )
            oracle.source = source
            self._prices[asset] = oracle

            if self._on_update:
                try:
                    self._on_update(oracle)
                except Exception:
                    pass

        # Always store in history (for price-to-beat lookups)
        # Only store Chainlink prices in history since that's the resolution source
        if is_chainlink:
            if asset not in self._history:
                self._history[asset] = deque(maxlen=HISTORY_MAX_ENTRIES)
            self._history[asset].append((updated_at_ms, price))

            # Prune old entries
            cutoff = int(time.time() * 1000) - HISTORY_MAX_AGE_MS
            while self._history[asset] and self._history[asset][0][0] < cutoff:
                self._history[asset].popleft()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[ChainlinkFeed] = None


def get_chainlink_feed() -> ChainlinkFeed:
    global _instance
    if _instance is None:
        _instance = ChainlinkFeed()
    return _instance
