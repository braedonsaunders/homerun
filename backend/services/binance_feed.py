"""Direct Binance WebSocket price feed for BTC/ETH/SOL/XRP.

Connects to Binance's combined ``bookTicker`` stream to receive best
bid/ask updates with sub-100ms latency.  This is significantly faster
than the RTDS-relayed Binance prices in :mod:`chainlink_feed` (which
pass through Polymarket's relay, adding 500ms-2s).

The feed writes prices into :class:`chainlink_feed.ChainlinkFeed` via
its ``update_from_binance_direct()`` method, tagged with source
``"binance_direct"`` to distinguish from the RTDS relay (``"binance"``).

See: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

import websockets

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

RECONNECT_BASE_MS = 200
RECONNECT_MAX_MS = 30_000
RECONNECT_AFTER_STALE_MS = 200

_BINANCE_SYMBOL_MAP = {
    "btcusdt": "BTC",
    "ethusdt": "ETH",
    "solusdt": "SOL",
    "xrpusdt": "XRP",
}


def _stale_data_timeout_s() -> float:
    raw = float(getattr(settings, "BINANCE_WS_STALE_DATA_TIMEOUT_SECONDS", 8.0) or 8.0)
    # Floor at 0.5s to avoid pathologically short windows that would
    # treat any normal tick gap as "stale" and reconnect-storm.
    return max(0.5, raw)


class BinanceFeed:
    """Direct WebSocket connection to Binance bookTicker stream.

    Fires ``on_update`` callback with ``(asset, mid, bid, ask, timestamp_ms)``
    on every book update.  Source tag is ``"binance_direct"`` to distinguish
    from the RTDS-relayed ``"binance"`` prices in ChainlinkFeed.

    Usage::

        feed = BinanceFeed()
        feed.on_update(lambda asset, mid, bid, ask, ts: print(asset, mid))
        await feed.start()
        # ...
        await feed.stop()
    """

    def __init__(self, ws_url: Optional[str] = None):
        self._ws_url = ws_url or settings.BINANCE_WS_URL
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._on_update: Optional[Callable] = None
        self._last_update_ms: int = 0
        # Per-asset last-message timestamp so we can spot one stream
        # going silent while the others tick (e.g. wrong URL format
        # silently subscribing to only the first symbol).
        self._last_update_ms_by_asset: dict[str, int] = {}
        self._message_count_by_asset: dict[str, int] = {}

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def last_update_ms(self) -> int:
        return self._last_update_ms

    def get_per_asset_status(self) -> dict[str, dict[str, int]]:
        """Snapshot of per-asset last-update + message count.

        Diagnostic only — used by status endpoints / tests to verify
        each subscribed stream is delivering, not just whichever symbol
        Binance picked off the URL.
        """
        now_ms = int(time.time() * 1000)
        out: dict[str, dict[str, int]] = {}
        for asset, last_ms in self._last_update_ms_by_asset.items():
            out[asset] = {
                "last_update_ms": int(last_ms),
                "age_ms": int(now_ms - last_ms) if last_ms > 0 else -1,
                "message_count": int(self._message_count_by_asset.get(asset, 0)),
            }
        return out

    def on_update(self, callback: Callable) -> None:
        """Register a callback for price updates.

        Signature: ``callback(asset: str, mid: float, bid: float, ask: float, timestamp_ms: int)``
        """
        self._on_update = callback

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run_loop())
        logger.info("BinanceFeed: starting connection to %s", self._ws_url)

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("BinanceFeed: stopped")

    async def _run_loop(self) -> None:
        """Reconnecting WebSocket loop with exponential backoff.

        Two liveness layers:
          * ``ping_interval`` / ``ping_timeout`` keep the TCP connection
            healthy at the protocol level.
          * Per-message ``wait_for(recv, timeout=stale)`` detects when
            Binance silently stops streaming even though pings still
            answer (their edge has been observed to do this during
            failover events).  On stale-timeout we tear the socket down
            and reconnect immediately — exponential backoff is reserved
            for true connection failures.

        Backoff resets on the first *message* received (not on connect)
        so a "false-success" socket that closes before delivering data
        keeps growing the backoff instead of resetting it.
        """
        reconnect_ms = RECONNECT_BASE_MS

        while not self._stopped:
            stale_reconnect = False
            received_any = False
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("BinanceFeed: connected to bookTicker stream")
                    stale_timeout = _stale_data_timeout_s()
                    while not self._stopped:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=stale_timeout)
                        except asyncio.TimeoutError:
                            # The wait_for timer fired, but the event loop may
                            # have been stalled (heavy strategy bursts, GC) —
                            # use wall-clock time vs the last *handled* message
                            # to confirm the feed is genuinely silent before
                            # tearing the socket down.  Suppresses false-positive
                            # reconnect storms under load.
                            now_ms = int(time.time() * 1000)
                            last_ms = self._last_update_ms
                            actual_age_ms = (now_ms - last_ms) if last_ms > 0 else None
                            stale_threshold_ms = int(stale_timeout * 1000)
                            if last_ms > 0 and actual_age_ms is not None and actual_age_ms < stale_threshold_ms:
                                # Data has actually arrived recently — keep waiting.
                                continue
                            logger.warning(
                                "BinanceFeed: no bookTicker data for %.1fs — forcing reconnect (actual_age_ms=%s, per-asset: %s)",
                                stale_timeout,
                                actual_age_ms,
                                self.get_per_asset_status(),
                            )
                            stale_reconnect = True
                            break
                        if not received_any:
                            received_any = True
                            reconnect_ms = RECONNECT_BASE_MS
                        self._handle_message(raw)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._stopped:
                    break
                logger.debug("BinanceFeed: connection error: %s", e)

            if self._stopped:
                break

            if stale_reconnect:
                # Connection was healthy enough to accept us — only the
                # data feed went silent.  Reconnect promptly without
                # exponential penalty and reset the backoff window.
                reconnect_ms = RECONNECT_BASE_MS
                await asyncio.sleep(RECONNECT_AFTER_STALE_MS / 1000.0)
                continue

            await asyncio.sleep(reconnect_ms / 1000.0)
            if not received_any:
                reconnect_ms = min(int(reconnect_ms * 1.5), RECONNECT_MAX_MS)
            else:
                reconnect_ms = RECONNECT_BASE_MS

    def _handle_message(self, raw: str | bytes) -> None:
        """Parse a Binance bookTicker message and fire callback."""
        msg_str = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if not msg_str.strip():
            return

        try:
            data = json.loads(msg_str)
        except json.JSONDecodeError:
            return

        # Combined stream format wraps ticker in {"stream": "...", "data": {...}}
        if isinstance(data, dict) and "data" in data:
            ticker = data["data"]
        else:
            ticker = data

        if not isinstance(ticker, dict):
            return

        symbol = str(ticker.get("s") or "").lower()
        asset = _BINANCE_SYMBOL_MAP.get(symbol)
        if not asset:
            return

        try:
            bid = float(ticker["b"])
            ask = float(ticker["a"])
        except (KeyError, TypeError, ValueError):
            return
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2.0
        now_ms = int(time.time() * 1000)
        self._last_update_ms = now_ms
        self._last_update_ms_by_asset[asset] = now_ms
        self._message_count_by_asset[asset] = self._message_count_by_asset.get(asset, 0) + 1

        # Crypto latency harness: record wire arrival per asset for
        # ``freshest_wire_ts_ms()`` lookups in ``place_order``.  Wrapped
        # in try/except inside the recorder; this call is sub-microsecond
        # and must never affect WS message handling.
        try:
            from services.crypto_latency_trace import record_wire_event

            record_wire_event("binance", asset, now_ms)
        except Exception:
            pass

        if self._on_update:
            try:
                self._on_update(asset, mid, bid, ask, now_ms)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[BinanceFeed] = None


def get_binance_feed() -> BinanceFeed:
    global _instance
    if _instance is None:
        _instance = BinanceFeed()
    return _instance
