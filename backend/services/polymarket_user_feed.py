"""WebSocket subscription to the Polymarket CLOB user channel.

**Architecture role**: pushes the wallet's own order/trade events
into ``WalletStateCache`` in real time so the orchestrator never
has to REST-poll wallet state on the hot path.

Endpoint: ``wss://ws-subscriptions-clob.polymarket.com/ws/user``

Auth: ``apiKey`` + ``secret`` + ``passphrase`` in the subscribe
message — the same trio managed by ``live_execution_service``.

Subscription unit: ``condition_id`` (market hex address), NOT
token_id.  Per the architecture mandate (option a), we subscribe
to ALL condition_ids the bot has ever held a position in;
bandwidth is cheap, churn is risky.

Lifecycle: started by ``feed_manager`` only on the trading plane.
On reconnect, replays the full subscription set.  The cache is
*not* cleared on reconnect — the reconciliation worker's REST
re-seed (every 30s) absorbs any drift from missed events.
"""

from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from typing import Any, Optional, Set

from utils.logger import get_logger

try:
    import websockets
    import websockets.exceptions
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False

from services.wallet_state_cache import WalletStateCache, get_wallet_state_cache

logger = get_logger("polymarket_user_feed")


POLYMARKET_USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
# Ping cadence — Polymarket docs specify 10s; we use a slightly
# generous 8s so timeouts on our side are unlikely to drift past
# their server-side timeout.
_PING_INTERVAL_SECONDS = 8.0
_PING_TIMEOUT_SECONDS = 15.0
_OPEN_TIMEOUT_SECONDS = 10.0
_CLOSE_TIMEOUT_SECONDS = 5.0
_MAX_FRAME_SIZE = 2 ** 22  # 4 MiB

_RECONNECT_BASE_DELAY = 1.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_MULTIPLIER = 2.0
_MAX_BACKOFF_ATTEMPTS = 30


class _ConnState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


def _is_expected_close(exc: BaseException) -> bool:
    if not _WEBSOCKETS_AVAILABLE:
        return False
    if isinstance(exc, (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError)):
        code = getattr(exc, "code", None)
        if code is None:
            received = getattr(exc, "rcvd", None)
            sent = getattr(exc, "sent", None)
            code = getattr(received, "code", None) or getattr(sent, "code", None)
        return code is None or code in {1000, 1001, 1005}
    return False


class PolymarketUserFeed:
    """WS client for the Polymarket CLOB user channel.

    Mirrors the structural pattern of ``PolymarketWSFeed`` (market
    channel) but speaks the user-channel subscription protocol and
    routes events into ``WalletStateCache``.
    """

    def __init__(
        self,
        *,
        cache: Optional[WalletStateCache] = None,
        ws_url: str = POLYMARKET_USER_WS_URL,
    ) -> None:
        self._cache: WalletStateCache = cache or get_wallet_state_cache()
        self._ws_url = ws_url

        # Auth — populated by ``configure_credentials`` before start.
        self._api_key: Optional[str] = None
        self._api_secret: Optional[str] = None
        self._api_passphrase: Optional[str] = None
        self._wallet_address: Optional[str] = None

        # Subscription state.
        self._subscribed_conditions: Set[str] = set()
        self._sub_lock = asyncio.Lock()

        # Connection state.
        self._ws: Any = None
        self._state = _ConnState.DISCONNECTED
        self._run_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._reconnect_attempt = 0

        # Diagnostics.
        self._messages_received = 0
        self._messages_parsed = 0
        self._parse_errors = 0
        self._reconnections = 0
        self._last_message_at_mono: Optional[float] = None

    # -------------------- Public API --------------------

    @property
    def state(self) -> _ConnState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    def configure_credentials(
        self,
        *,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        wallet_address: Optional[str] = None,
    ) -> None:
        """Provide auth before ``start()``.  Required."""
        self._api_key = str(api_key or "").strip() or None
        self._api_secret = str(api_secret or "").strip() or None
        self._api_passphrase = str(api_passphrase or "").strip() or None
        self._wallet_address = (str(wallet_address or "").strip().lower() or None)
        if self._wallet_address:
            self._cache.configure_wallet(self._wallet_address)

    async def start(self) -> None:
        """Idempotent start.  Refuses if creds are missing or
        ``websockets`` library isn't installed.
        """
        if not _WEBSOCKETS_AVAILABLE:
            logger.error("PolymarketUserFeed cannot start: websockets library missing")
            return
        if self._run_task is not None and not self._run_task.done():
            return
        if not (self._api_key and self._api_secret and self._api_passphrase):
            logger.warning(
                "PolymarketUserFeed start refused: missing credentials "
                "(call configure_credentials() first)"
            )
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self._run_loop(), name="polymarket-user-ws")
        logger.info("PolymarketUserFeed started", url=self._ws_url)

    async def stop(self) -> None:
        self._stop_event.set()
        self._state = _ConnState.CLOSED
        self._cache.mark_ws_state(False)
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
        logger.info("PolymarketUserFeed stopped")

    async def subscribe_conditions(self, condition_ids: list[str]) -> None:
        """Add condition_ids to the subscription set.  Sends a
        subscribe frame immediately if connected; otherwise the
        subscriptions are replayed on next connect.
        """
        if not condition_ids:
            return
        normalized = [str(cid).strip() for cid in condition_ids if str(cid).strip()]
        if not normalized:
            return
        async with self._sub_lock:
            new_ids = [cid for cid in normalized if cid not in self._subscribed_conditions]
            self._subscribed_conditions.update(normalized)
            should_send = (
                bool(new_ids)
                and self._ws is not None
                and self._state == _ConnState.CONNECTED
            )
        if should_send:
            await self._send_subscribe(new_ids)

    async def unsubscribe_conditions(self, condition_ids: list[str]) -> None:
        """Remove condition_ids.  Sends an unsubscribe frame if connected.
        We only call this when the bot has fully exited a market and we
        want to free the subscription slot (rare).
        """
        if not condition_ids:
            return
        normalized = [str(cid).strip() for cid in condition_ids if str(cid).strip()]
        if not normalized:
            return
        async with self._sub_lock:
            removed = [cid for cid in normalized if cid in self._subscribed_conditions]
            self._subscribed_conditions.difference_update(normalized)
            should_send = (
                bool(removed)
                and self._ws is not None
                and self._state == _ConnState.CONNECTED
            )
        if should_send:
            await self._send_unsubscribe(removed)

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "subscribed_condition_ids": len(self._subscribed_conditions),
            "messages_received": self._messages_received,
            "messages_parsed": self._messages_parsed,
            "parse_errors": self._parse_errors,
            "reconnections": self._reconnections,
            "last_message_age_seconds": (
                None if self._last_message_at_mono is None
                else round(time.monotonic() - self._last_message_at_mono, 1)
            ),
        }

    # -------------------- Connection loop --------------------

    async def _run_loop(self) -> None:
        self._reconnect_attempt = 0
        while not self._stop_event.is_set():
            self._state = (
                _ConnState.CONNECTING if self._reconnect_attempt == 0
                else _ConnState.RECONNECTING
            )
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._reconnect_attempt += 1
                self._reconnections += 1

                if self._reconnect_attempt >= _MAX_BACKOFF_ATTEMPTS:
                    delay = _RECONNECT_MAX_DELAY
                else:
                    delay = min(
                        _RECONNECT_BASE_DELAY
                        * (_RECONNECT_MULTIPLIER ** (self._reconnect_attempt - 1)),
                        _RECONNECT_MAX_DELAY,
                    )

                if _is_expected_close(exc):
                    logger.info(
                        "Polymarket user WS disconnected cleanly; reconnecting",
                        delay=round(delay, 1),
                        attempt=self._reconnect_attempt,
                    )
                else:
                    logger.warning(
                        "Polymarket user WS error: %r; reconnecting in %.1fs (attempt %d)",
                        exc,
                        delay,
                        self._reconnect_attempt,
                    )

                self._state = _ConnState.DISCONNECTED
                self._cache.mark_ws_state(False)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                    break
                except asyncio.TimeoutError:
                    pass
            else:
                # Clean exit from inner loop (server closed).  Reconnect
                # with a 1s pause unless we're shutting down.
                if not self._stop_event.is_set():
                    self._reconnect_attempt += 1
                    self._reconnections += 1
                    self._cache.mark_ws_state(False)
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        pass
                    continue
                break

        self._state = _ConnState.CLOSED
        self._cache.mark_ws_state(False)

    async def _connect_and_listen(self) -> None:
        async with websockets.connect(
            self._ws_url,
            ping_interval=None,  # we manage heartbeat
            ping_timeout=None,
            open_timeout=_OPEN_TIMEOUT_SECONDS,
            close_timeout=_CLOSE_TIMEOUT_SECONDS,
            max_size=_MAX_FRAME_SIZE,
        ) as ws:
            self._ws = ws
            self._state = _ConnState.CONNECTED
            self._cache.mark_ws_state(True)
            connect_time = time.monotonic()
            logger.info(
                "Polymarket user WS connected",
                attempt=self._reconnect_attempt,
                subscribed_conditions=len(self._subscribed_conditions),
            )

            # Polymarket's user channel drops the connection within a
            # few seconds if no auth/subscribe message arrives.  Send
            # one IMMEDIATELY on connect.  Per the official docs
            # (Polymarket/agent-skills/websocket.md) the ``markets``
            # field is optional — omitting it subscribes to ALL of
            # the wallet's events.  This matches the architectural
            # mandate (option a): subscribe to every condition the
            # wallet has ever held a position in.  Dynamic
            # subscribe/unsubscribe operations remain available via
            # ``subscribe_conditions()`` / ``unsubscribe_conditions()``
            # for callers that want narrower scope.
            async with self._sub_lock:
                tracked = sorted(self._subscribed_conditions)
                await self._send_subscribe(tracked, ws_override=ws)

            # Start heartbeat.
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ws),
                name="polymarket-user-ws-heartbeat",
            )

            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    self._messages_received += 1
                    self._last_message_at_mono = time.monotonic()
                    if self._reconnect_attempt > 0 and (
                        self._last_message_at_mono - connect_time
                    ) >= 10.0:
                        # Connection has been stable for 10s; reset backoff.
                        self._reconnect_attempt = 0
                    try:
                        data = json.loads(raw)
                    except (TypeError, ValueError):
                        self._parse_errors += 1
                        continue
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                self._handle_message(item)
                    elif isinstance(data, dict):
                        self._handle_message(data)
                    self._messages_parsed += 1
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                self._ws = None

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Polymarket docs: send ``PING`` every 10s; server responds
        with ``PONG``.  We use 8s to leave margin.  No pong-timeout
        enforcement here (the existing ``async for raw in ws`` will
        raise ``ConnectionClosed`` on dead socket, triggering reconnect).
        """
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(_PING_INTERVAL_SECONDS)
                if self._stop_event.is_set():
                    break
                try:
                    await ws.send("PING")
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    # -------------------- Subscribe message --------------------

    async def _send_subscribe(
        self,
        condition_ids: list[str],
        *,
        ws_override: Any = None,
    ) -> None:
        ws = ws_override if ws_override is not None else self._ws
        if ws is None:
            return
        # Per Polymarket docs (agent-skills/websocket.md):
        #   {"auth": {...}, "type": "user", "markets": [...]}
        # The ``markets`` field is optional — *omit it* to receive
        # events for ALL of the wallet's markets.  Sending an empty
        # array is not the same as omitting; some implementations
        # reject empty-array subscribes.  We omit the field when no
        # specific markets are provided so the wallet receives all
        # events globally — matches the architectural mandate (a).
        msg: dict[str, Any] = {
            "auth": {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "passphrase": self._api_passphrase,
            },
            "type": "user",
        }
        if condition_ids:
            msg["markets"] = condition_ids
        try:
            await ws.send(json.dumps(msg))
            logger.debug(
                "Polymarket user WS subscribe sent",
                condition_count=len(condition_ids),
            )
        except Exception as exc:
            if _is_expected_close(exc):
                logger.info("Polymarket user WS subscribe interrupted by clean close")
            else:
                logger.warning("Polymarket user WS subscribe failed: %r", exc)

    async def _send_unsubscribe(self, condition_ids: list[str]) -> None:
        ws = self._ws
        if ws is None:
            return
        msg = {
            "auth": {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "passphrase": self._api_passphrase,
            },
            "type": "user",
            "operation": "unsubscribe",
            "markets": condition_ids,
        }
        try:
            await ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("Polymarket user WS unsubscribe failed (non-fatal): %r", exc)

    # -------------------- Message routing --------------------

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route a parsed user-channel event to the cache.

        Spec event shapes (from Polymarket docs):
          * ``trade`` events: ``event_type == "trade"`` with
            asset_id/market/owner/price/size/side/status/id/matchtime
          * ``order`` events: ``event_type == "order"`` with
            id (order_id)/asset_id/market/side/price/original_size/
            size_matched/type (PLACEMENT|UPDATE|CANCELLATION)
        """
        event_type = str(
            msg.get("event_type") or msg.get("type") or ""
        ).strip().lower()
        if event_type == "trade":
            self._cache.apply_trade(msg)
        elif event_type == "order":
            self._cache.apply_order(msg)
        # Silently ignore unknown event types — Polymarket may add
        # event types we don't yet handle; logging at debug avoids
        # spamming the log on benign new events.


# Singleton.
_polymarket_user_feed: Optional[PolymarketUserFeed] = None


def get_polymarket_user_feed() -> PolymarketUserFeed:
    global _polymarket_user_feed
    if _polymarket_user_feed is None:
        _polymarket_user_feed = PolymarketUserFeed()
    return _polymarket_user_feed


def reset_polymarket_user_feed() -> None:
    """Test helper."""
    global _polymarket_user_feed
    _polymarket_user_feed = None
