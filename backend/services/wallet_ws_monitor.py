"""
Real-time WebSocket-based wallet monitoring for copy trading.

Connects to Polygon WebSocket RPC to subscribe to new blocks and detect
OrdersFilled events from the Polymarket CTF Exchange contract in near
real-time (<1 second latency vs 30-second polling).

Inspired by terauss/Polymarket-Copy-Trading-Bot architecture.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from utils.utcnow import utcnow, utcfromtimestamp
from typing import Callable, Optional

import httpx

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

from sqlalchemy import Column, String, Float, Integer, DateTime, Index

from config import settings
from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("wallet_ws_monitor")

# ==================== CONSTANTS ====================

# Polymarket CTF Exchange contract on Polygon
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)
# keccak256 topic hash
ORDER_FILLED_TOPIC = (
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
)

# Default Polygon WebSocket RPC endpoint
DEFAULT_WS_URL = settings.POLYGON_WS_URL

# Default HTTP RPC fallback endpoint
DEFAULT_HTTP_RPC_URL = settings.POLYGON_RPC_URL

# Fallback RPC endpoints (tried in order after the configured primary).
FALLBACK_HTTP_RPC_URLS = (
    "https://polygon-rpc.com",
    "https://polygon-bor-rpc.publicnode.com",
)

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=8.0, write=8.0, pool=8.0)


# ==================== DATA MODEL ====================


@dataclass
class WalletTradeEvent:
    """Represents a trade detected on-chain for a tracked wallet."""

    wallet_address: str
    token_id: str
    side: str  # BUY or SELL
    size: float
    price: float
    tx_hash: str
    block_number: int
    timestamp: datetime
    detected_at: datetime  # When we detected it
    latency_ms: float  # Detection latency


# ==================== SQLALCHEMY MODEL ====================


class WalletMonitorEvent(Base):
    """Persisted record of a wallet trade event detected via WebSocket monitoring."""

    __tablename__ = "wallet_monitor_events"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, index=True)
    token_id = Column(String, nullable=True)
    side = Column(String, nullable=True)
    size = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    tx_hash = Column(String, nullable=True)
    block_number = Column(Integer, nullable=True)
    detection_latency_ms = Column(Float, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_wme_wallet", "wallet_address"),
        Index("idx_wme_block", "block_number"),
    )


# ==================== HELPERS ====================


def _decode_uint256(hex_str: str) -> int:
    """Decode a 256-bit unsigned integer from a hex-encoded ABI word."""
    return int(hex_str, 16)


def _exception_text(exc: Exception) -> str:
    """Return a non-empty exception string for structured logging."""
    text = str(exc).strip()
    return text if text else repr(exc)


def _build_rpc_candidates(primary_url: str) -> list[str]:
    """Build de-duplicated RPC endpoints in failover order."""
    urls: list[str] = []
    for raw_url in (primary_url, *FALLBACK_HTTP_RPC_URLS):
        url = (raw_url or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _decode_address_from_topic(topic_hex: str) -> str:
    """Extract an Ethereum address from a 32-byte ABI-encoded topic.

    Addresses are left-padded with zeros in indexed event topics.
    """
    # Last 40 hex chars (20 bytes) represent the address
    clean = topic_hex.lower().replace("0x", "")
    return "0x" + clean[-40:]


def _parse_order_filled_log(log: dict) -> Optional[dict]:
    """Parse an OrderFilled event log into a structured dict.

    OrderFilled event layout:
        topic[0]: event signature hash
        topic[1]: orderHash (bytes32, indexed)
        data: maker (address), taker (address), makerAssetId (uint256),
              takerAssetId (uint256), makerAmountFilled (uint256),
              takerAmountFilled (uint256), fee (uint256)

    Note: Only orderHash is indexed. The remaining parameters are in
    the data field, each as a 32-byte ABI word.
    """
    topics = log.get("topics", [])
    data = log.get("data", "0x")

    if len(topics) < 2:
        return None

    # Strip 0x prefix from data
    data_hex = data[2:] if data.startswith("0x") else data

    # Each ABI word is 64 hex chars (32 bytes)
    if len(data_hex) < 64 * 7:
        return None

    words = [data_hex[i * 64 : (i + 1) * 64] for i in range(7)]

    order_hash = topics[1]
    maker = _decode_address_from_topic("0x" + words[0])
    taker = _decode_address_from_topic("0x" + words[1])
    maker_asset_id = _decode_uint256(words[2])
    taker_asset_id = _decode_uint256(words[3])
    maker_amount_filled = _decode_uint256(words[4])
    taker_amount_filled = _decode_uint256(words[5])
    fee = _decode_uint256(words[6])

    return {
        "order_hash": order_hash,
        "maker": maker,
        "taker": taker,
        "maker_asset_id": str(maker_asset_id),
        "taker_asset_id": str(taker_asset_id),
        "maker_amount_filled": maker_amount_filled,
        "taker_amount_filled": taker_amount_filled,
        "fee": fee,
    }


def _determine_trade_side_and_details(
    parsed: dict, wallet_address: str
) -> tuple[str, str, float, float]:
    """Determine trade side, token_id, size, and price from parsed log data.

    If the wallet is the maker, they placed the order:
      - token_id = makerAssetId (what they are giving)
      - size = makerAmountFilled
      - The maker is SELLING makerAssetId and BUYING takerAssetId
      - price = takerAmountFilled / makerAmountFilled (how much taker asset per maker asset)

    If the wallet is the taker, they filled an existing order:
      - token_id = takerAssetId (what they are giving)
      - size = takerAmountFilled
      - The taker is SELLING takerAssetId and BUYING makerAssetId
      - price = makerAmountFilled / takerAmountFilled

    In Polymarket context, BUY means acquiring outcome tokens, SELL means disposing.
    We simplify: if wallet is maker -> SELL, if wallet is taker -> BUY.

    Amounts are in raw units (typically 6 decimals for USDC, but token-dependent).
    We normalize to float using 1e6 divisor for USDC-denominated amounts.
    """
    wallet_lower = wallet_address.lower()
    maker_lower = parsed["maker"].lower()

    # Normalize amounts (Polymarket uses 6 decimal USDC)
    maker_amount = parsed["maker_amount_filled"] / 1e6
    taker_amount = parsed["taker_amount_filled"] / 1e6

    if wallet_lower == maker_lower:
        # Wallet is the maker (placed the order) -> they are selling their asset
        side = "SELL"
        token_id = parsed["maker_asset_id"]
        size = maker_amount
        price = taker_amount / maker_amount if maker_amount > 0 else 0.0
    else:
        # Wallet is the taker (filled the order) -> they are buying maker's asset
        side = "BUY"
        token_id = parsed["taker_asset_id"]
        size = taker_amount
        price = maker_amount / taker_amount if taker_amount > 0 else 0.0

    return side, token_id, size, price


# ==================== MONITOR SERVICE ====================


class WalletWebSocketMonitor:
    """Real-time wallet monitoring via Polygon WebSocket RPC.

    Subscribes to new block headers and, for each block, queries
    OrdersFilled event logs from the Polymarket CTF Exchange contract,
    filtering for tracked wallet addresses. Detected trades are emitted
    via registered callbacks and persisted to the database.

    Falls back to HTTP polling if the WebSocket connection drops, with
    exponential backoff for reconnection attempts.
    """

    def __init__(self):
        self._running: bool = False
        # address -> set of logical sources that requested tracking
        # (e.g. copy_trader, discovery_pool)
        self._tracked_sources: dict[str, set[str]] = {}
        self._callbacks: list[Callable] = []
        self._ws_url: str = DEFAULT_WS_URL
        self._http_rpc_url: str = DEFAULT_HTTP_RPC_URL or "https://polygon-rpc.com"
        self._rpc_urls: list[str] = _build_rpc_candidates(self._http_rpc_url)
        self._reconnect_delay: int = 5
        self._max_reconnect_delay: int = 60
        self._ws_connection = None
        self._last_processed_block: Optional[int] = None
        self._poll_fallback_task: Optional[asyncio.Task] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._mempool_mode = False  # Enable for pre-confirmation (<1s) latency
        self._stats = {
            "blocks_processed": 0,
            "events_detected": 0,
            "ws_reconnects": 0,
            "fallback_polls": 0,
            "errors": 0,
            "mempool_txns_seen": 0,
        }

    # ==================== WALLET MANAGEMENT ====================

    def _tracked_addresses(self) -> set[str]:
        """Flatten source-scoped memberships into a set of tracked addresses."""
        return set(self._tracked_sources.keys())

    def add_wallet(self, address: str, source: str = "default"):
        """Add a wallet address to the tracked set for a specific source.

        Args:
            address: Ethereum address (checksummed or lowercase).
            source: Logical owner of this membership.
        """
        normalized = address.lower()
        memberships = self._tracked_sources.setdefault(normalized, set())
        memberships.add(source)
        logger.info(
            "Added wallet to WS monitor",
            address=normalized,
            source=source,
            total_tracked=len(self._tracked_sources),
        )

    def remove_wallet(self, address: str, source: str = "default"):
        """Remove a source membership for a wallet.

        Args:
            address: Ethereum address to stop tracking.
            source: Logical owner whose membership should be removed.
        """
        normalized = address.lower()
        memberships = self._tracked_sources.get(normalized)
        if memberships is None:
            return
        memberships.discard(source)
        if not memberships:
            self._tracked_sources.pop(normalized, None)
        logger.info(
            "Removed wallet from WS monitor",
            address=normalized,
            source=source,
            total_tracked=len(self._tracked_sources),
        )

    def set_wallets_for_source(self, source: str, addresses: list[str]):
        """Replace all tracked wallets for a source in one operation."""
        target = {a.lower() for a in addresses if a}
        current = {
            addr for addr, memberships in self._tracked_sources.items() if source in memberships
        }

        for address in current - target:
            self.remove_wallet(address, source=source)
        for address in target - current:
            self.add_wallet(address, source=source)

        logger.info(
            "Updated source-scoped tracked wallets",
            source=source,
            total_tracked=len(self._tracked_sources),
            source_wallets=len(target),
        )

    def add_callback(self, callback: Callable):
        """Register a callback to be invoked when a tracked wallet trade is detected.

        The callback receives a single argument: a WalletTradeEvent instance.
        If the callback is a coroutine function, it will be awaited.

        Args:
            callback: Sync or async callable accepting a WalletTradeEvent.
        """
        self._callbacks.append(callback)

    # ==================== MEMPOOL MONITORING ====================

    def enable_mempool_mode(self):
        """Enable mempool monitoring for pre-confirmation detection.

        When enabled, the monitor will additionally subscribe to
        ``newPendingTransactions`` on the WebSocket connection, providing
        sub-second detection latency for tracked wallet activity before
        the transaction is included in a block.
        """
        self._mempool_mode = True
        logger.info("Mempool monitoring enabled - pre-confirmation detection active")

    def disable_mempool_mode(self):
        """Disable mempool monitoring, reverting to block-based detection only."""
        self._mempool_mode = False
        logger.info("Mempool monitoring disabled")

    async def _subscribe_mempool(self, ws):
        """Subscribe to pending transactions for pre-confirmation detection."""
        subscribe_msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "eth_subscribe",
                "params": ["newPendingTransactions"],
            }
        )
        await ws.send(subscribe_msg)

        response = await ws.recv()
        sub_result = json.loads(response)

        if "error" in sub_result:
            logger.warning(
                "Mempool subscription failed (node may not support it)",
                error=sub_result["error"],
            )
            return None

        subscription_id = sub_result.get("result")
        logger.info(
            "Subscribed to mempool (pending transactions)",
            subscription_id=subscription_id,
        )
        return subscription_id

    # ==================== LIFECYCLE ====================

    async def start(self):
        """Start WebSocket monitoring.

        Launches the WebSocket loop as a background task. If the monitor
        is already running, this is a no-op.
        """
        if self._running:
            logger.warning("WS monitor already running")
            return

        if websockets is None:
            logger.error(
                "websockets library not installed. Install with: pip install websockets"
            )
            return

        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info(
            "Started wallet WS monitor",
            ws_url=self._ws_url,
            tracked_wallets=len(self._tracked_sources),
        )

    def stop(self):
        """Stop the WebSocket monitor and all background tasks."""
        self._running = False

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._poll_fallback_task and not self._poll_fallback_task.done():
            self._poll_fallback_task.cancel()

        logger.info(
            "Stopped wallet WS monitor",
            stats=self._stats,
        )

    # ==================== WEBSOCKET LOOP ====================

    async def _ws_loop(self):
        """Main WebSocket loop with auto-reconnect and exponential backoff.

        Connects to the Polygon WebSocket RPC, subscribes to newHeads,
        and processes each new block for relevant events. On disconnection,
        starts a polling fallback and attempts to reconnect with exponential
        backoff up to _max_reconnect_delay seconds.
        """
        current_delay = self._reconnect_delay

        while self._running:
            try:
                logger.info("Connecting to Polygon WS RPC", url=self._ws_url)

                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
                ) as ws:
                    self._ws_connection = ws
                    current_delay = self._reconnect_delay  # Reset on success

                    # Stop fallback polling if it was running
                    if self._poll_fallback_task and not self._poll_fallback_task.done():
                        self._poll_fallback_task.cancel()
                        self._poll_fallback_task = None
                        logger.info("Stopped fallback polling, WS reconnected")

                    # Subscribe to newHeads
                    subscribe_msg = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "eth_subscribe",
                            "params": ["newHeads"],
                        }
                    )
                    await ws.send(subscribe_msg)

                    # Read subscription confirmation
                    response = await ws.recv()
                    sub_result = json.loads(response)

                    if "error" in sub_result:
                        logger.error(
                            "WS subscription failed",
                            error=sub_result["error"],
                        )
                        raise ConnectionError(
                            f"Subscription failed: {sub_result['error']}"
                        )

                    subscription_id = sub_result.get("result")
                    logger.info(
                        "Subscribed to newHeads",
                        subscription_id=subscription_id,
                    )

                    # Optionally subscribe to mempool for pre-confirmation detection
                    if self._mempool_mode:
                        await self._subscribe_mempool(ws)

                    # Process incoming messages
                    async for message in ws:
                        if not self._running:
                            break

                        try:
                            data = json.loads(message)
                            params = data.get("params", {})
                            result = params.get("result", {})

                            block_number_hex = result.get("number")
                            if block_number_hex:
                                block_number = int(block_number_hex, 16)
                                block_timestamp_hex = result.get("timestamp", "0x0")
                                await self._handle_block(
                                    block_number,
                                    block_timestamp_hex=block_timestamp_hex,
                                )

                        except json.JSONDecodeError:
                            self._stats["errors"] += 1
                        except Exception:
                            self._stats["errors"] += 1

            except asyncio.CancelledError:
                logger.info("WS loop cancelled")
                break

            except Exception as e:
                self._ws_connection = None
                self._stats["ws_reconnects"] += 1

                logger.warning(
                    "WS connection lost, will reconnect",
                    error_type=type(e).__name__,
                    error=_exception_text(e),
                    reconnect_delay=current_delay,
                )

                # Start fallback polling while disconnected
                if self._poll_fallback_task is None or self._poll_fallback_task.done():
                    self._poll_fallback_task = asyncio.create_task(
                        self._poll_fallback_loop()
                    )

                if self._running:
                    await asyncio.sleep(current_delay)
                    # Exponential backoff
                    current_delay = min(current_delay * 2, self._max_reconnect_delay)

        self._ws_connection = None

    # ==================== BLOCK PROCESSING ====================

    async def _handle_block(
        self,
        block_number: int,
        block_timestamp_hex: str = "0x0",
    ):
        """Process a new block by querying for OrdersFilled events.

        Calls eth_getLogs on the HTTP RPC endpoint for the specific block,
        filtering by the CTF Exchange address and the OrderFilled event topic.
        Any matching logs involving tracked wallets are processed.

        Args:
            block_number: The block number to process.
            block_timestamp_hex: Hex-encoded block timestamp.
        """
        if not self._tracked_sources:
            return

        # Avoid processing the same block twice
        if (
            self._last_processed_block is not None
            and block_number <= self._last_processed_block
        ):
            return

        block_hex = hex(block_number)

        try:
            block_timestamp = int(block_timestamp_hex, 16)
        except (ValueError, TypeError):
            block_timestamp = 0

        try:
            # Query logs via HTTP RPC (more reliable for getLogs)
            logs = await self._get_logs_for_block(block_hex)

            for log_entry in logs:
                try:
                    await self._process_log(
                        log_entry,
                        block_number=block_number,
                        block_timestamp=block_timestamp,
                    )
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(
                        "Error processing log entry",
                        block_number=block_number,
                        tx_hash=log_entry.get("transactionHash"),
                        error_type=type(e).__name__,
                        error=_exception_text(e),
                        exc_info=True,
                    )

            # Only advance the cursor after a successful RPC call so failed
            # blocks can be retried by fallback polling.
            self._last_processed_block = block_number
            self._stats["blocks_processed"] += 1

        except Exception as e:
            logger.error(
                "Error handling block",
                block_number=block_number,
                error_type=type(e).__name__,
                error=_exception_text(e),
                exc_info=True,
            )
            self._stats["errors"] += 1

    async def _get_logs_for_block(self, block_hex: str) -> list[dict]:
        """Fetch OrdersFilled event logs for a specific block from the HTTP RPC.

        Args:
            block_hex: Block number as a hex string (e.g. '0x1a2b3c').

        Returns:
            List of log entries matching the filter criteria.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getLogs",
            "params": [
                {
                    "fromBlock": block_hex,
                    "toBlock": block_hex,
                    "address": CTF_EXCHANGE_ADDRESS,
                    "topics": [ORDER_FILLED_TOPIC],
                }
            ],
        }

        result = await self._rpc_request(
            payload,
            method="eth_getLogs",
            block_hex=block_hex,
        )
        if result is None:
            raise RuntimeError(f"RPC request failed for eth_getLogs on block {block_hex}")

        if "error" in result:
            logger.error(
                "RPC error fetching logs",
                error=result["error"],
                block=block_hex,
            )
            return []

        logs = result.get("result", [])
        if isinstance(logs, list):
            return logs
        logger.warning(
            "Unexpected eth_getLogs result shape",
            block=block_hex,
            result_type=type(logs).__name__,
        )
        return []

    async def _rpc_request(
        self,
        payload: dict,
        *,
        method: str,
        block_hex: str = "",
    ) -> Optional[dict]:
        """Send an HTTP JSON-RPC request with endpoint failover."""
        last_error: Optional[Exception] = None
        for endpoint in self._rpc_urls:
            try:
                async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    result = response.json()

                if endpoint != self._http_rpc_url:
                    logger.warning(
                        "Wallet monitor RPC failover",
                        previous_endpoint=self._http_rpc_url,
                        active_endpoint=endpoint,
                    )
                    self._http_rpc_url = endpoint
                    self._rpc_urls = _build_rpc_candidates(self._http_rpc_url)

                if not isinstance(result, dict):
                    logger.warning(
                        "Unexpected RPC response payload",
                        method=method,
                        endpoint=endpoint,
                        response_type=type(result).__name__,
                    )
                    return None
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "Wallet monitor RPC request failed",
                    method=method,
                    endpoint=endpoint,
                    block=block_hex or None,
                    error_type=type(e).__name__,
                    error=_exception_text(e),
                )

        if last_error is not None:
            logger.error(
                "Wallet monitor RPC failed across all endpoints",
                method=method,
                block=block_hex or None,
                endpoints=self._rpc_urls,
                error_type=type(last_error).__name__,
                error=_exception_text(last_error),
            )
        return None

    # ==================== LOG PROCESSING ====================

    async def _process_log(
        self,
        log: dict,
        block_number: int = 0,
        block_timestamp: int = 0,
    ):
        """Process a single OrderFilled event log.

        Parses the event data, checks if the maker or taker is in our
        tracked set, and if so, creates a WalletTradeEvent, persists it,
        and invokes all registered callbacks.

        Args:
            log: Raw event log dict from eth_getLogs.
            block_number: The block number this log belongs to.
            block_timestamp: Unix timestamp of the block.
        """
        detected_at = utcnow()

        parsed = _parse_order_filled_log(log)
        if parsed is None:
            return

        # Check if maker or taker is in our tracked set
        maker_lower = parsed["maker"].lower()
        taker_lower = parsed["taker"].lower()

        matched_wallet = None
        if maker_lower in self._tracked_sources:
            matched_wallet = maker_lower
        elif taker_lower in self._tracked_sources:
            matched_wallet = taker_lower
        else:
            return  # Neither maker nor taker is tracked

        # Determine trade details
        side, token_id, size, price = _determine_trade_side_and_details(
            parsed, matched_wallet
        )

        tx_hash = log.get("transactionHash", "")

        # Calculate block timestamp as datetime
        if block_timestamp > 0:
            block_dt = utcfromtimestamp(block_timestamp)
        else:
            block_dt = detected_at

        # Calculate detection latency
        latency_ms = (detected_at - block_dt).total_seconds() * 1000.0

        trade_event = WalletTradeEvent(
            wallet_address=matched_wallet,
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            tx_hash=tx_hash,
            block_number=block_number,
            timestamp=block_dt,
            detected_at=detected_at,
            latency_ms=max(latency_ms, 0.0),
        )

        self._stats["events_detected"] += 1

        logger.debug(
            "Trade detected via WS",
            wallet=matched_wallet,
            side=side,
            token_id=token_id,
            size=size,
            price=price,
            block=block_number,
            latency_ms=round(latency_ms, 1),
            tx_hash=tx_hash,
        )

        # Persist to database
        await self._persist_event(trade_event)

        # Invoke callbacks
        await self._emit_callbacks(trade_event)

    # ==================== PERSISTENCE ====================

    async def _persist_event(self, event: WalletTradeEvent):
        """Save a wallet trade event to the database.

        Args:
            event: The WalletTradeEvent to persist.
        """
        try:
            async with AsyncSessionLocal() as session:
                db_event = WalletMonitorEvent(
                    id=str(uuid.uuid4()),
                    wallet_address=event.wallet_address,
                    token_id=event.token_id,
                    side=event.side,
                    size=event.size,
                    price=event.price,
                    tx_hash=event.tx_hash,
                    block_number=event.block_number,
                    detection_latency_ms=event.latency_ms,
                    detected_at=event.detected_at,
                )
                session.add(db_event)
                await session.commit()
        except Exception as e:
            logger.error(
                "Failed to persist wallet monitor event",
                error_type=type(e).__name__,
                error=_exception_text(e),
                wallet=event.wallet_address,
                tx_hash=event.tx_hash,
            )

    # ==================== CALLBACKS ====================

    async def _emit_callbacks(self, event: WalletTradeEvent):
        """Invoke all registered callbacks with the trade event.

        Supports both synchronous and asynchronous callbacks.

        Args:
            event: The WalletTradeEvent to emit.
        """
        for callback in self._callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "Callback error",
                    error_type=type(e).__name__,
                    error=_exception_text(e),
                    callback=getattr(callback, "__name__", str(callback)),
                )

    # ==================== POLLING FALLBACK ====================

    async def _poll_fallback_loop(self):
        """Fallback polling loop used when the WebSocket connection is down.

        Polls for the latest block number via HTTP RPC and processes any
        blocks that were missed since the last processed block. Runs
        every 2 seconds until the WebSocket reconnects.
        """
        logger.warning("Starting fallback HTTP polling")
        poll_interval = 2  # seconds

        while self._running:
            try:
                self._stats["fallback_polls"] += 1
                latest_block = await self._get_latest_block_number()

                if latest_block is None:
                    await asyncio.sleep(poll_interval)
                    continue

                if self._last_processed_block is None:
                    # First run in fallback: just process the latest block
                    await self._handle_block(latest_block)
                else:
                    # Process any missed blocks (cap at 10 to avoid overload)
                    start = self._last_processed_block + 1
                    end = min(latest_block, start + 10)

                    for block_num in range(start, end + 1):
                        if not self._running:
                            break
                        await self._handle_block(block_num)

            except asyncio.CancelledError:
                logger.info("Fallback polling cancelled")
                break
            except Exception as e:
                logger.error(
                    "Fallback polling error",
                    error_type=type(e).__name__,
                    error=_exception_text(e),
                )
                self._stats["errors"] += 1

            await asyncio.sleep(poll_interval)

    async def _get_latest_block_number(self) -> Optional[int]:
        """Get the latest block number from the HTTP RPC endpoint.

        Returns:
            The latest block number, or None if the request fails.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_blockNumber",
            "params": [],
        }

        result = await self._rpc_request(payload, method="eth_blockNumber")
        if result is None:
            return None

        if "error" in result:
            logger.error(
                "RPC error getting block number",
                error=result["error"],
            )
            return None

        block_hex = result.get("result")
        if not isinstance(block_hex, str):
            logger.warning(
                "Unexpected eth_blockNumber result shape",
                result_type=type(block_hex).__name__,
            )
            return None
        try:
            return int(block_hex, 16)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to parse block number result",
                raw_value=block_hex,
                error_type=type(e).__name__,
                error=_exception_text(e),
            )
            return None

    # ==================== STATUS ====================

    def get_status(self) -> dict:
        """Get current monitor status and statistics.

        Returns:
            Dict with running state, connection info, tracked wallets,
            and cumulative statistics.
        """
        return {
            "running": self._running,
            "ws_connected": self._ws_connection is not None,
            "ws_url": self._ws_url,
            "tracked_wallets": len(self._tracked_sources),
            "tracked_addresses": list(self._tracked_sources.keys()),
            "tracked_sources": {
                addr: sorted(list(sources))
                for addr, sources in self._tracked_sources.items()
            },
            "last_processed_block": self._last_processed_block,
            "fallback_polling": (
                self._poll_fallback_task is not None
                and not self._poll_fallback_task.done()
            ),
            "stats": dict(self._stats),
        }


# ==================== SINGLETON ====================

wallet_ws_monitor = WalletWebSocketMonitor()
