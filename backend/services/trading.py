"""
Trading Service - Real order execution on Polymarket

This service handles real trading on Polymarket using the CLOB API.
It integrates with py-clob-client for order placement and management.

IMPORTANT: Real trading involves real money. Use with caution.

Setup:
1. Get API credentials from https://polymarket.com/settings/api-keys
2. Set environment variables:
   - POLYMARKET_PRIVATE_KEY
   - POLYMARKET_API_KEY
   - POLYMARKET_API_SECRET
   - POLYMARKET_API_PASSPHRASE
3. Set TRADING_ENABLED=true to enable real trading
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
import uuid

from config import settings
from services.price_chaser import price_chaser
from services.execution_tiers import execution_tier_service
from services.trading_proxy import patch_clob_client_proxy, pre_trade_vpn_check
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    GTC = "GTC"  # Good Till Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date
    FAK = "FAK"  # Fill-and-Kill (immediate partial fill, cancel rest)


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class Order:
    """Represents a trading order"""

    id: str
    token_id: str
    side: OrderSide
    price: float
    size: float  # In shares
    order_type: OrderType = OrderType.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    clob_order_id: Optional[str] = None
    error_message: Optional[str] = None
    market_question: Optional[str] = None
    opportunity_id: Optional[str] = None


@dataclass
class Position:
    """Represents an open position"""

    token_id: str
    market_id: str
    market_question: str
    outcome: str  # YES or NO
    size: float  # Number of shares
    average_cost: float  # Average price paid
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradingStats:
    """Trading statistics"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    daily_volume: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    last_trade_at: Optional[datetime] = None


class TradingService:
    """
    Service for executing real trades on Polymarket.

    Uses the py-clob-client library for order placement.
    Implements safety limits and tracking.
    """

    def __init__(self):
        self._initialized = False
        self._client = None
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._stats = TradingStats()
        self._daily_volume_reset = datetime.utcnow().date()
        self._market_positions: dict[str, float] = {}  # token_id -> USD exposure
        self.MAX_PER_MARKET_USD = settings.MAX_PER_MARKET_USD

    async def initialize(self) -> bool:
        """
        Initialize the trading client with API credentials.

        Returns True if successfully initialized, False otherwise.
        """
        if not settings.TRADING_ENABLED:
            logger.warning("Trading is disabled. Set TRADING_ENABLED=true to enable.")
            return False

        if not all(
            [
                settings.POLYMARKET_PRIVATE_KEY,
                settings.POLYMARKET_API_KEY,
                settings.POLYMARKET_API_SECRET,
                settings.POLYMARKET_API_PASSPHRASE,
            ]
        ):
            logger.error(
                "Missing Polymarket API credentials. Cannot initialize trading."
            )
            return False

        try:
            # Import py-clob-client
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            # Create API credentials
            creds = ApiCreds(
                api_key=settings.POLYMARKET_API_KEY,
                api_secret=settings.POLYMARKET_API_SECRET,
                api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
            )

            # Initialize client
            self._client = ClobClient(
                host=settings.CLOB_API_URL,
                key=settings.POLYMARKET_PRIVATE_KEY,
                chain_id=settings.CHAIN_ID,
                creds=creds,
            )

            # Route trading HTTP requests through VPN proxy if configured
            if settings.TRADING_PROXY_ENABLED:
                patched = patch_clob_client_proxy()
                if patched:
                    logger.info("Trading requests will be routed through VPN proxy")
                else:
                    logger.warning(
                        "Trading proxy enabled but patch failed — "
                        "trades will use direct connection"
                    )

            self._initialized = True
            logger.info("Trading service initialized successfully")
            return True

        except ImportError:
            logger.error(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize trading client: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if trading service is ready"""
        return self._initialized and self._client is not None

    def _check_daily_reset(self):
        """Reset daily counters if it's a new day"""
        today = datetime.utcnow().date()
        if today != self._daily_volume_reset:
            self._stats.daily_volume = 0.0
            self._stats.daily_pnl = 0.0
            self._daily_volume_reset = today

    def _validate_order(
        self, size_usd: float, token_id: str = None
    ) -> tuple[bool, str]:
        """
        Validate order against safety limits.

        Returns (is_valid, error_message)
        """
        self._check_daily_reset()

        if not self.is_ready():
            return False, "Trading service not initialized"

        if not settings.TRADING_ENABLED:
            return False, "Trading is disabled"

        if size_usd < settings.MIN_ORDER_SIZE_USD:
            return (
                False,
                f"Order size ${size_usd:.2f} below minimum ${settings.MIN_ORDER_SIZE_USD:.2f}",
            )

        if size_usd > settings.MAX_TRADE_SIZE_USD:
            return (
                False,
                f"Order size ${size_usd:.2f} exceeds maximum ${settings.MAX_TRADE_SIZE_USD:.2f}",
            )

        projected_daily_volume = self._stats.daily_volume + size_usd
        if projected_daily_volume > settings.MAX_DAILY_TRADE_VOLUME:
            return (
                False,
                f"Would exceed daily volume limit (${projected_daily_volume:.2f} > ${settings.MAX_DAILY_TRADE_VOLUME:.2f})",
            )

        if len(self._positions) >= settings.MAX_OPEN_POSITIONS:
            return (
                False,
                f"Maximum open positions ({settings.MAX_OPEN_POSITIONS}) reached",
            )

        # Per-market position limit
        if token_id and token_id in self._market_positions:
            current = self._market_positions[token_id]
            if current + size_usd > self.MAX_PER_MARKET_USD:
                return (
                    False,
                    f"Per-market limit: ${current:.2f} + ${size_usd:.2f} exceeds ${self.MAX_PER_MARKET_USD:.2f}",
                )

        return True, ""

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.GTC,
        market_question: Optional[str] = None,
        opportunity_id: Optional[str] = None,
    ) -> Order:
        """
        Place an order on Polymarket.

        Args:
            token_id: The CLOB token ID (YES or NO token)
            side: BUY or SELL
            price: Price per share (0-1)
            size: Number of shares
            order_type: GTC, FOK, or GTD
            market_question: Optional market question for reference
            opportunity_id: Optional opportunity ID this trade is from

        Returns:
            Order object with status
        """
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            market_question=market_question,
            opportunity_id=opportunity_id,
        )

        # Calculate USD value
        size_usd = price * size

        # VPN pre-trade check (blocks if VPN required but unreachable)
        if settings.TRADING_PROXY_ENABLED and settings.TRADING_PROXY_REQUIRE_VPN:
            vpn_ok, vpn_reason = await pre_trade_vpn_check()
            if not vpn_ok:
                order.status = OrderStatus.FAILED
                order.error_message = f"VPN check failed: {vpn_reason}"
                self._orders[order_id] = order
                logger.error(f"Trade blocked by VPN check: {vpn_reason}")
                return order

        # Validate order
        is_valid, error = self._validate_order(size_usd, token_id=token_id)
        if not is_valid:
            order.status = OrderStatus.FAILED
            order.error_message = error
            self._orders[order_id] = order
            logger.warning(f"Order validation failed: {error}")
            return order

        try:
            # Build and sign order using py-clob-client
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL

            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY if side == OrderSide.BUY else SELL,
                token_id=token_id,
            )

            # Create and sign the order
            signed_order = self._client.create_order(order_args)

            # Post order to CLOB
            response = self._client.post_order(signed_order, order_type.value)

            if response.get("success"):
                order.status = OrderStatus.OPEN
                order.clob_order_id = response.get("orderID")
                self._stats.total_trades += 1
                self._stats.daily_volume += size_usd
                self._stats.total_volume += size_usd
                self._stats.last_trade_at = datetime.utcnow()
                logger.info(f"Order placed successfully: {order.clob_order_id}")

                # Update per-market position tracking
                if token_id not in self._market_positions:
                    self._market_positions[token_id] = 0.0
                self._market_positions[token_id] += size_usd
            else:
                order.status = OrderStatus.FAILED
                order.error_message = response.get("errorMsg", "Unknown error")
                logger.error(f"Order failed: {order.error_message}")

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            logger.error(f"Order execution error: {e}")

        order.updated_at = datetime.utcnow()
        self._orders[order_id] = order
        return order

    async def place_order_with_chase(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        tier: int = 2,
        order_type: OrderType = OrderType.GTC,
        market_question: Optional[str] = None,
        opportunity_id: Optional[str] = None,
    ) -> Order:
        """
        Place an order with price chasing retries.

        Uses the PriceChaserService to automatically adjust the price
        on each retry attempt, improving fill rates in fast-moving markets.

        Args:
            token_id: The CLOB token ID
            side: BUY or SELL
            price: Initial price per share
            size: Number of shares
            tier: Execution tier (1-4) for retry config
            order_type: Default order type
            market_question: Optional market reference
            opportunity_id: Optional opportunity ID
        """
        # Get tier config for max retries
        tier_config = execution_tier_service.TIERS.get(tier)
        if tier_config:
            from services.price_chaser import PriceChaseConfig

            chase_config = PriceChaseConfig(
                max_retries=tier_config.max_retries,
                max_slippage_percent=settings.MAX_SLIPPAGE_PERCENT,
            )
            chaser = price_chaser.__class__(config=chase_config)
        else:
            chaser = price_chaser

        async def _place_fn(token_id, side_str, adj_price, adj_size, order_type_str):
            ot = OrderType(order_type_str) if order_type_str else order_type
            os_side = OrderSide(side_str) if isinstance(side_str, str) else side
            return await self.place_order(
                token_id=token_id,
                side=os_side,
                price=adj_price,
                size=adj_size,
                order_type=ot,
                market_question=market_question,
                opportunity_id=opportunity_id,
            )

        async def _get_price_fn(tid, s):
            from services.polymarket import polymarket_client

            return await polymarket_client.get_price(tid, side=s)

        result = await chaser.execute_with_chase(
            token_id=token_id,
            side=side.value,
            price=price,
            size=size,
            place_order_fn=_place_fn,
            get_market_price_fn=_get_price_fn,
            opportunity_id=opportunity_id,
            tier=tier,
        )

        if result.get("success") and result.get("final_order"):
            return result["final_order"]

        # Fallback: return a failed order if chase didn't succeed
        return Order(
            id=str(uuid.uuid4()),
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            status=OrderStatus.FAILED,
            error_message=f"Price chase failed after {result.get('total_attempts', 0)} attempts",
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        if order_id not in self._orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        order = self._orders[order_id]
        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            logger.warning(f"Cannot cancel order in status: {order.status}")
            return False

        if not order.clob_order_id:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            return True

        try:
            response = self._client.cancel(order.clob_order_id)
            if response.get("canceled"):
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order: {response}")
                return False
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        cancelled = 0
        try:
            response = self._client.cancel_all()
            cancelled = len(response.get("canceled", []))

            # Update local order status
            for order in self._orders.values():
                if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.utcnow()

            logger.info(f"Cancelled {cancelled} orders")
        except Exception as e:
            logger.error(f"Cancel all orders error: {e}")

        return cancelled

    async def get_open_orders(self) -> list[Order]:
        """Get all open orders"""
        if not self.is_ready():
            return [o for o in self._orders.values() if o.status == OrderStatus.OPEN]

        try:
            response = self._client.get_orders()
            # Update local orders with server state
            for server_order in response:
                clob_id = server_order.get("id")
                for order in self._orders.values():
                    if order.clob_order_id == clob_id:
                        # Update fill status
                        order.filled_size = float(server_order.get("size_matched", 0))
                        if order.filled_size >= order.size:
                            order.status = OrderStatus.FILLED
                        elif order.filled_size > 0:
                            order.status = OrderStatus.PARTIALLY_FILLED
                        order.updated_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"Get orders error: {e}")

        return [
            o
            for o in self._orders.values()
            if o.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]

    async def sync_positions(self) -> list[Position]:
        """Sync positions from Polymarket"""
        if not self.is_ready():
            return list(self._positions.values())

        try:
            # Get positions from the wallet
            # Note: This uses the data API, not CLOB
            from services.polymarket import polymarket_client

            address = self._get_wallet_address()
            if not address:
                return list(self._positions.values())

            positions_data = await polymarket_client.get_wallet_positions(address)

            self._positions.clear()
            for pos in positions_data:
                token_id = pos.get("asset")
                position = Position(
                    token_id=token_id,
                    market_id=pos.get("market", ""),
                    market_question=pos.get("title", "Unknown"),
                    outcome=pos.get("outcome", ""),
                    size=float(pos.get("size", 0)),
                    average_cost=float(pos.get("avgCost", 0)),
                    current_price=float(pos.get("currentPrice", 0)),
                )
                position.unrealized_pnl = (
                    position.current_price - position.average_cost
                ) * position.size
                self._positions[token_id] = position

            self._stats.open_positions = len(self._positions)

        except Exception as e:
            logger.error(f"Sync positions error: {e}")

        return list(self._positions.values())

    def _get_wallet_address(self) -> Optional[str]:
        """Get wallet address from private key"""
        if not settings.POLYMARKET_PRIVATE_KEY:
            return None
        try:
            from eth_account import Account

            account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
            return account.address
        except Exception:
            return None

    async def execute_opportunity(
        self, opportunity_id: str, positions: list[dict], size_usd: float
    ) -> list[Order]:
        """
        Execute an arbitrage opportunity with PARALLEL order submission.

        Critical insight from research: CLOB execution is sequential, not atomic.
        If you execute orders one-by-one, prices move between legs, eating profits.

        This method submits ALL orders in parallel via asyncio.gather so they're
        included in the same block (~2 seconds on Polygon), eliminating sequential
        execution risk.

        Args:
            opportunity_id: ID of the opportunity
            positions: List of positions to take (from opportunity.positions_to_take)
            size_usd: Total USD amount to invest

        Returns:
            List of orders placed
        """

        # Pre-validate all positions before any execution
        valid_positions = []
        for position in positions:
            token_id = position.get("token_id")
            if not token_id:
                logger.warning(f"Position missing token_id: {position}")
                continue
            price = position.get("price", 0)
            if price <= 0:
                logger.warning(f"Invalid price {price} for {token_id}")
                continue
            valid_positions.append(position)

        if not valid_positions:
            logger.error("No valid positions to execute")
            return []

        # Build order coroutines for parallel execution
        async def place_single_order(position: dict) -> Order:
            token_id = position.get("token_id")
            price = position.get("price")
            position_usd = size_usd / len(valid_positions)
            shares = position_usd / price

            return await self.place_order(
                token_id=token_id,
                side=OrderSide.BUY,
                price=price,
                size=shares,
                market_question=position.get("market"),
                opportunity_id=opportunity_id,
            )

        # Execute ALL orders in PARALLEL - this is the critical change
        # asyncio.gather submits all coroutines before any await completes
        tasks = [place_single_order(pos) for pos in valid_positions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        orders = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Order failed with exception: {result}")
                failed_count += 1
            elif isinstance(result, Order):
                orders.append(result)
                if result.status == OrderStatus.FAILED:
                    failed_count += 1
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                failed_count += 1

        # Warn about partial execution (exposure risk)
        if 0 < failed_count < len(valid_positions):
            logger.warning(
                f"PARTIAL EXECUTION: {len(orders) - failed_count}/{len(valid_positions)} legs filled. "
                f"Position has EXPOSURE RISK!"
            )
            # Auto-reconcile: unwind filled legs from failed arbitrage
            asyncio.create_task(
                self._auto_reconcile(orders, valid_positions, failed_count)
            )

        return orders

    async def _auto_reconcile(self, orders: list, positions: list, failed_count: int):
        """Auto-unwind partial multi-leg fills to prevent one-sided exposure."""
        await asyncio.sleep(2)  # Brief delay before reconciliation
        logger.info(
            f"AUTO_RECONCILE: Unwinding {len(orders) - failed_count} filled legs"
        )
        for order in orders:
            if order.status in (
                OrderStatus.OPEN,
                OrderStatus.FILLED,
                OrderStatus.PARTIALLY_FILLED,
            ):
                if order.filled_size > 0:
                    try:
                        unwind = await self.place_order(
                            token_id=order.token_id,
                            side=OrderSide.SELL
                            if order.side == OrderSide.BUY
                            else OrderSide.BUY,
                            price=order.price * 0.95
                            if order.side == OrderSide.BUY
                            else order.price * 1.05,
                            size=order.filled_size,
                            order_type=OrderType.FOK,
                            market_question=f"AUTO_RECONCILE: {order.market_question}",
                        )
                        logger.info(
                            f"Reconciliation order placed: {unwind.status.value}"
                        )
                    except Exception as e:
                        logger.error(f"Reconciliation failed: {e}")

    def get_stats(self) -> TradingStats:
        """Get trading statistics"""
        self._check_daily_reset()
        return self._stats

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    def get_orders(self, limit: int = 100) -> list[Order]:
        """Get recent orders"""
        orders = sorted(self._orders.values(), key=lambda x: x.created_at, reverse=True)
        return orders[:limit]

    def get_positions(self) -> list[Position]:
        """Get current positions"""
        return list(self._positions.values())

    async def get_balance(self) -> dict:
        """Get wallet balance"""
        if not self.is_ready():
            return {"error": "Trading not initialized"}

        try:
            # Get USDC balance on Polygon
            address = self._get_wallet_address()
            if not address:
                return {"error": "Could not derive wallet address"}

            # This would need web3 integration to check actual balance
            # For now, return placeholder
            return {
                "address": address,
                "usdc_balance": 0.0,  # Would need web3 call
                "positions_value": sum(
                    p.size * p.current_price for p in self._positions.values()
                ),
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
trading_service = TradingService()
