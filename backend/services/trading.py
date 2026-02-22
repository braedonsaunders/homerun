"""
Trading Service - Real order execution on Polymarket

This service handles real trading on Polymarket using the CLOB API.
It integrates with py-clob-client for order placement and management.

IMPORTANT: Real trading involves real money. Use with caution.

Setup:
1. Get API credentials from https://polymarket.com/settings/api-keys
2. Provide credentials in Settings (DB-backed) or environment variables:
   - POLYMARKET_PRIVATE_KEY
   - POLYMARKET_API_KEY
   - POLYMARKET_API_SECRET
   - POLYMARKET_API_PASSPHRASE
3. Set TRADING_ENABLED=true to enable real trading
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from utils.utcnow import utcnow
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal
import uuid

from sqlalchemy import delete, select
from sqlalchemy.exc import InterfaceError, OperationalError

from config import settings
from services.pause_state import global_pause_state
from services.price_chaser import price_chaser
from services.execution_tiers import execution_tier_service
from services.trading_proxy import (
    patch_clob_client_proxy,
    pre_trade_vpn_check,
    _load_config_from_db as load_proxy_config,
)
from utils.logger import get_logger
from utils.secrets import decrypt_secret
from utils.converters import safe_float

logger = get_logger(__name__)

ZERO = Decimal("0")
USDC_BASE_UNITS = Decimal("1000000")
POLYMARKET_SIGNATURE_TYPES = (0, 1, 2)
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BASE_DELAY_SECONDS = 0.05
DB_RETRY_MAX_DELAY_SECONDS = 0.3


def _to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _first_float(data: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        parsed = safe_float(data.get(key))
        if parsed is not None:
            return float(parsed)
    return None


def _parse_collateral_amount(value: Any, *, assume_base_units: bool = False) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = Decimal(raw)
    except Exception:
        parsed_float = safe_float(value)
        if parsed_float is None:
            return None
        return float(parsed_float)
    if assume_base_units:
        return float(parsed / USDC_BASE_UNITS)
    return float(parsed)


def _is_retryable_db_error(exc: Exception) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    return any(
        marker in message
        for marker in (
            "deadlock detected",
            "serialization failure",
            "could not serialize access",
            "lock not available",
            "connection is closed",
            "underlying connection is closed",
            "connection has been closed",
            "closed the connection unexpectedly",
            "terminating connection",
            "connection reset by peer",
            "broken pipe",
        )
    )


def _db_retry_delay(attempt: int) -> float:
    return min(DB_RETRY_BASE_DELAY_SECONDS * (2**attempt), DB_RETRY_MAX_DELAY_SECONDS)


def _normalize_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_provider_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        normalized = _normalize_utc_datetime(value)
        return normalized if normalized is not None else utcnow()
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return utcnow()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        normalized = _normalize_utc_datetime(parsed)
        return normalized if normalized is not None else utcnow()
    except Exception:
        return utcnow()


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
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
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
    created_at: datetime = field(default_factory=utcnow)


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
        self._wallet_address: Optional[str] = None
        self._orders: OrderedDict[str, Order] = OrderedDict()
        self._positions: dict[str, Position] = {}
        self._stats = TradingStats()
        self._daily_volume_reset = utcnow().date()
        self._market_positions: OrderedDict[str, Decimal] = OrderedDict()  # token_id -> USD exposure
        self._stats_lock: Optional[asyncio.Lock] = None
        self._init_lock: Optional[asyncio.Lock] = None
        self._persist_lock: Optional[asyncio.Lock] = None
        self._balance_signature_type: Optional[int] = None
        self._runtime_state_loaded_for_wallet: Optional[str] = None
        self._daily_volume = ZERO
        self._daily_pnl = ZERO
        self._total_volume = ZERO
        self._total_pnl = ZERO
        self.MAX_PER_MARKET_USD = settings.MAX_PER_MARKET_USD
        self._max_order_history = max(
            100,
            int(getattr(settings, "TRADING_ORDER_HISTORY_LIMIT", 5000)),
        )
        self._max_market_position_entries = max(
            100,
            int(getattr(settings, "TRADING_MARKET_POSITION_LIMIT", 5000)),
        )

    def _get_stats_lock(self) -> asyncio.Lock:
        if self._stats_lock is None:
            self._stats_lock = asyncio.Lock()
        return self._stats_lock

    def _get_init_lock(self) -> asyncio.Lock:
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    def _get_persist_lock(self) -> asyncio.Lock:
        if self._persist_lock is None:
            self._persist_lock = asyncio.Lock()
        return self._persist_lock

    async def _load_db_polymarket_credentials(
        self,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        try:
            from sqlalchemy import select
            from models.database import AsyncSessionLocal, AppSettings

            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                row = result.scalar_one_or_none()
                if row is None:
                    return None, None, None, None
                return (
                    decrypt_secret(row.polymarket_private_key) or None,
                    decrypt_secret(row.polymarket_api_key) or None,
                    decrypt_secret(row.polymarket_api_secret) or None,
                    decrypt_secret(row.polymarket_api_passphrase) or None,
                )
        except Exception as e:
            logger.error("Failed to load Polymarket credentials from DB", exc_info=e)
            return None, None, None, None

    async def _resolve_polymarket_credentials(
        self,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
        db_creds = await self._load_db_polymarket_credentials()
        env_creds = (
            settings.POLYMARKET_PRIVATE_KEY,
            settings.POLYMARKET_API_KEY,
            settings.POLYMARKET_API_SECRET,
            settings.POLYMARKET_API_PASSPHRASE,
        )
        if all(db_creds):
            return (*db_creds, "db")
        if all(env_creds):
            return (*env_creds, "env")

        mixed = tuple(db_value or env_value for db_value, env_value in zip(db_creds, env_creds))
        if all(mixed):
            private_key, api_key, api_secret, api_passphrase = mixed
            return private_key, api_key, api_secret, api_passphrase, "mixed"
        return None, None, None, None, "missing"

    async def _sync_trading_transport(self) -> bool:
        await load_proxy_config()
        return patch_clob_client_proxy()

    async def ensure_initialized(self) -> bool:
        if self.is_ready():
            await self._sync_trading_transport()
            return True
        return await self.initialize()

    async def initialize(self) -> bool:
        """
        Initialize the trading client with API credentials.

        Returns True if successfully initialized, False otherwise.
        """
        init_lock = self._get_init_lock()
        async with init_lock:
            if not settings.TRADING_ENABLED:
                logger.warning("Trading is disabled. Set TRADING_ENABLED=true to enable.")
                return False

            private_key, api_key, api_secret, api_passphrase, credential_source = await self._resolve_polymarket_credentials()
            if not all([private_key, api_key, api_secret, api_passphrase]):
                logger.error("Missing Polymarket API credentials. Cannot initialize trading.")
                return False

            try:
                # Import py-clob-client
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds
                from eth_account import Account

                # Create API credentials
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )

                # Initialize client
                self._client = ClobClient(
                    host=settings.CLOB_API_URL,
                    key=private_key,
                    chain_id=settings.CHAIN_ID,
                    creds=creds,
                )
                self._wallet_address = Account.from_key(private_key).address
                self._initialized = True

                proxy_cfg = await load_proxy_config()
                patched = patch_clob_client_proxy()
                if patched and proxy_cfg.enabled and proxy_cfg.proxy_url:
                    logger.info("Trading requests will be routed through VPN proxy")
                elif patched:
                    logger.info("Trading requests will use direct connection")
                else:
                    logger.warning("Trading HTTP transport patch failed; using py-clob-client default transport")

                await self._restore_runtime_state()
                logger.info("Trading service initialized successfully", credential_source=credential_source)
                return True

            except ImportError:
                logger.error("py-clob-client not installed. Run: pip install py-clob-client")
                self._initialized = False
                self._client = None
                self._wallet_address = None
                return False
            except Exception as e:
                logger.error(f"Failed to initialize trading client: {e}")
                self._initialized = False
                self._client = None
                self._wallet_address = None
                return False

    def is_ready(self) -> bool:
        """Check if trading service is ready"""
        return self._initialized and self._client is not None

    def _sync_stats_from_decimals(self) -> None:
        self._stats.total_volume = float(self._total_volume)
        self._stats.total_pnl = float(self._total_pnl)
        self._stats.daily_volume = float(self._daily_volume)
        self._stats.daily_pnl = float(self._daily_pnl)

    def _prune_order_cache(self) -> None:
        if len(self._orders) <= self._max_order_history:
            return

        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        }
        for order_id, cached_order in list(self._orders.items()):
            if len(self._orders) <= self._max_order_history:
                break
            if cached_order.status not in active_statuses:
                self._orders.pop(order_id, None)

        while len(self._orders) > self._max_order_history:
            self._orders.popitem(last=False)

    def _remember_order(self, order: Order) -> None:
        self._orders[order.id] = order
        self._orders.move_to_end(order.id)
        self._prune_order_cache()

    def _runtime_state_id(self, wallet_address: str) -> str:
        return f"wallet:{wallet_address.lower()}"

    def _wallet_for_persistence(self) -> Optional[str]:
        wallet = str(self._wallet_address or "").strip()
        if wallet:
            return wallet.lower()
        derived = self._get_wallet_address()
        if not derived:
            return None
        return str(derived).strip().lower()

    async def _persist_orders(self, orders: list[Order]) -> None:
        if not orders:
            return
        wallet = self._wallet_for_persistence()
        if not wallet:
            return

        from models.database import AsyncSessionLocal, LiveTradingOrder

        unique_orders: dict[str, Order] = {}
        for order in orders:
            unique_orders[str(order.id)] = order
        order_ids = list(unique_orders)
        if not order_ids:
            return

        persist_lock = self._get_persist_lock()
        async with persist_lock:
            for attempt in range(DB_RETRY_ATTEMPTS):
                async with AsyncSessionLocal() as session:
                    try:
                        existing_result = await session.execute(
                            select(LiveTradingOrder).where(LiveTradingOrder.id.in_(order_ids))
                        )
                        existing_rows = {row.id: row for row in existing_result.scalars().all()}
                        for order in unique_orders.values():
                            row = existing_rows.get(order.id)
                            if row is None:
                                row = LiveTradingOrder(id=order.id, wallet_address=wallet)
                                session.add(row)
                            created_at = _normalize_utc_datetime(order.created_at) or utcnow()
                            updated_at = _normalize_utc_datetime(order.updated_at) or utcnow()
                            row.wallet_address = wallet
                            row.clob_order_id = str(order.clob_order_id or "").strip() or None
                            row.token_id = str(order.token_id or "").strip()
                            row.side = order.side.value
                            row.price = float(order.price)
                            row.size = float(order.size)
                            row.order_type = order.order_type.value
                            row.status = order.status.value
                            row.filled_size = float(order.filled_size)
                            row.average_fill_price = float(order.average_fill_price)
                            row.market_question = order.market_question
                            row.opportunity_id = order.opportunity_id
                            row.error_message = order.error_message
                            row.created_at = created_at
                            row.updated_at = updated_at
                        await session.commit()
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= DB_RETRY_ATTEMPTS - 1
                        if not _is_retryable_db_error(exc) or is_last:
                            logger.error("Failed to persist live trading orders", exc_info=exc)
                            return
                        await asyncio.sleep(_db_retry_delay(attempt))
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Failed to persist live trading orders", exc_info=exc)
                        return

    async def _persist_positions(self) -> None:
        wallet = self._wallet_for_persistence()
        if not wallet:
            return

        from models.database import AsyncSessionLocal, LiveTradingPosition

        positions = list(self._positions.values())
        persist_lock = self._get_persist_lock()
        async with persist_lock:
            for attempt in range(DB_RETRY_ATTEMPTS):
                async with AsyncSessionLocal() as session:
                    try:
                        await session.execute(
                            delete(LiveTradingPosition).where(LiveTradingPosition.wallet_address == wallet)
                        )
                        for position in positions:
                            token_id = str(position.token_id or "").strip()
                            if not token_id:
                                continue
                            row = LiveTradingPosition(
                                id=f"{wallet}:{token_id}",
                                wallet_address=wallet,
                                token_id=token_id,
                                market_id=str(position.market_id or "").strip(),
                                market_question=position.market_question,
                                outcome=position.outcome,
                                size=float(position.size),
                                average_cost=float(position.average_cost),
                                current_price=float(position.current_price),
                                unrealized_pnl=float(position.unrealized_pnl),
                                created_at=_normalize_utc_datetime(position.created_at) or utcnow(),
                                updated_at=utcnow(),
                            )
                            session.add(row)
                        await session.commit()
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= DB_RETRY_ATTEMPTS - 1
                        if not _is_retryable_db_error(exc) or is_last:
                            logger.error("Failed to persist live trading positions", exc_info=exc)
                            return
                        await asyncio.sleep(_db_retry_delay(attempt))
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Failed to persist live trading positions", exc_info=exc)
                        return

    async def _persist_runtime_state(self) -> None:
        wallet = self._wallet_for_persistence()
        if not wallet:
            return

        from models.database import AsyncSessionLocal, LiveTradingRuntimeState

        runtime_id = self._runtime_state_id(wallet)
        last_trade_at = _normalize_utc_datetime(self._stats.last_trade_at)
        daily_reset_at = datetime.combine(self._daily_volume_reset, datetime.min.time(), tzinfo=timezone.utc)
        market_positions_json = {
            str(token_id): str(exposure)
            for token_id, exposure in self._market_positions.items()
        }

        persist_lock = self._get_persist_lock()
        async with persist_lock:
            for attempt in range(DB_RETRY_ATTEMPTS):
                async with AsyncSessionLocal() as session:
                    try:
                        result = await session.execute(
                            select(LiveTradingRuntimeState).where(LiveTradingRuntimeState.id == runtime_id)
                        )
                        row = result.scalar_one_or_none()
                        if row is None:
                            row = LiveTradingRuntimeState(id=runtime_id, wallet_address=wallet)
                            session.add(row)

                        row.wallet_address = wallet
                        row.total_trades = int(self._stats.total_trades)
                        row.winning_trades = int(self._stats.winning_trades)
                        row.losing_trades = int(self._stats.losing_trades)
                        row.total_volume = float(self._total_volume)
                        row.total_pnl = float(self._total_pnl)
                        row.daily_volume = float(self._daily_volume)
                        row.daily_pnl = float(self._daily_pnl)
                        row.open_positions = int(self._stats.open_positions)
                        row.last_trade_at = last_trade_at
                        row.daily_volume_reset_at = daily_reset_at
                        row.market_positions_json = market_positions_json
                        row.balance_signature_type = self._balance_signature_type
                        row.updated_at = utcnow()
                        await session.commit()
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= DB_RETRY_ATTEMPTS - 1
                        if not _is_retryable_db_error(exc) or is_last:
                            logger.error("Failed to persist live trading runtime state", exc_info=exc)
                            return
                        await asyncio.sleep(_db_retry_delay(attempt))
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Failed to persist live trading runtime state", exc_info=exc)
                        return

    async def _restore_runtime_state(self) -> None:
        wallet = self._wallet_for_persistence()
        if not wallet:
            return
        if self._runtime_state_loaded_for_wallet == wallet:
            return

        from models.database import (
            AsyncSessionLocal,
            LiveTradingOrder,
            LiveTradingPosition,
            LiveTradingRuntimeState,
        )

        persist_lock = self._get_persist_lock()
        async with persist_lock:
            for attempt in range(DB_RETRY_ATTEMPTS):
                async with AsyncSessionLocal() as session:
                    try:
                        runtime_id = self._runtime_state_id(wallet)
                        runtime_result = await session.execute(
                            select(LiveTradingRuntimeState).where(LiveTradingRuntimeState.id == runtime_id)
                        )
                        runtime_row = runtime_result.scalar_one_or_none()

                        self._orders.clear()
                        orders_result = await session.execute(
                            select(LiveTradingOrder)
                            .where(LiveTradingOrder.wallet_address == wallet)
                            .order_by(LiveTradingOrder.created_at.desc())
                            .limit(self._max_order_history)
                        )
                        persisted_orders = list(orders_result.scalars().all())
                        persisted_orders.reverse()
                        for row in persisted_orders:
                            side_raw = str(row.side or "").strip().upper()
                            side = OrderSide.SELL if side_raw == OrderSide.SELL.value else OrderSide.BUY
                            order_type_raw = str(row.order_type or "").strip().upper()
                            try:
                                order_type = OrderType(order_type_raw)
                            except ValueError:
                                order_type = OrderType.GTC
                            status_raw = str(row.status or "").strip().lower()
                            try:
                                status = OrderStatus(status_raw)
                            except ValueError:
                                status = OrderStatus.PENDING
                            order = Order(
                                id=str(row.id),
                                token_id=str(row.token_id or ""),
                                side=side,
                                price=float(safe_float(row.price, 0.0) or 0.0),
                                size=float(safe_float(row.size, 0.0) or 0.0),
                                order_type=order_type,
                                status=status,
                                filled_size=float(safe_float(row.filled_size, 0.0) or 0.0),
                                average_fill_price=float(safe_float(row.average_fill_price, 0.0) or 0.0),
                                created_at=_normalize_utc_datetime(row.created_at) or utcnow(),
                                updated_at=_normalize_utc_datetime(row.updated_at) or utcnow(),
                                clob_order_id=str(row.clob_order_id or "").strip() or None,
                                error_message=row.error_message,
                                market_question=row.market_question,
                                opportunity_id=row.opportunity_id,
                            )
                            self._remember_order(order)

                        self._positions.clear()
                        positions_result = await session.execute(
                            select(LiveTradingPosition).where(LiveTradingPosition.wallet_address == wallet)
                        )
                        for row in positions_result.scalars().all():
                            token_id = str(row.token_id or "").strip()
                            if not token_id:
                                continue
                            self._positions[token_id] = Position(
                                token_id=token_id,
                                market_id=str(row.market_id or ""),
                                market_question=row.market_question or "Unknown",
                                outcome=row.outcome or "",
                                size=float(safe_float(row.size, 0.0) or 0.0),
                                average_cost=float(safe_float(row.average_cost, 0.0) or 0.0),
                                current_price=float(safe_float(row.current_price, 0.0) or 0.0),
                                unrealized_pnl=float(safe_float(row.unrealized_pnl, 0.0) or 0.0),
                                created_at=_normalize_utc_datetime(row.created_at) or utcnow(),
                            )

                        if runtime_row is not None:
                            self._stats.total_trades = int(runtime_row.total_trades or 0)
                            self._stats.winning_trades = int(runtime_row.winning_trades or 0)
                            self._stats.losing_trades = int(runtime_row.losing_trades or 0)
                            self._stats.total_volume = float(safe_float(runtime_row.total_volume, 0.0) or 0.0)
                            self._stats.total_pnl = float(safe_float(runtime_row.total_pnl, 0.0) or 0.0)
                            self._stats.daily_volume = float(safe_float(runtime_row.daily_volume, 0.0) or 0.0)
                            self._stats.daily_pnl = float(safe_float(runtime_row.daily_pnl, 0.0) or 0.0)
                            self._stats.open_positions = int(runtime_row.open_positions or len(self._positions))
                            self._stats.last_trade_at = _normalize_utc_datetime(runtime_row.last_trade_at)
                            self._total_volume = _to_decimal(runtime_row.total_volume or 0.0)
                            self._total_pnl = _to_decimal(runtime_row.total_pnl or 0.0)
                            self._daily_volume = _to_decimal(runtime_row.daily_volume or 0.0)
                            self._daily_pnl = _to_decimal(runtime_row.daily_pnl or 0.0)
                            daily_reset = _normalize_utc_datetime(runtime_row.daily_volume_reset_at)
                            if daily_reset is not None:
                                self._daily_volume_reset = daily_reset.date()
                            self._market_positions.clear()
                            if isinstance(runtime_row.market_positions_json, dict):
                                for token_id, raw_exposure in runtime_row.market_positions_json.items():
                                    token_key = str(token_id or "").strip()
                                    if not token_key:
                                        continue
                                    exposure = safe_float(raw_exposure)
                                    if exposure is None or exposure <= 0:
                                        continue
                                    self._market_positions[token_key] = _to_decimal(exposure)
                                    self._market_positions.move_to_end(token_key)
                                self._prune_market_positions()
                            if runtime_row.balance_signature_type is not None:
                                self._balance_signature_type = int(runtime_row.balance_signature_type)
                        else:
                            self._stats.open_positions = len(self._positions)

                        self._runtime_state_loaded_for_wallet = wallet
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= DB_RETRY_ATTEMPTS - 1
                        if not _is_retryable_db_error(exc) or is_last:
                            logger.error("Failed to restore live trading runtime state", exc_info=exc)
                            return
                        await asyncio.sleep(_db_retry_delay(attempt))
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Failed to restore live trading runtime state", exc_info=exc)
                        return

    def _prune_market_positions(self) -> None:
        while len(self._market_positions) > self._max_market_position_entries:
            self._market_positions.popitem(last=False)

    def _apply_market_exposure_delta(
        self,
        token_id: Optional[str],
        delta_usd: Decimal,
    ) -> None:
        if not token_id:
            return
        current = self._market_positions.get(token_id, ZERO)
        updated = current + delta_usd
        if updated <= ZERO:
            self._market_positions.pop(token_id, None)
            return
        self._market_positions[token_id] = updated
        self._market_positions.move_to_end(token_id)
        self._prune_market_positions()

    def _check_daily_reset(self) -> None:
        """Reset daily counters if it's a new day."""
        today = utcnow().date()
        if today != self._daily_volume_reset:
            self._daily_volume = ZERO
            self._daily_pnl = ZERO
            self._daily_volume_reset = today
            self._sync_stats_from_decimals()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_runtime_state())
            except RuntimeError:
                pass

    def _extract_server_orders(self, response: Any) -> list[dict[str, Any]]:
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        if isinstance(response, dict):
            for key in ("orders", "data", "items", "results"):
                items = response.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
            if "id" in response or "orderID" in response or "order_id" in response:
                return [response]
        return []

    def _normalize_provider_order_status(self, status: Any) -> str:
        status_key = str(status or "").strip().lower().replace("-", "_").replace(" ", "_")
        if status_key in {"filled", "matched", "executed", "complete", "completed"}:
            return "filled"
        if status_key in {"partial", "partially_filled", "partiallyfilled"}:
            return "partially_filled"
        if status_key in {"open", "live", "active", "working", "unmatched"}:
            return "open"
        if status_key in {"pending", "queued", "new", "received", "submitted"}:
            return "pending"
        if status_key in {"cancelled", "canceled", "killed", "void", "terminated"}:
            return "cancelled"
        if status_key in {"expired", "timed_out", "timeout"}:
            return "expired"
        if status_key in {"failed", "rejected", "error"}:
            return "failed"
        return status_key

    def _snapshot_from_cached_order(self, order: Order) -> Optional[dict[str, Any]]:
        clob_id = str(order.clob_order_id or "").strip()
        if not clob_id:
            return None
        status_map = {
            OrderStatus.PENDING: "pending",
            OrderStatus.OPEN: "open",
            OrderStatus.PARTIALLY_FILLED: "partially_filled",
            OrderStatus.FILLED: "filled",
            OrderStatus.CANCELLED: "cancelled",
            OrderStatus.EXPIRED: "expired",
            OrderStatus.FAILED: "failed",
        }
        normalized_status = status_map.get(order.status, "unknown")
        filled_size = max(0.0, safe_float(order.filled_size, 0.0) or 0.0)
        average_fill_price = safe_float(order.average_fill_price)
        filled_notional_usd = None
        if filled_size > 0 and average_fill_price is not None and average_fill_price > 0:
            filled_notional_usd = filled_size * average_fill_price
        return {
            "clob_order_id": clob_id,
            "normalized_status": normalized_status,
            "raw_status": str(getattr(order.status, "value", order.status) or ""),
            "size": max(0.0, safe_float(order.size, 0.0) or 0.0),
            "filled_size": filled_size,
            "remaining_size": max(0.0, (safe_float(order.size, 0.0) or 0.0) - filled_size),
            "average_fill_price": float(average_fill_price) if average_fill_price is not None else None,
            "limit_price": float(safe_float(order.price, 0.0) or 0.0),
            "filled_notional_usd": float(filled_notional_usd) if filled_notional_usd is not None else None,
            "raw": None,
        }

    def _parse_provider_order_snapshot(self, server_order: dict[str, Any]) -> Optional[dict[str, Any]]:
        clob_order_id = str(
            server_order.get("id") or server_order.get("orderID") or server_order.get("order_id") or ""
        ).strip()
        if not clob_order_id:
            return None

        normalized_status = self._normalize_provider_order_status(
            server_order.get("status")
            or server_order.get("state")
            or server_order.get("order_status")
            or server_order.get("orderState")
        )
        size = _first_float(server_order, "size", "original_size", "initial_size", "amount", "quantity")
        filled_size = _first_float(
            server_order,
            "size_matched",
            "sizeMatched",
            "matched_size",
            "filled_size",
            "executed_size",
            "filled",
        )
        remaining_size = _first_float(
            server_order,
            "size_remaining",
            "remaining_size",
            "unfilled_size",
            "sizeRemaining",
        )
        if filled_size is None and size is not None and remaining_size is not None:
            filled_size = max(0.0, size - remaining_size)
        if filled_size is None:
            filled_size = 0.0
        if size is None and remaining_size is not None:
            size = max(0.0, remaining_size + filled_size)

        average_fill_price = _first_float(
            server_order,
            "avg_price",
            "average_price",
            "average_fill_price",
            "avgFillPrice",
            "matched_price",
        )
        limit_price = _first_float(server_order, "price", "limit_price", "limitPrice", "initial_price")
        filled_notional_usd = _first_float(
            server_order,
            "filled_notional_usd",
            "filled_notional",
            "matched_notional",
            "matched_amount",
            "filled_value",
            "executed_notional",
        )
        if filled_notional_usd is None and filled_size > 0 and average_fill_price is not None:
            filled_notional_usd = filled_size * average_fill_price

        return {
            "clob_order_id": clob_order_id,
            "normalized_status": normalized_status,
            "raw_status": str(server_order.get("status") or server_order.get("state") or ""),
            "size": float(size) if size is not None else None,
            "filled_size": max(0.0, float(filled_size)),
            "remaining_size": float(remaining_size) if remaining_size is not None else None,
            "average_fill_price": float(average_fill_price) if average_fill_price is not None else None,
            "limit_price": float(limit_price) if limit_price is not None else None,
            "filled_notional_usd": float(filled_notional_usd) if filled_notional_usd is not None else None,
            "raw": server_order,
        }

    def _apply_snapshot_to_order(self, order: Order, snapshot: dict[str, Any]) -> None:
        normalized_status = str(snapshot.get("normalized_status") or "").strip().lower()
        if normalized_status == "filled":
            order.status = OrderStatus.FILLED
        elif normalized_status == "partially_filled":
            order.status = OrderStatus.PARTIALLY_FILLED
        elif normalized_status == "open":
            order.status = OrderStatus.OPEN
        elif normalized_status == "pending":
            order.status = OrderStatus.PENDING
        elif normalized_status == "cancelled":
            order.status = OrderStatus.CANCELLED
        elif normalized_status == "expired":
            order.status = OrderStatus.EXPIRED
        elif normalized_status == "failed":
            order.status = OrderStatus.FAILED

        filled_size = safe_float(snapshot.get("filled_size"))
        if filled_size is not None:
            order.filled_size = max(0.0, float(filled_size))
        average_fill_price = safe_float(snapshot.get("average_fill_price"))
        if average_fill_price is not None and average_fill_price > 0:
            order.average_fill_price = float(average_fill_price)
        order.updated_at = utcnow()

    async def get_order_snapshots_by_clob_ids(self, clob_order_ids: list[str]) -> dict[str, dict[str, Any]]:
        requested = {str(order_id or "").strip() for order_id in clob_order_ids if str(order_id or "").strip()}
        if not requested:
            return {}

        snapshots: dict[str, dict[str, Any]] = {}
        for order in self._orders.values():
            cached_snapshot = self._snapshot_from_cached_order(order)
            if cached_snapshot is None:
                continue
            clob_id = str(cached_snapshot["clob_order_id"])
            if clob_id in requested:
                snapshots[clob_id] = cached_snapshot

        if not self.is_ready():
            return snapshots

        try:
            response = self._client.get_orders()
            for server_order in self._extract_server_orders(response):
                snapshot = self._parse_provider_order_snapshot(server_order)
                if snapshot is None:
                    continue
                clob_id = str(snapshot["clob_order_id"])
                if clob_id in requested:
                    snapshots[clob_id] = snapshot
        except Exception as exc:
            logger.error("Failed to fetch open provider orders", exc_info=exc)

        missing = requested.difference(snapshots.keys())
        if missing and hasattr(self._client, "get_order"):
            for clob_id in sorted(missing):
                try:
                    single_response = self._client.get_order(clob_id)
                except Exception as exc:
                    logger.debug("Provider single-order lookup failed", clob_order_id=clob_id, exc_info=exc)
                    continue
                for server_order in self._extract_server_orders(single_response):
                    snapshot = self._parse_provider_order_snapshot(server_order)
                    if snapshot is None:
                        continue
                    if str(snapshot["clob_order_id"]) == clob_id:
                        snapshots[clob_id] = snapshot
                        break

        updated_orders: list[Order] = []
        for order in self._orders.values():
            clob_id = str(order.clob_order_id or "").strip()
            if not clob_id:
                continue
            snapshot = snapshots.get(clob_id)
            if snapshot is None:
                continue
            self._apply_snapshot_to_order(order, snapshot)
            updated_orders.append(order)

        if updated_orders:
            await self._persist_orders(updated_orders)

        return snapshots

    async def _sync_provider_open_orders(self) -> list[Order]:
        if not self.is_ready() and not await self.ensure_initialized():
            return []

        try:
            provider_response = await asyncio.to_thread(self._client.get_orders)
        except Exception as exc:
            logger.error("Failed to fetch provider open orders", exc_info=exc)
            return []

        provider_orders = self._extract_server_orders(provider_response)
        updated_orders: list[Order] = []
        for server_order in provider_orders:
            snapshot = self._parse_provider_order_snapshot(server_order)
            if snapshot is None:
                continue

            clob_order_id = str(snapshot["clob_order_id"])
            local_order: Optional[Order] = None
            for cached in self._orders.values():
                if str(cached.clob_order_id or "").strip() == clob_order_id:
                    local_order = cached
                    break

            if local_order is None:
                order_id = f"clob:{clob_order_id}"
                token_id = str(
                    server_order.get("asset_id")
                    or server_order.get("asset")
                    or server_order.get("token_id")
                    or server_order.get("tokenId")
                    or ""
                ).strip()
                if not token_id:
                    token_id = clob_order_id
                side_raw = str(
                    server_order.get("side")
                    or server_order.get("order_side")
                    or server_order.get("direction")
                    or "BUY"
                ).strip().upper()
                side = OrderSide.SELL if side_raw == OrderSide.SELL.value else OrderSide.BUY
                order_type_raw = str(
                    server_order.get("order_type")
                    or server_order.get("orderType")
                    or server_order.get("type")
                    or "GTC"
                ).strip().upper()
                try:
                    order_type = OrderType(order_type_raw)
                except ValueError:
                    order_type = OrderType.GTC

                created_at = _parse_provider_datetime(
                    server_order.get("created_at")
                    or server_order.get("createdAt")
                    or server_order.get("timestamp")
                )
                local_order = Order(
                    id=order_id,
                    token_id=token_id,
                    side=side,
                    price=float(snapshot.get("limit_price") or 0.0),
                    size=float(snapshot.get("size") or snapshot.get("filled_size") or 0.0),
                    order_type=order_type,
                    status=OrderStatus.PENDING,
                    created_at=created_at,
                    updated_at=created_at,
                    clob_order_id=clob_order_id,
                    market_question=str(
                        server_order.get("market_question")
                        or server_order.get("question")
                        or server_order.get("title")
                        or ""
                    )
                    or None,
                )
                self._remember_order(local_order)

            if not local_order.market_question:
                local_order.market_question = str(
                    server_order.get("market_question")
                    or server_order.get("question")
                    or server_order.get("title")
                    or ""
                ) or None
            if local_order.size <= 0:
                local_order.size = float(snapshot.get("size") or snapshot.get("filled_size") or 0.0)
            if local_order.price <= 0:
                local_order.price = float(snapshot.get("limit_price") or 0.0)

            self._apply_snapshot_to_order(local_order, snapshot)
            updated_orders.append(local_order)

        if updated_orders:
            await self._persist_orders(updated_orders)

        return [
            order
            for order in self._orders.values()
            if order.status in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING}
        ]

    async def get_recent_orders(
        self,
        limit: int = 100,
        status: Optional[OrderStatus] = None,
    ) -> list[Order]:
        if not self.is_ready():
            await self.ensure_initialized()

        await self._sync_provider_open_orders()
        orders = sorted(self._orders.values(), key=lambda x: x.created_at, reverse=True)
        if status is not None:
            orders = [order for order in orders if order.status == status]
        return orders[: max(1, int(limit))]

    def _validate_order(
        self,
        size_usd: Decimal,
        side: OrderSide,
        token_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Validate order against safety limits."""
        self._check_daily_reset()

        if global_pause_state.is_paused:
            return False, "Global pause is active"

        if not self.is_ready():
            return False, "Trading service not initialized"

        if not settings.TRADING_ENABLED:
            return False, "Trading is disabled"

        min_order_size = _to_decimal(settings.MIN_ORDER_SIZE_USD)
        max_trade_size = _to_decimal(settings.MAX_TRADE_SIZE_USD)
        max_daily_volume = _to_decimal(settings.MAX_DAILY_TRADE_VOLUME)
        max_per_market = _to_decimal(self.MAX_PER_MARKET_USD)

        if size_usd < min_order_size:
            return (
                False,
                f"Order size ${float(size_usd):.2f} below minimum ${settings.MIN_ORDER_SIZE_USD:.2f}",
            )

        if size_usd > max_trade_size:
            return (
                False,
                f"Order size ${float(size_usd):.2f} exceeds maximum ${settings.MAX_TRADE_SIZE_USD:.2f}",
            )

        projected_daily_volume = self._daily_volume + size_usd
        if projected_daily_volume > max_daily_volume:
            return (
                False,
                f"Would exceed daily volume limit (${float(projected_daily_volume):.2f} > ${settings.MAX_DAILY_TRADE_VOLUME:.2f})",
            )

        if len(self._positions) >= settings.MAX_OPEN_POSITIONS:
            return (
                False,
                f"Maximum open positions ({settings.MAX_OPEN_POSITIONS}) reached",
            )

        # Per-market position limit applies only to increased exposure.
        if token_id and side == OrderSide.BUY:
            current = self._market_positions.get(token_id, ZERO)
            if current + size_usd > max_per_market:
                return (
                    False,
                    f"Per-market limit: ${float(current):.2f} + ${float(size_usd):.2f} exceeds ${self.MAX_PER_MARKET_USD:.2f}",
                )

        return True, ""

    async def _validate_and_reserve_order(
        self,
        *,
        size_usd: Decimal,
        side: OrderSide,
        token_id: Optional[str],
    ) -> tuple[bool, str]:
        # Force refresh from shared DB controls so pause-all propagates quickly
        # across API and worker containers.
        await global_pause_state.refresh_from_db(force=True)

        reserved = False
        stats_lock = self._get_stats_lock()
        async with stats_lock:
            is_valid, error = self._validate_order(
                size_usd=size_usd,
                side=side,
                token_id=token_id,
            )
            if not is_valid:
                return False, error

            self._daily_volume += size_usd
            self._total_volume += size_usd
            delta = size_usd if side == OrderSide.BUY else -size_usd
            self._apply_market_exposure_delta(token_id, delta)
            self._sync_stats_from_decimals()
            reserved = True

        if reserved:
            await self._persist_runtime_state()
            return True, ""
        return False, "Order reservation failed"

    async def _release_reservation(
        self,
        *,
        size_usd: Decimal,
        side: OrderSide,
        token_id: Optional[str],
    ) -> None:
        stats_lock = self._get_stats_lock()
        async with stats_lock:
            self._daily_volume = max(ZERO, self._daily_volume - size_usd)
            self._total_volume = max(ZERO, self._total_volume - size_usd)
            delta = -size_usd if side == OrderSide.BUY else size_usd
            self._apply_market_exposure_delta(token_id, delta)
            self._sync_stats_from_decimals()
        await self._persist_runtime_state()

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

        # Calculate USD notional with Decimal to avoid float accumulation drift.
        size_usd = _to_decimal(price) * _to_decimal(size)
        reserved = False

        # VPN pre-trade check (blocks if VPN required but unreachable)
        vpn_ok, vpn_reason = await pre_trade_vpn_check()
        if not vpn_ok:
            order.status = OrderStatus.FAILED
            order.error_message = f"VPN check failed: {vpn_reason}"
            self._remember_order(order)
            await self._persist_orders([order])
            logger.error(f"Trade blocked by VPN check: {vpn_reason}")
            return order

        # Validate and reserve risk budget atomically to prevent async races.
        is_valid, error = await self._validate_and_reserve_order(
            size_usd=size_usd,
            side=side,
            token_id=token_id,
        )
        if not is_valid:
            order.status = OrderStatus.FAILED
            order.error_message = error
            self._remember_order(order)
            await self._persist_orders([order])
            logger.warning(f"Order validation failed: {error}")
            return order
        reserved = True

        try:
            await self._sync_trading_transport()

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
                stats_lock = self._get_stats_lock()
                async with stats_lock:
                    self._stats.total_trades += 1
                    self._stats.last_trade_at = utcnow()
                await self._persist_runtime_state()
                logger.info(f"Order placed successfully: {order.clob_order_id}")
            else:
                order.status = OrderStatus.FAILED
                order.error_message = response.get("errorMsg", "Unknown error")
                await self._release_reservation(
                    size_usd=size_usd,
                    side=side,
                    token_id=token_id,
                )
                reserved = False
                logger.error(f"Order failed: {order.error_message}")

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            if reserved:
                await self._release_reservation(
                    size_usd=size_usd,
                    side=side,
                    token_id=token_id,
                )
                reserved = False
            logger.error(f"Order execution error: {e}")

        order.updated_at = utcnow()
        self._remember_order(order)
        await self._persist_orders([order])
        await self._persist_runtime_state()
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
        order_key = str(order_id or "").strip()
        if not order_key:
            return False

        local_order = self._orders.get(order_key)
        if local_order is not None:
            if local_order.status not in {OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                logger.warning(f"Cannot cancel order in status: {local_order.status}")
                return False
            clob_order_id = str(local_order.clob_order_id or "").strip()
            if not clob_order_id:
                local_order.status = OrderStatus.CANCELLED
                local_order.updated_at = utcnow()
                await self._persist_orders([local_order])
                await self._persist_runtime_state()
                return True
        else:
            clob_order_id = order_key

        if not self.is_ready() and not await self.ensure_initialized():
            logger.warning("Trading service not ready for order cancellation", order_id=order_key)
            return False

        try:
            response = await asyncio.to_thread(self._client.cancel, clob_order_id)
        except Exception as exc:
            logger.error("Cancel order error", order_id=order_key, clob_order_id=clob_order_id, exc_info=exc)
            return False

        cancelled = False
        if isinstance(response, dict):
            canceled_field = response.get("canceled")
            if isinstance(canceled_field, bool):
                cancelled = canceled_field
            elif isinstance(canceled_field, list):
                for item in canceled_field:
                    if isinstance(item, dict):
                        if str(item.get("id") or item.get("orderID") or "").strip() == clob_order_id:
                            cancelled = True
                            break
                    elif str(item or "").strip() == clob_order_id:
                        cancelled = True
                        break
                if not cancelled and len(canceled_field) > 0:
                    cancelled = True
            elif isinstance(canceled_field, str):
                cancelled = canceled_field.strip() == clob_order_id
            if not cancelled and bool(response.get("success")):
                cancelled = True
            if not cancelled:
                error_text = str(response.get("error") or response.get("errorMsg") or "").strip().lower()
                if "already" in error_text and "cancel" in error_text:
                    cancelled = True
                elif "not found" in error_text:
                    cancelled = True
        elif isinstance(response, list):
            cancelled = any(str(item or "").strip() == clob_order_id for item in response) or bool(response)

        if not cancelled:
            logger.error("Failed to cancel order", order_id=order_key, clob_order_id=clob_order_id, response=response)
            return False

        now = utcnow()
        changed_orders: list[Order] = []
        if local_order is not None:
            local_order.status = OrderStatus.CANCELLED
            local_order.updated_at = now
            changed_orders.append(local_order)
        for order in self._orders.values():
            if str(order.clob_order_id or "").strip() == clob_order_id:
                order.status = OrderStatus.CANCELLED
                order.updated_at = now
                changed_orders.append(order)
        if changed_orders:
            await self._persist_orders(changed_orders)
            await self._persist_runtime_state()
        logger.info(f"Order cancelled: {order_key}")
        return True

    async def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all open orders with per-order success/failure reporting."""
        open_orders = await self.get_open_orders()
        targets: list[str] = []
        seen_targets: set[str] = set()
        for order in open_orders:
            target = str(order.clob_order_id or order.id or "").strip()
            if not target or target in seen_targets:
                continue
            seen_targets.add(target)
            targets.append(target)

        if not targets:
            return {
                "status": "success",
                "requested_count": 0,
                "cancelled_count": 0,
                "failed_count": 0,
                "failed_order_ids": [],
                "message": "No open orders to cancel.",
            }

        failed_order_ids: list[str] = []
        cancelled_count = 0
        for target in targets:
            if await self.cancel_order(target):
                cancelled_count += 1
            else:
                failed_order_ids.append(target)

        failed_count = len(failed_order_ids)
        if failed_count == 0:
            status = "success"
            message = f"Cancelled {cancelled_count} order(s)."
        elif cancelled_count > 0:
            status = "partial_failure"
            message = (
                f"Cancelled {cancelled_count} of {len(targets)} order(s); "
                f"{failed_count} cancellation(s) failed."
            )
        else:
            status = "failed"
            message = f"Failed to cancel {failed_count} order(s)."

        logger.info(
            "Cancel-all completed",
            status=status,
            requested_count=len(targets),
            cancelled_count=cancelled_count,
            failed_count=failed_count,
        )
        return {
            "status": status,
            "requested_count": len(targets),
            "cancelled_count": cancelled_count,
            "failed_count": failed_count,
            "failed_order_ids": failed_order_ids,
            "message": message,
        }

    async def get_open_orders(self) -> list[Order]:
        """Get all open orders"""
        await self._sync_provider_open_orders()
        clob_ids = [
            str(order.clob_order_id).strip()
            for order in self._orders.values()
            if str(order.clob_order_id or "").strip()
        ]
        if clob_ids:
            try:
                await self.get_order_snapshots_by_clob_ids(clob_ids)
            except Exception as exc:
                logger.error("Get orders error", exc_info=exc)

        return [
            o
            for o in self._orders.values()
            if o.status in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING}
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
                position.unrealized_pnl = (position.current_price - position.average_cost) * position.size
                self._positions[token_id] = position

            self._stats.open_positions = len(self._positions)
            await self._persist_positions()
            await self._persist_runtime_state()

        except Exception as e:
            logger.error(f"Sync positions error: {e}")

        return list(self._positions.values())

    def _get_wallet_address(self) -> Optional[str]:
        """Get wallet address from private key"""
        if self._wallet_address:
            return self._wallet_address
        if not settings.POLYMARKET_PRIVATE_KEY:
            return None
        try:
            from eth_account import Account

            account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
            return account.address
        except Exception:
            return None

    async def execute_opportunity(self, opportunity_id: str, positions: list[dict], size_usd: float) -> list[Order]:
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

            # Crypto 15-min markets: use maker mode to avoid taker fees
            # and earn rebates.  Place at best_bid (or 1 tick below ask)
            # to sit on the book as a maker order.
            if position.get("_maker_mode"):
                maker_price = position.get("_maker_price", price)
                # Round down to tick size (0.01 for crypto markets)
                maker_price = max(0.01, round(maker_price - 0.005, 2))
                price = maker_price
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
            asyncio.create_task(self._auto_reconcile(orders, valid_positions, failed_count))

        return orders

    async def _auto_reconcile(self, orders: list, positions: list, failed_count: int):
        """Auto-unwind partial multi-leg fills to prevent one-sided exposure."""
        await asyncio.sleep(2)  # Brief delay before reconciliation
        logger.info(f"AUTO_RECONCILE: Unwinding {len(orders) - failed_count} filled legs")
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
                            side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                            price=order.price * 0.95 if order.side == OrderSide.BUY else order.price * 1.05,
                            size=order.filled_size,
                            order_type=OrderType.FOK,
                            market_question=f"AUTO_RECONCILE: {order.market_question}",
                        )
                        logger.info(f"Reconciliation order placed: {unwind.status.value}")
                    except Exception as e:
                        logger.error(f"Reconciliation failed: {e}")

    def get_stats(self) -> TradingStats:
        """Get trading statistics"""
        self._check_daily_reset()
        self._sync_stats_from_decimals()
        return self._stats

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        key = str(order_id or "").strip()
        if not key:
            return None
        direct = self._orders.get(key)
        if direct is not None:
            return direct
        for order in self._orders.values():
            if str(order.clob_order_id or "").strip() == key:
                return order
        return None

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
            address = self._get_wallet_address()
            if not address:
                return {"error": "Could not derive wallet address"}

            try:
                from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

                def build_balance_params(signature_type: int):
                    return BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                        signature_type=signature_type,
                    )
            except Exception:

                class _FallbackBalanceParams:
                    def __init__(self, signature_type: int):
                        self.asset_type = "COLLATERAL"
                        self.signature_type = signature_type

                def build_balance_params(signature_type: int):
                    return _FallbackBalanceParams(signature_type)

            async def fetch_balance_snapshot(signature_type: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
                params = build_balance_params(signature_type)
                try:
                    await asyncio.to_thread(self._client.update_balance_allowance, params)
                except Exception as exc:
                    logger.warning(
                        "Balance allowance refresh failed; using cached values",
                        signature_type=signature_type,
                        exc_info=exc,
                    )
                try:
                    payload = await asyncio.to_thread(self._client.get_balance_allowance, params)
                except Exception as exc:
                    logger.warning(
                        "Balance allowance fetch failed",
                        signature_type=signature_type,
                        exc_info=exc,
                    )
                    return None, None
                if not isinstance(payload, dict):
                    return None, "Unexpected balance response"
                assume_base_units = isinstance(payload.get("allowances"), dict)
                balance = _parse_collateral_amount(
                    payload.get("balance"),
                    assume_base_units=assume_base_units,
                )
                if balance is None:
                    return None, "Balance value missing from response"

                allowance = _parse_collateral_amount(
                    payload.get("allowance"),
                    assume_base_units=assume_base_units,
                )
                allowances = payload.get("allowances")
                if isinstance(allowances, dict) and allowances:
                    max_allowance: Optional[float] = None
                    for raw_allowance in allowances.values():
                        parsed_allowance = _parse_collateral_amount(raw_allowance, assume_base_units=True)
                        if parsed_allowance is None:
                            continue
                        if max_allowance is None or parsed_allowance > max_allowance:
                            max_allowance = parsed_allowance
                    if max_allowance is not None:
                        allowance = max_allowance

                if allowance is None:
                    allowance = balance
                available = max(0.0, min(balance, allowance))
                reserved = max(0.0, balance - available)
                return {
                    "signature_type": signature_type,
                    "balance": balance,
                    "available": available,
                    "reserved": reserved,
                }, None

            builder = getattr(self._client, "builder", None)
            builder_signature_type = getattr(builder, "sig_type", None)
            if not isinstance(builder_signature_type, int):
                builder_signature_type = 0
            primary_signature_type = (
                self._balance_signature_type
                if isinstance(self._balance_signature_type, int)
                else builder_signature_type
            )

            primary_snapshot, primary_error = await fetch_balance_snapshot(primary_signature_type)
            if primary_error:
                return {"error": primary_error}

            best_snapshot = primary_snapshot
            needs_probe = (
                primary_snapshot is None
                or (
                    primary_snapshot["balance"] <= 0.0
                    and primary_snapshot["available"] <= 0.0
                )
            )

            if needs_probe:
                for signature_type in POLYMARKET_SIGNATURE_TYPES:
                    if signature_type == primary_signature_type:
                        continue
                    candidate_snapshot, candidate_error = await fetch_balance_snapshot(signature_type)
                    if candidate_error:
                        return {"error": candidate_error}
                    if candidate_snapshot is None:
                        continue
                    if best_snapshot is None:
                        best_snapshot = candidate_snapshot
                        continue
                    if candidate_snapshot["balance"] > best_snapshot["balance"]:
                        best_snapshot = candidate_snapshot
                        continue
                    if (
                        candidate_snapshot["balance"] == best_snapshot["balance"]
                        and candidate_snapshot["available"] > best_snapshot["available"]
                    ):
                        best_snapshot = candidate_snapshot

            if best_snapshot is None:
                return {"error": "Could not fetch balance from CLOB API"}

            selected_signature_type = int(best_snapshot["signature_type"])
            self._balance_signature_type = selected_signature_type
            if builder is not None and getattr(builder, "sig_type", None) != selected_signature_type:
                builder.sig_type = selected_signature_type

            return {
                "address": address,
                "balance": best_snapshot["balance"],
                "available": best_snapshot["available"],
                "reserved": best_snapshot["reserved"],
                "currency": "USDC",
                "timestamp": utcnow().isoformat(),
                "positions_value": sum(p.size * p.current_price for p in self._positions.values()),
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
trading_service = TradingService()
