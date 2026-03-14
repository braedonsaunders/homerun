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
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR
import uuid

from sqlalchemy import delete, select
from sqlalchemy.exc import InterfaceError, OperationalError

from config import settings
from services.pause_state import global_pause_state
from services.price_chaser import price_chaser
from services.execution_tiers import execution_tier_service
from services.strategy_sdk import StrategySDK
from services.trading_proxy import (
    patch_clob_client_proxy,
    pre_trade_vpn_check,
    _load_config_from_db as load_proxy_config,
)
from utils.logger import get_logger
from utils.secrets import decrypt_secret
from utils.converters import coerce_bool, safe_float

logger = get_logger(__name__)

ZERO = Decimal("0")
USDC_BASE_UNITS = Decimal("1000000")
POLYMARKET_SIGNATURE_TYPES = (0, 1, 2)
POST_ONLY_REPRICE_TICK = 0.01
INITIALIZATION_RETRY_BACKOFF_SECONDS = 60.0
MISSING_DEPENDENCY_RELOG_SECONDS = 300.0
PENDING_RECONCILIATION_MAX_ATTEMPTS = 6


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


def _parse_balance_allowance_amount(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return Decimal(raw)
    except Exception:
        parsed_float = safe_float(value)
        if parsed_float is None:
            return None
        return Decimal(str(parsed_float))


from utils.retry import DB_RETRY_ATTEMPTS as _DB_RETRY_ATTEMPTS  # noqa: E402
from utils.retry import is_retryable_db_error as _is_retryable_db_error  # noqa: E402
from utils.retry import db_retry_delay as _db_retry_delay  # noqa: E402


def _normalize_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_post_only_cross_reject(error_message: str | None) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    return "post-only" in text and "crosses book" in text


# Markers in exception text that indicate a transient network/proxy failure
# rather than a genuine order rejection.  py_clob_client wraps httpx transport
# errors as PolyApiException(error_msg="Request exception!"), so we match on
# the wrapper text and common network error strings.
_TRANSIENT_TRANSPORT_MARKERS = (
    "request exception",
    "proxy error",
    "proxyerror",
    "invalid username/password",
    "connection reset",
    "connection refused",
    "connection closed",
    "connect timeout",
    "read timeout",
    "timed out",
    "network is unreachable",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "broken pipe",
    "connection aborted",
    "remotedisconnected",
    "connectionerror",
)


def _is_transient_transport_error(exc: Exception) -> bool:
    """Check if an exception is a transient network/proxy error worth retrying."""
    try:
        import httpx as _httpx

        if isinstance(exc, (_httpx.TransportError, _httpx.TimeoutException, ConnectionError, OSError)):
            return True
    except ImportError:
        pass
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_TRANSPORT_MARKERS)


def _clamp_binary_price(value: float) -> float:
    return max(POST_ONLY_REPRICE_TICK, min(0.99, float(value)))


def _next_post_only_retry_price(side: "OrderSide", price: float) -> float:
    if side == OrderSide.BUY:
        return _clamp_binary_price(float(price) - POST_ONLY_REPRICE_TICK)
    return _clamp_binary_price(float(price) + POST_ONLY_REPRICE_TICK)


def _validated_positive_float(value: Any, *, field_name: str) -> float:
    parsed = safe_float(value, None, reject_nan_inf=True)
    if parsed is None or parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(parsed)


def _tick_size_from_position(position: dict[str, Any]) -> float:
    for key in (
        "_tick_size",
        "tick_size",
        "tickSize",
        "min_tick_size",
        "minimum_tick_size",
        "price_increment",
        "priceIncrement",
        "_price_increment",
    ):
        parsed = safe_float(position.get(key), None, reject_nan_inf=True)
        if parsed is not None and parsed > 0.0:
            return float(parsed)
    return POST_ONLY_REPRICE_TICK


def _round_down_to_tick(price: float, tick_size: float) -> float:
    tick = max(POST_ONLY_REPRICE_TICK, float(tick_size))
    normalized_price = max(tick, float(price))
    tick_decimal = Decimal(str(tick))
    price_decimal = Decimal(str(normalized_price))
    rounded = (price_decimal / tick_decimal).to_integral_value(rounding=ROUND_FLOOR) * tick_decimal
    return float(rounded)


def _pending_reconciliation_retry_delay(attempt: int) -> float:
    return min(60.0, float(2 ** max(0, int(attempt) - 1)))


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
    IOC = "IOC"  # Immediate Or Cancel (partial fill ok, cancel unfilled remainder)


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


def _normalize_order_type(value: Any) -> OrderType:
    raw = str(getattr(value, "value", value) or "").strip().upper()
    if raw == "FAK":
        return OrderType.IOC
    try:
        return OrderType(raw)
    except ValueError:
        return OrderType.GTC


def _provider_order_type_value(value: Any) -> str:
    order_type = _normalize_order_type(value)
    if order_type == OrderType.IOC:
        return OrderType.FAK.value
    return order_type.value


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


class LiveExecutionService:
    """
    Service for executing real trades on Polymarket.

    Uses the py-clob-client library for order placement.
    Implements safety limits and tracking.
    """

    def __init__(self):
        self._initialized = False
        self._client = None
        self._wallet_address: Optional[str] = None
        self._eoa_address: Optional[str] = None
        self._proxy_funder_address: Optional[str] = None
        self._last_init_error: Optional[str] = None
        self._init_retry_not_before: Optional[datetime] = None
        self._last_missing_dependency_log_at: Optional[datetime] = None
        self._orders: OrderedDict[str, Order] = OrderedDict()
        self._positions: dict[str, Position] = {}
        self._stats = TradingStats()
        self._daily_volume_reset = utcnow().date()
        self._market_positions: OrderedDict[str, Decimal] = OrderedDict()  # token_id -> USD exposure
        self._stats_lock: Optional[asyncio.Lock] = None
        self._init_lock: Optional[asyncio.Lock] = None
        self._client_io_lock: Optional[asyncio.Lock] = None
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
        self._background_tasks: set[asyncio.Task] = set()
        self._reconciliation_tasks: dict[str, asyncio.Task] = {}
        self._pending_reconciliations: list[dict[str, Any]] = []

    def _get_stats_lock(self) -> asyncio.Lock:
        if self._stats_lock is None:
            self._stats_lock = asyncio.Lock()
        return self._stats_lock

    def _get_init_lock(self) -> asyncio.Lock:
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    def _get_client_io_lock(self) -> asyncio.Lock:
        if self._client_io_lock is None:
            self._client_io_lock = asyncio.Lock()
        return self._client_io_lock

    def _get_persist_lock(self) -> asyncio.Lock:
        if self._persist_lock is None:
            self._persist_lock = asyncio.Lock()
        return self._persist_lock

    async def _run_client_io(self, func: Any, *args: Any) -> Any:
        async with self._get_client_io_lock():
            return await asyncio.to_thread(func, *args)

    async def check_buy_pre_submit_gate(
        self,
        *,
        token_id: str,
        required_notional_usd: float,
    ) -> tuple[bool, Optional[str]]:
        required_notional = _to_decimal(max(0.0, float(required_notional_usd)))
        return await self._enforce_buy_pre_submit_gate(
            token_id=token_id,
            required_notional_usd=required_notional,
        )

    def _track_background_task(
        self,
        task: asyncio.Task,
        *,
        description: str,
        registry: dict[str, asyncio.Task] | None = None,
        registry_key: str | None = None,
    ) -> asyncio.Task:
        self._background_tasks.add(task)
        if registry is not None and registry_key:
            registry[registry_key] = task

        def _finalize(done_task: asyncio.Task) -> None:
            self._background_tasks.discard(done_task)
            if registry is not None and registry_key:
                existing = registry.get(registry_key)
                if existing is done_task:
                    registry.pop(registry_key, None)
            try:
                done_task.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Background task failed", task_name=description, exc_info=exc)

        task.add_done_callback(_finalize)
        return task

    def _start_background_task(
        self,
        coro: Any,
        *,
        name: str,
        registry: dict[str, asyncio.Task] | None = None,
        registry_key: str | None = None,
    ) -> asyncio.Task | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Skipping background task spawn because no event loop is running", task_name=name)
            return None
        task = loop.create_task(coro, name=name)
        return self._track_background_task(task, description=name, registry=registry, registry_key=registry_key)

    def _pending_reconciliation_index(self, reconciliation_id: str) -> int | None:
        key = str(reconciliation_id or "").strip()
        if not key:
            return None
        for index, item in enumerate(self._pending_reconciliations):
            if str(item.get("id") or "").strip() == key:
                return index
        return None

    def _serialize_reconciliation_order(self, order: Order) -> dict[str, Any]:
        return {
            "order_id": str(order.id),
            "token_id": str(order.token_id or "").strip(),
            "side": order.side.value,
            "price": float(order.price),
            "filled_size": float(order.filled_size),
            "market_question": order.market_question,
            "opportunity_id": order.opportunity_id,
        }

    def _normalize_pending_reconciliation(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        reconciliation_id = str(raw.get("id") or "").strip()
        if not reconciliation_id:
            return None
        orders: list[dict[str, Any]] = []
        for raw_order in raw.get("orders") or []:
            if not isinstance(raw_order, dict):
                continue
            token_id = str(raw_order.get("token_id") or "").strip()
            side_raw = str(raw_order.get("side") or "").strip().upper()
            price = safe_float(raw_order.get("price"), None, reject_nan_inf=True)
            filled_size = safe_float(raw_order.get("filled_size"), None, reject_nan_inf=True)
            if not token_id or side_raw not in {OrderSide.BUY.value, OrderSide.SELL.value}:
                continue
            if price is None or price <= 0.0 or filled_size is None or filled_size <= 0.0:
                continue
            orders.append(
                {
                    "order_id": str(raw_order.get("order_id") or "").strip() or None,
                    "token_id": token_id,
                    "side": side_raw,
                    "price": float(price),
                    "filled_size": float(filled_size),
                    "market_question": str(raw_order.get("market_question") or "").strip() or None,
                    "opportunity_id": str(raw_order.get("opportunity_id") or "").strip() or None,
                }
            )
        if not orders:
            return None
        attempts = int(raw.get("attempts") or 0)
        return {
            "id": reconciliation_id,
            "created_at": str(raw.get("created_at") or utcnow().isoformat()),
            "last_attempt_at": str(raw.get("last_attempt_at") or "") or None,
            "last_error": str(raw.get("last_error") or "") or None,
            "attempts": max(0, attempts),
            "orders": orders,
        }

    async def _persist_runtime_state_now(self) -> None:
        if not self._wallet_for_persistence():
            return
        await self._persist_runtime_state()

    async def _run_pending_reconciliation(self, reconciliation_id: str) -> None:
        index = self._pending_reconciliation_index(reconciliation_id)
        if index is None:
            return
        payload = self._pending_reconciliations[index]
        attempts = int(payload.get("attempts") or 0) + 1
        payload["attempts"] = attempts
        payload["last_attempt_at"] = utcnow().isoformat()
        payload["last_error"] = None
        await self._persist_runtime_state_now()

        try:
            await self._auto_reconcile(payload.get("orders") or [])
        except Exception as exc:
            payload["last_error"] = str(exc)
            await self._persist_runtime_state_now()
            logger.error(
                "Pending partial-fill reconciliation failed",
                reconciliation_id=reconciliation_id,
                attempts=attempts,
                exc_info=exc,
            )
            if attempts >= PENDING_RECONCILIATION_MAX_ATTEMPTS:
                logger.error(
                    "Pending partial-fill reconciliation reached max attempts",
                    reconciliation_id=reconciliation_id,
                    attempts=attempts,
                )
                return
            retry_delay = _pending_reconciliation_retry_delay(attempts)
            logger.warning(
                "Scheduling pending partial-fill reconciliation retry",
                reconciliation_id=reconciliation_id,
                attempts=attempts,
                retry_delay_seconds=retry_delay,
            )
            self._start_background_task(
                self._retry_pending_reconciliation_after_delay(reconciliation_id, retry_delay),
                name=f"live-execution-reconcile-retry-{reconciliation_id}",
                registry=self._reconciliation_tasks,
                registry_key=reconciliation_id,
            )
            return

        index = self._pending_reconciliation_index(reconciliation_id)
        if index is not None:
            self._pending_reconciliations.pop(index)
            await self._persist_runtime_state_now()

    async def _enqueue_pending_reconciliation(self, orders: list[Order]) -> None:
        serialized_orders = [
            self._serialize_reconciliation_order(order)
            for order in orders
            if order.status in {OrderStatus.OPEN, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
            and float(order.filled_size or 0.0) > 0.0
        ]
        if not serialized_orders:
            return

        payload = {
            "id": uuid.uuid4().hex,
            "created_at": utcnow().isoformat(),
            "last_attempt_at": None,
            "last_error": None,
            "attempts": 0,
            "orders": serialized_orders,
        }
        self._pending_reconciliations.append(payload)
        await self._persist_runtime_state_now()
        self._start_background_task(
            self._run_pending_reconciliation(payload["id"]),
            name=f"live-execution-reconcile-{payload['id']}",
            registry=self._reconciliation_tasks,
            registry_key=payload["id"],
        )

    async def _retry_pending_reconciliation_after_delay(
        self,
        reconciliation_id: str,
        delay_seconds: float,
    ) -> None:
        await asyncio.sleep(max(0.0, float(delay_seconds)))
        await self._run_pending_reconciliation(reconciliation_id)

    def _schedule_restored_reconciliations(self) -> None:
        for payload in list(self._pending_reconciliations):
            reconciliation_id = str(payload.get("id") or "").strip()
            if not reconciliation_id or reconciliation_id in self._reconciliation_tasks:
                continue
            attempts = int(payload.get("attempts") or 0)
            if attempts >= PENDING_RECONCILIATION_MAX_ATTEMPTS:
                logger.error(
                    "Restored partial-fill reconciliation requires manual intervention",
                    reconciliation_id=reconciliation_id,
                    attempts=attempts,
                )
                continue
            self._start_background_task(
                self._run_pending_reconciliation(reconciliation_id),
                name=f"live-execution-reconcile-{reconciliation_id}",
                registry=self._reconciliation_tasks,
                registry_key=reconciliation_id,
            )

    def _normalize_evm_address(self, address: Any) -> Optional[str]:
        text = str(address or "").strip()
        if not text:
            return None
        try:
            from web3 import Web3

            return Web3.to_checksum_address(text)
        except Exception:
            return None

    def _funder_for_signature_type(self, signature_type: int) -> Optional[str]:
        if signature_type == 0:
            return self._eoa_address or self._wallet_address
        if signature_type in (1, 2):
            return self._proxy_funder_address
        return None

    def _signature_type_supported(self, signature_type: int) -> bool:
        return self._funder_for_signature_type(signature_type) is not None

    def _apply_signature_type_to_client(self, signature_type: Optional[int]) -> None:
        if not self.is_ready():
            return
        if not isinstance(signature_type, int):
            return
        if not (0 <= signature_type <= 2):
            return
        if self._client is None:
            return

        if getattr(self._client, "signature_type", None) != signature_type:
            try:
                self._client.signature_type = signature_type
            except (AttributeError, TypeError) as exc:
                logger.debug("Failed to apply signature_type to trading client", exc_info=exc)

        builder = getattr(self._client, "builder", None)
        if builder is not None and getattr(builder, "sig_type", None) != signature_type:
            try:
                builder.sig_type = signature_type
            except (AttributeError, TypeError) as exc:
                logger.debug("Failed to apply signature_type to trading client builder", exc_info=exc)
        funder = self._funder_for_signature_type(signature_type)
        if builder is not None and isinstance(funder, str) and getattr(builder, "funder", None) != funder:
            try:
                builder.funder = funder
            except (AttributeError, TypeError) as exc:
                logger.debug("Failed to apply funder to trading client builder", exc_info=exc)

    def _is_invalid_signature_error(self, error_text: Any) -> bool:
        if error_text is None:
            return False
        text = str(error_text).lower()
        return "invalid signature" in text

    async def _fetch_conditional_balance_snapshot(
        self,
        token_id: str,
        signature_type: int,
        *,
        refresh: bool,
    ) -> Optional[dict[str, Any]]:
        token_key = str(token_id or "").strip()
        if not token_key:
            return None
        if not self._signature_type_supported(signature_type):
            return None
        if not self.is_ready():
            return None

        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_key,
                signature_type=int(signature_type),
            )
        except Exception:
            return None

        if refresh:
            try:
                await self._run_client_io(self._client.update_balance_allowance, params)
            except Exception as exc:
                logger.debug(
                    "Conditional balance-allowance refresh failed",
                    token_id=token_key,
                    signature_type=signature_type,
                    exc_info=exc,
                )

        try:
            payload = await self._run_client_io(self._client.get_balance_allowance, params)
        except Exception as exc:
            logger.debug(
                "Conditional balance-allowance fetch failed",
                token_id=token_key,
                signature_type=signature_type,
                exc_info=exc,
            )
            return None
        if not isinstance(payload, dict):
            return None

        balance_raw = _parse_balance_allowance_amount(payload.get("balance")) or ZERO
        allowance_raw = _parse_balance_allowance_amount(payload.get("allowance"))
        allowances = payload.get("allowances")
        if isinstance(allowances, dict):
            for raw_allowance in allowances.values():
                parsed_allowance = _parse_balance_allowance_amount(raw_allowance)
                if parsed_allowance is None:
                    continue
                if allowance_raw is None or parsed_allowance > allowance_raw:
                    allowance_raw = parsed_allowance
        if allowance_raw is None:
            allowance_raw = balance_raw
        available_raw = min(balance_raw, allowance_raw)

        return {
            "signature_type": int(signature_type),
            "balance_raw": balance_raw,
            "allowance_raw": allowance_raw,
            "available_raw": available_raw,
        }

    async def _select_signature_type_for_conditional_token(self, token_id: str) -> Optional[int]:
        token_key = str(token_id or "").strip()
        if not token_key:
            return None
        if not self.is_ready() and not await self.ensure_initialized():
            return None
        if not self.is_ready():
            return None

        current_signature_type = self._resolved_signature_type()
        candidates: list[int] = []
        if self._signature_type_supported(current_signature_type):
            candidates.append(int(current_signature_type))
        for signature_type in POLYMARKET_SIGNATURE_TYPES:
            if signature_type in candidates:
                continue
            if not self._signature_type_supported(signature_type):
                continue
            candidates.append(signature_type)

        best_snapshot: Optional[dict[str, Any]] = None
        for signature_type in candidates:
            snapshot = await self._fetch_conditional_balance_snapshot(
                token_key,
                signature_type,
                refresh=True,
            )
            if snapshot is None:
                continue
            if best_snapshot is None:
                best_snapshot = snapshot
                continue
            if snapshot["available_raw"] > best_snapshot["available_raw"]:
                best_snapshot = snapshot
                continue
            if (
                snapshot["available_raw"] == best_snapshot["available_raw"]
                and snapshot["balance_raw"] > best_snapshot["balance_raw"]
            ):
                best_snapshot = snapshot

        if best_snapshot is None:
            return None

        selected_signature_type = int(best_snapshot["signature_type"])
        self._balance_signature_type = selected_signature_type
        self._apply_signature_type_to_client(selected_signature_type)
        return selected_signature_type

    async def _refresh_signature_type(self, *, force: bool = False) -> bool:
        if not self.is_ready():
            return False

        if not force and isinstance(self._balance_signature_type, int):
            if not self._signature_type_supported(int(self._balance_signature_type)):
                return False
            self._apply_signature_type_to_client(self._balance_signature_type)
            return True

        if force:
            self._balance_signature_type = None

        balance = await self.get_balance()
        if isinstance(balance, dict) and balance.get("error"):
            logger.warning("Signature refresh failed from balance probe: %s", balance["error"])
            return False

        signature_type = self._balance_signature_type
        if not isinstance(signature_type, int):
            builder = getattr(self._client, "builder", None)
            if builder is not None and isinstance(getattr(builder, "sig_type", None), int):
                signature_type = int(builder.sig_type)

        if not isinstance(signature_type, int):
            return False
        if not self._signature_type_supported(signature_type):
            return False

        self._balance_signature_type = signature_type
        self._apply_signature_type_to_client(signature_type)
        return True

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

    def _derive_poly_proxy_funder(self, eoa_address: str) -> Optional[str]:
        """Call CTFExchange.getPolyProxyWalletAddress(eoa) on-chain to get the
        proxy wallet (funder) address for proxy signature wallets.

        Returns the checksummed proxy address, or None if the call fails.
        """
        try:
            from web3 import Web3
            from py_clob_client.config import get_contract_config

            _rpc_candidates = [
                url
                for url in [
                    settings.POLYGON_RPC_URL,
                    "https://rpc-mainnet.matic.quiknode.pro",
                    "https://polygon.gateway.tenderly.co",
                ]
                if url
            ]
            w3 = None
            for rpc_url in _rpc_candidates:
                try:
                    candidate = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
                    candidate.eth.block_number
                    w3 = candidate
                    break
                except Exception:
                    continue
            if w3 is None:
                return None

            contract_cfg = get_contract_config(settings.CHAIN_ID)
            if contract_cfg is None:
                return None

            exchange_addr = Web3.to_checksum_address(contract_cfg.exchange)
            # getPolyProxyWalletAddress(address) → address
            _ABI = [
                {
                    "name": "getPolyProxyWalletAddress",
                    "type": "function",
                    "inputs": [{"name": "_addr", "type": "address"}],
                    "outputs": [{"name": "", "type": "address"}],
                    "stateMutability": "view",
                }
            ]
            exchange = w3.eth.contract(address=exchange_addr, abi=_ABI)
            proxy = exchange.functions.getPolyProxyWalletAddress(Web3.to_checksum_address(eoa_address)).call()
            return Web3.to_checksum_address(proxy)
        except Exception as exc:
            logger.warning("Failed to derive proxy funder address: %s", exc)
            return None

    def _lookup_data_api_proxy_funder(self, eoa_address: str) -> Optional[str]:
        try:
            import httpx

            data_api_base = str(getattr(settings, "DATA_API_URL", "") or "").rstrip("/")
            if not data_api_base:
                return None
            if not hasattr(self, "_data_api_client") or self._data_api_client is None or self._data_api_client.is_closed:
                self._data_api_client = httpx.Client(timeout=8.0, follow_redirects=True)
            response = self._data_api_client.get(
                f"{data_api_base}/profile",
                params={"address": eoa_address},
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            for key in ("proxyWallet", "proxyAddress", "wallet"):
                candidate = self._normalize_evm_address(payload.get(key))
                if candidate and candidate.lower() != eoa_address.lower():
                    return candidate
        except Exception as exc:
            logger.debug("Data API proxy funder lookup failed: %s", exc)
        return None

    def _resolve_polymarket_funder(self, eoa_address: str, signature_type: int) -> Optional[str]:
        if signature_type == 0:
            return eoa_address

        configured = self._normalize_evm_address(getattr(settings, "POLYMARKET_FUNDER", None))
        if configured:
            return configured

        profile_proxy = self._lookup_data_api_proxy_funder(eoa_address)
        if profile_proxy:
            return profile_proxy

        return self._derive_poly_proxy_funder(eoa_address)

    async def _sync_trading_transport(self) -> bool:
        await load_proxy_config()
        return patch_clob_client_proxy()

    async def _approve_clob_allowance(self) -> None:
        """Refresh CLOB collateral balance/allowance cache for supported signature types."""
        if not self.is_ready():
            return

        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            def build_params(sig_type: int) -> BalanceAllowanceParams:
                return BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=sig_type,
                )

            for sig_type in POLYMARKET_SIGNATURE_TYPES:
                if not self._signature_type_supported(sig_type):
                    continue
                try:
                    params = build_params(sig_type)
                    await self._run_client_io(self._client.update_balance_allowance, params)
                except Exception as exc:
                    logger.debug(
                        "CLOB balance-allowance cache refresh failed for sig_type=%d: %s",
                        sig_type,
                        exc,
                    )
        except Exception as exc:
            logger.warning("CLOB balance-allowance cache refresh failed (non-fatal): %s", exc)

    async def refresh_conditional_balance_allowance(self, token_id: str) -> bool:
        token_key = str(token_id or "").strip()
        if not token_key:
            return False
        if not self.is_ready() and not await self.ensure_initialized():
            return False
        if not self.is_ready():
            return False

        try:
            await self._refresh_signature_type()
        except Exception as exc:
            logger.debug("Conditional allowance refresh skipped because signature refresh failed", exc_info=exc)

        signature_type = self._resolved_signature_type()
        if not isinstance(signature_type, int) or not self._signature_type_supported(signature_type):
            return False

        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_key,
                signature_type=signature_type,
            )
            await self._run_client_io(self._client.update_balance_allowance, params)
            return True
        except Exception as exc:
            logger.warning(
                "Conditional balance-allowance refresh failed",
                token_id=token_key,
                signature_type=signature_type,
                exc_info=exc,
            )
            return False

    async def refresh_collateral_balance_allowance(self) -> bool:
        if not self.is_ready() and not await self.ensure_initialized():
            return False
        if not self.is_ready():
            return False

        try:
            await self._refresh_signature_type()
        except Exception as exc:
            logger.debug("Collateral allowance refresh skipped because signature refresh failed", exc_info=exc)

        signature_type = self._resolved_signature_type()
        if not isinstance(signature_type, int) or not self._signature_type_supported(signature_type):
            return False

        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=signature_type,
            )
            await self._run_client_io(self._client.update_balance_allowance, params)
            return True
        except Exception as exc:
            logger.warning(
                "Collateral balance-allowance refresh failed",
                signature_type=signature_type,
                exc_info=exc,
            )
            return False

    async def prepare_sell_balance_allowance(self, token_id: str) -> bool:
        token_key = str(token_id or "").strip()
        if token_key:
            try:
                await self._select_signature_type_for_conditional_token(token_key)
            except Exception as exc:
                logger.debug(
                    "Sell allowance preparation could not refresh signature type",
                    token_id=token_key,
                    exc_info=exc,
                )
        conditional_refreshed = False
        if token_key:
            conditional_refreshed = await self.refresh_conditional_balance_allowance(token_key)
            if not conditional_refreshed:
                for signature_type in POLYMARKET_SIGNATURE_TYPES:
                    if not self._signature_type_supported(signature_type):
                        continue
                    self._balance_signature_type = signature_type
                    self._apply_signature_type_to_client(signature_type)
                    if await self.refresh_conditional_balance_allowance(token_key):
                        conditional_refreshed = True
                        break
        return conditional_refreshed

    async def _enforce_buy_pre_submit_gate(
        self,
        *,
        token_id: str,
        required_notional_usd: Decimal,
    ) -> tuple[bool, Optional[str]]:
        token_key = str(token_id or "").strip()
        required_usdc = max(ZERO, required_notional_usd)
        min_account_balance_usd = max(ZERO, _to_decimal(settings.MIN_ACCOUNT_BALANCE_USD))
        required_total_usdc = required_usdc + min_account_balance_usd
        if required_usdc <= ZERO:
            return False, "BUY pre-submit gate failed: required notional must be greater than zero."

        balance = await self.get_balance()
        if not isinstance(balance, dict):
            logger.warning(
                "Buy pre-submit gate skipped; balance payload unavailable",
                token_id=token_key,
            )
            return True, None
        if balance.get("error"):
            logger.warning(
                "Buy pre-submit gate skipped; could not fetch collateral balance/allowance",
                token_id=token_key,
                error=str(balance.get("error")),
            )
            return True, None

        available_raw = safe_float(balance.get("available"))
        balance_raw = safe_float(balance.get("balance"))
        if available_raw is None or balance_raw is None:
            logger.warning(
                "Buy pre-submit gate skipped; missing collateral balance fields",
                token_id=token_key,
                payload_keys=sorted(balance.keys()),
            )
            return True, None

        available = max(ZERO, _to_decimal(available_raw))
        collateral_balance = max(ZERO, _to_decimal(balance_raw))
        if available >= required_total_usdc:
            return True, None

        if collateral_balance >= required_total_usdc and available < required_total_usdc:
            await self.refresh_collateral_balance_allowance()
            refreshed_balance = await self.get_balance()
            refreshed_available_raw = safe_float(refreshed_balance.get("available")) if isinstance(refreshed_balance, dict) else None
            refreshed_balance_raw = safe_float(refreshed_balance.get("balance")) if isinstance(refreshed_balance, dict) else None
            if refreshed_available_raw is not None:
                available = max(ZERO, _to_decimal(refreshed_available_raw))
            if refreshed_balance_raw is not None:
                collateral_balance = max(ZERO, _to_decimal(refreshed_balance_raw))
            if available >= required_total_usdc:
                return True, None

        signature_value_raw = balance.get("signature_type")
        signature_value = (
            int(signature_value_raw) if isinstance(signature_value_raw, int) else self._resolved_signature_type()
        )
        funder_wallet = str(
            self._funder_for_signature_type(signature_value) or self._execution_wallet_address() or ""
        ).strip()
        shortfall = max(ZERO, required_total_usdc - available)
        post_trade_available = max(ZERO, available - required_usdc)
        error_message = (
            "BUY pre-submit gate failed: not enough collateral balance/allowance. "
            f"token_id={token_key} "
            f"required_usdc={required_usdc} required_total_usdc={required_total_usdc} "
            f"minimum_account_balance_usd={min_account_balance_usd} "
            f"available_usdc={available} post_trade_available_usdc={post_trade_available} shortfall_usdc={shortfall} "
            f"balance_usdc={collateral_balance} "
            f"signature_type={signature_value} funder_wallet={funder_wallet or 'unknown'}. "
            "Collateral may be held under a different funder/signature wallet or reserved by open orders."
        )
        logger.info(
            "Buy pre-submit balance gate blocked order",
            token_id=token_key,
            required_usdc=str(required_usdc),
            required_total_usdc=str(required_total_usdc),
            minimum_account_balance_usd=str(min_account_balance_usd),
            available_usdc=str(available),
            post_trade_available_usdc=str(post_trade_available),
            balance_usdc=str(collateral_balance),
            signature_type=signature_value,
            funder_wallet=funder_wallet or "unknown",
        )
        return False, error_message

    async def _enforce_sell_pre_submit_gate(self, *, token_id: str, size: float) -> tuple[bool, Optional[str]]:
        token_key = str(token_id or "").strip()
        required_shares = _to_decimal(size)
        if not token_key:
            return False, "SELL pre-submit gate failed: token_id is missing."
        if required_shares <= ZERO:
            return False, "SELL pre-submit gate failed: order size must be greater than zero."
        if not self.is_ready() and not await self.ensure_initialized():
            return False, "SELL pre-submit gate failed: trading service is not initialized."
        if not self.is_ready():
            return False, "SELL pre-submit gate failed: trading service is not initialized."

        signature_type = await self._select_signature_type_for_conditional_token(token_key)
        if signature_type is None:
            resolved = self._resolved_signature_type()
            signature_type = resolved if self._signature_type_supported(resolved) else None

        if not isinstance(signature_type, int):
            logger.warning(
                "Sell pre-submit gate skipped; no supported signature type available",
                token_id=token_key,
            )
            return True, None

        snapshot = await self._fetch_conditional_balance_snapshot(
            token_key,
            signature_type,
            refresh=False,
        )
        snapshot_refreshed = False
        if snapshot is None:
            snapshot = await self._fetch_conditional_balance_snapshot(
                token_key,
                signature_type,
                refresh=True,
            )
            snapshot_refreshed = snapshot is not None
        if snapshot is None:
            logger.warning(
                "Sell pre-submit gate skipped; conditional balance snapshot unavailable",
                token_id=token_key,
                signature_type=signature_type,
            )
            return True, None

        balance_raw = max(ZERO, snapshot["balance_raw"])
        allowance_raw = max(ZERO, snapshot["allowance_raw"])
        available_raw = max(ZERO, snapshot["available_raw"])
        required_raw = max(ZERO, required_shares)

        if available_raw >= required_raw:
            return True, None

        if not snapshot_refreshed:
            refreshed_snapshot = await self._fetch_conditional_balance_snapshot(
                token_key,
                int(snapshot["signature_type"]),
                refresh=True,
            )
            if refreshed_snapshot is not None:
                snapshot = refreshed_snapshot
                balance_raw = max(ZERO, snapshot["balance_raw"])
                allowance_raw = max(ZERO, snapshot["allowance_raw"])
                available_raw = max(ZERO, snapshot["available_raw"])
                if available_raw >= required_raw:
                    return True, None

        if balance_raw >= required_raw and allowance_raw < required_raw:
            await self.refresh_conditional_balance_allowance(token_key)
            refreshed_signature_type = await self._select_signature_type_for_conditional_token(token_key)
            if isinstance(refreshed_signature_type, int):
                refreshed_snapshot = await self._fetch_conditional_balance_snapshot(
                    token_key,
                    refreshed_signature_type,
                    refresh=True,
                )
                if refreshed_snapshot is not None:
                    snapshot = refreshed_snapshot
                    balance_raw = max(ZERO, snapshot["balance_raw"])
                    allowance_raw = max(ZERO, snapshot["allowance_raw"])
                    available_raw = max(ZERO, snapshot["available_raw"])
                    if available_raw >= required_raw:
                        return True, None

        signature_value = int(snapshot["signature_type"])
        funder_wallet = str(
            self._funder_for_signature_type(signature_value) or self._execution_wallet_address() or ""
        ).strip()
        shortfall = max(ZERO, required_raw - available_raw)
        error_message = (
            "SELL pre-submit gate failed: not enough conditional token balance/allowance. "
            f"token_id={token_key} "
            f"required_shares={required_raw} available_shares={available_raw} shortfall_shares={shortfall} "
            f"balance_shares={balance_raw} allowance_shares={allowance_raw} "
            f"signature_type={signature_value} funder_wallet={funder_wallet or 'unknown'}. "
            "Shares may be held under a different funder/signature wallet or reserved by open orders."
        )
        logger.warning(
            "Sell pre-submit balance gate blocked order",
            token_id=token_key,
            required_shares=str(required_raw),
            available_shares=str(available_raw),
            balance_shares=str(balance_raw),
            allowance_shares=str(allowance_raw),
            signature_type=signature_value,
            funder_wallet=funder_wallet or "unknown",
        )
        return False, error_message

    async def ensure_initialized(self) -> bool:
        if self.is_ready():
            await self._sync_trading_transport()
            return True
        if self._init_retry_not_before is not None and utcnow() < self._init_retry_not_before:
            return False
        return await self.initialize()

    async def initialize(self) -> bool:
        """
        Initialize the trading client with API credentials.

        Returns True if successfully initialized, False otherwise.
        """
        init_lock = self._get_init_lock()
        async with init_lock:
            (
                private_key,
                api_key,
                api_secret,
                api_passphrase,
                credential_source,
            ) = await self._resolve_polymarket_credentials()
            if not all([private_key, api_key, api_secret, api_passphrase]):
                logger.error("Missing Polymarket API credentials. Cannot initialize trading.")
                self._last_init_error = "missing_polymarket_credentials"
                self._init_retry_not_before = None
                return False

            self._eoa_address = None
            self._proxy_funder_address = None
            self._last_init_error = None
            self._init_retry_not_before = None

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

                sig_type = int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1))
                eoa_address = Account.from_key(private_key).address
                self._eoa_address = eoa_address
                funder = await asyncio.to_thread(
                    self._resolve_polymarket_funder,
                    eoa_address,
                    sig_type,
                )
                if sig_type in (1, 2):
                    self._proxy_funder_address = funder
                    if funder:
                        logger.info(
                            "Resolved proxy funder=%s for EOA=%s signature_type=%s",
                            funder,
                            eoa_address,
                            sig_type,
                        )
                    else:
                        logger.error(
                            "Missing proxy funder for signature_type=%s. Set POLYMARKET_FUNDER or switch to signature_type=0.",
                            sig_type,
                        )
                        self._last_init_error = f"missing_proxy_funder_signature_type_{sig_type}"
                        self._init_retry_not_before = None
                        self._initialized = False
                        self._client = None
                        self._wallet_address = None
                        self._eoa_address = None
                        self._proxy_funder_address = None
                        return False

                self._client = ClobClient(
                    host=settings.CLOB_API_URL,
                    key=private_key,
                    chain_id=settings.CHAIN_ID,
                    creds=creds,
                    signature_type=sig_type,
                    funder=funder,
                )
                self._wallet_address = eoa_address
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
                # Apply restored sig_type to builder immediately so that even if
                # the get_balance() probe below fails, orders are signed correctly.
                self._apply_signature_type_to_client(self._balance_signature_type)
                if isinstance(self._balance_signature_type, int):
                    logger.info(
                        "Restored signature type=%s from runtime state",
                        self._balance_signature_type,
                    )
                await self._approve_clob_allowance()
                # Probe all signature types to find which one has balance/allowance.
                # This sets self._balance_signature_type and client signature settings
                # so orders are signed with the correct type (POLY_PROXY=1 for most wallets).
                try:
                    balance_info = await self.get_balance()
                    if "error" not in balance_info:
                        logger.info(
                            "Balance probe complete: sig_type=%s balance=%s",
                            self._balance_signature_type,
                            balance_info.get("balance"),
                        )
                except Exception as _bal_exc:
                    logger.warning("Balance probe during init failed (non-fatal): %s", _bal_exc)
                self._schedule_restored_reconciliations()
                logger.info("Trading service initialized successfully", credential_source=credential_source)
                self._last_init_error = None
                self._init_retry_not_before = None
                return True

            except ImportError:
                now = utcnow()
                if (
                    self._last_missing_dependency_log_at is None
                    or (now - self._last_missing_dependency_log_at).total_seconds() >= MISSING_DEPENDENCY_RELOG_SECONDS
                ):
                    logger.error("py-clob-client not installed. Run: pip install py-clob-client")
                    self._last_missing_dependency_log_at = now
                self._last_init_error = "py-clob-client not installed"
                self._init_retry_not_before = now + timedelta(seconds=INITIALIZATION_RETRY_BACKOFF_SECONDS)
                self._initialized = False
                self._client = None
                self._wallet_address = None
                self._eoa_address = None
                self._proxy_funder_address = None
                return False
            except Exception as e:
                logger.error(f"Failed to initialize trading client: {e}")
                self._last_init_error = str(e)
                self._init_retry_not_before = None
                self._initialized = False
                self._client = None
                self._wallet_address = None
                self._eoa_address = None
                self._proxy_funder_address = None
                return False

    def is_ready(self) -> bool:
        """Check if trading service is ready"""
        return self._initialized and self._client is not None

    def get_last_init_error(self) -> Optional[str]:
        return str(self._last_init_error or "").strip() or None

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

    def _resolved_signature_type(self) -> int:
        if isinstance(self._balance_signature_type, int):
            return int(self._balance_signature_type)
        builder = getattr(self._client, "builder", None)
        if builder is not None and isinstance(getattr(builder, "sig_type", None), int):
            return int(builder.sig_type)
        return int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1))

    def _execution_wallet_address(self) -> Optional[str]:
        signature_type = self._resolved_signature_type()
        funder = str(self._funder_for_signature_type(signature_type) or "").strip()
        if funder:
            return funder
        if self._wallet_address:
            return str(self._wallet_address).strip()
        if self._eoa_address:
            return str(self._eoa_address).strip()
        return self._get_wallet_address()

    def get_execution_wallet_address(self) -> Optional[str]:
        return self._execution_wallet_address()

    def _wallet_for_persistence(self) -> Optional[str]:
        wallet = str(self._execution_wallet_address() or "").strip()
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
            for attempt in range(_DB_RETRY_ATTEMPTS):
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
                        is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
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
            for attempt in range(_DB_RETRY_ATTEMPTS):
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
                        is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
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
        market_positions_json = {str(token_id): str(exposure) for token_id, exposure in self._market_positions.items()}
        pending_reconciliation_json = [dict(item) for item in self._pending_reconciliations]

        persist_lock = self._get_persist_lock()
        async with persist_lock:
            for attempt in range(_DB_RETRY_ATTEMPTS):
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
                        row.pending_reconciliation_json = pending_reconciliation_json
                        row.balance_signature_type = self._balance_signature_type
                        row.updated_at = utcnow()
                        await session.commit()
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
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
            for attempt in range(_DB_RETRY_ATTEMPTS):
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
                            order_type = _normalize_order_type(row.order_type)
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
                            restored_reconciliations: list[dict[str, Any]] = []
                            for raw_reconciliation in runtime_row.pending_reconciliation_json or []:
                                normalized = self._normalize_pending_reconciliation(raw_reconciliation)
                                if normalized is not None:
                                    restored_reconciliations.append(normalized)
                            self._pending_reconciliations = restored_reconciliations
                            if runtime_row.balance_signature_type is not None:
                                self._balance_signature_type = int(runtime_row.balance_signature_type)
                        else:
                            self._stats.open_positions = len(self._positions)
                            self._pending_reconciliations = []

                        self._runtime_state_loaded_for_wallet = wallet
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
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
            self._start_background_task(
                self._persist_runtime_state(),
                name="live-execution-persist-runtime-state",
            )

    def _extract_server_orders(self, response: Any) -> list[dict[str, Any]]:
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        if isinstance(response, dict):
            for key in ("orders", "data", "items", "results"):
                items = response.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
            for key in ("order", "result"):
                item = response.get(key)
                if isinstance(item, dict):
                    return [item]
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
        if status_key in {"canceling", "cancelling"}:
            return "pending"
        if status_key in {"cancelled", "canceled", "killed", "void", "terminated"}:
            return "cancelled"
        if status_key in {"expired", "timed_out", "timeout"}:
            return "expired"
        if status_key in {"failed", "rejected", "error", "invalid", "invalidated", "malformed", "dead"}:
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
            server_order.get("id")
            or server_order.get("orderID")
            or server_order.get("orderId")
            or server_order.get("order_id")
            or ""
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
            "filledAmount",
            "filledSize",
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
            "avgPrice",
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

        cached_fallback: dict[str, dict[str, Any]] = {}
        for order in self._orders.values():
            cached_snapshot = self._snapshot_from_cached_order(order)
            if cached_snapshot is None:
                continue
            clob_id = str(cached_snapshot["clob_order_id"])
            if clob_id in requested:
                cached_fallback[clob_id] = cached_snapshot

        if not self.is_ready():
            try:
                await self.ensure_initialized()
            except Exception as exc:
                logger.debug("Order snapshot refresh could not initialize trading client", exc_info=exc)
            if not self.is_ready():
                return cached_fallback

        snapshots: dict[str, dict[str, Any]] = {}
        provider_fetch_ok = False
        per_order_not_found: set[str] = set()

        def _ingest_open_orders(response_payload: Any) -> None:
            nonlocal provider_fetch_ok
            provider_fetch_ok = True
            for server_order in self._extract_server_orders(response_payload):
                snapshot = self._parse_provider_order_snapshot(server_order)
                if snapshot is None:
                    continue
                clob_id = str(snapshot["clob_order_id"])
                if clob_id in requested:
                    snapshots[clob_id] = snapshot

        try:
            response = await self._run_client_io(self._client.get_orders)
            _ingest_open_orders(response)
        except Exception as exc:
            if _is_transient_transport_error(exc):
                logger.warning("Failed to fetch open provider orders", exc_info=exc)
            else:
                logger.error("Failed to fetch open provider orders", exc_info=exc)
            try:
                reinitialized = await self.ensure_initialized()
            except Exception as reinit_exc:
                logger.warning(
                    "Trading client reinitialization failed while fetching order snapshots",
                    exc_info=reinit_exc,
                )
                reinitialized = False
            if reinitialized and self.is_ready():
                try:
                    response = await self._run_client_io(self._client.get_orders)
                    _ingest_open_orders(response)
                except Exception as retry_exc:
                    if _is_transient_transport_error(retry_exc):
                        logger.warning(
                            "Failed to fetch open provider orders after reinitializing trading client",
                            exc_info=retry_exc,
                        )
                    else:
                        logger.error(
                            "Failed to fetch open provider orders after reinitializing trading client",
                            exc_info=retry_exc,
                        )

        missing = requested.difference(snapshots.keys())
        if missing and hasattr(self._client, "get_order"):
            for clob_id in sorted(missing):
                try:
                    single_response = await self._run_client_io(self._client.get_order, clob_id)
                except Exception as exc:
                    error_text = str(exc).lower()
                    if "not found" in error_text or "does not exist" in error_text:
                        per_order_not_found.add(clob_id)
                    logger.debug("Provider single-order lookup failed", clob_order_id=clob_id, exc_info=exc)
                    continue
                parsed_exact = False
                for server_order in self._extract_server_orders(single_response):
                    snapshot = self._parse_provider_order_snapshot(server_order)
                    if snapshot is None:
                        continue
                    if str(snapshot["clob_order_id"]) == clob_id:
                        snapshots[clob_id] = snapshot
                        parsed_exact = True
                        break
                if not parsed_exact and isinstance(single_response, dict):
                    error_text = str(
                        single_response.get("error")
                        or single_response.get("errorMsg")
                        or single_response.get("message")
                        or ""
                    ).lower()
                    if "not found" in error_text or "does not exist" in error_text:
                        per_order_not_found.add(clob_id)

        order_by_clob: dict[str, Order] = {}
        for order in self._orders.values():
            clob_id = str(order.clob_order_id or "").strip()
            if clob_id:
                order_by_clob[clob_id] = order

        token_positions: dict[str, Position] = {}
        if provider_fetch_ok and per_order_not_found:
            try:
                await self.sync_positions()
                token_positions = dict(self._positions)
            except Exception as exc:
                logger.debug("Position sync for snapshot reconciliation failed", exc_info=exc)

        unresolved = requested.difference(snapshots.keys())
        for clob_id in sorted(unresolved):
            cached_snapshot = cached_fallback.get(clob_id)
            if provider_fetch_ok and clob_id in per_order_not_found:
                synthesized = dict(cached_snapshot or {"clob_order_id": clob_id})
                prior_filled_size = safe_float(synthesized.get("filled_size"), 0.0) or 0.0
                if prior_filled_size <= 0.0:
                    local_order = order_by_clob.get(clob_id)
                    if local_order is not None:
                        position = token_positions.get(str(local_order.token_id or ""))
                        if position is not None and float(position.size or 0.0) > 0:
                            inferred_filled_size = float(position.size)
                            inferred_avg_price = safe_float(position.average_cost)
                            if inferred_avg_price is None or inferred_avg_price <= 0:
                                inferred_avg_price = safe_float(synthesized.get("limit_price"))
                            synthesized["filled_size"] = inferred_filled_size
                            if inferred_avg_price is not None and inferred_avg_price > 0:
                                synthesized["average_fill_price"] = float(inferred_avg_price)
                                synthesized["filled_notional_usd"] = float(inferred_filled_size * inferred_avg_price)
                            prior_filled_size = inferred_filled_size
                synthesized["normalized_status"] = "filled" if prior_filled_size > 0 else "cancelled"
                synthesized["raw_status"] = "not_found"
                synthesized["raw"] = {"status": "not_found"}
                snapshots[clob_id] = synthesized
                continue
            if cached_snapshot is not None:
                snapshots[clob_id] = cached_snapshot

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
            provider_response = await self._run_client_io(self._client.get_orders)
        except Exception as exc:
            if _is_transient_transport_error(exc):
                logger.warning("Failed to fetch provider open orders", exc_info=exc)
            else:
                logger.error("Failed to fetch provider open orders", exc_info=exc)
            return []

        provider_orders = self._extract_server_orders(provider_response)
        updated_orders: list[Order] = []
        provider_clob_ids: set[str] = set()
        existing_by_clob: dict[str, list[Order]] = {}
        for cached in self._orders.values():
            cached_clob = str(cached.clob_order_id or "").strip()
            if not cached_clob:
                continue
            existing_by_clob.setdefault(cached_clob, []).append(cached)

        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        }
        for server_order in provider_orders:
            snapshot = self._parse_provider_order_snapshot(server_order)
            if snapshot is None:
                continue

            clob_order_id = str(snapshot["clob_order_id"])
            provider_clob_ids.add(clob_order_id)
            candidates = existing_by_clob.get(clob_order_id, [])
            local_order: Optional[Order] = None
            for candidate in candidates:
                if not str(candidate.id or "").startswith("clob:"):
                    local_order = candidate
                    break
            if local_order is None and candidates:
                local_order = max(candidates, key=lambda order: order.updated_at)

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
                side_raw = (
                    str(
                        server_order.get("side")
                        or server_order.get("order_side")
                        or server_order.get("direction")
                        or "BUY"
                    )
                    .strip()
                    .upper()
                )
                side = OrderSide.SELL if side_raw == OrderSide.SELL.value else OrderSide.BUY
                order_type_raw = (
                    str(
                        server_order.get("order_type")
                        or server_order.get("orderType")
                        or server_order.get("type")
                        or "GTC"
                    )
                    .strip()
                    .upper()
                )
                order_type = _normalize_order_type(order_type_raw)

                created_at = _parse_provider_datetime(
                    server_order.get("created_at") or server_order.get("createdAt") or server_order.get("timestamp")
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
                candidates = [local_order]
                existing_by_clob[clob_order_id] = candidates

            if not local_order.market_question:
                local_order.market_question = (
                    str(
                        server_order.get("market_question")
                        or server_order.get("question")
                        or server_order.get("title")
                        or ""
                    )
                    or None
                )
            if local_order.size <= 0:
                local_order.size = float(snapshot.get("size") or snapshot.get("filled_size") or 0.0)
            if local_order.price <= 0:
                local_order.price = float(snapshot.get("limit_price") or 0.0)

            self._apply_snapshot_to_order(local_order, snapshot)
            updated_orders.append(local_order)
            for duplicate in candidates:
                if duplicate.id == local_order.id:
                    continue
                duplicate.status = (
                    OrderStatus.FILLED if float(duplicate.filled_size or 0.0) > 0 else OrderStatus.CANCELLED
                )
                duplicate.updated_at = utcnow()
                updated_orders.append(duplicate)

        for cached in self._orders.values():
            clob_order_id = str(cached.clob_order_id or "").strip()
            if not clob_order_id:
                continue
            if clob_order_id in provider_clob_ids:
                continue
            if cached.status not in active_statuses:
                continue
            if not str(cached.id or "").startswith("clob:"):
                continue
            cached.status = OrderStatus.FILLED if float(cached.filled_size or 0.0) > 0 else OrderStatus.CANCELLED
            cached.updated_at = utcnow()
            updated_orders.append(cached)

        if updated_orders:
            await self._persist_orders(updated_orders)

        open_by_key: dict[str, Order] = {}
        for order in self._orders.values():
            if order.status not in active_statuses:
                continue
            clob_order_id = str(order.clob_order_id or "").strip()
            key = clob_order_id if clob_order_id else f"id:{order.id}"
            existing = open_by_key.get(key)
            if existing is None:
                open_by_key[key] = order
                continue
            existing_is_synthetic = str(existing.id or "").startswith("clob:")
            order_is_synthetic = str(order.id or "").startswith("clob:")
            if existing_is_synthetic and not order_is_synthetic:
                open_by_key[key] = order
                continue
            if existing.updated_at < order.updated_at:
                open_by_key[key] = order

        return sorted(open_by_key.values(), key=lambda order: order.created_at, reverse=True)

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
        min_order_size_usd: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Validate order against safety limits."""
        self._check_daily_reset()

        if global_pause_state.is_paused:
            return False, "Global pause is active"

        if not self.is_ready():
            return False, "Trading service not initialized"

        min_order_floor = StrategySDK.resolve_min_order_size_usd(
            {"min_order_size_usd": min_order_size_usd} if min_order_size_usd is not None else {},
            fallback=float(settings.MIN_ORDER_SIZE_USD),
        )
        min_order_size = _to_decimal(min_order_floor)
        max_trade_size = _to_decimal(settings.MAX_TRADE_SIZE_USD)
        max_daily_volume = _to_decimal(settings.MAX_DAILY_TRADE_VOLUME)
        max_per_market = _to_decimal(self.MAX_PER_MARKET_USD)

        if size_usd < min_order_size:
            return (
                False,
                f"Order size ${float(size_usd):.2f} below minimum ${float(min_order_floor):.2f}",
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
        min_order_size_usd: Optional[float] = None,
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
                min_order_size_usd=min_order_size_usd,
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
        post_only: bool = False,
        min_order_size_usd: Optional[float] = None,
        market_question: Optional[str] = None,
        opportunity_id: Optional[str] = None,
        skip_buy_pre_submit_gate: bool = False,
    ) -> Order:
        """
        Place an order on Polymarket.

        Args:
            token_id: The CLOB token ID (YES or NO token)
            side: BUY or SELL
            price: Price per share (0-1)
            size: Number of shares
            order_type: GTC, FOK, GTD, FAK, or IOC
            post_only: If True, order is rejected if it would immediately match.
                       Only valid with GTC or GTD order types.
            market_question: Optional market question for reference
            opportunity_id: Optional opportunity ID this trade is from

        Returns:
            Order object with status
        """
        token_key = str(token_id or "").strip()
        if not token_key:
            raise ValueError("token_id is required")
        normalized_price = _validated_positive_float(price, field_name="price")
        normalized_size = _validated_positive_float(size, field_name="size")
        if normalized_price > 1.0:
            raise ValueError("price must be less than or equal to 1.0")

        order_id = str(uuid.uuid4())
        normalized_order_type = _normalize_order_type(order_type)
        order = Order(
            id=order_id,
            token_id=token_key,
            side=side,
            price=normalized_price,
            size=normalized_size,
            order_type=normalized_order_type,
            market_question=market_question,
            opportunity_id=opportunity_id,
        )

        # Calculate USD notional with Decimal to avoid float accumulation drift.
        size_usd = _to_decimal(normalized_price) * _to_decimal(normalized_size)
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
            token_id=token_key,
            min_order_size_usd=min_order_size_usd,
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
            await self._refresh_signature_type()
            sell_allowance_retry_used = False
            if side == OrderSide.BUY and not skip_buy_pre_submit_gate:
                buy_gate_ok, buy_gate_error = await self._enforce_buy_pre_submit_gate(
                    token_id=token_key,
                    required_notional_usd=size_usd,
                )
                if not buy_gate_ok:
                    raise RuntimeError(buy_gate_error or "BUY pre-submit gate failed")
            if side == OrderSide.SELL:
                await self.prepare_sell_balance_allowance(token_key)
                sell_gate_ok, sell_gate_error = await self._enforce_sell_pre_submit_gate(
                    token_id=token_key,
                    size=normalized_size,
                )
                if not sell_gate_ok:
                    raise RuntimeError(sell_gate_error or "SELL pre-submit gate failed")

            # Build and sign order using py-clob-client
            from py_clob_client.clob_types import MarketOrderArgs, OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL

            submit_price = float(normalized_price)
            order_side = BUY if side == OrderSide.BUY else SELL
            provider_order_type = _provider_order_type_value(normalized_order_type)
            provider_market_order = provider_order_type in {OrderType.FAK.value, OrderType.FOK.value}

            transport_retries_used = 0
            max_transport_retries = 2
            max_attempts = 3 if side == OrderSide.SELL else 2
            for attempt in range(max_attempts + max_transport_retries):
                order.price = submit_price
                try:
                    async with self._get_client_io_lock():
                        if provider_market_order:
                            market_amount = float(normalized_size)
                            if side == OrderSide.BUY:
                                market_amount = float(max(0.0, submit_price) * max(0.0, normalized_size))
                            order_args = MarketOrderArgs(
                                token_id=token_key,
                                amount=market_amount,
                                side=order_side,
                                price=submit_price,
                                order_type=provider_order_type,
                            )
                            signed_order = await asyncio.to_thread(self._client.create_market_order, order_args)
                        else:
                            order_args = OrderArgs(
                                price=submit_price,
                                size=normalized_size,
                                side=order_side,
                                token_id=token_key,
                            )
                            signed_order = await asyncio.to_thread(self._client.create_order, order_args)
                        response = await asyncio.to_thread(
                            self._client.post_order,
                            signed_order,
                            provider_order_type,
                            post_only=post_only,
                        )
                    if not isinstance(response, dict):
                        raise RuntimeError("Trading provider returned malformed order response")
                except Exception as exc:
                    error_text = str(exc).lower()
                    if attempt == 0 and self._is_invalid_signature_error(str(exc)):
                        if await self._refresh_signature_type(force=True):
                            logger.warning(
                                "Order creation failed with invalid signature; refreshing and retrying",
                                attempt=attempt + 1,
                                token_id=token_key,
                                side=side.value,
                            )
                            await asyncio.sleep(0)
                            continue
                    if (
                        side == OrderSide.SELL
                        and not sell_allowance_retry_used
                        and "not enough balance / allowance" in error_text
                    ):
                        sell_allowance_retry_used = True
                        if await self.prepare_sell_balance_allowance(token_key):
                            logger.warning(
                                "Sell order creation failed with stale balance/allowance cache; refreshed allowances and retrying",
                                attempt=attempt + 1,
                                token_id=token_key,
                            )
                            await asyncio.sleep(0)
                            continue
                    if (
                        post_only
                        and _is_post_only_cross_reject(error_text)
                        and attempt < max_attempts - 1
                    ):
                        retry_price = _next_post_only_retry_price(side, submit_price)
                        if abs(retry_price - submit_price) >= 1e-9:
                            logger.warning(
                                "Post-only order crossed book; repricing one tick and retrying",
                                attempt=attempt + 1,
                                token_id=token_key,
                                side=side.value,
                                from_price=round(submit_price, 6),
                                to_price=round(retry_price, 6),
                            )
                            submit_price = retry_price
                            await asyncio.sleep(0)
                            continue
                    if (
                        _is_transient_transport_error(exc)
                        and transport_retries_used < max_transport_retries
                    ):
                        transport_retries_used += 1
                        delay = 0.5 * (2 ** (transport_retries_used - 1))
                        logger.warning(
                            "Order submission failed with transient transport error; retrying",
                            attempt=attempt + 1,
                            transport_retry=transport_retries_used,
                            token_id=token_key,
                            side=side.value,
                            error_type=type(exc).__name__,
                            error=str(exc)[:200],
                            delay=delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

                if response.get("success"):
                    order.status = OrderStatus.OPEN
                    order.clob_order_id = response.get("orderID")
                    setattr(order, "_provider_order_type_sent", provider_order_type)
                    setattr(
                        order,
                        "_submit_method",
                        "create_market_order" if provider_market_order else "create_order",
                    )
                    immediate_snapshot = self._parse_provider_order_snapshot(response)
                    if immediate_snapshot is not None:
                        self._apply_snapshot_to_order(order, immediate_snapshot)
                    stats_lock = self._get_stats_lock()
                    async with stats_lock:
                        self._stats.total_trades += 1
                        self._stats.last_trade_at = utcnow()
                    await self._persist_runtime_state()
                    logger.info(f"Order placed successfully: {order.clob_order_id}")
                    break

                error_message = str(response.get("errorMsg", response.get("error", "Unknown error")))
                if (
                    attempt == 0
                    and self._is_invalid_signature_error(error_message)
                    and await self._refresh_signature_type(force=True)
                ):
                    logger.warning(
                        "Order rejected with invalid signature; refreshing and retrying",
                        attempt=attempt + 1,
                        token_id=token_key,
                        side=side.value,
                    )
                    await asyncio.sleep(0)
                    continue
                if (
                    side == OrderSide.SELL
                    and not sell_allowance_retry_used
                    and "not enough balance / allowance" in error_message.lower()
                ):
                    sell_allowance_retry_used = True
                    if await self.prepare_sell_balance_allowance(token_key):
                        logger.warning(
                            "Sell order rejected with stale balance/allowance cache; refreshed allowances and retrying",
                            attempt=attempt + 1,
                            token_id=token_key,
                        )
                        await asyncio.sleep(0)
                        continue
                if (
                    post_only
                    and _is_post_only_cross_reject(error_message)
                    and attempt < max_attempts - 1
                ):
                    retry_price = _next_post_only_retry_price(side, submit_price)
                    if abs(retry_price - submit_price) >= 1e-9:
                        logger.warning(
                            "Post-only order crossed book; repricing one tick and retrying",
                            attempt=attempt + 1,
                            token_id=token_key,
                            side=side.value,
                            from_price=round(submit_price, 6),
                            to_price=round(retry_price, 6),
                        )
                        submit_price = retry_price
                        await asyncio.sleep(0)
                        continue

                order.status = OrderStatus.FAILED
                order.error_message = error_message
                await self._release_reservation(
                    size_usd=size_usd,
                    side=side,
                    token_id=token_key,
                )
                reserved = False
                logger.error(f"Order failed: {order.error_message}")
                break

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            if reserved:
                await self._release_reservation(
                    size_usd=size_usd,
                    side=side,
                    token_id=token_key,
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
        post_only: bool = False,
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
            post_only: If True, reject if order would immediately match
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
                post_only=post_only,
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

        async def _release_cancelled_reservation(order: Order) -> None:
            remaining_shares = max(0.0, float(order.size or 0.0) - float(order.filled_size or 0.0))
            if remaining_shares <= 0.0:
                return
            price = safe_float(order.price, None)
            if price is None or price <= 0.0:
                return
            side = order.side if isinstance(order.side, OrderSide) else None
            if side is None:
                side_text = str(order.side or "").strip().upper()
                if side_text not in {"BUY", "SELL"}:
                    return
                side = OrderSide(side_text)
            token_id = str(order.token_id or "").strip() or None
            await self._release_reservation(
                size_usd=_to_decimal(price) * _to_decimal(remaining_shares),
                side=side,
                token_id=token_id,
            )

        local_order = self._orders.get(order_key)
        if local_order is not None:
            if local_order.status not in {OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                logger.warning(f"Cannot cancel order in status: {local_order.status}")
                return False
            clob_order_id = str(local_order.clob_order_id or "").strip()
            if not clob_order_id:
                local_order.status = OrderStatus.CANCELLED
                local_order.updated_at = utcnow()
                try:
                    await _release_cancelled_reservation(local_order)
                except Exception as exc:
                    logger.warning(
                        "Failed to release reservation for locally cancelled order",
                        order_id=local_order.id,
                        exc_info=exc,
                    )
                await self._persist_orders([local_order])
                await self._persist_runtime_state()
                return True
        else:
            clob_order_id = order_key

        if not self.is_ready() and not await self.ensure_initialized():
            logger.warning("Trading service not ready for order cancellation", order_id=order_key)
            return False

        try:
            response = await self._run_client_io(self._client.cancel, clob_order_id)
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
        changed_orders_by_id: dict[str, Order] = {}
        if local_order is not None:
            local_order.status = OrderStatus.CANCELLED
            local_order.updated_at = now
            changed_orders_by_id[str(local_order.id)] = local_order
        for order in self._orders.values():
            if str(order.clob_order_id or "").strip() == clob_order_id:
                order.status = OrderStatus.CANCELLED
                order.updated_at = now
                changed_orders_by_id[str(order.id)] = order
        changed_orders = list(changed_orders_by_id.values())
        for order in changed_orders:
            try:
                await _release_cancelled_reservation(order)
            except Exception as exc:
                logger.warning(
                    "Failed to release reservation for cancelled order",
                    order_id=order.id,
                    clob_order_id=order.clob_order_id,
                    exc_info=exc,
                )
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
            message = f"Cancelled {cancelled_count} of {len(targets)} order(s); {failed_count} cancellation(s) failed."
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
        open_orders = await self._sync_provider_open_orders()
        clob_ids = [str(order.clob_order_id).strip() for order in open_orders if str(order.clob_order_id or "").strip()]
        if clob_ids:
            try:
                await self.get_order_snapshots_by_clob_ids(clob_ids)
            except Exception as exc:
                logger.error("Get orders error", exc_info=exc)

        return [
            self._orders.get(order.id, order)
            for order in open_orders
            if self._orders.get(order.id, order).status
            in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING}
        ]

    async def sync_positions(self) -> list[Position]:
        """Sync positions from Polymarket"""
        if not self.is_ready() and not await self.ensure_initialized():
            return list(self._positions.values())

        try:
            # Get positions from the wallet
            # Note: This uses the data API, not CLOB
            from services.polymarket import polymarket_client

            address = self._execution_wallet_address()
            if not address:
                return list(self._positions.values())

            positions_data = await polymarket_client.get_wallet_positions_with_prices(address)

            def _read_float(data: dict[str, Any], *keys: str) -> Optional[float]:
                for key in keys:
                    value = safe_float(data.get(key))
                    if value is not None:
                        return float(value)
                return None

            def _read_text(data: dict[str, Any], *keys: str) -> str:
                for key in keys:
                    value = str(data.get(key) or "").strip()
                    if value:
                        return value
                return ""

            next_positions: dict[str, Position] = {}
            for pos in positions_data:
                token_id = _read_text(pos, "asset", "asset_id", "assetId", "token_id", "tokenId")
                if not token_id:
                    continue

                market_id = (
                    _read_text(
                        pos,
                        "market",
                        "conditionId",
                        "condition_id",
                        "market_id",
                        "marketId",
                    )
                    or token_id
                )
                market_question = _read_text(pos, "title", "market_question", "marketQuestion", "question") or "Unknown"
                outcome = _read_text(pos, "outcome", "position_side", "side") or "UNKNOWN"

                size = _read_float(pos, "size", "amount", "shares", "position_size")
                average_cost = _read_float(
                    pos,
                    "avgCost",
                    "avg_cost",
                    "avgPrice",
                    "avg_price",
                    "average_cost",
                )
                current_price = _read_float(
                    pos,
                    "currentPrice",
                    "current_price",
                    "curPrice",
                    "cur_price",
                    "price",
                    "markPrice",
                    "mark_price",
                )
                current_value = _read_float(pos, "currentValue", "current_value")
                initial_value = _read_float(pos, "initialValue", "initial_value")
                unrealized_pnl = _read_float(
                    pos,
                    "unrealized_pnl",
                    "unrealizedPnl",
                    "cashPnl",
                    "cash_pnl",
                )

                if (size is None or size <= 0.0) and current_value is not None and current_value > 0.0:
                    if current_price is not None and current_price > 0.0:
                        size = current_value / current_price

                if size is None or size <= 0.0:
                    continue

                if (average_cost is None or average_cost <= 0.0) and initial_value is not None and initial_value > 0.0:
                    average_cost = initial_value / size

                if average_cost is None:
                    average_cost = 0.0

                if (
                    (current_price is None or current_price <= 0.0)
                    and current_value is not None
                    and current_value > 0.0
                ):
                    current_price = current_value / size

                if current_price is None:
                    current_price = 0.0

                if unrealized_pnl is None:
                    unrealized_pnl = (current_price - average_cost) * size

                next_positions[token_id] = Position(
                    token_id=token_id,
                    market_id=market_id,
                    market_question=market_question,
                    outcome=outcome,
                    size=float(size),
                    average_cost=float(average_cost),
                    current_price=float(current_price),
                    unrealized_pnl=float(unrealized_pnl),
                )

            self._positions = next_positions

            self._stats.open_positions = len(self._positions)
            await self._persist_positions()
            await self._persist_runtime_state()

        except Exception as e:
            logger.error("Sync positions error", exc_info=e)

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

        normalized_size_usd = _validated_positive_float(size_usd, field_name="size_usd")

        # Pre-validate all positions before any execution.
        valid_positions: list[dict[str, Any]] = []
        for position in positions:
            if not isinstance(position, dict):
                logger.warning("Execution position payload must be a dict", payload_type=type(position).__name__)
                continue
            token_key = str(position.get("token_id") or "").strip()
            if not token_key:
                logger.warning("Execution position missing token_id", position=position)
                continue
            execution_price = safe_float(position.get("price"), None, reject_nan_inf=True)
            if execution_price is None or execution_price <= 0.0 or execution_price > 1.0:
                logger.warning(
                    "Execution position has invalid price",
                    token_id=token_key,
                    price=position.get("price"),
                )
                continue
            normalized_position = dict(position)
            normalized_position["token_id"] = token_key
            normalized_position["price"] = float(execution_price)
            valid_positions.append(normalized_position)

        if not valid_positions:
            logger.error("No valid positions to execute")
            return []

        # Build order coroutines for parallel execution
        async def place_single_order(position: dict[str, Any]) -> Order:
            token_key = str(position.get("token_id") or "").strip()
            price = _validated_positive_float(position.get("price"), field_name="position.price")
            position_usd = normalized_size_usd / len(valid_positions)
            shares = _validated_positive_float(position_usd / price, field_name="shares")
            maker_mode = bool(coerce_bool(position.get("_maker_mode"), False))

            post_only_raw = position.get("post_only")
            if post_only_raw is None:
                post_only_raw = position.get("_post_only")
            post_only = bool(coerce_bool(post_only_raw, False)) or maker_mode
            tick_size = _tick_size_from_position(position)

            # Crypto 15-min markets: use maker mode to avoid taker fees
            # and earn rebates.  Place at best_bid (or 1 tick below ask)
            # to sit on the book as a maker order.
            if maker_mode:
                maker_price = safe_float(position.get("_maker_price"), None, reject_nan_inf=True)
                if maker_price is None or maker_price <= 0.0:
                    maker_price = price
                maker_price = _round_down_to_tick(float(maker_price) - (tick_size / 2.0), tick_size)
                price = min(0.99, max(tick_size, float(maker_price)))
                shares = _validated_positive_float(position_usd / price, field_name="shares")

            return await self.place_order(
                token_id=token_key,
                side=OrderSide.BUY,
                price=price,
                size=shares,
                post_only=post_only,
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
            await self._enqueue_pending_reconciliation(orders)

        return orders

    async def _auto_reconcile(self, orders: list[dict[str, Any]]) -> None:
        """Auto-unwind partial multi-leg fills to prevent one-sided exposure."""
        await asyncio.sleep(2)  # Brief delay before reconciliation
        valid_orders = [order for order in orders if isinstance(order, dict)]
        logger.warning("AUTO_RECONCILE: Unwinding filled legs", leg_count=len(valid_orders))
        failures: list[str] = []
        for order in valid_orders:
            token_key = str(order.get("token_id") or "").strip()
            side_raw = str(order.get("side") or "").strip().upper()
            filled_size = safe_float(order.get("filled_size"), None, reject_nan_inf=True)
            filled_price = safe_float(order.get("price"), None, reject_nan_inf=True)
            if not token_key or side_raw not in {OrderSide.BUY.value, OrderSide.SELL.value}:
                failures.append(f"invalid_order:{order.get('order_id') or 'unknown'}")
                continue
            if filled_size is None or filled_size <= 0.0 or filled_price is None or filled_price <= 0.0:
                failures.append(f"invalid_fill:{order.get('order_id') or token_key}")
                continue
            try:
                unwind = await self.place_order(
                    token_id=token_key,
                    side=OrderSide.SELL if side_raw == OrderSide.BUY.value else OrderSide.BUY,
                    price=_clamp_binary_price(filled_price * (0.95 if side_raw == OrderSide.BUY.value else 1.05)),
                    size=float(filled_size),
                    order_type=OrderType.FOK,
                    market_question=f"AUTO_RECONCILE: {order.get('market_question')}",
                    opportunity_id=str(order.get("opportunity_id") or "").strip() or None,
                )
            except Exception as exc:
                logger.error("Reconciliation order placement raised", token_id=token_key, exc_info=exc)
                failures.append(f"exception:{token_key}")
                continue

            if unwind.status == OrderStatus.FAILED:
                failures.append(f"failed:{token_key}:{unwind.error_message or 'unknown'}")
            else:
                logger.warning(
                    "Reconciliation order placed",
                    token_id=token_key,
                    status=unwind.status.value,
                    clob_order_id=unwind.clob_order_id,
                )

        if failures:
            raise RuntimeError("; ".join(failures[:5]))

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
            await self.ensure_initialized()
        if not self.is_ready():
            return {"error": "Polymarket credentials not configured"}

        try:
            address = self._execution_wallet_address()
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
                    await self._run_client_io(self._client.update_balance_allowance, params)
                except Exception as exc:
                    logger.warning(
                        "Balance allowance refresh failed; using cached values",
                        signature_type=signature_type,
                        exc_info=exc,
                    )
                try:
                    payload = await self._run_client_io(self._client.get_balance_allowance, params)
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
            if not self._signature_type_supported(int(primary_signature_type)):
                primary_signature_type = 0

            primary_snapshot, primary_error = await fetch_balance_snapshot(primary_signature_type)
            if primary_error:
                primary_snapshot = None

            best_snapshot = primary_snapshot
            needs_probe = primary_snapshot is None or (
                primary_snapshot["balance"] <= 0.0 and primary_snapshot["available"] <= 0.0
            )

            if needs_probe:
                for signature_type in POLYMARKET_SIGNATURE_TYPES:
                    if signature_type == primary_signature_type:
                        continue
                    if not self._signature_type_supported(signature_type):
                        continue
                    candidate_snapshot, candidate_error = await fetch_balance_snapshot(signature_type)
                    if candidate_error:
                        continue
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
            self._apply_signature_type_to_client(selected_signature_type)

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
live_execution_service = LiveExecutionService()
