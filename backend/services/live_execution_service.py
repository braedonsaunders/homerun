"""
Trading Service - Real order execution on Polymarket

This service handles real trading on Polymarket using the CLOB API.
It integrates with py-clob-client-v2 for order placement and management.

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
import os
import re
import time as _time
import collections
import concurrent.futures
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR
import uuid

from sqlalchemy import case, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

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
# Per-call cap on the synchronous Polymarket SDK methods
# (``create_order`` / ``create_market_order`` / ``post_order``).  Each
# of those is wrapped in ``asyncio.wait_for(asyncio.to_thread(...),
# timeout=_ORDER_SUBMIT_TIMEOUT_SECONDS)`` so the event loop can
# unblock if the gateway is unresponsive.  Two SDK calls per
# submission (sign + post) means the worst-case wall time of a single
# ``place_order`` is ~2 × this value; the OUTER ``asyncio.wait_for``
# at the lifecycle layer (``_LIVE_EXIT_ORDER_TIMEOUT_SECONDS`` /
# ``_LIVE_EXIT_RETRY_TIMEOUT_SECONDS``) MUST exceed that or it will
# cancel the SDK mid-protocol and leak the underlying thread (and
# half-submit a request to Polymarket).  20.0s pre-incident produced
# exactly that pathology against the lifecycle's 12s retry outer.
# 8s here keeps each gateway round-trip bounded under load while
# leaving room for the 22s outer to absorb both calls plus buffer.
_ORDER_SUBMIT_TIMEOUT_SECONDS = 8.0
_CLIENT_IO_TIMEOUT_SECONDS = 10.0
_CLOB_READ_TIMEOUT_SECONDS = 3.0
_CLOB_READ_CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive failures before opening
_CLOB_READ_CIRCUIT_BREAKER_COOLDOWN = 30.0  # seconds to wait before retrying
_CLOB_READ_FAILURE_LOG_INTERVAL = 30.0  # seconds between repeated failure logs
_SNAPSHOT_SINGLE_LOOKUP_BUDGET_SECONDS = 6.0  # cap the per-call single-order fallback loop
_SNAPSHOT_SINGLE_LOOKUP_MAX = 8  # cap how many single-order fetches we attempt per call
_OPEN_ORDER_SNAPSHOT_CACHE_TTL_SECONDS = 2.0
# PnL counters cache — 30s TTL.  ``_derive_pnl_counters_from_orders``
# runs a double-aggregate scan on ``trader_orders`` inside EVERY
# ``_persist_runtime_state`` call.  Production soak showed it hit
# 1.4s avg under DB pressure (5/2026/05 cycle 1 harness report).
# The result feeds the runtime_state snapshot row only — no
# decision logic depends on it, so 30s staleness is safe.
_PNL_COUNTERS_TTL_SECONDS = 300.0  # was 30s; cycle 7 of the perf-harness
# loop verified the underlying SQL is 1ms in isolation, so the 3.5s avg
# cache-miss latency observed under DB pressure is pool contention, not
# query speed.  Reducing miss FREQUENCY is therefore the right lever
# (an index can't help when the wait is for a connection).  300s is
# safe because the PnL counters power only the runtime_state snapshot
# row — they are informational telemetry, not decisional inputs.  The
# verifier writes ``actual_profit`` whenever an exit is matched on-chain,
# so a 5-min stale total_pnl/winning_trades count never miscounts a
# trader's actual position state.


# Balance cache TTL — was 5s, but the production soak (5/2026/05)
# showed every cycle paying a 6-SDK-call ``get_balance`` round trip
# (3 sig types × 2 ops each, all serialized on ``_client_io_lock``)
# whenever consecutive cycles fell outside the 5s window.  Each
# individual SDK call has a 10s timeout, so a worst-case fan-out under
# lock contention can eat 30+ s of ``ps_submit_order`` budget.
# Balance only changes when WE place an order (we know about that
# locally — ``_validate_and_reserve_order`` reserves against
# ``_daily_volume_usd`` and the per-trade gate computes
# ``post_trade_available`` deterministically) or when an external
# transfer funds the wallet (rare; the next cycle just past TTL
# will pick it up).  30s is a safe window that eliminates almost all
# the repeat fetches in normal operation.
_BALANCE_CACHE_TTL_SECONDS = 30.0


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


def _normalize_utc_datetime(value: Any) -> datetime | None:
    """Coerce ``value`` to a tz-aware UTC datetime, accepting either a
    real ``datetime`` or an ISO-8601 string.  The previous version
    declared ``datetime | None`` and then accessed ``value.tzinfo``
    unguarded — when the caller (or upstream JSON/DB deserialization)
    handed in a string, it raised ``AttributeError: 'str' object has
    no attribute 'tzinfo'``, which surfaced as ``Worker freshness
    check failed plane=all`` and triggered spurious worker restarts.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
        except Exception:
            return None
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_post_only_cross_reject(error_message: str | None) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    return "post-only" in text and "crosses book" in text


# CLOB rejects sells with this exact format when the wallet does not hold
# enough outcome-token shares to cover the order:
#   "the balance is not enough -> balance: 9080, order amount: 10790000"
# Both numbers are atomic units (10**6 per share).  This is a *share*
# shortage and is unrelated to the wallet's USDC allowance — refreshing
# the allowance cannot fix it.  When we can parse this shape out of an
# error, we know the live_trading_positions ledger has drifted from the
# chain and further retries with the same exit_size are futile.
_INSUFFICIENT_SHARE_BALANCE_PATTERN = re.compile(
    r"balance\s+is\s+not\s+enough.*?balance:\s*(\d+)\s*,\s*order\s+amount:\s*(\d+)",
    re.DOTALL,
)


def _parse_clob_share_balance_shortage(error_text: str | None) -> tuple[int, int] | None:
    """Return (actual_balance_atomic, requested_amount_atomic) or None.

    None means the message does NOT carry a share-balance shortage
    (e.g. it might be an allowance-only failure, which the caller
    should still retry by refreshing the USDC allowance).
    """
    if not error_text:
        return None
    match = _INSUFFICIENT_SHARE_BALANCE_PATTERN.search(error_text)
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except (TypeError, ValueError):
        return None


# Markers in exception text that indicate a transient network/proxy failure
# rather than a genuine order rejection.  py_clob_client_v2 wraps httpx transport
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
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
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


def _normalize_clob_metadata(value: Any) -> Optional[str]:
    """Coerce a CLOB ``OrderArgs.metadata`` payload to a valid bytes32 hex
    string, or ``None`` if the input is not usable.

    The ``py_clob_client_v2`` order builder feeds ``metadata`` through
    ``bytes.fromhex(value.replace("0x", "").zfill(64))`` so any non-hex
    character (a colon, brace, comma, etc.) raises ``ValueError`` from
    deep inside the signing thread and the order is lost. Upstream
    callers historically pass either:

      * a deterministic 0x-prefixed bytes32 (the fast-tier idempotency
        key) — keep as-is,
      * ``None`` / empty string — drop, the SDK substitutes BYTES32_ZERO,
      * a stringified dict / opportunity id / debug tag — drop with a
        warning; the order still places, it just loses venue-side
        discoverability.

    Returning ``None`` here is equivalent to omitting the kwarg, so the
    SDK falls back to its default zero metadata.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    body = text[2:] if text.lower().startswith("0x") else text
    if len(body) > 64:
        return None
    body = body.zfill(64)
    try:
        bytes.fromhex(body)
    except ValueError:
        return None
    return "0x" + body


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
    market_id: Optional[str] = None


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
    redeemable: bool = False
    counts_as_open: bool = True
    end_date: Optional[str] = None
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

    Uses the py-clob-client-v2 library for order placement.
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
        # ``_client_io_lock`` was historically a Lock — single-flight for
        # every CLOB SDK call.  In practice the SDK's create/post path is
        # thread-safe (signer is stateless eth_account, ``_http_client``
        # is a shared httpx.Client which httpx documents as concurrency-
        # safe), so the single-flight serialization was paying p90 ~1.2 s
        # of ``io_lock_wait`` for no correctness benefit when several
        # crypto signals arrived within the same tick.  A bounded
        # semaphore caps inflight CLOB calls so we don't pile a runaway
        # 30-call burst onto the venue, while letting the typical 2-4
        # parallel orders proceed concurrently.
        self._client_io_lock: Optional[asyncio.Semaphore] = None
        self._client_io_lock_concurrency = max(
            1, int(getattr(settings, "POLYMARKET_CLIENT_IO_CONCURRENCY", 8))
        )
        # Split off balance / read-only SDK calls to a separate lock so
        # they don't queue behind in-flight order submissions.  Pre-split
        # production saw ``_client_io_lock`` serialize ALL SDK calls — a
        # buy gate's 6 ``update_balance_allowance`` round-trips (3 sig
        # types × 2 ops each) would block any concurrent ``post_order``,
        # contributing to ``ps_submit_order`` blowing past 30 s.  The
        # order-creation lock still serializes ``create_order`` +
        # ``post_order`` pairs (the SDK signs once and submits with the
        # cached signature; pairing them avoids cross-order signature
        # interleaving).
        self._client_balance_io_lock: Optional[asyncio.Lock] = None
        # Persist locks — split per-table after production saw
        # ``persist_lock_wait`` of 4-6 s on every call.  All four
        # consumers (``_persist_orders``, ``_persist_positions``,
        # ``_persist_runtime_state``, ``_restore_runtime_state``) write
        # to DIFFERENT tables; serializing them on a single lock
        # turned independent operations into a queue.  Per-table locks
        # let the orders write run in parallel with a runtime-state
        # write, since they touch distinct rows on distinct tables.
        # ``_persist_lock`` remains defined for back-compat with
        # ``_get_persist_lock()`` callers (now an alias for the
        # runtime-state lock).
        self._persist_lock: Optional[asyncio.Lock] = None
        self._orders_persist_lock: Optional[asyncio.Lock] = None
        self._positions_persist_lock: Optional[asyncio.Lock] = None
        self._runtime_state_persist_dirty = False
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
        # Circuit breaker for CLOB API read operations (order sync, balance fetch)
        self._clob_read_consecutive_failures: int = 0
        self._clob_read_circuit_open_until: Optional[float] = None
        self._clob_read_last_failure_logged: float = 0.0
        self._open_order_snapshot_cache: Optional[dict[str, dict[str, Any]]] = None
        self._open_order_snapshot_cache_at: float = 0.0
        # Short-lived balance cache to avoid repeated CLOB calls within the same
        # order submission pipeline (signature refresh + buy gate both call get_balance).
        self._balance_cache: Optional[dict] = None
        self._balance_cache_at: float = 0.0
        # HTTP/2 keepalive task — pings /ok every few seconds so the
        # shared httpx.Client (and the underlying TLS+H2 stream) stays
        # warm.  httpx default ``keepalive_expiry=5.0s`` means a cold-
        # start submit pays a full TLS handshake (≈150-300 ms over the
        # VPN) any time trading idles past 5 s.  A cheap unauth GET
        # below the expiry window avoids that on every order placement.
        self._clob_keepalive_task: Optional[asyncio.Task] = None
        self._clob_keepalive_last_ok_at: float = 0.0
        self._clob_keepalive_last_fail_at: float = 0.0
        # Rolling window of recent keepalive RTTs (ms).  This is the
        # warm-connection baseline against which place_order's
        # ``post_order`` stage is decomposed: post_order minus this RTT
        # ≈ venue-side processing time, which tells us whether the next
        # latency win comes from network/TLS or from the venue.
        self._clob_keepalive_recent_ms: collections.deque = collections.deque(maxlen=20)
        self._clob_keepalive_log_every = 10  # log a stats summary every N pings
        # Dedicated thread pool for SDK calls on the CLOB hot path.
        # Default ``asyncio.to_thread`` uses the shared default executor
        # which is contended by everything else in the process (DB
        # adapter setup, pickle round-trips, file I/O on logs etc.) —
        # a transient burst on the default pool blocks order submission
        # for tens of ms.  A dedicated pool sized to match the order-
        # submission semaphore guarantees we never queue behind an
        # unrelated thread task.  Lazy-created on first hot-path call.
        self._clob_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # TTL cache for ``prepare_sell_balance_allowance``.  Each call
        # makes 2-4 HTTP roundtrips (signature_type select + conditional
        # balance refresh + collateral balance refresh).  ITER-2 traces
        # showed it costing 4-5 seconds on the sell hot path.  Sell-
        # frequent strategies (e.g. exit cascades after a fill) re-
        # refresh those values within seconds of each other for no
        # information gain — the venue's cache hasn't moved that fast.
        # A short TTL skips the refresh on consecutive sells while
        # still catching genuine drift after idle gaps.  ``last_at`` is
        # per-token so a sell on token A doesn't satisfy a sell on
        # token B (different conditional balance lookups).
        self._sell_allowance_last_at: dict[str, float] = {}
        self._sell_allowance_ttl_seconds = float(
            getattr(settings, "POLYMARKET_SELL_ALLOWANCE_TTL_SECONDS", 5.0)
        )

        # 2026-05-10: per-token cache for ``_select_signature_type_for_
        # conditional_token``.  That function is the dominant cost in
        # the SELL pre-submit critical path: it loops over every
        # supported Polymarket signature type (typically 3) calling
        # ``_fetch_conditional_balance_snapshot(refresh=True)`` for each
        # — i.e. 3 sequential venue refresh-fetches just to pick the
        # signature with the largest balance.  Without dedup,
        # ``place_order`` for a SELL invokes it BOTH from
        # ``prepare_sell_balance_allowance`` AND from
        # ``_enforce_sell_pre_submit_gate``, duplicating those 3 fetches
        # back-to-back (= 6 total).  Live traces showed this dominating
        # the per-candidate reconcile time at 13s on the slow
        # candidate, with breakdown ``prepare_sell_allowance=3344ms``
        # plus ``sell_pre_submit_gate=2563ms`` ≈ 5.9s of pre-submit
        # before the actual order even hits the wire.  The signature
        # type with the largest conditional balance for a given token
        # only moves when WE buy/sell that token — a 5 s TTL is safe
        # (matches ``_sell_allowance_ttl_seconds``) and lets the second
        # call be a near-zero-cost cache hit.
        self._signature_type_cache: dict[str, tuple[float, int]] = {}
        self._signature_type_cache_ttl_seconds = float(
            getattr(settings, "POLYMARKET_SIGNATURE_TYPE_CACHE_TTL_SECONDS", 5.0)
        )

    def _get_stats_lock(self) -> asyncio.Lock:
        if self._stats_lock is None:
            self._stats_lock = asyncio.Lock()
        return self._stats_lock

    def _get_init_lock(self) -> asyncio.Lock:
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    def _get_clob_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Dedicated thread pool for CLOB SDK calls.

        Sized to ``_client_io_lock_concurrency + 4`` so the order-
        submission semaphore can saturate the pool without blocking,
        with a small buffer for the keepalive ping and prewarm calls.
        Threads are tagged so stack dumps make their purpose obvious.
        """
        if self._clob_executor is None or self._clob_executor._shutdown:  # type: ignore[attr-defined]
            self._clob_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(4, self._client_io_lock_concurrency + 4),
                thread_name_prefix="clob-sdk",
            )
        return self._clob_executor

    def _get_client_io_lock(self) -> asyncio.Semaphore:
        # Returns a bounded semaphore (NOT an asyncio.Lock) so multiple
        # CLOB SDK calls can be in flight concurrently up to the
        # configured cap.  Callers use ``async with self._get_client_io
        # _lock():`` exactly as before — Semaphore implements the same
        # async-context interface so existing call sites Just Work.
        if self._client_io_lock is None:
            self._client_io_lock = asyncio.Semaphore(self._client_io_lock_concurrency)
        return self._client_io_lock

    def _get_client_balance_io_lock(self) -> asyncio.Lock:
        if self._client_balance_io_lock is None:
            self._client_balance_io_lock = asyncio.Lock()
        return self._client_balance_io_lock

    def _get_persist_lock(self) -> asyncio.Lock:
        # Back-compat alias; runtime-state callers pick up this lock.
        # ``_persist_orders`` and ``_persist_positions`` use their own
        # dedicated locks now (no cross-table serialization).
        if self._persist_lock is None:
            self._persist_lock = asyncio.Lock()
        return self._persist_lock

    def _get_orders_persist_lock(self) -> asyncio.Lock:
        if self._orders_persist_lock is None:
            self._orders_persist_lock = asyncio.Lock()
        return self._orders_persist_lock

    def _get_positions_persist_lock(self) -> asyncio.Lock:
        if self._positions_persist_lock is None:
            self._positions_persist_lock = asyncio.Lock()
        return self._positions_persist_lock

    async def _run_client_io(
        self,
        func: Any,
        *args: Any,
        timeout: float | None = _CLIENT_IO_TIMEOUT_SECONDS,
        lock: str = "order",
    ) -> Any:
        # ``lock="order"`` is the default — protects create/post/cancel
        # order pairings.  ``lock="balance"`` uses the dedicated balance/
        # read-only lock so concurrent get-balance / update-allowance
        # calls don't queue behind an in-flight order submission (and
        # vice versa).  The two SDK call families don't share mutable
        # client state on the py-clob-client-v2 side; only the in-flight
        # serialization expectation differs.
        if lock == "balance":
            target_lock = self._get_client_balance_io_lock()
        else:
            target_lock = self._get_client_io_lock()
        async with target_lock:
            task = asyncio.to_thread(func, *args)
            if timeout is None:
                return await task
            return await asyncio.wait_for(task, timeout=float(max(0.1, timeout)))

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
        # Safety cap: drop oldest entries to prevent unbounded memory growth
        # if reconciliations accumulate faster than they're resolved.
        _MAX_PENDING_RECONCILIATIONS = 500
        if len(self._pending_reconciliations) > _MAX_PENDING_RECONCILIATIONS:
            dropped = self._pending_reconciliations[: len(self._pending_reconciliations) - _MAX_PENDING_RECONCILIATIONS]
            self._pending_reconciliations = self._pending_reconciliations[-_MAX_PENDING_RECONCILIATIONS:]
            logger.warning(
                "Dropped %d oldest pending reconciliations (cap=%d)",
                len(dropped),
                _MAX_PENDING_RECONCILIATIONS,
            )
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

        # py-clob-client-v2 stores signature_type and funder on the OrderBuilder,
        # not on the ClobClient itself. The constructor wires the builder, so we
        # only need to mutate the builder when the signature type changes after init.
        builder = getattr(self._client, "builder", None)
        if builder is not None and getattr(builder, "signature_type", None) != signature_type:
            try:
                builder.signature_type = signature_type
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
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_key,
                signature_type=int(signature_type),
            )
        except Exception:
            return None

        if refresh:
            try:
                await self._run_client_io(self._client.update_balance_allowance, params, lock="balance")
            except Exception as exc:
                logger.debug(
                    "Conditional balance-allowance refresh failed",
                    token_id=token_key,
                    signature_type=signature_type,
                    exc_info=exc,
                )

        try:
            payload = await self._run_client_io(self._client.get_balance_allowance, params, lock="balance")
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

        # 2026-05-10: TTL cache.  The expensive part of this method is
        # the loop below — N sequential venue refresh-fetches to pick
        # the signature type with the largest conditional balance.
        # That property only changes when WE buy or sell this token,
        # so within a few-second window the answer is stable.  On a
        # cache hit we still re-apply the signature type to the client
        # (cheap attribute mutation on the OrderBuilder) so callers
        # observing the client's signature_type read consistent state.
        cached = self._signature_type_cache.get(token_key)
        if cached is not None:
            cached_at, cached_signature_type = cached
            if (_time.monotonic() - cached_at) < self._signature_type_cache_ttl_seconds:
                if self._signature_type_supported(cached_signature_type):
                    self._balance_signature_type = cached_signature_type
                    self._apply_signature_type_to_client(cached_signature_type)
                    return cached_signature_type
                # Supported set changed since the cache was stamped;
                # fall through and re-probe.
                self._signature_type_cache.pop(token_key, None)

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
        # Stamp the cache with the freshly selected signature type so
        # subsequent calls within the TTL window short-circuit the
        # N-fetch probe.
        self._signature_type_cache[token_key] = (_time.monotonic(), selected_signature_type)
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
            if builder is not None and isinstance(getattr(builder, "signature_type", None), int):
                signature_type = int(builder.signature_type)

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
            from py_clob_client_v2.config import get_contract_config

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
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

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
                    await self._run_client_io(self._client.update_balance_allowance, params, lock="balance")
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
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_key,
                signature_type=signature_type,
            )
            await self._run_client_io(self._client.update_balance_allowance, params, lock="balance")
            return True
        except Exception as exc:
            logger.warning(
                "Conditional balance-allowance refresh failed",
                token_id=token_key,
                signature_type=signature_type,
                exc_info=exc,
            )
            return False

    @staticmethod
    def _parse_balance_rejection(error_text: str) -> Optional[dict[str, float]]:
        """Extract structured numbers from Polymarket's balance rejection.

        The venue formats it as::

            balance: 156452220, sum of active orders: 122020000,
            sum of matched orders: 0, order amount (inc. fees): 61010000

        All values are USDC base-units (× 10^6).  Returns None when
        the format doesn't match — keeps the caller decoupled from
        venue copy changes.
        """
        if not error_text:
            return None
        text = error_text.lower()
        try:
            import re

            m = re.search(
                r"balance:\s*(\d+).*?sum of active orders:\s*(\d+).*?"
                r"sum of matched orders:\s*(\d+).*?order amount.*?:\s*(\d+)",
                text,
                flags=re.DOTALL,
            )
            if not m:
                return None
            return {
                "balance_usdc": float(m.group(1)) / 1_000_000.0,
                "active_orders_usdc": float(m.group(2)) / 1_000_000.0,
                "matched_orders_usdc": float(m.group(3)) / 1_000_000.0,
                "order_amount_usdc": float(m.group(4)) / 1_000_000.0,
            }
        except Exception:
            return None

    def _emit_balance_rejection_diagnostic(
        self, *, side_text: str, token_id: str, error_text: str
    ) -> None:
        """Surface the venue's balance rejection as an interpretive warning.

        The raw SDK error tells you the numbers; this line tells the
        operator what they mean and points to the manual remediation
        paths.  No automated mutations — placing/cancelling orders is
        a user action and the system surfaces the diagnostic for the
        operator to decide.
        """
        parsed = self._parse_balance_rejection(error_text)
        if not parsed:
            return
        free = parsed["balance_usdc"] - parsed["active_orders_usdc"]
        shortfall = parsed["order_amount_usdc"] - free
        active_pct = (
            (parsed["active_orders_usdc"] / parsed["balance_usdc"]) * 100.0
            if parsed["balance_usdc"] > 0
            else 0.0
        )
        # Three remediation paths the operator can take, in order of
        # increasing scope.  We do NOT pick one for them — auto-cancel
        # of user orders to make a SELL go through is not institutional-
        # grade behavior, even with safety heuristics.
        logger.warning(
            "Balance rejection diagnosed: %s blocked by collateral reservation. "
            "venue_balance=$%.2f active_orders=$%.2f (%.1f%% of balance) "
            "order_needs=$%.2f free_margin=$%.2f shortfall=$%.2f. "
            "Remediation: (a) wait for periodic stale-order sweep "
            "(STALE_ORDER_AGE_HOURS=%.1fh, every "
            "%.0fs); (b) operator cancel specific open orders via "
            "POST /api/operator/orders/{order_id}/cancel; "
            "(c) deposit additional USDC to the funder wallet. "
            "Note: the per-signature_type balance snapshot follows; "
            "compare ``balance snapshot`` rows against ``venue_balance`` "
            "to confirm whether the venue's view matches the wallet "
            "the SDK is signing under.",
            side_text,
            parsed["balance_usdc"],
            parsed["active_orders_usdc"],
            active_pct,
            parsed["order_amount_usdc"],
            free,
            max(0.0, shortfall),
            float(getattr(settings, "STALE_ORDER_AGE_HOURS", 2.0)),
            60.0,  # _STALE_OPEN_ORDER_SWEEP_INTERVAL_SECONDS — kept inline to avoid worker import
        )

    async def _log_balance_snapshot_per_signature_type(self, *, context: str) -> None:
        """Log the venue's balance + active-order view for every supported
        signature_type so an operator can see *which wallet* the SDK was
        looking at when it threw "not enough balance / allowance".

        The error message itself only reports one signature_type's slice
        of the truth (the one the SDK currently has selected) — funds in
        a different funder wallet are invisible to that slice.  Without
        this snapshot the operator has to run an out-of-band script to
        find which funder actually has free cash; with it, every
        balance rejection in the worker logs is self-explanatory.

        Read-only.  All exceptions are swallowed: this is best-effort
        diagnostic output and must not break the calling order path.
        """
        try:
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
        except Exception:
            logger.warning(
                "balance snapshot unavailable: could not import BalanceAllowanceParams",
                context=context,
            )
            return

        if not self.is_ready():
            logger.warning(
                "balance snapshot unavailable: trading client not ready",
                context=context,
            )
            return

        prior = self._balance_signature_type
        # Capture the SDK's CURRENTLY-active signature_type + funder
        # (i.e. the wallet the failing order was signed under) so we
        # can mark it in each row.  Knowing which row was the active
        # one is essential — sig 1 having $236 doesn't help if the
        # SDK signed under sig 2 with $0.
        active_sig: int | None = None
        active_funder: str = ""
        try:
            builder = getattr(self._client, "builder", None)
            if builder is not None and isinstance(getattr(builder, "signature_type", None), int):
                active_sig = int(builder.signature_type)
                active_funder = str(getattr(builder, "funder", "") or "")
        except Exception:
            pass
        if active_sig is None and isinstance(prior, int):
            active_sig = int(prior)
            active_funder = str(self._funder_for_signature_type(active_sig) or "")
        logger.warning(
            "balance snapshot active signature_type at order submission",
            context=context,
            signature_type=active_sig,
            funder=active_funder or None,
        )

        try:
            for sig in POLYMARKET_SIGNATURE_TYPES:
                if not self._signature_type_supported(int(sig)):
                    logger.warning(
                        "balance snapshot",
                        context=context,
                        signature_type=int(sig),
                        supported=False,
                        funder=None,
                        active=(active_sig == int(sig)),
                    )
                    continue
                funder = self._funder_for_signature_type(int(sig))
                # Re-point the client so ``get_balance_allowance`` reads
                # this signature_type's wallet.  Restored in the finally.
                try:
                    self._balance_signature_type = int(sig)
                    self._apply_signature_type_to_client(int(sig))
                except Exception:
                    pass

                params = BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=int(sig),
                )

                balance_str: str | None = None
                allowance_str: str | None = None
                err: str | None = None

                try:
                    await self._run_client_io(
                        self._client.update_balance_allowance, params, lock="balance"
                    )
                except Exception as exc:
                    err = f"refresh_failed:{type(exc).__name__}:{str(exc)[:80]}"

                try:
                    payload = await self._run_client_io(
                        self._client.get_balance_allowance, params, lock="balance"
                    )
                except Exception as exc:
                    err = (err + " | " if err else "") + (
                        f"fetch_failed:{type(exc).__name__}:{str(exc)[:80]}"
                    )
                    payload = None

                if isinstance(payload, dict):
                    assume_base_units = isinstance(payload.get("allowances"), dict)
                    bal = _parse_collateral_amount(
                        payload.get("balance"),
                        assume_base_units=assume_base_units,
                    )
                    alw = _parse_collateral_amount(
                        payload.get("allowance"),
                        assume_base_units=assume_base_units,
                    )
                    if isinstance(payload.get("allowances"), dict):
                        for raw in payload["allowances"].values():
                            parsed = _parse_collateral_amount(
                                raw, assume_base_units=True
                            )
                            if parsed is not None and (
                                alw is None or parsed > alw
                            ):
                                alw = parsed
                    if bal is not None:
                        balance_str = f"${float(bal):.2f}"
                    if alw is not None:
                        allowance_str = f"${float(alw):.2f}"

                logger.warning(
                    "balance snapshot",
                    context=context,
                    signature_type=int(sig),
                    supported=True,
                    funder=funder,
                    balance_usdc=balance_str,
                    allowance_usdc=allowance_str,
                    error=err,
                    active=(active_sig == int(sig)),
                )
        except Exception as exc:
            logger.warning(
                "balance snapshot dump failed",
                context=context,
                exc_info=exc,
            )
        finally:
            try:
                self._balance_signature_type = prior
                self._apply_signature_type_to_client(prior)
            except Exception:
                pass

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
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=signature_type,
            )
            await self._run_client_io(self._client.update_balance_allowance, params, lock="balance")
            self._invalidate_balance_cache()
            return True
        except Exception as exc:
            logger.warning(
                "Collateral balance-allowance refresh failed",
                signature_type=signature_type,
                exc_info=exc,
            )
            return False

    async def prepare_sell_balance_allowance(
        self,
        token_id: str,
        *,
        force_refresh: bool = False,
    ) -> bool:
        """Pre-flight refresh for a SELL submission.

        Refreshes BOTH the conditional balance/allowance (the venue
        needs to know we hold the tokens we're trying to sell) AND
        the collateral balance/allowance (the venue charges a USDC
        fee on SELLs and rejects the order if its cached view of our
        USDC balance is below the fee — even if the actual on-chain
        balance is fine).

        The 2026-04-28 cascade was the collateral half of this:
        Polymarket's CLOB cached our maker wallet at $8.33 USDC with
        $8.32 reserved by old orders, leaving $0.01 free for fees.
        Every SELL retry hit "not enough balance / allowance" because
        we only refreshed the conditional side; the venue's stale
        collateral view never got nudged.  Per-signature_type wallet
        snapshot showed the actual on-chain balance was $236 in the
        proxy.

        Returns True if EITHER refresh succeeded — failure to refresh
        is non-fatal, the SDK has its own on-the-fly approval
        fall-through.

        TTL cache: if a successful refresh ran in the last
        ``_sell_allowance_ttl_seconds`` (default 5 s) for this token,
        skip the 4-5 s of HTTP roundtrips and trust the cached venue-
        side view.  ITER-2 traces showed sells inheriting up to 8 s
        of pre-submit overhead from this function's serial refreshes;
        for sell-frequent strategies (exit cascades right after a
        fill) the second-and-later sells gain nothing from re-asking
        the venue what its balance view is — the venue has not had
        time to forget.  Pass ``force_refresh=True`` to bypass.
        """
        token_key = str(token_id or "").strip()
        if not force_refresh and token_key:
            last_at = self._sell_allowance_last_at.get(token_key, 0.0)
            if last_at > 0.0 and (_time.monotonic() - last_at) < self._sell_allowance_ttl_seconds:
                # Cache hit — recent refresh is still good enough.
                return True
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
        # Also force-refresh the collateral cache.  This is what fixes
        # "balance: $8.33, sum of active orders: $8.32" rejections on
        # SELL retries — without it the venue's stale USDC view sits
        # at the same value across every retry, so the price walk-down
        # and conditional refresh both run against an unfixable error.
        collateral_refreshed = False
        try:
            collateral_refreshed = await self.refresh_collateral_balance_allowance()
        except Exception as exc:
            logger.debug(
                "Sell allowance preparation could not refresh collateral",
                token_id=token_key,
                exc_info=exc,
            )
        # Stamp the TTL gate only when the work that hit the wire
        # actually succeeded — a failed refresh shouldn't suppress the
        # next attempt.
        if (conditional_refreshed or collateral_refreshed) and token_key:
            self._sell_allowance_last_at[token_key] = _time.monotonic()
        return conditional_refreshed or collateral_refreshed

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
            self._invalidate_balance_cache()
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

    async def _bootstrap_wallet_state_cache(self, wallet_address: Optional[str]) -> None:
        """Synchronous WalletStateCache REST seed at init time.

        ARCHITECTURAL CONTRACT: when ``initialize()`` returns ``True`` on
        the trading plane, every dependency the orchestrator hot path
        reads MUST already be populated.  That includes the cache's
        wallet positions snapshot — without it, the orchestrator's
        freshness gate refuses every cycle until the reconciliation
        worker's first 30s tick.

        This method blocks the init path on the REST fetch (with a
        bounded timeout) so success here means the cache is ready to
        serve hot-path reads.  Failure here is also a valid outcome —
        the cache records a failed-seed state, and the freshness gate
        keeps refusing trades until the reconciliation worker eventually
        succeeds.  Either way, no trading happens on stale state.
        """
        if not wallet_address:
            return
        wallet_lower = str(wallet_address).strip().lower()
        if not wallet_lower:
            return
        try:
            from services.polymarket import polymarket_client
            from services.wallet_state_cache import get_wallet_state_cache
        except Exception as imp_exc:
            logger.warning(
                "WalletStateCache bootstrap import failed (non-fatal)",
                exc_info=imp_exc,
            )
            return
        cache = get_wallet_state_cache()
        # Open positions: REST fetch with a tight 15s budget — cache
        # seed is on the critical bootstrap path, can't dawdle.
        bootstrap_timeout_s = 15.0
        try:
            open_positions = await asyncio.wait_for(
                polymarket_client.get_wallet_positions(wallet_lower),
                timeout=bootstrap_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WalletStateCache bootstrap open-positions REST timed out after %.0fs; "
                "cache will not be fresh until reconciliation worker reseeds",
                bootstrap_timeout_s,
                extra={"wallet": wallet_lower},
            )
            cache.seed_from_rest(
                wallet_address=wallet_lower,
                positions=[],
                closed_positions=[],
                succeeded=False,
            )
            return
        except Exception as exc:
            logger.warning(
                "WalletStateCache bootstrap open-positions REST failed",
                wallet=wallet_lower,
                exc_info=exc,
            )
            cache.seed_from_rest(
                wallet_address=wallet_lower,
                positions=[],
                closed_positions=[],
                succeeded=False,
            )
            return
        if not isinstance(open_positions, list):
            open_positions = []
        # Closed positions are larger and less time-critical — fetch
        # with a generous budget but don't fail bootstrap if they're
        # slow.  An empty closed list at boot is acceptable; the
        # reconciliation worker's slower cadence (every 5 min) will
        # populate it later.
        closed_positions: list[dict[str, Any]] = []
        try:
            closed_positions_raw = await asyncio.wait_for(
                polymarket_client.get_closed_positions_paginated(
                    wallet_lower,
                    max_positions=2000,
                ),
                timeout=15.0,
            )
            if isinstance(closed_positions_raw, list):
                closed_positions = closed_positions_raw
        except Exception as exc:
            logger.debug(
                "WalletStateCache bootstrap closed-positions fetch failed (non-fatal)",
                wallet=wallet_lower,
                exc_info=exc,
            )
        result = cache.seed_from_rest(
            wallet_address=wallet_lower,
            positions=open_positions,
            closed_positions=closed_positions,
            succeeded=True,
        )
        # Replay user-channel WS subscription set with whatever
        # condition_ids the REST seed populated.  Keeps the option (a)
        # contract: subscribe to every market the wallet has held.
        try:
            from services.ws_feeds import get_feed_manager

            condition_ids = cache.iter_tracked_condition_ids()
            if condition_ids:
                await get_feed_manager().ensure_user_subscribed(condition_ids)
        except Exception as sub_exc:
            logger.debug(
                "Bootstrap user-channel WS subscription refresh failed (non-fatal)",
                exc_info=sub_exc,
            )
        logger.info(
            "WalletStateCache bootstrap complete",
            wallet=wallet_lower,
            open_positions=result.get("open_seeded", 0),
            closed_positions=result.get("closed_seeded", 0),
        )

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
                # Import py-clob-client-v2
                from py_clob_client_v2.client import ClobClient
                from py_clob_client_v2.clob_types import ApiCreds, BuilderConfig
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

                builder_config = None
                builder_code = (getattr(settings, "POLYMARKET_BUILDER_CODE", None) or "").strip() or None
                if builder_code:
                    builder_config = BuilderConfig(builder_code=builder_code)

                self._client = ClobClient(
                    host=settings.CLOB_API_URL,
                    key=private_key,
                    chain_id=settings.CHAIN_ID,
                    creds=creds,
                    signature_type=sig_type,
                    funder=funder,
                    builder_config=builder_config,
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
                    logger.warning("Trading HTTP transport patch failed; using py-clob-client-v2 default transport")

                await self._restore_runtime_state()
                # Apply restored sig_type to builder immediately so that even if
                # the get_balance() probe below fails, orders are signed correctly.
                self._apply_signature_type_to_client(self._balance_signature_type)
                if isinstance(self._balance_signature_type, int):
                    logger.info(
                        "Restored signature type=%s from runtime state",
                        self._balance_signature_type,
                    )
                await self.sync_positions()
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
                # Wire the wallet's API credentials into the user-channel
                # WS feed so it can subscribe to the wallet's own
                # order/trade events — but ONLY in the trading worker
                # plane.  Polymarket's user channel rejects a second
                # concurrent connection from the same wallet, so we
                # must not start it in both the API process and the
                # trading plane.  ``HOMERUN_WORKER_PLANE=trading`` is
                # set by ``workers.host.main`` on the trading plane;
                # the API process leaves it unset and skips this.
                worker_plane = os.environ.get("HOMERUN_WORKER_PLANE", "")
                if worker_plane == "trading":
                    user_feed_wallet = (
                        self._proxy_funder_address
                        or self._wallet_address
                        or eoa_address
                    )
                    # ARCHITECTURAL CONTRACT: when ``initialize()`` returns
                    # ``True`` on the trading plane, the orchestrator may
                    # immediately begin trading.  That requires:
                    #   (1) Polymarket user-channel WS credentials wired
                    #   (2) WalletStateCache seeded from REST (open
                    #       positions + closed positions snapshot)
                    # Both must complete BEFORE initialize() returns,
                    # otherwise the freshness gate refuses every cycle
                    # until the reconciliation worker's first 30s tick
                    # — that's seconds of forced idle on every restart,
                    # and a flood of "stale" warnings from the fast
                    # trader.  We do them synchronously here.
                    try:
                        from services.ws_feeds import get_feed_manager

                        user_feed_configured = get_feed_manager().configure_user_feed_credentials(
                            api_key=api_key,
                            api_secret=api_secret,
                            api_passphrase=api_passphrase,
                            wallet_address=user_feed_wallet,
                        )
                        if user_feed_configured:
                            logger.info(
                                "Polymarket user-channel WS credentials configured",
                                wallet=user_feed_wallet,
                                plane=worker_plane,
                            )
                    except Exception as user_feed_exc:
                        logger.warning(
                            "Failed to configure Polymarket user-channel WS credentials (non-fatal)",
                            exc_info=user_feed_exc,
                        )

                    # Synchronous WalletStateCache bootstrap.  Block until
                    # we have a fresh REST snapshot or a clear failure
                    # signal — both are valid outcomes that establish the
                    # cache's freshness state.
                    await self._bootstrap_wallet_state_cache(user_feed_wallet)
                else:
                    logger.debug(
                        "Skipping Polymarket user-channel WS credential wiring "
                        "(only the trading worker plane runs the feed)",
                        process_role=os.environ.get("HOMERUN_PROCESS_ROLE", ""),
                        plane=worker_plane,
                    )
                # ITER-? (Fix HH): SYNCHRONOUS prewarm.  Fire-and-forget
                # prewarm let the FIRST order race ahead of the cache
                # population — measured cold-start ``create_order`` of
                # 968 ms and ``post_order`` of 1625 ms, vs the warm-path
                # 78 ms / 610 ms baseline.  Block ``initialize`` until
                # every lazy SDK path is warm so no order can be cold.
                #
                # Order matters: spin the executor BEFORE we submit
                # tasks to it, run ``get_version`` on the same executor
                # so we exercise the to_thread dispatch path the hot-
                # path will use, and finally fire one ``get_ok`` round-
                # trip through the proxy so the SOCKS5+TLS+H2 connection
                # is fully established (the 1050 ms cold-vs-148 ms warm
                # gap we measured against the proxy directly).
                try:
                    executor = self._get_clob_executor()
                    pool_size = self._client_io_lock_concurrency + 4
                    spinup_futures = [executor.submit(lambda: None) for _ in range(pool_size)]
                    # Wait for every worker thread to actually exist
                    # (Python lazy-spawns on submit; futures.result()
                    # returns once the thread has picked up the noop).
                    for fut in spinup_futures:
                        try:
                            fut.result(timeout=1.0)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.debug("CLOB executor prewarm failed (non-fatal)", exc_info=exc)

                # Prime ``__cached_version`` and the SOCKS5+TLS+H2
                # connection in ONE call: ``get_ok`` is unauthenticated
                # so it's the cheapest endpoint that establishes the
                # full network path; ``get_version`` populates the
                # SDK's lazy version cache.  Both run on the dedicated
                # executor so the pool's threads are also exercised.
                # ``await asyncio.wait_for`` blocks until done — that's
                # the whole point.  Failures here just log; the SDK
                # falls back to lazy init at first order, which is the
                # pre-Fix-HH behaviour and acceptable.
                loop = asyncio.get_running_loop()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(self._get_clob_executor(), self._client.get_ok),
                        timeout=5.0,
                    )
                except Exception as exc:
                    logger.debug("CLOB connection warmup failed (non-fatal)", exc_info=exc)
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(self._get_clob_executor(), self._client.get_version),
                        timeout=5.0,
                    )
                except Exception as exc:
                    logger.debug("CLOB version prewarm failed (non-fatal)", exc_info=exc)

                logger.info(
                    "Trading service initialized successfully (prewarm complete)",
                    credential_source=credential_source,
                )
                self._last_init_error = None
                self._init_retry_not_before = None
                # Now the connection is warm we can start the keepalive
                # loop — the loop's first ping will renew the warm
                # connection, not establish a cold one.
                self._start_clob_keepalive_loop()
                return True

            except ImportError:
                now = utcnow()
                if (
                    self._last_missing_dependency_log_at is None
                    or (now - self._last_missing_dependency_log_at).total_seconds() >= MISSING_DEPENDENCY_RELOG_SECONDS
                ):
                    logger.error("py-clob-client-v2 not installed. Run: pip install py-clob-client-v2")
                    self._last_missing_dependency_log_at = now
                self._last_init_error = "py-clob-client-v2 not installed"
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

    def _start_clob_keepalive_loop(self) -> None:
        # Trading-plane only.  The API process never submits orders, so
        # paying for a periodic CLOB GET there is pure waste — and would
        # double the /ok request rate on no benefit.
        if os.environ.get("HOMERUN_WORKER_PLANE", "") != "trading":
            return
        existing = self._clob_keepalive_task
        if existing is not None and not existing.done():
            return
        task = self._start_background_task(
            self._run_clob_keepalive_loop(),
            name="clob_keepalive_loop",
        )
        # Store the handle so a second ``initialize()`` (e.g. credential
        # refresh path) doesn't spawn a duplicate keepalive loop —
        # without this assignment the guard above always falls through.
        self._clob_keepalive_task = task

    async def _run_clob_keepalive_loop(self) -> None:
        # Cadence < httpx default ``keepalive_expiry`` (5 s).  3 s gives
        # a 2 s safety margin against jitter and lets one stretched-out
        # tick still land before the connection is reaped.
        TICK_SECONDS = 3.0
        # Stagger the first tick so the very first order after init
        # also benefits from a primed connection — without blocking
        # initialize() on the round-trip.
        await asyncio.sleep(0.0)
        ping_counter = 0
        while True:
            try:
                client = self._client
                if client is None or not self._initialized:
                    await asyncio.sleep(TICK_SECONDS)
                    continue
                t0 = _time.monotonic()
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(client.get_ok),
                        timeout=2.0,
                    )
                    elapsed_ms = (_time.monotonic() - t0) * 1000.0
                    self._clob_keepalive_last_ok_at = _time.monotonic()
                    self._clob_keepalive_recent_ms.append(elapsed_ms)
                    ping_counter += 1
                    # Periodic summary so the operator (and the
                    # crypto_latency_harness) can see the warm-
                    # connection baseline that ``post_order`` is being
                    # measured against.  Logged at INFO so the harness
                    # can pick it up alongside ``crypto_latency_trace``.
                    if (
                        ping_counter % self._clob_keepalive_log_every == 0
                        and self._clob_keepalive_recent_ms
                    ):
                        samples = sorted(self._clob_keepalive_recent_ms)
                        n = len(samples)
                        p50 = samples[n // 2]
                        p90 = samples[min(n - 1, max(0, int(round(0.9 * (n - 1)))))]
                        logger.info(
                            "clob_keepalive_rtt",
                            samples=n,
                            min_ms=round(samples[0], 1),
                            p50_ms=round(p50, 1),
                            p90_ms=round(p90, 1),
                            max_ms=round(samples[-1], 1),
                            last_ms=round(elapsed_ms, 1),
                        )
                except asyncio.TimeoutError:
                    self._clob_keepalive_last_fail_at = _time.monotonic()
                    logger.debug(
                        "CLOB keepalive ping timed out",
                        elapsed_ms=round((_time.monotonic() - t0) * 1000.0, 1),
                    )
                except Exception as exc:
                    self._clob_keepalive_last_fail_at = _time.monotonic()
                    logger.debug(
                        "CLOB keepalive ping failed",
                        elapsed_ms=round((_time.monotonic() - t0) * 1000.0, 1),
                        exc=str(exc),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as outer_exc:
                logger.warning(
                    "CLOB keepalive loop iteration raised", exc_info=outer_exc
                )
            await asyncio.sleep(TICK_SECONDS)

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
        if builder is not None and isinstance(getattr(builder, "signature_type", None), int):
            return int(builder.signature_type)
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

        from models.database import AsyncSessionLocal, LiveTradingOrder, TradeSignal
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        unique_orders: dict[str, Order] = {}
        for order in orders:
            unique_orders[str(order.id)] = order
        order_ids = list(unique_orders)
        if not order_ids:
            return

        # Per-stage breakdown for slow-log diagnosis.  Cycle 2 of the
        # perf-harness loop showed ``persist_lock_wait=884ms`` avg
        # and ``select_existing=1262ms`` avg dominating the cost.
        # The fix here:
        #  - Drop ``_orders_persist_lock`` entirely.  Each call writes
        #    distinct order_ids (UUIDs from the calling submission);
        #    same-id concurrent writes are race-safe via
        #    ``ON CONFLICT DO UPDATE``.  Concurrent persists can now
        #    parallelize through the asyncpg pool (200 max conns).
        #  - Replace the SELECT-then-UPDATE pattern with a single
        #    ``INSERT ... ON CONFLICT DO UPDATE``.  Saves one network
        #    round-trip per call (the ``select_existing`` SELECT)
        #    and halves the DB-side parse + planner work.
        _persist_started = _time.monotonic()
        _persist_breakdown: dict[str, float] = {"orders": len(order_ids)}

        def _persist_record(stage: str, started_mono: float) -> None:
            elapsed_ms = (_time.monotonic() - started_mono) * 1000.0
            _persist_breakdown[stage] = round(
                _persist_breakdown.get(stage, 0.0) + elapsed_ms, 1
            )

        for attempt in range(_DB_RETRY_ATTEMPTS):
            needs_retry = False
            _persist_breakdown["attempts"] = float(attempt + 1)
            _stage_started = _time.monotonic()
            async with AsyncSessionLocal() as session:
                _persist_record("session_checkout", _stage_started)
                try:
                    await session.execute(text("SET LOCAL lock_timeout = '1500ms'"))
                    # Cycle 9 of the perf-harness loop showed ``select
                    # _signals`` averaging 1.9s/call under DB pool
                    # contention (30 slow events, 56s cumulative).
                    # The SELECT was previously eager: every persist
                    # call ran it whether or not the orders actually
                    # needed market_id resolution.  In practice the
                    # call site populates ``order.market_id`` BEFORE
                    # persist (or the position cache fills it from
                    # ``self._positions``), so the SELECT is a pure
                    # defensive fallback for newly-placed orders that
                    # haven't been linked to a position yet.
                    #
                    # Restructure to:
                    #   1. First pass — check ``order.market_id`` and
                    #      the position cache (no DB I/O).
                    #   2. Collect ONLY the opportunity_ids that still
                    #      lack a market_id.
                    #   3. Run ``select_signals`` ONLY if step 2 left
                    #      anything unresolved.
                    #   4. Second pass — finalize using the resolved
                    #      map and build the UPSERT values list.
                    #
                    # The 99% case has zero unresolved orders → the
                    # SELECT is skipped entirely.

                    # First pass: try in-memory resolution (no DB I/O).
                    pending_resolution: list[tuple[Order, str, str | None, str | None]] = []
                    needs_signal_lookup: set[str] = set()
                    for order in unique_orders.values():
                        token_key = str(order.token_id or "").strip()
                        authoritative_market_question = order.market_question
                        authoritative_market_id = str(getattr(order, "market_id", "") or "").strip() or None
                        if token_key:
                            position = self._positions.get(token_key)
                            if position is not None:
                                position_market_id = str(position.market_id or "").strip() or None
                                if position_market_id:
                                    authoritative_market_id = position_market_id
                                    order.market_id = position_market_id
                                position_market_question = str(position.market_question or "").strip()
                                if position_market_question:
                                    authoritative_market_question = position_market_question
                                    order.market_question = position_market_question
                        if authoritative_market_id is None:
                            opportunity_key = str(order.opportunity_id or "").strip()
                            if opportunity_key:
                                needs_signal_lookup.add(opportunity_key)
                        pending_resolution.append((
                            order,
                            token_key,
                            authoritative_market_id,
                            authoritative_market_question,
                        ))

                    # Step 3: only fetch signals that we couldn't resolve
                    # locally.  Skipped entirely in the common case.
                    market_ids_by_signal_id: dict[str, str] = {}
                    if needs_signal_lookup:
                        _stage_started = _time.monotonic()
                        signal_result = await session.execute(
                            select(TradeSignal.id, TradeSignal.market_id).where(
                                TradeSignal.id.in_(sorted(needs_signal_lookup))
                            )
                        )
                        market_ids_by_signal_id = {
                            str(signal_id): str(market_id)
                            for signal_id, market_id in signal_result.all()
                            if str(signal_id or "").strip() and str(market_id or "").strip()
                        }
                        _persist_record("select_signals", _stage_started)

                    # Second pass: finalize and build UPSERT values.
                    values_list: list[dict[str, Any]] = []
                    for order, token_key, authoritative_market_id, authoritative_market_question in pending_resolution:
                        if authoritative_market_id is None:
                            resolved_market_id = (
                                market_ids_by_signal_id.get(str(order.opportunity_id or "").strip()) or None
                            )
                            if resolved_market_id:
                                authoritative_market_id = resolved_market_id
                                order.market_id = resolved_market_id
                        created_at = _normalize_utc_datetime(order.created_at) or utcnow()
                        updated_at = _normalize_utc_datetime(order.updated_at) or utcnow()
                        values_list.append({
                            "id": order.id,
                            "wallet_address": wallet,
                            "market_id": authoritative_market_id,
                            "clob_order_id": str(order.clob_order_id or "").strip() or None,
                            "token_id": token_key,
                            "side": order.side.value,
                            "price": float(order.price),
                            "size": float(order.size),
                            "order_type": order.order_type.value,
                            "status": order.status.value,
                            "filled_size": float(order.filled_size),
                            "average_fill_price": float(order.average_fill_price),
                            "market_question": authoritative_market_question,
                            "opportunity_id": order.opportunity_id,
                            "error_message": order.error_message,
                            "created_at": created_at,
                            "updated_at": updated_at,
                        })

                    # Single UPSERT — the ``DO UPDATE SET`` covers
                    # every column except ``id`` and ``created_at``
                    # (insert-only).  ``excluded`` is the candidate
                    # row's values.
                    _stage_started = _time.monotonic()
                    stmt = pg_insert(LiveTradingOrder).values(values_list)
                    update_cols = {
                        col: stmt.excluded[col]
                        for col in (
                            "wallet_address", "market_id", "clob_order_id", "token_id",
                            "side", "price", "size", "order_type", "status",
                            "filled_size", "average_fill_price", "market_question",
                            "opportunity_id", "error_message", "updated_at",
                        )
                    }
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["id"], set_=update_cols
                    )
                    await session.execute(stmt)
                    _persist_record("upsert", _stage_started)
                    _stage_started = _time.monotonic()
                    await session.commit()
                    _persist_record("commit", _stage_started)
                    _persist_breakdown["total_ms"] = round(
                        (_time.monotonic() - _persist_started) * 1000.0, 1
                    )
                    if _persist_breakdown["total_ms"] >= 2000.0:
                        try:
                            logger.warning(
                                "_persist_orders slow",
                                breakdown=_persist_breakdown,
                            )
                        except Exception:
                            pass
                    return
                except (DBAPIError, OperationalError, InterfaceError) as exc:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
                    if not _is_retryable_db_error(exc) or is_last:
                        logger.error("Failed to persist live trading orders", exc_info=exc)
                        return
                    needs_retry = True
                except Exception as exc:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    logger.error("Failed to persist live trading orders", exc_info=exc)
                    return
            if needs_retry:
                await asyncio.sleep(_db_retry_delay(attempt))

    async def _persist_positions(self) -> None:
        wallet = self._wallet_for_persistence()
        if not wallet:
            return

        from models.database import AsyncSessionLocal, LiveTradingPosition

        positions = list(self._positions.values())
        # Positions-specific lock; doesn't block order or runtime
        # state writes.  See split rationale on ``_persist_orders``.
        persist_lock = self._get_positions_persist_lock()
        async with persist_lock:
            for attempt in range(_DB_RETRY_ATTEMPTS):
                needs_retry = False
                async with AsyncSessionLocal() as session:
                    try:
                        wallet_key = str(wallet).strip().lower()
                        persisted_at = utcnow()
                        position_ids: set[str] = set()
                        position_rows: list[dict[str, Any]] = []
                        for position in positions:
                            token_id = str(position.token_id or "").strip()
                            if not token_id:
                                continue
                            position_id = f"{wallet_key}:{token_id}"
                            position_ids.add(position_id)
                            position_rows.append(
                                {
                                    "id": position_id,
                                    "wallet_address": wallet_key,
                                    "token_id": token_id,
                                    "market_id": str(position.market_id or "").strip(),
                                    "market_question": position.market_question,
                                    "outcome": position.outcome,
                                    "size": float(position.size),
                                    "average_cost": float(position.average_cost),
                                    "current_price": float(position.current_price),
                                    "unrealized_pnl": float(position.unrealized_pnl),
                                    "redeemable": bool(position.redeemable),
                                    "counts_as_open": bool(position.counts_as_open),
                                    "end_date": str(position.end_date or "").strip() or None,
                                    "created_at": _normalize_utc_datetime(position.created_at) or persisted_at,
                                    "updated_at": persisted_at,
                                }
                            )
                        # Wipe-and-replace guard: if the new positions list is
                        # EMPTY but a non-trivial number of rows currently
                        # exist for this wallet, refuse to wipe.  An empty
                        # ``self._positions`` map can be the legitimate
                        # "wallet truly has no positions" signal, but it is
                        # also the indistinguishable shape produced by a
                        # transient Polymarket data-API blip (HTTP 200 with
                        # empty ``data`` array, or pagination short-circuit).
                        # The 2026-04-28 incident saw ~$200 of unrealised gain
                        # written off because that blip wiped the local cache,
                        # which the lifecycle then read as "wallet absent".
                        # Trust the previous DB state until we get a non-empty
                        # confirmation; the next sync will reconcile if the
                        # wallet did genuinely flatten.
                        if not position_rows:
                            existing_count = (
                                await session.execute(
                                    select(func.count())
                                    .select_from(LiveTradingPosition)
                                    .where(
                                        func.lower(
                                            func.coalesce(
                                                LiveTradingPosition.wallet_address, ""
                                            )
                                        )
                                        == wallet_key
                                    )
                                )
                            ).scalar_one() or 0
                            if int(existing_count) > 0:
                                logger.warning(
                                    "Refusing to wipe live_trading_positions on "
                                    "empty-result sync (likely Polymarket data-API blip)",
                                    wallet=wallet_key,
                                    existing_rows=int(existing_count),
                                )
                                await session.commit()
                                return
                        stale_rows_query = delete(LiveTradingPosition).where(
                            func.lower(func.coalesce(LiveTradingPosition.wallet_address, "")) == wallet_key
                        )
                        if position_ids:
                            stale_rows_query = stale_rows_query.where(~LiveTradingPosition.id.in_(list(position_ids)))
                        await session.execute(stale_rows_query)
                        if position_rows:
                            insert_stmt = pg_insert(LiveTradingPosition).values(position_rows)
                            await session.execute(
                                insert_stmt.on_conflict_do_update(
                                    index_elements=[LiveTradingPosition.id],
                                    set_={
                                        "wallet_address": insert_stmt.excluded.wallet_address,
                                        "token_id": insert_stmt.excluded.token_id,
                                        "market_id": insert_stmt.excluded.market_id,
                                        "market_question": insert_stmt.excluded.market_question,
                                        "outcome": insert_stmt.excluded.outcome,
                                        "size": insert_stmt.excluded.size,
                                        "average_cost": insert_stmt.excluded.average_cost,
                                        "current_price": insert_stmt.excluded.current_price,
                                        "unrealized_pnl": insert_stmt.excluded.unrealized_pnl,
                                        "redeemable": insert_stmt.excluded.redeemable,
                                        "counts_as_open": insert_stmt.excluded.counts_as_open,
                                        "end_date": insert_stmt.excluded.end_date,
                                        "updated_at": insert_stmt.excluded.updated_at,
                                    },
                                )
                            )
                        await session.commit()
                        return
                    except (OperationalError, InterfaceError) as exc:
                        await session.rollback()
                        is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
                        if not _is_retryable_db_error(exc) or is_last:
                            logger.error("Failed to persist live trading positions", exc_info=exc)
                            return
                        needs_retry = True
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Failed to persist live trading positions", exc_info=exc)
                        return
                if needs_retry:
                    await asyncio.sleep(_db_retry_delay(attempt))

    _PNL_COUNTERS_PLACEHOLDER: dict[str, Any] = {
        "total_pnl": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "daily_pnl": 0.0,
    }

    async def _derive_pnl_counters_from_orders(
        self, session: Any, wallet: str
    ) -> dict[str, Any]:
        # 300s TTL cache (see ``_PNL_COUNTERS_TTL_SECONDS``).  This used to
        # run the double-aggregate query inline on the caller's
        # ``session`` whenever the cache missed — under DB pool pressure
        # the 1ms query stretched to 2-3s of *event-loop blocking*
        # because the persist's session was waiting for a connection.
        # The 12-hour soak on 2026-05-05 showed ``derive_pnl=2578ms``
        # repeating ~every 5 min as the dominant single contributor to
        # event-loop stalls.
        #
        # Fix RR: never block the persist hot path on this.  PnL
        # counters are informational telemetry written to the
        # runtime_state snapshot row — no decision logic depends on
        # them.  The pattern below:
        #   * Cache fresh → return cached (unchanged)
        #   * Cache stale → return stale immediately, kick off a
        #     background refresh on a fresh session
        #   * Cache cold (process startup) → return zeros, kick off a
        #     background populate
        # The background compute uses its own DB session via
        # ``AsyncSessionLocal`` so it doesn't compete with the caller's
        # session pool slot, and the existing single-flight lock
        # prevents duplicate refreshes when several wallets pipe
        # through the same TTL boundary.
        cache = getattr(self, "_pnl_counters_cache", None)
        if cache is None:
            cache = {}
            self._pnl_counters_cache = cache
        wallet_key = (wallet or "").lower()
        cached = cache.get(wallet_key)
        now_mono = _time.monotonic()
        if cached is not None:
            cached_at, cached_result = cached
            if (now_mono - cached_at) < _PNL_COUNTERS_TTL_SECONDS:
                return cached_result
            # Stale — return immediately, kick off background refresh.
            self._schedule_pnl_counters_refresh(wallet, wallet_key, cache)
            return cached_result
        # Cold cache (first call for this wallet since process start).
        # Schedule a background populate and return placeholder zeros.
        self._schedule_pnl_counters_refresh(wallet, wallet_key, cache)
        return self._PNL_COUNTERS_PLACEHOLDER

    def _schedule_pnl_counters_refresh(
        self,
        wallet: str,
        wallet_key: str,
        cache: dict[str, tuple[float, dict[str, Any]]],
    ) -> None:
        """Fire a background compute of PnL counters on a fresh session.

        Single-flight via the existing per-wallet lock dict.  The lock
        guards both the compute and the cache write, so concurrent
        callers see exactly one DB query per TTL window per wallet.
        """
        locks = getattr(self, "_pnl_counters_locks", None)
        if locks is None:
            locks = {}
            self._pnl_counters_locks = locks
        lock = locks.get(wallet_key)
        if lock is None:
            lock = asyncio.Lock()
            locks[wallet_key] = lock
        # If a refresh is already in flight for this wallet, the lock
        # is held — don't queue another.  The acquire/release pattern
        # below handles that with ``locked()`` so we don't spawn a
        # background task that just blocks on the lock.
        if lock.locked():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._refresh_pnl_counters_background(wallet, wallet_key, cache, lock)
        )

    async def _refresh_pnl_counters_background(
        self,
        wallet: str,
        wallet_key: str,
        cache: dict[str, tuple[float, dict[str, Any]]],
        lock: asyncio.Lock,
    ) -> None:
        async with lock:
            # Re-check freshness inside the lock — another waiter
            # may have refreshed the cache while we were queued.
            cached = cache.get(wallet_key)
            if cached is not None:
                cached_at, _ = cached
                if (_time.monotonic() - cached_at) < _PNL_COUNTERS_TTL_SECONDS:
                    return
            try:
                from models.database import AsyncSessionLocal
                async with AsyncSessionLocal() as bg_session:
                    await self._compute_pnl_counters_from_orders(
                        bg_session, wallet, wallet_key, cache
                    )
            except Exception as exc:
                # Best-effort telemetry — don't surface as an error.
                # The cache simply stays stale until the next call.
                logger.debug(
                    "PnL counters background refresh failed",
                    wallet=wallet_key[:12],
                    exc_info=exc,
                )

    async def _compute_pnl_counters_from_orders(
        self,
        session: Any,
        wallet: str,
        wallet_key: str,
        cache: dict[str, tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        # ``TraderOrder.actual_profit`` is the verified-truth ledger maintained
        # by ``polymarket_trade_verifier``; aggregating it here keeps the
        # runtime-state row honest without inventing a parallel accumulator.
        from sqlalchemy import func as _func
        from models.database import TraderOrder

        wallet_lower = (wallet or "").lower()
        wallet_addr = _func.lower(_func.coalesce(TraderOrder.execution_wallet_address, ""))
        base_filter = (wallet_addr == wallet_lower) & (
            TraderOrder.actual_profit.isnot(None)
        )
        totals = (
            await session.execute(
                select(
                    _func.coalesce(_func.sum(TraderOrder.actual_profit), 0.0),
                    _func.coalesce(
                        _func.sum(
                            case((TraderOrder.actual_profit > 0, 1), else_=0)
                        ),
                        0,
                    ),
                    _func.coalesce(
                        _func.sum(
                            case((TraderOrder.actual_profit < 0, 1), else_=0)
                        ),
                        0,
                    ),
                ).where(base_filter)
            )
        ).one()

        daily_cutoff = datetime.combine(
            self._daily_volume_reset, datetime.min.time(), tzinfo=timezone.utc
        )
        daily_pnl = (
            await session.execute(
                select(
                    _func.coalesce(_func.sum(TraderOrder.actual_profit), 0.0)
                ).where(
                    base_filter
                    & (
                        _func.coalesce(
                            TraderOrder.executed_at, TraderOrder.created_at
                        )
                        >= daily_cutoff
                    )
                )
            )
        ).scalar() or 0.0

        result = {
            "total_pnl": float(totals[0] or 0.0),
            "winning_trades": int(totals[1] or 0),
            "losing_trades": int(totals[2] or 0),
            "daily_pnl": float(daily_pnl),
        }
        cache[wallet_key] = (_time.monotonic(), result)
        return result

    async def _persist_runtime_state(self) -> None:
        lock = self._get_persist_lock()
        if lock.locked():
            self._runtime_state_persist_dirty = True
            return
        async with lock:
            while True:
                self._runtime_state_persist_dirty = False
                # 2026-05-08: wall-clock bound on _persist_runtime_state_once.
                # The 2026-05-07 soak showed individual upserts taking
                # 1.4-3.1s under DB pool contention. Without this bound,
                # a single stalled asyncpg connection would hold the per-
                # instance persist lock indefinitely and starve the
                # coalesce queue (every subsequent persist request would
                # just set the dirty bit and return, losing observability
                # of how deep the backlog got). 15s is comfortably above
                # the DB-level statement_timeout (main pool: 30s) — wait,
                # that's backwards. We want the client bound to fire
                # BEFORE statement_timeout so we can rollback cleanly,
                # log that the persist cycle overran, and let the next
                # tick retry. 20s — below statement_timeout, above the
                # 99p observed latency (~3s) with margin.
                try:
                    await asyncio.wait_for(
                        self._persist_runtime_state_once(),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    # Keep dirty bit set so next caller re-enters the
                    # coalesce loop and retries. The stalled SQL will
                    # eventually be cancelled by statement_timeout or
                    # the pool reaper; we're just releasing the app-
                    # level lock here so we don't compound the stall.
                    self._runtime_state_persist_dirty = True
                    try:
                        logger.warning(
                            "_persist_runtime_state exceeded 20s wall-clock budget; will retry",
                        )
                    except Exception:
                        pass
                    return
                if not self._runtime_state_persist_dirty:
                    return

    async def _persist_runtime_state_once(self) -> None:
        wallet = self._wallet_for_persistence()
        if not wallet:
            return

        from models.database import AsyncSessionLocal, LiveTradingRuntimeState

        runtime_id = self._runtime_state_id(wallet)
        last_trade_at = _normalize_utc_datetime(self._stats.last_trade_at)
        daily_reset_at = datetime.combine(self._daily_volume_reset, datetime.min.time(), tzinfo=timezone.utc)
        market_positions_json = {str(token_id): str(exposure) for token_id, exposure in self._market_positions.items()}
        pending_reconciliation_json = [dict(item) for item in self._pending_reconciliations]

        # Per-stage breakdown for slow-log diagnosis.  Cycle 3 of the
        # perf-harness loop (post-_persist_orders UPSERT) showed
        # ``select_state`` averaging 2.0-2.6s per slow event.  This
        # is the same SELECT-then-UPDATE anti-pattern we just killed
        # in ``_persist_orders``.  Keep the single UPSERT shape:
        #  - The outer wrapper coalesces overlapping persist calls.
        #  - PG resolves insert/update races via ON CONFLICT on the PK.
        #  - Replace SELECT-then-INSERT-or-UPDATE with single
        #    ``INSERT ... ON CONFLICT DO UPDATE``.  Saves the PK
        #    SELECT round-trip; PG does the existence check inline
        #    against the PK index.
        _prs_started = _time.monotonic()
        _prs_breakdown: dict[str, float] = {}

        def _prs_record(stage: str, started_mono: float) -> None:
            elapsed_ms = (_time.monotonic() - started_mono) * 1000.0
            _prs_breakdown[stage] = round(
                _prs_breakdown.get(stage, 0.0) + elapsed_ms, 1
            )

        for attempt in range(_DB_RETRY_ATTEMPTS):
            _prs_breakdown["attempts"] = float(attempt + 1)
            _stage_started = _time.monotonic()
            async with AsyncSessionLocal() as session:
                _prs_record("session_checkout", _stage_started)
                try:
                    # Derive realized P&L counters from the verified
                    # ground truth (TraderOrder.actual_profit).  The legacy
                    # in-memory accumulators on ``self._stats``/``self._total_pnl``
                    # were never wired to the close-of-position path, which
                    # left ``winning_trades`` / ``losing_trades`` /
                    # ``total_pnl`` stuck at 0 across every wallet.  Sourcing
                    # from TraderOrder also picks up the verifier's
                    # corrections automatically.  Cached for 30s in
                    # ``_derive_pnl_counters_from_orders`` so most calls
                    # never round-trip to the DB.
                    _stage_started = _time.monotonic()
                    derived = await self._derive_pnl_counters_from_orders(
                        session, wallet
                    )
                    _prs_record("derive_pnl", _stage_started)

                    values = {
                        "id": runtime_id,
                        "wallet_address": wallet,
                        "total_trades": int(self._stats.total_trades),
                        "winning_trades": int(derived["winning_trades"]),
                        "losing_trades": int(derived["losing_trades"]),
                        "total_volume": float(self._total_volume),
                        "total_pnl": float(derived["total_pnl"]),
                        "daily_volume": float(self._daily_volume),
                        "daily_pnl": float(derived["daily_pnl"]),
                        "open_positions": int(self._stats.open_positions),
                        "last_trade_at": last_trade_at,
                        "daily_volume_reset_at": daily_reset_at,
                        "market_positions_json": market_positions_json,
                        "pending_reconciliation_json": pending_reconciliation_json,
                        "balance_signature_type": self._balance_signature_type,
                        "updated_at": utcnow(),
                    }

                    _stage_started = _time.monotonic()
                    stmt = pg_insert(LiveTradingRuntimeState).values(values)
                    update_cols = {
                        col: stmt.excluded[col]
                        for col in (
                            "wallet_address", "total_trades", "winning_trades",
                            "losing_trades", "total_volume", "total_pnl",
                            "daily_volume", "daily_pnl", "open_positions",
                            "last_trade_at", "daily_volume_reset_at",
                            "market_positions_json", "pending_reconciliation_json",
                            "balance_signature_type", "updated_at",
                        )
                    }
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["id"], set_=update_cols
                    )
                    await session.execute(stmt)
                    _prs_record("upsert", _stage_started)
                    _stage_started = _time.monotonic()
                    await session.commit()
                    _prs_record("commit", _stage_started)
                    _prs_breakdown["total_ms"] = round(
                        (_time.monotonic() - _prs_started) * 1000.0, 1
                    )
                    if _prs_breakdown["total_ms"] >= 2000.0:
                        try:
                            logger.warning(
                                "_persist_runtime_state slow",
                                breakdown=_prs_breakdown,
                            )
                        except Exception:
                            pass
                    return
                except (OperationalError, InterfaceError) as exc:
                    await session.rollback()
                    is_last = attempt >= _DB_RETRY_ATTEMPTS - 1
                    if not _is_retryable_db_error(exc) or is_last:
                        logger.error("Failed to persist live trading runtime state", exc_info=exc)
                        return
                except Exception as exc:
                    await session.rollback()
                    logger.error("Failed to persist live trading runtime state", exc_info=exc)
                    return
            await asyncio.sleep(_db_retry_delay(attempt))

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

                        self._positions.clear()
                        wallet_key = str(wallet).strip().lower()
                        positions_result = await session.execute(
                            select(LiveTradingPosition).where(
                                func.lower(func.coalesce(LiveTradingPosition.wallet_address, "")) == wallet_key
                            )
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
                                redeemable=bool(getattr(row, "redeemable", False)),
                                counts_as_open=coerce_bool(getattr(row, "counts_as_open", True), default=True),
                                end_date=str(getattr(row, "end_date", "") or "").strip() or None,
                                created_at=_normalize_utc_datetime(row.created_at) or utcnow(),
                            )
                        self._stats.open_positions = sum(
                            1 for position in self._positions.values() if bool(position.counts_as_open)
                        )

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
                            token_key = str(row.token_id or "").strip()
                            side_raw = str(row.side or "").strip().upper()
                            side = OrderSide.SELL if side_raw == OrderSide.SELL.value else OrderSide.BUY
                            order_type = _normalize_order_type(row.order_type)
                            status_raw = str(row.status or "").strip().lower()
                            try:
                                status = OrderStatus(status_raw)
                            except ValueError:
                                status = OrderStatus.PENDING
                            market_question = row.market_question
                            position = self._positions.get(token_key)
                            if position is not None:
                                position_market_question = str(position.market_question or "").strip()
                                if position_market_question:
                                    market_question = position_market_question
                            order = Order(
                                id=str(row.id),
                                market_id=str(row.market_id or "").strip() or None,
                                token_id=token_key,
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
                                market_question=market_question,
                                opportunity_id=row.opportunity_id,
                            )
                            self._remember_order(order)

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
        order_size = max(0.0, safe_float(order.size, 0.0) or 0.0)
        filled_size = max(0.0, safe_float(order.filled_size, 0.0) or 0.0)
        if order_size > 0.0 and filled_size > order_size:
            filled_size = order_size
        average_fill_price = safe_float(order.average_fill_price)
        filled_notional_usd = None
        if filled_size > 0 and average_fill_price is not None and average_fill_price > 0:
            filled_notional_usd = filled_size * average_fill_price
        return {
            "clob_order_id": clob_id,
            "normalized_status": normalized_status,
            "raw_status": str(getattr(order.status, "value", order.status) or ""),
            "size": order_size,
            "filled_size": filled_size,
            "remaining_size": max(0.0, order_size - filled_size),
            "average_fill_price": float(average_fill_price) if average_fill_price is not None else None,
            "limit_price": float(safe_float(order.price, 0.0) or 0.0),
            "filled_notional_usd": float(filled_notional_usd) if filled_notional_usd is not None else None,
            "raw": None,
        }

    def _remember_open_order_snapshot_cache(self, snapshots: dict[str, dict[str, Any]]) -> None:
        cached: dict[str, dict[str, Any]] = {}
        for raw_clob_id, raw_snapshot in (snapshots or {}).items():
            clob_id = str(raw_clob_id or "").strip()
            if not clob_id or not isinstance(raw_snapshot, dict):
                continue
            cached[clob_id] = dict(raw_snapshot)
        self._open_order_snapshot_cache = cached
        self._open_order_snapshot_cache_at = _time.monotonic()

    def _get_recent_open_order_snapshot_cache(self) -> Optional[dict[str, dict[str, Any]]]:
        if self._open_order_snapshot_cache is None:
            return None
        age_seconds = _time.monotonic() - float(self._open_order_snapshot_cache_at or 0.0)
        if age_seconds > _OPEN_ORDER_SNAPSHOT_CACHE_TTL_SECONDS:
            return None
        return {
            clob_id: dict(snapshot)
            for clob_id, snapshot in self._open_order_snapshot_cache.items()
            if isinstance(snapshot, dict)
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
        if size is not None and size > 0.0 and filled_size > size:
            filled_size = size

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
        elif filled_notional_usd is not None and average_fill_price is not None and average_fill_price > 0:
            max_notional = filled_size * average_fill_price
            if max_notional >= 0.0 and filled_notional_usd > max_notional:
                filled_notional_usd = max_notional

        # Polymarket's CLOB returns the EIP712 ``metadata`` field on each
        # open order. The fast tier stamps a deterministic
        # ``fast_idempotency_key`` there so the orphan-reconcile sweep can
        # match a venue order back to a local TraderOrder row that lost
        # its provider_clob_order_id link (e.g. process killed between
        # CLOB success and DB write). Carry it through unchanged so the
        # caller can compare directly against
        # ``derive_fast_idempotency_key`` outputs.
        metadata_raw = (
            server_order.get("metadata")
            or server_order.get("metaData")
            or server_order.get("orderMetadata")
            or ""
        )
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
            "metadata": str(metadata_raw or ""),
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
            normalized_filled_size = max(0.0, float(filled_size))
            order_size = max(0.0, safe_float(order.size, 0.0) or 0.0)
            if order_size > 0.0 and normalized_filled_size > order_size:
                normalized_filled_size = order_size
            order.filled_size = normalized_filled_size
        average_fill_price = safe_float(snapshot.get("average_fill_price"))
        if average_fill_price is not None and average_fill_price > 0:
            order.average_fill_price = float(average_fill_price)
        else:
            # Provider snapshots sometimes omit ``average_fill_price`` (or report
            # 0) on partially/fully filled orders.  When that happens but the
            # snapshot carries a fill notional, infer the average price so the
            # persisted row isn't ``status=filled, average_fill_price=0``.
            current_filled_size = max(0.0, safe_float(order.filled_size, 0.0) or 0.0)
            if current_filled_size > 0:
                filled_notional_usd = safe_float(snapshot.get("filled_notional_usd"))
                if filled_notional_usd is not None and filled_notional_usd > 0:
                    inferred = float(filled_notional_usd) / current_filled_size
                    if inferred > 0:
                        order.average_fill_price = inferred
                elif (
                    not order.average_fill_price or order.average_fill_price <= 0
                ) and order.price and float(order.price) > 0:
                    # Last-resort: use the limit price.  For GTC limits the
                    # fill price equals the limit (or better), and Polymarket
                    # clamps to whole-cent ticks, so this is safe.  This keeps
                    # downstream P&L math from dividing by zero.
                    order.average_fill_price = float(order.price)
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
        provider_bulk_fetch_failed = False
        per_order_not_found: set[str] = set()
        used_recent_open_order_snapshot_cache = False

        def _ingest_open_orders(response_payload: Any) -> None:
            nonlocal provider_fetch_ok
            provider_fetch_ok = True
            parsed_snapshots: dict[str, dict[str, Any]] = {}
            for server_order in self._extract_server_orders(response_payload):
                snapshot = self._parse_provider_order_snapshot(server_order)
                if snapshot is None:
                    continue
                clob_id = str(snapshot["clob_order_id"])
                parsed_snapshots[clob_id] = snapshot
                if clob_id in requested:
                    snapshots[clob_id] = snapshot
            self._remember_open_order_snapshot_cache(parsed_snapshots)

        recent_open_order_snapshots = self._get_recent_open_order_snapshot_cache()
        if recent_open_order_snapshots is not None:
            used_recent_open_order_snapshot_cache = True
            provider_fetch_ok = True
            for clob_id in requested:
                snapshot = recent_open_order_snapshots.get(clob_id)
                if snapshot is not None:
                    snapshots[clob_id] = dict(snapshot)
            per_order_not_found.update(requested.difference(snapshots.keys()))
        elif self._clob_read_circuit_open():
            return cached_fallback
        else:
            try:
                response = await self._run_client_io(self._client.get_open_orders, timeout=_CLOB_READ_TIMEOUT_SECONDS)
                _ingest_open_orders(response)
                self._clob_read_record_success("Open order snapshots fetch")
            except Exception as exc:
                provider_bulk_fetch_failed = True
                self._clob_read_record_failure(exc, "Open order snapshots fetch")
                try:
                    reinitialized = await self.ensure_initialized()
                except Exception as reinit_exc:
                    logger.warning(
                        "Trading client reinitialization failed while fetching order snapshots",
                        exc_info=reinit_exc,
                    )
                    reinitialized = False
                if reinitialized and self.is_ready() and not self._clob_read_circuit_open():
                    try:
                        response = await self._run_client_io(self._client.get_open_orders, timeout=_CLOB_READ_TIMEOUT_SECONDS)
                        _ingest_open_orders(response)
                        provider_bulk_fetch_failed = False
                        self._clob_read_record_success("Open order snapshots fetch")
                    except Exception as retry_exc:
                        self._clob_read_record_failure(retry_exc, "Open order snapshots retry")

        missing = requested.difference(snapshots.keys())
        if (
            missing
            and hasattr(self._client, "get_order")
            and not provider_bulk_fetch_failed
            and not used_recent_open_order_snapshot_cache
            and not self._clob_read_circuit_open()
        ):
            single_lookup_deadline = _time.monotonic() + _SNAPSHOT_SINGLE_LOOKUP_BUDGET_SECONDS
            single_lookup_attempts = 0
            for clob_id in sorted(missing):
                if single_lookup_attempts >= _SNAPSHOT_SINGLE_LOOKUP_MAX:
                    break
                if _time.monotonic() >= single_lookup_deadline:
                    break
                single_lookup_attempts += 1
                try:
                    single_response = await self._run_client_io(
                        self._client.get_order,
                        clob_id,
                        timeout=_CLOB_READ_TIMEOUT_SECONDS,
                    )
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
                            local_order_size = max(0.0, safe_float(local_order.size, 0.0) or 0.0)
                            if local_order_size > 0.0 and inferred_filled_size > local_order_size:
                                inferred_filled_size = local_order_size
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

    def _clob_read_circuit_open(self) -> bool:
        """Check if the CLOB API read circuit breaker is open (API known unreachable)."""
        if self._clob_read_circuit_open_until is None:
            return False
        now_mono = _time.monotonic()
        if now_mono >= self._clob_read_circuit_open_until:
            self._clob_read_circuit_open_until = None
            return False
        return True

    def clob_read_circuit_open(self) -> bool:
        return self._clob_read_circuit_open()

    def _cached_active_open_orders(self) -> list[Order]:
        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        }
        return [
            order
            for order in sorted(self._orders.values(), key=lambda item: item.created_at, reverse=True)
            if order.status in active_statuses
        ]

    def _clob_read_record_failure(self, exc: Exception, operation: str) -> None:
        """Record a CLOB API read failure and open circuit breaker if threshold met."""
        now_mono = _time.monotonic()
        self._clob_read_consecutive_failures += 1
        if self._clob_read_consecutive_failures >= _CLOB_READ_CIRCUIT_BREAKER_THRESHOLD:
            self._clob_read_circuit_open_until = now_mono + _CLOB_READ_CIRCUIT_BREAKER_COOLDOWN
        if now_mono - self._clob_read_last_failure_logged >= _CLOB_READ_FAILURE_LOG_INTERVAL:
            self._clob_read_last_failure_logged = now_mono
            if _is_transient_transport_error(exc):
                if self._clob_read_consecutive_failures < _CLOB_READ_CIRCUIT_BREAKER_THRESHOLD:
                    logger.debug(
                        "%s transient failure",
                        operation,
                        consecutive_failures=self._clob_read_consecutive_failures,
                        circuit_open=False,
                        error_type=type(exc).__name__,
                    )
                    return
                logger.warning(
                    "%s failed",
                    operation,
                    consecutive_failures=self._clob_read_consecutive_failures,
                    circuit_open=self._clob_read_circuit_open_until is not None,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                logger.error(
                    "%s failed",
                    operation,
                    consecutive_failures=self._clob_read_consecutive_failures,
                    exc_info=exc,
                )

    def _clob_read_record_success(self, operation: str) -> None:
        """Record a successful CLOB API read and reset circuit breaker."""
        if self._clob_read_consecutive_failures > 0:
            logger.info(
                "%s recovered",
                operation,
                after_failures=self._clob_read_consecutive_failures,
            )
        self._clob_read_consecutive_failures = 0
        self._clob_read_circuit_open_until = None

    async def _sync_provider_open_orders(self) -> list[Order]:
        if not self.is_ready() and not await self.ensure_initialized():
            return []

        if self._clob_read_circuit_open():
            return self._cached_active_open_orders()

        try:
            provider_response = await self._run_client_io(
                self._client.get_open_orders,
                timeout=_CLOB_READ_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            self._clob_read_record_failure(exc, "Provider open orders sync")
            return self._cached_active_open_orders()

        self._clob_read_record_success("Provider open orders sync")

        provider_orders = self._extract_server_orders(provider_response)
        provider_snapshots_by_clob: dict[str, dict[str, Any]] = {}
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
            provider_snapshots_by_clob[clob_order_id] = dict(snapshot)
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
            cached.status = OrderStatus.FILLED if float(cached.filled_size or 0.0) > 0 else OrderStatus.CANCELLED
            cached.updated_at = utcnow()
            updated_orders.append(cached)

        self._remember_open_order_snapshot_cache(provider_snapshots_by_clob)

        now = utcnow()
        immediate_order_cutoff = 30.0
        for cached in self._orders.values():
            if cached.status not in active_statuses:
                continue
            if cached.order_type not in {OrderType.IOC, OrderType.FAK, OrderType.FOK}:
                continue
            age_seconds = max(0.0, (now - cached.created_at).total_seconds())
            if age_seconds < immediate_order_cutoff:
                continue
            cached.status = OrderStatus.FILLED if float(cached.filled_size or 0.0) > 0 else OrderStatus.CANCELLED
            cached.updated_at = now
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

        if size_usd > max_trade_size and side == OrderSide.BUY:
            return (
                False,
                f"Order size ${float(size_usd):.2f} exceeds maximum ${settings.MAX_TRADE_SIZE_USD:.2f}",
            )

        # Daily volume limit applies only to BUY orders (new exposure).
        # SELL orders (exits/closes) must always be allowed so positions can be unwound.
        projected_daily_volume = self._daily_volume + size_usd
        if projected_daily_volume > max_daily_volume and side == OrderSide.BUY:
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
        # Crypto-latency Fix W: removed the ``await refresh_from_db()``
        # call that previously sat at the head of this function.  On a
        # cache miss it issued 8 parallel ``AsyncSessionLocal`` checkouts
        # against the shared-controls table (scanner / news / weather /
        # discovery / orchestrator / crypto / tracked / events) and
        # took 1.9-3.2 s wall time on the synchronous order submission
        # path — directly observed at 2,187 ms in the 5/2026/05 latency
        # harness ``place_order`` breakdown.  The earlier 2 s TTL fix
        # capped the worst case but every cold-cache moment still paid
        # the full multi-second cost.
        #
        # The actual ``global_pause_state.is_paused`` check already
        # happens 3 lines below inside ``_validate_order`` (live_execution
        # _service.py:3681).  That property accessor is non-blocking
        # (sub-microsecond) and *also* schedules a background
        # ``refresh_from_db()`` if the cache is stale — same correctness,
        # zero blocking on the order's serial path.  Pause-all from the
        # operator UI now propagates to live execution within one
        # background-refresh tick (≤2 s + DB query time) instead of
        # being held up for an explicit pre-flight refresh.
        pass

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

            # Only track BUY volume toward the daily limit — SELL orders
            # (position exits) must always be allowed and should not inflate
            # the counter that gates new entries.
            if side == OrderSide.BUY:
                self._daily_volume += size_usd
            self._total_volume += size_usd
            delta = size_usd if side == OrderSide.BUY else -size_usd
            self._apply_market_exposure_delta(token_id, delta)
            self._sync_stats_from_decimals()
            reserved = True

        if reserved:
            # Persist removed: ``_persist_runtime_state`` was the third
            # call in a chain on every order (validate-reserve here +
            # success-branch inline + post-loop outer).  Each call
            # serializes on ``_persist_lock`` and opens a fresh
            # ``AsyncSessionLocal``; under DB pool pressure this turned
            # the local-only volume bump into a 3-9s wait
            # (``submit_validate_reserve=6297ms`` in production).  The
            # subsequent inner persist (success path) or
            # ``_release_reservation`` persist (failure path) writes the
            # SAME state moments later, so the volume bump always
            # reaches the DB before ``place_order`` returns.  The only
            # window where in-memory volume diverges from persisted is
            # an OS-level worker kill between this point and the next
            # persist — same window that already existed on the
            # exception path; deemed acceptable.
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
            if side == OrderSide.BUY:
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
        metadata: Optional[str] = None,
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

        # Validate the CLOB metadata BEFORE reserving risk budget or
        # touching the trading transport.  Metadata is the only mechanism
        # the reconcile sweep has to recover an orphan venue order whose
        # post-submit DB write was lost (see fast_idempotency.py); we
        # would rather refuse the trade and surface the bug than submit
        # without the safety key and leave a position undiscoverable if
        # the worker crashes mid-flush.  If a caller passed metadata at
        # all, it must be a 0x-prefixed bytes32 hex.  Empty / None means
        # "no idempotency key", which is fine — the SDK substitutes
        # BYTES32_ZERO and reconcile simply skips that submission.
        normalized_metadata: Optional[str] = None
        if metadata is not None:
            normalized_metadata = _normalize_clob_metadata(metadata)
            if normalized_metadata is None:
                order.status = OrderStatus.FAILED
                order.error_message = (
                    "invalid_clob_metadata: metadata must be a 0x-prefixed bytes32 hex string"
                )
                self._remember_order(order)
                await self._persist_orders([order])
                logger.error(
                    "Refusing live order: malformed CLOB metadata (would orphan the venue submission)",
                    token_id=token_key,
                    side=side.value,
                    metadata_excerpt=str(metadata)[:80],
                )
                return order

        # Calculate USD notional with Decimal to avoid float accumulation drift.
        size_usd = _to_decimal(normalized_price) * _to_decimal(normalized_size)
        reserved = False

        # Per-stage breakdown — every async step in this method updates a
        # bucket so the slow log at the end (and the structured payload
        # returned to the orchestrator) names which stage ate the budget.
        # Without this, the orchestrator's ``ps_submit_order`` bucket sees
        # the whole 35-40s wall time as a single opaque blob — production
        # soak (5/2026/05) had no way to tell whether the cost was lock
        # contention, balance fetch fan-out, VPN, or the actual SDK
        # round-trip.
        _po_started = _time.monotonic()
        _po_breakdown: dict[str, float] = {}

        def _po_record(stage: str, started_mono: float) -> None:
            elapsed_ms = (_time.monotonic() - started_mono) * 1000.0
            _po_breakdown[stage] = round(_po_breakdown.get(stage, 0.0) + elapsed_ms, 1)

        # Crypto latency harness: capture the freshest wire ts the system
        # had access to at place_order entry.  This is the conservative
        # ``t0_wire`` baseline — the actual decision may have been made
        # against slightly older data, but no wire event the system saw
        # AFTER this point could have driven this trade.  Used at
        # function exit to compute ``wire_to_*_ms`` deltas in the
        # ``crypto_latency_trace`` log line.  Recorder is non-raising;
        # failure here returns None and the trace just omits the field.
        _wire_ts_at_entry_ms: Optional[int] = None
        try:
            from services.crypto_latency_trace import freshest_wire_ts_ms

            _wire_ts_at_entry_ms = freshest_wire_ts_ms()
        except Exception:
            _wire_ts_at_entry_ms = None

        _stage_started = _time.monotonic()
        # VPN pre-trade check (blocks if VPN required but unreachable)
        vpn_ok, vpn_reason = await pre_trade_vpn_check()
        _po_record("vpn_check", _stage_started)
        if not vpn_ok:
            order.status = OrderStatus.FAILED
            order.error_message = f"VPN check failed: {vpn_reason}"
            self._remember_order(order)
            await self._persist_orders([order])
            logger.error(f"Trade blocked by VPN check: {vpn_reason}")
            return order

        _stage_started = _time.monotonic()
        # Validate and reserve risk budget atomically to prevent async races.
        is_valid, error = await self._validate_and_reserve_order(
            size_usd=size_usd,
            side=side,
            token_id=token_key,
            min_order_size_usd=min_order_size_usd,
        )
        _po_record("validate_reserve", _stage_started)
        if not is_valid:
            order.status = OrderStatus.FAILED
            order.error_message = error
            self._remember_order(order)
            await self._persist_orders([order])
            logger.warning(f"Order validation failed: {error}")
            return order
        reserved = True

        try:
            _stage_started = _time.monotonic()
            await self._sync_trading_transport()
            _po_record("sync_transport", _stage_started)
            _stage_started = _time.monotonic()
            await self._refresh_signature_type()
            _po_record("refresh_signature_type", _stage_started)
            sell_allowance_retry_used = False
            buy_allowance_retry_used = False
            # One-shot guard so we dump the per-signature_type wallet
            # snapshot at most once per ``place_order`` call even if
            # multiple retries hit "not enough balance / allowance".
            # The snapshot itself is 9 SDK calls (3 sig types × refresh +
            # fetch + apply) so spamming it on every retry would amplify
            # gateway pressure exactly when the venue is already
            # rejecting us.
            runtime_state_persisted_inline = False
            # In-flight task for the success-path runtime-state
            # persist.  When ``post_order`` succeeds, we kick this
            # off concurrently with the unconditional
            # ``_persist_orders`` call below — both target different
            # tables with different sessions, so pipelining saves
            # 200-3000ms wall time.  Awaited at the persist-orders
            # site so failures still propagate through the same
            # error-handling pathway the original sequential code
            # used.
            runtime_state_persist_task: asyncio.Task[None] | None = None
            balance_snapshot_logged = False
            if side == OrderSide.BUY and not skip_buy_pre_submit_gate:
                _stage_started = _time.monotonic()
                buy_gate_ok, buy_gate_error = await self._enforce_buy_pre_submit_gate(
                    token_id=token_key,
                    required_notional_usd=size_usd,
                )
                _po_record("buy_pre_submit_gate", _stage_started)
                if not buy_gate_ok:
                    raise RuntimeError(buy_gate_error or "BUY pre-submit gate failed")
            if side == OrderSide.SELL:
                _stage_started = _time.monotonic()
                await self.prepare_sell_balance_allowance(token_key)
                _po_record("prepare_sell_allowance", _stage_started)
                _stage_started = _time.monotonic()
                sell_gate_ok, sell_gate_error = await self._enforce_sell_pre_submit_gate(
                    token_id=token_key,
                    size=normalized_size,
                )
                _po_record("sell_pre_submit_gate", _stage_started)
                if not sell_gate_ok:
                    raise RuntimeError(sell_gate_error or "SELL pre-submit gate failed")

            # Build and sign order using py-clob-client-v2
            from py_clob_client_v2.clob_types import MarketOrderArgs, OrderArgs
            from py_clob_client_v2.order_builder.constants import BUY, SELL

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
                    _po_breakdown["attempts"] = float(attempt + 1)
                    # Time spent waiting for ``_client_io_lock`` is the
                    # most common 30+ s offender — every other concurrent
                    # SDK call (balance refresh, order placement, cancel)
                    # serializes through this single asyncio.Lock.  Split
                    # ``io_lock_wait`` from the actual SDK round-trip so
                    # the slow log can name lock contention vs venue
                    # latency.
                    _lock_wait_started = _time.monotonic()
                    async with self._get_client_io_lock():
                        _po_record("io_lock_wait", _lock_wait_started)
                        if provider_market_order:
                            market_amount = float(normalized_size)
                            if side == OrderSide.BUY:
                                market_amount = float(max(0.0, submit_price) * max(0.0, normalized_size))
                            market_order_kwargs = dict(
                                token_id=token_key,
                                amount=market_amount,
                                side=order_side,
                                price=submit_price,
                                order_type=provider_order_type,
                            )
                            if normalized_metadata is not None:
                                market_order_kwargs["metadata"] = normalized_metadata
                            order_args = MarketOrderArgs(**market_order_kwargs)
                            # ITER-4 (Fix EE): Combine create + post into ONE
                            # ``asyncio.to_thread`` dispatch.  Pre-fix code paid
                            # two executor hops (~30-50 ms each) plus an event-
                            # loop turn between them; the SDK's create + post
                            # have a strict data dependency (post takes the
                            # signed order) so there's nothing to gain from
                            # interleaving on the asyncio side, but every
                            # millisecond of dispatch is hot-path time we don't
                            # want.  Inner stage timings (create + post) are
                            # captured by ``_po_breakdown_ref`` so the harness
                            # still sees them split out.
                            _stage_started = _time.monotonic()
                            client_ref = self._client
                            _po_breakdown_ref = _po_breakdown

                            def _create_and_post_market() -> dict:
                                t_create = _time.monotonic()
                                signed = client_ref.create_market_order(order_args)
                                _po_breakdown_ref["create_market_order"] = round(
                                    _po_breakdown_ref.get("create_market_order", 0.0)
                                    + (_time.monotonic() - t_create) * 1000.0,
                                    1,
                                )
                                t_post = _time.monotonic()
                                resp = client_ref.post_order(
                                    signed, provider_order_type, post_only=post_only
                                )
                                _po_breakdown_ref["post_order"] = round(
                                    _po_breakdown_ref.get("post_order", 0.0)
                                    + (_time.monotonic() - t_post) * 1000.0,
                                    1,
                                )
                                return resp

                            # ITER-5 (Fix FF): dispatch to dedicated CLOB
                            # executor instead of the default thread pool so
                            # this hot-path call never queues behind unrelated
                            # blocking work elsewhere in the process.
                            response = await asyncio.wait_for(
                                asyncio.get_running_loop().run_in_executor(
                                    self._get_clob_executor(), _create_and_post_market
                                ),
                                timeout=_ORDER_SUBMIT_TIMEOUT_SECONDS,
                            )
                            # ``signed_order`` left undefined on this branch —
                            # only the limit-order branch references it later
                            # for retry-on-version-mismatch.  Set to None to
                            # surface any accidental cross-branch usage.
                            signed_order = None
                            _po_record("clob_create_post_combined", _stage_started)
                        else:
                            limit_order_kwargs = dict(
                                price=submit_price,
                                size=normalized_size,
                                side=order_side,
                                token_id=token_key,
                            )
                            if normalized_metadata is not None:
                                limit_order_kwargs["metadata"] = normalized_metadata
                            order_args = OrderArgs(**limit_order_kwargs)
                            # Same combined create + post for the limit-order
                            # path.  The SDK's ``post_order`` for a GTC limit
                            # is the same call shape as for FAK/FOK (just a
                            # different ``order_type``).
                            _stage_started = _time.monotonic()
                            client_ref = self._client
                            _po_breakdown_ref = _po_breakdown
                            _post_only_ref = post_only
                            _provider_order_type_ref = provider_order_type

                            def _create_and_post_limit() -> tuple:
                                t_create = _time.monotonic()
                                signed = client_ref.create_order(order_args)
                                _po_breakdown_ref["create_order"] = round(
                                    _po_breakdown_ref.get("create_order", 0.0)
                                    + (_time.monotonic() - t_create) * 1000.0,
                                    1,
                                )
                                t_post = _time.monotonic()
                                resp = client_ref.post_order(
                                    signed, _provider_order_type_ref, post_only=_post_only_ref
                                )
                                _po_breakdown_ref["post_order"] = round(
                                    _po_breakdown_ref.get("post_order", 0.0)
                                    + (_time.monotonic() - t_post) * 1000.0,
                                    1,
                                )
                                return signed, resp

                            # ITER-5 (Fix FF): dedicated CLOB executor on
                            # the limit-order path too.
                            signed_order, response = await asyncio.wait_for(
                                asyncio.get_running_loop().run_in_executor(
                                    self._get_clob_executor(), _create_and_post_limit
                                ),
                                timeout=_ORDER_SUBMIT_TIMEOUT_SECONDS,
                            )
                            _po_record("clob_create_post_combined", _stage_started)
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
                        and "not enough balance / allowance" in error_text
                    ):
                        share_drift = _parse_clob_share_balance_shortage(error_text)
                        if share_drift is not None:
                            actual_atomic, requested_atomic = share_drift
                            logger.error(
                                "Sell order rejected: outcome-token share balance below requested exit size. "
                                "Refusing further retries — refreshing USDC allowance cannot fix a share shortage; "
                                "the live_trading_positions ledger has drifted from the chain.",
                                token_id=token_key,
                                actual_balance_atomic=actual_atomic,
                                requested_amount_atomic=requested_atomic,
                                shortfall_atomic=requested_atomic - actual_atomic,
                                attempt=attempt + 1,
                            )
                            # Fall through; the for-loop will not retry because
                            # we don't `continue`, and after this block the
                            # exception is re-raised at the end of the handler.
                        elif not sell_allowance_retry_used:
                            sell_allowance_retry_used = True
                            if await self.prepare_sell_balance_allowance(token_key):
                                logger.warning(
                                    "Sell order creation failed with stale balance/allowance cache; refreshed allowances and retrying",
                                    attempt=attempt + 1,
                                    token_id=token_key,
                                )
                                await asyncio.sleep(0)
                                continue
                        if not balance_snapshot_logged:
                            balance_snapshot_logged = True
                            self._emit_balance_rejection_diagnostic(
                                side_text="SELL",
                                token_id=token_key,
                                error_text=error_text,
                            )
                            await self._log_balance_snapshot_per_signature_type(
                                context=f"sell_balance_rejection:{token_key}"
                            )
                    if (
                        side == OrderSide.BUY
                        and "not enough balance / allowance" in error_text
                    ):
                        # Polymarket's CLOB caches USDC balance/allowance
                        # server-side for performance; a fill on a previous
                        # order that freed collateral, or an on-chain allowance
                        # bump, doesn't show up until that cache is refreshed.
                        # Force the venue to re-read the wallet from chain and
                        # retry once. Mirrors the SELL path (line 3052) — until
                        # this branch existed, BUY orders that hit a stale cache
                        # would fail and stay failed (visible as a recurring
                        # "balance: $X, sum of active orders: $X" rejection
                        # even when the wallet on-chain has plenty of USDC).
                        if not buy_allowance_retry_used:
                            buy_allowance_retry_used = True
                            if await self.refresh_collateral_balance_allowance():
                                logger.warning(
                                    "Buy order rejected by stale balance/allowance cache; refreshed cache and retrying",
                                    attempt=attempt + 1,
                                    token_id=token_key,
                                    error_excerpt=error_text[:200],
                                )
                                await asyncio.sleep(0)
                                continue
                        if not balance_snapshot_logged:
                            balance_snapshot_logged = True
                            self._emit_balance_rejection_diagnostic(
                                side_text="BUY",
                                token_id=token_key,
                                error_text=error_text,
                            )
                            await self._log_balance_snapshot_per_signature_type(
                                context=f"buy_balance_rejection:{token_key}"
                            )
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
                        # For FAK/IOC BUY orders, do NOT retry on transport errors.
                        # The order may have been accepted by the CLOB before the
                        # timeout — retrying would create a duplicate on-chain order
                        # (since FAK has no stable nonce for idempotency).  The
                        # reconciliation worker will discover any fills.
                        if side == OrderSide.BUY and provider_market_order:
                            logger.warning(
                                "FAK/IOC BUY transport error — skipping retry to avoid duplicate on-chain order",
                                attempt=attempt + 1,
                                token_id=token_key,
                                error_type=type(exc).__name__,
                                error=str(exc)[:200],
                            )
                            raise
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
                    # Crypto-latency Fix Z: post-placement fill fetch is now
                    # FIRE-AND-FORGET for IOC/FAK/FOK orders.  The order is
                    # already at the venue (post_order returned an order_id);
                    # this HTTP GET to ``client.get_order`` fetches the fill
                    # snapshot so the local Order has ``size_matched`` and
                    # ``average_fill_price`` populated.  Synchronous it cost
                    # 531-3937 ms in the 5/2026/05 latency harness — directly
                    # on the order-return path.  Background-task it costs zero
                    # wall time on the caller; the snapshot is applied
                    # ASYNCHRONOUSLY when the fetch completes.
                    #
                    # Trade-off: in the brief window between place_order
                    # return and the background fetch completing, the
                    # in-memory Order object reads as ``OPEN`` even though
                    # the venue already filled it.  Callers that need
                    # synchronous fill data should call ``get_order``
                    # directly post-placement.  The reconciliation worker
                    # syncs venue → local state on every cycle, so even if
                    # the background fetch fails we converge within one
                    # reconciliation pass.
                    #
                    # Optimization: skip the fetch entirely when the
                    # immediate response already carried fill data
                    # (``size_matched > 0``).  Polymarket's place_order
                    # response sometimes embeds the fill snapshot —
                    # checking saves the HTTP round-trip.
                    needs_fill_fetch = (
                        order.clob_order_id
                        and normalized_order_type in {OrderType.IOC, OrderType.FAK, OrderType.FOK}
                        and hasattr(self._client, "get_order")
                        and float(getattr(order, "filled_size", 0.0) or 0.0) <= 0.0
                    )
                    if needs_fill_fetch:
                        _stage_started = _time.monotonic()
                        clob_id_for_fetch = order.clob_order_id

                        async def _bg_fill_fetch() -> None:
                            try:
                                detail = await self._run_client_io(
                                    self._client.get_order,
                                    clob_id_for_fetch,
                                    timeout=_CLOB_READ_TIMEOUT_SECONDS,
                                )
                                for srv in self._extract_server_orders(
                                    detail if isinstance(detail, (list, dict)) else {}
                                ):
                                    snap = self._parse_provider_order_snapshot(srv)
                                    if (
                                        snap is not None
                                        and str(snap.get("clob_order_id")) == clob_id_for_fetch
                                    ):
                                        self._apply_snapshot_to_order(order, snap)
                                        break
                            except Exception as exc:
                                logger.debug(
                                    "Post-placement fill-price fetch (bg) failed for %s: %s",
                                    clob_id_for_fetch,
                                    exc,
                                )

                        bg_task = asyncio.create_task(
                            _bg_fill_fetch(),
                            name=f"post-fill-fetch-{clob_id_for_fetch[:16]}",
                        )

                        def _log_bg_fill_fetch_failure(task: asyncio.Task) -> None:
                            try:
                                task.result()
                            except asyncio.CancelledError:
                                return
                            except Exception as exc:
                                logger.debug(
                                    "Post-placement fill-fetch background task error",
                                    exc_info=exc,
                                )

                        bg_task.add_done_callback(_log_bg_fill_fetch_failure)
                        # Record near-zero on the synchronous breakdown — the
                        # actual HTTP cost moves to a background task that
                        # the harness will see as a separate ``post_fill_fetch_bg``
                        # log line if it becomes a problem.
                        _po_record("post_placement_fill_fetch", _time.monotonic())
                    _stage_started = _time.monotonic()
                    stats_lock = self._get_stats_lock()
                    async with stats_lock:
                        self._stats.total_trades += 1
                        self._stats.last_trade_at = utcnow()
                    _po_record("stats_lock_update", _stage_started)
                    # Pipelined: kick off runtime-state persistence as
                    # a background task and let it run concurrently
                    # with the unconditional ``_persist_orders`` call
                    # past the retry loop.  Both target different
                    # tables (``LiveTradingRuntimeState`` vs
                    # ``LiveTradingOrder``) with their own sessions,
                    # so there's no row-lock interaction.  The task is
                    # awaited via ``asyncio.gather`` at the persist-
                    # orders site so any exception still propagates
                    # through the original error-handling pathway.
                    runtime_state_persist_task = asyncio.create_task(
                        self._persist_runtime_state(),
                        name="persist_runtime_state_inline",
                    )
                    runtime_state_persisted_inline = True
                    self._invalidate_balance_cache()
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
                    and "not enough balance / allowance" in error_message.lower()
                ):
                    share_drift = _parse_clob_share_balance_shortage(error_message.lower())
                    if share_drift is not None:
                        actual_atomic, requested_atomic = share_drift
                        logger.error(
                            "Sell order rejected: outcome-token share balance below requested exit size. "
                            "Refusing further retries — refreshing USDC allowance cannot fix a share shortage; "
                            "the live_trading_positions ledger has drifted from the chain.",
                            token_id=token_key,
                            actual_balance_atomic=actual_atomic,
                            requested_amount_atomic=requested_atomic,
                            shortfall_atomic=requested_atomic - actual_atomic,
                            attempt=attempt + 1,
                        )
                        # Fall through — exit the retry loop with a normal failure.
                    elif not sell_allowance_retry_used:
                        sell_allowance_retry_used = True
                        if await self.prepare_sell_balance_allowance(token_key):
                            logger.warning(
                                "Sell order rejected with stale balance/allowance cache; refreshed allowances and retrying",
                                attempt=attempt + 1,
                                token_id=token_key,
                            )
                            await asyncio.sleep(0)
                            continue
                    if not balance_snapshot_logged:
                        balance_snapshot_logged = True
                        self._emit_balance_rejection_diagnostic(
                            side_text="SELL",
                            token_id=token_key,
                            error_text=error_message,
                        )
                        await self._log_balance_snapshot_per_signature_type(
                            context=f"sell_balance_rejection:{token_key}"
                        )
                if (
                    side == OrderSide.BUY
                    and "not enough balance / allowance" in error_message.lower()
                ):
                    # Same stale-cache root cause as the create_order path
                    # above — Polymarket's CLOB caches USDC balance/allowance
                    # server-side and rejects BUY orders against the cached
                    # view even when the wallet has plenty on-chain. Force a
                    # cache refresh + one retry. Without this branch BUYs
                    # against a stale cache fail permanently (the recurring
                    # "balance: $X, sum of active orders: $X" rejection users
                    # see despite having ample USDC in the proxy wallet).
                    if not buy_allowance_retry_used:
                        buy_allowance_retry_used = True
                        if await self.refresh_collateral_balance_allowance():
                            logger.warning(
                                "Buy order rejected by stale balance/allowance cache; refreshed cache and retrying",
                                attempt=attempt + 1,
                                token_id=token_key,
                                error_excerpt=error_message[:200],
                            )
                            await asyncio.sleep(0)
                            continue
                    if not balance_snapshot_logged:
                        balance_snapshot_logged = True
                        self._emit_balance_rejection_diagnostic(
                            side_text="BUY",
                            token_id=token_key,
                            error_text=error_message,
                        )
                        await self._log_balance_snapshot_per_signature_type(
                            context=f"buy_balance_rejection:{token_key}"
                        )
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

        except (asyncio.CancelledError, KeyboardInterrupt):
            # CancelledError is BaseException in Python 3.9+ and bypasses
            # ``except Exception``.  Release the reservation before re-raising
            # so the daily volume counter doesn't leak phantom volume on every
            # trader cycle timeout / task cancellation.
            if reserved:
                try:
                    await self._release_reservation(
                        size_usd=size_usd,
                        side=side,
                        token_id=token_key,
                    )
                except Exception:
                    pass
                reserved = False
            raise
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
            error_str = str(e).lower()
            if "no orders found to match" in error_str or "fak" in error_str:
                logger.info(f"Order execution no-fill (FAK/FOK no liquidity): {e}")
            else:
                logger.error(f"Order execution error: {e}")

        order.updated_at = utcnow()
        self._remember_order(order)
        _stage_started = _time.monotonic()
        if runtime_state_persist_task is not None:
            # Crypto-latency Fix X: success path is now FIRE-AND-FORGET
            # for both persist tasks.  The CLOB has already ack'd the
            # order at this point (post_order returned an order_id) so
            # the order is at the venue regardless of the local DB
            # write.  Previously this gather() blocked place_order for
            # 3.6 s wall time waiting for ``_persist_runtime_state`` +
            # ``_persist_orders`` to commit — directly observed at
            # 3,656 ms each in the 5/2026/05 latency harness, dominating
            # 49 % of the 7.5 s place_order wall time.
            #
            # The orchestrator's caller does not need the local DB row
            # to exist before returning — it has the in-memory ``order``
            # object with the venue order_id and uses that for
            # downstream decisions.  The reconciliation worker
            # (``trader_reconciliation_worker``) syncs venue → local
            # DB on every cycle, so any persist failure here is caught
            # and corrected within one reconciliation pass without
            # blocking trade flow.  Failure semantic in the persist
            # task is logged inside ``_persist_orders`` /
            # ``_persist_runtime_state`` themselves; we explicitly
            # attach a done-callback so an exception doesn't get
            # silently swallowed by the asyncio task GC.
            persist_orders_task = asyncio.create_task(
                self._persist_orders([order]),
                name="persist_orders_inline",
            )

            def _log_persist_failure(task: asyncio.Task) -> None:
                try:
                    task.result()
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.warning(
                        "Background persist task failed (order is at venue, "
                        "reconciler will sync)",
                        task_name=task.get_name(),
                        exc_info=exc,
                    )

            runtime_state_persist_task.add_done_callback(_log_persist_failure)
            persist_orders_task.add_done_callback(_log_persist_failure)
            # Record zero ms on the synchronous side — both persists
            # complete asynchronously on a background task; their wall
            # time no longer blocks the order-return path.
            _po_record("persist_runtime_state_inner", _time.monotonic())
            _po_record("persist_orders", _time.monotonic())
        else:
            # Failure path: no in-flight runtime-state task to join.
            # Persist orders synchronously, then runtime state if
            # the success-branch never ran it.
            await self._persist_orders([order])
            _po_record("persist_orders", _stage_started)
            # Skip the outer ``_persist_runtime_state`` when the
            # success-branch already ran it inline.  Was a free 3-10s
            # win per successful order — the same function called
            # twice within microseconds, second call rewriting
            # identical state under DB pool pressure.  Failure paths
            # still go through here so the runtime state captures the
            # OrderStatus.FAILED transition.
            if not runtime_state_persisted_inline:
                _stage_started = _time.monotonic()
                await self._persist_runtime_state()
                _po_record("persist_runtime_state_outer", _stage_started)

        # Stash the breakdown on the Order so the orchestrator's
        # ps_submit_order slow-log can surface it (live_execution_adapter
        # propagates ``order._submit_breakdown`` into the LiveOrderExecution
        # payload).  Always log when the wall time exceeds 5 s so we have
        # data on which sub-stage owns the cost — production saw
        # ps_submit_order = 39.6 s with no visibility into the breakdown.
        _po_total_ms = round((_time.monotonic() - _po_started) * 1000.0, 1)
        _po_breakdown["total_ms"] = _po_total_ms
        try:
            order._submit_breakdown = dict(_po_breakdown)
        except Exception:
            pass
        if _po_total_ms >= 5000.0:
            try:
                logger.warning(
                    "place_order slow",
                    token_id=token_key,
                    side=side.value if hasattr(side, "value") else str(side),
                    total_ms=_po_total_ms,
                    status=str(getattr(order.status, "value", order.status) or ""),
                    breakdown=_po_breakdown,
                )
            except Exception:
                pass

        # Crypto latency harness: emit a structured ``crypto_latency_trace``
        # line on EVERY place_order (success OR failure), so the harness
        # aggregator gets a per-trade sample feed instead of only the
        # >=5s slow-events.  ``wire_to_ack_ms`` is computed against the
        # snapshot taken at function entry; by the time we emit, the
        # CLOB ack has already happened (it's the post_order stage,
        # captured earlier in _po_breakdown).  Pure-additive — never
        # raises into the order-return path.
        try:
            from services.crypto_latency_trace import emit_trace

            now_ms = int(_time.time() * 1000)
            trace_breakdown = dict(_po_breakdown)
            wire_ts_for_trace: Optional[int] = _wire_ts_at_entry_ms
            if wire_ts_for_trace is not None:
                trace_breakdown["wire_to_place_order_end"] = max(
                    0.0, float(now_ms - wire_ts_for_trace)
                )
            emit_trace(
                signal_id=str(getattr(order, "opportunity_id", "") or "") or None,
                token_id=str(token_key) if token_key else None,
                status=str(getattr(order.status, "value", order.status) or ""),
                wire_ts_ms=wire_ts_for_trace,
                breakdown_ms=trace_breakdown,
            )
        except Exception:
            # Never let an instrumentation bug change the order-return
            # path.  The order is fully placed at this point — nothing
            # the harness does should affect the caller.
            pass

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
            response = await self._run_client_io(self._client.cancel_orders, [clob_order_id])
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

    async def get_open_order_snapshots_by_metadata(self) -> dict[str, list[dict[str, Any]]]:
        """Return live open orders indexed by their EIP712 ``metadata`` field.

        Used by the fast-tier orphan reconcile sweep: when a TraderOrder
        row's ``provider_clob_order_id`` was lost (e.g. CLOB-success →
        process-killed → DB-write-never-happened), the only durable link
        back to the venue order is the deterministic metadata key the
        fast path stamps on every submission. This helper does a single
        ``get_open_orders`` round-trip and groups parsed snapshots by
        metadata so the reconciler can do an O(1) lookup per orphan.

        Orders without a metadata field (legacy / non-fast-tier) end up
        under the empty-string key and are simply ignored by the
        reconciler. Failures return an empty dict — callers must treat
        that as "venue is unreachable, retry later".
        """
        if not self.is_ready() and not await self.ensure_initialized():
            return {}

        try:
            provider_response = await self._run_client_io(
                self._client.get_open_orders,
                timeout=_CLOB_READ_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("get_open_order_snapshots_by_metadata: get_open_orders failed", exc_info=exc)
            return {}

        provider_orders = self._extract_server_orders(provider_response)
        out: dict[str, list[dict[str, Any]]] = {}
        for server_order in provider_orders:
            snapshot = self._parse_provider_order_snapshot(server_order)
            if snapshot is None:
                continue
            metadata_key = str(snapshot.get("metadata") or "").strip().lower()
            out.setdefault(metadata_key, []).append(snapshot)
        return out

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
                redeemable = bool(pos.get("redeemable"))
                end_date = _read_text(pos, "endDate", "end_date")

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
                    redeemable=redeemable,
                    counts_as_open=not redeemable,
                    end_date=end_date or None,
                )

            self._positions = next_positions

            self._stats.open_positions = sum(1 for position in self._positions.values() if bool(position.counts_as_open))
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

    def _invalidate_balance_cache(self) -> None:
        self._balance_cache = None
        self._balance_cache_at = 0.0

    async def get_balance(self) -> dict:
        """Get wallet balance.  Results are cached for a few seconds to avoid
        redundant CLOB API round-trips within the same order pipeline."""
        import time as _time

        if (
            self._balance_cache is not None
            and (_time.monotonic() - self._balance_cache_at) < _BALANCE_CACHE_TTL_SECONDS
        ):
            return self._balance_cache

        if not self.is_ready():
            await self.ensure_initialized()
        if not self.is_ready():
            return {"error": "Polymarket credentials not configured"}

        if self._clob_read_circuit_open():
            return {"error": "CLOB API temporarily unreachable (circuit breaker open)"}

        try:
            address = self._execution_wallet_address()
            if not address:
                return {"error": "Could not derive wallet address"}

            try:
                from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

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
                    await self._run_client_io(self._client.update_balance_allowance, params, lock="balance")
                except Exception as exc:
                    self._clob_read_record_failure(exc, "Balance allowance refresh")
                try:
                    payload = await self._run_client_io(self._client.get_balance_allowance, params, lock="balance")
                except Exception as exc:
                    self._clob_read_record_failure(exc, "Balance allowance fetch")
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
            builder_signature_type = getattr(builder, "signature_type", None)
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

            self._clob_read_record_success("Balance fetch")
            selected_signature_type = int(best_snapshot["signature_type"])
            self._balance_signature_type = selected_signature_type
            self._apply_signature_type_to_client(selected_signature_type)

            result = {
                "address": address,
                "balance": best_snapshot["balance"],
                "available": best_snapshot["available"],
                "reserved": best_snapshot["reserved"],
                "currency": "USDC",
                "timestamp": utcnow().isoformat(),
                "positions_value": sum(p.size * p.current_price for p in self._positions.values()),
                "signature_type": selected_signature_type,
            }
            self._balance_cache = result
            self._balance_cache_at = __import__("time").monotonic()
            return result
        except Exception as e:
            return {"error": str(e)}

    _prewarm_in_progress: bool = False

    async def prewarm_clob_market_info_cache(
        self,
        condition_ids: list[str],
        *,
        max_concurrent: int = 2,
    ) -> dict[str, Any]:
        # Fix Y addendum: dedup overlapping prewarm calls.  market_runtime
        # fires one prewarm task per crypto refresh (~3 min); under
        # network jitter the prior call may not have finished before the
        # next refresh kicks off, stacking concurrent _warm_one tasks
        # in the asyncio task tree (observed at 11 in flight in stall
        # dumps despite a per-call Semaphore(4)).  Class-level boolean
        # gate ensures only one prewarm sweep is alive at a time;
        # subsequent overlap calls return immediately with the
        # ``in_progress`` sentinel.
        if self.__class__._prewarm_in_progress:
            return {"prewarmed": 0, "skipped": "in_progress"}
        self.__class__._prewarm_in_progress = True
        try:
            return await self._prewarm_clob_market_info_cache_inner(
                condition_ids, max_concurrent=max_concurrent
            )
        finally:
            self.__class__._prewarm_in_progress = False

    async def _prewarm_clob_market_info_cache_inner(
        self,
        condition_ids: list[str],
        *,
        max_concurrent: int = 2,
    ) -> dict[str, Any]:
        """Pre-populate the SDK's per-token caches so ``create_order`` is HTTP-free.

        Crypto-latency Fix Y: ``py_clob_client_v2.create_order`` makes up to
        three SYNCHRONOUS HTTP round-trips on the order-creation path
        (``__resolve_tick_size``, ``get_neg_risk``, ``__resolve_fee_rate_bps``)
        for any token whose metadata isn't yet in the SDK's per-instance
        dicts.  ``get_clob_market_info(condition_id)`` is ONE HTTP call
        that populates ``__tick_sizes``, ``__neg_risk``, AND
        ``__fee_infos`` for both YES + NO tokens of the market in one
        shot.

        Wired in from ``market_runtime._refresh_crypto_markets`` so each
        time the runtime discovers / re-fetches the crypto market list,
        a background task warms the SDK cache.  ``create_order`` then
        hits cache (sub-microsecond) on the hot path.

        Returns a small status dict so the caller can log/aggregate.
        Never raises — failures degrade silently and the next order
        falls back to the SDK's lazy-fetch path.
        """
        if not self.is_ready() or self._client is None:
            return {"prewarmed": 0, "skipped": "not_ready"}
        unique_ids = sorted({cid for cid in (condition_ids or []) if cid})
        if not unique_ids:
            return {"prewarmed": 0, "skipped": "no_condition_ids"}

        # Probe what's already cached so we don't re-issue HTTP calls
        # for markets the SDK has already seen.  ``__token_condition_map``
        # is populated by ``get_clob_market_info`` and indirectly by
        # ``get_tick_size`` etc., so its presence is the cheapest way
        # to know the row is warm.
        try:
            already_cached = set(getattr(self._client, "_BaseClobClientV2__token_condition_map", {}).values())
        except Exception:
            already_cached = set()
        # Some Python implementations expose name-mangled attrs without
        # the class prefix or via a different path; fall back gracefully
        # if the introspection didn't pick anything up.
        if not already_cached:
            for attr in (
                "__token_condition_map",
                "_token_condition_map",
                "_BaseClobClient__token_condition_map",
            ):
                try:
                    already_cached = set(getattr(self._client, attr, {}).values())
                    if already_cached:
                        break
                except Exception:
                    pass

        targets = [cid for cid in unique_ids if cid not in already_cached]
        if not targets:
            return {"prewarmed": 0, "skipped": "all_cached", "total": len(unique_ids)}

        succeeded = 0
        failed = 0
        queue: asyncio.Queue[str] = asyncio.Queue()
        for cid in targets:
            queue.put_nowait(cid)

        async def _warm_worker() -> None:
            nonlocal succeeded, failed
            while True:
                try:
                    cid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await asyncio.to_thread(self._client.get_clob_market_info, cid)
                    succeeded += 1
                except Exception:
                    failed += 1
                finally:
                    queue.task_done()
                    await asyncio.sleep(0)

        workers = [
            asyncio.create_task(_warm_worker(), name=f"clob-market-info-prewarm-{idx}")
            for idx in range(min(max(1, int(max_concurrent)), len(targets)))
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        except Exception:
            return {
                "prewarmed": succeeded,
                "failed": failed,
                "total": len(unique_ids),
                "targets": len(targets),
            }
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
        if succeeded > 0:
            try:
                logger.info(
                    "Prewarmed CLOB market info cache",
                    targets=len(targets),
                    succeeded=succeeded,
                    failed=failed,
                    total=len(unique_ids),
                )
            except Exception:
                pass
        return {
            "prewarmed": succeeded,
            "failed": failed,
            "total": len(unique_ids),
            "targets": len(targets),
        }


# Singleton instance
live_execution_service = LiveExecutionService()
