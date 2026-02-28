"""
Orchestrator live execution API routes.

Endpoints for real trading on Polymarket.

IMPORTANT: These endpoints execute real trades with real money.
Ensure proper credentials are configured.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import asyncio

from config import settings
from services.polymarket import polymarket_client
from services.live_execution_service import (
    live_execution_service,
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus,
)
from services.trading_proxy import verify_vpn_active, _get_config as get_proxy_config
from utils.converters import safe_float
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)
router = APIRouter(prefix="/trader-orchestrator/live", tags=["Trader Orchestrator"])


# ==================== REQUEST/RESPONSE MODELS ====================


class PlaceOrderRequest(BaseModel):
    token_id: str = Field(..., description="CLOB token ID")
    side: str = Field(..., description="BUY or SELL")
    price: float = Field(..., ge=0.01, le=0.99, description="Price per share")
    size: float = Field(..., gt=0, description="Number of shares")
    order_type: str = Field(default="GTC", description="GTC, FOK, or GTD")
    market_question: Optional[str] = None


class ExecuteOpportunityRequest(BaseModel):
    opportunity_id: str
    positions: list[dict]
    size_usd: float = Field(..., gt=0, le=10000, description="Total USD to invest")


class OrderResponse(BaseModel):
    id: str
    token_id: str
    side: str
    price: float
    size: float
    order_type: str
    status: str
    filled_size: float
    clob_order_id: Optional[str]
    error_message: Optional[str]
    market_question: Optional[str]
    created_at: datetime

    @classmethod
    def from_order(cls, order: Order) -> "OrderResponse":
        return cls(
            id=order.id,
            token_id=order.token_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            order_type=order.order_type.value,
            status=order.status.value,
            filled_size=order.filled_size,
            clob_order_id=order.clob_order_id,
            error_message=order.error_message,
            market_question=order.market_question,
            created_at=order.created_at,
        )


class PositionResponse(BaseModel):
    token_id: str
    market_id: str
    market_slug: Optional[str] = None
    event_slug: Optional[str] = None
    market_question: str
    outcome: str
    size: float
    average_cost: float
    current_price: float
    unrealized_pnl: float

    @classmethod
    def from_position(cls, pos: Position, market_info: Optional[dict] = None) -> "PositionResponse":
        market_info = market_info or {}
        resolved_market_id = str(market_info.get("condition_id") or pos.market_id or "").strip()
        if not resolved_market_id:
            resolved_market_id = pos.market_id
        return cls(
            token_id=pos.token_id,
            market_id=resolved_market_id,
            market_slug=market_info.get("slug") or None,
            event_slug=market_info.get("event_slug") or None,
            market_question=pos.market_question,
            outcome=pos.outcome,
            size=pos.size,
            average_cost=pos.average_cost,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
        )


class CancelAllOrdersResponse(BaseModel):
    status: str
    requested_count: int
    cancelled_count: int
    failed_count: int
    failed_order_ids: list[str]
    message: str


class NativeGasStatusResponse(BaseModel):
    wallet_address: Optional[str]
    affordable_for_approval: bool
    balance_wei: int
    balance_native: float
    gas_price_wei: int
    required_wei_for_approval: int
    required_native_for_approval: float
    error: Optional[str]


class ExecutionPathStatusResponse(BaseModel):
    normal_trading: str
    direct_ctf_actions: str


class TradingStatusResponse(BaseModel):

    initialized: bool
    authenticated: bool
    credentials_configured: bool
    wallet_address: Optional[str]
    execution_wallet_address: Optional[str]
    eoa_wallet_address: Optional[str]
    proxy_funder_wallet: Optional[str]
    auth_error: Optional[str]
    native_gas: NativeGasStatusResponse
    execution_paths: ExecutionPathStatusResponse
    stats: dict
    limits: dict
    vpn: dict


class LiveWalletFillResponse(BaseModel):
    id: str
    side: str
    condition_id: str
    token_id: str
    market_title: str
    market_slug: Optional[str] = None
    event_slug: Optional[str] = None
    category: Optional[str] = None
    outcome: Optional[str] = None
    size: float
    price: float
    notional: float
    timestamp: datetime
    transaction_hash: Optional[str] = None


class LiveWalletRoundTripResponse(BaseModel):
    id: str
    condition_id: str
    token_id: str
    market_title: str
    market_slug: Optional[str] = None
    event_slug: Optional[str] = None
    category: Optional[str] = None
    outcome: Optional[str] = None
    quantity: float
    avg_buy_price: float
    avg_sell_price: float
    buy_notional: float
    sell_notional: float
    realized_pnl: float
    roi_percent: float
    opened_at: datetime
    closed_at: datetime
    hold_minutes: float
    lots_matched: int


class LiveWalletOpenLotResponse(BaseModel):
    token_id: str
    condition_id: str
    market_title: str
    market_slug: Optional[str] = None
    event_slug: Optional[str] = None
    category: Optional[str] = None
    outcome: Optional[str] = None
    remaining_size: float
    avg_cost: float
    cost_basis: float
    opened_at: datetime


class LiveWalletPerformanceResponse(BaseModel):
    wallet_address: str
    generated_at: datetime
    fills: list[LiveWalletFillResponse]
    round_trips: list[LiveWalletRoundTripResponse]
    open_lots: list[LiveWalletOpenLotResponse]
    summary: dict[str, float | int]


def _parse_trade_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        as_float = float(text)
    except Exception:
        as_float = None
    if as_float is not None:
        if as_float > 1_000_000_000_000:
            as_float /= 1000.0
        try:
            return datetime.fromtimestamp(as_float, tz=timezone.utc)
        except Exception:
            return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_trade_timestamp(trade: dict[str, Any]) -> Optional[datetime]:
    for key in ("timestamp_iso", "match_time", "timestamp", "time", "created_at", "createdAt"):
        parsed = _parse_trade_timestamp(trade.get(key))
        if parsed is not None:
            return parsed
    return None


def _normalize_trade_side(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"buy", "b", "bid", "buy_yes", "buy_no"}:
        return "buy"
    if raw in {"sell", "s", "ask", "sell_yes", "sell_no"}:
        return "sell"
    if raw in {"long"}:
        return "buy"
    if raw in {"short"}:
        return "sell"
    return raw


def _normalize_outcome(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.upper()
    if normalized in {"Y", "YES", "TRUE", "UP"}:
        return "YES"
    if normalized in {"N", "NO", "FALSE", "DOWN"}:
        return "NO"
    return text


def _extract_category_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("label", "name", "value", "category"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    text = str(value or "").strip()
    return text


def _extract_tag_values(value: Any) -> list[str]:
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    tags: list[str] = []
    for item in items:
        tag = _extract_category_text(item)
        if tag:
            tags.append(tag)
    return tags


def _resolve_trade_category(trade: dict[str, Any], market_info: Optional[dict[str, Any]]) -> Optional[str]:
    sources: list[Any] = [
        trade.get("category"),
        trade.get("market_category"),
        trade.get("event_category"),
    ]

    event = trade.get("event")
    if isinstance(event, dict):
        sources.append(event.get("category"))
        sources.extend(_extract_tag_values(event.get("tags")))
    sources.extend(_extract_tag_values(trade.get("tags")))

    if isinstance(market_info, dict):
        sources.extend(
            [
                market_info.get("category"),
                market_info.get("market_category"),
                market_info.get("event_category"),
            ]
        )
        sources.extend(_extract_tag_values(market_info.get("tags")))
        events = market_info.get("events")
        if isinstance(events, list):
            for row in events:
                if not isinstance(row, dict):
                    continue
                sources.append(row.get("category"))
                sources.extend(_extract_tag_values(row.get("tags")))

    for source in sources:
        category = _extract_category_text(source)
        if category:
            return category
    return None


def _extract_trade_condition_id(trade: dict[str, Any]) -> str:
    condition_id = str(trade.get("conditionId") or trade.get("condition_id") or "").strip()
    if condition_id:
        return condition_id
    market = str(trade.get("market") or "").strip()
    if market.startswith("0x"):
        return market
    return ""


def _extract_trade_token_id(trade: dict[str, Any], market_info: Optional[dict[str, Any]]) -> str:
    direct = str(
        trade.get("asset_id")
        or trade.get("asset")
        or trade.get("token_id")
        or trade.get("tokenId")
        or ""
    ).strip()
    if direct:
        return direct

    if not isinstance(market_info, dict):
        return ""

    token_ids_raw = market_info.get("token_ids")
    if not isinstance(token_ids_raw, list):
        token_ids_raw = market_info.get("tokenIds")
    token_ids = [str(token or "").strip() for token in (token_ids_raw or []) if str(token or "").strip()]
    if not token_ids:
        return ""

    outcome_index = safe_float(
        trade.get("outcomeIndex") if trade.get("outcomeIndex") is not None else trade.get("outcome_index"),
        None,
    )
    if outcome_index is not None:
        idx = int(outcome_index)
        if 0 <= idx < len(token_ids):
            return token_ids[idx]

    outcome_text = str(trade.get("outcome") or trade.get("token_outcome") or "").strip().lower()
    if outcome_text:
        outcomes_raw = market_info.get("outcomes")
        if isinstance(outcomes_raw, list):
            for idx, label in enumerate(outcomes_raw):
                if idx < len(token_ids) and str(label or "").strip().lower() == outcome_text:
                    return token_ids[idx]

    return token_ids[0]


def _resolve_trade_market_title(trade: dict[str, Any], market_info: Optional[dict[str, Any]], fallback: str) -> str:
    title = str(
        trade.get("market_title")
        or trade.get("title")
        or trade.get("question")
        or ""
    ).strip()
    if title:
        return title
    if isinstance(market_info, dict):
        from_market = str(market_info.get("groupItemTitle") or market_info.get("question") or "").strip()
        if from_market:
            return from_market
    return fallback


def _resolve_trade_market_slug(trade: dict[str, Any], market_info: Optional[dict[str, Any]]) -> Optional[str]:
    slug = str(trade.get("market_slug") or trade.get("slug") or "").strip()
    if slug:
        return slug
    if isinstance(market_info, dict):
        from_market = str(market_info.get("slug") or "").strip()
        if from_market:
            return from_market
    return None


def _resolve_trade_event_slug(trade: dict[str, Any], market_info: Optional[dict[str, Any]]) -> Optional[str]:
    slug = str(trade.get("event_slug") or trade.get("eventSlug") or "").strip()
    if slug:
        return slug
    if isinstance(market_info, dict):
        from_market = str(market_info.get("event_slug") or "").strip()
        if from_market:
            return from_market
    return None


async def _resolve_live_wallet_address() -> str:
    try:
        await live_execution_service.ensure_initialized()
    except Exception as exc:
        logger.debug("Live wallet resolution skipped service initialization", exc_info=exc)

    wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    if wallet:
        return wallet

    wallet = str(live_execution_service._get_wallet_address() or "").strip()
    if wallet:
        return wallet

    private_key, _api_key, _api_secret, _api_passphrase, _ = await live_execution_service._resolve_polymarket_credentials()
    if not private_key:
        return ""

    try:
        from eth_account import Account

        return str(Account.from_key(private_key).address or "").strip()
    except Exception:
        return ""


def _empty_live_performance_response(wallet: str) -> LiveWalletPerformanceResponse:
    return LiveWalletPerformanceResponse(
        wallet_address=wallet,
        generated_at=utcnow(),
        fills=[],
        round_trips=[],
        open_lots=[],
        summary={
            "total_fills": 0,
            "buy_fills": 0,
            "sell_fills": 0,
            "total_round_trips": 0,
            "winning_round_trips": 0,
            "losing_round_trips": 0,
            "win_rate_percent": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_realized_pnl": 0.0,
            "total_buy_notional": 0.0,
            "total_sell_notional": 0.0,
            "avg_roi_percent": 0.0,
            "avg_hold_minutes": 0.0,
            "profit_factor": 0.0,
            "unmatched_sell_size": 0.0,
            "open_inventory_notional": 0.0,
            "open_inventory_size": 0.0,
            "open_lot_count": 0,
        },
    )


# ==================== ENDPOINTS ====================


@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status():
    """Get trading service status and configuration"""
    stats = live_execution_service.get_stats()
    initialized = live_execution_service.is_ready()
    authenticated = initialized
    credentials_configured = False
    auth_error: Optional[str] = None
    execution_wallet_address = str(live_execution_service.get_execution_wallet_address() or "").strip() or None
    eoa_wallet_address = str(getattr(live_execution_service, "_eoa_address", "") or "").strip() or None
    proxy_funder_wallet = str(getattr(live_execution_service, "_proxy_funder_address", "") or "").strip() or None
    wallet_address = execution_wallet_address or live_execution_service._get_wallet_address()

    private_key, api_key, api_secret, api_passphrase, _ = await live_execution_service._resolve_polymarket_credentials()
    credentials_configured = bool(private_key and api_key and api_secret and api_passphrase)

    if not wallet_address and private_key:
        try:
            from eth_account import Account

            wallet_address = Account.from_key(private_key).address
            if not eoa_wallet_address:
                eoa_wallet_address = wallet_address
        except Exception:
            wallet_address = None

    if not authenticated and credentials_configured:
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            client = ClobClient(
                host=settings.CLOB_API_URL,
                key=private_key,
                chain_id=settings.CHAIN_ID,
                creds=ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                ),
            )
            await asyncio.to_thread(client.get_orders)
            authenticated = True
        except Exception as exc:
            auth_error = str(exc)

    native_gas = {
        "wallet_address": eoa_wallet_address or execution_wallet_address or wallet_address,
        "affordable_for_approval": False,
        "balance_wei": 0,
        "balance_native": 0.0,
        "gas_price_wei": 0,
        "required_wei_for_approval": 0,
        "required_native_for_approval": 0.0,
        "error": None,
    }
    try:
        from services.ctf_execution import ctf_execution_service

        gas_snapshot = await ctf_execution_service.get_native_gas_affordability(gas_limit=170_000)
        balance_wei = int(gas_snapshot.get("balance_wei") or 0)
        required_wei = int(gas_snapshot.get("required_wei") or 0)
        gas_price_wei = int(gas_snapshot.get("gas_price_wei") or 0)
        gas_wallet_address = str(gas_snapshot.get("wallet_address") or native_gas["wallet_address"] or "").strip() or None
        native_gas = {
            "wallet_address": gas_wallet_address,
            "affordable_for_approval": bool(gas_snapshot.get("affordable", False)),
            "balance_wei": balance_wei,
            "balance_native": float(balance_wei / 10**18),
            "gas_price_wei": gas_price_wei,
            "required_wei_for_approval": required_wei,
            "required_native_for_approval": float(required_wei / 10**18),
            "error": str(gas_snapshot.get("error") or "").strip() or None,
        }
    except Exception as exc:
        native_gas["error"] = str(exc)

    return TradingStatusResponse(
        initialized=initialized,
        authenticated=authenticated,
        credentials_configured=credentials_configured,
        wallet_address=wallet_address,
        execution_wallet_address=execution_wallet_address or wallet_address,
        eoa_wallet_address=eoa_wallet_address,
        proxy_funder_wallet=proxy_funder_wallet,
        auth_error=auth_error,
        native_gas=native_gas,
        execution_paths={
            "normal_trading": "clob_only",
            "direct_ctf_actions": "explicit_only",
        },
        stats={
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "total_volume": stats.total_volume,
            "total_pnl": stats.total_pnl,
            "daily_volume": stats.daily_volume,
            "daily_pnl": stats.daily_pnl,
            "open_positions": stats.open_positions,
            "last_trade_at": stats.last_trade_at.isoformat() if stats.last_trade_at else None,
        },
        limits={
            "max_trade_size_usd": settings.MAX_TRADE_SIZE_USD,
            "max_daily_volume": settings.MAX_DAILY_TRADE_VOLUME,
            "max_open_positions": settings.MAX_OPEN_POSITIONS,
            "min_order_size_usd": settings.MIN_ORDER_SIZE_USD,
            "max_slippage_percent": settings.MAX_SLIPPAGE_PERCENT,
        },
        vpn={
            "proxy_enabled": get_proxy_config().enabled,
            "require_vpn": get_proxy_config().require_vpn,
        },
    )


@router.get("/vpn-status")
async def get_vpn_status():
    """
    Check trading VPN proxy status.

    Verifies proxy connectivity and compares direct vs proxy IP
    to confirm the VPN is active.
    """
    status = await verify_vpn_active()
    return status


@router.post("/initialize")
async def initialize_trading():
    """Initialize the trading service with configured credentials.
    If already initialized, refreshes CLOB balance/allowance cache."""
    if live_execution_service.is_ready():
        await live_execution_service._approve_clob_allowance()
        return {
            "status": "already_initialized",
            "message": "Trading service already initialized; CLOB allowance cache refreshed",
        }

    success = await live_execution_service.initialize()
    if success:
        return {"status": "success", "message": "Trading service initialized"}
    else:
        raise HTTPException(
            status_code=400,
            detail="Failed to initialize trading. Check credentials are configured.",
        )


@router.post("/approve-allowance")
async def approve_clob_allowance():
    """Refresh CLOB collateral balance/allowance cache for supported signature types."""
    if not live_execution_service.is_ready():
        raise HTTPException(status_code=400, detail="Trading service not initialized")
    await live_execution_service._approve_clob_allowance()
    return {"status": "success", "message": "CLOB allowance cache refresh completed"}


@router.post("/orders", response_model=OrderResponse)
async def place_order(request: PlaceOrderRequest):
    """
    Place a new order.

    Requires trading to be initialized.
    """
    if not live_execution_service.is_ready():
        raise HTTPException(status_code=400, detail="Trading service not initialized")

    try:
        side = OrderSide(request.side.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid side. Must be BUY or SELL")

    try:
        order_type = OrderType(request.order_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid order type. Must be GTC, FOK, or GTD")

    order = await live_execution_service.place_order(
        token_id=request.token_id,
        side=side,
        price=request.price,
        size=request.size,
        order_type=order_type,
        market_question=request.market_question,
    )

    if order.status == OrderStatus.FAILED:
        raise HTTPException(status_code=400, detail=order.error_message)

    return OrderResponse.from_order(order)


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(limit: int = 100, status: Optional[str] = None):
    """Get recent orders"""
    filter_status: Optional[OrderStatus] = None
    if status:
        try:
            filter_status = OrderStatus(status.lower())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid order status filter") from exc

    orders = await live_execution_service.get_recent_orders(limit=limit, status=filter_status)

    return [OrderResponse.from_order(o) for o in orders]


@router.get("/orders/open", response_model=list[OrderResponse])
async def get_open_orders():
    """Get all open orders"""
    orders = await live_execution_service.get_open_orders()
    return [OrderResponse.from_order(o) for o in orders]


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get a specific order by ID"""
    order = live_execution_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse.from_order(order)


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    success = await live_execution_service.cancel_order(order_id)
    if success:
        return {"status": "cancelled", "order_id": order_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to cancel order")


@router.delete("/orders", response_model=CancelAllOrdersResponse)
async def cancel_all_orders():
    """Cancel all open orders"""
    result = await live_execution_service.cancel_all_orders()
    status = str(result.get("status") or "")
    if status == "partial_failure":
        raise HTTPException(status_code=409, detail=result)
    if status == "failed":
        raise HTTPException(status_code=502, detail=result)
    return result


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions():
    """Get current open positions"""
    positions = await live_execution_service.sync_positions()
    if not positions:
        return []

    lookups: dict[str, Awaitable[Optional[dict]]] = {}
    for pos in positions:
        market_id = pos.market_id
        if not market_id or market_id in lookups:
            continue
        if market_id.startswith("0x"):
            lookups[market_id] = polymarket_client.get_market_by_condition_id(market_id)
        else:
            lookups[market_id] = polymarket_client.get_market_by_token_id(market_id)

    market_info_by_id: dict[str, dict] = {}
    if lookups:
        results = await asyncio.gather(*lookups.values(), return_exceptions=True)
        for market_id, info in zip(lookups.keys(), results):
            if isinstance(info, dict):
                market_info_by_id[market_id] = info

    return [PositionResponse.from_position(p, market_info_by_id.get(p.market_id)) for p in positions]


@router.get("/performance", response_model=LiveWalletPerformanceResponse)
async def get_live_wallet_performance(limit: int = Query(default=1000, ge=1, le=5000)):
    wallet_address = await _resolve_live_wallet_address()
    if not wallet_address:
        raise HTTPException(status_code=409, detail="Live execution wallet is not configured.")

    try:
        raw_trades = await polymarket_client.get_wallet_trades(wallet_address, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Failed to fetch wallet trades from Polymarket.") from exc

    if not raw_trades:
        return _empty_live_performance_response(wallet_address)

    try:
        enriched = await polymarket_client.enrich_trades_with_market_info(raw_trades)
    except Exception:
        enriched = raw_trades

    condition_ids: set[str] = set()
    token_ids: set[str] = set()
    for trade in enriched:
        if not isinstance(trade, dict):
            continue
        condition_id = _extract_trade_condition_id(trade)
        if condition_id:
            condition_ids.add(condition_id)
        token_id = str(
            trade.get("asset_id")
            or trade.get("asset")
            or trade.get("token_id")
            or trade.get("tokenId")
            or ""
        ).strip()
        if token_id:
            token_ids.add(token_id)

    market_info_by_condition: dict[str, Optional[dict[str, Any]]] = {}
    market_info_by_token: dict[str, Optional[dict[str, Any]]] = {}

    semaphore = asyncio.Semaphore(12)

    async def _fetch_condition(condition_id: str) -> None:
        async with semaphore:
            try:
                market_info_by_condition[condition_id] = await polymarket_client.get_market_by_condition_id(condition_id)
            except Exception:
                market_info_by_condition[condition_id] = None

    async def _fetch_token(token_id: str) -> None:
        async with semaphore:
            try:
                info = await polymarket_client.get_market_by_token_id(token_id)
            except Exception:
                info = None
            market_info_by_token[token_id] = info
            if isinstance(info, dict):
                condition_id = str(info.get("condition_id") or "").strip()
                if condition_id and condition_id not in market_info_by_condition:
                    market_info_by_condition[condition_id] = info

    if condition_ids:
        await asyncio.gather(*[_fetch_condition(condition_id) for condition_id in sorted(condition_ids)])
    if token_ids:
        await asyncio.gather(*[_fetch_token(token_id) for token_id in sorted(token_ids)])

    fills: list[dict[str, Any]] = []
    for idx, item in enumerate(enriched):
        if not isinstance(item, dict):
            continue
        timestamp = _read_trade_timestamp(item)
        if timestamp is None:
            continue

        side = _normalize_trade_side(item.get("side") or item.get("trade_side") or item.get("type"))
        if side not in {"buy", "sell"}:
            continue

        size = safe_float(item.get("size"), None)
        if size is None:
            size = safe_float(item.get("amount"), None)
        if size is None:
            size = safe_float(item.get("shares"), None)
        if size is None or size <= 0:
            continue

        price = safe_float(item.get("price"), None)
        if price is None:
            price = safe_float(item.get("avg_price"), None)
        if price is None or price < 0:
            continue

        condition_id = _extract_trade_condition_id(item)
        market_info = market_info_by_condition.get(condition_id) if condition_id else None
        token_id = _extract_trade_token_id(item, market_info)
        if not token_id:
            token_id = str(item.get("market") or "").strip()
        if not token_id:
            token_id = f"unknown:{idx}"

        if market_info is None:
            market_info = market_info_by_token.get(token_id)
            if isinstance(market_info, dict) and not condition_id:
                condition_id = str(market_info.get("condition_id") or "").strip()
        if not condition_id:
            condition_id = str(item.get("market") or "").strip()
            if not condition_id:
                condition_id = token_id

        fill_id = str(
            item.get("id")
            or item.get("transactionHash")
            or item.get("transaction_hash")
            or f"{token_id}:{timestamp.timestamp()}:{idx}"
        ).strip()
        if not fill_id:
            fill_id = f"{token_id}:{timestamp.timestamp()}:{idx}"

        category = _resolve_trade_category(item, market_info)

        outcome = _normalize_outcome(item.get("outcome") or item.get("token_outcome"))
        fills.append(
            {
                "id": fill_id,
                "side": side,
                "condition_id": condition_id,
                "token_id": token_id,
                "market_title": _resolve_trade_market_title(item, market_info, condition_id),
                "market_slug": _resolve_trade_market_slug(item, market_info),
                "event_slug": _resolve_trade_event_slug(item, market_info),
                "category": category,
                "outcome": outcome,
                "size": float(size),
                "price": float(price),
                "notional": float(size * price),
                "timestamp": timestamp,
                "transaction_hash": str(item.get("transactionHash") or item.get("transaction_hash") or "").strip() or None,
            }
        )

    if not fills:
        return _empty_live_performance_response(wallet_address)

    fills.sort(key=lambda row: (row["timestamp"], row["id"]))

    inventory_by_token: dict[str, deque[dict[str, Any]]] = {}
    round_trips: list[dict[str, Any]] = []
    unmatched_sell_size = 0.0
    fill_epsilon = 1e-9

    for fill in fills:
        token_id = str(fill["token_id"])
        token_inventory = inventory_by_token.get(token_id)
        if token_inventory is None:
            token_inventory = deque()
            inventory_by_token[token_id] = token_inventory

        if fill["side"] == "buy":
            token_inventory.append(
                {
                    "remaining_size": float(fill["size"]),
                    "price": float(fill["price"]),
                    "timestamp": fill["timestamp"],
                    "condition_id": fill["condition_id"],
                    "market_title": fill["market_title"],
                    "market_slug": fill["market_slug"],
                    "event_slug": fill["event_slug"],
                    "category": fill["category"],
                    "outcome": fill["outcome"],
                }
            )
            continue

        remaining_sell = float(fill["size"])
        matched_size = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        hold_minutes_weighted = 0.0
        earliest_buy_ts: Optional[datetime] = None
        matched_category: Optional[str] = None
        lots_matched = 0

        while remaining_sell > fill_epsilon and token_inventory:
            lot = token_inventory[0]
            lot_remaining = float(lot.get("remaining_size") or 0.0)
            if lot_remaining <= fill_epsilon:
                token_inventory.popleft()
                continue

            matched = min(remaining_sell, lot_remaining)
            lots_matched += 1
            matched_size += matched
            buy_notional += matched * float(lot["price"])
            sell_notional += matched * float(fill["price"])
            if isinstance(lot.get("timestamp"), datetime):
                hold_minutes = max(0.0, (fill["timestamp"] - lot["timestamp"]).total_seconds() / 60.0)
                hold_minutes_weighted += hold_minutes * matched
                if earliest_buy_ts is None or lot["timestamp"] < earliest_buy_ts:
                    earliest_buy_ts = lot["timestamp"]
            if matched_category is None:
                lot_category = str(lot.get("category") or "").strip()
                if lot_category:
                    matched_category = lot_category

            lot["remaining_size"] = max(0.0, lot_remaining - matched)
            if float(lot["remaining_size"]) <= fill_epsilon:
                token_inventory.popleft()
            remaining_sell -= matched

        if matched_size > fill_epsilon:
            pnl = sell_notional - buy_notional
            round_trips.append(
                {
                    "id": f"{fill['id']}:{len(round_trips) + 1}",
                    "condition_id": str(fill["condition_id"]),
                    "token_id": token_id,
                    "market_title": str(fill["market_title"]),
                    "market_slug": fill["market_slug"],
                    "event_slug": fill["event_slug"],
                    "category": fill["category"] or matched_category,
                    "outcome": fill["outcome"],
                    "quantity": matched_size,
                    "avg_buy_price": (buy_notional / matched_size) if matched_size > 0 else 0.0,
                    "avg_sell_price": float(fill["price"]),
                    "buy_notional": buy_notional,
                    "sell_notional": sell_notional,
                    "realized_pnl": pnl,
                    "roi_percent": ((pnl / buy_notional) * 100.0) if buy_notional > 0 else 0.0,
                    "opened_at": earliest_buy_ts or fill["timestamp"],
                    "closed_at": fill["timestamp"],
                    "hold_minutes": (hold_minutes_weighted / matched_size) if matched_size > 0 else 0.0,
                    "lots_matched": lots_matched,
                }
            )

        if remaining_sell > fill_epsilon:
            unmatched_sell_size += remaining_sell

    round_trips.sort(key=lambda row: row["closed_at"], reverse=True)

    open_lots: list[dict[str, Any]] = []
    for token_id, token_inventory in inventory_by_token.items():
        for lot in token_inventory:
            remaining_size = float(lot.get("remaining_size") or 0.0)
            if remaining_size <= fill_epsilon:
                continue
            avg_cost = float(lot.get("price") or 0.0)
            open_lots.append(
                {
                    "token_id": token_id,
                    "condition_id": str(lot.get("condition_id") or token_id),
                    "market_title": str(lot.get("market_title") or token_id),
                    "market_slug": lot.get("market_slug"),
                    "event_slug": lot.get("event_slug"),
                    "category": lot.get("category"),
                    "outcome": lot.get("outcome"),
                    "remaining_size": remaining_size,
                    "avg_cost": avg_cost,
                    "cost_basis": remaining_size * avg_cost,
                    "opened_at": lot.get("timestamp") or utcnow(),
                }
            )

    open_lots.sort(key=lambda row: row["opened_at"], reverse=True)

    winning_round_trips = sum(1 for row in round_trips if float(row["realized_pnl"]) > 0.0)
    losing_round_trips = sum(1 for row in round_trips if float(row["realized_pnl"]) < 0.0)
    gross_profit = sum(max(0.0, float(row["realized_pnl"])) for row in round_trips)
    gross_loss = sum(abs(min(0.0, float(row["realized_pnl"]))) for row in round_trips)
    net_realized_pnl = sum(float(row["realized_pnl"]) for row in round_trips)
    total_buy_notional = sum(float(row["buy_notional"]) for row in round_trips)
    total_sell_notional = sum(float(row["sell_notional"]) for row in round_trips)
    avg_roi_percent = (
        sum(float(row["roi_percent"]) for row in round_trips) / len(round_trips)
        if round_trips
        else 0.0
    )
    avg_hold_minutes = (
        sum(float(row["hold_minutes"]) for row in round_trips) / len(round_trips)
        if round_trips
        else 0.0
    )
    buy_fills = sum(1 for row in fills if row["side"] == "buy")
    sell_fills = sum(1 for row in fills if row["side"] == "sell")
    open_inventory_notional = sum(float(row["cost_basis"]) for row in open_lots)
    open_inventory_size = sum(float(row["remaining_size"]) for row in open_lots)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0

    return LiveWalletPerformanceResponse(
        wallet_address=wallet_address,
        generated_at=utcnow(),
        fills=fills,
        round_trips=round_trips,
        open_lots=open_lots,
        summary={
            "total_fills": len(fills),
            "buy_fills": buy_fills,
            "sell_fills": sell_fills,
            "total_round_trips": len(round_trips),
            "winning_round_trips": winning_round_trips,
            "losing_round_trips": losing_round_trips,
            "win_rate_percent": (winning_round_trips / len(round_trips) * 100.0) if round_trips else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_realized_pnl": net_realized_pnl,
            "total_buy_notional": total_buy_notional,
            "total_sell_notional": total_sell_notional,
            "avg_roi_percent": avg_roi_percent,
            "avg_hold_minutes": avg_hold_minutes,
            "profit_factor": profit_factor,
            "unmatched_sell_size": unmatched_sell_size,
            "open_inventory_notional": open_inventory_notional,
            "open_inventory_size": open_inventory_size,
            "open_lot_count": len(open_lots),
        },
    )


@router.get("/balance")
async def get_balance():
    """Get wallet balance"""
    balance = await live_execution_service.get_balance()
    if "error" in balance:
        raise HTTPException(status_code=400, detail=balance["error"])
    return balance


@router.post("/execute-opportunity")
async def execute_opportunity(request: ExecuteOpportunityRequest):
    """
    Execute an arbitrage opportunity.

    Takes the positions from an opportunity and places orders.
    """
    if not live_execution_service.is_ready():
        raise HTTPException(status_code=400, detail="Trading service not initialized")

    orders = await live_execution_service.execute_opportunity(
        opportunity_id=request.opportunity_id,
        positions=request.positions,
        size_usd=request.size_usd,
    )

    failed_orders = [o for o in orders if o.status == OrderStatus.FAILED]
    if failed_orders:
        return {
            "status": "partial_failure",
            "message": f"{len(failed_orders)} of {len(orders)} orders failed",
            "orders": [OrderResponse.from_order(o).model_dump() for o in orders],
        }

    return {
        "status": "success",
        "orders": [OrderResponse.from_order(o).model_dump() for o in orders],
    }


# ==================== SAFETY ENDPOINTS ====================


@router.post("/emergency-stop")
async def emergency_stop():
    """
    Emergency stop - cancel all orders immediately.

    Use in case of unexpected behavior or market conditions.
    """
    logger.warning("EMERGENCY STOP triggered")

    # Cancel all open orders
    cancellation_result = await live_execution_service.cancel_all_orders()
    cancellation_status = str(cancellation_result.get("status") or "")

    if cancellation_status == "partial_failure":
        raise HTTPException(
            status_code=409,
            detail={
                "status": "emergency_stop_partial_failure",
                "message": "Emergency stop completed with cancellation failures.",
                "cancel_result": cancellation_result,
            },
        )
    if cancellation_status == "failed":
        raise HTTPException(
            status_code=502,
            detail={
                "status": "emergency_stop_failed",
                "message": "Emergency stop failed to cancel open orders.",
                "cancel_result": cancellation_result,
            },
        )

    return {
        "status": "emergency_stop_executed",
        "cancelled_orders": cancellation_result.get("cancelled_count", 0),
        "message": "All open orders cancelled. Trading service remains active.",
        "cancel_result": cancellation_result,
    }
