"""
Orchestrator live execution API routes.

Endpoints for real trading on Polymarket.

IMPORTANT: These endpoints execute real trades with real money.
Ensure proper credentials are configured.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Awaitable, Optional
from datetime import datetime
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
from utils.logger import get_logger

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
        return cls(
            token_id=pos.token_id,
            market_id=pos.market_id,
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


class TradingStatusResponse(BaseModel):
    initialized: bool
    authenticated: bool
    credentials_configured: bool
    wallet_address: Optional[str]
    execution_wallet_address: Optional[str]
    eoa_wallet_address: Optional[str]
    proxy_funder_wallet: Optional[str]
    auth_error: Optional[str]
    stats: dict
    limits: dict
    vpn: dict


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

    return TradingStatusResponse(
        initialized=initialized,
        authenticated=authenticated,
        credentials_configured=credentials_configured,
        wallet_address=wallet_address,
        execution_wallet_address=execution_wallet_address or wallet_address,
        eoa_wallet_address=eoa_wallet_address,
        proxy_funder_wallet=proxy_funder_wallet,
        auth_error=auth_error,
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
    If already initialized, re-runs the USDC allowance approval (useful
    after funding the wallet with MATIC for gas)."""
    if live_execution_service.is_ready():
        # Re-run allowance approval in case the wallet was just funded with MATIC.
        await live_execution_service._approve_clob_allowance()
        return {
            "status": "already_initialized",
            "message": "Trading service already initialized; USDC allowance re-checked",
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
    """Re-run the on-chain USDC approve for the CLOB exchange contract.
    Call this after funding the trading wallet with MATIC for gas fees."""
    if not live_execution_service.is_ready():
        raise HTTPException(status_code=400, detail="Trading service not initialized")
    await live_execution_service._approve_clob_allowance()
    return {"status": "success", "message": "USDC allowance check/approve completed"}


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
            "orders": [OrderResponse.from_order(o).dict() for o in orders],
        }

    return {
        "status": "success",
        "orders": [OrderResponse.from_order(o).dict() for o in orders],
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
