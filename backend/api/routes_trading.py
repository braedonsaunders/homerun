"""
Trading API Routes

Endpoints for real trading on Polymarket.

IMPORTANT: These endpoints execute real trades with real money.
Ensure TRADING_ENABLED=true and proper credentials are configured.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from config import settings
from services.trading import (
    trading_service,
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/trading", tags=["Trading"])


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
            created_at=order.created_at
        )


class PositionResponse(BaseModel):
    token_id: str
    market_id: str
    market_question: str
    outcome: str
    size: float
    average_cost: float
    current_price: float
    unrealized_pnl: float

    @classmethod
    def from_position(cls, pos: Position) -> "PositionResponse":
        return cls(
            token_id=pos.token_id,
            market_id=pos.market_id,
            market_question=pos.market_question,
            outcome=pos.outcome,
            size=pos.size,
            average_cost=pos.average_cost,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl
        )


class TradingStatusResponse(BaseModel):
    enabled: bool
    initialized: bool
    wallet_address: Optional[str]
    stats: dict
    limits: dict


# ==================== ENDPOINTS ====================

@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status():
    """Get trading service status and configuration"""
    stats = trading_service.get_stats()

    return TradingStatusResponse(
        enabled=settings.TRADING_ENABLED,
        initialized=trading_service.is_ready(),
        wallet_address=trading_service._get_wallet_address(),
        stats={
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "total_volume": stats.total_volume,
            "total_pnl": stats.total_pnl,
            "daily_volume": stats.daily_volume,
            "daily_pnl": stats.daily_pnl,
            "open_positions": stats.open_positions,
            "last_trade_at": stats.last_trade_at.isoformat() if stats.last_trade_at else None
        },
        limits={
            "max_trade_size_usd": settings.MAX_TRADE_SIZE_USD,
            "max_daily_volume": settings.MAX_DAILY_TRADE_VOLUME,
            "max_open_positions": settings.MAX_OPEN_POSITIONS,
            "min_order_size_usd": settings.MIN_ORDER_SIZE_USD,
            "max_slippage_percent": settings.MAX_SLIPPAGE_PERCENT
        }
    )


@router.post("/initialize")
async def initialize_trading():
    """Initialize the trading service with configured credentials"""
    if trading_service.is_ready():
        return {"status": "already_initialized", "message": "Trading service already initialized"}

    success = await trading_service.initialize()
    if success:
        return {"status": "success", "message": "Trading service initialized"}
    else:
        raise HTTPException(
            status_code=400,
            detail="Failed to initialize trading. Check credentials and TRADING_ENABLED setting."
        )


@router.post("/orders", response_model=OrderResponse)
async def place_order(request: PlaceOrderRequest):
    """
    Place a new order.

    Requires trading to be enabled and initialized.
    """
    if not trading_service.is_ready():
        raise HTTPException(status_code=400, detail="Trading service not initialized")

    try:
        side = OrderSide(request.side.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid side. Must be BUY or SELL")

    try:
        order_type = OrderType(request.order_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid order type. Must be GTC, FOK, or GTD")

    order = await trading_service.place_order(
        token_id=request.token_id,
        side=side,
        price=request.price,
        size=request.size,
        order_type=order_type,
        market_question=request.market_question
    )

    if order.status == OrderStatus.FAILED:
        raise HTTPException(status_code=400, detail=order.error_message)

    return OrderResponse.from_order(order)


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(limit: int = 100, status: Optional[str] = None):
    """Get recent orders"""
    orders = trading_service.get_orders(limit)

    if status:
        try:
            filter_status = OrderStatus(status.lower())
            orders = [o for o in orders if o.status == filter_status]
        except ValueError:
            pass

    return [OrderResponse.from_order(o) for o in orders]


@router.get("/orders/open", response_model=list[OrderResponse])
async def get_open_orders():
    """Get all open orders"""
    orders = await trading_service.get_open_orders()
    return [OrderResponse.from_order(o) for o in orders]


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get a specific order by ID"""
    order = trading_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse.from_order(order)


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    success = await trading_service.cancel_order(order_id)
    if success:
        return {"status": "cancelled", "order_id": order_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to cancel order")


@router.delete("/orders")
async def cancel_all_orders():
    """Cancel all open orders"""
    count = await trading_service.cancel_all_orders()
    return {"status": "success", "cancelled_count": count}


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions():
    """Get current open positions"""
    positions = await trading_service.sync_positions()
    return [PositionResponse.from_position(p) for p in positions]


@router.get("/balance")
async def get_balance():
    """Get wallet balance"""
    balance = await trading_service.get_balance()
    if "error" in balance:
        raise HTTPException(status_code=400, detail=balance["error"])
    return balance


@router.post("/execute-opportunity")
async def execute_opportunity(request: ExecuteOpportunityRequest):
    """
    Execute an arbitrage opportunity.

    Takes the positions from an opportunity and places orders.
    """
    if not trading_service.is_ready():
        raise HTTPException(status_code=400, detail="Trading service not initialized")

    orders = await trading_service.execute_opportunity(
        opportunity_id=request.opportunity_id,
        positions=request.positions,
        size_usd=request.size_usd
    )

    failed_orders = [o for o in orders if o.status == OrderStatus.FAILED]
    if failed_orders:
        return {
            "status": "partial_failure",
            "message": f"{len(failed_orders)} of {len(orders)} orders failed",
            "orders": [OrderResponse.from_order(o).dict() for o in orders]
        }

    return {
        "status": "success",
        "orders": [OrderResponse.from_order(o).dict() for o in orders]
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
    cancelled_count = await trading_service.cancel_all_orders()

    return {
        "status": "emergency_stop_executed",
        "cancelled_orders": cancelled_count,
        "message": "All open orders cancelled. Trading service remains active."
    }
