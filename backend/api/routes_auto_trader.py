"""
Auto Trader API Routes

Endpoints for configuring and controlling the autonomous trading engine.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from services.auto_trader import auto_trader, AutoTraderMode
from models import StrategyType
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auto-trader", tags=["Auto Trader"])


# ==================== REQUEST/RESPONSE MODELS ====================

class AutoTraderConfigRequest(BaseModel):
    mode: Optional[str] = Field(None, description="disabled, paper, live, or shadow")
    enabled_strategies: Optional[list[str]] = None
    min_roi_percent: Optional[float] = Field(None, ge=0, le=100)
    max_risk_score: Optional[float] = Field(None, ge=0, le=1)
    min_liquidity_usd: Optional[float] = Field(None, ge=0)
    base_position_size_usd: Optional[float] = Field(None, ge=1)
    max_position_size_usd: Optional[float] = Field(None, ge=1)
    max_daily_trades: Optional[int] = Field(None, ge=1)
    max_daily_loss_usd: Optional[float] = Field(None, ge=0)
    require_confirmation: Optional[bool] = None


class AutoTraderStatusResponse(BaseModel):
    mode: str
    running: bool
    config: dict
    stats: dict


# ==================== ENDPOINTS ====================

@router.get("/status", response_model=AutoTraderStatusResponse)
async def get_auto_trader_status():
    """Get auto trader status, configuration, and statistics"""
    return AutoTraderStatusResponse(
        mode=auto_trader.config.mode.value,
        running=auto_trader._running,
        config=auto_trader.get_config(),
        stats=auto_trader.get_stats()
    )


@router.post("/start")
async def start_auto_trader(mode: Optional[str] = None):
    """
    Start the auto trader.

    Modes:
    - paper: Simulation trading (safe, no real money)
    - live: Real trading (requires credentials)
    - shadow: Record trades without executing

    Default mode is 'paper' if not specified.
    """
    if auto_trader._running:
        return {"status": "already_running", "mode": auto_trader.config.mode.value}

    if mode:
        try:
            auto_trader.config.mode = AutoTraderMode(mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Must be one of: disabled, paper, live, shadow"
            )
    elif auto_trader.config.mode == AutoTraderMode.DISABLED:
        auto_trader.config.mode = AutoTraderMode.PAPER

    await auto_trader.start()

    logger.info(f"Auto trader started in {auto_trader.config.mode.value} mode")

    return {
        "status": "started",
        "mode": auto_trader.config.mode.value,
        "message": f"Auto trader started in {auto_trader.config.mode.value} mode"
    }


@router.post("/stop")
async def stop_auto_trader():
    """Stop the auto trader"""
    if not auto_trader._running:
        return {"status": "not_running"}

    auto_trader.stop()

    return {"status": "stopped", "message": "Auto trader stopped"}


@router.put("/config")
async def update_auto_trader_config(config: AutoTraderConfigRequest):
    """
    Update auto trader configuration.

    Only provided fields will be updated.
    """
    updates = {}

    if config.mode is not None:
        try:
            updates["mode"] = AutoTraderMode(config.mode.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {config.mode}")

    if config.enabled_strategies is not None:
        try:
            strategies = [StrategyType(s.lower()) for s in config.enabled_strategies]
            updates["enabled_strategies"] = strategies
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {e}")

    if config.min_roi_percent is not None:
        updates["min_roi_percent"] = config.min_roi_percent

    if config.max_risk_score is not None:
        updates["max_risk_score"] = config.max_risk_score

    if config.min_liquidity_usd is not None:
        updates["min_liquidity_usd"] = config.min_liquidity_usd

    if config.base_position_size_usd is not None:
        updates["base_position_size_usd"] = config.base_position_size_usd

    if config.max_position_size_usd is not None:
        updates["max_position_size_usd"] = config.max_position_size_usd

    if config.max_daily_trades is not None:
        updates["max_daily_trades"] = config.max_daily_trades

    if config.max_daily_loss_usd is not None:
        updates["max_daily_loss_usd"] = config.max_daily_loss_usd

    if config.require_confirmation is not None:
        updates["require_confirmation"] = config.require_confirmation

    auto_trader.configure(**updates)

    return {
        "status": "updated",
        "config": auto_trader.get_config()
    }


@router.get("/trades")
async def get_auto_trader_trades(limit: int = 100, status: Optional[str] = None):
    """Get recent auto-executed trades"""
    trades = auto_trader.get_trades(limit)

    if status:
        trades = [t for t in trades if t["status"] == status]

    return trades


@router.get("/stats")
async def get_auto_trader_stats():
    """Get detailed auto trader statistics"""
    return auto_trader.get_stats()


@router.post("/reset-stats")
async def reset_auto_trader_stats():
    """Reset auto trader statistics (does not affect trade history)"""
    from services.auto_trader import AutoTraderStats

    auto_trader.stats = AutoTraderStats()
    auto_trader._daily_reset_date = datetime.utcnow().date()

    return {"status": "reset", "message": "Statistics reset"}


@router.post("/reset-circuit-breaker")
async def reset_circuit_breaker():
    """Manually reset the circuit breaker"""
    auto_trader.stats.circuit_breaker_until = None
    auto_trader.stats.consecutive_losses = 0

    logger.info("Circuit breaker manually reset")

    return {"status": "reset", "message": "Circuit breaker reset"}


# ==================== QUICK ACTIONS ====================

@router.post("/enable-live-trading")
async def enable_live_trading(
    confirm: bool = False,
    max_daily_loss: float = 100.0
):
    """
    Enable live trading mode.

    Requires explicit confirmation.
    Sets conservative default limits.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to enable live trading. This will trade REAL MONEY."
        )

    # Check if trading service can be initialized
    from services.trading import trading_service

    if not trading_service.is_ready():
        success = await trading_service.initialize()
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to initialize trading service. Check credentials."
            )

    # Set conservative limits
    auto_trader.configure(
        mode=AutoTraderMode.LIVE,
        max_daily_loss_usd=max_daily_loss,
        require_confirmation=False  # Can be set to True for extra safety
    )

    if not auto_trader._running:
        await auto_trader.start()

    logger.warning("LIVE TRADING ENABLED")

    return {
        "status": "live_trading_enabled",
        "warning": "Auto trader is now executing REAL trades",
        "max_daily_loss": max_daily_loss,
        "config": auto_trader.get_config()
    }


@router.post("/emergency-stop")
async def emergency_stop():
    """
    Emergency stop - immediately stop all auto trading.

    Also cancels all open orders from the trading service.
    """
    logger.warning("EMERGENCY STOP triggered for auto trader")

    # Stop auto trader
    auto_trader.stop()
    auto_trader.config.mode = AutoTraderMode.DISABLED

    # Cancel all open orders
    from services.trading import trading_service
    cancelled_orders = 0

    if trading_service.is_ready():
        cancelled_orders = await trading_service.cancel_all_orders()

    return {
        "status": "emergency_stop_executed",
        "auto_trader": "stopped",
        "mode": "disabled",
        "cancelled_orders": cancelled_orders,
        "message": "All automated trading has been stopped"
    }
