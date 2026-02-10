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
    paper_account_capital: Optional[float] = Field(
        None, ge=100, description="Starting capital for paper trading"
    )

    # Settlement time filters
    max_end_date_days: Optional[int] = Field(
        None,
        ge=1,
        description="Max days until settlement (skip markets further out). Set to null to disable.",
    )
    min_end_date_days: Optional[int] = Field(
        None,
        ge=0,
        description="Min days until settlement (skip markets settling too soon)",
    )
    prefer_near_settlement: Optional[bool] = Field(
        None, description="Boost score for markets settling sooner"
    )

    # Opportunity prioritization
    priority_method: Optional[str] = Field(
        None,
        description="How to rank opportunities: 'roi', 'annualized_roi', or 'composite'",
    )
    settlement_weight: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Weight for settlement proximity in composite score",
    )
    roi_weight: Optional[float] = Field(
        None, ge=0, le=1, description="Weight for ROI in composite score"
    )
    liquidity_weight: Optional[float] = Field(
        None, ge=0, le=1, description="Weight for liquidity in composite score"
    )
    risk_weight: Optional[float] = Field(
        None, ge=0, le=1, description="Weight for (inverse) risk in composite score"
    )

    # Event concentration limits
    max_trades_per_event: Optional[int] = Field(
        None, ge=1, description="Max trades on markets within the same event"
    )
    max_exposure_per_event_usd: Optional[float] = Field(
        None, ge=0, description="Max total $ exposure per event"
    )

    # Exclusion filters
    excluded_categories: Optional[list[str]] = Field(
        None, description="Categories to exclude (e.g. ['Politics', 'Sports'])"
    )
    excluded_keywords: Optional[list[str]] = Field(
        None,
        description="Keywords to exclude from titles/descriptions (e.g. ['presidential', '2028'])",
    )
    excluded_event_slugs: Optional[list[str]] = Field(
        None, description="Event slugs to exclude (partial match)"
    )

    # Volume filter
    min_volume_usd: Optional[float] = Field(
        None, ge=0, description="Minimum market trading volume in USD"
    )

    # Spread trading exits
    take_profit_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Sell when price rises X% above entry (0 = disabled)",
    )
    stop_loss_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Sell when price drops X% below entry (0 = disabled)",
    )
    enable_spread_exits: Optional[bool] = Field(
        None, description="Whether to set TP/SL on new positions"
    )

    # AI: Resolution analysis gate
    ai_resolution_gate: Optional[bool] = Field(
        None,
        description="Require AI resolution analysis before trading (cached per market, 24h TTL)",
    )
    ai_max_resolution_risk: Optional[float] = Field(
        None, ge=0, le=1, description="Block trades if resolution risk exceeds this"
    )
    ai_min_resolution_clarity: Optional[float] = Field(
        None, ge=0, le=1, description="Block trades if resolution clarity is below this"
    )
    ai_resolution_block_avoid: Optional[bool] = Field(
        None, description="Hard block when resolution analysis recommends 'avoid'"
    )
    ai_resolution_model: Optional[str] = Field(
        None,
        description="LLM model for resolution analysis (e.g. 'gpt-4o-mini', 'gemini-2.0-flash')",
    )
    ai_skip_on_analysis_failure: Optional[bool] = Field(
        None,
        description="If true, skip trade when analysis fails. If false, allow trade through (fail-open).",
    )

    # AI: Opportunity judge position sizing
    ai_position_sizing: Optional[bool] = Field(
        None, description="Use AI judge score to scale position sizes"
    )
    ai_min_score_to_trade: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Hard floor: skip if AI overall_score below this (0 = disabled)",
    )
    ai_score_size_multiplier: Optional[bool] = Field(
        None, description="Scale position size by AI score (0.8 score = 80% size)"
    )
    ai_score_boost_threshold: Optional[float] = Field(
        None, ge=0, le=1, description="Boost size when AI score exceeds this"
    )
    ai_score_boost_multiplier: Optional[float] = Field(
        None, ge=1.0, le=3.0, description="Multiplier for high-confidence AI trades"
    )
    ai_judge_model: Optional[str] = Field(
        None,
        description="LLM model for opportunity judging (e.g. 'gpt-4o-mini', 'gemini-2.0-flash')",
    )

    # LLM verification before trading
    llm_verify_trades: Optional[bool] = Field(
        None,
        description="When enabled, consult AI before executing each trade. Trades recommended as skip/strong_skip are blocked.",
    )
    llm_verify_strategies: Optional[list[str]] = Field(
        None,
        description="Strategy types to LLM-verify (empty list = verify all if llm_verify_trades is True)",
    )

    # Scanner auto AI scoring
    auto_ai_scoring: Optional[bool] = Field(
        None,
        description="When enabled, the scanner automatically AI-scores all opportunities after each scan. Default is OFF.",
    )


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
        stats=auto_trader.get_stats(),
    )


@router.post("/start")
async def start_auto_trader(
    mode: Optional[str] = None, account_id: Optional[str] = None
):
    """
    Start the auto trader.

    Modes:
    - paper: Simulation trading (safe, no real money)
    - live: Real trading (requires credentials)
    - shadow: Record trades without executing

    Default mode is 'paper' if not specified.
    If account_id is provided, the auto trader will use that simulation account
    instead of creating a new one.
    """
    if auto_trader._running:
        return {"status": "already_running", "mode": auto_trader.config.mode.value}

    if mode:
        try:
            auto_trader.config.mode = AutoTraderMode(mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Must be one of: disabled, paper, live, shadow",
            )
    elif auto_trader.config.mode == AutoTraderMode.DISABLED:
        auto_trader.config.mode = AutoTraderMode.PAPER

    # Set selected paper account if provided
    if account_id:
        auto_trader.config.paper_account_id = account_id

    await auto_trader.start()

    logger.info(f"Auto trader started in {auto_trader.config.mode.value} mode")

    return {
        "status": "started",
        "mode": auto_trader.config.mode.value,
        "message": f"Auto trader started in {auto_trader.config.mode.value} mode",
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
        strategies = []
        for s in config.enabled_strategies:
            s_lower = s.lower()
            try:
                strategies.append(StrategyType(s_lower))
            except ValueError:
                # Accept plugin slugs as plain strings
                strategies.append(s_lower)
        updates["enabled_strategies"] = strategies

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

    if config.paper_account_capital is not None:
        updates["paper_account_capital"] = config.paper_account_capital

    # Settlement time filters
    if config.max_end_date_days is not None:
        updates["max_end_date_days"] = config.max_end_date_days

    if config.min_end_date_days is not None:
        updates["min_end_date_days"] = config.min_end_date_days

    if config.prefer_near_settlement is not None:
        updates["prefer_near_settlement"] = config.prefer_near_settlement

    # Priority/scoring
    if config.priority_method is not None:
        valid_methods = ("roi", "annualized_roi", "composite")
        if config.priority_method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority_method: {config.priority_method}. Must be one of: {valid_methods}",
            )
        updates["priority_method"] = config.priority_method

    if config.settlement_weight is not None:
        updates["settlement_weight"] = config.settlement_weight

    if config.roi_weight is not None:
        updates["roi_weight"] = config.roi_weight

    if config.liquidity_weight is not None:
        updates["liquidity_weight"] = config.liquidity_weight

    if config.risk_weight is not None:
        updates["risk_weight"] = config.risk_weight

    # Event concentration
    if config.max_trades_per_event is not None:
        updates["max_trades_per_event"] = config.max_trades_per_event

    if config.max_exposure_per_event_usd is not None:
        updates["max_exposure_per_event_usd"] = config.max_exposure_per_event_usd

    # Exclusions
    if config.excluded_categories is not None:
        updates["excluded_categories"] = config.excluded_categories

    if config.excluded_keywords is not None:
        updates["excluded_keywords"] = config.excluded_keywords

    if config.excluded_event_slugs is not None:
        updates["excluded_event_slugs"] = config.excluded_event_slugs

    # Volume
    if config.min_volume_usd is not None:
        updates["min_volume_usd"] = config.min_volume_usd

    # Spread trading exits
    if config.take_profit_pct is not None:
        updates["take_profit_pct"] = config.take_profit_pct

    if config.stop_loss_pct is not None:
        updates["stop_loss_pct"] = config.stop_loss_pct

    if config.enable_spread_exits is not None:
        updates["enable_spread_exits"] = config.enable_spread_exits

    # AI: Resolution analysis gate
    if config.ai_resolution_gate is not None:
        updates["ai_resolution_gate"] = config.ai_resolution_gate

    if config.ai_max_resolution_risk is not None:
        updates["ai_max_resolution_risk"] = config.ai_max_resolution_risk

    if config.ai_min_resolution_clarity is not None:
        updates["ai_min_resolution_clarity"] = config.ai_min_resolution_clarity

    if config.ai_resolution_block_avoid is not None:
        updates["ai_resolution_block_avoid"] = config.ai_resolution_block_avoid

    if config.ai_resolution_model is not None:
        updates["ai_resolution_model"] = config.ai_resolution_model

    if config.ai_skip_on_analysis_failure is not None:
        updates["ai_skip_on_analysis_failure"] = config.ai_skip_on_analysis_failure

    # AI: Opportunity judge position sizing
    if config.ai_position_sizing is not None:
        updates["ai_position_sizing"] = config.ai_position_sizing

    if config.ai_min_score_to_trade is not None:
        updates["ai_min_score_to_trade"] = config.ai_min_score_to_trade

    if config.ai_score_size_multiplier is not None:
        updates["ai_score_size_multiplier"] = config.ai_score_size_multiplier

    if config.ai_score_boost_threshold is not None:
        updates["ai_score_boost_threshold"] = config.ai_score_boost_threshold

    if config.ai_score_boost_multiplier is not None:
        updates["ai_score_boost_multiplier"] = config.ai_score_boost_multiplier

    if config.ai_judge_model is not None:
        updates["ai_judge_model"] = config.ai_judge_model

    # LLM verification before trading
    if config.llm_verify_trades is not None:
        updates["llm_verify_trades"] = config.llm_verify_trades

    if config.llm_verify_strategies is not None:
        updates["llm_verify_strategies"] = config.llm_verify_strategies

    # Scanner auto AI scoring (applied to scanner singleton, not auto_trader config)
    if config.auto_ai_scoring is not None:
        from services.scanner import scanner

        scanner.set_auto_ai_scoring(config.auto_ai_scoring)

    auto_trader.configure(**updates)

    return {"status": "updated", "config": auto_trader.get_config()}


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
async def enable_live_trading(confirm: bool = False, max_daily_loss: float = 100.0):
    """
    Enable live trading mode.

    Requires explicit confirmation.
    Sets conservative default limits.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to enable live trading. This will trade REAL MONEY.",
        )

    # Check if trading service can be initialized
    from services.trading import trading_service

    if not trading_service.is_ready():
        success = await trading_service.initialize()
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to initialize trading service. Check credentials.",
            )

    # Set conservative limits
    auto_trader.configure(
        mode=AutoTraderMode.LIVE,
        max_daily_loss_usd=max_daily_loss,
        require_confirmation=False,  # Can be set to True for extra safety
    )

    if not auto_trader._running:
        await auto_trader.start()

    logger.warning("LIVE TRADING ENABLED")

    return {
        "status": "live_trading_enabled",
        "warning": "Auto trader is now executing REAL trades",
        "max_daily_loss": max_daily_loss,
        "config": auto_trader.get_config(),
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
        "message": "All automated trading has been stopped",
    }
