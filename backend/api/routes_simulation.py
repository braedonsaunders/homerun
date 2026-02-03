from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field

from services.simulation import simulation_service
from services.scanner import scanner
from utils.validation import validate_positive_number

simulation_router = APIRouter()


class CreateAccountRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    initial_capital: float = Field(default=10000.0, ge=100.0, le=10000000.0)
    max_position_pct: float = Field(default=10.0, ge=1.0, le=100.0)
    max_positions: int = Field(default=10, ge=1, le=100)


class ExecuteTradeRequest(BaseModel):
    opportunity_id: str
    position_size: Optional[float] = Field(default=None, ge=10.0)


# ==================== ACCOUNTS ====================

@simulation_router.post("/accounts")
async def create_simulation_account(request: CreateAccountRequest):
    """Create a new simulation account for paper trading"""
    account = await simulation_service.create_account(
        name=request.name,
        initial_capital=request.initial_capital,
        max_position_pct=request.max_position_pct,
        max_positions=request.max_positions
    )
    return {
        "account_id": account.id,
        "name": account.name,
        "initial_capital": account.initial_capital,
        "message": "Simulation account created successfully"
    }


@simulation_router.get("/accounts")
async def list_simulation_accounts():
    """List all simulation accounts"""
    accounts = await simulation_service.get_all_accounts()
    return [{
        "id": acc.id,
        "name": acc.name,
        "initial_capital": acc.initial_capital,
        "current_capital": acc.current_capital,
        "total_pnl": acc.total_pnl,
        "total_trades": acc.total_trades,
        "win_rate": acc.winning_trades / acc.total_trades * 100 if acc.total_trades > 0 else 0
    } for acc in accounts]


@simulation_router.get("/accounts/{account_id}")
async def get_simulation_account(account_id: str):
    """Get detailed simulation account information"""
    stats = await simulation_service.get_account_stats(account_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Account not found")
    return stats


@simulation_router.get("/accounts/{account_id}/positions")
async def get_account_positions(account_id: str):
    """Get open positions for a simulation account"""
    positions = await simulation_service.get_open_positions(account_id)
    return [{
        "id": pos.id,
        "market_id": pos.market_id,
        "market_question": pos.market_question,
        "side": pos.side.value,
        "quantity": pos.quantity,
        "entry_price": pos.entry_price,
        "entry_cost": pos.entry_cost,
        "current_price": pos.current_price,
        "unrealized_pnl": pos.unrealized_pnl,
        "opened_at": pos.opened_at.isoformat(),
        "status": pos.status.value
    } for pos in positions]


@simulation_router.get("/accounts/{account_id}/trades")
async def get_account_trades(
    account_id: str,
    limit: int = Query(default=50, ge=1, le=500)
):
    """Get trade history for a simulation account"""
    trades = await simulation_service.get_trade_history(account_id, limit)
    return [{
        "id": trade.id,
        "opportunity_id": trade.opportunity_id,
        "strategy_type": trade.strategy_type,
        "total_cost": trade.total_cost,
        "expected_profit": trade.expected_profit,
        "slippage": trade.slippage,
        "status": trade.status.value,
        "actual_payout": trade.actual_payout,
        "actual_pnl": trade.actual_pnl,
        "fees_paid": trade.fees_paid,
        "executed_at": trade.executed_at.isoformat(),
        "resolved_at": trade.resolved_at.isoformat() if trade.resolved_at else None,
        "copied_from": trade.copied_from_wallet
    } for trade in trades]


# ==================== TRADING ====================

@simulation_router.post("/accounts/{account_id}/execute")
async def execute_opportunity(
    account_id: str,
    request: ExecuteTradeRequest
):
    """Execute an arbitrage opportunity in simulation"""
    # Find the opportunity
    opportunities = scanner.get_opportunities()
    opportunity = next(
        (o for o in opportunities if o.id == request.opportunity_id),
        None
    )

    if not opportunity:
        raise HTTPException(
            status_code=404,
            detail=f"Opportunity not found: {request.opportunity_id}"
        )

    try:
        trade = await simulation_service.execute_opportunity(
            account_id=account_id,
            opportunity=opportunity,
            position_size=request.position_size
        )

        return {
            "trade_id": trade.id,
            "status": trade.status.value,
            "total_cost": trade.total_cost,
            "expected_profit": trade.expected_profit,
            "slippage": trade.slippage,
            "message": "Trade executed successfully in simulation"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@simulation_router.post("/trades/{trade_id}/resolve")
async def resolve_trade(
    trade_id: str,
    winning_outcome: str = Query(..., description="The outcome that won (YES or NO)")
):
    """Manually resolve a simulated trade (for testing)"""
    try:
        trade = await simulation_service.resolve_trade(
            trade_id=trade_id,
            winning_outcome=winning_outcome
        )

        return {
            "trade_id": trade.id,
            "status": trade.status.value,
            "actual_payout": trade.actual_payout,
            "actual_pnl": trade.actual_pnl,
            "fees_paid": trade.fees_paid,
            "message": f"Trade resolved as {trade.status.value}"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== PERFORMANCE ====================

@simulation_router.get("/accounts/{account_id}/performance")
async def get_account_performance(account_id: str):
    """Get detailed performance metrics for an account"""
    stats = await simulation_service.get_account_stats(account_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Account not found")

    trades = await simulation_service.get_trade_history(account_id, 1000)

    # Calculate additional metrics
    profits = [t.actual_pnl for t in trades if t.actual_pnl and t.actual_pnl > 0]
    losses = [t.actual_pnl for t in trades if t.actual_pnl and t.actual_pnl < 0]

    avg_win = sum(profits) / len(profits) if profits else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Calculate max drawdown
    cumulative_pnl = 0
    peak = 0
    max_drawdown = 0
    for trade in reversed(trades):
        if trade.actual_pnl:
            cumulative_pnl += trade.actual_pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return {
        **stats,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(sum(profits) / sum(losses)) if losses else 0,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": (max_drawdown / stats["initial_capital"]) * 100 if stats["initial_capital"] > 0 else 0
    }
