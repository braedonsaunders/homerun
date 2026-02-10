import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    SimulationAccount,
    SimulationPosition,
    SimulationTrade,
    TradeStatus,
    PositionSide,
    AsyncSessionLocal,
)
from models.opportunity import ArbitrageOpportunity
from utils.logger import get_logger

logger = get_logger("simulation")


class SlippageModel:
    """Calculate execution slippage"""

    @staticmethod
    def fixed(base_price: float, slippage_bps: float) -> float:
        """Fixed slippage in basis points"""
        return base_price * (1 + slippage_bps / 10000)

    @staticmethod
    def linear(
        base_price: float, size: float, liquidity: float, slippage_bps: float
    ) -> float:
        """Linear slippage based on size vs liquidity"""
        impact = (size / liquidity) * slippage_bps / 10000
        return base_price * (1 + impact)

    @staticmethod
    def sqrt(
        base_price: float, size: float, liquidity: float, slippage_bps: float
    ) -> float:
        """Square root slippage (more realistic for large orders)"""
        impact = (size / liquidity) ** 0.5 * slippage_bps / 10000
        return base_price * (1 + impact)


class SimulationService:
    """Paper trading simulation service"""

    POLYMARKET_FEE = 0.02  # 2% winner fee

    async def create_account(
        self,
        name: str,
        initial_capital: float = 10000.0,
        max_position_pct: float = 10.0,
        max_positions: int = 10,
    ) -> SimulationAccount:
        """Create a new simulation account"""
        async with AsyncSessionLocal() as session:
            account = SimulationAccount(
                id=str(uuid.uuid4()),
                name=name,
                initial_capital=initial_capital,
                current_capital=initial_capital,
                max_position_size_pct=max_position_pct,
                max_open_positions=max_positions,
            )
            session.add(account)
            await session.commit()
            await session.refresh(account)

            logger.info(
                "Created simulation account",
                account_id=account.id,
                name=name,
                capital=initial_capital,
            )

            return account

    async def get_account(self, account_id: str) -> Optional[SimulationAccount]:
        """Get simulation account by ID"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(SimulationAccount).where(SimulationAccount.id == account_id)
            )
            return result.scalar_one_or_none()

    async def get_all_accounts(self) -> list[SimulationAccount]:
        """Get all simulation accounts"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(SimulationAccount))
            return list(result.scalars().all())

    async def delete_account(self, account_id: str) -> bool:
        """Delete a simulation account and all related records"""
        async with AsyncSessionLocal() as session:
            # Check if account exists
            account = await session.get(SimulationAccount, account_id)
            if not account:
                return False

            # Delete related positions
            await session.execute(
                select(SimulationPosition).where(
                    SimulationPosition.account_id == account_id
                )
            )
            positions = await session.execute(
                select(SimulationPosition).where(
                    SimulationPosition.account_id == account_id
                )
            )
            for pos in positions.scalars():
                await session.delete(pos)

            # Delete related trades
            trades = await session.execute(
                select(SimulationTrade).where(SimulationTrade.account_id == account_id)
            )
            for trade in trades.scalars():
                await session.delete(trade)

            # Delete the account
            await session.delete(account)
            await session.commit()

            logger.info(
                "Deleted simulation account", account_id=account_id, name=account.name
            )

            return True

    async def execute_opportunity(
        self,
        account_id: str,
        opportunity: ArbitrageOpportunity,
        position_size: Optional[float] = None,
        copied_from: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> SimulationTrade:
        """Execute an arbitrage opportunity in simulation"""
        async with AsyncSessionLocal() as session:
            # Get account
            account = await session.get(SimulationAccount, account_id)
            if not account:
                raise ValueError(f"Account not found: {account_id}")

            # Calculate position size
            if position_size is None:
                max_size = account.current_capital * (
                    account.max_position_size_pct / 100
                )
                position_size = min(max_size, opportunity.max_position_size)

            if position_size > account.current_capital:
                raise ValueError(
                    f"Insufficient capital: {account.current_capital} < {position_size}"
                )

            # Calculate total cost with slippage
            base_cost = opportunity.total_cost * position_size
            slippage = self._calculate_slippage(
                account.slippage_model,
                base_cost,
                position_size,
                opportunity.min_liquidity,
                account.slippage_bps,
            )
            total_cost = base_cost + slippage

            # Create trade record
            trade = SimulationTrade(
                id=str(uuid.uuid4()),
                account_id=account_id,
                opportunity_id=opportunity.id,
                strategy_type=opportunity.strategy,
                positions_data=opportunity.positions_to_take,
                total_cost=total_cost,
                expected_profit=opportunity.net_profit * position_size,
                slippage=slippage,
                status=TradeStatus.OPEN,
                copied_from_wallet=copied_from,
            )
            session.add(trade)

            # Create position records
            for pos in opportunity.positions_to_take:
                position = SimulationPosition(
                    id=str(uuid.uuid4()),
                    account_id=account_id,
                    opportunity_id=opportunity.id,
                    market_id=pos.get("market", ""),
                    market_question=pos.get("market", ""),
                    token_id=pos.get("token_id"),
                    side=PositionSide.YES
                    if pos.get("outcome") == "YES"
                    else PositionSide.NO,
                    quantity=position_size,
                    entry_price=pos.get("price", 0),
                    entry_cost=pos.get("price", 0) * position_size,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    resolution_date=opportunity.resolution_date,
                    status=TradeStatus.OPEN,
                )
                session.add(position)

            # Update account balance
            account.current_capital -= total_cost
            account.total_trades += 1

            await session.commit()
            await session.refresh(trade)

            logger.info(
                "Executed simulation trade",
                account_id=account_id,
                trade_id=trade.id,
                opportunity_id=opportunity.id,
                position_size=position_size,
                total_cost=total_cost,
                slippage=slippage,
            )

            return trade

    async def resolve_trade(
        self,
        trade_id: str,
        winning_outcome: str,  # Which outcome won
        session: AsyncSession = None,
    ) -> SimulationTrade:
        """Resolve a trade when market settles"""
        should_close = session is None
        if session is None:
            session = AsyncSessionLocal()

        try:
            trade = await session.get(SimulationTrade, trade_id)
            if not trade:
                raise ValueError(f"Trade not found: {trade_id}")

            if trade.status != TradeStatus.OPEN:
                raise ValueError(f"Trade already resolved: {trade.status}")

            account = await session.get(SimulationAccount, trade.account_id)

            # Calculate payout
            payout = 0.0
            for pos in trade.positions_data:
                if pos.get("outcome") == winning_outcome:
                    # This position won
                    payout += 1.0 * (trade.total_cost / len(trade.positions_data))

            # Apply fee on winnings
            fee = payout * self.POLYMARKET_FEE if payout > 0 else 0
            net_payout = payout - fee

            # Calculate PnL
            pnl = net_payout - trade.total_cost
            is_win = pnl > 0

            # Update trade
            trade.status = (
                TradeStatus.RESOLVED_WIN if is_win else TradeStatus.RESOLVED_LOSS
            )
            trade.actual_payout = net_payout
            trade.actual_pnl = pnl
            trade.fees_paid = fee
            trade.resolved_at = datetime.utcnow()

            # Update account
            account.current_capital += net_payout
            account.total_pnl += pnl
            if is_win:
                account.winning_trades += 1
            else:
                account.losing_trades += 1

            # Close positions
            positions = await session.execute(
                select(SimulationPosition).where(
                    SimulationPosition.opportunity_id == trade.opportunity_id
                )
            )
            for pos in positions.scalars():
                pos.status = (
                    TradeStatus.RESOLVED_WIN if is_win else TradeStatus.RESOLVED_LOSS
                )

            await session.commit()
            await session.refresh(trade)

            logger.info(
                "Resolved simulation trade",
                trade_id=trade_id,
                winning_outcome=winning_outcome,
                payout=net_payout,
                pnl=pnl,
                is_win=is_win,
            )

            return trade

        finally:
            if should_close:
                await session.close()

    async def get_open_positions(self, account_id: str) -> list[SimulationPosition]:
        """Get all open positions for an account"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(SimulationPosition).where(
                    SimulationPosition.account_id == account_id,
                    SimulationPosition.status == TradeStatus.OPEN,
                )
            )
            return list(result.scalars().all())

    async def get_trade_history(
        self, account_id: str, limit: int = 100
    ) -> list[SimulationTrade]:
        """Get trade history for an account"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(SimulationTrade)
                .where(SimulationTrade.account_id == account_id)
                .order_by(SimulationTrade.executed_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_account_stats(self, account_id: str) -> dict:
        """Get comprehensive stats for an account"""
        async with AsyncSessionLocal() as session:
            account = await session.get(SimulationAccount, account_id)
            if not account:
                return None

            # Calculate additional stats
            win_rate = (
                account.winning_trades / account.total_trades * 100
                if account.total_trades > 0
                else 0
            )
            roi = (
                (account.current_capital - account.initial_capital)
                / account.initial_capital
                * 100
            )

            # Get open positions count
            positions = await self.get_open_positions(account_id)

            return {
                "account_id": account.id,
                "name": account.name,
                "initial_capital": account.initial_capital,
                "current_capital": account.current_capital,
                "total_pnl": account.total_pnl,
                "roi_percent": roi,
                "total_trades": account.total_trades,
                "winning_trades": account.winning_trades,
                "losing_trades": account.losing_trades,
                "win_rate": win_rate,
                "open_positions": len(positions),
                "max_positions": account.max_open_positions,
                "created_at": account.created_at.isoformat(),
            }

    def _calculate_slippage(
        self,
        model: str,
        base_cost: float,
        size: float,
        liquidity: float,
        slippage_bps: float,
    ) -> float:
        """Calculate slippage based on model"""
        if liquidity <= 0:
            liquidity = 10000  # Default if unknown

        if model == "linear":
            factor = 1 + (size / liquidity) * slippage_bps / 10000
        elif model == "sqrt":
            factor = 1 + (size / liquidity) ** 0.5 * slippage_bps / 10000
        else:  # fixed
            factor = 1 + slippage_bps / 10000

        return base_cost * (factor - 1)


# Singleton instance
simulation_service = SimulationService()
