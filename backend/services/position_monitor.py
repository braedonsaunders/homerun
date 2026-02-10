"""
Position Monitor Service - Spread Trading Exit Strategies

Monitors open positions and automatically exits when:
- Price reaches take-profit target
- Price drops to stop-loss level
- Configurable trailing stop-loss

This is the key service that enables "trading the spread" vs holding to resolution.
Instead of waiting for market resolution, positions are bought and sold based
on price movement, just like stock trading.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    SimulationPosition,
    SimulationTrade,
    SimulationAccount,
    TradeStatus,
    AsyncSessionLocal,
)
from utils.logger import get_logger

logger = get_logger("position_monitor")


class PositionMonitor:
    """
    Background service that monitors open positions and exits based on price targets.

    Supports:
    - Take profit: Sell when current price >= take_profit_price
    - Stop loss: Sell when current price <= stop_loss_price
    - Works for both simulation and live positions
    """

    def __init__(self):
        self._running = False
        self._poll_interval = 15  # seconds (WS cache makes this cheap)
        self._positions_checked = 0
        self._exits_triggered = 0
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the position monitoring loop"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Position monitor started", poll_interval=self._poll_interval)

    def stop(self):
        """Stop the position monitoring loop"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Position monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_positions()
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            await asyncio.sleep(self._poll_interval)

    async def _check_positions(self):
        """Check all open positions with price targets"""
        async with AsyncSessionLocal() as session:
            # Get all open positions that have take_profit or stop_loss set
            result = await session.execute(
                select(SimulationPosition).where(
                    SimulationPosition.status == TradeStatus.OPEN,
                )
            )
            positions = list(result.scalars().all())

            # Filter to only positions with price targets
            monitored = [
                p
                for p in positions
                if p.take_profit_price is not None or p.stop_loss_price is not None
            ]

            if not monitored:
                return

            # Get current prices for all monitored token_ids
            token_ids = list(set(p.token_id for p in monitored if p.token_id))
            if not token_ids:
                return

            prices = await self._fetch_prices(token_ids)
            self._positions_checked += len(monitored)

            for position in monitored:
                if not position.token_id or position.token_id not in prices:
                    continue

                current_price = prices[position.token_id]
                if current_price <= 0:
                    continue

                # Update stored current price
                position.current_price = current_price
                position.unrealized_pnl = (
                    current_price - position.entry_price
                ) * position.quantity

                # Check take profit
                if (
                    position.take_profit_price is not None
                    and current_price >= position.take_profit_price
                ):
                    logger.info(
                        "Take profit triggered",
                        position_id=position.id,
                        entry_price=position.entry_price,
                        current_price=current_price,
                        take_profit=position.take_profit_price,
                    )
                    await self._exit_position(
                        session, position, current_price, "take_profit"
                    )
                    continue

                # Check stop loss
                if (
                    position.stop_loss_price is not None
                    and current_price <= position.stop_loss_price
                ):
                    logger.info(
                        "Stop loss triggered",
                        position_id=position.id,
                        entry_price=position.entry_price,
                        current_price=current_price,
                        stop_loss=position.stop_loss_price,
                    )
                    await self._exit_position(
                        session, position, current_price, "stop_loss"
                    )
                    continue

            await session.commit()

    async def _fetch_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Fetch current prices, preferring WS cache over HTTP.

        Uses the real-time WS price cache for tokens that have fresh data,
        and falls back to HTTP batch fetch only for stale/missing tokens.
        This avoids unnecessary API calls when WS feeds are healthy.
        """
        prices: dict[str, float] = {}
        stale_ids: list[str] = []

        # Try WS cache first
        try:
            from config import settings as app_settings

            if app_settings.WS_FEED_ENABLED:
                from services.ws_feeds import get_feed_manager

                feed_mgr = get_feed_manager()
                if feed_mgr._started:
                    for tid in token_ids:
                        if feed_mgr.is_fresh(tid):
                            mid = feed_mgr.cache.get_mid_price(tid)
                            if mid is not None and mid > 0:
                                prices[tid] = mid
                                continue
                        stale_ids.append(tid)
                else:
                    stale_ids = list(token_ids)
            else:
                stale_ids = list(token_ids)
        except Exception:
            stale_ids = list(token_ids)

        # HTTP fallback for stale/missing tokens only
        if stale_ids:
            try:
                from services.polymarket import polymarket_client

                http_prices = await polymarket_client.get_prices_batch(stale_ids)
                for tid, data in http_prices.items():
                    prices[tid] = (
                        data.get("mid", 0) if isinstance(data, dict) else float(data)
                    )
            except Exception as e:
                logger.error(f"Failed to fetch prices via HTTP: {e}")

        if prices and stale_ids:
            ws_count = len(token_ids) - len(stale_ids)
            logger.debug(
                "Position prices: %d from WS cache, %d from HTTP",
                ws_count,
                len(stale_ids),
            )

        return prices

    async def _exit_position(
        self,
        session: AsyncSession,
        position: SimulationPosition,
        exit_price: float,
        reason: str,
    ):
        """Exit a position at the given price"""
        pnl = (exit_price - position.entry_price) * position.quantity
        is_win = pnl > 0

        # Apply 2% Polymarket fee on winnings
        fee = 0.0
        if pnl > 0:
            fee = pnl * 0.02
            pnl -= fee

        # Update position
        position.status = (
            TradeStatus.RESOLVED_WIN if is_win else TradeStatus.RESOLVED_LOSS
        )
        position.current_price = exit_price
        position.unrealized_pnl = 0  # Now realized

        # Find and update the associated trade
        if position.opportunity_id:
            trade_result = await session.execute(
                select(SimulationTrade).where(
                    SimulationTrade.opportunity_id == position.opportunity_id,
                    SimulationTrade.account_id == position.account_id,
                    SimulationTrade.status == TradeStatus.OPEN,
                )
            )
            trade = trade_result.scalar_one_or_none()
            if trade:
                trade.status = (
                    TradeStatus.RESOLVED_WIN if is_win else TradeStatus.RESOLVED_LOSS
                )
                trade.actual_pnl = pnl
                trade.actual_payout = position.entry_cost + pnl
                trade.fees_paid = fee
                trade.resolved_at = datetime.now(timezone.utc)

        # Update account balance
        account = await session.get(SimulationAccount, position.account_id)
        if account:
            # Return the exit proceeds to the account
            exit_proceeds = exit_price * position.quantity - fee
            account.current_capital += exit_proceeds
            account.total_pnl += pnl
            if is_win:
                account.winning_trades += 1
            else:
                account.losing_trades += 1

        self._exits_triggered += 1
        logger.info(
            f"Position exited ({reason})",
            position_id=position.id,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            is_win=is_win,
        )

    def get_status(self) -> dict:
        """Get monitor status"""
        return {
            "running": self._running,
            "poll_interval": self._poll_interval,
            "positions_checked": self._positions_checked,
            "exits_triggered": self._exits_triggered,
        }


# Singleton
position_monitor = PositionMonitor()
