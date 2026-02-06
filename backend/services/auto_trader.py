"""
Autonomous Trading Engine

This service runs in the background and automatically executes trades
based on detected arbitrage opportunities.

Features:
- Continuously monitors opportunities from the scanner
- Automatically executes trades that meet criteria
- Manages position sizing and risk
- Tracks performance and adjusts strategy
- Implements safety limits and circuit breakers

IMPORTANT: This executes REAL trades with REAL money.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
import uuid

from config import settings
from services.trading import trading_service, OrderStatus
from services.scanner import scanner
from models import ArbitrageOpportunity, StrategyType
from utils.logger import get_logger

logger = get_logger(__name__)


class AutoTraderMode(str, Enum):
    """Operating modes for auto trader"""

    DISABLED = "disabled"  # No automatic trading
    PAPER = "paper"  # Simulation only (uses simulation service)
    LIVE = "live"  # Real trading
    SHADOW = "shadow"  # Track what would be traded but don't execute


@dataclass
class AutoTraderConfig:
    """Configuration for automatic trading"""

    mode: AutoTraderMode = AutoTraderMode.DISABLED

    # Strategy filters
    enabled_strategies: list[StrategyType] = field(
        default_factory=lambda: [
            StrategyType.BASIC,
            StrategyType.NEGRISK,
            StrategyType.MUTUALLY_EXCLUSIVE,
            StrategyType.MUST_HAPPEN,
            StrategyType.MIRACLE,
            StrategyType.SETTLEMENT_LAG,
        ]
    )

    # Entry criteria
    min_roi_percent: float = 2.5  # Minimum ROI to trade
    max_risk_score: float = 0.5  # Maximum acceptable risk
    min_liquidity_usd: float = 5000.0  # Minimum market liquidity
    min_impossibility_score: float = 0.8  # For miracle strategy

    # Profit guarantee thresholds (from article Proposition 4.1)
    min_guaranteed_profit: float = 0.05  # $0.05 min from research (gas + slippage)
    use_profit_guarantee: bool = True  # Enable Proposition 4.1 filtering

    # Position sizing
    base_position_size_usd: float = 10.0  # Base position size
    max_position_size_usd: float = 100.0  # Maximum per trade
    position_size_method: str = "fixed"  # fixed, kelly, volatility_adjusted
    paper_account_capital: float = 10000.0  # Starting capital for paper trading

    # Risk management
    max_daily_trades: int = 50  # Maximum trades per day
    max_daily_loss_usd: float = 100.0  # Stop trading if daily loss exceeds
    max_concurrent_positions: int = 10  # Maximum open positions
    cooldown_after_loss_seconds: int = 60  # Wait after losing trade

    # Execution
    execution_delay_seconds: float = 0.0  # Delay before executing (0 for speed)
    require_confirmation: bool = False  # Require manual confirmation
    auto_retry_failed: bool = True  # Retry failed orders

    # Circuit breakers
    circuit_breaker_losses: int = 3  # Pause after N consecutive losses
    circuit_breaker_duration_minutes: int = 30


@dataclass
class TradeRecord:
    """Record of an automatic trade"""

    id: str
    opportunity_id: str
    strategy: StrategyType
    executed_at: datetime
    positions: list[dict]
    total_cost: float
    expected_profit: float
    actual_profit: Optional[float] = None
    guaranteed_profit: Optional[float] = None  # From Proposition 4.1
    capture_ratio: Optional[float] = None  # % of max profit captured
    mispricing_type: Optional[str] = None  # within_market, cross_market, settlement_lag
    status: str = "pending"  # pending, open, resolved_win, resolved_loss, failed
    order_ids: list[str] = field(default_factory=list)
    mode: AutoTraderMode = AutoTraderMode.PAPER


@dataclass
class AutoTraderStats:
    """Statistics for auto trader"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_invested: float = 0.0

    daily_trades: int = 0
    daily_profit: float = 0.0
    daily_invested: float = 0.0

    consecutive_losses: int = 0
    last_trade_at: Optional[datetime] = None
    circuit_breaker_until: Optional[datetime] = None

    opportunities_seen: int = 0
    opportunities_skipped: int = 0
    opportunities_executed: int = 0


class AutoTrader:
    """
    Autonomous trading engine.

    Monitors the scanner for opportunities and automatically
    executes trades based on configured criteria.
    """

    def __init__(self):
        self.config = AutoTraderConfig()
        self.stats = AutoTraderStats()
        self._running = False
        self._trades: dict[str, TradeRecord] = {}
        self._processed_opportunities: set[str] = set()
        self._callbacks: list[Callable] = []
        self._daily_reset_date = datetime.utcnow().date()

    def configure(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Auto trader config updated: {key}={value}")

    def add_callback(self, callback: Callable):
        """Add callback for trade events"""
        self._callbacks.append(callback)

    async def _notify_callbacks(self, event: str, data: dict):
        """Notify all callbacks of an event"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.utcnow().date()
        if today != self._daily_reset_date:
            self.stats.daily_trades = 0
            self.stats.daily_profit = 0.0
            self.stats.daily_invested = 0.0
            self._daily_reset_date = today
            logger.info("Daily stats reset")

    def _check_circuit_breaker(self) -> tuple[bool, str]:
        """Check if circuit breaker is active"""
        self._check_daily_reset()

        # Check circuit breaker timeout
        if self.stats.circuit_breaker_until:
            if datetime.utcnow() < self.stats.circuit_breaker_until:
                remaining = (
                    self.stats.circuit_breaker_until - datetime.utcnow()
                ).seconds
                return False, f"Circuit breaker active ({remaining}s remaining)"
            else:
                self.stats.circuit_breaker_until = None
                self.stats.consecutive_losses = 0
                logger.info("Circuit breaker reset")

        # Check consecutive losses
        if self.stats.consecutive_losses >= self.config.circuit_breaker_losses:
            self.stats.circuit_breaker_until = datetime.utcnow() + timedelta(
                minutes=self.config.circuit_breaker_duration_minutes
            )
            logger.warning(
                f"Circuit breaker triggered: {self.stats.consecutive_losses} consecutive losses"
            )
            return (
                False,
                f"Circuit breaker triggered after {self.stats.consecutive_losses} losses",
            )

        # Check daily limits
        if self.stats.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        if self.stats.daily_profit < -self.config.max_daily_loss_usd:
            return (
                False,
                f"Daily loss limit exceeded (${abs(self.stats.daily_profit):.2f})",
            )

        return True, ""

    def _should_trade_opportunity(self, opp: ArbitrageOpportunity) -> tuple[bool, str]:
        """Determine if an opportunity should be traded"""

        # Already processed
        if opp.id in self._processed_opportunities:
            return False, "Already processed"

        # Check mode
        if self.config.mode == AutoTraderMode.DISABLED:
            return False, "Auto trading disabled"

        # Check circuit breaker
        can_trade, reason = self._check_circuit_breaker()
        if not can_trade:
            return False, reason

        # Check strategy filter
        if opp.strategy not in self.config.enabled_strategies:
            return False, f"Strategy {opp.strategy.value} not enabled"

        # Check ROI threshold
        if opp.roi_percent < self.config.min_roi_percent:
            return False, f"ROI {opp.roi_percent:.2f}% below threshold"

        # Check risk score
        if opp.risk_score > self.config.max_risk_score:
            return False, f"Risk score {opp.risk_score:.2f} above threshold"

        # Check liquidity
        if opp.min_liquidity < self.config.min_liquidity_usd:
            return False, f"Liquidity ${opp.min_liquidity:.0f} below threshold"

        # For miracle strategy, check impossibility score
        if opp.strategy == StrategyType.MIRACLE:
            # Extract impossibility score from description
            # Format: "... | Impossibility: XX%"
            import re

            match = re.search(r"Impossibility: (\d+)%", opp.description)
            if match:
                impossibility = int(match.group(1)) / 100
                if impossibility < self.config.min_impossibility_score:
                    return False, f"Impossibility {impossibility:.0%} below threshold"

        # Check profit guarantee (Proposition 4.1 from article)
        # If the opportunity has a guaranteed_profit from Frank-Wolfe, use it
        if self.config.use_profit_guarantee and opp.guaranteed_profit is not None:
            if opp.guaranteed_profit < self.config.min_guaranteed_profit:
                return False, (
                    f"Guaranteed profit ${opp.guaranteed_profit:.4f} below "
                    f"threshold ${self.config.min_guaranteed_profit:.4f}"
                )

        # Check max concurrent positions
        open_positions = len([t for t in self._trades.values() if t.status == "open"])
        if open_positions >= self.config.max_concurrent_positions:
            return (
                False,
                f"Max concurrent positions ({self.config.max_concurrent_positions}) reached",
            )

        return True, "Opportunity meets criteria"

    def _calculate_position_size(self, opp: ArbitrageOpportunity) -> float:
        """
        Calculate position size with execution risk adjustment.

        Research paper insight: Modified Kelly accounting for execution risk:
        f* = (b×p - q) / b × √p

        Where p is EXECUTION probability, not win probability.
        Arbitrage has ~100% win rate IF executed correctly, so the key
        risk is partial execution / slippage.
        """
        if self.config.position_size_method == "fixed":
            size = self.config.base_position_size_usd

        elif self.config.position_size_method == "kelly":
            # Estimate execution probability from liquidity
            # Research: only count opportunities with >= $0.05 margin
            if opp.min_liquidity < 1000:
                exec_prob = 0.5
            elif opp.min_liquidity < 5000:
                exec_prob = 0.75
            elif opp.min_liquidity < 20000:
                exec_prob = 0.9
            else:
                exec_prob = 0.95

            # Reduce exec_prob based on number of legs (more legs = more risk)
            num_positions = len(opp.positions_to_take) if opp.positions_to_take else 1
            if num_positions > 2:
                exec_prob *= 0.95 ** (num_positions - 2)

            expected_return = opp.roi_percent / 100

            if expected_return > 0:
                # Modified Kelly with execution risk
                # Standard: f = (bp - q) / b where q = 1 - p
                # Modified: f = (bp - q) / b × √p (conservative adjustment)
                q = 1 - exec_prob
                standard_kelly = (expected_return * exec_prob - q) / expected_return

                # Apply √p adjustment for execution uncertainty
                adjusted_kelly = standard_kelly * (exec_prob**0.5)
                kelly_fraction = max(0, min(adjusted_kelly, 0.25))
            else:
                kelly_fraction = 0

            size = self.config.max_position_size_usd * kelly_fraction

            # Minimum viable: must exceed fees by enough to be worth it
            min_viable = 0.02 * 10  # At least 10x the 2% fee
            if size < min_viable and size > 0:
                size = 0  # Not worth executing

        else:
            size = self.config.base_position_size_usd

        # Apply limits
        size = max(size, settings.MIN_ORDER_SIZE_USD) if size > 0 else 0
        size = min(size, self.config.max_position_size_usd)
        size = min(size, opp.max_position_size)  # Don't exceed 10% of liquidity

        return size

    async def _execute_trade(self, opp: ArbitrageOpportunity) -> TradeRecord:
        """Execute a trade for an opportunity"""
        trade_id = str(uuid.uuid4())
        position_size = self._calculate_position_size(opp)

        trade = TradeRecord(
            id=trade_id,
            opportunity_id=opp.id,
            strategy=opp.strategy,
            executed_at=datetime.utcnow(),
            positions=opp.positions_to_take,
            total_cost=position_size,
            expected_profit=position_size * (opp.roi_percent / 100),
            guaranteed_profit=opp.guaranteed_profit,
            capture_ratio=opp.capture_ratio,
            mispricing_type=opp.mispricing_type.value if opp.mispricing_type else None,
            mode=self.config.mode,
        )

        guarantee_str = ""
        if opp.guaranteed_profit is not None:
            guarantee_str = f" | Guaranteed: ${opp.guaranteed_profit:.4f}"
            if opp.capture_ratio is not None:
                guarantee_str += f" ({opp.capture_ratio:.0%} capture)"

        mispricing_str = ""
        if opp.mispricing_type:
            mispricing_str = f" | Type: {opp.mispricing_type.value}"

        logger.info(
            f"Executing trade: {opp.strategy.value} | "
            f"ROI: {opp.roi_percent:.2f}% | "
            f"Size: ${position_size:.2f} | "
            f"Mode: {self.config.mode.value}"
            f"{guarantee_str}{mispricing_str}"
        )

        if self.config.mode == AutoTraderMode.LIVE:
            # Real trading
            if not trading_service.is_ready():
                trade.status = "failed"
                logger.error("Trading service not ready")
                return trade

            # Add execution delay
            if self.config.execution_delay_seconds > 0:
                await asyncio.sleep(self.config.execution_delay_seconds)

            # Execute orders
            orders = await trading_service.execute_opportunity(
                opportunity_id=opp.id,
                positions=opp.positions_to_take,
                size_usd=position_size,
            )

            trade.order_ids = [o.id for o in orders]

            # Check if all orders succeeded
            failed_orders = [o for o in orders if o.status == OrderStatus.FAILED]
            if failed_orders:
                trade.status = "failed"
                logger.error(f"Trade failed: {len(failed_orders)} orders failed")
            else:
                trade.status = "open"

        elif self.config.mode == AutoTraderMode.PAPER:
            # Simulation mode
            from services.simulation import simulation_service

            # Get or create a simulation account for auto trading
            accounts = await simulation_service.get_all_accounts()
            auto_trader_account = None
            for acc in accounts:
                if acc.name == "Auto Trader":
                    auto_trader_account = acc
                    break

            if not auto_trader_account:
                auto_trader_account = await simulation_service.create_account(
                    name="Auto Trader",
                    initial_capital=self.config.paper_account_capital,
                )

            account = auto_trader_account

            # Execute in simulation
            sim_trade = await simulation_service.execute_opportunity(
                account_id=account.id, opportunity=opp, position_size=position_size
            )

            if sim_trade:
                trade.status = "open"
                trade.order_ids = [sim_trade.id]
            else:
                trade.status = "failed"

        elif self.config.mode == AutoTraderMode.SHADOW:
            # Shadow mode - just record what would happen
            trade.status = "shadow"
            logger.info(f"Shadow trade recorded: {opp.title}")

        # Update stats
        self.stats.total_trades += 1
        self.stats.daily_trades += 1
        self.stats.total_invested += position_size
        self.stats.daily_invested += position_size
        self.stats.last_trade_at = datetime.utcnow()
        self.stats.opportunities_executed += 1

        # Store trade
        self._trades[trade_id] = trade
        self._processed_opportunities.add(opp.id)

        # Notify callbacks
        await self._notify_callbacks(
            "trade_executed", {"trade": trade, "opportunity": opp}
        )

        return trade

    async def _process_opportunities(self, opportunities: list[ArbitrageOpportunity]):
        """Process a batch of opportunities"""
        self.stats.opportunities_seen += len(opportunities)

        for opp in opportunities:
            should_trade, reason = self._should_trade_opportunity(opp)

            if should_trade:
                logger.info(f"Auto trading opportunity: {opp.title[:50]}...")

                if self.config.require_confirmation:
                    # Would need to implement confirmation mechanism
                    logger.info("Confirmation required - skipping")
                    continue

                try:
                    await self._execute_trade(opp)
                except Exception as e:
                    logger.error(f"Trade execution error: {e}")
                    self.stats.opportunities_skipped += 1
            else:
                self.stats.opportunities_skipped += 1
                if reason != "Already processed":
                    logger.debug(f"Skipping opportunity: {reason}")

    async def start(self):
        """Start the auto trader"""
        if self._running:
            logger.warning("Auto trader already running")
            return

        self._running = True
        logger.info(f"Auto trader started in {self.config.mode.value} mode")

        # Register callback with scanner
        scanner.add_callback(self._on_scan_complete)

        # If in live mode, initialize trading service
        if self.config.mode == AutoTraderMode.LIVE:
            if not trading_service.is_ready():
                success = await trading_service.initialize()
                if not success:
                    logger.error(
                        "Failed to initialize trading service, falling back to paper mode"
                    )
                    self.config.mode = AutoTraderMode.PAPER

    async def _on_scan_complete(self, opportunities: list[ArbitrageOpportunity]):
        """Callback when scanner completes a scan"""
        if not self._running:
            return

        if self.config.mode == AutoTraderMode.DISABLED:
            return

        await self._process_opportunities(opportunities)

    def stop(self):
        """Stop the auto trader"""
        self._running = False
        logger.info("Auto trader stopped")

    def record_trade_result(self, trade_id: str, won: bool, actual_profit: float):
        """Record the result of a resolved trade"""
        if trade_id not in self._trades:
            return

        trade = self._trades[trade_id]
        trade.actual_profit = actual_profit
        trade.status = "resolved_win" if won else "resolved_loss"

        # Update stats
        self.stats.total_profit += actual_profit
        self.stats.daily_profit += actual_profit

        if won:
            self.stats.winning_trades += 1
            self.stats.consecutive_losses = 0
        else:
            self.stats.losing_trades += 1
            self.stats.consecutive_losses += 1

        logger.info(
            f"Trade resolved: {'WIN' if won else 'LOSS'} | "
            f"Profit: ${actual_profit:.2f} | "
            f"Total P/L: ${self.stats.total_profit:.2f}"
        )

    def get_stats(self) -> dict:
        """Get auto trader statistics"""
        self._check_daily_reset()

        win_rate = 0.0
        if self.stats.winning_trades + self.stats.losing_trades > 0:
            win_rate = self.stats.winning_trades / (
                self.stats.winning_trades + self.stats.losing_trades
            )

        return {
            "mode": self.config.mode.value,
            "running": self._running,
            "total_trades": self.stats.total_trades,
            "winning_trades": self.stats.winning_trades,
            "losing_trades": self.stats.losing_trades,
            "win_rate": win_rate,
            "total_profit": self.stats.total_profit,
            "total_invested": self.stats.total_invested,
            "roi_percent": (self.stats.total_profit / self.stats.total_invested * 100)
            if self.stats.total_invested > 0
            else 0,
            "daily_trades": self.stats.daily_trades,
            "daily_profit": self.stats.daily_profit,
            "consecutive_losses": self.stats.consecutive_losses,
            "circuit_breaker_active": self.stats.circuit_breaker_until is not None,
            "last_trade_at": self.stats.last_trade_at.isoformat()
            if self.stats.last_trade_at
            else None,
            "opportunities_seen": self.stats.opportunities_seen,
            "opportunities_executed": self.stats.opportunities_executed,
            "opportunities_skipped": self.stats.opportunities_skipped,
        }

    def get_config(self) -> dict:
        """Get current configuration"""
        return {
            "mode": self.config.mode.value,
            "enabled_strategies": [s.value for s in self.config.enabled_strategies],
            "min_roi_percent": self.config.min_roi_percent,
            "max_risk_score": self.config.max_risk_score,
            "min_liquidity_usd": self.config.min_liquidity_usd,
            "base_position_size_usd": self.config.base_position_size_usd,
            "max_position_size_usd": self.config.max_position_size_usd,
            "max_daily_trades": self.config.max_daily_trades,
            "max_daily_loss_usd": self.config.max_daily_loss_usd,
            "circuit_breaker_losses": self.config.circuit_breaker_losses,
            "require_confirmation": self.config.require_confirmation,
            "paper_account_capital": self.config.paper_account_capital,
            "min_guaranteed_profit": self.config.min_guaranteed_profit,
            "use_profit_guarantee": self.config.use_profit_guarantee,
        }

    def get_trades(self, limit: int = 100) -> list[dict]:
        """Get recent trades"""
        trades = sorted(
            self._trades.values(), key=lambda t: t.executed_at, reverse=True
        )[:limit]

        return [
            {
                "id": t.id,
                "opportunity_id": t.opportunity_id,
                "strategy": t.strategy.value,
                "executed_at": t.executed_at.isoformat(),
                "total_cost": t.total_cost,
                "expected_profit": t.expected_profit,
                "actual_profit": t.actual_profit,
                "guaranteed_profit": t.guaranteed_profit,
                "capture_ratio": t.capture_ratio,
                "mispricing_type": t.mispricing_type,
                "status": t.status,
                "mode": t.mode.value,
            }
            for t in trades
        ]


# Singleton instance
auto_trader = AutoTrader()
