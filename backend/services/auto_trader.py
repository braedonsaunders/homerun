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
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
import uuid

from config import settings
from services.trading import trading_service, OrderStatus
from services.scanner import scanner
from services.depth_analyzer import depth_analyzer
from services.token_circuit_breaker import token_circuit_breaker
from services.execution_tiers import execution_tier_service
from services.category_buffers import category_buffer_service
from models import ArbitrageOpportunity, StrategyType
from utils.logger import get_logger

logger = get_logger(__name__)


class AutoTraderMode(str, Enum):
    """Operating modes for auto trader"""

    DISABLED = "disabled"  # No automatic trading
    PAPER = "paper"  # Simulation only (uses simulation service)
    LIVE = "live"  # Real trading
    SHADOW = "shadow"  # Track what would be traded but don't execute
    MOCK = "mock"  # Full pipeline with simulated execution


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

    # Settlement time filtering
    max_end_date_days: Optional[int] = 30  # Skip markets settling more than N days out (None = no limit)
    min_end_date_days: Optional[int] = None  # Skip markets settling sooner than N days (None = no limit)
    prefer_near_settlement: bool = True  # Boost score for markets settling sooner

    # Opportunity prioritization
    priority_method: str = "composite"  # "roi", "annualized_roi", "composite"
    settlement_weight: float = 0.3  # Weight for settlement proximity in composite score (0-1)
    roi_weight: float = 0.5  # Weight for ROI in composite score (0-1)
    liquidity_weight: float = 0.1  # Weight for liquidity in composite score (0-1)
    risk_weight: float = 0.1  # Weight for (inverse) risk in composite score (0-1)

    # Event concentration limits
    max_trades_per_event: int = 3  # Max trades on markets within same event
    max_exposure_per_event_usd: float = 50.0  # Max total $ exposure per event

    # Exclusion filters
    excluded_categories: list[str] = field(default_factory=list)  # e.g. ["Politics"]
    excluded_keywords: list[str] = field(default_factory=list)  # e.g. ["presidential", "2028"]
    excluded_event_slugs: list[str] = field(default_factory=list)  # e.g. ["will-donald-trump-win"]

    # Volume filter
    min_volume_usd: float = 0.0  # Minimum market trading volume

    # AI: Resolution analysis gate (Option B)
    ai_resolution_gate: bool = True  # Require resolution analysis before trading
    ai_max_resolution_risk: float = 0.5  # Block if resolution risk_score exceeds this
    ai_min_resolution_clarity: float = 0.5  # Block if clarity_score below this
    ai_resolution_block_avoid: bool = True  # Hard block if recommendation is "avoid"
    ai_resolution_model: Optional[str] = None  # LLM model override (None = default gpt-4o-mini)
    ai_skip_on_analysis_failure: bool = False  # If True, skip trade when analysis fails; if False, allow trade through

    # AI: Opportunity judge position sizing (Option C)
    ai_position_sizing: bool = True  # Use AI judge score to scale position sizes
    ai_min_score_to_trade: float = 0.0  # Hard floor: skip if overall_score below this (0 = no floor)
    ai_score_size_multiplier: bool = True  # Scale position size by AI score (e.g. 0.8 score = 80% size)
    ai_score_boost_threshold: float = 0.85  # Boost size by 1.2x if score exceeds this
    ai_score_boost_multiplier: float = 1.2  # How much to boost high-confidence trades
    ai_judge_model: Optional[str] = None  # LLM model override (None = default gpt-4o-mini)


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
    event_id: Optional[str] = None
    days_to_settlement: Optional[float] = None
    composite_score: Optional[float] = None
    ai_resolution_recommendation: Optional[str] = None  # safe/caution/avoid
    ai_resolution_risk: Optional[float] = None
    ai_judge_score: Optional[float] = None  # overall_score from opportunity judge
    ai_judge_recommendation: Optional[str] = None  # strong_execute/execute/review/skip/strong_skip
    ai_size_adjustment: Optional[float] = None  # multiplier applied from AI scoring


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
        self._executing_opportunities: set[str] = set()  # Currently being executed
        self._execution_lock = asyncio.Lock()
        self._callbacks: list[Callable] = []
        self._daily_reset_date = datetime.utcnow().date()

        # AI integration caches
        self._resolution_cache: dict[str, dict] = {}  # market_id -> resolution analysis
        self._judge_cache: dict[str, dict] = {}  # opportunity_id -> judge result
        self._ai_available: Optional[bool] = None  # Lazily checked

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

    def _days_until_settlement(self, opp: ArbitrageOpportunity) -> Optional[float]:
        """Calculate days until market settlement/resolution.

        Returns None if no end/resolution date is available.
        """
        end_date = opp.resolution_date
        if end_date is None:
            return None
        now = datetime.utcnow()
        if end_date <= now:
            return 0.0
        return (end_date - now).total_seconds() / 86400.0

    def _annualized_roi(self, roi_percent: float, days_to_settlement: Optional[float]) -> float:
        """Convert raw ROI to annualized ROI based on settlement time.

        A 5% ROI in 7 days is much better than 5% in 365 days.
        If settlement date is unknown, returns raw ROI as-is.
        """
        if days_to_settlement is None or days_to_settlement <= 0:
            return roi_percent
        # Annualize: (1 + roi)^(365/days) - 1, capped to avoid absurd values
        periods_per_year = 365.0 / max(days_to_settlement, 0.5)
        try:
            annualized = (math.pow(1 + roi_percent / 100, periods_per_year) - 1) * 100
        except (OverflowError, ValueError):
            annualized = 999999.0
        return min(annualized, 999999.0)  # Cap at a reasonable max

    def _settlement_score(self, days_to_settlement: Optional[float]) -> float:
        """Score settlement proximity from 0.0 (far away) to 1.0 (imminent).

        Uses exponential decay: markets settling within a week score ~1.0,
        markets settling in 30+ days score low, 365+ days score near 0.
        """
        if days_to_settlement is None:
            return 0.3  # Unknown settlement gets a neutral-low score
        if days_to_settlement <= 0:
            return 1.0  # Already past resolution
        # Exponential decay with half-life of ~14 days
        return math.exp(-days_to_settlement / 20.0)

    def _composite_score(self, opp: ArbitrageOpportunity) -> float:
        """Calculate a composite priority score for an opportunity.

        Combines ROI, settlement proximity, liquidity, and risk into a
        single score using configurable weights. Higher is better.
        """
        days = self._days_until_settlement(opp)

        # ROI component (normalized: 10% ROI -> 1.0 score)
        if self.config.priority_method == "annualized_roi":
            roi_score = min(self._annualized_roi(opp.roi_percent, days) / 10.0, 10.0)
        else:
            roi_score = min(opp.roi_percent / 10.0, 10.0)

        # Settlement component
        settle_score = self._settlement_score(days)

        # Liquidity component (normalized: $50k -> 1.0 score)
        liq_score = min(opp.min_liquidity / 50000.0, 1.0)

        # Risk component (inverted: low risk = high score)
        risk_score = 1.0 - opp.risk_score

        composite = (
            self.config.roi_weight * roi_score
            + self.config.settlement_weight * settle_score
            + self.config.liquidity_weight * liq_score
            + self.config.risk_weight * risk_score
        )

        return composite

    def _get_event_trade_count(self, event_id: Optional[str]) -> int:
        """Count how many open/pending trades exist for a given event."""
        if not event_id:
            return 0
        return sum(
            1 for t in self._trades.values()
            if t.status in ("open", "pending") and t.event_id == event_id
        )

    def _get_event_exposure(self, event_id: Optional[str]) -> float:
        """Sum total cost of open/pending trades for a given event."""
        if not event_id:
            return 0.0
        return sum(
            t.total_cost for t in self._trades.values()
            if t.status in ("open", "pending") and t.event_id == event_id
        )

    def _is_ai_available(self) -> bool:
        """Check if the AI subsystem is initialized and usable."""
        if self._ai_available is not None:
            return self._ai_available
        try:
            from services.ai import get_llm_manager
            manager = get_llm_manager()
            self._ai_available = manager.is_available()
        except Exception:
            self._ai_available = False
        return self._ai_available

    async def _get_resolution_analysis(self, opp: ArbitrageOpportunity) -> Optional[dict]:
        """Get resolution analysis for an opportunity's markets, using cache.

        Checks the in-memory cache first, then the DB-level 24h cache inside
        the resolution analyzer. Only the first market's ID is used as the
        cache key since most opportunities have a primary market.

        Returns the aggregate analysis dict, or None if analysis fails / AI unavailable.
        """
        if not self._is_ai_available():
            return None

        # Use event_id or first market id as cache key
        cache_key = opp.event_id
        if not cache_key and opp.markets:
            cache_key = opp.markets[0].get("id") or opp.markets[0].get("condition_id", "")
        if not cache_key:
            return None

        # Check in-memory cache
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]

        try:
            from services.ai.resolution_analyzer import resolution_analyzer

            result = await resolution_analyzer.analyze_opportunity_markets(
                opportunity=opp,
                model=self.config.ai_resolution_model,
            )
            self._resolution_cache[cache_key] = result
            logger.info(
                f"Resolution analysis for {opp.title[:40]}: "
                f"clarity={result.get('overall_clarity', '?'):.2f}, "
                f"risk={result.get('overall_risk', '?'):.2f}, "
                f"rec={result.get('overall_recommendation', '?')}"
            )
            return result
        except Exception as e:
            logger.error(f"Resolution analysis failed for {cache_key}: {e}")
            return None

    async def _get_ai_judgment(self, opp: ArbitrageOpportunity, resolution_analysis: Optional[dict] = None) -> Optional[dict]:
        """Get AI judge score for an opportunity, using cache.

        Returns the judgment dict, or None if unavailable.
        """
        if not self._is_ai_available():
            return None

        # Check in-memory cache
        if opp.id in self._judge_cache:
            return self._judge_cache[opp.id]

        try:
            from services.ai.opportunity_judge import opportunity_judge

            result = await opportunity_judge.judge_opportunity(
                opportunity=opp,
                resolution_analysis=resolution_analysis,
                model=self.config.ai_judge_model,
            )
            self._judge_cache[opp.id] = result
            logger.info(
                f"AI judgment for {opp.title[:40]}: "
                f"score={result.get('overall_score', '?'):.2f}, "
                f"rec={result.get('recommendation', '?')}"
            )
            return result
        except Exception as e:
            logger.error(f"AI judgment failed for {opp.id}: {e}")
            return None

    def _matches_exclusion_keywords(self, opp: ArbitrageOpportunity) -> Optional[str]:
        """Check if opportunity matches any exclusion keyword. Returns matched keyword or None."""
        if not self.config.excluded_keywords:
            return None
        text = f"{opp.title} {opp.description} {opp.event_title or ''}".lower()
        for kw in self.config.excluded_keywords:
            if kw.lower() in text:
                return kw
        return None

    def _matches_exclusion_slug(self, opp: ArbitrageOpportunity) -> Optional[str]:
        """Check if opportunity's event slug matches exclusion list."""
        if not self.config.excluded_event_slugs:
            return None
        for market in (opp.markets or []):
            slug = market.get("slug", "")
            for excl_slug in self.config.excluded_event_slugs:
                if excl_slug.lower() in slug.lower():
                    return excl_slug
        return None

    async def _should_trade_opportunity(self, opp: ArbitrageOpportunity) -> tuple[bool, str]:
        """Determine if an opportunity should be traded.

        This is async because the resolution analysis gate requires an LLM call
        (cached after first call per market).
        """

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

        # Check per-token circuit breaker (trip mechanism)
        if opp.positions_to_take:
            for pos in opp.positions_to_take:
                token_id = pos.get("token_id", "")
                if token_id:
                    is_tripped, trip_reason = token_circuit_breaker.is_tripped(token_id)
                    if is_tripped:
                        return False, f"Token tripped: {trip_reason}"

        # Check strategy filter
        if opp.strategy not in self.config.enabled_strategies:
            return False, f"Strategy {opp.strategy.value} not enabled"

        # === EXCLUSION FILTERS ===

        # Category exclusion
        category = getattr(opp, "category", None)
        if category and self.config.excluded_categories:
            if category.lower() in [c.lower() for c in self.config.excluded_categories]:
                return False, f"Category '{category}' is excluded"

        # Keyword exclusion
        matched_kw = self._matches_exclusion_keywords(opp)
        if matched_kw:
            return False, f"Matches excluded keyword '{matched_kw}'"

        # Slug exclusion
        matched_slug = self._matches_exclusion_slug(opp)
        if matched_slug:
            return False, f"Matches excluded slug '{matched_slug}'"

        # === SETTLEMENT TIME FILTERS ===

        days_to_settle = self._days_until_settlement(opp)

        # Filter out markets that settle too far in the future
        if self.config.max_end_date_days is not None and days_to_settle is not None:
            if days_to_settle > self.config.max_end_date_days:
                return (
                    False,
                    f"Settles in {days_to_settle:.0f} days, max allowed is {self.config.max_end_date_days}",
                )

        # Filter out markets that settle too soon (if configured)
        if self.config.min_end_date_days is not None and days_to_settle is not None:
            if days_to_settle < self.config.min_end_date_days:
                return (
                    False,
                    f"Settles in {days_to_settle:.0f} days, min required is {self.config.min_end_date_days}",
                )

        # === STANDARD FILTERS ===

        # Check ROI threshold
        if opp.roi_percent < self.config.min_roi_percent:
            return False, f"ROI {opp.roi_percent:.2f}% below threshold"

        # Check risk score
        if opp.risk_score > self.config.max_risk_score:
            return False, f"Risk score {opp.risk_score:.2f} above threshold"

        # Apply category-specific liquidity adjustments
        adjusted_min_liquidity = (
            category_buffer_service.adjust_min_liquidity(
                self.config.min_liquidity_usd, category
            )
            if category
            else self.config.min_liquidity_usd
        )

        # Check liquidity with category adjustment
        if opp.min_liquidity < adjusted_min_liquidity:
            return (
                False,
                f"Liquidity ${opp.min_liquidity:.0f} below threshold (${adjusted_min_liquidity:.0f} for {category or 'default'})",
            )

        # Check volume filter
        if self.config.min_volume_usd > 0:
            opp_volume = sum(m.get("volume", 0) for m in (opp.markets or []))
            if opp_volume < self.config.min_volume_usd:
                return False, f"Volume ${opp_volume:.0f} below threshold ${self.config.min_volume_usd:.0f}"

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

        # === EVENT CONCENTRATION LIMITS ===

        event_id = opp.event_id
        if event_id and self.config.max_trades_per_event > 0:
            event_trades = self._get_event_trade_count(event_id)
            if event_trades >= self.config.max_trades_per_event:
                return (
                    False,
                    f"Event already has {event_trades} trades (max {self.config.max_trades_per_event})",
                )

        if event_id and self.config.max_exposure_per_event_usd > 0:
            event_exposure = self._get_event_exposure(event_id)
            if event_exposure >= self.config.max_exposure_per_event_usd:
                return (
                    False,
                    f"Event exposure ${event_exposure:.2f} exceeds max ${self.config.max_exposure_per_event_usd:.2f}",
                )

        # Check max concurrent positions
        open_positions = len([t for t in self._trades.values() if t.status == "open"])
        if open_positions >= self.config.max_concurrent_positions:
            return (
                False,
                f"Max concurrent positions ({self.config.max_concurrent_positions}) reached",
            )

        # === AI RESOLUTION ANALYSIS GATE (Option B) ===
        # Run after all cheap filters pass to minimize LLM costs.
        # Results are cached per market (24h TTL in analyzer + in-memory here).

        if self.config.ai_resolution_gate and self._is_ai_available():
            analysis = await self._get_resolution_analysis(opp)

            if analysis is None:
                # AI unavailable or analysis failed
                if self.config.ai_skip_on_analysis_failure:
                    return False, "Resolution analysis unavailable (configured to skip)"
                # else: allow trade through (fail-open)
            else:
                rec = analysis.get("overall_recommendation", "caution")
                risk = analysis.get("overall_risk", 0.5)
                clarity = analysis.get("overall_clarity", 0.5)

                # Hard block on "avoid" recommendation
                if self.config.ai_resolution_block_avoid and rec == "avoid":
                    return (
                        False,
                        f"AI resolution analysis: AVOID (clarity={clarity:.2f}, risk={risk:.2f})",
                    )

                # Check risk threshold
                if risk > self.config.ai_max_resolution_risk:
                    return (
                        False,
                        f"AI resolution risk {risk:.2f} exceeds max {self.config.ai_max_resolution_risk}",
                    )

                # Check clarity threshold
                if clarity < self.config.ai_min_resolution_clarity:
                    return (
                        False,
                        f"AI resolution clarity {clarity:.2f} below min {self.config.ai_min_resolution_clarity}",
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
            MIN_CASH_VALUE = 1.01  # From terauss settings
            min_viable = max(
                0.02 * 10, MIN_CASH_VALUE
            )  # At least 10x the 2% fee or min cash value
            if size < min_viable and size > 0:
                probability = size / min_viable
                if random.random() < probability:
                    size = min_viable  # Execute at minimum size
                    logger.info(
                        f"Probabilistic execution: size below minimum, executing at ${min_viable:.2f} (prob={probability:.2f})"
                    )
                else:
                    size = 0  # Skip this time
                    logger.debug(
                        f"Probabilistic skip: size below minimum (prob={probability:.2f})"
                    )

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

        # Classify into execution tier for price buffers and retry config
        category = getattr(opp, "category", None)
        tier = execution_tier_service.classify_opportunity(
            roi_percent=opp.roi_percent,
            liquidity=opp.min_liquidity,
            strategy=opp.strategy.value,
            category=category,
        )

        # Apply category-specific position size adjustment
        position_size = self._calculate_position_size(opp)
        if category:
            position_size = category_buffer_service.adjust_position_size(
                position_size, category
            )
        position_size *= tier.size_multiplier

        # === AI OPPORTUNITY JUDGE POSITION SIZING (Option C) ===
        # Run the AI judge to get a quality score, then adjust position size.
        # This is non-blocking in the sense that if AI is unavailable, we
        # proceed with the base size.
        ai_judge_result = None
        ai_size_adjustment = 1.0

        if self.config.ai_position_sizing and self._is_ai_available():
            # Pass resolution analysis if we already have it from the gate
            cache_key = opp.event_id
            if not cache_key and opp.markets:
                cache_key = opp.markets[0].get("id") or opp.markets[0].get("condition_id", "")
            resolution_data = self._resolution_cache.get(cache_key) if cache_key else None

            ai_judge_result = await self._get_ai_judgment(opp, resolution_analysis=resolution_data)

            if ai_judge_result:
                ai_score = ai_judge_result.get("overall_score", 0.5)
                ai_rec = ai_judge_result.get("recommendation", "review")

                # Hard floor: skip if score too low
                if self.config.ai_min_score_to_trade > 0 and ai_score < self.config.ai_min_score_to_trade:
                    logger.info(
                        f"AI judge blocked trade: score {ai_score:.2f} < min {self.config.ai_min_score_to_trade}"
                    )
                    trade = TradeRecord(
                        id=trade_id,
                        opportunity_id=opp.id,
                        strategy=opp.strategy,
                        executed_at=datetime.utcnow(),
                        positions=opp.positions_to_take,
                        total_cost=0,
                        expected_profit=0,
                        status="failed",
                        mode=self.config.mode,
                        ai_judge_score=ai_score,
                        ai_judge_recommendation=ai_rec,
                    )
                    self._trades[trade_id] = trade
                    self._processed_opportunities.add(opp.id)
                    return trade

                # Scale position size by AI score
                if self.config.ai_score_size_multiplier:
                    ai_size_adjustment = ai_score  # 0.8 score = 80% of base size

                    # Boost high-confidence trades
                    if ai_score >= self.config.ai_score_boost_threshold:
                        ai_size_adjustment *= self.config.ai_score_boost_multiplier

                    position_size *= ai_size_adjustment
                    logger.info(
                        f"AI size adjustment: {ai_size_adjustment:.2f}x "
                        f"(score={ai_score:.2f}, rec={ai_rec})"
                    )

        # Apply limits after tier/category/AI adjustments
        position_size = min(position_size, self.config.max_position_size_usd)
        position_size = (
            max(position_size, settings.MIN_ORDER_SIZE_USD) if position_size > 0 else 0
        )

        # Depth check for each position leg (only for live/paper/mock trades)
        if self.config.mode in (
            AutoTraderMode.LIVE,
            AutoTraderMode.PAPER,
            AutoTraderMode.MOCK,
        ):
            if opp.positions_to_take:
                for pos in opp.positions_to_take:
                    token_id = pos.get("token_id", "")
                    if not token_id:
                        continue
                    price = pos.get("price", 0)
                    try:
                        depth_result = await depth_analyzer.check_depth(
                            token_id=token_id,
                            side="BUY",
                            target_price=price + tier.price_buffer,
                            required_size_usd=position_size
                            / max(len(opp.positions_to_take), 1),
                            trade_context="auto_trader",
                        )
                        if not depth_result.has_sufficient_depth:
                            # Trip the token on insufficient depth
                            token_circuit_breaker.trip_token(
                                token_id,
                                "insufficient_depth",
                                {"available": depth_result.available_depth_usd},
                            )
                            logger.warning(
                                "Insufficient depth, blocking trade",
                                token_id=token_id,
                                available=depth_result.available_depth_usd,
                            )
                            trade = TradeRecord(
                                id=trade_id,
                                opportunity_id=opp.id,
                                strategy=opp.strategy,
                                executed_at=datetime.utcnow(),
                                positions=opp.positions_to_take,
                                total_cost=0,
                                expected_profit=0,
                                status="failed",
                                mode=self.config.mode,
                            )
                            self._trades[trade_id] = trade
                            self._processed_opportunities.add(opp.id)
                            return trade

                        # After depth check succeeds, use VWAP price if available
                        if depth_result and depth_result.vwap_price > 0:
                            # Use depth-aware executable price instead of raw price
                            pos["effective_price"] = depth_result.vwap_price
                    except Exception as e:
                        logger.error(f"Depth check failed: {e}")

        # Record tier assignment for analytics
        try:
            await execution_tier_service.record_tier_assignment(
                tier=tier,
                roi_percent=opp.roi_percent,
                liquidity=opp.min_liquidity,
                strategy=opp.strategy.value,
                category=category,
                opportunity_id=opp.id,
            )
        except Exception as e:
            logger.error(f"Failed to record tier assignment: {e}")

        days = self._days_until_settlement(opp)

        # Collect resolution analysis info for the trade record
        res_cache_key = opp.event_id
        if not res_cache_key and opp.markets:
            res_cache_key = opp.markets[0].get("id") or opp.markets[0].get("condition_id", "")
        res_data = self._resolution_cache.get(res_cache_key) if res_cache_key else None

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
            event_id=opp.event_id,
            days_to_settlement=days,
            composite_score=self._composite_score(opp),
            ai_resolution_recommendation=res_data.get("overall_recommendation") if res_data else None,
            ai_resolution_risk=res_data.get("overall_risk") if res_data else None,
            ai_judge_score=ai_judge_result.get("overall_score") if ai_judge_result else None,
            ai_judge_recommendation=ai_judge_result.get("recommendation") if ai_judge_result else None,
            ai_size_adjustment=ai_size_adjustment if ai_size_adjustment != 1.0 else None,
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

        elif self.config.mode == AutoTraderMode.MOCK:
            # Run full real pipeline but simulate the fill
            logger.info(
                f"MOCK execution: would place {len(opp.positions_to_take)} orders, size=${position_size:.2f}"
            )
            trade.status = "open"
            trade.order_ids = [
                f"mock_{uuid.uuid4().hex[:8]}" for _ in (opp.positions_to_take or [])
            ]

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
        """Process a batch of opportunities, sorted by priority method."""
        self.stats.opportunities_seen += len(opportunities)

        # Sort opportunities by configured priority method
        if self.config.priority_method == "composite":
            sorted_opps = sorted(
                opportunities,
                key=lambda o: self._composite_score(o),
                reverse=True,
            )
        elif self.config.priority_method == "annualized_roi":
            sorted_opps = sorted(
                opportunities,
                key=lambda o: self._annualized_roi(
                    o.roi_percent, self._days_until_settlement(o)
                ),
                reverse=True,
            )
        else:
            # Default: raw ROI
            sorted_opps = sorted(
                opportunities,
                key=lambda o: o.roi_percent,
                reverse=True,
            )

        for opp in sorted_opps:
            should_trade, reason = await self._should_trade_opportunity(opp)

            if should_trade:
                days = self._days_until_settlement(opp)
                settle_str = f" | Settles: {days:.0f}d" if days is not None else ""
                score_str = ""
                if self.config.priority_method == "composite":
                    score_str = f" | Score: {self._composite_score(opp):.3f}"
                elif self.config.priority_method == "annualized_roi":
                    ann_roi = self._annualized_roi(opp.roi_percent, days)
                    score_str = f" | Ann.ROI: {ann_roi:.1f}%"

                logger.info(
                    f"Auto trading opportunity: {opp.title[:50]}...{settle_str}{score_str}"
                )

                if self.config.require_confirmation:
                    # Would need to implement confirmation mechanism
                    logger.info("Confirmation required - skipping")
                    continue

                # In-flight deduplication: prevent concurrent execution of same opportunity
                async with self._execution_lock:
                    if opp.id in self._executing_opportunities:
                        continue  # Already being executed concurrently
                    self._executing_opportunities.add(opp.id)

                try:
                    await self._execute_trade(opp)
                except Exception as e:
                    logger.error(f"Trade execution error: {e}")
                    self.stats.opportunities_skipped += 1
                finally:
                    self._executing_opportunities.discard(opp.id)
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
            # Settlement time
            "max_end_date_days": self.config.max_end_date_days,
            "min_end_date_days": self.config.min_end_date_days,
            "prefer_near_settlement": self.config.prefer_near_settlement,
            # Priority/scoring
            "priority_method": self.config.priority_method,
            "settlement_weight": self.config.settlement_weight,
            "roi_weight": self.config.roi_weight,
            "liquidity_weight": self.config.liquidity_weight,
            "risk_weight": self.config.risk_weight,
            # Event concentration
            "max_trades_per_event": self.config.max_trades_per_event,
            "max_exposure_per_event_usd": self.config.max_exposure_per_event_usd,
            # Exclusions
            "excluded_categories": self.config.excluded_categories,
            "excluded_keywords": self.config.excluded_keywords,
            "excluded_event_slugs": self.config.excluded_event_slugs,
            # Volume
            "min_volume_usd": self.config.min_volume_usd,
            # AI: Resolution analysis gate
            "ai_resolution_gate": self.config.ai_resolution_gate,
            "ai_max_resolution_risk": self.config.ai_max_resolution_risk,
            "ai_min_resolution_clarity": self.config.ai_min_resolution_clarity,
            "ai_resolution_block_avoid": self.config.ai_resolution_block_avoid,
            "ai_resolution_model": self.config.ai_resolution_model,
            "ai_skip_on_analysis_failure": self.config.ai_skip_on_analysis_failure,
            # AI: Opportunity judge position sizing
            "ai_position_sizing": self.config.ai_position_sizing,
            "ai_min_score_to_trade": self.config.ai_min_score_to_trade,
            "ai_score_size_multiplier": self.config.ai_score_size_multiplier,
            "ai_score_boost_threshold": self.config.ai_score_boost_threshold,
            "ai_score_boost_multiplier": self.config.ai_score_boost_multiplier,
            "ai_judge_model": self.config.ai_judge_model,
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
                "event_id": t.event_id,
                "days_to_settlement": t.days_to_settlement,
                "composite_score": t.composite_score,
                "ai_resolution_recommendation": t.ai_resolution_recommendation,
                "ai_resolution_risk": t.ai_resolution_risk,
                "ai_judge_score": t.ai_judge_score,
                "ai_judge_recommendation": t.ai_judge_recommendation,
                "ai_size_adjustment": t.ai_size_adjustment,
            }
            for t in trades
        ]


# Singleton instance
auto_trader = AutoTrader()
