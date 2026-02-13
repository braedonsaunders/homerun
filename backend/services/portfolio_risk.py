"""
Portfolio-Level Risk Management with Kelly Criterion Position Sizing

Prevents concentrated exposure in correlated markets by tracking portfolio state,
enforcing exposure limits per category/event, and sizing positions using the
Kelly criterion (half-Kelly by default for conservatism).

Correlated positions are detected via shared event IDs, categories, and
overlapping market IDs, and their recommended sizes are discounted accordingly.
"""

from dataclasses import dataclass
from datetime import datetime
from utils.utcnow import utcnow
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger("portfolio_risk")


# ==================== DATA CLASSES ====================


@dataclass
class Position:
    """A currently open position in the portfolio."""

    opportunity_id: str
    market_ids: list[str]
    event_id: Optional[str]
    category: Optional[str]
    size_usd: float
    entry_time: datetime
    expected_roi: float
    risk_score: float


@dataclass
class PortfolioState:
    """Snapshot of the current portfolio."""

    positions: list[Position]
    total_exposure_usd: float
    exposure_by_category: dict[str, float]
    exposure_by_event: dict[str, float]
    num_positions: int


# ==================== KELLY CRITERION ====================


class KellyCriterion:
    """Kelly criterion position sizing for arbitrage opportunities.

    For arbitrage the "win" is a successful fill that captures the spread,
    and the "loss" accounts for partial-fill risk or adverse price movement.
    A maximum fraction cap (default 5% of bankroll) prevents over-betting on
    any single trade regardless of what the formula outputs.
    """

    MAX_KELLY_FRACTION: float = 0.05  # Never bet more than 5% of bankroll

    @staticmethod
    def calculate_kelly_fraction(win_prob: float, win_amount: float, loss_amount: float) -> float:
        """Classic Kelly formula: f* = (p * b - q) / b

        where b = win_amount / loss_amount, p = win_prob, q = 1 - p.

        Returns the fraction of bankroll to wager. Clamped to [0, 1].
        """
        if loss_amount <= 0 or win_amount <= 0 or win_prob <= 0:
            return 0.0
        if win_prob >= 1.0:
            return 1.0

        b = win_amount / loss_amount
        q = 1.0 - win_prob
        fraction = (win_prob * b - q) / b

        return max(0.0, min(fraction, 1.0))

    @staticmethod
    def calculate_half_kelly(win_prob: float, win_amount: float, loss_amount: float) -> float:
        """Half-Kelly: more conservative, reduces variance by ~75% while
        capturing ~75% of the growth rate of full Kelly."""
        full = KellyCriterion.calculate_kelly_fraction(win_prob, win_amount, loss_amount)
        return full / 2.0

    @classmethod
    def optimal_position_size(
        cls,
        bankroll: float,
        opportunity,  # ArbitrageOpportunity (avoid circular import)
    ) -> float:
        """Calculate the optimal position size in USD for an arbitrage opportunity.

        For arbitrage specifically:
        - win_prob is estimated from the risk_score (lower risk = higher fill
          probability).  A risk_score of 0 maps to ~95% fill probability; a
          risk_score of 1 maps to ~50%.
        - win_amount is the net_profit per dollar of cost (i.e. roi_percent / 100).
        - loss_amount accounts for partial-fill risk: we assume worst case we
          lose the fee plus slippage (~2-3% of trade cost).
        """
        if bankroll <= 0:
            return 0.0

        # Estimate fill probability from risk score
        # risk_score 0 -> 0.95, risk_score 1 -> 0.50 (linear interpolation)
        win_prob = 0.95 - 0.45 * min(max(opportunity.risk_score, 0.0), 1.0)

        # Win amount per dollar wagered
        win_amount = max(opportunity.roi_percent / 100.0, 0.0)

        # Loss amount per dollar wagered: fees + assumed slippage
        loss_amount = max(opportunity.fee / opportunity.total_cost, 0.02) + 0.01
        if opportunity.total_cost > 0:
            loss_amount = max(loss_amount, 0.02)

        fraction = cls.calculate_half_kelly(win_prob, win_amount, loss_amount)

        # Apply maximum fraction cap
        fraction = min(fraction, cls.MAX_KELLY_FRACTION)

        size = bankroll * fraction

        # Floor at minimum order size, cap at opportunity's max position
        size = max(size, 0.0)
        if opportunity.max_position_size > 0:
            size = min(size, opportunity.max_position_size)

        logger.debug(
            "Kelly sizing computed",
            bankroll=bankroll,
            win_prob=round(win_prob, 3),
            win_amount=round(win_amount, 4),
            loss_amount=round(loss_amount, 4),
            fraction=round(fraction, 5),
            size_usd=round(size, 2),
        )

        return round(size, 2)


# ==================== PORTFOLIO RISK MANAGER ====================


class PortfolioRiskManager:
    """Manages portfolio-level risk across all open arbitrage positions.

    Enforces exposure limits per category, per event, and overall.  Detects
    correlation between a new opportunity and existing positions to reduce
    recommended sizes.  Provides a risk report with concentration metrics
    and worst-case analysis.
    """

    # Default exposure limits (USD)
    DEFAULT_MAX_TOTAL_EXPOSURE: float = 5000.0
    DEFAULT_MAX_CATEGORY_EXPOSURE: float = 2000.0
    DEFAULT_MAX_EVENT_EXPOSURE: float = 1000.0
    LIQUIDITY_FRACTION_CAP: float = 0.10  # Don't exceed 10% of market liquidity

    def __init__(
        self,
        max_total_exposure: float = DEFAULT_MAX_TOTAL_EXPOSURE,
        max_category_exposure: float = DEFAULT_MAX_CATEGORY_EXPOSURE,
        max_event_exposure: float = DEFAULT_MAX_EVENT_EXPOSURE,
    ):
        self._positions: dict[str, Position] = {}
        self.max_total_exposure = max_total_exposure
        self.max_category_exposure = max_category_exposure
        self.max_event_exposure = max_event_exposure
        self.kelly = KellyCriterion()

        logger.info(
            "PortfolioRiskManager initialized",
            max_total=max_total_exposure,
            max_category=max_category_exposure,
            max_event=max_event_exposure,
        )

    # ---- Position tracking ----

    def add_position(self, opportunity, size_usd: float) -> Position:
        """Record a new open position from an executed opportunity."""
        market_ids = [m.get("id", "") for m in opportunity.markets if m.get("id")]

        position = Position(
            opportunity_id=opportunity.id,
            market_ids=market_ids,
            event_id=opportunity.event_id,
            category=opportunity.category,
            size_usd=size_usd,
            entry_time=utcnow(),
            expected_roi=opportunity.roi_percent,
            risk_score=opportunity.risk_score,
        )

        self._positions[opportunity.id] = position

        logger.info(
            "Position added",
            opportunity_id=opportunity.id,
            size_usd=size_usd,
            event_id=opportunity.event_id,
            category=opportunity.category,
            total_positions=len(self._positions),
        )

        return position

    def remove_position(self, opportunity_id: str) -> Optional[Position]:
        """Remove a closed position. Returns the removed Position or None."""
        position = self._positions.pop(opportunity_id, None)
        if position:
            logger.info(
                "Position removed",
                opportunity_id=opportunity_id,
                size_usd=position.size_usd,
                total_positions=len(self._positions),
            )
        else:
            logger.warning(
                "Attempted to remove unknown position",
                opportunity_id=opportunity_id,
            )
        return position

    def get_portfolio_state(self) -> PortfolioState:
        """Build a current snapshot of the portfolio."""
        positions = list(self._positions.values())

        exposure_by_category: dict[str, float] = {}
        exposure_by_event: dict[str, float] = {}

        for pos in positions:
            if pos.category:
                exposure_by_category[pos.category] = exposure_by_category.get(pos.category, 0.0) + pos.size_usd
            if pos.event_id:
                exposure_by_event[pos.event_id] = exposure_by_event.get(pos.event_id, 0.0) + pos.size_usd

        total_exposure = sum(p.size_usd for p in positions)

        return PortfolioState(
            positions=positions,
            total_exposure_usd=round(total_exposure, 2),
            exposure_by_category=exposure_by_category,
            exposure_by_event=exposure_by_event,
            num_positions=len(positions),
        )

    # ---- Correlation detection ----

    def get_correlation_score(self, opportunity) -> float:
        """Compute the maximum correlation between an opportunity and existing
        positions.

        Returns a score in [0, 1]:
        - 1.0 if the opportunity shares an event_id with an existing position
        - 0.5 if same category only
        - 0.3 if overlapping market IDs only
        - 0.0 if fully independent
        """
        if not self._positions:
            return 0.0

        opp_market_ids = set(m.get("id", "") for m in opportunity.markets if m.get("id"))

        max_score = 0.0

        for pos in self._positions.values():
            # Same event is highest correlation
            if opportunity.event_id and pos.event_id and opportunity.event_id == pos.event_id:
                max_score = max(max_score, 1.0)
                break  # Can't go higher

            # Same category is moderate correlation
            if opportunity.category and pos.category and opportunity.category == pos.category:
                max_score = max(max_score, 0.5)

            # Overlapping market IDs is mild correlation
            pos_market_ids = set(pos.market_ids)
            if opp_market_ids & pos_market_ids:
                max_score = max(max_score, 0.3)

        return max_score

    # ---- Position checks ----

    def check_position_allowed(self, opportunity) -> tuple[bool, str, float]:
        """Determine whether a new position is allowed given portfolio constraints.

        Returns:
            (allowed, reason, recommended_size)
            - allowed: True if the position can be opened at all
            - reason: Human-readable explanation (empty string if allowed)
            - recommended_size: Suggested size in USD (0 if not allowed)
        """
        state = self.get_portfolio_state()

        # 1. Max open positions
        max_positions = getattr(settings, "MAX_OPEN_POSITIONS", 10)
        if state.num_positions >= max_positions:
            reason = f"Maximum open positions reached ({max_positions})"
            logger.warning(reason, opportunity_id=opportunity.id)
            return False, reason, 0.0

        # 2. Total exposure limit
        remaining_total = self.max_total_exposure - state.total_exposure_usd
        if remaining_total <= 0:
            reason = f"Total exposure limit reached (${state.total_exposure_usd:.2f} / ${self.max_total_exposure:.2f})"
            logger.warning(reason, opportunity_id=opportunity.id)
            return False, reason, 0.0

        # 3. Per-category limit
        remaining_category = self.max_category_exposure
        if opportunity.category:
            current_cat = state.exposure_by_category.get(opportunity.category, 0.0)
            remaining_category = self.max_category_exposure - current_cat
            if remaining_category <= 0:
                reason = (
                    f"Category '{opportunity.category}' exposure limit reached "
                    f"(${current_cat:.2f} / ${self.max_category_exposure:.2f})"
                )
                logger.warning(reason, opportunity_id=opportunity.id)
                return False, reason, 0.0

        # 4. Per-event limit
        remaining_event = self.max_event_exposure
        if opportunity.event_id:
            current_evt = state.exposure_by_event.get(opportunity.event_id, 0.0)
            remaining_event = self.max_event_exposure - current_evt
            if remaining_event <= 0:
                reason = (
                    f"Event '{opportunity.event_id}' exposure limit reached "
                    f"(${current_evt:.2f} / ${self.max_event_exposure:.2f})"
                )
                logger.warning(reason, opportunity_id=opportunity.id)
                return False, reason, 0.0

        # 5. Correlation check -- we still allow but warn and reduce
        correlation = self.get_correlation_score(opportunity)
        if correlation >= 1.0:
            # Same event: allow but strongly reduce
            logger.warning(
                "High correlation: same event as existing position",
                opportunity_id=opportunity.id,
                event_id=opportunity.event_id,
                correlation=correlation,
            )

        # Compute recommended size constrained by limits
        max_allowed = min(remaining_total, remaining_category, remaining_event)

        # Also respect per-market limit from config
        max_per_market = getattr(settings, "MAX_PER_MARKET_USD", 500.0)
        max_allowed = min(max_allowed, max_per_market)

        # Also respect max trade size from config
        max_trade = getattr(settings, "MAX_TRADE_SIZE_USD", 100.0)
        max_allowed = min(max_allowed, max_trade)

        recommended = max(0.0, round(max_allowed, 2))

        logger.info(
            "Position check passed",
            opportunity_id=opportunity.id,
            recommended_size=recommended,
            correlation=correlation,
            remaining_total=round(remaining_total, 2),
        )

        return True, "", recommended

    # ---- Size recommendation ----

    def calculate_recommended_size(self, opportunity, bankroll: float) -> float:
        """Calculate the final recommended position size incorporating Kelly sizing,
        portfolio constraints, liquidity limits, and correlation discounts.

        Args:
            opportunity: An ArbitrageOpportunity instance.
            bankroll: Current available capital in USD.

        Returns:
            Recommended position size in USD (may be 0 if disallowed).
        """
        # Step 0: Check if position is allowed at all
        allowed, reason, max_allowed = self.check_position_allowed(opportunity)
        if not allowed:
            logger.info(
                "Position disallowed, recommended size 0",
                reason=reason,
                opportunity_id=opportunity.id,
            )
            return 0.0

        # Step 1: Kelly-optimal size
        kelly_size = self.kelly.optimal_position_size(bankroll, opportunity)

        # Step 2: Apply portfolio constraint (don't exceed max_allowed from checks)
        size = min(kelly_size, max_allowed)

        # Step 3: Liquidity constraint -- don't exceed 10% of market liquidity
        if opportunity.min_liquidity > 0:
            liquidity_cap = opportunity.min_liquidity * self.LIQUIDITY_FRACTION_CAP
            if size > liquidity_cap:
                logger.debug(
                    "Liquidity cap applied",
                    original=round(size, 2),
                    liquidity_cap=round(liquidity_cap, 2),
                )
                size = liquidity_cap

        # Step 4: Correlation discount
        correlation = self.get_correlation_score(opportunity)
        if correlation > 0:
            # Discount factor: 1.0 (no correlation) -> 0.25 (fully correlated)
            discount = 1.0 - 0.75 * correlation
            original = size
            size *= discount
            logger.debug(
                "Correlation discount applied",
                correlation=correlation,
                discount=round(discount, 3),
                original=round(original, 2),
                adjusted=round(size, 2),
            )

        # Step 5: Floor at minimum order size, but don't go below 0
        min_order = getattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
        if 0 < size < min_order:
            logger.debug(
                "Size below minimum order, setting to 0",
                size=round(size, 2),
                min_order=min_order,
            )
            size = 0.0

        size = round(size, 2)

        logger.info(
            "Recommended size calculated",
            opportunity_id=opportunity.id,
            kelly_size=kelly_size,
            max_allowed=max_allowed,
            correlation=correlation,
            final_size=size,
        )

        return size

    # ---- Risk report ----

    def risk_report(self) -> dict:
        """Generate a comprehensive portfolio risk summary.

        Returns a dict with:
        - total_exposure: Current total USD exposure
        - num_positions: Number of open positions
        - max_drawdown_estimate: Estimated worst-case loss
        - concentration_hhi: Herfindahl-Hirschman index (0 = diversified, 1 = concentrated)
        - top_category: Category with highest exposure
        - top_event: Event with highest exposure
        - weighted_expected_return: Size-weighted expected ROI
        - sharpe_like_ratio: Expected return / risk (using risk_score as vol proxy)
        - worst_case_loss: Sum of all positions (total wipeout scenario)
        - category_breakdown: Exposure per category
        - event_breakdown: Exposure per event
        """
        state = self.get_portfolio_state()
        positions = state.positions

        report: dict = {
            "total_exposure": state.total_exposure_usd,
            "num_positions": state.num_positions,
            "max_total_limit": self.max_total_exposure,
            "utilization_pct": round(
                (state.total_exposure_usd / self.max_total_exposure * 100) if self.max_total_exposure > 0 else 0.0,
                1,
            ),
            "category_breakdown": state.exposure_by_category,
            "event_breakdown": state.exposure_by_event,
        }

        if not positions:
            report.update(
                {
                    "max_drawdown_estimate": 0.0,
                    "concentration_hhi": 0.0,
                    "top_category": None,
                    "top_event": None,
                    "weighted_expected_return": 0.0,
                    "sharpe_like_ratio": 0.0,
                    "worst_case_loss": 0.0,
                }
            )
            return report

        total = state.total_exposure_usd

        # Concentration: Herfindahl-Hirschman Index on position sizes
        if total > 0:
            shares = [p.size_usd / total for p in positions]
            hhi = sum(s * s for s in shares)
        else:
            hhi = 0.0

        # Size-weighted expected return
        weighted_roi = sum(p.size_usd * p.expected_roi for p in positions) / total if total > 0 else 0.0

        # Size-weighted risk (used as volatility proxy)
        weighted_risk = sum(p.size_usd * p.risk_score for p in positions) / total if total > 0 else 0.0

        # Sharpe-like ratio: expected return / risk
        sharpe = weighted_roi / weighted_risk if weighted_risk > 0 else 0.0

        # Max drawdown estimate: assume worst case each position loses
        # (risk_score * size).  This is a rough heuristic.
        max_drawdown = sum(p.size_usd * p.risk_score for p in positions)

        # Top category and event
        top_category = (
            max(state.exposure_by_category, key=state.exposure_by_category.get) if state.exposure_by_category else None
        )
        top_event = max(state.exposure_by_event, key=state.exposure_by_event.get) if state.exposure_by_event else None

        report.update(
            {
                "max_drawdown_estimate": round(max_drawdown, 2),
                "concentration_hhi": round(hhi, 4),
                "top_category": top_category,
                "top_event": top_event,
                "weighted_expected_return": round(weighted_roi, 2),
                "sharpe_like_ratio": round(sharpe, 2),
                "worst_case_loss": round(total, 2),
            }
        )

        logger.info(
            "Risk report generated",
            total_exposure=report["total_exposure"],
            num_positions=report["num_positions"],
            hhi=report["concentration_hhi"],
            sharpe=report["sharpe_like_ratio"],
        )

        return report


# ==================== SINGLETON ====================

portfolio_risk_manager = PortfolioRiskManager()
