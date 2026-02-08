from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime, timezone

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings


def utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime"""
    return datetime.now(timezone.utc)


def make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Make a datetime timezone-aware (UTC) if it isn't already"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class BaseStrategy(ABC):
    """Base class for arbitrage detection strategies"""

    strategy_type: StrategyType
    name: str
    description: str

    def __init__(self):
        self.fee = settings.POLYMARKET_FEE
        self.min_profit = settings.MIN_PROFIT_THRESHOLD

    @abstractmethod
    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Detect arbitrage opportunities"""
        pass

    def calculate_risk_score(
        self, markets: list[Market], resolution_date: Optional[datetime] = None
    ) -> tuple[float, list[str]]:
        """Calculate risk score (0-1) and return risk factors"""
        score = 0.0
        factors = []

        # Time to resolution risk
        if resolution_date:
            resolution_aware = make_aware(resolution_date)
            days_until = (resolution_aware - utcnow()).days
            if days_until < 2:
                score += 0.4
                factors.append("Very short time to resolution (<2 days)")
            elif days_until < 7:
                score += 0.2
                factors.append("Short time to resolution (<7 days)")
            # Long-duration capital lockup risk
            elif days_until > 180:
                score += 0.3
                factors.append(f"Long capital lockup ({days_until} days to resolution)")
            elif days_until > 90:
                score += 0.2
                factors.append(
                    f"Extended capital lockup ({days_until} days to resolution)"
                )

        # Liquidity risk
        min_liquidity = min((m.liquidity for m in markets), default=0)
        if min_liquidity < 1000:
            score += 0.3
            factors.append(f"Low liquidity (${min_liquidity:.0f})")
        elif min_liquidity < 5000:
            score += 0.15
            factors.append(f"Moderate liquidity (${min_liquidity:.0f})")

        # Number of markets (complexity risk) — slippage compounds per leg
        if len(markets) > 5:
            score += 0.2
            factors.append(f"Complex trade ({len(markets)} legs — high slippage risk)")
        elif len(markets) > 3:
            score += 0.1
            factors.append(f"Multiple positions ({len(markets)} markets)")

        return min(score, 1.0), factors

    def _calculate_annualized_roi(
        self, roi_percent: float, resolution_date: Optional[datetime]
    ) -> Optional[float]:
        """Calculate annualized ROI based on days to resolution.

        Returns None if resolution_date is unknown.
        """
        if not resolution_date:
            return None
        resolution_aware = make_aware(resolution_date)
        days_until = max((resolution_aware - utcnow()).days, 1)
        return roi_percent * (365.0 / days_until)

    def create_opportunity(
        self,
        title: str,
        description: str,
        total_cost: float,
        markets: list[Market],
        positions: list[dict],
        event: Optional[Event] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """Create an ArbitrageOpportunity if profitable.

        Applies hard rejection filters:
        1. ROI must exceed MIN_PROFIT_THRESHOLD
        2. Min liquidity must exceed MIN_LIQUIDITY_HARD
        3. Max position size must exceed MIN_POSITION_SIZE
        4. Absolute profit on max position must exceed MIN_ABSOLUTE_PROFIT
        5. Annualized ROI must exceed MIN_ANNUALIZED_ROI (if resolution date known)
        6. Resolution must be within MAX_RESOLUTION_MONTHS
        """

        expected_payout = 1.0
        gross_profit = expected_payout - total_cost
        fee = expected_payout * self.fee
        net_profit = gross_profit - fee
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        # Check if profitable after fees
        if roi < self.min_profit * 100:
            return None

        # Calculate max position size based on liquidity
        min_liquidity = min((m.liquidity for m in markets), default=0)
        max_position = min_liquidity * 0.1  # Don't exceed 10% of liquidity

        # --- Hard filter: minimum liquidity ---
        if min_liquidity < settings.MIN_LIQUIDITY_HARD:
            return None

        # --- Hard filter: minimum position size ---
        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # --- Hard filter: minimum absolute profit ---
        # The profit from max_position investment (scaled proportionally)
        absolute_profit = (
            max_position * (net_profit / total_cost) if total_cost > 0 else 0
        )
        if absolute_profit < settings.MIN_ABSOLUTE_PROFIT:
            return None

        # Calculate risk
        resolution_date = None
        if markets and markets[0].end_date:
            resolution_date = markets[0].end_date

        # --- Hard filter: maximum resolution timeframe ---
        if resolution_date:
            resolution_aware = make_aware(resolution_date)
            days_until = (resolution_aware - utcnow()).days
            max_days = settings.MAX_RESOLUTION_MONTHS * 30
            if days_until > max_days:
                return None

            # --- Hard filter: minimum annualized ROI ---
            annualized_roi = self._calculate_annualized_roi(roi, resolution_date)
            if (
                annualized_roi is not None
                and annualized_roi < settings.MIN_ANNUALIZED_ROI
            ):
                return None

        risk_score, risk_factors = self.calculate_risk_score(markets, resolution_date)

        return ArbitrageOpportunity(
            strategy=self.strategy_type,
            title=title,
            description=description,
            total_cost=total_cost,
            expected_payout=expected_payout,
            gross_profit=gross_profit,
            fee=fee,
            net_profit=net_profit,
            roi_percent=roi,
            risk_score=risk_score,
            risk_factors=risk_factors,
            markets=[
                {
                    "id": m.id,
                    "slug": m.slug,
                    "question": m.question,
                    "yes_price": m.yes_price,
                    "no_price": m.no_price,
                    "liquidity": m.liquidity,
                }
                for m in markets
            ],
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=resolution_date,
            positions_to_take=positions,
        )
