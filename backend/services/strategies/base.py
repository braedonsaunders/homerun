from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime, timezone

from models import Market, Event, ArbitrageOpportunity
from config import settings
from services.fee_model import fee_model


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
    """Base class for arbitrage detection strategies.

    Built-in strategies set strategy_type to a StrategyType enum value.
    Plugin strategies set strategy_type to their unique slug string.
    """

    strategy_type: str  # StrategyType value or plugin slug
    name: str
    description: str

    def __init__(self):
        self.fee = settings.POLYMARKET_FEE
        self.min_profit = settings.MIN_PROFIT_THRESHOLD

    @abstractmethod
    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
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
            very_short = getattr(settings, "RISK_VERY_SHORT_DAYS", 2)
            short = getattr(settings, "RISK_SHORT_DAYS", 7)
            long_lockup = getattr(settings, "RISK_LONG_LOCKUP_DAYS", 180)
            ext_lockup = getattr(settings, "RISK_EXTENDED_LOCKUP_DAYS", 90)
            if days_until < very_short:
                score += 0.4
                factors.append(f"Very short time to resolution (<{very_short} days)")
            elif days_until < short:
                score += 0.2
                factors.append(f"Short time to resolution (<{short} days)")
            # Long-duration capital lockup risk
            elif days_until > long_lockup:
                score += 0.3
                factors.append(f"Long capital lockup ({days_until} days to resolution)")
            elif days_until > ext_lockup:
                score += 0.2
                factors.append(f"Extended capital lockup ({days_until} days to resolution)")

        # Liquidity risk
        low_liq = getattr(settings, "RISK_LOW_LIQUIDITY", 1000.0)
        mod_liq = getattr(settings, "RISK_MODERATE_LIQUIDITY", 5000.0)
        min_liquidity = min((m.liquidity for m in markets), default=0)
        if min_liquidity < low_liq:
            score += 0.3
            factors.append(f"Low liquidity (${min_liquidity:.0f})")
        elif min_liquidity < mod_liq:
            score += 0.15
            factors.append(f"Moderate liquidity (${min_liquidity:.0f})")

        # Number of markets (complexity risk) — slippage compounds per leg
        complex_legs = getattr(settings, "RISK_COMPLEX_LEGS", 5)
        multi_legs = getattr(settings, "RISK_MULTIPLE_LEGS", 3)
        if len(markets) > complex_legs:
            score += 0.2
            factors.append(f"Complex trade ({len(markets)} legs — high slippage risk)")
        elif len(markets) > multi_legs:
            score += 0.1
            factors.append(f"Multiple positions ({len(markets)} markets)")

        # Multi-leg execution risk
        num_legs = len(markets)
        if num_legs > 1:
            # Each additional leg adds ~5% probability of partial fill
            partial_fill_risk = 1 - (0.95 ** (num_legs - 1))
            score += partial_fill_risk * 0.3  # Weight execution risk
            if partial_fill_risk > 0.2:
                factors.append(
                    f"Multi-leg execution risk ({num_legs} legs, {partial_fill_risk:.0%} partial fill chance)"
                )
            elif num_legs > 1:
                factors.append(f"Multi-leg trade ({num_legs} legs)")

        return min(score, 1.0), factors

    def _calculate_annualized_roi(self, roi_percent: float, resolution_date: Optional[datetime]) -> Optional[float]:
        """Calculate annualized ROI based on days to resolution.

        Returns None if resolution_date is unknown.
        """
        if not resolution_date:
            return None
        resolution_aware = make_aware(resolution_date)
        days_until = max((resolution_aware - utcnow()).total_seconds() / 86400.0, 1.0)
        return roi_percent * (365.0 / days_until)

    def create_opportunity(
        self,
        title: str,
        description: str,
        total_cost: float,
        markets: list[Market],
        positions: list[dict],
        event: Optional[Event] = None,
        expected_payout: float = 1.0,  # Override for strategies with non-$1 payouts
        is_guaranteed: bool = True,  # False for directional/statistical strategies
        # VWAP-adjusted parameters (all optional for backward compatibility)
        vwap_total_cost: Optional[float] = None,  # Realistic cost from order book
        spread_bps: Optional[float] = None,  # Actual spread in basis points
        fill_probability: Optional[float] = None,  # Probability all legs fill
        # Strategy-specific threshold overrides (high-freq crypto uses thinner books)
        min_liquidity_hard: Optional[float] = None,
        min_position_size: Optional[float] = None,
        min_absolute_profit: Optional[float] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """Create an ArbitrageOpportunity if profitable.

        Applies hard rejection filters:
        1. ROI must exceed MIN_PROFIT_THRESHOLD
        2. Min liquidity must exceed MIN_LIQUIDITY_HARD (or min_liquidity_hard override)
        3. Max position size must exceed MIN_POSITION_SIZE (or min_position_size override)
        4. Absolute profit on max position must exceed MIN_ABSOLUTE_PROFIT (or override)
        5. Annualized ROI must exceed MIN_ANNUALIZED_ROI (if resolution date known)
        6. Resolution must be within MAX_RESOLUTION_MONTHS
        """

        gross_profit = expected_payout - total_cost
        fee = expected_payout * self.fee
        net_profit = gross_profit - fee
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        # --- Comprehensive fee model (gas, spread, multi-leg slippage) ---
        is_negrisk = any(getattr(m, "neg_risk", False) for m in markets)
        total_liquidity = sum(m.liquidity for m in markets)
        fee_breakdown = fee_model.calculate_fees(
            expected_payout=expected_payout,
            num_legs=len(markets),
            is_negrisk=is_negrisk,
            spread_bps=spread_bps,
            total_cost=total_cost,
            maker_mode=settings.FEE_MODEL_MAKER_MODE,
            total_liquidity=total_liquidity,
        )

        # --- VWAP-adjusted realistic profit ---
        realistic_cost = vwap_total_cost if vwap_total_cost is not None else total_cost
        realistic_gross = expected_payout - realistic_cost
        realistic_net = realistic_gross - fee_breakdown.total_fees
        realistic_roi = (realistic_net / realistic_cost) * 100 if realistic_cost > 0 else 0

        # Use realistic profit for filtering when VWAP data is available
        effective_roi = realistic_roi if vwap_total_cost is not None else roi

        # Check if profitable after fees
        if effective_roi < self.min_profit * 100:
            return None

        # --- Hard filter: implausible directional ROI ---
        # Directional strategies are not guaranteed spreads. Extremely high
        # ROI readings here are usually payout-model artifacts (e.g. buying a
        # $0.05 leg and treating it like guaranteed $1 settlement).
        if not is_guaranteed:
            # Keep directional ROI guard fixed and conservative. Tying this
            # directly to configurable arbitrage ROI caps can let clearly
            # implausible directional opportunities leak through.
            directional_roi_cap = 120.0
            if effective_roi > directional_roi_cap:
                return None

        # --- Hard filter: suspiciously high ROI ---
        # In efficient prediction markets, genuine arbitrage is 1-5%.
        # ROI > 30% almost always indicates non-exhaustive outcomes, stale
        # order books, or missing data — not a real mispricing.
        # Skip this check for directional/statistical strategies where
        # "ROI" represents potential return, not guaranteed profit.
        if is_guaranteed and effective_roi > settings.MAX_PLAUSIBLE_ROI:
            return None

        # --- Hard filter: too many legs ---
        # Slippage compounds per leg. An 8-leg trade where each leg has
        # even 0.5% slippage loses 4% of margin before execution completes.
        if len(markets) > settings.MAX_TRADE_LEGS:
            return None

        # --- Hard filter: minimum liquidity per leg ---
        # Multi-leg trades need proportionally more liquidity to avoid
        # slippage cascading across legs.
        min_liq_per_leg = getattr(settings, "MIN_LIQUIDITY_PER_LEG", 500.0)
        if len(markets) > 1:
            required_liquidity = min_liq_per_leg * len(markets)
            if total_liquidity < required_liquidity:
                return None

        # Calculate max position size based on liquidity
        min_liquidity = min((m.liquidity for m in markets), default=0)
        max_position = min_liquidity * 0.1  # Don't exceed 10% of liquidity

        # Resolve thresholds (allow strategy-specific overrides for thin-book markets)
        eff_min_liquidity = min_liquidity_hard if min_liquidity_hard is not None else settings.MIN_LIQUIDITY_HARD
        eff_min_position = min_position_size if min_position_size is not None else settings.MIN_POSITION_SIZE
        eff_min_absolute = min_absolute_profit if min_absolute_profit is not None else settings.MIN_ABSOLUTE_PROFIT

        # --- Hard filter: minimum liquidity ---
        if min_liquidity < eff_min_liquidity:
            return None

        # --- Hard filter: minimum position size ---
        if max_position < eff_min_position:
            return None

        # --- Hard filter: minimum absolute profit ---
        # Use realistic net profit for the absolute profit check when available
        effective_net = realistic_net if vwap_total_cost is not None else net_profit
        effective_cost = realistic_cost if vwap_total_cost is not None else total_cost
        absolute_profit = max_position * (effective_net / effective_cost) if effective_cost > 0 else 0
        if absolute_profit < eff_min_absolute:
            return None

        # Calculate risk
        resolution_date = None
        if markets and markets[0].end_date:
            resolution_date = markets[0].end_date

        # --- Hard filter: maximum resolution timeframe ---
        if resolution_date:
            resolution_aware = make_aware(resolution_date)
            days_until = (resolution_aware - utcnow()).total_seconds() / 86400.0
            max_days = settings.MAX_RESOLUTION_MONTHS * 30
            if days_until > max_days:
                return None

            # --- Hard filter: minimum annualized ROI ---
            annualized_roi = self._calculate_annualized_roi(effective_roi, resolution_date)
            if annualized_roi is not None and annualized_roi < settings.MIN_ANNUALIZED_ROI:
                return None

        risk_score, risk_factors = self.calculate_risk_score(markets, resolution_date)

        # Build enriched market dicts with VWAP metadata
        market_dicts = []
        for m in markets:
            entry: dict = {
                "id": m.id,
                "slug": m.slug,
                "question": m.question,
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "liquidity": m.liquidity,
                "platform": getattr(m, "platform", "polymarket"),
            }
            market_dicts.append(entry)

        # Attach realistic-profit metadata to positions_to_take
        enriched_positions = list(positions)  # shallow copy
        if enriched_positions:
            enriched_positions[0] = {
                **enriched_positions[0],
                "_fee_breakdown": {
                    "winner_fee": fee_breakdown.winner_fee,
                    "gas_cost_usd": fee_breakdown.gas_cost_usd,
                    "spread_cost": fee_breakdown.spread_cost,
                    "multi_leg_slippage": fee_breakdown.multi_leg_slippage,
                    "total_fees": fee_breakdown.total_fees,
                    "fee_as_pct_of_payout": fee_breakdown.fee_as_pct_of_payout,
                },
                "_realistic_profit": {
                    "vwap_total_cost": vwap_total_cost,
                    "realistic_gross": realistic_gross,
                    "realistic_net": realistic_net,
                    "realistic_roi": realistic_roi,
                    "theoretical_roi": roi,
                    "spread_bps": spread_bps,
                    "fill_probability": fill_probability,
                },
            }

        roi_type = "guaranteed_spread" if is_guaranteed else "directional_payout"

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
            is_guaranteed=is_guaranteed,
            roi_type=roi_type,
            risk_score=risk_score,
            risk_factors=risk_factors,
            markets=market_dicts,
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=resolution_date,
            positions_to_take=enriched_positions,
        )
