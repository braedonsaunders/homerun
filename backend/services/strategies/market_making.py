"""
Strategy: Market Making - Provide Liquidity for Profit

Instead of taking existing arbitrage, CREATE profit by providing
liquidity. Place limit orders on both sides of the book and earn
the bid-ask spread.

Key insight: On Polymarket, MAKERS pay 0% fees. This means the
full bid-ask spread is profit.

This is how the most profitable Polymarket bots actually work.
It's not "arbitrage" per se but is the dominant profit mechanism.

The strategy identifies ideal market-making candidates:
- High volume (lots of crossing flow)
- Wide spread (more profit per fill)
- Stable price (less inventory risk)
- Binary markets near 50/50 (maximum two-sided flow)
"""

from datetime import datetime
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, utcnow, make_aware


class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy

    Provide liquidity on both sides of a binary market to earn the
    bid-ask spread. On Polymarket makers pay 0% fees, so the full
    spread is profit when both sides fill.

    Ideal candidates:
    - Binary markets (exactly 2 outcomes)
    - High volume (frequent crossing flow fills both sides)
    - Wide spread (more profit per round-trip)
    - Prices near 50/50 (balanced order flow, lower inventory risk)
    """

    strategy_type = StrategyType.MARKET_MAKING
    name = "Market Making"
    description = "Provide liquidity on both sides to earn the bid-ask spread"

    def __init__(self):
        super().__init__()
        self.min_spread_bps = settings.MARKET_MAKING_SPREAD_BPS
        self.max_inventory_usd = settings.MARKET_MAKING_MAX_INVENTORY_USD
        self.min_liquidity = 1000.0  # Minimum $1000 liquidity
        self.min_volume = 500.0  # Minimum trading volume

    def _get_prices(
        self, market: Market, prices: dict[str, dict]
    ) -> tuple[float, float]:
        """Get live YES/NO prices, falling back to market snapshot."""
        yes_price = market.yes_price
        no_price = market.no_price

        if market.clob_token_ids:
            if len(market.clob_token_ids) > 0:
                yes_token = market.clob_token_ids[0]
                if yes_token in prices:
                    yes_price = prices[yes_token].get("mid", yes_price)
            if len(market.clob_token_ids) > 1:
                no_token = market.clob_token_ids[1]
                if no_token in prices:
                    no_price = prices[no_token].get("mid", no_price)

        return yes_price, no_price

    def _calculate_spread(self, yes_price: float, no_price: float) -> float:
        """Calculate the bid-ask spread.

        If actual order book data were available we would use ask - bid.
        With snapshot prices we estimate spread from the gap between
        the YES and NO prices and the theoretical $1 payout:
            spread = 1.0 - (yes_price + no_price)

        A positive value means the combined cost is less than $1, which
        represents the effective spread available to a market maker.
        When yes + no > 1 the spread estimate becomes the overround,
        and we still treat it as the available spread (the book is wide
        enough that a maker can quote inside).
        """
        spread = 1.0 - (yes_price + no_price)
        # A negative spread means overround (yes + no > 1.0).
        # No genuine market-making opportunity exists in overround.
        if spread < 0:
            return 0.0
        return spread

    def _calculate_inventory_risk(self, yes_price: float) -> float:
        """Score inventory risk based on how far the price is from 50/50.

        Markets near 0.50 have balanced two-sided flow -> low risk.
        Markets near extremes (0.10 or 0.90) attract one-sided flow -> high risk.

        Returns a score in [0, 1] where 0 = lowest risk, 1 = highest.
        """
        return abs(yes_price - 0.5) * 2.0

    def _calculate_risk_score(
        self,
        spread: float,
        yes_price: float,
        market: Market,
        resolution_date: Optional[datetime],
    ) -> tuple[float, list[str]]:
        """Calculate a risk score in the 0.30-0.50 range for market making.

        Lower risk for:
        - Wide spread (more margin of safety)
        - Price near 50/50 (balanced flow)
        - High volume (frequent fills)
        - Far resolution (more time to earn)

        Higher risk for:
        - Narrow spread (thin margins)
        - Extreme prices (one-sided flow)
        - Low volume (slow fills)
        - Near resolution (inventory may be stranded)
        """
        base_risk = 0.40  # Start in the middle of [0.30, 0.50]
        factors: list[str] = []

        # --- Spread width adjustment ---
        # Wider spread -> lower risk (more margin)
        if spread >= 0.04:
            base_risk -= 0.04
            factors.append(f"Wide spread ({spread:.1%}) provides good margin")
        elif spread >= 0.02:
            base_risk -= 0.02
            factors.append(f"Moderate spread ({spread:.1%})")
        else:
            base_risk += 0.03
            factors.append(f"Narrow spread ({spread:.1%}) - thin margins")

        # --- Inventory risk from price extremes ---
        inv_risk = self._calculate_inventory_risk(yes_price)
        if inv_risk < 0.20:
            base_risk -= 0.03
            factors.append(f"Price near 50/50 (YES={yes_price:.2f}) - balanced flow")
        elif inv_risk > 0.60:
            base_risk += 0.04
            factors.append(f"Extreme price (YES={yes_price:.2f}) - one-sided flow risk")
        else:
            factors.append(f"Moderate inventory risk (YES={yes_price:.2f})")

        # --- Volume adjustment ---
        if market.volume >= 50000:
            base_risk -= 0.03
            factors.append(f"High volume (${market.volume:,.0f}) - frequent fills")
        elif market.volume >= 10000:
            base_risk -= 0.01
            factors.append(f"Moderate volume (${market.volume:,.0f})")
        else:
            base_risk += 0.02
            factors.append(f"Low volume (${market.volume:,.0f}) - slow fill risk")

        # --- Time to resolution ---
        if resolution_date:
            resolution_aware = make_aware(resolution_date)
            days_until = (resolution_aware - utcnow()).days
            if days_until < 2:
                base_risk += 0.05
                factors.append(
                    f"Very near resolution ({days_until}d) - inventory may be stranded"
                )
            elif days_until < 7:
                base_risk += 0.02
                factors.append(f"Near resolution ({days_until}d)")
            elif days_until > 90:
                base_risk -= 0.02
                factors.append(f"Far resolution ({days_until}d) - ample time to earn")

        # Clamp to [0.30, 0.50]
        risk_score = max(0.30, min(0.50, base_risk))
        return risk_score, factors

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Detect market-making opportunities.

        Scans all binary markets for candidates with:
        1. Sufficient liquidity and volume
        2. A spread wide enough to profit as a maker
        3. Acceptable inventory risk (price not too extreme)

        Returns ranked opportunities.
        """
        if not settings.MARKET_MAKING_ENABLED:
            return []

        opportunities: list[ArbitrageOpportunity] = []

        # Build an event lookup for enrichment
        event_by_market: dict[str, Event] = {}
        for event in events:
            for em in event.markets:
                event_by_market[em.id] = event

        for market in markets:
            # --- Filter: binary markets only ---
            if len(market.outcome_prices) != 2:
                continue

            # --- Filter: active and not closed ---
            if market.closed or not market.active:
                continue

            # --- Filter: minimum liquidity ---
            if market.liquidity < self.min_liquidity:
                continue

            # --- Filter: minimum volume ---
            if market.volume < self.min_volume:
                continue

            # --- Get live prices ---
            yes_price, no_price = self._get_prices(market, prices)

            # Sanity check prices
            if yes_price <= 0 or no_price <= 0:
                continue
            if yes_price >= 1.0 or no_price >= 1.0:
                continue

            # --- Calculate spread ---
            spread = self._calculate_spread(yes_price, no_price)

            # Minimum spread filter (in terms of mid-price basis points)
            mid_price = yes_price  # For YES side, mid_price is the YES price
            min_spread_absolute = (self.min_spread_bps / 10000.0) * mid_price
            if spread < min_spread_absolute:
                continue

            # --- Fair value and quote prices ---
            fair_value = yes_price
            half_spread = spread / 2.0
            buy_price = round(fair_value - half_spread, 4)
            sell_price = round(fair_value + half_spread, 4)

            # Ensure prices are valid (within 0-1 range)
            if buy_price <= 0 or sell_price >= 1.0:
                continue

            # --- Inventory risk ---
            inv_risk = self._calculate_inventory_risk(yes_price)

            # Skip markets with extreme inventory risk
            if inv_risk > 0.80:
                continue

            # --- Profit calculation ---
            # Market making profit: if both sides fill, we earn the spread.
            # Maker fee on Polymarket is 0%, so the full spread is profit.
            # total_cost = buy_price (capital needed to buy the YES side)
            # expected_payout = sell_price (revenue when we sell the YES side)
            total_cost = buy_price
            expected_payout = sell_price
            gross_profit = expected_payout - total_cost  # = spread
            fee = 0.0  # Makers pay 0% on Polymarket
            net_profit = gross_profit - fee
            roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

            # Skip if profit is negligible
            if net_profit <= 0:
                continue

            # Position sizing (moved up for hard filter checks)
            max_position = min(
                market.liquidity * 0.05,
                self.max_inventory_usd,
            )

            # --- Hard filters (matching create_opportunity gate) ---
            if roi > settings.MAX_PLAUSIBLE_ROI:
                continue
            if market.liquidity < settings.MIN_LIQUIDITY_HARD:
                continue
            if max_position < settings.MIN_POSITION_SIZE:
                continue
            absolute_profit = (
                max_position * (net_profit / total_cost) if total_cost > 0 else 0
            )
            if absolute_profit < settings.MIN_ABSOLUTE_PROFIT:
                continue
            if market.end_date:
                resolution_aware = make_aware(market.end_date)
                days_until = (resolution_aware - utcnow()).days
                if days_until > settings.MAX_RESOLUTION_MONTHS * 30:
                    continue

            # --- Risk score ---
            resolution_date = market.end_date
            risk_score, risk_factors = self._calculate_risk_score(
                spread, yes_price, market, resolution_date
            )

            # --- Build positions ---
            yes_token_id = (
                market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None
            )

            positions = [
                {
                    "action": "LIMIT_BUY",
                    "outcome": "YES",
                    "price": buy_price,
                    "token_id": yes_token_id,
                },
                {
                    "action": "LIMIT_SELL",
                    "outcome": "YES",
                    "price": sell_price,
                    "token_id": yes_token_id,
                },
            ]

            # --- Enrich with event data ---
            event = event_by_market.get(market.id)
            market_platform = str(getattr(market, "platform", "") or "").strip().lower()
            if market_platform not in {"polymarket", "kalshi"}:
                market_platform = (
                    "kalshi"
                    if str(getattr(market, "id", "")).upper().startswith("KX")
                    else "polymarket"
                )

            # --- Build the opportunity directly (like Miracle strategy) ---
            opp = ArbitrageOpportunity(
                strategy=self.strategy_type,
                is_guaranteed=False,
                title=f"MM: {market.question[:60]}",
                description=(
                    f"Market make YES @ bid ${buy_price:.3f} / ask ${sell_price:.3f} | "
                    f"Spread {spread:.1%} | Inv risk {inv_risk:.0%}"
                ),
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
                        "id": market.id,
                        "condition_id": market.condition_id,
                        "slug": market.slug,
                        "event_slug": market.event_slug,
                        "platform": market_platform,
                        "question": market.question,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "liquidity": market.liquidity,
                        "volume": market.volume,
                    }
                ],
                event_id=event.id if event else None,
                event_slug=event.slug if event else None,
                event_title=event.title if event else None,
                category=event.category if event else None,
                min_liquidity=market.liquidity,
                max_position_size=max_position,
                resolution_date=resolution_date,
                positions_to_take=positions,
            )

            opportunities.append(opp)

        # --- Rank by expected profit vs risk ---
        # Score = ROI / risk_score (higher ROI and lower risk = better)
        opportunities.sort(
            key=lambda o: o.roi_percent / max(o.risk_score, 0.01), reverse=True
        )

        return opportunities
