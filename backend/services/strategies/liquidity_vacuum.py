"""
Strategy: Liquidity Vacuum - Order Book Imbalance Exploitation

Monitors order book depth on all markets. When one side has a large
imbalance (e.g., $50K bid depth vs $500 ask depth), the price is
about to move toward the thicker side.

This is a well-known strategy in traditional finance (order flow)
but nobody has systematically implemented it for Polymarket's CLOB.

The "vacuum" is the thin side of the book - prices get sucked
toward the thick side because there's no resistance.

NOT risk-free arbitrage - this is statistical edge trading.
"""

from typing import Optional
from datetime import datetime, timezone

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, utcnow, make_aware


class LiquidityVacuumStrategy(BaseStrategy):
    """Exploit order book imbalances for directional edge.

    Two detection modes:
    1. Direct depth analysis -- when prices dict contains bid_depth/ask_depth
       data, compute the raw imbalance ratio.
    2. Synthetic imbalance detector -- when depth data is absent, infer
       pressure from price spreads (yes + no deviating from 1.0) and from
       price velocity across consecutive scans.
    """

    strategy_type = StrategyType.LIQUIDITY_VACUUM
    name = "Liquidity Vacuum"
    description = "Exploit order book imbalances for directional edge"

    def __init__(self):
        super().__init__()
        self.min_imbalance_ratio = settings.LIQUIDITY_VACUUM_MIN_IMBALANCE_RATIO
        self.min_depth_usd = settings.LIQUIDITY_VACUUM_MIN_DEPTH_USD
        # Track previous prices for velocity calculation
        self._prev_prices: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # detect() -- main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        if not settings.LIQUIDITY_VACUUM_ENABLED:
            return []

        opportunities: list[ArbitrageOpportunity] = []

        # Build event lookup for enriching opportunities
        event_by_market: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                event_by_market[m.id] = event

        for market in markets:
            # Only binary markets
            if len(market.outcome_prices) != 2:
                continue
            if market.closed or not market.active:
                continue

            # Resolve live prices
            yes_price, no_price = self._resolve_prices(market, prices)

            if yes_price <= 0 or no_price <= 0:
                continue

            # Attempt direct depth-based detection first
            signal = self._detect_depth_imbalance(market, prices)

            # Fall back to synthetic detection
            if signal is None:
                signal = self._detect_synthetic_imbalance(
                    market, yes_price, no_price, prices
                )

            if signal is None:
                continue

            direction: str = signal["direction"]  # "YES" or "NO"
            imbalance_ratio: float = signal["imbalance_ratio"]
            reason: str = signal["reason"]
            thick_depth: float = signal.get("thick_depth", market.liquidity)

            # Filter: thick side must meet minimum depth
            if thick_depth < self.min_depth_usd:
                continue

            # Determine which side to buy and at what price
            if direction == "YES":
                buy_price = yes_price
                buy_outcome = "YES"
                token_index = 0
            else:
                buy_price = no_price
                buy_outcome = "NO"
                token_index = 1

            # Skip degenerate prices
            if buy_price <= 0.01 or buy_price >= 0.99:
                continue

            total_cost = buy_price

            # Build position
            token_id = (
                market.clob_token_ids[token_index]
                if len(market.clob_token_ids) > token_index
                else None
            )

            positions = [
                {
                    "action": "BUY",
                    "outcome": buy_outcome,
                    "market": market.question[:80],
                    "price": buy_price,
                    "token_id": token_id,
                    "imbalance_ratio": round(imbalance_ratio, 2),
                    "signal_reason": reason,
                }
            ]

            event = event_by_market.get(market.id)

            opp = self.create_opportunity(
                title=f"Vacuum: BUY {buy_outcome} {market.question[:50]}",
                description=(
                    f"Order book imbalance ({imbalance_ratio:.1f}x) "
                    f"detected — BUY {buy_outcome} @ ${buy_price:.3f} | {reason}"
                ),
                total_cost=total_cost,
                markets=[market],
                positions=positions,
                event=event,
            )

            if opp is not None:
                # Override risk_score with vacuum-specific scoring
                opp.risk_score = self._calculate_vacuum_risk(
                    imbalance_ratio, market, thick_depth
                )
                opp.risk_factors = self._build_risk_factors(
                    imbalance_ratio, market, thick_depth
                )
                opportunities.append(opp)

        # Update stored prices for next scan velocity calculation
        self._update_prev_prices(markets, prices)

        # Sort by imbalance ratio (strongest signal first)
        opportunities.sort(
            key=lambda o: next(
                (
                    p.get("imbalance_ratio", 0)
                    for p in o.positions_to_take
                    if "imbalance_ratio" in p
                ),
                0,
            ),
            reverse=True,
        )

        return opportunities

    # ------------------------------------------------------------------
    # Price resolution helpers
    # ------------------------------------------------------------------

    def _resolve_prices(
        self, market: Market, prices: dict[str, dict]
    ) -> tuple[float, float]:
        """Return (yes_price, no_price) using live data when available."""
        yes_price = market.yes_price
        no_price = market.no_price

        if market.clob_token_ids:
            yes_token = (
                market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None
            )
            no_token = (
                market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
            )

            if yes_token and yes_token in prices:
                yes_price = prices[yes_token].get("mid", yes_price)
            if no_token and no_token in prices:
                no_price = prices[no_token].get("mid", no_price)

        return yes_price, no_price

    # ------------------------------------------------------------------
    # Mode 1: Direct depth imbalance from order book data
    # ------------------------------------------------------------------

    def _detect_depth_imbalance(
        self, market: Market, prices: dict[str, dict]
    ) -> Optional[dict]:
        """Detect imbalance from explicit bid_depth / ask_depth in prices dict.

        Returns a signal dict or None.
        """
        if not market.clob_token_ids or len(market.clob_token_ids) < 2:
            return None

        yes_token = market.clob_token_ids[0]
        no_token = market.clob_token_ids[1]

        # Check if depth data is present for the YES token
        yes_data = prices.get(yes_token, {})
        bid_depth = yes_data.get("bid_depth")
        ask_depth = yes_data.get("ask_depth")

        if bid_depth is None or ask_depth is None:
            # Also check the NO token side
            no_data = prices.get(no_token, {})
            bid_depth_no = no_data.get("bid_depth")
            ask_depth_no = no_data.get("ask_depth")
            if bid_depth_no is None or ask_depth_no is None:
                return None
            # NO token depth: bids on NO = asks on YES equivalent
            bid_depth = ask_depth_no
            ask_depth = bid_depth_no

        bid_depth = float(bid_depth)
        ask_depth = float(ask_depth)

        if bid_depth <= 0 and ask_depth <= 0:
            return None

        # Avoid division by zero
        min_side = min(bid_depth, ask_depth)
        max_side = max(bid_depth, ask_depth)
        if min_side <= 0:
            # One side completely empty -- extreme imbalance
            imbalance_ratio = max_side / max(min_side, 0.01)
        else:
            imbalance_ratio = max_side / min_side

        if imbalance_ratio < self.min_imbalance_ratio:
            return None

        # Direction prediction
        if bid_depth > ask_depth:
            direction = "YES"  # Strong bids -> price going up
            thick_depth = bid_depth
        else:
            direction = "NO"  # Strong asks -> price going down, buy NO
            thick_depth = ask_depth

        return {
            "direction": direction,
            "imbalance_ratio": imbalance_ratio,
            "thick_depth": thick_depth,
            "reason": (
                f"Depth imbalance: bid ${bid_depth:,.0f} vs ask ${ask_depth:,.0f}"
            ),
        }

    # ------------------------------------------------------------------
    # Mode 2: Synthetic imbalance detector (no depth data)
    # ------------------------------------------------------------------

    def _detect_synthetic_imbalance(
        self,
        market: Market,
        yes_price: float,
        no_price: float,
        prices: dict[str, dict],
    ) -> Optional[dict]:
        """Infer order book imbalance from price spread and velocity.

        Synthetic signals:
        1. Spread analysis: yes + no deviating from 1.0 indicates pressure.
           - sum > 1.02  => excess selling pressure (vacuum on buy side)
           - sum < 0.96  => excess buying demand (vacuum on sell side)

        2. Price velocity: rapid price movement across scans amplifies signal.
        """
        combined = yes_price + no_price

        spread_signal = self._analyze_spread(combined, yes_price, no_price)
        velocity_signal = self._analyze_velocity(market, yes_price, no_price)

        # Merge signals -- either alone can trigger, velocity amplifies
        if spread_signal is None and velocity_signal is None:
            return None

        # Start with spread signal if present
        if spread_signal is not None:
            signal = spread_signal
            # Amplify imbalance ratio if velocity confirms direction
            if (
                velocity_signal is not None
                and velocity_signal["direction"] == signal["direction"]
            ):
                signal["imbalance_ratio"] = (
                    signal["imbalance_ratio"] + velocity_signal["imbalance_ratio"]
                ) / 2.0 + 1.0  # Bonus for confirmation
                signal["reason"] += f" + {velocity_signal['reason']}"
            return signal

        # Velocity-only signal (no spread signal)
        return velocity_signal

    def _analyze_spread(
        self, combined: float, yes_price: float, no_price: float
    ) -> Optional[dict]:
        """Detect imbalance from yes + no price deviation from 1.0."""
        if combined > 1.02:
            # Excess selling pressure -- both sides priced high.
            # The overpriced side is the one being sold into.
            # Vacuum is on the buy side: price will drop to equilibrium.
            # Buy the cheaper side (it's undervalued relative to the spread).
            deviation = combined - 1.0
            imbalance_ratio = 1.0 + (deviation * 50)  # Scale deviation to ratio

            if imbalance_ratio < self.min_imbalance_ratio:
                return None

            # Buy the side with lower price -- it has more upside
            if yes_price <= no_price:
                direction = "YES"
            else:
                direction = "NO"

            return {
                "direction": direction,
                "imbalance_ratio": imbalance_ratio,
                "thick_depth": market_liquidity_estimate(0),  # unknown, use 0
                "reason": f"Spread overpriced (sum={combined:.3f})",
            }

        elif combined < 0.96:
            # Excess buying demand -- both sides underpriced.
            # Strong buyers are present. Buy the side with more buying pressure.
            deviation = 1.0 - combined
            imbalance_ratio = 1.0 + (deviation * 50)

            if imbalance_ratio < self.min_imbalance_ratio:
                return None

            # Buy the side with higher price -- it has the buying pressure
            if yes_price >= no_price:
                direction = "YES"
            else:
                direction = "NO"

            return {
                "direction": direction,
                "imbalance_ratio": imbalance_ratio,
                "thick_depth": 0,
                "reason": f"Spread underpriced (sum={combined:.3f})",
            }

        return None

    def _analyze_velocity(
        self,
        market: Market,
        yes_price: float,
        no_price: float,
    ) -> Optional[dict]:
        """Detect imbalance from price movement across scans."""
        market_id = market.id
        prev = self._prev_prices.get(market_id)

        if prev is None:
            return None

        prev_yes = prev.get("yes", 0)
        prev_no = prev.get("no", 0)

        if prev_yes <= 0 or prev_no <= 0:
            return None

        yes_delta = yes_price - prev_yes
        no_delta = no_price - prev_no

        # Net directional movement
        # If YES is rising and NO is falling, strong upward pressure
        net_move = yes_delta - no_delta

        abs_move = abs(net_move)
        if abs_move < 0.02:  # Less than 2 cent move -- ignore noise
            return None

        # Convert movement magnitude to imbalance ratio
        # A 5-cent net move is significant in prediction markets
        imbalance_ratio = 1.0 + (abs_move * 100)  # 0.05 move -> 6.0 ratio

        if imbalance_ratio < self.min_imbalance_ratio:
            return None

        if net_move > 0:
            direction = "YES"  # Price moving up
        else:
            direction = "NO"  # Price moving down

        return {
            "direction": direction,
            "imbalance_ratio": imbalance_ratio,
            "thick_depth": market.liquidity,  # Best estimate
            "reason": (
                f"Price velocity: YES {yes_delta:+.3f}, NO {no_delta:+.3f}"
            ),
        }

    # ------------------------------------------------------------------
    # Price tracking for velocity
    # ------------------------------------------------------------------

    def _update_prev_prices(
        self, markets: list[Market], prices: dict[str, dict]
    ) -> None:
        """Store current prices for velocity calculation on next scan."""
        new_prev: dict[str, dict[str, float]] = {}
        for market in markets:
            if len(market.outcome_prices) != 2:
                continue
            yes_price, no_price = self._resolve_prices(market, prices)
            if yes_price > 0 and no_price > 0:
                new_prev[market.id] = {"yes": yes_price, "no": no_price}
        self._prev_prices = new_prev

    # ------------------------------------------------------------------
    # Vacuum-specific risk scoring
    # ------------------------------------------------------------------

    def _calculate_vacuum_risk(
        self,
        imbalance_ratio: float,
        market: Market,
        thick_depth: float,
    ) -> float:
        """Calculate risk score specific to liquidity vacuum trades.

        Base risk: 0.4 (these are NOT risk-free)
        Adjustments:
          - Higher imbalance ratio -> lower risk (stronger signal)
          - Higher market liquidity -> lower risk (more reliable)
          - Shorter time to resolution -> higher risk (news risk)
        """
        score = 0.4

        # Imbalance magnitude adjustment
        # Ratio of 5 = no adjustment; ratio of 20+ = significant reduction
        if imbalance_ratio >= 20:
            score -= 0.15
        elif imbalance_ratio >= 10:
            score -= 0.10
        elif imbalance_ratio >= 7:
            score -= 0.05

        # Market liquidity adjustment
        liquidity = market.liquidity
        if liquidity >= 50000:
            score -= 0.10  # Very liquid market, more reliable
        elif liquidity >= 10000:
            score -= 0.05
        elif liquidity < 2000:
            score += 0.10  # Thin market, less reliable signal

        # Time to resolution adjustment
        if market.end_date:
            end_aware = make_aware(market.end_date)
            days_until = (end_aware - utcnow()).days
            if days_until < 1:
                score += 0.15  # Imminent resolution, high news risk
            elif days_until < 3:
                score += 0.10
            elif days_until < 7:
                score += 0.05

        # Thick side depth adjustment
        if thick_depth >= 10000:
            score -= 0.05  # Very deep thick side
        elif thick_depth < 500:
            score += 0.05  # Thin even on "thick" side

        return max(0.05, min(score, 0.95))

    def _build_risk_factors(
        self,
        imbalance_ratio: float,
        market: Market,
        thick_depth: float,
    ) -> list[str]:
        """Build descriptive risk factor list for the opportunity."""
        factors = [
            "Statistical edge trade (NOT risk-free arbitrage)",
            f"Imbalance ratio: {imbalance_ratio:.1f}x",
        ]

        if market.liquidity < 2000:
            factors.append(f"Low market liquidity (${market.liquidity:,.0f})")
        elif market.liquidity < 10000:
            factors.append(f"Moderate market liquidity (${market.liquidity:,.0f})")

        if thick_depth > 0 and thick_depth < 500:
            factors.append(f"Thin depth even on strong side (${thick_depth:,.0f})")

        if market.end_date:
            end_aware = make_aware(market.end_date)
            days_until = (end_aware - utcnow()).days
            if days_until < 3:
                factors.append(f"Near resolution ({days_until} days) - news risk")
            elif days_until < 7:
                factors.append(f"Short time to resolution ({days_until} days)")

        if imbalance_ratio < 7:
            factors.append("Moderate imbalance signal")
        elif imbalance_ratio >= 15:
            factors.append("Very strong imbalance signal")

        return factors


def market_liquidity_estimate(default: float = 0) -> float:
    """Placeholder for when we cannot determine depth from prices dict."""
    return default
