"""
Strategy 6: Miracle Market Scanner (Garbage Collection)

Based on the Swisstony strategy that turned $5 into $3.7M.

This strategy identifies markets asking about highly improbable "miracle" events
and bets NO on them. These are not risk-free arbitrage opportunities - they have
a tiny probability of loss. But the probability is so low that consistent execution
across many markets generates reliable profit.

Key insight: "He does not predict the future. He bets against miracles."

Categories of miracle events:
- Apocalypse/World-ending events (WW3, asteroid impact)
- Supernatural events (aliens landing, divine interventions)
- Impossible physics (time travel, FTL communication)
- Celebrity impossibilities (certain deaths by tomorrow, pregnancies)
- Extreme deadline impossibilities (Bitcoin to $1M by Friday)
- Political impossibilities (certain resignations by tomorrow)
- Logical impossibilities (events that already didn't happen)

The strategy:
1. Scan markets for absurd/impossible questions using keyword patterns
2. Find markets where NO is priced at $0.90+ (indicating near-certainty)
3. Calculate expected value: tiny profit per trade, but near-guaranteed
4. Flag stale markets where events already became logically impossible
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


# Keywords indicating highly improbable events
MIRACLE_KEYWORDS = {
    # Supernatural/Paranormal
    "alien": 0.95,
    "aliens": 0.95,
    "ufo": 0.95,
    "extraterrestrial": 0.95,
    "ghost": 0.90,
    "supernatural": 0.90,
    "paranormal": 0.90,
    "miracle": 0.85,
    "divine": 0.85,
    "god appear": 0.95,
    "resurrection": 0.95,
    "rapture": 0.95,
    "second coming": 0.95,

    # Apocalypse/World-ending
    "world war 3": 0.80,
    "ww3": 0.80,
    "nuclear war": 0.80,
    "apocalypse": 0.95,
    "end of the world": 0.95,
    "extinction": 0.85,
    "asteroid impact": 0.90,
    "meteor strike": 0.90,

    # Impossible physics
    "time travel": 0.95,
    "faster than light": 0.95,
    "teleportation": 0.90,
    "free energy": 0.90,
    "perpetual motion": 0.95,

    # Extreme claims
    "prove": 0.70,  # "Will X prove..." often absurd
    "confirm existence": 0.80,
    "discovered": 0.60,  # Context dependent

    # Celebrity/Political impossibilities
    "resign by tomorrow": 0.90,
    "resign this week": 0.85,
    "die by": 0.75,  # Needs short timeframe
    "assassinated": 0.70,

    # Crypto impossibilities (extreme short-term moves)
    "bitcoin.*1 million": 0.85,
    "btc.*1m": 0.85,
    "eth.*100k": 0.85,

    # Hoax indicators
    "hoax": 0.80,
    "fake": 0.70,
    "debunked": 0.85,
}

# Phrases that boost impossibility score
IMPOSSIBILITY_PHRASES = [
    (r"by (tomorrow|tonight|today|this week|friday|monday|end of week)", 0.15),
    (r"before (january|february|march|april|may|june|july|august|september|october|november|december) \d+", 0.10),
    (r"in the next (\d+) (hour|day|week)", 0.15),
    (r"within (\d+) (hour|day)", 0.20),
    (r"will .* ever", -0.10),  # "Will X ever happen" is less predictable
    (r"confirm.*alien", 0.20),
    (r"land on (earth|times square|white house)", 0.15),
    (r"declare war", 0.10),
]

# Categories for classification
MIRACLE_CATEGORIES = {
    "apocalypse": ["ww3", "world war", "nuclear", "apocalypse", "extinction", "end of world"],
    "supernatural": ["alien", "ufo", "ghost", "paranormal", "miracle", "divine", "rapture"],
    "celebrity_hoax": ["die by", "death hoax", "pregnant", "resign"],
    "crypto_extreme": ["bitcoin", "btc", "eth", "crypto", "1 million", "100k"],
    "impossible_physics": ["time travel", "teleport", "faster than light"],
    "political_impossible": ["resign", "impeach", "removed from office"],
}


class MiracleStrategy(BaseStrategy):
    """
    Strategy 6: Miracle Market Scanner

    Bet NO on events that are almost certainly never going to happen.
    Not risk-free, but probability of loss is extremely low.

    Expected return: 1-6% per trade
    Expected win rate: 99%+
    Risk: Black swan events (very rare)
    """

    strategy_type = StrategyType.MIRACLE
    name = "Miracle Scanner"
    description = "Bet NO on impossible/absurd events (garbage collection)"

    def __init__(self):
        super().__init__()
        self.min_no_price = 0.90  # Only consider NO prices >= 90 cents
        self.max_no_price = 0.995  # Skip if NO is already at 99.5%+
        self.min_impossibility_score = 0.70  # Minimum confidence it's impossible

    def calculate_impossibility_score(self, question: str, end_date: Optional[datetime] = None) -> tuple[float, str, list[str]]:
        """
        Calculate how "impossible" an event seems based on the question text.

        Returns:
            - score (0-1): Higher = more impossible
            - category: What type of miracle event
            - reasons: Why we think it's impossible
        """
        question_lower = question.lower()
        score = 0.0
        reasons = []
        category = "unknown"

        # Check for miracle keywords
        for keyword, weight in MIRACLE_KEYWORDS.items():
            if re.search(keyword, question_lower):
                score = max(score, weight)
                reasons.append(f"Contains '{keyword}' (base score: {weight})")

                # Determine category
                for cat_name, cat_keywords in MIRACLE_CATEGORIES.items():
                    if any(k in keyword for k in cat_keywords) or any(k in question_lower for k in cat_keywords):
                        category = cat_name
                        break

        # Apply phrase modifiers
        for pattern, modifier in IMPOSSIBILITY_PHRASES:
            if re.search(pattern, question_lower):
                score += modifier
                if modifier > 0:
                    reasons.append(f"Phrase pattern '{pattern}' adds {modifier}")

        # Time-based impossibility boost
        if end_date:
            days_until = (end_date - datetime.utcnow()).days
            if days_until <= 1:
                score += 0.15
                reasons.append("Resolves within 1 day (very short window)")
            elif days_until <= 7:
                score += 0.10
                reasons.append("Resolves within 1 week (short window)")

        # Check for logical impossibilities (past events)
        if "2023" in question_lower or "2022" in question_lower:
            if datetime.utcnow().year >= 2024:
                score += 0.30
                reasons.append("Question references past year")

        # Cap score at 1.0
        score = min(score, 1.0)

        return score, category, reasons

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Detect miracle betting opportunities"""
        opportunities = []

        for market in markets:
            # Skip non-binary markets
            if len(market.outcome_prices) != 2:
                continue

            # Skip inactive or closed markets
            if market.closed or not market.active:
                continue

            # Get NO price (use live prices if available)
            no_price = market.no_price
            yes_price = market.yes_price

            if market.clob_token_ids and len(market.clob_token_ids) > 1:
                no_token = market.clob_token_ids[1]
                if no_token in prices:
                    no_price = prices[no_token].get("mid", no_price)
                yes_token = market.clob_token_ids[0]
                if yes_token in prices:
                    yes_price = prices[yes_token].get("mid", yes_price)

            # Only interested in markets where NO is expensive (event is unlikely)
            if no_price < self.min_no_price or no_price > self.max_no_price:
                continue

            # Calculate impossibility score
            impossibility_score, category, reasons = self.calculate_impossibility_score(
                market.question,
                market.end_date
            )

            # Skip if not confident enough it's impossible
            if impossibility_score < self.min_impossibility_score:
                continue

            # Calculate profit metrics
            # We buy NO at current price, get $1 when event doesn't happen
            total_cost = no_price
            expected_payout = 1.0
            gross_profit = expected_payout - total_cost
            fee = expected_payout * self.fee
            net_profit = gross_profit - fee
            roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

            # Skip if ROI too low after fees
            if roi < 0.5:  # At least 0.5% profit
                continue

            # Risk assessment for miracle strategy is different
            # Lower NO price = higher risk (less consensus that event won't happen)
            risk_score = 1.0 - no_price  # Risk inversely proportional to NO price
            risk_score = max(0.05, risk_score)  # Minimum 5% risk always

            risk_factors = [
                f"Impossibility confidence: {impossibility_score:.0%}",
                f"Category: {category}",
            ]
            risk_factors.extend(reasons[:3])  # Add top 3 reasons

            if no_price < 0.95:
                risk_factors.append("NO price below 95% - higher uncertainty")

            # Calculate max position based on liquidity
            min_liquidity = market.liquidity
            max_position = min_liquidity * 0.05  # Conservative: 5% of liquidity

            positions = [
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "market": market.question[:50],
                    "price": no_price,
                    "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
                }
            ]

            opp = ArbitrageOpportunity(
                strategy=self.strategy_type,
                title=f"Miracle: {market.question[:60]}...",
                description=f"Buy NO @ ${no_price:.3f} | {category} | Impossibility: {impossibility_score:.0%}",
                total_cost=total_cost,
                expected_payout=expected_payout,
                gross_profit=gross_profit,
                fee=fee,
                net_profit=net_profit,
                roi_percent=roi,
                risk_score=risk_score,
                risk_factors=risk_factors,
                markets=[{
                    "id": market.id,
                    "question": market.question,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "liquidity": market.liquidity
                }],
                min_liquidity=min_liquidity,
                max_position_size=max_position,
                resolution_date=market.end_date,
                positions_to_take=positions
            )

            opportunities.append(opp)

        # Sort by ROI (higher profit opportunities first)
        opportunities.sort(key=lambda x: x.roi_percent, reverse=True)

        return opportunities

    def find_stale_markets(
        self,
        markets: list[Market],
        resolved_events: list[str]
    ) -> list[ArbitrageOpportunity]:
        """
        Find markets that are now logically impossible due to resolved events.

        This implements the "logical holes" strategy - when Event A resolves,
        related Event B markets may not update immediately.

        Args:
            markets: Active markets to check
            resolved_events: List of event slugs/titles that have resolved

        Returns:
            Opportunities where the market should now be 100% NO
        """
        # This is a more advanced feature that requires:
        # 1. Tracking relationships between markets
        # 2. Understanding logical dependencies
        # 3. Monitoring resolution events

        # For now, return empty - this can be enhanced later
        # with NLP or manual relationship mapping
        return []
