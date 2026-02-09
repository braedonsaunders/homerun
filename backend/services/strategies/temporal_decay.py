"""
Strategy: Temporal Decay Arbitrage

Markets with time-based questions should follow predictable decay curves,
similar to options theta. When prices deviate from the expected curve,
there's a trading opportunity.

Key insight: A "BTC above $100K by June" market should lose value
as time passes without BTC reaching $100K. If it doesn't decay
as expected, it's mispriced.

Uses a modified Black-Scholes-like decay model adapted for binary outcomes.
"""

import re
import time
from datetime import datetime, timezone
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, utcnow, make_aware
from utils.logger import get_logger

logger = get_logger(__name__)

# Regex patterns to extract deadlines from market questions
_DEADLINE_PATTERNS = [
    # "by December 2025", "by Dec 2025", "by December 31, 2025"
    re.compile(
        r"\bby\s+(\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    # "before March 2026", "before Mar 1, 2026"
    re.compile(
        r"\bbefore\s+(\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    # "in January 2026", "in Jan 2026"
    re.compile(
        r"\bin\s+(\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    # "by end of 2026", "by the end of 2025"
    re.compile(
        r"\bby\s+(?:the\s+)?end\s+of\s+(\d{4})",
        re.IGNORECASE,
    ),
]

# Month name -> number mapping
_MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# Default decay rate (square root decay for most markets)
_DEFAULT_DECAY_RATE = 0.5

# Minimum deviation from expected price to flag opportunity (5%)
_MIN_DEVIATION = 0.05

# Only consider markets within 30 days of deadline (steepest decay)
_MAX_DAYS_TO_DEADLINE = 30

# Minimum days to deadline (avoid markets about to expire)
_MIN_DAYS_TO_DEADLINE = 1


class TemporalDecayStrategy(BaseStrategy):
    """
    Temporal Decay Arbitrage: Exploit time-decay mispricing in deadline markets.

    Identifies markets with time-based questions (e.g. "X by [date]") and
    calculates expected price decay using a square-root decay model. When
    actual prices deviate significantly from expected decay, flags as
    opportunity.

    This is a statistical edge strategy, not risk-free arbitrage.
    """

    strategy_type = StrategyType.TEMPORAL_DECAY
    name = "Temporal Decay"
    description = "Exploit time-decay mispricing in deadline markets"

    def __init__(self):
        super().__init__()
        # market_id -> [(timestamp, yes_price)] for tracking decay over time
        self._price_history: dict[str, list[tuple[float, float]]] = {}
        # market_id -> (deadline_dt, first_seen_price) for decay calculation
        self._market_baselines: dict[str, tuple[datetime, float]] = {}

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        if not settings.TEMPORAL_DECAY_ENABLED:
            return []

        now = utcnow()
        scan_time = time.time()
        opportunities: list[ArbitrageOpportunity] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) < 2:
                continue

            yes_price = self._get_live_price(market, prices)

            # Record price history
            if market.id not in self._price_history:
                self._price_history[market.id] = []
            self._price_history[market.id].append((scan_time, yes_price))
            # Keep bounded
            if len(self._price_history[market.id]) > 100:
                self._price_history[market.id] = self._price_history[market.id][-100:]

            # Try to extract a deadline from the question
            deadline = self._extract_deadline(market)
            if deadline is None:
                continue

            # Calculate days remaining
            days_remaining = (deadline - now).total_seconds() / 86400.0
            if days_remaining < _MIN_DAYS_TO_DEADLINE:
                continue  # Too close to expiry, too risky
            if days_remaining > _MAX_DAYS_TO_DEADLINE:
                continue  # Too far from deadline, decay is gentle

            # Establish or retrieve baseline for this market
            if market.id not in self._market_baselines:
                # Use current price as proxy for initial price on first scan
                initial_price = max(yes_price, 0.10)  # Floor at 0.10
                self._market_baselines[market.id] = (deadline, initial_price)
            else:
                # Use stored baseline
                _, initial_price = self._market_baselines[market.id]

            # Calculate total days from first observation to deadline
            first_seen_time = self._price_history[market.id][0][0]
            first_seen_dt = datetime.fromtimestamp(first_seen_time, tz=timezone.utc)
            total_days = max((deadline - first_seen_dt).total_seconds() / 86400.0, 1.0)

            # Expected decay: p_expected = p_initial * (days_remaining / total_days)^decay_rate
            ratio = min(days_remaining / total_days, 1.0)
            p_expected = initial_price * (ratio**_DEFAULT_DECAY_RATE)

            # Compare actual price to expected
            deviation = yes_price - p_expected

            if abs(deviation) < _MIN_DEVIATION:
                continue  # Within normal range

            # Build opportunity
            opp = self._create_decay_opportunity(
                market,
                yes_price,
                p_expected,
                deviation,
                days_remaining,
                deadline,
                prices,
            )
            if opp:
                opportunities.append(opp)

        if opportunities:
            logger.info(
                f"Temporal Decay: found {len(opportunities)} decay mispricing(s)"
            )

        return opportunities

    def _get_live_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Get the best available YES price for a market."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            token = market.clob_token_ids[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    def _extract_deadline(self, market: Market) -> Optional[datetime]:
        """
        Extract a deadline datetime from the market question text.

        First checks the market's end_date field, then tries regex extraction
        from the question text.
        """
        # Prefer the market's own end_date if available
        if market.end_date:
            return make_aware(market.end_date)

        question = market.question

        for pattern in _DEADLINE_PATTERNS:
            match = pattern.search(question)
            if match:
                date_str = match.group(1).strip().rstrip(",")
                parsed = self._parse_date_string(date_str)
                if parsed:
                    return parsed

        return None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string extracted from a market question."""
        parts = date_str.lower().split()

        if len(parts) == 1:
            # Just a year like "2026"
            try:
                year = int(parts[0])
                return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            except ValueError:
                return None

        if len(parts) >= 2:
            month_str = parts[0]
            month = _MONTH_MAP.get(month_str)
            if month is None:
                return None

            if len(parts) == 2:
                # "March 2026" -> end of that month
                try:
                    year = int(parts[1])
                    # End of month: use day 28 as safe default
                    # (approximation, but close enough for decay calculation)
                    day = 28
                    return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
                except ValueError:
                    return None

            if len(parts) >= 3:
                # "March 15 2026" or "March 15, 2026"
                try:
                    day_str = parts[1].rstrip(",")
                    day = int(day_str)
                    year = int(parts[2])
                    return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
                except ValueError:
                    return None

        return None

    def _create_decay_opportunity(
        self,
        market: Market,
        actual_price: float,
        expected_price: float,
        deviation: float,
        days_remaining: float,
        deadline: datetime,
        prices: dict[str, dict],
    ) -> Optional[ArbitrageOpportunity]:
        """Create an opportunity from a temporal decay deviation."""
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            token = market.clob_token_ids[1]
            if token in prices:
                no_price = prices[token].get("mid", no_price)

        if deviation > 0:
            # Market is OVERPRICED relative to decay expectation -> buy NO
            action = "BUY"
            outcome = "NO"
            entry_price = no_price
            token_id = (
                market.clob_token_ids[1]
                if market.clob_token_ids and len(market.clob_token_ids) > 1
                else None
            )
            direction_desc = "overpriced"
        else:
            # Market is UNDERPRICED relative to decay expectation -> buy YES
            action = "BUY"
            outcome = "YES"
            entry_price = actual_price
            token_id = market.clob_token_ids[0] if market.clob_token_ids else None
            direction_desc = "underpriced"

        total_cost = entry_price

        # Skip extreme prices
        if entry_price < 0.05 or entry_price > 0.95:
            return None

        # Risk score: 0.35 - 0.50
        # Closer to deadline = steeper decay = higher confidence = lower risk
        base_risk = 0.50
        # Reduce risk as deviation increases (stronger signal)
        deviation_adjustment = min(abs(deviation) * 2.0, 0.15)
        risk_score = max(base_risk - deviation_adjustment, 0.35)

        question_short = market.question[:50]
        positions = [
            {
                "action": action,
                "outcome": outcome,
                "market": question_short,
                "price": entry_price,
                "token_id": token_id,
                "rationale": (
                    f"Expected decay price: ${expected_price:.3f}, "
                    f"actual: ${actual_price:.3f} "
                    f"({direction_desc} by {abs(deviation):.3f})"
                ),
            },
        ]

        opp = self.create_opportunity(
            title=f"Temporal Decay: {question_short}...",
            description=(
                f"Deadline market {direction_desc} vs expected decay curve. "
                f"YES actual: ${actual_price:.3f}, expected: ${expected_price:.3f} "
                f"(deviation: {abs(deviation):.3f}). "
                f"{days_remaining:.1f} days to deadline. "
                f"Buy {outcome} at ${entry_price:.3f}."
            ),
            total_cost=total_cost,
            markets=[market],
            positions=positions,
        )

        if opp:
            # Override risk score to our statistical range
            opp.risk_score = risk_score
            opp.risk_factors.append(
                f"Statistical edge (not risk-free): "
                f"decay deviation {abs(deviation):.1%}"
            )
            opp.risk_factors.append(
                f"Deadline in {days_remaining:.0f} days "
                f"({deadline.strftime('%Y-%m-%d')})"
            )
            if days_remaining < 7:
                opp.risk_factors.append(
                    "Near-deadline: steep decay but higher event uncertainty"
                )

        return opp
