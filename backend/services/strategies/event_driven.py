"""
Strategy: Event-Driven Arbitrage

Detects markets where prices haven't adjusted to recent significant
moves in related markets. When a "catalyst" market moves sharply,
related markets often lag behind.

Example:
- "Fed raises rates" jumps from 30% to 70% (catalyst)
- "Mortgage rates increase" still at 35% (should be higher)
- Buy YES on the lagging market

This exploits the information propagation delay across markets.
"""

import time
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

# Stop words for keyword extraction
_STOP_WORDS = frozenset(
    {
        "will",
        "the",
        "a",
        "an",
        "in",
        "on",
        "by",
        "to",
        "be",
        "is",
        "are",
        "of",
        "for",
        "with",
        "this",
        "that",
        "it",
        "at",
        "from",
        "or",
        "and",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "was",
        "were",
        "been",
        "not",
        "no",
        "yes",
        "what",
        "when",
        "how",
        "who",
        "which",
        "if",
        "but",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "than",
        "then",
        "so",
        "as",
        "its",
        "their",
        "there",
        "they",
        "them",
        "market",
        "price",
        "2024",
        "2025",
        "2026",
        "2027",
    }
)

# Minimum word length for keyword significance
_MIN_WORD_LENGTH = 3
# Minimum shared keywords for relatedness
_MIN_SHARED_KEYWORDS = 2
# Catalyst move threshold (10%)
_CATALYST_THRESHOLD = 0.10
# Minimum proportional lag to flag an opportunity
_MIN_LAG_RATIO = 0.3


class EventDrivenStrategy(BaseStrategy):
    """
    Event-Driven Arbitrage: Exploit price lag after significant market moves.

    Tracks prices across scans and detects "catalyst" moves (>10% change).
    When a catalyst fires, related markets that didn't move proportionally
    are flagged as lagging opportunities.

    This is a statistical edge strategy, not risk-free arbitrage.
    """

    strategy_type = StrategyType.EVENT_DRIVEN
    name = "Event-Driven"
    description = "Exploit price lag after significant market moves"

    def __init__(self):
        super().__init__()
        # market_id -> [(timestamp, yes_price)]
        self._price_history: dict[str, list[tuple[float, float]]] = {}
        # market_id -> event_id (for fast lookup)
        self._market_to_event: dict[str, str] = {}
        # market_id -> category
        self._market_to_category: dict[str, str] = {}
        # market_id -> set of keywords
        self._market_keywords: dict[str, set[str]] = {}
        # market_id -> Market (latest snapshot)
        self._market_cache: dict[str, Market] = {}

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        if not settings.EVENT_DRIVEN_ENABLED:
            return []

        now = time.time()
        opportunities: list[ArbitrageOpportunity] = []

        # Build event/category/keyword mappings from the current scan
        event_market_ids: dict[str, list[str]] = {}
        for event in events:
            for m in event.markets:
                self._market_to_event[m.id] = event.id
                self._market_to_category[m.id] = event.category or ""
                if event.id not in event_market_ids:
                    event_market_ids[event.id] = []
                event_market_ids[event.id].append(m.id)

        # Record current prices and extract keywords
        for market in markets:
            if market.closed or not market.active:
                continue

            yes_price = self._get_live_price(market, prices)
            self._market_cache[market.id] = market
            self._market_keywords[market.id] = self._extract_keywords(market.question)

            if market.id not in self._price_history:
                self._price_history[market.id] = []
            self._price_history[market.id].append((now, yes_price))

            # Keep history bounded (last 100 data points)
            if len(self._price_history[market.id]) > 100:
                self._price_history[market.id] = self._price_history[market.id][-100:]

        # Need at least 2 scans to detect moves
        catalysts = self._detect_catalysts()
        if not catalysts:
            return []

        logger.info(f"Event-Driven: detected {len(catalysts)} catalyst move(s)")

        # For each catalyst, find related lagging markets
        for catalyst_id, move_direction, move_magnitude in catalysts:
            related = self._find_related_markets(catalyst_id, event_market_ids)

            for related_id in related:
                if related_id == catalyst_id:
                    continue

                history = self._price_history.get(related_id, [])
                if len(history) < 2:
                    continue

                # Check if this related market also moved
                related_move = history[-1][1] - history[-2][1]
                related_magnitude = abs(related_move)

                # If the related market barely moved relative to catalyst, it's lagging
                if related_magnitude < move_magnitude * _MIN_LAG_RATIO:
                    opp = self._create_lag_opportunity(
                        catalyst_id,
                        related_id,
                        move_direction,
                        move_magnitude,
                        related_move,
                        prices,
                    )
                    if opp:
                        opportunities.append(opp)

        return opportunities

    def _get_live_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Get the best available YES price for a market."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            token = market.clob_token_ids[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    def _extract_keywords(self, question: str) -> set[str]:
        """Extract significant keywords from a market question."""
        words = question.lower().split()
        # Strip punctuation from each word
        cleaned = set()
        for w in words:
            w = w.strip("?.,!:;\"'()[]{}#")
            if len(w) >= _MIN_WORD_LENGTH and w not in _STOP_WORDS:
                cleaned.add(w)
        return cleaned

    def _detect_catalysts(self) -> list[tuple[str, float, float]]:
        """
        Detect catalyst moves: markets that moved > 10% since last scan.

        Returns list of (market_id, move_direction, move_magnitude).
        move_direction is positive for upward moves, negative for downward.
        """
        catalysts = []
        for market_id, history in self._price_history.items():
            if len(history) < 2:
                continue
            prev_price = history[-2][1]
            curr_price = history[-1][1]
            move = curr_price - prev_price

            if abs(move) >= _CATALYST_THRESHOLD:
                catalysts.append((market_id, move, abs(move)))

        return catalysts

    def _find_related_markets(
        self, catalyst_id: str, event_market_ids: dict[str, list[str]]
    ) -> list[str]:
        """
        Find markets related to the catalyst via:
        1. Same event
        2. Same category
        3. Shared keywords (2+ significant words)
        """
        related: dict[str, bool] = {}

        catalyst_event = self._market_to_event.get(catalyst_id)
        catalyst_category = self._market_to_category.get(catalyst_id, "")
        catalyst_keywords = self._market_keywords.get(catalyst_id, set())

        # 1. Same event
        if catalyst_event and catalyst_event in event_market_ids:
            for mid in event_market_ids[catalyst_event]:
                if mid != catalyst_id:
                    related[mid] = True

        # 2. Same category and 3. Shared keywords
        for mid in self._price_history:
            if mid == catalyst_id or mid in related:
                continue

            # Same category (non-empty)
            mid_category = self._market_to_category.get(mid, "")
            if catalyst_category and mid_category and catalyst_category == mid_category:
                related[mid] = True
                continue

            # Shared keywords
            mid_keywords = self._market_keywords.get(mid, set())
            shared = catalyst_keywords & mid_keywords
            if len(shared) >= _MIN_SHARED_KEYWORDS:
                related[mid] = True

        return list(related.keys())

    def _create_lag_opportunity(
        self,
        catalyst_id: str,
        lagging_id: str,
        catalyst_direction: float,
        catalyst_magnitude: float,
        lagging_move: float,
        prices: dict[str, dict],
    ) -> Optional[ArbitrageOpportunity]:
        """Create an opportunity from a catalyst-lag pair."""
        catalyst_market = self._market_cache.get(catalyst_id)
        lagging_market = self._market_cache.get(lagging_id)

        if not catalyst_market or not lagging_market:
            return None
        if lagging_market.closed or not lagging_market.active:
            return None
        if len(lagging_market.outcome_prices) < 2:
            return None

        lagging_yes = self._get_live_price(lagging_market, prices)
        lagging_no = lagging_market.no_price
        if lagging_market.clob_token_ids and len(lagging_market.clob_token_ids) > 1:
            token = lagging_market.clob_token_ids[1]
            if token in prices:
                lagging_no = prices[token].get("mid", lagging_no)

        # Determine the direction the lagging market should move:
        # If catalyst moved UP, related market should also move UP (buy YES)
        # If catalyst moved DOWN, related market should move DOWN (buy NO)
        if catalyst_direction > 0:
            # Catalyst moved up -> lagging should move up -> buy YES
            action = "BUY"
            outcome = "YES"
            entry_price = lagging_yes
            token_id = (
                lagging_market.clob_token_ids[0]
                if lagging_market.clob_token_ids
                else None
            )
        else:
            # Catalyst moved down -> lagging should move down -> buy NO
            action = "BUY"
            outcome = "NO"
            entry_price = lagging_no
            token_id = (
                lagging_market.clob_token_ids[1]
                if lagging_market.clob_token_ids
                and len(lagging_market.clob_token_ids) > 1
                else None
            )

        # The "total cost" for a directional bet is the entry price
        total_cost = entry_price

        # Skip if price is already extreme (no room for movement)
        if entry_price < 0.05 or entry_price > 0.95:
            return None

        # Risk score: 0.35 - 0.50 depending on magnitude of catalyst
        base_risk = 0.35
        # Larger catalyst moves are more convincing -> slightly lower risk
        risk_adjustment = max(0, 0.15 - catalyst_magnitude * 0.5)
        risk_score = min(base_risk + risk_adjustment, 0.50)

        catalyst_q = catalyst_market.question[:50]
        lagging_q = lagging_market.question[:50]
        direction_word = "UP" if catalyst_direction > 0 else "DOWN"

        positions = [
            {
                "action": action,
                "outcome": outcome,
                "market": lagging_q,
                "price": entry_price,
                "token_id": token_id,
                "rationale": (
                    f"Catalyst '{catalyst_q}' moved {direction_word} "
                    f"{catalyst_magnitude:.1%}, lagging market should follow"
                ),
            },
        ]

        # Find the event for the lagging market
        event = self._find_event_for_market(lagging_market.id)

        opp = self.create_opportunity(
            title=f"Event-Driven: {lagging_q}...",
            description=(
                f"Catalyst move detected: '{catalyst_q}' moved "
                f"{direction_word} {catalyst_magnitude:.1%}. "
                f"Related market '{lagging_q}' lagged "
                f"(moved only {abs(lagging_move):.1%}). "
                f"Buy {outcome} at ${entry_price:.3f}."
            ),
            total_cost=total_cost,
            markets=[lagging_market],
            positions=positions,
            event=event,
        )

        if opp:
            # Override risk score to our statistical range
            opp.risk_score = risk_score
            opp.risk_factors.append(
                f"Statistical edge (not risk-free): catalyst {catalyst_magnitude:.1%} move"
            )
            opp.risk_factors.append(
                "Price lag may reflect legitimate market disagreement"
            )

        return opp

    def _find_event_for_market(self, market_id: str) -> Optional[Event]:
        """Find the Event object associated with a market_id (best effort)."""
        # We don't cache Event objects in this strategy, so return None.
        # The create_opportunity method handles None events gracefully.
        return None
