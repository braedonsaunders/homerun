from __future__ import annotations

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy


class MutuallyExclusiveStrategy(BaseStrategy):
    """
    Strategy 2: Mutually Exclusive Arbitrage

    Find two events where only one can be true, buy YES on both for < $1

    WARNING: This strategy has SIGNIFICANT RISKS!

    KNOWN ISSUES:
    1. Third-party candidates: "Democrats win" vs "Republicans win" ignores independents
    2. Pattern matching false positives: Keywords may match unrelated markets
    3. Draw/tie outcomes: In sports, both "win" options could lose if there's a draw
    4. Different thresholds: "above $100K" vs "below $100K" leaves a gap at exactly $100K

    ONLY use this strategy when:
    - The event has EXACTLY 2 markets (enforced)
    - You have MANUALLY verified the outcomes are truly exhaustive
    - No third option exists (no independents, no draws, no boundary cases)

    Example of FAILURE:
    - "Democrats win 2028" YES: $0.45
    - "Republicans win 2028" YES: $0.52
    - Independent candidate wins
    - BOTH positions resolve to $0

    Example (true binary):
    - "Bill passes" YES: $0.45
    - "Bill fails" YES: $0.52
    - Total: $0.97 (assuming pass/fail is truly exhaustive)
    - One MUST win = $1.00 payout
    - Profit: $0.03
    """

    strategy_type = StrategyType.MUTUALLY_EXCLUSIVE
    name = "Mutually Exclusive"
    description = "Two-market events - REQUIRES MANUAL VERIFICATION of exhaustiveness"

    # Pairs of mutually exclusive patterns to look for
    # WARNING: These are HEURISTICS that may produce false positives!
    EXCLUSIVE_PATTERNS = [
        # Political - RISKY: ignores third-party candidates!
        (["democrat", "biden", "harris", "democratic"], ["republican", "trump", "gop"]),
        # REMOVED: Too many false positives
        # (["yes", "will"], ["no", "won't", "will not"]),
        # Sports - RISKY: ignores draws/ties!
        (["home", "team a"], ["away", "team b"]),
        # REMOVED: Boundary case issues (exactly equal)
        # (["above", "over", "more than", "higher"], ["below", "under", "less than", "lower"]),
        # Time-based - RISKY: "on the date" could be neither before nor after
        # (["before", "by"], ["after", "not by"]),
        # Win/lose - RISKY: ignores draws
        (["win", "victory"], ["lose", "defeat"]),
    ]

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        # Group markets by potential mutual exclusivity
        # This is a simplified approach - real implementation would need NLP

        # Look within events first (related markets)
        for event in events:
            if len(event.markets) < 2:
                continue

            opps = self._find_exclusive_pairs_in_event(event, prices)
            opportunities.extend(opps)

        # Also check across all markets for obvious pairs
        opps = self._find_exclusive_pairs_across_markets(markets, prices)
        opportunities.extend(opps)

        return opportunities

    def _find_exclusive_pairs_in_event(
        self, event: Event, prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Find mutually exclusive pairs within an event"""
        opportunities = []
        active_markets = [m for m in event.markets if m.active and not m.closed]

        # IMPORTANT: Only apply to events with EXACTLY 2 markets
        # If there are more than 2, it's a multi-outcome scenario (use must_happen/negrisk instead)
        # Two random candidates from a multi-candidate race are NOT mutually exclusive
        # because a third candidate could win
        if len(active_markets) != 2:
            return opportunities

        market_a, market_b = active_markets
        if self._are_mutually_exclusive(market_a, market_b):
            opp = self._check_pair(market_a, market_b, prices, event)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _find_exclusive_pairs_across_markets(
        self, markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Find mutually exclusive pairs across all markets"""
        opportunities = []

        # This is expensive, so we limit to high-volume markets
        high_volume = sorted(markets, key=lambda m: m.volume, reverse=True)[:100]

        for i, market_a in enumerate(high_volume):
            for market_b in high_volume[i + 1 :]:
                if self._are_mutually_exclusive(market_a, market_b):
                    opp = self._check_pair(market_a, market_b, prices)
                    if opp:
                        opportunities.append(opp)

        return opportunities

    def _are_mutually_exclusive(self, market_a: Market, market_b: Market) -> bool:
        """Check if two markets are mutually exclusive based on patterns"""
        q_a = market_a.question.lower()
        q_b = market_b.question.lower()

        for pattern_a, pattern_b in self.EXCLUSIVE_PATTERNS:
            a_matches_first = any(p in q_a for p in pattern_a)
            b_matches_second = any(p in q_b for p in pattern_b)
            a_matches_second = any(p in q_a for p in pattern_b)
            b_matches_first = any(p in q_b for p in pattern_a)

            # Check if they're opposite patterns
            if (a_matches_first and b_matches_second) or (
                a_matches_second and b_matches_first
            ):
                # Additional check: questions should be about the same topic
                # Simple heuristic: share significant words
                words_a = set(q_a.split())
                words_b = set(q_b.split())
                common = words_a & words_b
                # Remove common stop words
                stop_words = {
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
                }
                common = common - stop_words

                if len(common) >= 2:
                    return True

        return False

    def _is_election_pair(self, market_a: Market, market_b: Market) -> bool:
        """Check if a pair of markets is an election/political race.

        Election markets with only 2 candidates listed (Dem vs Rep) are
        NEVER truly exhaustive — independent/third-party candidates can win.
        These should be rejected outright, not just warned about.
        """
        q_combined = (market_a.question + market_b.question).lower()
        election_keywords = [
            "election", "house", "senate", "governor", "congress",
            "president", "democrat", "republican", "gop",
            "primary", "nominee", "caucus", "special election",
        ]
        return any(kw in q_combined for kw in election_keywords)

    def _check_pair(
        self,
        market_a: Market,
        market_b: Market,
        prices: dict[str, dict],
        event: Event = None,
    ) -> ArbitrageOpportunity | None:
        """Check if a pair offers arbitrage opportunity"""

        # Reject election markets outright — two candidates are never exhaustive
        if self._is_election_pair(market_a, market_b):
            return None

        # Get YES prices
        yes_a = market_a.yes_price
        yes_b = market_b.yes_price

        # Use live prices if available
        if market_a.clob_token_ids:
            token = market_a.clob_token_ids[0]
            if token in prices:
                yes_a = prices[token].get("mid", yes_a)

        if market_b.clob_token_ids:
            token = market_b.clob_token_ids[0]
            if token in prices:
                yes_b = prices[token].get("mid", yes_b)

        total_cost = yes_a + yes_b

        # Require total very close to 1.0 — wider spreads indicate
        # non-exhaustive outcomes rather than mispricing
        if total_cost < settings.NEGRISK_MIN_TOTAL_YES:
            return None

        if total_cost >= 1.0:
            return None

        positions = [
            {
                "action": "BUY",
                "outcome": "YES",
                "market": market_a.question[:50],
                "price": yes_a,
                "token_id": market_a.clob_token_ids[0]
                if market_a.clob_token_ids
                else None,
            },
            {
                "action": "BUY",
                "outcome": "YES",
                "market": market_b.question[:50],
                "price": yes_b,
                "token_id": market_b.clob_token_ids[0]
                if market_b.clob_token_ids
                else None,
            },
        ]

        opp = self.create_opportunity(
            title=f"⚠️ Exclusive: {market_a.question[:20]}... vs {market_b.question[:20]}...",
            description=f"VERIFY MANUALLY: Check no third outcome exists. YES on both: ${yes_a:.3f} + ${yes_b:.3f} = ${total_cost:.3f}",
            total_cost=total_cost,
            markets=[market_a, market_b],
            positions=positions,
            event=event,
        )

        if opp:
            opp.risk_factors.insert(
                0, "REQUIRES MANUAL VERIFICATION - check for third-party outcomes"
            )
            q_combined = (market_a.question + market_b.question).lower()
            if any(p in q_combined for p in ["win", "lose", "victory", "defeat"]):
                opp.risk_factors.insert(1, "Win/lose market: Draw/tie outcome possible")

        return opp
