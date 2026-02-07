from __future__ import annotations

from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


class ContradictionStrategy(BaseStrategy):
    """
    Strategy 3: Contradiction Arbitrage

    Find two markets that say opposite things - buy YES in one, NO in the other

    WARNING: This strategy has SIGNIFICANT RISKS and may produce FALSE POSITIVES!

    KNOWN ISSUES:
    1. "Above X" vs "Below X" are NOT exhaustive - value could be EXACTLY X
    2. Timeframe mismatches: "by March" vs "in March" are different conditions
    3. Different thresholds: "above $100K" vs "below $95K" leaves a gap
    4. Context differences: Markets may look contradictory but refer to different scenarios

    ONLY use this strategy when you can MANUALLY VERIFY:
    - The markets cover ALL possible outcomes (no gaps)
    - The timeframes and conditions are identical
    - One outcome MUST be true (exhaustive)

    Example that LOOKS like arbitrage but ISN'T:
    - Market A: "BTC above $100K by March" YES: $0.30
    - Market B: "BTC below $100K in March" YES: $0.65
    - PROBLEM: What if BTC is exactly $100K? Both could resolve NO!
    - PROBLEM: "by March" vs "in March" are different timeframes!
    """

    strategy_type = StrategyType.CONTRADICTION
    name = "Contradiction"
    description = "Two markets say opposite things - REQUIRES MANUAL VERIFICATION"

    # Contradiction patterns (word pairs that indicate opposite meanings)
    # WARNING: These are HEURISTICS that may produce false positives!
    # "above/below" is particularly dangerous - doesn't cover "exactly equal"
    CONTRADICTION_PAIRS = [
        # DANGEROUS: These don't cover the boundary case (exactly equal)
        # ("above", "below"),  # REMOVED - not exhaustive
        # ("over", "under"),   # REMOVED - not exhaustive
        # ("more", "less"),    # REMOVED - not exhaustive
        # ("higher", "lower"), # REMOVED - not exhaustive
        # SAFER: These are more likely to be true contradictions
        ("before", "after"),  # Still risky if "on the date" is possible
        ("win", "lose"),  # Usually exhaustive in head-to-head
        ("pass", "fail"),  # Usually binary
        ("approve", "reject"),  # Usually binary
    ]

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        # Index markets by key topics for efficient matching
        topic_index = self._build_topic_index(markets)

        # Find contradicting pairs
        checked_pairs = set()

        for market in markets:
            if market.closed or not market.active:
                continue

            # Find potential contradictions
            candidates = self._find_contradiction_candidates(market, topic_index)

            for candidate in candidates:
                # Avoid duplicate checks
                pair_key = tuple(sorted([market.id, candidate.id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Check for actual contradiction
                if self._are_contradictory(market, candidate):
                    opp = self._check_contradiction(market, candidate, prices)
                    if opp:
                        opportunities.append(opp)

        return opportunities

    def _build_topic_index(self, markets: list[Market]) -> dict[str, list[Market]]:
        """Index markets by key topic words for efficient lookup"""
        index = {}

        # Common words to ignore
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
            "market",
            "price",
            "2025",
            "2026",
            "2027",
        }

        for market in markets:
            if market.closed or not market.active:
                continue

            words = market.question.lower().split()
            key_words = [w for w in words if w not in stop_words and len(w) > 3]

            for word in key_words:
                if word not in index:
                    index[word] = []
                index[word].append(market)

        return index

    def _find_contradiction_candidates(
        self, market: Market, topic_index: dict[str, list[Market]]
    ) -> list[Market]:
        """Find markets that might contradict this one"""
        candidates = {}  # Use dict keyed by ID since Market isn't hashable
        question = market.question.lower()

        # Get markets sharing topic words
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
            "of",
        }
        words = [w for w in question.split() if w not in stop_words and len(w) > 3]

        for word in words:
            if word in topic_index:
                for m in topic_index[word]:
                    if m.id != market.id:
                        candidates[m.id] = m

        return list(candidates.values())

    def _are_contradictory(self, market_a: Market, market_b: Market) -> bool:
        """Check if two markets are contradictory"""
        q_a = market_a.question.lower()
        q_b = market_b.question.lower()

        # Check for contradiction patterns
        for word_a, word_b in self.CONTRADICTION_PAIRS:
            # Market A has word_a and Market B has word_b
            if word_a in q_a and word_b in q_b:
                # Verify they share topic context
                if self._share_topic(q_a, q_b):
                    return True

            # Or vice versa
            if word_b in q_a and word_a in q_b:
                if self._share_topic(q_a, q_b):
                    return True

        return False

    def _share_topic(self, q_a: str, q_b: str) -> bool:
        """Check if two questions share enough topic context"""
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
            "of",
            "for",
            "with",
            "above",
            "below",
            "over",
            "under",
            "more",
            "less",
        }

        words_a = set(q_a.split()) - stop_words
        words_b = set(q_b.split()) - stop_words

        # Filter to meaningful words (length > 3)
        words_a = {w for w in words_a if len(w) > 3}
        words_b = {w for w in words_b if len(w) > 3}

        common = words_a & words_b

        # Need at least 2 common topic words
        return len(common) >= 2

    def _check_contradiction(
        self, market_a: Market, market_b: Market, prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """
        Check contradiction arbitrage.

        If markets contradict, we can either:
        1. Buy YES on both (if one saying "above" and one "below")
        2. Buy YES on one and NO on another

        We check both approaches and take the profitable one.
        """
        # Approach 1: Buy YES on both
        yes_a = market_a.yes_price
        yes_b = market_b.yes_price

        if market_a.clob_token_ids:
            token = market_a.clob_token_ids[0]
            if token in prices:
                yes_a = prices[token].get("mid", yes_a)

        if market_b.clob_token_ids:
            token = market_b.clob_token_ids[0]
            if token in prices:
                yes_b = prices[token].get("mid", yes_b)

        cost_both_yes = yes_a + yes_b

        if cost_both_yes < 1.0:
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
                title=f"⚠️ Contradiction: {market_a.question[:25]}...",
                description=f"VERIFY MANUALLY: Check markets are truly exhaustive. YES+YES: ${yes_a:.3f} + ${yes_b:.3f} = ${cost_both_yes:.3f}",
                total_cost=cost_both_yes,
                markets=[market_a, market_b],
                positions=positions,
            )
            # Add extra risk factor for contradiction strategy
            if opp:
                opp.risk_factors.insert(
                    0, "⚠️ REQUIRES MANUAL VERIFICATION - may not be exhaustive"
                )
            return opp

        # Approach 2: Buy YES on A, NO on B
        no_b = market_b.no_price
        if len(market_b.clob_token_ids) > 1:
            token = market_b.clob_token_ids[1]
            if token in prices:
                no_b = prices[token].get("mid", no_b)

        cost_yes_no = yes_a + no_b

        if cost_yes_no < 1.0:
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
                    "outcome": "NO",
                    "market": market_b.question[:50],
                    "price": no_b,
                    "token_id": market_b.clob_token_ids[1]
                    if len(market_b.clob_token_ids) > 1
                    else None,
                },
            ]

            opp = self.create_opportunity(
                title=f"⚠️ Contradiction: {market_a.question[:25]}...",
                description=f"VERIFY MANUALLY: Check markets are truly exhaustive. YES+NO: ${yes_a:.3f} + ${no_b:.3f} = ${cost_yes_no:.3f}",
                total_cost=cost_yes_no,
                markets=[market_a, market_b],
                positions=positions,
            )
            # Add extra risk factor for contradiction strategy
            if opp:
                opp.risk_factors.insert(
                    0, "⚠️ REQUIRES MANUAL VERIFICATION - may not be exhaustive"
                )
            return opp

        return None
