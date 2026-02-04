"""
Strategy 7: Combinatorial Arbitrage

Detects arbitrage opportunities across multiple dependent markets using
integer programming and Bregman projection.

Research paper findings:
- 17,218 conditions examined
- 1,576 dependent market pairs in 2024 US election
- 13 confirmed exploitable cross-market arbitrage opportunities
- $95,634 extracted from combinatorial arbitrage alone

Key insight: Markets that look independent may have logical dependencies.
"Trump wins PA" implies "Republican wins PA" - if you can buy both
contradicting positions for less than $1, arbitrage exists.

This strategy:
1. Uses LLM/heuristics to detect market dependencies
2. Builds constraint matrix for valid outcome combinations
3. Solves integer program to find minimum-cost coverage
4. If cost < $1, arbitrage exists with profit = $1 - cost
"""

import asyncio
from typing import Optional
from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

# Import optimization modules
try:
    from services.optimization import (
        constraint_solver,
        dependency_detector,
        bregman_projector,
        MarketInfo,
        Dependency,
        DependencyType
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logger.warning("Optimization module not available, combinatorial strategy limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class CombinatorialStrategy(BaseStrategy):
    """
    Combinatorial Arbitrage: Cross-market arbitrage using integer programming.

    Detects logical dependencies between markets and exploits mispricing
    that arises when markets are not properly correlated.

    Example:
    - Market A: "Trump wins Pennsylvania" YES: $0.48
    - Market B: "Republican wins Pennsylvania" YES: $0.55
    - Dependency: A implies B (if Trump wins, Republican wins)
    - Arbitrage: If prices violate this constraint, profit exists

    This is the strategy that extracted $95K+ in the research paper.
    """

    strategy_type = StrategyType.COMBINATORIAL
    name = "Combinatorial Arbitrage"
    description = "Cross-market arbitrage via integer programming"

    def __init__(self):
        super().__init__()
        self._dependency_cache: dict[tuple, list] = {}
        self._pair_cache: dict[str, bool] = {}

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """
        Detect combinatorial arbitrage opportunities.

        This runs synchronously but may trigger async dependency detection
        for uncached pairs.
        """
        if not OPTIMIZATION_AVAILABLE or not NUMPY_AVAILABLE:
            return []

        opportunities = []

        # Group markets by event for potential dependencies
        event_markets: dict[str, list[Market]] = {}
        for market in markets:
            if market.closed or not market.active:
                continue
            event_id = market.event_id or "unknown"
            if event_id not in event_markets:
                event_markets[event_id] = []
            event_markets[event_id].append(market)

        # Also look for cross-event dependencies
        all_active = [m for m in markets if not m.closed and m.active]

        # Check pairs within same event (higher likelihood of dependency)
        for event_id, event_mkts in event_markets.items():
            if len(event_mkts) < 2:
                continue

            for i, market_a in enumerate(event_mkts):
                for market_b in event_mkts[i+1:]:
                    opp = self._check_pair(market_a, market_b, prices)
                    if opp:
                        opportunities.append(opp)

        # Check high-potential cross-event pairs (limited to avoid explosion)
        # Focus on markets with similar keywords
        checked = set()
        for market_a in all_active[:100]:  # Limit for performance
            candidates = self._find_related_markets(market_a, all_active)
            for market_b in candidates[:5]:  # Top 5 most related
                pair_key = tuple(sorted([market_a.id, market_b.id]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                opp = self._check_pair(market_a, market_b, prices)
                if opp:
                    opportunities.append(opp)

        return opportunities

    def _check_pair(
        self,
        market_a: Market,
        market_b: Market,
        prices: dict[str, dict]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check a market pair for combinatorial arbitrage.

        Uses integer programming to determine if prices violate
        the marginal polytope constraints.
        """
        # Get live prices
        prices_a = self._get_market_prices(market_a, prices)
        prices_b = self._get_market_prices(market_b, prices)

        if not prices_a or not prices_b:
            return None

        # Detect dependencies (cached)
        cache_key = (market_a.id, market_b.id)
        if cache_key not in self._dependency_cache:
            deps = self._detect_dependencies_sync(market_a, market_b)
            self._dependency_cache[cache_key] = deps

        dependencies = self._dependency_cache[cache_key]

        if not dependencies:
            return None  # Independent markets, no combinatorial arb

        # Build constraint matrix and run IP solver
        try:
            result = constraint_solver.detect_cross_market_arbitrage(
                prices_a, prices_b, dependencies
            )

            if result.arbitrage_found and result.profit > self.min_profit:
                return self._create_combinatorial_opportunity(
                    market_a, market_b,
                    prices_a, prices_b,
                    result, dependencies
                )

        except Exception as e:
            logger.debug(f"IP solver error for {market_a.id}/{market_b.id}: {e}")

        return None

    def _get_market_prices(
        self,
        market: Market,
        prices: dict[str, dict]
    ) -> list[float]:
        """Get outcome prices for a market."""
        if len(market.outcome_prices) == 2:
            # Binary market
            yes_price = market.yes_price
            no_price = market.no_price

            if market.clob_token_ids:
                if len(market.clob_token_ids) > 0:
                    token = market.clob_token_ids[0]
                    if token in prices:
                        yes_price = prices[token].get("mid", yes_price)
                if len(market.clob_token_ids) > 1:
                    token = market.clob_token_ids[1]
                    if token in prices:
                        no_price = prices[token].get("mid", no_price)

            return [yes_price, no_price]

        elif len(market.outcome_prices) > 2:
            # Multi-outcome market
            result = list(market.outcome_prices)
            if market.clob_token_ids:
                for i, token in enumerate(market.clob_token_ids):
                    if token in prices and i < len(result):
                        result[i] = prices[token].get("mid", result[i])
            return result

        return []

    def _detect_dependencies_sync(
        self,
        market_a: Market,
        market_b: Market
    ) -> list[Dependency]:
        """
        Synchronously detect dependencies using heuristics.

        For async LLM detection, use detect_dependencies_async.
        """
        dependencies = []
        q_a = market_a.question.lower()
        q_b = market_b.question.lower()

        # Extract outcomes
        outcomes_a = ["YES", "NO"] if len(market_a.outcome_prices) == 2 else [f"Outcome {i}" for i in range(len(market_a.outcome_prices))]
        outcomes_b = ["YES", "NO"] if len(market_b.outcome_prices) == 2 else [f"Outcome {i}" for i in range(len(market_b.outcome_prices))]

        # Heuristic: Check for implies relationships
        # Pattern: Specific candidate implies party win
        implies_patterns = [
            (["trump", "desantis", "haley"], ["republican"]),
            (["biden", "harris", "newsom"], ["democrat"]),
            (["wins by", "margin"], ["wins"]),
        ]

        for specific_terms, general_terms in implies_patterns:
            has_specific_a = any(t in q_a for t in specific_terms)
            has_general_b = any(t in q_b for t in general_terms)
            has_specific_b = any(t in q_b for t in specific_terms)
            has_general_a = any(t in q_a for t in general_terms)

            # Check if they share context (same state, same election, etc)
            if not self._share_context(q_a, q_b):
                continue

            if has_specific_a and has_general_b:
                # Market A specific implies Market B general
                # YES in A implies YES in B
                dependencies.append(Dependency(
                    market_a_idx=0,
                    outcome_a_idx=0,  # YES
                    market_b_idx=1,
                    outcome_b_idx=0,  # YES
                    dep_type=DependencyType.IMPLIES,
                    reason=f"Specific outcome implies general: {specific_terms} -> {general_terms}"
                ))

            if has_specific_b and has_general_a:
                # Market B specific implies Market A general
                dependencies.append(Dependency(
                    market_a_idx=0,
                    outcome_a_idx=0,
                    market_b_idx=1,
                    outcome_b_idx=0,
                    dep_type=DependencyType.IMPLIES,
                    reason=f"Specific outcome implies general"
                ))

        # Heuristic: Check for exclusion relationships
        exclusion_patterns = [
            (["above", "over", "more than"], ["below", "under", "less than"]),
            (["before"], ["after"]),
            (["win"], ["lose"]),
        ]

        for terms_a, terms_b in exclusion_patterns:
            has_a = any(t in q_a for t in terms_a)
            has_b = any(t in q_b for t in terms_b)

            if has_a and has_b and self._share_context(q_a, q_b):
                # YES in A excludes YES in B
                dependencies.append(Dependency(
                    market_a_idx=0,
                    outcome_a_idx=0,
                    market_b_idx=1,
                    outcome_b_idx=0,
                    dep_type=DependencyType.EXCLUDES,
                    reason=f"Contradictory outcomes: {terms_a} vs {terms_b}"
                ))

        return dependencies

    def _share_context(self, q_a: str, q_b: str) -> bool:
        """Check if two questions share enough context to be related."""
        # Extract significant words
        stop_words = {
            "will", "the", "a", "an", "in", "on", "by", "to", "be", "is",
            "of", "for", "with", "this", "that", "it", "at", "from", "or",
            "and", "yes", "no", "market", "price"
        }

        words_a = set(q_a.split()) - stop_words
        words_b = set(q_b.split()) - stop_words

        # Filter short words
        words_a = {w for w in words_a if len(w) > 3}
        words_b = {w for w in words_b if len(w) > 3}

        common = words_a & words_b
        return len(common) >= 2

    def _find_related_markets(
        self,
        market: Market,
        all_markets: list[Market]
    ) -> list[Market]:
        """Find markets potentially related to the given market."""
        q = market.question.lower()

        # Extract key entities
        entities = set()
        entity_patterns = [
            "trump", "biden", "harris", "republican", "democrat",
            "bitcoin", "btc", "ethereum", "eth",
            "pennsylvania", "georgia", "michigan", "arizona", "nevada"
        ]
        for e in entity_patterns:
            if e in q:
                entities.add(e)

        if not entities:
            return []

        # Find markets with shared entities
        related = []
        for m in all_markets:
            if m.id == market.id:
                continue
            m_q = m.question.lower()
            shared = sum(1 for e in entities if e in m_q)
            if shared > 0:
                related.append((m, shared))

        # Sort by number of shared entities
        related.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in related]

    def _create_combinatorial_opportunity(
        self,
        market_a: Market,
        market_b: Market,
        prices_a: list[float],
        prices_b: list[float],
        result,  # ArbitrageResult
        dependencies: list[Dependency]
    ) -> ArbitrageOpportunity:
        """Create opportunity from IP solver result."""
        # Build positions to take
        positions = []
        n_a = len(prices_a)

        for pos in result.positions:
            idx = pos["index"]
            if idx < n_a:
                # Position in market A
                token_id = None
                if market_a.clob_token_ids and idx < len(market_a.clob_token_ids):
                    token_id = market_a.clob_token_ids[idx]
                positions.append({
                    "action": "BUY",
                    "outcome": "YES" if idx == 0 else "NO",
                    "market": market_a.question[:50],
                    "price": pos["price"],
                    "token_id": token_id
                })
            else:
                # Position in market B
                b_idx = idx - n_a
                token_id = None
                if market_b.clob_token_ids and b_idx < len(market_b.clob_token_ids):
                    token_id = market_b.clob_token_ids[b_idx]
                positions.append({
                    "action": "BUY",
                    "outcome": "YES" if b_idx == 0 else "NO",
                    "market": market_b.question[:50],
                    "price": pos["price"],
                    "token_id": token_id
                })

        dep_desc = ", ".join([d.reason for d in dependencies[:2]])

        return self.create_opportunity(
            title=f"Combinatorial: {market_a.question[:30]}... + {market_b.question[:30]}...",
            description=f"Cross-market arbitrage via IP solver. Dependencies: {dep_desc}. Cost: ${result.total_cost:.3f}",
            total_cost=result.total_cost,
            markets=[market_a, market_b],
            positions=positions
        )

    async def detect_async(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """
        Async version with LLM dependency detection.

        Use this for more accurate dependency detection at the cost of
        increased latency.
        """
        if not OPTIMIZATION_AVAILABLE:
            return []

        opportunities = []

        # Build market pairs to check
        pairs = []
        all_active = [m for m in markets if not m.closed and m.active]

        for i, market_a in enumerate(all_active[:50]):
            candidates = self._find_related_markets(market_a, all_active)
            for market_b in candidates[:3]:
                pairs.append((market_a, market_b))

        # Batch LLM dependency detection
        market_infos = []
        for market_a, market_b in pairs:
            info_a = MarketInfo(
                id=market_a.id,
                question=market_a.question,
                outcomes=["YES", "NO"],
                prices=self._get_market_prices(market_a, prices)
            )
            info_b = MarketInfo(
                id=market_b.id,
                question=market_b.question,
                outcomes=["YES", "NO"],
                prices=self._get_market_prices(market_b, prices)
            )
            market_infos.append((info_a, info_b))

        # Run LLM detection in parallel
        analyses = await dependency_detector.batch_detect(
            [(a, b) for a, b in market_infos],
            concurrency=5
        )

        # Check each pair with detected dependencies
        for (market_a, market_b), analysis in zip(pairs, analyses):
            if analysis.is_independent:
                continue

            prices_a = self._get_market_prices(market_a, prices)
            prices_b = self._get_market_prices(market_b, prices)

            try:
                result = constraint_solver.detect_cross_market_arbitrage(
                    prices_a, prices_b, analysis.dependencies
                )

                if result.arbitrage_found and result.profit > self.min_profit:
                    opp = self._create_combinatorial_opportunity(
                        market_a, market_b,
                        prices_a, prices_b,
                        result, analysis.dependencies
                    )
                    if opp:
                        opportunities.append(opp)

            except Exception as e:
                logger.debug(f"IP solver error: {e}")

        return opportunities
