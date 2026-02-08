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

from __future__ import annotations

import re
import time
from collections import defaultdict
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
        MarketInfo,
        Dependency,
        DependencyType,
    )

    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logger.warning("Optimization module not available, combinatorial strategy limited")

try:
    import numpy  # noqa: F401

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Confidence tiers for combinatorial trades
# Research shows 62% failure rate for LLM-detected dependencies.
# These tiers gate which opportunities are surfaced.
# ---------------------------------------------------------------------------
HIGH_CONFIDENCE = 0.90    # Heuristic + LLM agree, structural valid, price consistent
MEDIUM_CONFIDENCE = 0.75  # LLM confident + structural valid
LOW_CONFIDENCE = 0.60     # LLM says yes but structural unclear
REJECT_THRESHOLD = 0.60   # Below this: do not trade

# Minimum LLM confidence to even consider executing
MIN_LLM_CONFIDENCE = 0.85

# Month ordering for date comparisons
_MONTH_ORDER = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ---------------------------------------------------------------------------
# Known Dependency Patterns (bypass LLM for well-understood relationships)
# ---------------------------------------------------------------------------
class KnownPatternMatcher:
    """
    Hard-coded validated dependency patterns that bypass LLM detection.

    These patterns are structurally certain and do not need probabilistic
    confirmation. Each pattern returns a (DependencyType, reason) tuple
    if matched, or None otherwise.
    """

    # Candidate-to-party implies patterns
    _CANDIDATE_PARTY = [
        # (candidate_regex, party_regex, party_label)
        (r"\btrump\b", r"\brepublican\b", "Trump -> Republican"),
        (r"\bdesantis\b", r"\brepublican\b", "DeSantis -> Republican"),
        (r"\bhaley\b", r"\brepublican\b", "Haley -> Republican"),
        (r"\bvance\b", r"\brepublican\b", "Vance -> Republican"),
        (r"\bbiden\b", r"\bdemocrat\b", "Biden -> Democrat"),
        (r"\bharris\b", r"\bdemocrat\b", "Harris -> Democrat"),
        (r"\bnewsom\b", r"\bdemocrat\b", "Newsom -> Democrat"),
    ]

    # Championship implies playoffs pattern
    _CHAMPIONSHIP_PLAYOFF = re.compile(
        r"\b(?P<team>.+?)\s+wins?\s+(?:the\s+)?(?:championship|title|finals|super bowl|world series)\b",
        re.IGNORECASE,
    )
    _PLAYOFF_PATTERN = re.compile(
        r"\b(?P<team>.+?)\s+(?:makes?|reaches?|qualifies?)\s+(?:the\s+)?(?:playoffs?|postseason|finals)\b",
        re.IGNORECASE,
    )

    # Price threshold implies pattern: "above X" implies "above Y" when Y < X
    _PRICE_ABOVE = re.compile(
        r"(?:above|over|more than|greater than|exceeds?)\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE
    )
    _PRICE_BELOW = re.compile(
        r"(?:below|under|less than|fewer than)\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE
    )

    # Date-ordered cumulative pattern: "by <date_a>" implies "by <date_b>" when A < B
    _BY_DATE = re.compile(
        r"\bby\s+(january|february|march|april|may|june|july|august|september|october|november|december"
        r"|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b",
        re.IGNORECASE,
    )

    # Mutually exclusive outcomes in the same event
    _WINS_PATTERN = re.compile(r"\b(?P<entity>.+?)\s+wins?\b", re.IGNORECASE)

    @classmethod
    def match(cls, q_a: str, q_b: str, share_context: bool) -> Optional[tuple[DependencyType, str]]:
        """
        Check two market questions against known patterns.

        Returns (DependencyType, reason) if a known pattern matches, else None.
        Direction is always A -> B (A's YES implies/excludes B's YES).
        """
        q_a_lower = q_a.lower()
        q_b_lower = q_b.lower()

        # 1. Candidate wins State -> Party of candidate wins State (IMPLIES)
        if share_context:
            for cand_re, party_re, label in cls._CANDIDATE_PARTY:
                if re.search(cand_re, q_a_lower) and re.search(party_re, q_b_lower):
                    return (DependencyType.IMPLIES, f"Known pattern: {label} (candidate implies party)")
                # Also check reversed direction
                if re.search(cand_re, q_b_lower) and re.search(party_re, q_a_lower):
                    # B implies A in this case, but we report direction for the caller
                    return (DependencyType.IMPLIES, f"Known pattern: {label} (candidate implies party, reversed)")

        # 2. Event by Date A -> Event by Date B where A < B (CUMULATIVE)
        if share_context:
            dates_a = cls._BY_DATE.findall(q_a_lower)
            dates_b = cls._BY_DATE.findall(q_b_lower)
            if dates_a and dates_b:
                order_a = cls._date_to_ordinal(dates_a[0])
                order_b = cls._date_to_ordinal(dates_b[0])
                if order_a is not None and order_b is not None and order_a < order_b:
                    return (
                        DependencyType.CUMULATIVE,
                        f"Known pattern: event by {dates_a[0]} implies event by {dates_b[0]}",
                    )

        # 3. Price above X -> Price above Y where Y < X (IMPLIES)
        above_a = cls._PRICE_ABOVE.findall(q_a_lower)
        above_b = cls._PRICE_ABOVE.findall(q_b_lower)
        if above_a and above_b and share_context:
            val_a = float(above_a[0].replace(",", ""))
            val_b = float(above_b[0].replace(",", ""))
            if val_a > val_b:
                return (
                    DependencyType.IMPLIES,
                    f"Known pattern: above ${val_a:,.0f} implies above ${val_b:,.0f}",
                )

        # 4. Team wins championship -> Team makes playoffs (IMPLIES)
        champ_a = cls._CHAMPIONSHIP_PLAYOFF.search(q_a)
        playoff_b = cls._PLAYOFF_PATTERN.search(q_b)
        if champ_a and playoff_b:
            team_a = champ_a.group("team").strip().lower()
            team_b = playoff_b.group("team").strip().lower()
            if team_a and team_b and (team_a in team_b or team_b in team_a):
                return (
                    DependencyType.IMPLIES,
                    f"Known pattern: {team_a} wins championship implies makes playoffs",
                )

        # 5. Mutually exclusive outcomes in same event (EXCLUDES)
        #    E.g. "Team A wins <event>" vs "Team B wins <event>" where A != B
        if share_context:
            wins_a = cls._WINS_PATTERN.search(q_a)
            wins_b = cls._WINS_PATTERN.search(q_b)
            if wins_a and wins_b:
                entity_a = wins_a.group("entity").strip().lower()
                entity_b = wins_b.group("entity").strip().lower()
                # Different entities competing for the same thing
                if entity_a and entity_b and entity_a != entity_b:
                    # Rough check: the rest of the questions match enough
                    rest_a = q_a_lower.replace(entity_a, "").strip()
                    rest_b = q_b_lower.replace(entity_b, "").strip()
                    # If the non-entity part is similar, they're competing
                    common_words = set(rest_a.split()) & set(rest_b.split())
                    if len(common_words) >= 3:
                        return (
                            DependencyType.EXCLUDES,
                            f"Known pattern: '{entity_a}' and '{entity_b}' are mutually exclusive winners",
                        )

        return None

    @classmethod
    def _date_to_ordinal(cls, date_str: str) -> Optional[int]:
        """Convert a date string to a comparable ordinal."""
        date_str = date_str.lower().strip()
        if date_str in _MONTH_ORDER:
            return _MONTH_ORDER[date_str]
        # Try numeric date (M/D or M/D/Y)
        parts = re.split(r"[/-]", date_str)
        if len(parts) >= 2:
            try:
                month = int(parts[0])
                day = int(parts[1])
                return month * 100 + day
            except ValueError:
                pass
        return None


# ---------------------------------------------------------------------------
# Historical Accuracy Tracking
# ---------------------------------------------------------------------------
class DependencyAccuracyTracker:
    """
    In-memory tracker for dependency prediction accuracy.

    Records which dependencies were acted on and whether they turned out
    to be correct after market resolution. Automatically raises the
    confidence threshold if accuracy drops below 70%.
    """

    def __init__(self, base_threshold: float = MIN_LLM_CONFIDENCE):
        self._base_threshold = base_threshold
        self._records: list[dict] = []  # {dep_key, correct, timestamp}
        self._accuracy_cache: Optional[float] = None
        self._threshold_override: Optional[float] = None

    def record_dependency(self, market_a_id: str, market_b_id: str,
                          dep_type: str, acted_on: bool = True) -> str:
        """Record that a dependency was detected and (optionally) acted upon."""
        dep_key = f"{market_a_id}:{market_b_id}:{dep_type}"
        self._records.append({
            "dep_key": dep_key,
            "correct": None,  # Unknown until resolution
            "acted_on": acted_on,
            "timestamp": time.time(),
        })
        self._accuracy_cache = None
        return dep_key

    def record_resolution(self, market_a_id: str, market_b_id: str,
                          dep_type: str, was_correct: bool) -> None:
        """Record whether a dependency turned out to be correct after resolution."""
        dep_key = f"{market_a_id}:{market_b_id}:{dep_type}"
        for record in reversed(self._records):
            if record["dep_key"] == dep_key and record["correct"] is None:
                record["correct"] = was_correct
                self._accuracy_cache = None
                self._update_threshold()
                return

    @property
    def accuracy(self) -> Optional[float]:
        """Running accuracy rate of resolved dependencies."""
        if self._accuracy_cache is not None:
            return self._accuracy_cache

        resolved = [r for r in self._records if r["correct"] is not None and r["acted_on"]]
        if not resolved:
            return None

        correct = sum(1 for r in resolved if r["correct"])
        self._accuracy_cache = correct / len(resolved)
        return self._accuracy_cache

    @property
    def total_acted(self) -> int:
        return sum(1 for r in self._records if r["acted_on"])

    @property
    def total_resolved(self) -> int:
        return sum(1 for r in self._records if r["correct"] is not None and r["acted_on"])

    @property
    def effective_threshold(self) -> float:
        """Current confidence threshold, auto-raised if accuracy is poor."""
        if self._threshold_override is not None:
            return self._threshold_override
        return self._base_threshold

    def _update_threshold(self) -> None:
        """Auto-raise confidence threshold if accuracy drops below 70%."""
        acc = self.accuracy
        if acc is not None and self.total_resolved >= 5:
            if acc < 0.70:
                # Raise threshold proportionally to how bad accuracy is
                boost = (0.70 - acc) * 0.5  # e.g. 60% accuracy -> +0.05
                self._threshold_override = min(0.95, self._base_threshold + boost)
                logger.warning(
                    f"Dependency accuracy at {acc:.0%} (< 70%), raising confidence "
                    f"threshold to {self._threshold_override:.2f}"
                )
            else:
                self._threshold_override = None

    def get_stats(self) -> dict:
        """Return tracker statistics."""
        return {
            "total_acted": self.total_acted,
            "total_resolved": self.total_resolved,
            "accuracy": self.accuracy,
            "effective_threshold": self.effective_threshold,
        }


# ---------------------------------------------------------------------------
# Dependency Validator (multi-source validation + contradiction detection)
# ---------------------------------------------------------------------------
class DependencyValidator:
    """
    Multi-source validation pipeline for market dependencies.

    Requires multiple confirmation sources before trusting a dependency,
    reducing the 62% false positive rate from LLM-only detection.

    Validation stages:
    1. Known pattern matching (hard-coded, highest confidence)
    2. Heuristic check (rule-based)
    3. LLM confidence threshold (raised to 0.85)
    4. Structural validation (type-specific sanity checks)
    5. Price sanity check (prices should be consistent with dependency)
    6. Contradiction detection (no internal conflicts)
    """

    def __init__(self, accuracy_tracker: Optional[DependencyAccuracyTracker] = None):
        self.accuracy_tracker = accuracy_tracker or DependencyAccuracyTracker()

    # -- Public API ----------------------------------------------------------

    def validate_dependencies(
        self,
        dependencies: list[Dependency],
        market_a_question: str,
        market_b_question: str,
        prices_a: list[float],
        prices_b: list[float],
        llm_confidence: float,
        heuristic_found: bool,
        share_context: bool,
    ) -> tuple[list[Dependency], float, str]:
        """
        Run the full validation pipeline on a set of dependencies.

        Returns:
            (validated_deps, confidence_score, confidence_tier)
            where confidence_tier is one of "HIGH", "MEDIUM", "LOW", "REJECT".
        """
        if not dependencies:
            return [], 0.0, "REJECT"

        # Step 1: Check for known patterns (bypass LLM entirely)
        known = KnownPatternMatcher.match(market_a_question, market_b_question, share_context)
        known_pattern_match = known is not None

        # Step 2: Structural validation for each dependency
        structural_results = [
            self._structural_check(dep, market_a_question, market_b_question)
            for dep in dependencies
        ]
        structural_valid_count = sum(1 for ok in structural_results if ok)
        structural_pass = structural_valid_count > 0

        # Step 3: Price sanity check
        price_consistent = self._price_sanity_check(
            dependencies, prices_a, prices_b
        )

        # Step 4: LLM confidence gate
        effective_threshold = self.accuracy_tracker.effective_threshold
        llm_passes = llm_confidence >= effective_threshold

        # Step 5: Contradiction detection
        contradictions = self._detect_contradictions(dependencies)
        has_contradictions = len(contradictions) > 0

        if has_contradictions:
            logger.warning(
                f"Dependency contradictions detected between "
                f"'{market_a_question[:40]}' and '{market_b_question[:40]}': "
                f"{contradictions}"
            )
            return [], 0.0, "REJECT"

        # Step 6: Filter to structurally valid dependencies only
        validated = [
            dep for dep, ok in zip(dependencies, structural_results) if ok
        ]
        # If no deps pass structural check, keep originals only if known pattern
        if not validated and known_pattern_match:
            validated = dependencies

        if not validated:
            return [], 0.0, "REJECT"

        # Step 7: Assign confidence tier
        confidence, tier = self._assign_confidence_tier(
            known_pattern_match=known_pattern_match,
            heuristic_found=heuristic_found,
            llm_passes=llm_passes,
            llm_confidence=llm_confidence,
            structural_pass=structural_pass,
            price_consistent=price_consistent,
        )

        return validated, confidence, tier

    # -- Structural Validation -----------------------------------------------

    def _structural_check(
        self, dep: Dependency, q_a: str, q_b: str
    ) -> bool:
        """
        Check if a dependency makes structural sense for its type.

        - IMPLIES: A must be a subset/specialization of B
        - EXCLUDES: A and B must concern the same measurable quantity
                    with non-overlapping ranges
        - CUMULATIVE: Must have clear date ordering
        """
        q_a_lower = q_a.lower()
        q_b_lower = q_b.lower()

        if dep.dep_type == DependencyType.IMPLIES:
            return self._check_implies_structure(q_a_lower, q_b_lower)
        elif dep.dep_type == DependencyType.EXCLUDES:
            return self._check_excludes_structure(q_a_lower, q_b_lower)
        elif dep.dep_type == DependencyType.CUMULATIVE:
            return self._check_cumulative_structure(q_a_lower, q_b_lower)

        return False

    def _check_implies_structure(self, q_a: str, q_b: str) -> bool:
        """
        IMPLIES validation: A should be a specialization of B.

        Valid examples:
          - "Trump wins PA" implies "Republican wins PA" (specific candidate -> party)
          - "BTC above $150K" implies "BTC above $100K" (stricter threshold)
          - "Wins by 10+ points" implies "wins" (specific margin -> general)
        """
        # Specific-to-general: A has more restrictive terms than B
        specificity_markers = [
            r"\bby\s+\d+\+?\s*points?\b",   # Margin of victory
            r"\babove\s+\$?[\d,]+\b",         # Price threshold
            r"\bover\s+\$?[\d,]+\b",
            r"\bmore\s+than\s+\$?[\d,]+\b",
            r"\bwins?\s+championship\b",       # Championship implies playoffs
            r"\bwins?\s+(?:the\s+)?finals\b",
        ]

        # Check if A is more specific than B (A has specificity markers B lacks)
        a_specific = any(re.search(pat, q_a) for pat in specificity_markers)
        b_general = not any(re.search(pat, q_b) for pat in specificity_markers)

        if a_specific and b_general:
            return True

        # Candidate -> Party relationship
        candidates = [r"\btrump\b", r"\bbiden\b", r"\bharris\b", r"\bdesantis\b",
                       r"\bhaley\b", r"\bnewsom\b", r"\bvance\b"]
        parties = [r"\brepublican\b", r"\bdemocrat\b", r"\bgop\b"]

        a_has_candidate = any(re.search(c, q_a) for c in candidates)
        b_has_party = any(re.search(p, q_b) for p in parties)
        if a_has_candidate and b_has_party:
            return True

        # Price thresholds: "above X" implies "above Y" where X > Y
        above_re = re.compile(r"(?:above|over|more than)\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
        a_vals = above_re.findall(q_a)
        b_vals = above_re.findall(q_b)
        if a_vals and b_vals:
            try:
                va = float(a_vals[0].replace(",", ""))
                vb = float(b_vals[0].replace(",", ""))
                if va > vb:
                    return True
            except ValueError:
                pass

        return False

    def _check_excludes_structure(self, q_a: str, q_b: str) -> bool:
        """
        EXCLUDES validation: A and B should concern the same measurable
        quantity with non-overlapping ranges or mutually exclusive entities.
        """
        # Opposing directions on the same quantity
        opposing_pairs = [
            (r"\babove\b|\bover\b|\bmore\s+than\b", r"\bbelow\b|\bunder\b|\bless\s+than\b"),
            (r"\bbefore\b", r"\bafter\b"),
            (r"\bwin\b", r"\blose\b"),
        ]
        for pat_a, pat_b in opposing_pairs:
            if re.search(pat_a, q_a) and re.search(pat_b, q_b):
                return True
            if re.search(pat_b, q_a) and re.search(pat_a, q_b):
                return True

        # Different entities winning the same thing
        wins_re = re.compile(r"(\w+(?:\s+\w+)?)\s+wins?\b", re.IGNORECASE)
        wa = wins_re.findall(q_a)
        wb = wins_re.findall(q_b)
        if wa and wb:
            entity_a = wa[0].strip().lower()
            entity_b = wb[0].strip().lower()
            if entity_a != entity_b:
                return True

        return False

    def _check_cumulative_structure(self, q_a: str, q_b: str) -> bool:
        """
        CUMULATIVE validation: Must have clear date ordering.
        "Event by March" implies "Event by June".
        """
        date_re = re.compile(
            r"\bby\s+(january|february|march|april|may|june|july|august|september|"
            r"october|november|december|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b",
            re.IGNORECASE,
        )
        dates_a = date_re.findall(q_a)
        dates_b = date_re.findall(q_b)

        if not dates_a or not dates_b:
            return False

        ord_a = KnownPatternMatcher._date_to_ordinal(dates_a[0])
        ord_b = KnownPatternMatcher._date_to_ordinal(dates_b[0])

        if ord_a is not None and ord_b is not None:
            # Cumulative: earlier date being true implies later date true
            return ord_a != ord_b

        return False

    # -- Price Sanity Check --------------------------------------------------

    def _price_sanity_check(
        self,
        dependencies: list[Dependency],
        prices_a: list[float],
        prices_b: list[float],
    ) -> bool:
        """
        Verify that current prices are roughly consistent with the
        alleged dependency. Large contradictions suggest the dependency
        is wrong, not that there's a huge arbitrage.

        Rules:
        - IMPLIES (A->B): price(A_yes) should be <= price(B_yes)
          If A_yes >> B_yes, the market doesn't believe A implies B.
        - EXCLUDES: price(A_yes) + price(B_yes) should be <= ~1.05
          If sum >> 1, market doesn't believe they're exclusive.
        - CUMULATIVE (earlier -> later): price(earlier_yes) <= price(later_yes)
        """
        for dep in dependencies:
            try:
                p_a = prices_a[dep.outcome_a_idx] if dep.outcome_a_idx < len(prices_a) else None
                p_b = prices_b[dep.outcome_b_idx] if dep.outcome_b_idx < len(prices_b) else None

                if p_a is None or p_b is None:
                    continue

                if dep.dep_type == DependencyType.IMPLIES:
                    # If A implies B, then P(A) <= P(B).
                    # Allow small tolerance for market noise.
                    # Flag if P(A) > P(B) + 0.15 (large contradiction)
                    if p_a > p_b + 0.15:
                        logger.debug(
                            f"Price sanity fail (IMPLIES): P(A)={p_a:.3f} > P(B)={p_b:.3f} + 0.15"
                        )
                        return False

                elif dep.dep_type == DependencyType.EXCLUDES:
                    # If A and B are exclusive, P(A) + P(B) <= 1 + fee margin
                    if p_a + p_b > 1.15:
                        logger.debug(
                            f"Price sanity fail (EXCLUDES): P(A)+P(B)={p_a + p_b:.3f} > 1.15"
                        )
                        return False

                elif dep.dep_type == DependencyType.CUMULATIVE:
                    # Earlier date YES should be <= later date YES
                    if p_a > p_b + 0.15:
                        logger.debug(
                            f"Price sanity fail (CUMULATIVE): P(earlier)={p_a:.3f} > P(later)={p_b:.3f} + 0.15"
                        )
                        return False

            except (IndexError, TypeError):
                continue

        return True

    # -- Contradiction Detection ---------------------------------------------

    def _detect_contradictions(self, dependencies: list[Dependency]) -> list[str]:
        """
        Check a set of dependencies for internal contradictions.

        Contradiction types:
        1. A implies B AND A excludes B
        2. A implies B AND B implies C AND A excludes C (transitive)
        3. Circular exclusion chains

        Builds a simple directed constraint graph and checks for
        inconsistencies.
        """
        contradictions = []

        # Build adjacency maps
        # Key: (market_idx, outcome_idx) -> node identifier
        implies_edges: dict[tuple, set[tuple]] = defaultdict(set)  # A -> {B, C, ...}
        excludes_edges: dict[tuple, set[tuple]] = defaultdict(set)  # A -> {X, Y, ...}

        for dep in dependencies:
            node_a = (dep.market_a_idx, dep.outcome_a_idx)
            node_b = (dep.market_b_idx, dep.outcome_b_idx)

            if dep.dep_type == DependencyType.IMPLIES or dep.dep_type == DependencyType.CUMULATIVE:
                implies_edges[node_a].add(node_b)
            elif dep.dep_type == DependencyType.EXCLUDES:
                excludes_edges[node_a].add(node_b)
                excludes_edges[node_b].add(node_a)  # Symmetric

        # Check 1: A implies B AND A excludes B
        for node_a, implied_nodes in implies_edges.items():
            excluded_by_a = excludes_edges.get(node_a, set())
            overlap = implied_nodes & excluded_by_a
            if overlap:
                for node_b in overlap:
                    contradictions.append(
                        f"Node {node_a} both implies and excludes {node_b}"
                    )

        # Check 2: Transitive contradiction
        # A implies B, B implies C ... then A should not exclude any node
        # reachable via implies chain.
        # BFS/DFS from each node through implies edges
        for start_node in implies_edges:
            reachable = set()
            stack = list(implies_edges[start_node])
            while stack:
                current = stack.pop()
                if current in reachable:
                    continue
                reachable.add(current)
                for next_node in implies_edges.get(current, set()):
                    if next_node not in reachable:
                        stack.append(next_node)

            # Check if any reachable node is excluded by start
            excluded_by_start = excludes_edges.get(start_node, set())
            transitive_conflict = reachable & excluded_by_start
            if transitive_conflict:
                for node in transitive_conflict:
                    msg = f"Transitive contradiction: {start_node} implies (chain) {node} but also excludes it"
                    if msg not in contradictions:
                        contradictions.append(msg)

        return contradictions

    # -- Confidence Tier Assignment ------------------------------------------

    def _assign_confidence_tier(
        self,
        known_pattern_match: bool,
        heuristic_found: bool,
        llm_passes: bool,
        llm_confidence: float,
        structural_pass: bool,
        price_consistent: bool,
    ) -> tuple[float, str]:
        """
        Assign a confidence score and tier based on validation results.

        Returns (confidence_score, tier_name).
        """
        # Known pattern match = highest confidence regardless
        if known_pattern_match and price_consistent:
            return (0.95, "HIGH")

        # HIGH: heuristic + LLM agree, structural valid, price consistent
        if heuristic_found and llm_passes and structural_pass and price_consistent:
            return (max(0.90, llm_confidence), "HIGH")

        # MEDIUM: LLM confident + structural valid
        if llm_passes and structural_pass:
            conf = llm_confidence if price_consistent else llm_confidence * 0.9
            if conf >= MEDIUM_CONFIDENCE:
                return (conf, "MEDIUM")

        # MEDIUM: heuristic + structural, no LLM
        if heuristic_found and structural_pass and price_consistent:
            return (0.78, "MEDIUM")

        # LOW: LLM says yes but structural unclear
        if llm_confidence >= LOW_CONFIDENCE:
            conf = llm_confidence * 0.85  # Penalize for lack of structural support
            if conf >= LOW_CONFIDENCE:
                return (conf, "LOW")

        return (llm_confidence * 0.5, "REJECT")


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
        self._accuracy_tracker = DependencyAccuracyTracker()
        self._validator = DependencyValidator(self._accuracy_tracker)

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """
        Detect combinatorial arbitrage opportunities.

        This runs synchronously but may trigger async dependency detection
        for uncached pairs.
        """
        if not OPTIMIZATION_AVAILABLE or not NUMPY_AVAILABLE:
            return []

        opportunities = []

        # Filter active markets
        all_active = [m for m in markets if not m.closed and m.active]

        # Check high-potential pairs based on keyword similarity
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
        self, market_a: Market, market_b: Market, prices: dict[str, dict]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check a market pair for combinatorial arbitrage.

        Uses integer programming to determine if prices violate
        the marginal polytope constraints. Dependencies are validated
        through a multi-source pipeline before execution.
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

        # --- Validation pipeline ---
        share_context = self._share_context(
            market_a.question.lower(), market_b.question.lower()
        )
        validated_deps, confidence, tier = self._validator.validate_dependencies(
            dependencies=dependencies,
            market_a_question=market_a.question,
            market_b_question=market_b.question,
            prices_a=prices_a,
            prices_b=prices_b,
            llm_confidence=0.6,  # Heuristic-only path has implicit 0.6 confidence
            heuristic_found=True,  # Sync path is always heuristic
            share_context=share_context,
        )

        # Only proceed with HIGH or MEDIUM confidence
        if tier not in ("HIGH", "MEDIUM"):
            logger.debug(
                f"Dependency rejected ({tier}, conf={confidence:.2f}) for "
                f"{market_a.question[:40]}... / {market_b.question[:40]}..."
            )
            return None

        # Track the dependency for accuracy monitoring
        for dep in validated_deps:
            self._accuracy_tracker.record_dependency(
                market_a.id, market_b.id, dep.dep_type.value
            )

        # Build constraint matrix and run IP solver
        try:
            result = constraint_solver.detect_cross_market_arbitrage(
                prices_a, prices_b, validated_deps
            )

            if result.arbitrage_found and result.profit > self.min_profit:
                return self._create_combinatorial_opportunity(
                    market_a, market_b, prices_a, prices_b, result,
                    validated_deps, confidence, tier,
                )

        except Exception as e:
            logger.debug(f"IP solver error for {market_a.id}/{market_b.id}: {e}")

        return None

    def _get_market_prices(
        self, market: Market, prices: dict[str, dict]
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
        self, market_a: Market, market_b: Market
    ) -> list[Dependency]:
        """
        Synchronously detect dependencies using heuristics.

        For async LLM detection, use detect_dependencies_async.
        """
        dependencies = []
        q_a = market_a.question.lower()
        q_b = market_b.question.lower()

        # Extract outcomes
        ["YES", "NO"] if len(market_a.outcome_prices) == 2 else [
            f"Outcome {i}" for i in range(len(market_a.outcome_prices))
        ]
        ["YES", "NO"] if len(market_b.outcome_prices) == 2 else [
            f"Outcome {i}" for i in range(len(market_b.outcome_prices))
        ]

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
                dependencies.append(
                    Dependency(
                        market_a_idx=0,
                        outcome_a_idx=0,  # YES
                        market_b_idx=1,
                        outcome_b_idx=0,  # YES
                        dep_type=DependencyType.IMPLIES,
                        reason=f"Specific outcome implies general: {specific_terms} -> {general_terms}",
                    )
                )

            if has_specific_b and has_general_a:
                # Market B specific implies Market A general
                dependencies.append(
                    Dependency(
                        market_a_idx=0,
                        outcome_a_idx=0,
                        market_b_idx=1,
                        outcome_b_idx=0,
                        dep_type=DependencyType.IMPLIES,
                        reason="Specific outcome implies general",
                    )
                )

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
                dependencies.append(
                    Dependency(
                        market_a_idx=0,
                        outcome_a_idx=0,
                        market_b_idx=1,
                        outcome_b_idx=0,
                        dep_type=DependencyType.EXCLUDES,
                        reason=f"Contradictory outcomes: {terms_a} vs {terms_b}",
                    )
                )

        return dependencies

    def _share_context(self, q_a: str, q_b: str) -> bool:
        """Check if two questions share enough context to be related."""
        # Extract significant words
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
            "yes",
            "no",
            "market",
            "price",
        }

        words_a = set(q_a.split()) - stop_words
        words_b = set(q_b.split()) - stop_words

        # Filter short words
        words_a = {w for w in words_a if len(w) > 3}
        words_b = {w for w in words_b if len(w) > 3}

        common = words_a & words_b
        return len(common) >= 2

    def _find_related_markets(
        self, market: Market, all_markets: list[Market]
    ) -> list[Market]:
        """Find markets potentially related to the given market."""
        q = market.question.lower()

        # Extract key entities
        entities = set()
        entity_patterns = [
            "trump",
            "biden",
            "harris",
            "republican",
            "democrat",
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "pennsylvania",
            "georgia",
            "michigan",
            "arizona",
            "nevada",
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
        dependencies: list[Dependency],
        confidence: float = 0.0,
        confidence_tier: str = "UNKNOWN",
    ) -> ArbitrageOpportunity:
        """Create opportunity from IP solver result, including validation metadata."""
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
                positions.append(
                    {
                        "action": "BUY",
                        "outcome": "YES" if idx == 0 else "NO",
                        "market": market_a.question[:50],
                        "price": pos["price"],
                        "token_id": token_id,
                    }
                )
            else:
                # Position in market B
                b_idx = idx - n_a
                token_id = None
                if market_b.clob_token_ids and b_idx < len(market_b.clob_token_ids):
                    token_id = market_b.clob_token_ids[b_idx]
                positions.append(
                    {
                        "action": "BUY",
                        "outcome": "YES" if b_idx == 0 else "NO",
                        "market": market_b.question[:50],
                        "price": pos["price"],
                        "token_id": token_id,
                    }
                )

        dep_desc = ", ".join([d.reason for d in dependencies[:2]])
        tracker_stats = self._accuracy_tracker.get_stats()
        accuracy_info = (
            f"Dep accuracy: {tracker_stats['accuracy']:.0%}"
            if tracker_stats["accuracy"] is not None
            else "Dep accuracy: N/A (no resolved data)"
        )

        description = (
            f"Cross-market arbitrage via IP solver. "
            f"Dependencies: {dep_desc}. "
            f"Cost: ${result.total_cost:.3f}. "
            f"Validation: {confidence_tier} confidence ({confidence:.2f}). "
            f"{accuracy_info}"
        )

        return self.create_opportunity(
            title=f"Combinatorial: {market_a.question[:30]}... + {market_b.question[:30]}...",
            description=description,
            total_cost=result.total_cost,
            markets=[market_a, market_b],
            positions=positions,
        )

    async def detect_async(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """
        Async version with LLM dependency detection.

        Use this for more accurate dependency detection at the cost of
        increased latency. Dependencies are validated through a multi-source
        pipeline before execution.
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
                prices=self._get_market_prices(market_a, prices),
            )
            info_b = MarketInfo(
                id=market_b.id,
                question=market_b.question,
                outcomes=["YES", "NO"],
                prices=self._get_market_prices(market_b, prices),
            )
            market_infos.append((info_a, info_b))

        # Run LLM detection in parallel
        analyses = await dependency_detector.batch_detect(
            [(a, b) for a, b in market_infos], concurrency=5
        )

        # Also run heuristic detection for each pair to compare
        heuristic_results = {}
        for market_a, market_b in pairs:
            h_deps = self._detect_dependencies_sync(market_a, market_b)
            heuristic_results[(market_a.id, market_b.id)] = len(h_deps) > 0

        # Check each pair with detected dependencies
        for (market_a, market_b), analysis in zip(pairs, analyses):
            if analysis.is_independent:
                continue

            prices_a = self._get_market_prices(market_a, prices)
            prices_b = self._get_market_prices(market_b, prices)

            if not prices_a or not prices_b:
                continue

            # --- Validation pipeline ---
            heuristic_found = heuristic_results.get(
                (market_a.id, market_b.id), False
            )
            share_context = self._share_context(
                market_a.question.lower(), market_b.question.lower()
            )

            validated_deps, confidence, tier = self._validator.validate_dependencies(
                dependencies=analysis.dependencies,
                market_a_question=market_a.question,
                market_b_question=market_b.question,
                prices_a=prices_a,
                prices_b=prices_b,
                llm_confidence=analysis.confidence,
                heuristic_found=heuristic_found,
                share_context=share_context,
            )

            # Only proceed with HIGH or MEDIUM confidence
            if tier not in ("HIGH", "MEDIUM"):
                logger.debug(
                    f"Async dependency rejected ({tier}, conf={confidence:.2f}) for "
                    f"{market_a.question[:40]}... / {market_b.question[:40]}..."
                )
                continue

            # Track the dependency for accuracy monitoring
            for dep in validated_deps:
                self._accuracy_tracker.record_dependency(
                    market_a.id, market_b.id, dep.dep_type.value
                )

            try:
                result = constraint_solver.detect_cross_market_arbitrage(
                    prices_a, prices_b, validated_deps
                )

                if result.arbitrage_found and result.profit > self.min_profit:
                    opp = self._create_combinatorial_opportunity(
                        market_a,
                        market_b,
                        prices_a,
                        prices_b,
                        result,
                        validated_deps,
                        confidence,
                        tier,
                    )
                    if opp:
                        opportunities.append(opp)

            except Exception as e:
                logger.debug(f"IP solver error: {e}")

        return opportunities

    def record_resolution(
        self, market_a_id: str, market_b_id: str,
        dep_type: str, was_correct: bool,
    ) -> None:
        """
        Record whether a dependency turned out to be correct after
        market resolution. Call this when markets resolve so the
        accuracy tracker can auto-adjust thresholds.
        """
        self._accuracy_tracker.record_resolution(
            market_a_id, market_b_id, dep_type, was_correct
        )

    def get_validation_stats(self) -> dict:
        """Return validation and accuracy tracking statistics."""
        return self._accuracy_tracker.get_stats()
