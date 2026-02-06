import asyncio
import re
from typing import Optional
from datetime import datetime, timedelta

from models import Event, Market, ArbitrageOpportunity
from models.opportunity import StrategyType
from services.polymarket import polymarket_client
from services.kalshi_client import kalshi_client
from utils.logger import get_logger

logger = get_logger("cross_platform_scanner")

# Fee schedules (conservative estimates)
POLYMARKET_FEE = 0.02  # 2 % winner-take fee
KALSHI_FEE = 0.07  # 7 % Kalshi fee on winnings (per their fee schedule)


class CrossPlatformScanner:
    """Detect arbitrage opportunities across Polymarket and Kalshi.

    The scanner pulls events from both platforms, matches them by title
    similarity (keyword overlap), resolution date proximity, and category,
    then compares prices to identify cross-platform arbitrage.

    An arbitrage exists when the sum of the cheapest YES on one platform
    and the cheapest NO on the other platform is less than $1.00 (after
    accounting for fees on both sides).
    """

    def __init__(self):
        self._poly_client = polymarket_client
        self._kalshi_client = kalshi_client

    # ------------------------------------------------------------------ #
    #  Text normalisation and similarity
    # ------------------------------------------------------------------ #

    _STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "will",
            "be",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "and",
            "or",
            "not",
            "with",
            "it",
            "this",
            "that",
            "from",
            "as",
            "if",
            "do",
            "does",
            "did",
            "has",
            "have",
            "had",
            "but",
            "so",
            "than",
            "when",
            "what",
            "which",
            "who",
            "whom",
            "how",
            "no",
            "yes",
            "before",
            "after",
            "between",
        }
    )

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """Lower-case, strip punctuation, remove stop words."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in cls._STOP_WORDS and len(w) > 1}

    @classmethod
    def _title_similarity(cls, a: str, b: str) -> float:
        """Keyword-overlap Jaccard similarity in [0, 1]."""
        tokens_a = cls._tokenize(a)
        tokens_b = cls._tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------ #
    #  Date proximity check
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dates_close(
        d1: Optional[datetime],
        d2: Optional[datetime],
        tolerance_days: int = 3,
    ) -> bool:
        """Return True if both dates exist and are within *tolerance_days*."""
        if d1 is None or d2 is None:
            # If either side has no date, we cannot disqualify the match,
            # so treat as acceptable.
            return True
        # Ensure both are naive UTC for a fair comparison
        d1_naive = d1.replace(tzinfo=None) if d1.tzinfo else d1
        d2_naive = d2.replace(tzinfo=None) if d2.tzinfo else d2
        return abs(d1_naive - d2_naive) <= timedelta(days=tolerance_days)

    # ------------------------------------------------------------------ #
    #  Category matching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _categories_compatible(cat_a: Optional[str], cat_b: Optional[str]) -> bool:
        """Fuzzy category match (case-insensitive substring)."""
        if cat_a is None or cat_b is None:
            return True  # unknown category is not a disqualifier
        a = cat_a.lower().strip()
        b = cat_b.lower().strip()
        if a == b:
            return True
        # Allow substring containment (e.g. "politics" in "US Politics")
        return a in b or b in a

    # ------------------------------------------------------------------ #
    #  Event matching
    # ------------------------------------------------------------------ #

    async def find_matching_events(
        self,
        similarity_threshold: float = 0.45,
    ) -> list[tuple[Event, Event]]:
        """Find pairs of (Polymarket event, Kalshi event) that represent
        the same real-world question.

        Matching criteria:
            1. Title similarity >= *similarity_threshold* (Jaccard on keywords)
            2. Resolution dates within 3 days (when both are known)
            3. Categories are compatible
        """
        # Fetch events from both platforms concurrently
        try:
            poly_events, kalshi_events = await asyncio.gather(
                self._poly_client.get_all_events(closed=False),
                self._kalshi_client.get_all_events(closed=False),
            )
        except Exception as exc:
            logger.error("Failed to fetch events for matching", error=str(exc))
            return []

        logger.info(
            "Loaded events for matching",
            polymarket_count=len(poly_events),
            kalshi_count=len(kalshi_events),
        )

        if not poly_events or not kalshi_events:
            return []

        matched: list[tuple[Event, Event]] = []

        for p_event in poly_events:
            best_score = 0.0
            best_kalshi: Optional[Event] = None

            for k_event in kalshi_events:
                # Quick category gate
                if not self._categories_compatible(p_event.category, k_event.category):
                    continue

                score = self._title_similarity(p_event.title, k_event.title)
                if score < similarity_threshold:
                    continue

                # Check resolution dates across markets
                p_end = _earliest_end_date(p_event)
                k_end = _earliest_end_date(k_event)
                if not self._dates_close(p_end, k_end):
                    continue

                if score > best_score:
                    best_score = score
                    best_kalshi = k_event

            if best_kalshi is not None:
                matched.append((p_event, best_kalshi))
                logger.debug(
                    "Matched cross-platform event",
                    poly=p_event.title[:60],
                    kalshi=best_kalshi.title[:60],
                    score=round(best_score, 3),
                )

        logger.info("Cross-platform event matches", count=len(matched))
        return matched

    # ------------------------------------------------------------------ #
    #  Arbitrage calculation for a single market pair
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_cross_platform_arb(
        poly_market: Market,
        kalshi_market: Market,
        poly_fee: float = POLYMARKET_FEE,
        kalshi_fee: float = KALSHI_FEE,
    ) -> Optional[ArbitrageOpportunity]:
        """Detect whether buying YES on one platform and NO on the other
        yields a risk-free profit after fees.

        Two legs to check:
            Leg A: buy Polymarket YES  +  buy Kalshi NO
            Leg B: buy Kalshi YES      +  buy Polymarket NO

        For each leg the guaranteed payout is $1.00 (one side always wins).
        Profit = payout - fees_on_winning_side - total_cost.
        """
        p_yes = poly_market.yes_price
        p_no = poly_market.no_price
        k_yes = kalshi_market.yes_price
        k_no = kalshi_market.no_price

        # Need valid prices on both sides
        if not (0 < p_yes < 1 and 0 < p_no < 1):
            return None
        if not (0 < k_yes < 1 and 0 < k_no < 1):
            return None

        best_opportunity: Optional[ArbitrageOpportunity] = None
        best_net = 0.0

        legs = [
            # (description, cost_platform_a, cost_platform_b, fee_on_a_win, fee_on_b_win, label)
            (
                "Buy YES on Polymarket + NO on Kalshi",
                p_yes,  # cost if Poly YES wins
                k_no,  # cost if Kalshi NO wins
                poly_fee,
                kalshi_fee,
                "poly_yes_kalshi_no",
            ),
            (
                "Buy NO on Polymarket + YES on Kalshi",
                p_no,  # cost if Poly NO wins
                k_yes,  # cost if Kalshi YES wins
                poly_fee,
                kalshi_fee,
                "poly_no_kalshi_yes",
            ),
        ]

        for desc, cost_a, cost_b, fee_a, fee_b, label in legs:
            total_cost = cost_a + cost_b
            if total_cost >= 1.0:
                continue  # no arb

            # When the first leg wins, the payout is $1 on that platform
            # minus the fee on the winning amount.
            # The *winning* amount is (1 - cost_of_that_side).
            # Worst-case fee is max of the two scenarios.
            profit_if_a_wins = (1.0 - cost_a) * (1.0 - fee_a) - cost_b
            profit_if_b_wins = (1.0 - cost_b) * (1.0 - fee_b) - cost_a

            # Guaranteed profit is the *minimum* of the two scenarios
            guaranteed = min(profit_if_a_wins, profit_if_b_wins)
            if guaranteed <= 0:
                continue

            gross = 1.0 - total_cost
            fee_estimate = gross - guaranteed
            roi = (guaranteed / total_cost) * 100.0 if total_cost > 0 else 0.0

            if guaranteed > best_net:
                best_net = guaranteed

                # Build the positions-to-take list
                if label == "poly_yes_kalshi_no":
                    positions = [
                        {
                            "platform": "polymarket",
                            "market_id": poly_market.id,
                            "side": "YES",
                            "price": p_yes,
                        },
                        {
                            "platform": "kalshi",
                            "market_id": kalshi_market.id,
                            "side": "NO",
                            "price": k_no,
                        },
                    ]
                else:
                    positions = [
                        {
                            "platform": "polymarket",
                            "market_id": poly_market.id,
                            "side": "NO",
                            "price": p_no,
                        },
                        {
                            "platform": "kalshi",
                            "market_id": kalshi_market.id,
                            "side": "YES",
                            "price": k_yes,
                        },
                    ]

                best_opportunity = ArbitrageOpportunity(
                    strategy=StrategyType.CROSS_PLATFORM,
                    title=f"Cross-platform arb: {poly_market.question[:80]}",
                    description=(
                        f"{desc}. "
                        f"Poly YES={p_yes:.3f} NO={p_no:.3f} | "
                        f"Kalshi YES={k_yes:.3f} NO={k_no:.3f}"
                    ),
                    total_cost=round(total_cost, 6),
                    expected_payout=1.0,
                    gross_profit=round(gross, 6),
                    fee=round(fee_estimate, 6),
                    net_profit=round(guaranteed, 6),
                    roi_percent=round(roi, 4),
                    risk_score=0.15,  # low risk: guaranteed profit
                    risk_factors=_risk_factors(poly_market, kalshi_market),
                    markets=[
                        {
                            "id": poly_market.id,
                            "platform": "polymarket",
                            "question": poly_market.question,
                            "yes_price": p_yes,
                            "no_price": p_no,
                        },
                        {
                            "id": kalshi_market.id,
                            "platform": "kalshi",
                            "question": kalshi_market.question,
                            "yes_price": k_yes,
                            "no_price": k_no,
                        },
                    ],
                    category=None,
                    min_liquidity=min(poly_market.liquidity, kalshi_market.liquidity),
                    positions_to_take=positions,
                    resolution_date=poly_market.end_date or kalshi_market.end_date,
                )

        return best_opportunity

    # ------------------------------------------------------------------ #
    #  Full scan
    # ------------------------------------------------------------------ #

    async def scan_cross_platform(
        self,
        similarity_threshold: float = 0.45,
    ) -> list[ArbitrageOpportunity]:
        """Run the full cross-platform arbitrage scan.

        Steps:
            1. Find matching events on Polymarket and Kalshi.
            2. For every matched event pair, compare each Polymarket market
               against each Kalshi market for arbitrage.
            3. Return all detected opportunities sorted by ROI descending.
        """
        logger.info("Starting cross-platform arbitrage scan")

        matches = await self.find_matching_events(
            similarity_threshold=similarity_threshold,
        )
        if not matches:
            logger.info("No cross-platform matches found")
            return []

        opportunities: list[ArbitrageOpportunity] = []

        for poly_event, kalshi_event in matches:
            poly_markets = poly_event.markets
            kalshi_markets = kalshi_event.markets

            # If either event has no inline markets, try to match the
            # events' titles directly and use the whole-event-level
            # market lists (often a single binary market per event).
            if not poly_markets or not kalshi_markets:
                continue

            for p_mkt in poly_markets:
                for k_mkt in kalshi_markets:
                    # Optionally also check market-level title similarity
                    # for multi-market events so we pair the right legs.
                    if len(poly_markets) > 1 and len(kalshi_markets) > 1:
                        mkt_sim = self._title_similarity(p_mkt.question, k_mkt.question)
                        if mkt_sim < 0.35:
                            continue

                    opp = self.calculate_cross_platform_arb(p_mkt, k_mkt)
                    if opp is not None:
                        opp.event_id = poly_event.id
                        opp.event_title = poly_event.title
                        opp.category = poly_event.category or kalshi_event.category
                        opportunities.append(opp)

        # Highest ROI first
        opportunities.sort(key=lambda o: o.roi_percent, reverse=True)
        logger.info(
            "Cross-platform scan complete",
            opportunities=len(opportunities),
        )
        return opportunities


# ------------------------------------------------------------------ #
#  Module-level helpers
# ------------------------------------------------------------------ #


def _earliest_end_date(event: Event) -> Optional[datetime]:
    """Return the earliest end_date across all markets in an event."""
    dates = [m.end_date for m in event.markets if m.end_date is not None]
    return min(dates) if dates else None


def _risk_factors(poly_market: Market, kalshi_market: Market) -> list[str]:
    """Compile a list of risk factors for the opportunity."""
    factors: list[str] = []
    if poly_market.liquidity < 1000:
        factors.append("Low Polymarket liquidity")
    if kalshi_market.liquidity < 1000:
        factors.append("Low Kalshi liquidity")
    if poly_market.volume < 500:
        factors.append("Low Polymarket volume")
    if kalshi_market.volume < 500:
        factors.append("Low Kalshi volume")
    factors.append("Execution on two separate platforms required")
    factors.append("Settlement timing may differ across platforms")
    return factors


# Singleton instance
cross_platform_scanner = CrossPlatformScanner()
