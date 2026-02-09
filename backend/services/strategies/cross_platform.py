"""
Strategy: Cross-Platform Oracle Arbitrage

Monitors the same events across Polymarket and Kalshi prediction markets.
When prices diverge beyond a threshold (after fees), executes arbitrage.

Example:
- Polymarket: "BTC above $100K by March" YES = $0.55
- Kalshi: same event YES = $0.62
- Buy YES on Polymarket ($0.55), sell YES on Kalshi ($0.62) = $0.07 profit

This is the most reliably profitable strategy because cross-platform
price discovery is slow and fee structures differ.

Detection approach:
1. Fetch/cache Kalshi markets (refreshed every 60 seconds)
2. For each Polymarket market, fuzzy-match against Kalshi markets
   using Jaccard similarity on word tokens
3. For matched pairs, check both arb legs:
   - Leg A: Buy YES on Polymarket + Buy NO on Kalshi
   - Leg B: Buy NO on Polymarket + Buy YES on Kalshi
4. If guaranteed profit after fees > 0, emit an opportunity
"""

import re
import string
import time
from typing import Optional
from datetime import datetime

import httpx

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  Fee constants
# ---------------------------------------------------------------------------
POLYMARKET_FEE = 0.02  # 2% winner fee
KALSHI_FEE = 0.07  # 7% Kalshi fee on winnings

# ---------------------------------------------------------------------------
#  Text normalisation for fuzzy matching
# ---------------------------------------------------------------------------

_ABBREVIATIONS: dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "xrp": "ripple",
    "doge": "dogecoin",
    "bnb": "binance coin",
    "ada": "cardano",
    "dot": "polkadot",
    "avax": "avalanche",
    "matic": "polygon",
    "link": "chainlink",
    "ltc": "litecoin",
    "uni": "uniswap",
    "gop": "republican",
    "dem": "democrat",
    "dems": "democrats",
    "rep": "republican",
    "reps": "republicans",
    "potus": "president of the united states",
    "scotus": "supreme court",
    "gdp": "gross domestic product",
    "cpi": "consumer price index",
    "ppi": "producer price index",
    "fed": "federal reserve",
    "fomc": "federal open market committee",
    "nfp": "nonfarm payrolls",
    "eod": "end of day",
    "eoy": "end of year",
    "eom": "end of month",
    "q1": "first quarter",
    "q2": "second quarter",
    "q3": "third quarter",
    "q4": "fourth quarter",
    "super bowl": "superbowl",
    "nba": "national basketball association",
    "nfl": "national football league",
    "mlb": "major league baseball",
    "ufc": "ultimate fighting championship",
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}

_RE_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")
_RE_MULTI_SPACE = re.compile(r"\s+")

# Minimum Jaccard similarity to consider two markets as matching
_MATCH_THRESHOLD = 0.35

# Common stop words that inflate the union and dilute Jaccard scores.
# These add no discriminative value when matching market questions.
_STOP_WORDS = frozenset(
    {
        "will",
        "the",
        "be",
        "in",
        "by",
        "on",
        "of",
        "to",
        "at",
        "is",
        "it",
        "an",
        "or",
        "if",
        "as",
        "do",
        "no",
        "so",
        "up",
        "has",
        "for",
        "not",
        "its",
        "was",
        "are",
        "did",
        "can",
        "than",
        "from",
        "this",
        "that",
        "what",
        "which",
        "who",
        "how",
        "does",
        "have",
        "been",
        "above",
        "below",
        "before",
        "after",
        "between",
        "during",
    }
)


def _normalize_text(text: str) -> str:
    """Normalise a market question for comparison.

    Pipeline:
        1. Lower-case
        2. Expand known abbreviations
        3. Remove punctuation
        4. Collapse whitespace
    """
    text = text.lower().strip()

    # Expand abbreviations (whole-word only)
    for abbr, expansion in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", expansion, text)

    text = _RE_PUNCTUATION.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    """Tokenize normalised text into a set of words for Jaccard similarity.

    Removes stop words that inflate the union and dilute discriminative
    signal. For example, "Will BTC be above $100K by March?" and
    "Bitcoin above $100,000 March" should match on the meaningful words
    (bitcoin, 100, march) without "will", "be", "by" hurting the score.
    """
    normalized = _normalize_text(text)
    # Filter out very short tokens and stop words that add noise
    return {w for w in normalized.split() if len(w) > 1 and w not in _STOP_WORDS}


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
#  Lightweight Kalshi market cache
# ---------------------------------------------------------------------------


class _KalshiMarketCache:
    """In-memory cache of Kalshi markets with a configurable TTL.

    Uses httpx synchronous client to fetch from the Kalshi public API.
    The synchronous approach is intentional: BaseStrategy.detect() is
    synchronous, and the 60-second cache means API calls are infrequent.
    """

    def __init__(self, api_url: str, ttl_seconds: int = 60):
        self._api_url = api_url.rstrip("/")
        self._ttl = ttl_seconds
        self._markets: list[Market] = []
        self._last_fetch: float = 0.0

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self._last_fetch) > self._ttl

    def _parse_kalshi_market(self, data: dict) -> Optional[Market]:
        """Convert a Kalshi market JSON dict into the app's Market model.

        Kalshi prices are in cents (0-100); normalise to 0.0-1.0.
        """
        try:
            ticker = data.get("ticker", "")
            title = data.get("title", "") or data.get("subtitle", "")

            yes_bid = (data.get("yes_bid", 0) or 0) / 100.0
            yes_ask = (data.get("yes_ask", 0) or 0) / 100.0
            no_bid = (data.get("no_bid", 0) or 0) / 100.0
            no_ask = (data.get("no_ask", 0) or 0) / 100.0
            last_price = (data.get("last_price", 0) or 0) / 100.0

            # Use midpoint of bid/ask when available; fall back to last_price
            yes_price = (
                (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else last_price
            )
            no_price = (
                (no_bid + no_ask) / 2.0 if (no_bid + no_ask) > 0 else (1.0 - yes_price)
            )

            # Skip markets with zero prices (no liquidity)
            if yes_price <= 0 and no_price <= 0:
                return None

            outcome_prices = [yes_price, no_price]

            # Determine active/closed from status
            status = (data.get("status", "") or "").lower()
            is_active = status in ("open", "active", "")
            is_closed = status in ("closed", "settled", "finalized")

            if is_closed or not is_active:
                return None

            # Parse close_time / expiration_time as end_date
            end_date: Optional[datetime] = None
            for date_key in (
                "close_time",
                "expiration_time",
                "expected_expiration_time",
            ):
                raw = data.get(date_key)
                if raw and isinstance(raw, str):
                    try:
                        end_date = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                        break
                    except (ValueError, TypeError):
                        pass

            volume_raw = data.get("volume", 0) or 0
            liquidity_raw = (
                data.get("liquidity", 0) or data.get("open_interest", 0) or 0
            )

            from models.market import Token

            return Market(
                id=ticker,
                condition_id=ticker,
                question=title,
                slug=ticker,
                tokens=[
                    Token(
                        token_id=f"{ticker}_yes",
                        outcome="Yes",
                        price=yes_price,
                    ),
                    Token(
                        token_id=f"{ticker}_no",
                        outcome="No",
                        price=no_price,
                    ),
                ],
                clob_token_ids=[f"{ticker}_yes", f"{ticker}_no"],
                outcome_prices=outcome_prices,
                active=True,
                closed=False,
                neg_risk=False,
                volume=float(volume_raw),
                liquidity=float(liquidity_raw),
                end_date=end_date,
            )
        except Exception as exc:
            logger.debug("Failed to parse Kalshi market", error=str(exc))
            return None

    def _fetch_markets(self) -> list[Market]:
        """Fetch all active Kalshi markets via paginated GET requests.

        Uses httpx synchronous client.  Resilient: returns an empty list
        on any HTTP or parsing failure so the strategy never crashes.
        Includes rate limiting (200ms between pages) and 429 retry with
        exponential backoff.
        """
        all_markets: list[Market] = []
        cursor: Optional[str] = None
        max_pages = 30

        try:
            with httpx.Client(
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "homerun-arb-scanner/1.0",
                },
            ) as client:
                for page in range(max_pages):
                    params: dict = {"limit": 200, "status": "open"}
                    if cursor:
                        params["cursor"] = cursor

                    # Rate limit: pause between pages to avoid 429
                    if page > 0:
                        time.sleep(0.2)

                    data = None
                    max_retries = 3
                    for attempt in range(max_retries + 1):
                        try:
                            resp = client.get(f"{self._api_url}/markets", params=params)
                            if resp.status_code == 429 and attempt < max_retries:
                                backoff = 2**attempt  # 1s, 2s, 4s
                                logger.warning(
                                    "Kalshi markets 429, retrying",
                                    attempt=attempt + 1,
                                    backoff_seconds=backoff,
                                )
                                time.sleep(backoff)
                                continue
                            resp.raise_for_status()
                            data = resp.json()
                            break
                        except httpx.HTTPStatusError as exc:
                            logger.warning(
                                "Kalshi markets HTTP error",
                                status=exc.response.status_code,
                            )
                            break
                        except Exception as exc:
                            logger.warning(
                                "Kalshi markets request failed",
                                error=str(exc),
                            )
                            break

                    if data is None:
                        break

                    raw_markets = data.get("markets", [])
                    if not raw_markets:
                        break

                    for m_data in raw_markets:
                        parsed = self._parse_kalshi_market(m_data)
                        if parsed is not None:
                            all_markets.append(parsed)

                    cursor = data.get("cursor") or None
                    if cursor is None:
                        break

        except Exception as exc:
            logger.warning("Kalshi client creation failed", error=str(exc))
            return []

        logger.info("Kalshi market cache refreshed", count=len(all_markets))
        return all_markets

    def get_markets(self) -> list[Market]:
        """Return cached Kalshi markets, refreshing if stale."""
        if self.is_stale:
            fetched = self._fetch_markets()
            if fetched:
                self._markets = fetched
                self._last_fetch = time.monotonic()
            elif not self._markets:
                # First fetch failed and cache is empty
                self._last_fetch = time.monotonic()
        return self._markets


# ---------------------------------------------------------------------------
#  Strategy implementation
# ---------------------------------------------------------------------------


# Keywords that indicate player prop / contingent markets where
# Polymarket and Kalshi may have different DNP (Did Not Play) or
# void/cancellation rules.  These markets carry resolution divergence
# risk: one platform may void while the other resolves as NO.
_RESOLUTION_DIVERGENCE_KEYWORDS = frozenset(
    {
        # Player performance thresholds
        "points",
        "rebounds",
        "assists",
        "yards",
        "touchdowns",
        "goals",
        "saves",
        "strikeouts",
        "hits",
        "home runs",
        "passes",
        "tackles",
        "sacks",
        "receptions",
        "carries",
        "completions",
        "interceptions",
        # Common prop patterns
        "scorer",
        "anytime",
        "first goal",
        "last goal",
        "most",
        "mvp",
        "prop",
        # Threshold markers (e.g., "20+", "over 10.5")
        "over",
        "under",
    }
)


def _has_resolution_divergence_risk(question: str) -> bool:
    """Check if a market question involves player/game props where
    platforms may resolve differently on DNP, void, or cancellation."""
    q = question.lower()
    # Check for numeric thresholds like "20+", "10.5+"
    import re

    if re.search(r"\d+\.?\d*\+", q):
        return True
    return any(kw in q for kw in _RESOLUTION_DIVERGENCE_KEYWORDS)


class CrossPlatformStrategy(BaseStrategy):
    """Cross-platform arbitrage between Polymarket and Kalshi.

    Detects price discrepancies for the same event listed on both platforms.
    Uses fuzzy text matching (Jaccard on word tokens) to identify equivalent
    markets, then checks both arbitrage legs accounting for each platform's
    fee structure.
    """

    strategy_type = StrategyType.CROSS_PLATFORM
    name = "Cross-Platform Oracle"
    description = "Cross-platform arbitrage between Polymarket and Kalshi"

    def __init__(self):
        super().__init__()
        self._kalshi_cache = _KalshiMarketCache(
            api_url=settings.KALSHI_API_URL,
            ttl_seconds=60,
        )
        # Pre-compute token sets for Kalshi markets (refreshed with cache)
        self._kalshi_tokens: dict[str, set[str]] = {}

    def _refresh_kalshi_tokens(
        self, kalshi_markets: list[Market]
    ) -> dict[str, set[str]]:
        """Build/update the tokenized question index for Kalshi markets."""
        token_index: dict[str, set[str]] = {}
        for km in kalshi_markets:
            token_index[km.id] = _tokenize(km.question)
        return token_index

    def _find_best_match(
        self,
        pm_tokens: set[str],
        kalshi_markets: list[Market],
        kalshi_token_index: dict[str, set[str]],
    ) -> Optional[tuple[Market, float]]:
        """Find the best-matching Kalshi market for a Polymarket question.

        Returns (kalshi_market, similarity_score) or None if no match
        exceeds the threshold.
        """
        best_market: Optional[Market] = None
        best_score = 0.0

        for km in kalshi_markets:
            km_tokens = kalshi_token_index.get(km.id)
            if not km_tokens:
                continue

            score = _jaccard_similarity(pm_tokens, km_tokens)
            if score > best_score:
                best_score = score
                best_market = km

        if best_market is not None and best_score >= _MATCH_THRESHOLD:
            return best_market, best_score
        return None

    @staticmethod
    def _calculate_arb(
        pm_yes: float,
        pm_no: float,
        k_yes: float,
        k_no: float,
        poly_fee: float = POLYMARKET_FEE,
        kalshi_fee: float = KALSHI_FEE,
    ) -> Optional[dict]:
        """Check both arb legs and return the best one if profitable.

        Leg A: Buy YES on Polymarket + Buy NO on Kalshi
        Leg B: Buy NO on Polymarket + Buy YES on Kalshi

        Returns a dict with leg details or None if no arb exists.
        """
        # Validate prices are within (0, 1)
        for p in (pm_yes, pm_no, k_yes, k_no):
            if not (0 < p < 1):
                return None

        best_leg: Optional[dict] = None
        best_net = 0.0

        legs = [
            {
                "label": "poly_yes_kalshi_no",
                "desc": "Buy YES on Polymarket + NO on Kalshi",
                "cost_a": pm_yes,
                "cost_b": k_no,
                "fee_a": poly_fee,
                "fee_b": kalshi_fee,
                "pm_action": "BUY",
                "pm_outcome": "YES",
                "k_action": "BUY",
                "k_outcome": "NO",
            },
            {
                "label": "poly_no_kalshi_yes",
                "desc": "Buy NO on Polymarket + YES on Kalshi",
                "cost_a": pm_no,
                "cost_b": k_yes,
                "fee_a": poly_fee,
                "fee_b": kalshi_fee,
                "pm_action": "BUY",
                "pm_outcome": "NO",
                "k_action": "BUY",
                "k_outcome": "YES",
            },
        ]

        for leg in legs:
            total_cost = leg["cost_a"] + leg["cost_b"]
            if total_cost >= 1.0:
                continue

            # Guaranteed payout is $1.00 (one side always wins).
            # Profit depends on which side wins (different fees apply).
            profit_if_a_wins = (1.0 - leg["cost_a"]) * (1.0 - leg["fee_a"]) - leg[
                "cost_b"
            ]
            profit_if_b_wins = (1.0 - leg["cost_b"]) * (1.0 - leg["fee_b"]) - leg[
                "cost_a"
            ]

            # Guaranteed profit is the minimum of both scenarios
            guaranteed = min(profit_if_a_wins, profit_if_b_wins)
            if guaranteed <= 0:
                continue

            if guaranteed > best_net:
                best_net = guaranteed
                gross = 1.0 - total_cost
                fee_estimate = gross - guaranteed
                roi = (guaranteed / total_cost) * 100.0

                best_leg = {
                    "label": leg["label"],
                    "desc": leg["desc"],
                    "total_cost": total_cost,
                    "gross_profit": gross,
                    "fee_estimate": fee_estimate,
                    "net_profit": guaranteed,
                    "roi": roi,
                    "pm_action": leg["pm_action"],
                    "pm_outcome": leg["pm_outcome"],
                    "pm_price": leg["cost_a"],
                    "k_action": leg["k_action"],
                    "k_outcome": leg["k_outcome"],
                    "k_price": leg["cost_b"],
                }

        return best_leg

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Detect cross-platform arbitrage opportunities.

        Takes Polymarket events/markets/prices as input (standard interface),
        fetches/uses cached Kalshi markets, finds matching pairs via text
        similarity, and calculates cross-platform arb for each pair.
        """
        if not settings.CROSS_PLATFORM_ENABLED:
            return []

        # Fetch (or use cached) Kalshi markets
        kalshi_markets = self._kalshi_cache.get_markets()
        if not kalshi_markets:
            logger.info("No Kalshi markets available, skipping cross-platform scan")
            return []

        logger.info(
            "Cross-platform scan starting",
            kalshi_markets=len(kalshi_markets),
            polymarket_markets=len(markets),
        )

        # Build token index for Kalshi markets
        kalshi_token_index = self._refresh_kalshi_tokens(kalshi_markets)

        opportunities: list[ArbitrageOpportunity] = []
        pairs_checked = 0
        pairs_matched = 0
        best_unmatched_score = 0.0
        best_unmatched_pm = ""
        best_unmatched_k = ""

        for pm_market in markets:
            # Skip inactive, closed, or non-binary markets
            if pm_market.closed or not pm_market.active:
                continue
            if len(pm_market.outcome_prices) != 2:
                continue

            # Get Polymarket prices (use live prices if available)
            pm_yes = pm_market.yes_price
            pm_no = pm_market.no_price

            if pm_market.clob_token_ids:
                yes_token = (
                    pm_market.clob_token_ids[0]
                    if len(pm_market.clob_token_ids) > 0
                    else None
                )
                no_token = (
                    pm_market.clob_token_ids[1]
                    if len(pm_market.clob_token_ids) > 1
                    else None
                )
                if yes_token and yes_token in prices:
                    pm_yes = prices[yes_token].get("mid", pm_yes)
                if no_token and no_token in prices:
                    pm_no = prices[no_token].get("mid", pm_no)

            # Tokenize Polymarket question
            pm_tokens = _tokenize(pm_market.question)
            if not pm_tokens:
                continue

            pairs_checked += 1

            # Find the best-matching Kalshi market
            match = self._find_best_match(pm_tokens, kalshi_markets, kalshi_token_index)
            if match is None:
                # Track best near-miss for debugging
                best_score_for_pm = 0.0
                best_km_question = ""
                for km in kalshi_markets:
                    km_tokens = kalshi_token_index.get(km.id)
                    if not km_tokens:
                        continue
                    score = _jaccard_similarity(pm_tokens, km_tokens)
                    if score > best_score_for_pm:
                        best_score_for_pm = score
                        best_km_question = km.question
                if best_score_for_pm > best_unmatched_score:
                    best_unmatched_score = best_score_for_pm
                    best_unmatched_pm = pm_market.question[:80]
                    best_unmatched_k = best_km_question[:80]
                continue

            kalshi_market, similarity = match
            pairs_matched += 1

            # Get Kalshi prices
            k_yes = kalshi_market.yes_price
            k_no = kalshi_market.no_price

            # Calculate cross-platform arb
            arb = self._calculate_arb(pm_yes, pm_no, k_yes, k_no)
            if arb is None:
                continue

            # Build positions list with both platform references
            pm_token_id = None
            if pm_market.clob_token_ids:
                if arb["pm_outcome"] == "YES" and len(pm_market.clob_token_ids) > 0:
                    pm_token_id = pm_market.clob_token_ids[0]
                elif arb["pm_outcome"] == "NO" and len(pm_market.clob_token_ids) > 1:
                    pm_token_id = pm_market.clob_token_ids[1]

            positions = [
                {
                    "action": arb["pm_action"],
                    "outcome": arb["pm_outcome"],
                    "platform": "polymarket",
                    "price": arb["pm_price"],
                    "token_id": pm_token_id,
                    "market_id": pm_market.id,
                },
                {
                    "action": arb["k_action"],
                    "outcome": arb["k_outcome"],
                    "platform": "kalshi",
                    "price": arb["k_price"],
                    "ticker": kalshi_market.id,
                    "market_id": kalshi_market.id,
                },
            ]

            # Build description with price details and match confidence
            description = (
                f"{arb['desc']}. "
                f"Poly YES=${pm_yes:.3f} NO=${pm_no:.3f} | "
                f"Kalshi YES=${k_yes:.3f} NO=${k_no:.3f} | "
                f"Match similarity={similarity:.2f}"
            )

            # Find the parent event for this Polymarket market (if any)
            parent_event: Optional[Event] = None
            for event in events:
                for em in event.markets:
                    if em.id == pm_market.id:
                        parent_event = event
                        break
                if parent_event:
                    break

            # Use both markets for risk assessment.
            # Create a synthetic market list with both platforms' liquidity.
            risk_markets = [pm_market, kalshi_market]

            opp = self.create_opportunity(
                title=f"Cross-Platform: {pm_market.question[:60]}",
                description=description,
                total_cost=arb["total_cost"],
                markets=risk_markets,
                positions=positions,
                event=parent_event,
            )

            if opp:
                # Override the risk score to account for match confidence.
                # Lower match similarity -> higher risk.
                confidence_penalty = max(0.0, 0.30 * (1.0 - similarity))
                opp.risk_score = min(1.0, opp.risk_score + confidence_penalty)
                if similarity < 0.6:
                    opp.risk_factors.append(
                        f"Moderate match confidence ({similarity:.2f})"
                    )

                # Resolution divergence risk for player props and contingent markets.
                # Different platforms often have different rules for DNP (Did Not
                # Play), void, and cancellation — e.g., Polymarket may void and
                # refund while Kalshi resolves as NO.  This breaks the "guaranteed
                # profit" assumption because one leg can lose without the hedge.
                if _has_resolution_divergence_risk(
                    pm_market.question
                ) or _has_resolution_divergence_risk(kalshi_market.question):
                    opp.risk_score = min(1.0, opp.risk_score + 0.25)
                    opp.risk_factors.insert(
                        0,
                        "Resolution divergence risk: player prop / contingent market "
                        "— platforms may have different DNP/void/cancellation rules",
                    )

                opportunities.append(opp)

        logger.info(
            "Cross-platform scan complete",
            pairs_checked=pairs_checked,
            pairs_matched=pairs_matched,
            opportunities=len(opportunities),
        )

        # Log best near-miss to help debug matching issues
        if pairs_matched == 0 and best_unmatched_score > 0:
            logger.info(
                "Cross-platform best near-miss (no match reached threshold)",
                best_score=round(best_unmatched_score, 3),
                threshold=_MATCH_THRESHOLD,
                polymarket=best_unmatched_pm,
                kalshi=best_unmatched_k,
            )

        return opportunities
