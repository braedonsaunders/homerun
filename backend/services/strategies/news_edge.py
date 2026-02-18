"""
Strategy 10: News-Driven Edge Detection

Detects when breaking news creates an informational edge on a prediction
market — i.e., the LLM-estimated probability (given the news) diverges
from the current market price by more than a threshold.

Unlike the other 9 strategies which detect STRUCTURAL mispricings
(mathematical guarantees like YES+NO < $1), this strategy detects
INFORMATIONAL mispricings (the market hasn't priced in the news yet).

Pipeline:
  1. News feed service fetches articles from RSS/GDELT
  2. Semantic matcher embeds articles + markets, finds matches
  3. Edge detector estimates probability via LLM, computes edge
  4. This strategy converts edges into Opportunity objects

Because this is async (LLM calls), it runs differently from the sync
strategies. The scanner calls detect_async() instead of detect().
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.news.edge_detector import NewsEdge
from services.news.feed_service import news_feed_service
from services.news.semantic_matcher import MarketInfo, semantic_matcher
from services.strategies.base import BaseStrategy, DecisionCheck, ScoringWeights, SizingConfig, ExitDecision
from services.data_events import DataEvent
from services.quality_filter import QualityFilterOverrides
from services.strategy_sdk import StrategySDK

logger = logging.getLogger(__name__)

# Single-thread executor so all semantic_matcher + FAISS native calls run on
# the same OS thread.  PyTorch / FAISS use thread-local state; dispatching
# to arbitrary pool threads via the default executor causes segfaults.
_MATCHER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic")


class NewsEdgeStrategy(BaseStrategy):
    """
    Strategy 10: News-Driven Edge Detection

    Detects informational mispricings by matching news articles to
    markets and estimating probability shifts via LLM.
    """

    strategy_type = "news_edge"
    name = "News Edge"
    description = "Detect news-driven mispricings via semantic matching + LLM probability estimation"
    mispricing_type = "news_information"
    source_key = "news"
    worker_affinity = "news"
    requires_news_data = True
    allow_deduplication = False
    subscriptions = ["news_update"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=3.0,
        max_resolution_months=6.0,
    )

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:
        """Sync detect -- not used for this strategy.

        NewsEdgeStrategy requires async I/O (news fetching, LLM calls).
        The scanner calls detect_async() instead.
        """
        return []

    @staticmethod
    def _market_from_intent(intent: dict) -> Market:
        """Deserialize a news intent dict into a typed Market object.

        Intent dicts crossing the DataEvent boundary carry all the fields
        needed to construct a Market. We do this immediately at the boundary
        so every downstream code path (create_opportunity, fee model, risk
        scoring) sees a typed object — never a raw dict.
        """
        market_id = str(intent.get("market_id") or "")
        question = intent.get("market_question") or ""
        entry_price = float(intent.get("entry_price") or 0.0)
        # outcome_prices: [yes_price, no_price] — derive from direction + entry
        direction = intent.get("direction") or ""
        if direction == "buy_yes":
            yes_price = entry_price
            no_price = round(1.0 - entry_price, 6)
        else:
            no_price = entry_price
            yes_price = round(1.0 - entry_price, 6)
        liquidity = float(intent.get("liquidity") or intent.get("market_liquidity") or 500.0)
        return Market(
            id=market_id,
            condition_id=market_id,
            question=question,
            slug=intent.get("market_slug") or market_id,
            outcome_prices=[yes_price, no_price],
            liquidity=liquidity,
            platform="polymarket",
        )

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        """Convert news intents dispatched by the news worker into opportunities."""
        if event.event_type != "news_update":
            return []
        intents = event.payload.get("intents") or []
        if not intents:
            return []

        opportunities: list[Opportunity] = []
        for intent in intents:
            market_id = str(intent.get("market_id") or "")
            if not market_id:
                continue
            direction = intent.get("direction")
            entry_price = float(intent.get("entry_price") or 0.0)
            edge_percent = float(intent.get("edge_percent") or 0.0)
            confidence = float(intent.get("confidence") or 0.0)
            question = intent.get("market_question") or ""
            metadata = {
                k: v
                for k, v in intent.items()
                if k
                not in {
                    "id",
                    "market_id",
                    "market_question",
                    "direction",
                    "entry_price",
                    "edge_percent",
                    "confidence",
                    "status",
                    "created_at",
                }
            }

            side = "YES" if direction == "buy_yes" else "NO"
            target_price = entry_price + (edge_percent / 100.0) if entry_price > 0 else 0.0

            # Deserialize intent dict into a typed Market at the event boundary.
            # All downstream logic (fee model, risk scoring, create_opportunity)
            # sees a typed Market — never a raw dict.
            market = self._market_from_intent(intent)

            opp = self.create_opportunity(
                title=f"News Edge: {question[:50]}",
                description=f"News-driven {side} at ${entry_price:.2f} (edge: {edge_percent:.1f}%)",
                total_cost=entry_price,
                expected_payout=target_price,
                markets=[market],
                positions=[
                    {
                        "action": "BUY",
                        "outcome": side,
                        "price": entry_price,
                        "news_metadata": metadata,
                    }
                ],
                is_guaranteed=False,
                skip_fee_model=True,  # News edge uses its own net_profit calc
                custom_roi_percent=edge_percent,
                custom_risk_score=1.0 - confidence,
                confidence=confidence,
            )
            if opp is not None:
                opp.mispricing_type = MispricingType.NEWS_INFORMATION
                opportunities.append(opp)
        return opportunities

    async def detect_async(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        """Async detection: fetch news, match, estimate, generate opportunities."""
        if not settings.NEWS_EDGE_ENABLED:
            return []

        try:
            # Step 1: Fetch new articles
            await news_feed_service.fetch_all()
            all_articles = news_feed_service.get_articles(max_age_hours=settings.NEWS_ARTICLE_TTL_HOURS)

            if not all_articles:
                logger.debug("News Edge: no articles available")
                return []

            # Step 2: Build market index from scanner's market data
            market_infos = self._build_market_infos(events, markets, prices)
            if not market_infos:
                return []

            loop = asyncio.get_running_loop()

            if not semantic_matcher._initialized:
                await loop.run_in_executor(_MATCHER_EXECUTOR, semantic_matcher.initialize)

            await loop.run_in_executor(_MATCHER_EXECUTOR, semantic_matcher.update_market_index, market_infos)

            # Step 3: Embed new articles
            await loop.run_in_executor(_MATCHER_EXECUTOR, semantic_matcher.embed_articles, all_articles)

            # Step 4: Match articles to markets
            matches = await loop.run_in_executor(
                _MATCHER_EXECUTOR,
                semantic_matcher.match_articles_to_markets,
                all_articles,
                3,
                settings.NEWS_SIMILARITY_THRESHOLD,
            )

            if not matches:
                logger.debug("News Edge: no matches found")
                return []

            logger.info(
                "News Edge: %d articles, %d markets, %d matches",
                len(all_articles),
                len(market_infos),
                len(matches),
            )

            # Step 5: Estimate edges via LLM
            from services.news.edge_detector import edge_detector

            edges = await edge_detector.detect_edges(matches)

            # Step 6: Convert edges to Opportunity objects
            opportunities = []
            for edge in edges:
                opp = self._edge_to_opportunity(edge, markets, events, prices)
                if opp:
                    opportunities.append(opp)

            logger.info("News Edge: %d opportunities generated", len(opportunities))
            return opportunities

        except Exception as e:
            logger.error("News Edge strategy error: %s", e, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_market_infos(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[MarketInfo]:
        """Convert scanner's market data into MarketInfo for the matcher."""
        # Build event lookup
        event_map: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                event_map[m.id] = event

        infos: list[MarketInfo] = []
        for market in markets:
            if market.closed or not market.active:
                continue
            if market.liquidity < settings.MIN_LIQUIDITY_HARD:
                continue

            # Get live prices
            yes_price = market.yes_price
            no_price = market.no_price
            if market.clob_token_ids:
                if len(market.clob_token_ids) > 0:
                    tid = market.clob_token_ids[0]
                    if tid in prices:
                        yes_price = prices[tid].get("mid", yes_price)
                if len(market.clob_token_ids) > 1:
                    tid = market.clob_token_ids[1]
                    if tid in prices:
                        no_price = prices[tid].get("mid", no_price)

            event = event_map.get(market.id)
            infos.append(
                MarketInfo(
                    market_id=market.id,
                    question=market.question,
                    event_title=event.title if event else "",
                    category=event.category if event else "",
                    yes_price=yes_price,
                    no_price=no_price,
                    liquidity=market.liquidity,
                    slug=market.slug,
                    end_date=market.end_date.isoformat() if market.end_date else None,
                )
            )

        return infos

    def _edge_to_opportunity(
        self,
        edge: NewsEdge,
        markets: list[Market],
        events: list[Event],
        prices: dict[str, dict],
    ) -> Optional[Opportunity]:
        """Convert a NewsEdge into an Opportunity."""
        mi = edge.match.market

        # Find the actual Market object
        market = next((m for m in markets if m.id == mi.market_id), None)
        if not market:
            return None

        # Find the event
        event = None
        for e in events:
            if any(m.id == mi.market_id for m in e.markets):
                event = e
                break

        # Determine position
        if edge.direction == "buy_yes":
            side = "YES"
            entry_price = mi.yes_price
            target_price = edge.model_probability
            token_id = market.clob_token_ids[0] if market.clob_token_ids else None
        else:
            side = "NO"
            entry_price = mi.no_price
            target_price = 1.0 - edge.model_probability
            token_id = market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None

        # Profit calculation: if we buy at entry_price and the true probability
        # is target_price, our expected value is target_price per share.
        # Expected profit = target_price - entry_price (per $1 of shares).
        expected_payout = target_price
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee = expected_payout * self.fee
        net_profit = gross_profit - fee
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if roi < settings.NEWS_MIN_EDGE_PERCENT / 2:
            return None

        sizing = StrategySDK.resolve_position_sizing(
            liquidity_usd=market.liquidity,
            liquidity_fraction=0.05,
            hard_cap_usd=500.0,
            signal=None,
            default_min_size=float(settings.MIN_POSITION_SIZE),
        )
        min_liquidity = float(sizing.get("liquidity_usd", 0.0) or 0.0)
        max_position = float(sizing.get("max_position_size", 0.0) or 0.0)
        if not bool(sizing.get("is_tradeable", False)):
            return None

        # Risk scoring for news-driven trades (higher risk than pure arb)
        risk_score = 0.4  # Base: news trades are inherently riskier
        risk_factors = [
            "News-driven directional bet (not structural arbitrage)",
            f"Model confidence: {edge.confidence:.0%}",
            f"Semantic similarity: {edge.match.similarity:.2f}",
        ]

        if edge.confidence < 0.7:
            risk_score += 0.2
            risk_factors.append("Low model confidence")
        if edge.match.similarity < 0.6:
            risk_score += 0.1
            risk_factors.append("Moderate semantic match quality")

        risk_score = min(risk_score, 1.0)

        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                "_news_edge": {
                    "article_title": edge.match.article.title,
                    "article_url": edge.match.article.url,
                    "article_source": edge.match.article.source,
                    "model_probability": edge.model_probability,
                    "market_price": edge.market_price,
                    "edge_percent": edge.edge_percent,
                    "direction": edge.direction,
                    "confidence": edge.confidence,
                    "reasoning": edge.reasoning,
                    "similarity": edge.match.similarity,
                    "match_method": edge.match.match_method,
                },
            }
        ]

        opp = self.create_opportunity(
            title=f"News Edge: {market.question[:50]}...",
            description=(
                f"News suggests {side} at ${entry_price:.2f} "
                f"(model: {edge.model_probability:.0%}, "
                f"market: {edge.market_price:.0%}, "
                f"edge: {edge.edge_percent:.1f}%). "
                f"Source: {edge.match.article.title[:80]}"
            ),
            total_cost=total_cost,
            expected_payout=expected_payout,
            markets=[market],
            positions=positions,
            event=event,
            is_guaranteed=False,
            custom_roi_percent=roi,
            custom_risk_score=risk_score,
            confidence=edge.confidence,
        )
        if opp is not None:
            opp.risk_factors = risk_factors
            opp.min_liquidity = min_liquidity
            opp.max_position_size = max_position
            opp.mispricing_type = MispricingType.NEWS_INFORMATION

        return opp

    scoring_weights = ScoringWeights()
    sizing_config = SizingConfig()

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in {"news"}
        return [
            DecisionCheck("source", "News-capable source", source_ok, detail="news"),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        return (edge * 0.55) + (confidence * 45.0)

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        return max(1.0, min(max_size, base_size * (1.0 + confidence)))

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """News edge decays quickly -- exit on time or standard TP/SL."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        age_minutes = float(getattr(position, "age_minutes", 0) or 0)
        max_hold = float(config.get("max_hold_minutes", 240) or 240)
        if age_minutes > max_hold:
            current_price = market_state.get("current_price")
            return ExitDecision("close", f"News cycle decay ({age_minutes:.0f} min)", close_price=current_price)
        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
