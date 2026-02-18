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
  4. This strategy converts edges into ArbitrageOpportunity objects

Because this is async (LLM calls), it runs differently from the sync
strategies. The scanner calls detect_async() instead of detect().
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.news.edge_detector import NewsEdge
from services.news.feed_service import news_feed_service
from services.news.semantic_matcher import MarketInfo, semantic_matcher
from services.strategies.base import BaseStrategy, DecisionCheck, ScoringWeights, SizingConfig, StrategyDecision, ExitDecision
from services.data_events import DataEvent
from utils.converters import to_float, to_confidence, clamp
from utils.signal_helpers import signal_payload, days_to_resolution, selected_probability, live_move

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

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
        """Sync detect -- not used for this strategy.

        NewsEdgeStrategy requires async I/O (news fetching, LLM calls).
        The scanner calls detect_async() instead.
        """
        return []

    async def on_event(self, event: DataEvent) -> list[ArbitrageOpportunity]:
        """Convert news intents dispatched by the news worker into opportunities."""
        if event.event_type != "news_update":
            return []
        intents = event.payload.get("intents") or []
        if not intents:
            return []

        opportunities: list[ArbitrageOpportunity] = []
        for intent in intents:
            market_id = str(intent.get("market_id") or "")
            if not market_id:
                continue
            direction = intent.get("direction")
            entry_price = float(intent.get("entry_price") or 0.0)
            edge_percent = float(intent.get("edge_percent") or 0.0)
            confidence = float(intent.get("confidence") or 0.0)
            question = intent.get("market_question") or ""
            metadata = {k: v for k, v in intent.items() if k not in {
                "id", "market_id", "market_question", "direction",
                "entry_price", "edge_percent", "confidence", "status", "created_at",
            }}

            side = "YES" if direction == "buy_yes" else "NO"
            target_price = entry_price + (edge_percent / 100.0) if entry_price > 0 else 0.0
            net_profit = (target_price - entry_price) - (target_price * self.fee) if target_price > entry_price else 0.0

            # NOTE: markets are pre-enriched dicts (only market_id/question from intent payload);
            # no ORM Market object is available here, so cannot route through create_opportunity().
            opp = ArbitrageOpportunity(
                strategy=self.strategy_type,
                title=f"News Edge: {question[:50]}",
                description=f"News-driven {side} at ${entry_price:.2f} (edge: {edge_percent:.1f}%)",
                total_cost=entry_price,
                expected_payout=target_price,
                gross_profit=target_price - entry_price,
                fee=target_price * self.fee,
                net_profit=net_profit,
                roi_percent=edge_percent,
                is_guaranteed=False,
                roi_type="directional_payout",
                risk_score=max(0.0, min(1.0, 1.0 - confidence)),
                risk_factors=["News-driven informational edge"],
                confidence=confidence,
                markets=[{"id": market_id, "question": question}],
                min_liquidity=0.0,
                max_position_size=500.0,
                positions_to_take=[{
                    "action": "BUY",
                    "outcome": side,
                    "price": entry_price,
                }],
                mispricing_type=MispricingType.NEWS_INFORMATION,
                strategy_context={"news_metadata": metadata},
            )
            opportunities.append(opp)
        return opportunities

    async def detect_async(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
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

            # Step 6: Convert edges to ArbitrageOpportunity objects
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
    ) -> Optional[ArbitrageOpportunity]:
        """Convert a NewsEdge into an ArbitrageOpportunity."""
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

        # Position sizing: conservative for directional bets
        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.05, 500.0)  # 5% of liquidity, max $500

        if max_position < settings.MIN_POSITION_SIZE:
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

        resolution_date = market.end_date

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

        market_dict = {
            "id": market.id,
            "slug": market.slug,
            "question": market.question,
            "yes_price": mi.yes_price,
            "no_price": mi.no_price,
            "liquidity": market.liquidity,
        }

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

    def compute_score(self, edge: float, confidence: float, risk_score: float,
                      market_count: int, payload: dict) -> float:
        return (edge * 0.55) + (confidence * 45.0)

    def compute_size(self, base_size: float, max_size: float, edge: float,
                     confidence: float, risk_score: float, market_count: int) -> float:
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
