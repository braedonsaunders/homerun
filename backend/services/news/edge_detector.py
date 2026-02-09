"""
News edge detector.

Takes matched (article, market) pairs from the semantic matcher and
uses an LLM to estimate the true probability of the market outcome
given the news. When the model probability diverges from the market
price by more than the configured threshold, an edge is detected.

This is the core bridge between Options 1/2 (ingestion + matching)
and the scanner strategy output.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from services.news.semantic_matcher import NewsMarketMatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NewsEdge:
    """A detected informational edge from news analysis."""

    match: NewsMarketMatch
    model_probability: float  # LLM-estimated P(YES)
    market_price: float  # Current market YES price
    edge_percent: float  # |model_prob - market_price| * 100
    direction: str  # "buy_yes" or "buy_no"
    confidence: float  # 0-1
    reasoning: str
    estimated_at: datetime

    @property
    def market_id(self) -> str:
        return self.match.market.market_id

    @property
    def article_title(self) -> str:
        return self.match.article.title


# ---------------------------------------------------------------------------
# Probability estimation schema
# ---------------------------------------------------------------------------

PROBABILITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability_yes": {
            "type": "number",
            "minimum": 0.01,
            "maximum": 0.99,
            "description": (
                "Your estimated probability that this market resolves YES, "
                "as a decimal between 0.01 and 0.99."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "How confident you are in this estimate (0=no idea, 1=certain). "
                "Consider the quality and relevance of the news evidence."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Brief reasoning explaining how the news affects the probability. "
                "1-3 sentences max."
            ),
        },
        "news_relevance": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
            "description": "How relevant this news article actually is to the market.",
        },
    },
    "required": ["probability_yes", "confidence", "reasoning", "news_relevance"],
}


# ---------------------------------------------------------------------------
# Edge Detector
# ---------------------------------------------------------------------------


class NewsEdgeDetector:
    """
    Estimates probabilities and detects edges for news-market matches.

    For each (article, market) pair, asks the LLM:
    "Given this news, what is the probability this market resolves YES?"

    Then compares the LLM estimate to the current market price.
    """

    # Cost control: max concurrent LLM calls
    _CONCURRENCY = 3

    async def detect_edges(
        self,
        matches: list[NewsMarketMatch],
        model: Optional[str] = None,
    ) -> list[NewsEdge]:
        """Evaluate matches and return those with significant edges.

        Args:
            matches: Semantic matches from the matcher.
            model: LLM model to use.

        Returns:
            NewsEdge list, sorted by edge_percent descending.
        """
        if not matches:
            return []

        # Deduplicate by (article_id, market_id) — keep highest similarity
        seen: dict[str, NewsMarketMatch] = {}
        for match in matches:
            key = f"{match.article.article_id}:{match.market.market_id}"
            if key not in seen or match.similarity > seen[key].similarity:
                seen[key] = match

        unique_matches = list(seen.values())

        # Cap to avoid runaway LLM costs
        max_evals = settings.NEWS_MAX_OPPORTUNITIES_PER_SCAN
        unique_matches = unique_matches[:max_evals]

        # Evaluate in parallel with concurrency limit
        sem = asyncio.Semaphore(self._CONCURRENCY)
        edges: list[NewsEdge] = []

        async def _evaluate_one(match: NewsMarketMatch) -> Optional[NewsEdge]:
            async with sem:
                try:
                    return await self._estimate_edge(match, model=model)
                except Exception as e:
                    logger.debug("Edge estimation failed: %s", e)
                    return None

        tasks = [_evaluate_one(m) for m in unique_matches]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None and result.edge_percent >= settings.NEWS_MIN_EDGE_PERCENT:
                edges.append(result)

        edges.sort(key=lambda e: e.edge_percent, reverse=True)
        logger.info("Edge detection: %d edges found from %d matches", len(edges), len(unique_matches))
        return edges

    async def _estimate_edge(
        self,
        match: NewsMarketMatch,
        model: Optional[str] = None,
    ) -> Optional[NewsEdge]:
        """Estimate probability for a single match and compute edge."""
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return None
        except Exception:
            return None

        market = match.market
        article = match.article

        system_prompt = (
            "You are a calibrated prediction market forecaster. "
            "Given a news article and a prediction market question, estimate "
            "the probability that the market resolves YES.\n\n"
            "IMPORTANT:\n"
            "- Base your estimate on the news evidence provided.\n"
            "- If the news is not relevant to the market, set news_relevance "
            "to 'none' and set confidence to 0.\n"
            "- Be well-calibrated: don't be overconfident. When uncertain, "
            "stay closer to 50%.\n"
            "- Consider that prediction markets are often efficient — "
            "the current price reflects information from many participants.\n"
            "- Only deviate significantly from the market price if the news "
            "provides STRONG, CLEAR evidence of a probability shift."
        )

        user_prompt = (
            f"MARKET QUESTION: {market.question}\n"
            f"EVENT: {market.event_title}\n"
            f"CATEGORY: {market.category}\n"
            f"CURRENT YES PRICE: ${market.yes_price:.2f}\n"
            f"CURRENT NO PRICE: ${market.no_price:.2f}\n\n"
            f"NEWS ARTICLE:\n"
            f"  Title: {article.title}\n"
            f"  Source: {article.source}\n"
            f"  Published: {article.published or 'Unknown'}\n"
        )
        if article.summary:
            user_prompt += f"  Summary: {article.summary}\n"

        user_prompt += (
            "\nBased on this news, what is the probability that the market "
            "resolves YES? Consider the current market price as a baseline."
        )

        result = await manager.structured_output(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            schema=PROBABILITY_SCHEMA,
            model=model,
            purpose="news_edge_detection",
        )

        # Skip irrelevant matches
        if result.get("news_relevance") == "none":
            return None

        prob_yes = float(result.get("probability_yes", 0.5))
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")

        # Skip low-confidence estimates
        if confidence < settings.NEWS_MIN_CONFIDENCE:
            return None

        # Compute edge
        market_price = market.yes_price
        edge = abs(prob_yes - market_price) * 100

        # Direction: should we buy YES or NO?
        if prob_yes > market_price:
            direction = "buy_yes"
        else:
            direction = "buy_no"

        return NewsEdge(
            match=match,
            model_probability=prob_yes,
            market_price=market_price,
            edge_percent=edge,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            estimated_at=datetime.now(timezone.utc),
        )


# ======================================================================
# Singleton
# ======================================================================

edge_detector = NewsEdgeDetector()
