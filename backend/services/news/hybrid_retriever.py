"""
Hybrid Retriever -- Weighted composite market retrieval.

Given an extracted event and article text, retrieves candidate markets
using a weighted combination of:
  - keyword_score (BM25 of event entities against market text)
  - semantic_score (cosine similarity of article embedding vs market embedding)
  - event_score (event-type to market-category affinity matrix)

Configurable weights from AppSettings.

Pattern from: Quant-tool (multi-factor scoring), Polymarket Agents (RAG retrieval).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from services.news.event_extractor import ExtractedEvent, EVENT_CATEGORY_AFFINITY
from services.news.market_watcher_index import (
    MarketWatcherIndex,
    SearchResult,
    _tokenize,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A market candidate with full score breakdown."""

    market_id: str
    question: str
    event_title: str
    category: str
    yes_price: float
    no_price: float
    liquidity: float
    slug: str
    keyword_score: float
    semantic_score: float
    event_score: float
    combined_score: float


class HybridRetriever:
    """Retrieves candidate markets for an extracted event using hybrid scoring."""

    def __init__(self, index: MarketWatcherIndex) -> None:
        self._index = index

    def retrieve(
        self,
        event: ExtractedEvent,
        article_text: str,
        top_k: int = 8,
        keyword_weight: float = 0.25,
        semantic_weight: float = 0.45,
        event_weight: float = 0.30,
        min_liquidity: float = 0.0,
        similarity_threshold: float = 0.05,
    ) -> list[RetrievalCandidate]:
        """Retrieve candidate markets for an event.

        Args:
            event: Extracted event from the article.
            article_text: Full text for embedding (title + summary).
            top_k: Max candidates to return.
            keyword_weight: Weight for BM25 keyword score.
            semantic_weight: Weight for semantic similarity.
            event_weight: Weight for event-type category affinity.
            min_liquidity: Minimum liquidity filter.
            similarity_threshold: Minimum combined score to include.

        Returns:
            List of RetrievalCandidate sorted by combined_score desc.
        """
        # Build query tokens from event
        query_terms = _tokenize(" ".join(event.search_terms))

        # Get article embedding for semantic search
        article_embedding: Optional[np.ndarray] = None
        if self._index.is_ml_mode:
            article_embedding = self._index.embed_text(article_text)

        # Category filter based on event type affinity
        affinity_categories = event.category_affinities

        # Search the index (keyword + semantic)
        # Don't category-filter at the index level -- we'll boost by affinity instead
        raw_results = self._index.search(
            query_terms=query_terms,
            query_embedding=article_embedding,
            category_filter=None,
            min_liquidity=min_liquidity,
            top_k=top_k * 3,  # Get more for re-scoring
            keyword_weight=1.0,  # Raw scores, we'll re-weight
            semantic_weight=1.0,
        )

        # Re-score with event affinity
        candidates: list[RetrievalCandidate] = []
        for result in raw_results:
            market = result.market

            # Event-type to category affinity score
            event_score = 0.0
            if affinity_categories and market.category:
                if market.category in affinity_categories:
                    event_score = 1.0
                else:
                    event_score = 0.1  # Small boost for any active market

            # Weighted combination
            combined = (
                keyword_weight * result.keyword_score
                + semantic_weight * result.semantic_score
                + event_weight * event_score
            )

            if combined >= similarity_threshold:
                candidates.append(
                    RetrievalCandidate(
                        market_id=market.market_id,
                        question=market.question,
                        event_title=market.event_title,
                        category=market.category,
                        yes_price=market.yes_price,
                        no_price=market.no_price,
                        liquidity=market.liquidity,
                        slug=market.slug,
                        keyword_score=result.keyword_score,
                        semantic_score=result.semantic_score,
                        event_score=event_score,
                        combined_score=combined,
                    )
                )

        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates[:top_k]
