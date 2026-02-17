"""
Hybrid Retriever -- Weighted composite market retrieval.

Given an extracted event and article text, retrieves candidate markets
using a weighted combination of:
  - keyword_score (BM25 of event entities against market text)
  - semantic_score (cosine similarity of article embedding vs market embedding)
  - event_score (event-type to market-category affinity matrix)
  - entity_overlap_score (entity-type aware overlap between event and market)

Configurable weights from AppSettings.

Pattern from: Quant-tool (multi-factor scoring), Polymarket Agents (RAG retrieval).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from services.news.event_extractor import ExtractedEvent
from services.news.market_watcher_index import (
    MarketWatcherIndex,
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
    end_date: Optional[str]
    tags: list[str]
    keyword_score: float
    semantic_score: float
    event_score: float
    combined_score: float
    entity_overlap_score: float = 0.0


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
        entity_weight: float = 0.0,
        min_liquidity: float = 0.0,
        similarity_threshold: float = 0.42,
        min_keyword_signal: float = 0.04,
        min_semantic_signal: float = 0.22,
        min_text_overlap_tokens: int = 1,
    ) -> list[RetrievalCandidate]:
        """Retrieve candidate markets for an event.

        Args:
            event: Extracted event from the article.
            article_text: Full text for embedding (title + summary).
            top_k: Max candidates to return.
            keyword_weight: Weight for BM25 keyword score.
            semantic_weight: Weight for semantic similarity.
            event_weight: Weight for event-type category affinity.
            entity_weight: Weight for entity-type overlap score (0-1).
                When non-zero, the other weights are renormalised so they
                still sum to 1.0.  Recommended: 0.0 to 0.20.
            min_liquidity: Minimum liquidity filter.
            similarity_threshold: Minimum combined score to include.
            min_keyword_signal: Floor for lexical signal.
            min_semantic_signal: Floor for semantic signal.
            min_text_overlap_tokens: Minimum overlap between event and market tokens.

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
        event_tokens = self._event_alignment_tokens(event)

        # Pre-compute article entity info for entity overlap scoring.
        # Lazy import to avoid circular dependency.
        article_ent_info = None
        if entity_weight > 0:
            try:
                from services.news.reranker import _extract_entities
                article_ent_info = _extract_entities(article_text)
            except Exception:
                article_ent_info = None

        # If entity_weight is non-zero, renormalise the other three weights so
        # the total still sums to ~1.0.  This keeps existing thresholds valid.
        eff_kw_weight = keyword_weight
        eff_sem_weight = semantic_weight
        eff_evt_weight = event_weight
        eff_ent_weight = max(0.0, min(1.0, entity_weight))
        if eff_ent_weight > 0:
            base_sum = eff_kw_weight + eff_sem_weight + eff_evt_weight
            if base_sum > 0:
                scale = (1.0 - eff_ent_weight) / base_sum
                eff_kw_weight *= scale
                eff_sem_weight *= scale
                eff_evt_weight *= scale

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

        # Re-score with event affinity and entity overlap
        candidates: list[RetrievalCandidate] = []
        for result in raw_results:
            market = result.market

            market_tokens = set(
                _tokenize(
                    " ".join(
                        [
                            market.question,
                            market.event_title,
                            market.slug,
                            " ".join(market.tags or []),
                        ]
                    )
                )
            )
            overlap_count = len(event_tokens.intersection(market_tokens))
            overlap_ratio = overlap_count / max(1, len(event_tokens)) if event_tokens else 0.0

            # Event alignment score blends category affinity and entity overlap.
            category_score = 0.0
            if affinity_categories and market.category:
                if market.category in affinity_categories:
                    category_score = 1.0
            event_score = min(1.0, (0.35 * category_score) + (0.65 * overlap_ratio))

            has_textual_signal = (
                result.keyword_score >= min_keyword_signal or result.semantic_score >= min_semantic_signal
            )
            if not has_textual_signal:
                continue

            if min_text_overlap_tokens > 0 and overlap_count < min_text_overlap_tokens:
                continue

            # Entity-type overlap score (0-1).  Falls back to 0.5 (neutral)
            # when entity extraction is unavailable or weight is zero.
            ent_overlap_score = 0.5
            if eff_ent_weight > 0 and article_ent_info is not None:
                try:
                    from services.news.reranker import (
                        _compute_entity_overlap,
                        _extract_entities,
                    )
                    market_text = f"{market.question} {market.event_title} {market.category} {' '.join(market.tags or [])}"
                    market_ent_info = _extract_entities(market_text)
                    overlap_result = _compute_entity_overlap(article_ent_info, market_ent_info)
                    ent_overlap_score = overlap_result.get("entity_overlap_score", 0.5)
                except Exception:
                    ent_overlap_score = 0.5

            # Weighted combination
            combined = (
                eff_kw_weight * result.keyword_score
                + eff_sem_weight * result.semantic_score
                + eff_evt_weight * event_score
                + eff_ent_weight * ent_overlap_score
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
                        end_date=market.end_date,
                        tags=list(market.tags or []),
                        keyword_score=result.keyword_score,
                        semantic_score=result.semantic_score,
                        event_score=event_score,
                        combined_score=combined,
                        entity_overlap_score=ent_overlap_score,
                    )
                )

        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _event_alignment_tokens(event: ExtractedEvent) -> set[str]:
        terms = []
        terms.extend(event.key_entities or [])
        terms.extend(event.actors or [])
        if event.action:
            terms.append(event.action)
        return set(_tokenize(" ".join(t for t in terms if isinstance(t, str))))
