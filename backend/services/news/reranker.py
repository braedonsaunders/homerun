"""
LLM Reranker -- Cuts false positives before expensive edge estimation.

Takes top-K retrieval candidates and calls an LLM to score relevance
of each (article, market) pair. Returns top-N with relevance score
and brief rationale.

Pattern from: Polymarket Agents (RAG reranking pipeline).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.news.hybrid_retriever import RetrievalCandidate

logger = logging.getLogger(__name__)


@dataclass
class RerankedCandidate:
    """A candidate after LLM reranking."""

    candidate: RetrievalCandidate
    relevance: float  # 0-1 from LLM
    rationale: str
    rerank_score: float  # Final combined score
    used_llm: bool = False

    @property
    def market_id(self) -> str:
        return self.candidate.market_id


# LLM reranking schema
RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The index of this market pair (0-based).",
                    },
                    "relevance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "How relevant this news article is to this specific market. "
                            "1.0 = directly about the same event/question. "
                            "0.0 = completely unrelated."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence explaining why this is or isn't relevant.",
                    },
                },
                "required": ["index", "relevance", "rationale"],
            },
        },
    },
    "required": ["pairs"],
}


class Reranker:
    """LLM-based reranking of retrieval candidates."""

    async def rerank(
        self,
        article_title: str,
        article_summary: str,
        candidates: list[RetrievalCandidate],
        top_n: int = 5,
        model: Optional[str] = None,
        allow_llm: bool = True,
    ) -> list[RerankedCandidate]:
        """Rerank candidates using LLM.

        Falls back to retrieval scores if LLM is unavailable.

        Args:
            article_title: News article title.
            article_summary: Article summary text.
            candidates: Candidates from hybrid retriever.
            top_n: Number of top candidates to return.
            model: LLM model override.

        Returns:
            Top-N RerankedCandidate sorted by rerank_score desc.
        """
        if not candidates:
            return []

        # Try LLM reranking
        reranked = None
        if allow_llm:
            reranked = await self._rerank_llm(
                article_title, article_summary, candidates, model=model
            )

        if reranked is not None:
            reranked.sort(key=lambda r: r.rerank_score, reverse=True)
            return reranked[:top_n]

        # Fallback: use retrieval scores directly
        results = [
            RerankedCandidate(
                candidate=c,
                relevance=c.combined_score,
                rationale="Retrieval score (LLM unavailable)",
                rerank_score=c.combined_score,
                used_llm=False,
            )
            for c in candidates
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

    async def _rerank_llm(
        self,
        article_title: str,
        article_summary: str,
        candidates: list[RetrievalCandidate],
        model: Optional[str] = None,
    ) -> Optional[list[RerankedCandidate]]:
        """LLM-based reranking."""
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return None
        except Exception:
            return None

        system_prompt = (
            "You are a prediction market relevance scorer. "
            "Given a news article and a list of prediction market questions, "
            "score how relevant the article is to each market.\n\n"
            "IMPORTANT:\n"
            "- A score of 1.0 means the article is DIRECTLY about the event "
            "the market is asking about.\n"
            "- A score of 0.5 means the article is tangentially related.\n"
            "- A score of 0.0 means the article has nothing to do with the market.\n"
            "- Be strict: most articles are NOT directly relevant to most markets.\n"
            "- Only score high when the article and market reference the same "
            "underlying event, not just the same broad category."
        )

        # Build market list
        market_lines = []
        for i, c in enumerate(candidates):
            market_lines.append(
                f"[{i}] {c.question}"
                + (f" (Event: {c.event_title})" if c.event_title else "")
                + (f" [{c.category}]" if c.category else "")
            )

        user_prompt = (
            f"NEWS ARTICLE:\n"
            f"  Title: {article_title}\n"
        )
        if article_summary:
            user_prompt += f"  Summary: {article_summary[:400]}\n"

        user_prompt += (
            f"\nMARKET QUESTIONS ({len(candidates)}):\n"
            + "\n".join(market_lines)
            + "\n\nScore each market's relevance to this article."
        )

        try:
            result = await manager.structured_output(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                schema=RERANK_SCHEMA,
                model=model,
                purpose="news_workflow_rerank",
            )

            pairs = result.get("pairs", [])
            reranked: list[RerankedCandidate] = []

            for pair in pairs:
                idx = pair.get("index", -1)
                if idx < 0 or idx >= len(candidates):
                    continue
                relevance = float(pair.get("relevance", 0.0))
                rationale = pair.get("rationale", "")

                c = candidates[idx]
                # Combine retrieval score with LLM relevance
                rerank_score = 0.3 * c.combined_score + 0.7 * relevance

                reranked.append(
                    RerankedCandidate(
                        candidate=c,
                        relevance=relevance,
                        rationale=rationale,
                        rerank_score=rerank_score,
                        used_llm=True,
                    )
                )

            # Include candidates not scored by LLM (fallback)
            scored_indices = {p.get("index") for p in pairs}
            for i, c in enumerate(candidates):
                if i not in scored_indices:
                    reranked.append(
                        RerankedCandidate(
                            candidate=c,
                            relevance=0.0,
                            rationale="Not scored by LLM",
                            rerank_score=c.combined_score * 0.3,
                            used_llm=False,
                        )
                    )

            return reranked

        except Exception as e:
            logger.debug("LLM reranking failed: %s", e)
            return None


# Singleton
reranker = Reranker()
