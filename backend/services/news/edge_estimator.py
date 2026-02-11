"""
Edge Estimator -- LLM probability estimation with full evidence chain.

Improved version of edge_detector.py for the independent news workflow.
Produces persisted NewsWorkflowFinding objects with complete audit trail.

Improvements over edge_detector.py:
- Full evidence chain (article snippet, event graph, retrieval scores, reasoning)
- Calibration prompt improvements (from KalshiBench findings)
- news_relevance filter before edge calc
- Confidence calibration with explicit uncertainty acknowledgment

Pattern from: KalshiBench (calibration), Quant-tool (evidence trail).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from services.news.event_extractor import ExtractedEvent
from services.news.reranker import RerankedCandidate

logger = logging.getLogger(__name__)


@dataclass
class WorkflowFinding:
    """A finding from the news workflow pipeline with full evidence chain."""

    id: str = ""
    article_id: str = ""
    market_id: str = ""
    article_title: str = ""
    article_source: str = ""
    article_url: str = ""
    market_question: str = ""
    market_price: float = 0.5
    model_probability: float = 0.5
    edge_percent: float = 0.0
    direction: str = "buy_yes"
    confidence: float = 0.0
    retrieval_score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    event_score: float = 0.0
    rerank_score: float = 0.0
    event_graph: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    reasoning: str = ""
    actionable: bool = False
    signal_key: Optional[str] = None
    cache_key: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# LLM probability estimation schema (calibrated)
# ---------------------------------------------------------------------------

EDGE_ESTIMATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability_yes": {
            "type": "number",
            "minimum": 0.01,
            "maximum": 0.99,
            "description": (
                "Your estimated probability that this market resolves YES, "
                "as a decimal between 0.01 and 0.99. "
                "IMPORTANT: prediction markets are often efficient. "
                "Only diverge significantly from the current price if the "
                "news provides STRONG, CLEAR evidence of a probability shift."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "How confident you are in this estimate (0=complete guess, 1=certain). "
                "Consider: Is the news directly about this market? "
                "Is the information new (not already priced in)? "
                "Could there be other interpretations?"
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Explain in 2-4 sentences how the news affects probability. "
                "Include: what the news says, how it relates to the market, "
                "and why you believe the market hasn't priced this in yet."
            ),
        },
        "news_relevance": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
            "description": (
                "How relevant this news article actually is to the market. "
                "'high' = directly about the market topic. "
                "'medium' = related but indirect. "
                "'low' = tangentially related. "
                "'none' = not relevant at all."
            ),
        },
        "information_novelty": {
            "type": "string",
            "enum": ["breaking", "recent", "known", "stale"],
            "description": (
                "How new is this information? "
                "'breaking' = just happened, unlikely priced in. "
                "'recent' = hours old, partially priced in. "
                "'known' = widely known, likely already priced in. "
                "'stale' = old news, definitely priced in."
            ),
        },
    },
    "required": [
        "probability_yes",
        "confidence",
        "reasoning",
        "news_relevance",
        "information_novelty",
    ],
}


class EdgeEstimator:
    """Estimates probability edges for reranked article-market pairs."""

    _CONCURRENCY = 3

    async def estimate_batch(
        self,
        article_title: str,
        article_summary: str,
        article_source: str,
        article_url: str,
        article_id: str,
        event: ExtractedEvent,
        reranked: list[RerankedCandidate],
        min_edge_percent: float = 8.0,
        min_confidence: float = 0.6,
        model: Optional[str] = None,
        allow_llm: bool = True,
        max_llm_calls: Optional[int] = None,
    ) -> list[WorkflowFinding]:
        """Estimate edges for a batch of reranked candidates.

        Returns only findings that pass the edge and confidence thresholds.
        """
        import asyncio

        if not reranked:
            return []

        sem = asyncio.Semaphore(self._CONCURRENCY)
        findings: list[WorkflowFinding] = []

        llm_budget = (
            max(0, max_llm_calls if max_llm_calls is not None else len(reranked))
            if allow_llm
            else 0
        )

        async def _one(i: int, rc: RerankedCandidate) -> Optional[WorkflowFinding]:
            async with sem:
                return await self._estimate_one(
                    article_title=article_title,
                    article_summary=article_summary,
                    article_source=article_source,
                    article_url=article_url,
                    article_id=article_id,
                    event=event,
                    rc=rc,
                    model=model,
                    allow_llm=i < llm_budget,
                )

        results = await asyncio.gather(*[_one(i, rc) for i, rc in enumerate(reranked)])

        for finding in results:
            if finding is None:
                continue
            if finding.edge_percent >= min_edge_percent and finding.confidence >= min_confidence:
                finding.actionable = True
            findings.append(finding)

        findings.sort(key=lambda f: f.edge_percent, reverse=True)
        return findings

    async def _estimate_one(
        self,
        article_title: str,
        article_summary: str,
        article_source: str,
        article_url: str,
        article_id: str,
        event: ExtractedEvent,
        rc: RerankedCandidate,
        model: Optional[str] = None,
        allow_llm: bool = True,
    ) -> Optional[WorkflowFinding]:
        """Estimate edge for a single article-market pair."""
        c = rc.candidate

        # Build evidence chain
        evidence = {
            "retrieval": {
                "keyword_score": round(c.keyword_score, 4),
                "semantic_score": round(c.semantic_score, 4),
                "event_score": round(c.event_score, 4),
                "combined_score": round(c.combined_score, 4),
            },
            "rerank": {
                "relevance": round(rc.relevance, 4),
                "rationale": rc.rationale,
                "rerank_score": round(rc.rerank_score, 4),
            },
            "event": {
                "type": event.event_type,
                "actors": event.actors,
                "action": event.action,
                "key_entities": event.key_entities,
            },
        }

        event_graph = {
            "event_type": event.event_type,
            "actors": event.actors,
            "action": event.action,
            "date": event.date,
            "region": event.region,
            "impact_direction": event.impact_direction,
            "key_entities": event.key_entities,
        }

        # Try LLM estimation
        llm_result = None
        if allow_llm:
            llm_result = await self._call_llm(
                article_title=article_title,
                article_summary=article_summary,
                market_question=c.question,
                event_title=c.event_title,
                category=c.category,
                yes_price=c.yes_price,
                no_price=c.no_price,
                model=model,
            )

        if llm_result is None:
            # No probability estimate means no actionable signal.
            return None

        # Filter irrelevant
        if llm_result.get("news_relevance") in {"none", "low"}:
            return None

        # Filter stale info (likely already priced in)
        if llm_result.get("information_novelty") == "stale":
            return None

        prob_yes = float(llm_result.get("probability_yes", 0.5))
        confidence = float(llm_result.get("confidence", 0.0))
        reasoning = llm_result.get("reasoning", "")

        # Compute edge
        market_price = c.yes_price
        edge = abs(prob_yes - market_price) * 100
        direction = "buy_yes" if prob_yes > market_price else "buy_no"

        # Novelty-adjusted confidence
        novelty = llm_result.get("information_novelty", "known")
        novelty_mult = {"breaking": 1.0, "recent": 0.85, "known": 0.5, "stale": 0.1}
        confidence *= novelty_mult.get(novelty, 0.5)
        if confidence < 0.2:
            return None

        evidence["llm"] = {
            "probability_yes": prob_yes,
            "confidence": confidence,
            "news_relevance": llm_result.get("news_relevance"),
            "information_novelty": novelty,
        }

        return WorkflowFinding(
            id=uuid.uuid4().hex[:16],
            article_id=article_id,
            market_id=c.market_id,
            article_title=article_title,
            article_source=article_source,
            article_url=article_url,
            market_question=c.question,
            market_price=market_price,
            model_probability=prob_yes,
            edge_percent=edge,
            direction=direction,
            confidence=confidence,
            retrieval_score=c.combined_score,
            semantic_score=c.semantic_score,
            keyword_score=c.keyword_score,
            event_score=c.event_score,
            rerank_score=rc.rerank_score,
            event_graph=event_graph,
            evidence=evidence,
            reasoning=reasoning,
        )

    async def _call_llm(
        self,
        article_title: str,
        article_summary: str,
        market_question: str,
        event_title: str,
        category: str,
        yes_price: float,
        no_price: float,
        model: Optional[str] = None,
    ) -> Optional[dict]:
        """Call LLM for probability estimation."""
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return None
        except Exception:
            return None

        system_prompt = (
            "You are a calibrated prediction market forecaster. "
            "Given a news article and a prediction market question, estimate "
            "the probability that the market resolves YES.\n\n"
            "CALIBRATION GUIDELINES (from KalshiBench research):\n"
            "- Prediction markets are often efficient. The current price "
            "reflects information from many participants.\n"
            "- LLMs tend to be OVERCONFIDENT. When uncertain, stay closer "
            "to the current market price.\n"
            "- Only deviate significantly (>10%) from the market price if "
            "the news provides STRONG, CLEAR, and NOVEL evidence.\n"
            "- Consider whether this information is already priced in.\n"
            "- A 90%+ confidence rating should be reserved for near-certain outcomes."
        )

        user_prompt = (
            f"MARKET QUESTION: {market_question}\n"
            f"EVENT: {event_title}\n"
            f"CATEGORY: {category}\n"
            f"CURRENT YES PRICE: ${yes_price:.2f}\n"
            f"CURRENT NO PRICE: ${no_price:.2f}\n\n"
            f"NEWS ARTICLE:\n"
            f"  Title: {article_title}\n"
        )
        if article_summary:
            user_prompt += f"  Summary: {article_summary[:500]}\n"

        user_prompt += (
            "\nBased on this news, what is the probability that the market "
            "resolves YES? Consider the current market price as a strong baseline."
        )

        try:
            return await manager.structured_output(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                schema=EDGE_ESTIMATION_SCHEMA,
                model=model,
                purpose="news_workflow_edge_estimation",
            )
        except Exception as e:
            logger.debug("Edge estimation LLM call failed: %s", e)
            return None


# Singleton
edge_estimator = EdgeEstimator()
