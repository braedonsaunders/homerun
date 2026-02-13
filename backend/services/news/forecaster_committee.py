"""
Forecaster Committee — Multi-Agent Probability Engine.

Implements a superforecasting-inspired multi-agent system:

1. Outside View Agent — Reference class forecasting with base rates
2. Inside View Agent  — Causal mechanism analysis of specific evidence
3. Adversarial Critic — Challenges both views, finds biases and flaws
4. Aggregator         — Weighted median with extremization

This is ON-DEMAND only — too expensive for continuous scanning.
Call via the API when you want deep analysis of a specific market.

Inspired by:
- Tetlock's superforecasting methodology
- Halawi et al. "Approaching Human-Level Forecasting with LLMs"
- Metaculus AI Benchmarking (multi-agent bots outperform single-agent)
- Panshul42/Forecasting_Bot_Q2 (inside/outside view committee)
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentEstimate:
    """Probability estimate from a single agent."""

    agent_name: str
    probability: float  # P(YES) between 0.01 and 0.99
    confidence: float  # 0-1
    reasoning: str
    model_used: str = ""
    tokens_used: int = 0


@dataclass
class CommitteeResult:
    """Aggregated result from the full committee."""

    market_question: str
    market_price: float
    final_probability: float
    edge_percent: float
    direction: str  # "buy_yes" or "buy_no"
    confidence: float
    agent_estimates: list[AgentEstimate] = field(default_factory=list)
    aggregation_method: str = "weighted_median_extremized"
    news_context: str = ""
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "market_question": self.market_question,
            "market_price": self.market_price,
            "final_probability": self.final_probability,
            "edge_percent": self.edge_percent,
            "direction": self.direction,
            "confidence": self.confidence,
            "aggregation_method": self.aggregation_method,
            "news_context": self.news_context[:500] if self.news_context else "",
            "analyzed_at": self.analyzed_at.isoformat(),
            "agents": [
                {
                    "name": a.agent_name,
                    "probability": a.probability,
                    "confidence": a.confidence,
                    "reasoning": a.reasoning,
                    "model": a.model_used,
                }
                for a in self.agent_estimates
            ],
        }


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

_ESTIMATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability_yes": {
            "type": "number",
            "minimum": 0.01,
            "maximum": 0.99,
            "description": "Your estimated probability that this market resolves YES.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How confident you are in this estimate.",
        },
        "reasoning": {
            "type": "string",
            "description": "Your step-by-step reasoning (2-5 sentences).",
        },
    },
    "required": ["probability_yes", "confidence", "reasoning"],
}

_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "adjusted_probability_yes": {
            "type": "number",
            "minimum": 0.01,
            "maximum": 0.99,
            "description": "Your bias-adjusted probability estimate.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in your adjusted estimate.",
        },
        "biases_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cognitive biases identified in the other agents' reasoning.",
        },
        "reasoning": {
            "type": "string",
            "description": "Your critique and justification for adjustments.",
        },
    },
    "required": [
        "adjusted_probability_yes",
        "confidence",
        "biases_found",
        "reasoning",
    ],
}


# ---------------------------------------------------------------------------
# Agent Prompts
# ---------------------------------------------------------------------------

_OUTSIDE_VIEW_SYSTEM = """\
You are a superforecaster using the OUTSIDE VIEW methodology (reference class forecasting).

Your approach:
1. IDENTIFY the reference class: What category of event is this? What is the \
broadest applicable class?
2. FIND the base rate: How often do events in this reference class resolve YES \
historically? If you don't know an exact number, estimate the order of magnitude.
3. Make SMALL adjustments: Starting from the base rate, adjust by no more than \
10-15% based on the specific details, UNLESS the evidence is definitively \
conclusive (e.g., an official announcement that directly resolves the question).
4. RESIST the temptation to over-adjust. The outside view's power comes from \
anchoring to base rates and resisting narrative thinking.

Common reference classes and base rates to consider:
- Political promises fulfilled: ~30-60% depending on type
- Legislation passing: ~10-30% for novel proposals, 50-80% for renewals
- International agreements: ~20-40% within stated timeline
- Tech product launches on schedule: ~40-70%
- Geopolitical conflicts escalating: ~15-30% in any given period
- Economic targets being met: ~30-50%

Adjust these based on available evidence, but stay anchored."""

_INSIDE_VIEW_SYSTEM = """\
You are a superforecaster using the INSIDE VIEW methodology (causal reasoning).

Your approach:
1. MAP the causal chain: What specific sequence of events would lead to YES? \
What would lead to NO? Be concrete and specific.
2. ANALYZE the news evidence: How does this specific news shift the causal \
balance? Which causal paths does it strengthen or weaken?
3. Consider SECOND-ORDER effects: What consequences follow from this news? \
Who else will react, and how will their reactions affect the outcome?
4. Identify MISSING information: What key facts would change your estimate \
significantly? How should this uncertainty affect your probability?
5. Consider TIMING: Is this news early enough to matter, or is the resolution \
date too far away for it to be predictive?

Be specific about mechanisms. Don't just say "this increases the probability" — \
explain exactly WHY and through what causal path."""

_CRITIC_SYSTEM = """\
You are a forecasting critic and debiaser. Your job is to find FLAWS in \
probability estimates from two other forecasters.

You receive:
- The Outside View estimate (base-rate anchored)
- The Inside View estimate (causal reasoning)
- The current market price (reflects crowd wisdom)

Your job:
1. IDENTIFY cognitive biases in both estimates:
   - Anchoring bias (over-relying on the market price or a salient number)
   - Recency bias (over-weighting recent news vs. base rates)
   - Availability bias (over-weighting vivid/memorable events)
   - Narrative bias (constructing a compelling story and over-trusting it)
   - Overconfidence (too extreme a probability given available evidence)
   - Conjunction fallacy (overestimating probability of specific scenarios)
2. CHECK if the news actually supports the claimed probability shift. Is the \
evidence strong enough to justify deviating from the market consensus?
3. Consider why the MARKET might be RIGHT. Markets aggregate information from \
many participants, many of whom have seen this same news already.
4. Provide your own ADJUSTED probability that corrects for the biases found.

IMPORTANT: The market is often efficient. Genuine informational edges require \
strong, unambiguous evidence. When in doubt, move TOWARD the market price, not \
away from it."""


# ---------------------------------------------------------------------------
# ForecasterCommittee
# ---------------------------------------------------------------------------


class ForecasterCommittee:
    """
    Multi-agent forecasting system for deep market analysis.

    Usage:
        committee = ForecasterCommittee()
        result = await committee.analyze(
            market_question="Will X happen by Y?",
            market_price=0.55,
            news_context="Recent article says...",
        )
    """

    async def analyze(
        self,
        market_question: str,
        market_price: float,
        news_context: str = "",
        event_title: str = "",
        category: str = "",
        model: Optional[str] = None,
    ) -> CommitteeResult:
        """Run the full committee analysis.

        Args:
            market_question: The market question text.
            market_price: Current YES price (0-1).
            news_context: Optional news articles/text for context.
            event_title: Optional event title for context.
            category: Optional market category.
            model: LLM model to use (defaults to provider's default).

        Returns:
            CommitteeResult with aggregated probability and agent details.
        """
        # Build context that all agents share
        context = self._build_context(market_question, market_price, news_context, event_title, category)

        # Run Outside View and Inside View in parallel
        outside_task = asyncio.create_task(self._run_outside_view(context, model))
        inside_task = asyncio.create_task(self._run_inside_view(context, model))

        outside_estimate, inside_estimate = await asyncio.gather(outside_task, inside_task)

        # Run Critic with both estimates as input
        critic_estimate = await self._run_critic(context, outside_estimate, inside_estimate, model)

        # Aggregate
        all_estimates = [outside_estimate, inside_estimate, critic_estimate]
        final_prob, final_confidence = self._aggregate(all_estimates)

        # Compute edge
        edge = abs(final_prob - market_price) * 100
        direction = "buy_yes" if final_prob > market_price else "buy_no"

        return CommitteeResult(
            market_question=market_question,
            market_price=market_price,
            final_probability=final_prob,
            edge_percent=edge,
            direction=direction,
            confidence=final_confidence,
            agent_estimates=all_estimates,
            news_context=news_context,
        )

    # ------------------------------------------------------------------
    # Agent runners
    # ------------------------------------------------------------------

    async def _run_outside_view(self, context: str, model: Optional[str]) -> AgentEstimate:
        """Run the Outside View (base rate) agent."""
        return await self._run_agent(
            agent_name="Outside View",
            system_prompt=_OUTSIDE_VIEW_SYSTEM,
            user_prompt=context,
            schema=_ESTIMATE_SCHEMA,
            model=model,
        )

    async def _run_inside_view(self, context: str, model: Optional[str]) -> AgentEstimate:
        """Run the Inside View (causal reasoning) agent."""
        return await self._run_agent(
            agent_name="Inside View",
            system_prompt=_INSIDE_VIEW_SYSTEM,
            user_prompt=context,
            schema=_ESTIMATE_SCHEMA,
            model=model,
        )

    async def _run_critic(
        self,
        context: str,
        outside: AgentEstimate,
        inside: AgentEstimate,
        model: Optional[str],
    ) -> AgentEstimate:
        """Run the Adversarial Critic agent."""
        critic_context = (
            f"{context}\n\n"
            f"--- OUTSIDE VIEW AGENT ---\n"
            f"Probability: {outside.probability:.2f}\n"
            f"Confidence: {outside.confidence:.2f}\n"
            f"Reasoning: {outside.reasoning}\n\n"
            f"--- INSIDE VIEW AGENT ---\n"
            f"Probability: {inside.probability:.2f}\n"
            f"Confidence: {inside.confidence:.2f}\n"
            f"Reasoning: {inside.reasoning}\n\n"
            f"Identify biases, check evidence quality, and provide your "
            f"adjusted probability."
        )

        result = await self._run_agent(
            agent_name="Adversarial Critic",
            system_prompt=_CRITIC_SYSTEM,
            user_prompt=critic_context,
            schema=_CRITIC_SCHEMA,
            model=model,
            prob_key="adjusted_probability_yes",
        )

        return result

    async def _run_agent(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        model: Optional[str],
        prob_key: str = "probability_yes",
    ) -> AgentEstimate:
        """Run a single agent and return its estimate."""
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                raise RuntimeError("No LLM provider available")

            result = await manager.structured_output(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                schema=schema,
                model=model,
                purpose=f"forecaster_{agent_name.lower().replace(' ', '_')}",
            )

            prob = float(result.get(prob_key, 0.5))
            prob = max(0.01, min(0.99, prob))

            return AgentEstimate(
                agent_name=agent_name,
                probability=prob,
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                model_used=model or "default",
            )

        except Exception as e:
            logger.error("Agent '%s' failed: %s", agent_name, e)
            return AgentEstimate(
                agent_name=agent_name,
                probability=0.5,
                confidence=0.0,
                reasoning=f"Agent failed: {e}",
                model_used=model or "default",
            )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, estimates: list[AgentEstimate]) -> tuple[float, float]:
        """Aggregate multiple agent estimates into a final probability.

        Uses weighted median (weighted by confidence) with extremization.

        Returns (final_probability, confidence).
        """
        # Filter out failed agents (confidence=0)
        valid = [e for e in estimates if e.confidence > 0]
        if not valid:
            return 0.5, 0.0

        if len(valid) == 1:
            return valid[0].probability, valid[0].confidence

        # Weighted average (simpler and more robust than weighted median for 3 agents)
        total_weight = sum(e.confidence for e in valid)
        if total_weight == 0:
            return 0.5, 0.0

        weighted_prob = sum(e.probability * e.confidence for e in valid) / total_weight
        avg_confidence = total_weight / len(valid)

        # Extremization: when independent agents agree on a direction,
        # push the probability further that way.
        # This captures the "information diversity premium" (Ungar et al., 2012).
        # d > 1 extremizes, d = 1 is no change, d < 1 shrinks toward 0.5
        agreement = self._compute_agreement(valid)
        d = 1.0 + 0.5 * agreement  # d ranges from 1.0 (no agreement) to 1.5 (full)

        extremized = self._extremize(weighted_prob, d)

        return extremized, avg_confidence

    @staticmethod
    def _compute_agreement(estimates: list[AgentEstimate]) -> float:
        """Compute agreement score (0-1) among agents.

        1.0 means all agents agree on direction and magnitude.
        0.0 means agents disagree completely.
        """
        if len(estimates) < 2:
            return 0.0

        probs = [e.probability for e in estimates]
        mean = sum(probs) / len(probs)

        # Check directional agreement: do all agents agree on >0.5 or <0.5?
        above = sum(1 for p in probs if p > 0.5)
        below = sum(1 for p in probs if p < 0.5)
        directional = max(above, below) / len(probs)

        # Check magnitude agreement: how close are the estimates?
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        magnitude = max(0, 1.0 - variance * 10)  # Penalize high variance

        return directional * magnitude

    @staticmethod
    def _extremize(prob: float, d: float) -> float:
        """Apply extremization transform.

        Pushes probability away from 0.5 by factor d.
        Uses the log-odds extremization: logit(p_new) = d * logit(p).
        """
        if prob <= 0.01 or prob >= 0.99:
            return prob

        # logit transform
        logit_p = math.log(prob / (1.0 - prob))
        # extremize
        logit_new = d * logit_p
        # inverse logit
        extremized = 1.0 / (1.0 + math.exp(-logit_new))
        # Clamp
        return max(0.01, min(0.99, extremized))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        question: str,
        market_price: float,
        news_context: str,
        event_title: str,
        category: str,
    ) -> str:
        """Build the shared context string for all agents."""
        parts = [
            f"MARKET QUESTION: {question}",
            f"CURRENT MARKET PRICE (YES): ${market_price:.2f} ({market_price:.0%})",
        ]
        if event_title:
            parts.append(f"EVENT: {event_title}")
        if category:
            parts.append(f"CATEGORY: {category}")

        parts.append("")

        if news_context:
            parts.append("RELEVANT NEWS:")
            parts.append(news_context)
            parts.append("")

        parts.append("Based on the above, estimate the probability that this market resolves YES.")

        return "\n".join(parts)


# ======================================================================
# Singleton
# ======================================================================

forecaster_committee = ForecasterCommittee()
