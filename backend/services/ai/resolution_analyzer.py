"""
Resolution Criteria Analyzer.

Analyzes Polymarket resolution criteria using LLM intelligence to:
1. Identify ambiguities in resolution rules
2. Flag potential edge cases
3. Assess resolution source reliability
4. Provide safety recommendations (safe/caution/avoid)

This is the highest-impact AI feature -- it prevents the most
dangerous kind of loss: betting on a market that resolves
unexpectedly due to ambiguous criteria.

Inspired by virattt/dexter's autonomous research agent pattern.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from typing import TYPE_CHECKING

from sqlalchemy import select, desc

from models.database import (
    AsyncSessionLocal,
    ResolutionAnalysis,
    ResearchSession,
)
from services.ai import get_llm_manager
from services.ai.scratchpad import ScratchpadService

if TYPE_CHECKING:
    from models.opportunity import ArbitrageOpportunity

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

_CACHE_TTL_HOURS = 24

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RESOLUTION_ANALYSIS_SYSTEM_PROMPT = """\
You are a Polymarket Resolution Criteria Analyst. Your job is to evaluate \
prediction market resolution rules for clarity, safety, and risk of \
unexpected outcomes. You protect traders from the most dangerous kind of \
loss: betting on a market whose resolution criteria are ambiguous, \
unreliable, or prone to edge-case disputes.

## Polymarket Background

Polymarket is a prediction market platform on Polygon where users buy \
YES and NO outcome tokens priced between $0 and $1. When a market \
resolves, winning tokens pay out $1 each and losing tokens pay $0. \
Polymarket charges a 2% fee on winning positions. Markets are resolved \
by UMA's Optimistic Oracle, which relies on the resolution source and \
rules specified in the market description.

## Your Analysis Framework

For every market, systematically evaluate:

### 1. Resolution Source Reliability
- Is the resolution source clearly identified (e.g., official government data, \
  specific news outlet, a named organization)?
- Could the source change methodology mid-market (e.g., a poll aggregator \
  revising its model)?
- Is there a single authoritative source, or could different valid sources \
  give conflicting answers?
- Is the source freely accessible, or is it behind paywalls or subject to \
  delays that complicate verification?

### 2. Language Precision
- Are there undefined or vague terms (e.g., "significant", "major", \
  "approximately", "around")?
- Is the scope well-defined (what counts and what doesn't)?
- Are thresholds explicit (exact numbers vs. qualitative judgments)?
- Could reasonable people disagree on whether criteria are met?

### 3. Temporal Issues
- Is the resolution deadline clearly stated with timezone?
- Are there race conditions (e.g., an event could happen right at the \
  deadline)?
- Could the relevant event be delayed, postponed, or rescheduled?
- Is there ambiguity about when exactly a threshold is "met" vs. \
  "announced"?

### 4. Edge Cases
Consider specific real-world scenarios that could cause disputes:
- Partial fulfillment (e.g., "Will X happen?" but X partially happens)
- Technicalities vs. spirit (e.g., a bill is "passed" in one chamber but \
  not the other)
- Data revisions (initial reports revised later)
- Cancellations, postponements, or force majeure events
- Definitional boundaries (what counts as a "recession", "war", "deal"?)
- Split outcomes or ties where rules don't specify a tiebreaker

### 5. NegRisk and Multi-Outcome Markets
- In NegRisk markets (multiple mutually exclusive outcomes under one event), \
  check if the outcomes are truly exhaustive. A missing outcome means all \
  NO tokens could pay out.
- Check if the ordering of resolution matters (first to reach threshold? \
  final count?).
- Look for correlated resolution risks: if one outcome resolves in an \
  unexpected way, it cascades to others.

## Scoring Guidelines

**clarity_score** (0-1):
- 0.9-1.0: Crystal clear, no ambiguity, single authoritative source, \
  explicit thresholds and deadlines.
- 0.7-0.89: Mostly clear with minor ambiguities that are unlikely to \
  matter in practice.
- 0.5-0.69: Moderate ambiguity; some terms are vague or the resolution \
  source has known issues.
- 0.3-0.49: Significant ambiguity; multiple reasonable interpretations exist.
- 0.0-0.29: Highly ambiguous; resolution could go either way based on \
  interpretation alone.

**risk_score** (0-1):
- 0.0-0.1: Virtually no risk of unexpected resolution.
- 0.1-0.3: Low risk; minor edge cases exist but are unlikely.
- 0.3-0.5: Moderate risk; plausible scenarios could cause disputes.
- 0.5-0.7: High risk; realistic edge cases are concerning.
- 0.7-1.0: Very high risk; resolution is actively contentious or the \
  criteria are fundamentally flawed.

**recommendation**:
- "safe": clarity >= 0.7 AND risk <= 0.3 -- proceed with normal trading.
- "caution": clarity >= 0.5 AND risk <= 0.5 -- trade with reduced size and \
  extra monitoring.
- "avoid": clarity < 0.5 OR risk > 0.5 -- do not trade; the resolution \
  risk outweighs potential profit.

**confidence**: How confident you are in your own analysis (0-1). Lower \
confidence when you lack domain expertise on the topic, when the market \
question is about niche subject matter, or when the resolution source is \
one you cannot verify.

**resolution_likelihood**: Estimate the probability of each outcome based \
purely on the resolution criteria analysis (not a prediction of the event \
itself). This represents how likely each outcome is to be the "cleanly \
resolved" one given the criteria. If the criteria are clear and symmetric, \
this may simply mirror current market prices. If there is asymmetric \
ambiguity (e.g., one side benefits from vague language), flag that.

Respond with valid JSON matching the required schema. Be specific in your \
ambiguities and edge_cases -- cite the exact wording that concerns you.\
"""

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

RESOLUTION_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "clarity_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "How clear and unambiguous the resolution criteria are (0=opaque, 1=crystal clear)",
        },
        "risk_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Risk of unexpected or disputed resolution (0=no risk, 1=near certain dispute)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in this analysis (0=guessing, 1=expert certainty)",
        },
        "recommendation": {
            "type": "string",
            "enum": ["safe", "caution", "avoid"],
            "description": "Trading recommendation based on resolution safety",
        },
        "ambiguities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific ambiguities identified in the resolution criteria, citing exact wording where possible",
        },
        "edge_cases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete real-world scenarios that could cause unexpected or disputed resolution",
        },
        "key_dates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important dates and deadlines relevant to resolution (with timezone notes)",
        },
        "resolution_likelihood": {
            "type": "object",
            "description": "Estimated probability for each outcome based on criteria clarity analysis",
        },
        "summary": {
            "type": "string",
            "description": "One paragraph human-readable summary of the analysis and key concerns",
        },
    },
    "required": [
        "clarity_score",
        "risk_score",
        "confidence",
        "recommendation",
        "ambiguities",
        "edge_cases",
        "summary",
    ],
}


# ---------------------------------------------------------------------------
# ResolutionAnalyzer
# ---------------------------------------------------------------------------


class ResolutionAnalyzer:
    """Analyzes market resolution criteria for safety and clarity.

    This is the highest-impact AI feature. Before betting on any market,
    it evaluates the resolution rules to assess whether the market could
    resolve unexpectedly due to ambiguous criteria, unreliable sources,
    temporal edge cases, or definitional disputes.

    Results are cached for 24 hours per market_id to avoid redundant
    LLM calls.
    """

    def __init__(self) -> None:
        self._scratchpad = ScratchpadService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_market(
        self,
        market_id: str,
        question: str,
        description: str = "",
        resolution_source: str = "",
        end_date: str = "",
        outcomes: list[str] | None = None,
        force_refresh: bool = False,
        model: str | None = None,
    ) -> dict:
        """Analyze a market's resolution criteria.

        Parameters
        ----------
        market_id:
            The Polymarket market (condition) ID.
        question:
            The market question (e.g., "Will X happen by Y?").
        description:
            Full market description including resolution rules.
        resolution_source:
            The stated resolution source (e.g., "Associated Press").
        end_date:
            Market end / resolution date as a string.
        outcomes:
            List of possible outcomes (e.g., ["Yes", "No"]).
        force_refresh:
            If True, bypass cache and re-analyze.
        model:
            LLM model override. Defaults to the configured default model.

        Returns
        -------
        dict with clarity_score, risk_score, confidence, recommendation,
        ambiguities, edge_cases, key_dates, resolution_likelihood,
        summary, market_id, and session_id.
        """
        # 1. Check cache
        if not force_refresh:
            cached = await self.get_cached_analysis(market_id)
            if cached is not None:
                logger.info(
                    "Returning cached resolution analysis",
                    market_id=market_id,
                )
                return cached

        session_id = uuid.uuid4().hex[:16]
        started_at = utcnow()

        try:
            # 2. Build the user prompt with all market details
            user_prompt = self._build_analysis_prompt(
                market_id=market_id,
                question=question,
                description=description,
                resolution_source=resolution_source,
                end_date=end_date,
                outcomes=outcomes or ["Yes", "No"],
            )

            # 3. Call LLM with structured output
            manager = get_llm_manager()
            from services.ai.llm_provider import LLMMessage

            messages = [
                LLMMessage(role="system", content=RESOLUTION_ANALYSIS_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ]

            llm_response = await manager.structured_output(
                messages=messages,
                schema=RESOLUTION_ANALYSIS_SCHEMA,
                model=model,
                purpose="resolution_analysis",
            )

            # structured_output() returns a dict directly
            if llm_response:
                analysis = llm_response
            else:
                analysis = self._fallback_analysis("LLM returned empty response")

            # Ensure all required fields exist with defaults
            analysis = self._validate_analysis(analysis)

            # 4. Build the result dict
            result = {
                "market_id": market_id,
                "clarity_score": analysis["clarity_score"],
                "risk_score": analysis["risk_score"],
                "confidence": analysis["confidence"],
                "recommendation": analysis["recommendation"],
                "ambiguities": analysis.get("ambiguities", []),
                "edge_cases": analysis.get("edge_cases", []),
                "key_dates": analysis.get("key_dates", []),
                "resolution_likelihood": analysis.get("resolution_likelihood", {}),
                "summary": analysis["summary"],
                "session_id": session_id,
                "model_used": model,
                "analyzed_at": started_at.isoformat(),
            }

            # 5. Persist to database
            await self._store_analysis(
                result=result,
                question=question,
                description=description,
                resolution_source=resolution_source,
                session_id=session_id,
                model_used=model,
                started_at=started_at,
            )

            logger.info(
                "Resolution analysis complete",
                market_id=market_id,
                clarity=result["clarity_score"],
                risk=result["risk_score"],
                recommendation=result["recommendation"],
            )
            return result

        except Exception as exc:
            logger.error(
                "Resolution analysis failed",
                market_id=market_id,
                error=str(exc),
            )
            # Record the failed session
            await self._record_failed_session(
                session_id=session_id,
                market_id=market_id,
                error=str(exc),
                started_at=started_at,
            )
            # Return a conservative fallback so callers always get a result
            fallback = self._fallback_analysis(str(exc))
            fallback["market_id"] = market_id
            fallback["session_id"] = session_id
            return fallback

    async def analyze_opportunity_markets(
        self,
        opportunity: "ArbitrageOpportunity",
        model: str | None = None,
    ) -> dict:
        """Analyze all markets in an arbitrage opportunity.

        Fetches market details for each market in the opportunity and
        runs resolution analysis on each. The aggregate result is driven
        by the *worst* individual market -- a chain is only as strong as
        its weakest link.

        Parameters
        ----------
        opportunity:
            The arbitrage opportunity containing market references.
        model:
            LLM model override.

        Returns
        -------
        dict with per-market analyses and an aggregate assessment.
        """
        markets = opportunity.markets or []
        if not markets:
            return {
                "opportunity_id": opportunity.id,
                "overall_recommendation": "caution",
                "overall_clarity": 0.5,
                "overall_risk": 0.5,
                "market_analyses": [],
                "summary": "No market details available for analysis.",
            }

        # Analyze each market concurrently
        tasks = []
        for mkt in markets:
            market_id = mkt.get("id") or mkt.get("condition_id", "")
            question = mkt.get("question", "")
            description = mkt.get("description", "")
            resolution_source = mkt.get("resolution_source", "")
            end_date = mkt.get("end_date", "")
            outcomes = mkt.get("outcomes", ["Yes", "No"])

            tasks.append(
                self.analyze_market(
                    market_id=market_id,
                    question=question,
                    description=description,
                    resolution_source=resolution_source,
                    end_date=end_date,
                    outcomes=outcomes,
                    model=model,
                )
            )

        analyses = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful analyses
        market_analyses = []
        for i, result in enumerate(analyses):
            if isinstance(result, Exception):
                logger.error(
                    "Market analysis failed in opportunity batch",
                    market_index=i,
                    error=str(result),
                )
                market_analyses.append(self._fallback_analysis(f"Analysis failed: {result}"))
            else:
                market_analyses.append(result)

        # Aggregate: worst market drives overall assessment
        clarity_scores = [a.get("clarity_score", 0.5) for a in market_analyses]
        risk_scores = [a.get("risk_score", 0.5) for a in market_analyses]

        overall_clarity = min(clarity_scores) if clarity_scores else 0.5
        overall_risk = max(risk_scores) if risk_scores else 0.5

        # Determine aggregate recommendation
        if overall_clarity >= 0.7 and overall_risk <= 0.3:
            overall_recommendation = "safe"
        elif overall_clarity >= 0.5 and overall_risk <= 0.5:
            overall_recommendation = "caution"
        else:
            overall_recommendation = "avoid"

        # Collect all risk flags
        all_ambiguities = []
        all_edge_cases = []
        for a in market_analyses:
            all_ambiguities.extend(a.get("ambiguities", []))
            all_edge_cases.extend(a.get("edge_cases", []))

        return {
            "opportunity_id": opportunity.id,
            "overall_recommendation": overall_recommendation,
            "overall_clarity": overall_clarity,
            "overall_risk": overall_risk,
            "market_analyses": market_analyses,
            "all_ambiguities": all_ambiguities,
            "all_edge_cases": all_edge_cases,
            "summary": (
                f"Analyzed {len(market_analyses)} market(s). "
                f"Overall clarity: {overall_clarity:.2f}, "
                f"overall risk: {overall_risk:.2f}. "
                f"Recommendation: {overall_recommendation}."
            ),
        }

    async def get_cached_analysis(self, market_id: str) -> dict | None:
        """Get cached analysis for a market if it exists and is not expired.

        Parameters
        ----------
        market_id:
            The market (condition) ID to look up.

        Returns
        -------
        The cached analysis dict, or None if no valid cache entry exists.
        """
        try:
            cutoff = utcnow() - timedelta(hours=_CACHE_TTL_HOURS)
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ResolutionAnalysis)
                    .where(
                        ResolutionAnalysis.market_id == market_id,
                        ResolutionAnalysis.analyzed_at >= cutoff,
                    )
                    .order_by(desc(ResolutionAnalysis.analyzed_at))
                    .limit(1)
                )
                row = result.scalar_one_or_none()

                if row is None:
                    return None

                return {
                    "market_id": row.market_id,
                    "clarity_score": row.clarity_score,
                    "risk_score": row.risk_score,
                    "confidence": row.confidence,
                    "recommendation": row.recommendation,
                    "ambiguities": row.ambiguities or [],
                    "edge_cases": row.edge_cases or [],
                    "key_dates": row.key_dates or [],
                    "resolution_likelihood": row.resolution_likelihood or {},
                    "summary": row.summary or "",
                    "session_id": row.session_id,
                    "model_used": row.model_used,
                    "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
                    "cached": True,
                }
        except Exception as exc:
            logger.error(
                "Failed to fetch cached analysis",
                market_id=market_id,
                error=str(exc),
            )
            return None

    async def get_analysis_history(
        self,
        market_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get recent resolution analyses.

        Parameters
        ----------
        market_id:
            If provided, filter to a specific market.
        limit:
            Maximum number of records to return.

        Returns
        -------
        List of analysis dicts, newest first.
        """
        try:
            async with AsyncSessionLocal() as session:
                query = select(ResolutionAnalysis).order_by(desc(ResolutionAnalysis.analyzed_at))
                if market_id:
                    query = query.where(ResolutionAnalysis.market_id == market_id)
                query = query.limit(limit)

                result = await session.execute(query)
                rows = result.scalars().all()

                return [
                    {
                        "id": row.id,
                        "market_id": row.market_id,
                        "question": row.question,
                        "clarity_score": row.clarity_score,
                        "risk_score": row.risk_score,
                        "confidence": row.confidence,
                        "recommendation": row.recommendation,
                        "ambiguities": row.ambiguities or [],
                        "edge_cases": row.edge_cases or [],
                        "key_dates": row.key_dates or [],
                        "resolution_likelihood": row.resolution_likelihood or {},
                        "summary": row.summary or "",
                        "model_used": row.model_used,
                        "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Failed to fetch analysis history", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_analysis_prompt(
        self,
        market_id: str,
        question: str,
        description: str,
        resolution_source: str,
        end_date: str,
        outcomes: list[str],
    ) -> str:
        """Build the user prompt for the LLM with all available market details."""
        sections = [
            "## Market to Analyze\n",
            f"**Market ID:** {market_id}",
            f"**Question:** {question}",
        ]

        if description:
            sections.append(f"\n**Full Description / Resolution Rules:**\n{description}")
        else:
            sections.append(
                "\n**Description:** Not provided. This itself is a risk factor "
                "-- markets without explicit resolution rules are inherently ambiguous."
            )

        if resolution_source:
            sections.append(f"\n**Resolution Source:** {resolution_source}")
        else:
            sections.append("\n**Resolution Source:** Not specified. Flag this as an ambiguity.")

        if end_date:
            sections.append(f"**End Date:** {end_date}")

        sections.append(f"**Possible Outcomes:** {', '.join(outcomes)}")

        sections.append(
            "\n---\n"
            "Analyze the above market's resolution criteria. Be specific about "
            "any ambiguities -- cite the exact wording. Consider real-world "
            "scenarios that could cause disputes. Provide your structured analysis."
        )

        return "\n".join(sections)

    def _validate_analysis(self, analysis: dict) -> dict:
        """Ensure all required fields exist with sensible defaults."""
        defaults = {
            "clarity_score": 0.5,
            "risk_score": 0.5,
            "confidence": 0.5,
            "recommendation": "caution",
            "ambiguities": [],
            "edge_cases": [],
            "key_dates": [],
            "resolution_likelihood": {},
            "summary": "Analysis completed but some fields were missing.",
        }
        for key, default in defaults.items():
            if key not in analysis or analysis[key] is None:
                analysis[key] = default

        # Clamp numeric scores to [0, 1]
        for score_key in ("clarity_score", "risk_score", "confidence"):
            val = analysis[score_key]
            if isinstance(val, (int, float)):
                analysis[score_key] = max(0.0, min(1.0, float(val)))
            else:
                analysis[score_key] = defaults[score_key]

        # Validate recommendation enum
        if analysis["recommendation"] not in ("safe", "caution", "avoid"):
            analysis["recommendation"] = "caution"

        return analysis

    def _fallback_analysis(self, reason: str) -> dict:
        """Return a conservative fallback analysis when the LLM call fails."""
        return {
            "clarity_score": 0.3,
            "risk_score": 0.7,
            "confidence": 0.1,
            "recommendation": "avoid",
            "ambiguities": [f"Analysis could not be completed: {reason}"],
            "edge_cases": [],
            "key_dates": [],
            "resolution_likelihood": {},
            "summary": (
                f"Automated analysis was unable to complete ({reason}). "
                "Defaulting to 'avoid' recommendation as a safety measure."
            ),
        }

    async def _store_analysis(
        self,
        result: dict,
        question: str,
        description: str,
        resolution_source: str,
        session_id: str,
        model_used: str,
        started_at: datetime,
    ) -> None:
        """Persist the analysis result to the database."""
        try:
            completed_at = utcnow()
            duration = (completed_at - started_at).total_seconds()

            async with AsyncSessionLocal() as session:
                # Store the research session record
                research_session = ResearchSession(
                    id=session_id,
                    session_type="resolution_analysis",
                    query=question,
                    market_id=result["market_id"],
                    status="completed",
                    result=result,
                    iterations=1,
                    tools_called=0,
                    model_used=model_used,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                )
                session.add(research_session)

                # Store the resolution analysis record
                analysis_row = ResolutionAnalysis(
                    id=uuid.uuid4().hex[:16],
                    market_id=result["market_id"],
                    question=question,
                    resolution_source=resolution_source or None,
                    resolution_rules=description or None,
                    clarity_score=result["clarity_score"],
                    risk_score=result["risk_score"],
                    confidence=result["confidence"],
                    ambiguities=result.get("ambiguities"),
                    edge_cases=result.get("edge_cases"),
                    key_dates=result.get("key_dates"),
                    resolution_likelihood=result.get("resolution_likelihood"),
                    summary=result.get("summary"),
                    recommendation=result["recommendation"],
                    session_id=session_id,
                    model_used=model_used,
                    analyzed_at=started_at,
                    expires_at=started_at + timedelta(hours=_CACHE_TTL_HOURS),
                )
                session.add(analysis_row)
                await session.commit()

            logger.info(
                "Resolution analysis stored",
                market_id=result["market_id"],
                session_id=session_id,
                duration_s=round(duration, 2),
            )
        except Exception as exc:
            logger.error(
                "Failed to store resolution analysis",
                market_id=result.get("market_id"),
                error=str(exc),
            )

    async def _record_failed_session(
        self,
        session_id: str,
        market_id: str,
        error: str,
        started_at: datetime,
    ) -> None:
        """Record a failed research session for observability."""
        try:
            completed_at = utcnow()
            duration = (completed_at - started_at).total_seconds()

            async with AsyncSessionLocal() as session:
                research_session = ResearchSession(
                    id=session_id,
                    session_type="resolution_analysis",
                    query=f"Resolution analysis for market {market_id}",
                    market_id=market_id,
                    status="failed",
                    error=error,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                )
                session.add(research_session)
                await session.commit()
        except Exception as exc:
            # Logging failures must not propagate
            logger.error(
                "Failed to record failed session",
                session_id=session_id,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

resolution_analyzer = ResolutionAnalyzer()
