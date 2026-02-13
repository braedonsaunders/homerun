"""
LLM-as-Judge Opportunity Scorer.

Evaluates arbitrage opportunities using LLM intelligence,
providing a second opinion alongside the ML classifier.

Scores each opportunity on multiple dimensions:
- Profit viability: Will the theoretical profit actually materialize?
- Resolution safety: Will markets resolve as expected?
- Execution feasibility: Can we execute at the quoted prices?
- Market efficiency: Is this a real inefficiency or just noise/fees?

Inspired by virattt/dexter's LLM-as-judge evaluation pattern.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from utils.utcnow import utcnow
from typing import TYPE_CHECKING

from sqlalchemy import select, desc, func

from models.database import (
    AsyncSessionLocal,
    OpportunityJudgment,
    ResearchSession,
)
from services.ai import get_llm_manager
from services.ai.scratchpad import ScratchpadService

if TYPE_CHECKING:
    from models.opportunity import ArbitrageOpportunity

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

OPPORTUNITY_JUDGE_SYSTEM_PROMPT = """\
You are an expert Polymarket Arbitrage Judge. Your job is to evaluate \
whether a detected arbitrage opportunity is a genuine, executable, \
profitable trade -- or a false positive that will lose money. You \
provide a dispassionate second opinion alongside a machine learning \
classifier.

## Polymarket Mechanics You Must Consider

### Fee Structure
- Polymarket charges a **2% fee on net winnings** (not on the cost). \
  If you buy YES at $0.60 and it resolves to $1.00, you pay 2% of the \
  $0.40 profit = $0.008 per share.
- This fee is already factored into the opportunity's `net_profit` field, \
  but verify it looks correct.

### CLOB Order Book
- Polymarket uses a Central Limit Order Book (CLOB). Quoted prices are \
  the best bid/ask at the moment of detection.
- **Slippage is the #1 killer of theoretical profits.** A 5% ROI on paper \
  can vanish if the order book is thin and you move the price by even 1-2 \
  cents executing.
- Check `min_liquidity` and `max_position_size` -- if liquidity is low \
  relative to the required trade size, the opportunity is likely unexecutable.

### NegRisk Bundles
- NegRisk markets allow buying a complete set of NO tokens for $1. \
  Arbitrage in NegRisk works by buying NO tokens whose prices sum to \
  less than $1, guaranteeing profit regardless of outcome.
- The most reliable arbitrage type, but profits are often tiny (1-3%) \
  and compete with bots that execute in milliseconds.

### Resolution Risk
- Markets resolve via UMA's Optimistic Oracle. If a market resolves \
  unexpectedly, your entire position can go to zero.
- Arbitrage strategies that depend on correlated resolution across \
  multiple markets carry higher resolution risk.
- Markets near their end date are riskier because resolution timing \
  matters more.

### Common False Positive Patterns
1. **Stale prices**: Prices haven't updated yet; by the time you \
   execute, the spread is gone.
2. **Fee miscalculation**: ROI looks positive before fees but is \
   negative after the 2% winner fee.
3. **Thin liquidity**: ROI exists on paper but you can only fill \
   10 shares before moving the price.
4. **Correlated markets mislabeled as independent**: Two markets \
   seem to offer arbitrage but are actually about different \
   resolution criteria.
5. **Settlement lag illusion**: A market is detected as mispriced \
   but it's actually already been resolved off-chain and prices \
   are adjusting.
6. **Cross-event confusion**: Markets from different events have \
   similar questions but different resolution rules.

## Scoring Dimensions

### profit_viability (0-1)
Will the stated profit actually materialize?
- 0.9-1.0: Guaranteed profit (NegRisk complete set, prices locked, fees verified)
- 0.7-0.89: Highly likely profit (strong spread, fees accounted for, reasonable assumptions)
- 0.5-0.69: Uncertain (thin margins, some assumptions may not hold)
- 0.3-0.49: Doubtful (margins likely consumed by fees, slippage, or price movement)
- 0.0-0.29: Almost certainly unprofitable after real-world execution costs

### resolution_safety (0-1)
Will all involved markets resolve cleanly and as expected?
- 0.9-1.0: Clear resolution criteria, reliable sources, no edge cases
- 0.7-0.89: Minor concerns but resolution is likely straightforward
- 0.5-0.69: Some ambiguity; resolution could plausibly go wrong
- 0.3-0.49: Significant resolution risk; edge cases are realistic
- 0.0-0.29: Resolution is actively uncertain or criteria are flawed

### execution_feasibility (0-1)
Can we actually execute this trade at the prices shown?
- 0.9-1.0: Deep liquidity, large max position, prices unlikely to move
- 0.7-0.89: Adequate liquidity, can execute most of the size needed
- 0.5-0.69: Liquidity is marginal; may get partial fills or slippage
- 0.3-0.49: Thin book; execution will likely degrade the profit significantly
- 0.0-0.29: Not executable at the quoted prices

### market_efficiency (0-1)
Is this a real market inefficiency, or noise?
- 0.9-1.0: Clear structural mispricing (e.g., NegRisk sum > 1, identical markets priced differently)
- 0.7-0.89: Likely real inefficiency with a plausible explanation for why it persists
- 0.5-0.69: Unclear; could be real or could be noise / stale data
- 0.3-0.49: Probably noise, fee artifact, or already-closing spread
- 0.0-0.29: Almost certainly not a real inefficiency

### overall_score (0-1)
Composite score. Should roughly equal the minimum of the four dimension \
scores, because an opportunity is only as strong as its weakest link. \
An opportunity with 0.95 profit viability but 0.2 execution feasibility \
should score ~0.2-0.3 overall.

### recommendation
- "strong_execute": overall_score >= 0.8 -- high confidence, execute immediately
- "execute": overall_score >= 0.65 -- positive signal, proceed with standard size
- "review": overall_score >= 0.45 -- mixed signals, requires human review
- "skip": overall_score >= 0.25 -- probably not worth it, high risk of loss
- "strong_skip": overall_score < 0.25 -- almost certainly a false positive

## Your Task

Given the opportunity details below, score it on all dimensions. \
Provide a concise rationale for each score. Identify specific risk \
factors. Be calibrated -- most detected opportunities in prediction \
markets ARE false positives due to fees, slippage, or stale data. \
A skeptical prior is appropriate.

Respond with valid JSON matching the required schema.\
"""

WEATHER_OPPORTUNITY_JUDGE_SYSTEM_PROMPT = """\
You are a risk-first weather prediction market trade reviewer.

Score the provided opportunity on:
- profit_viability
- resolution_safety
- execution_feasibility
- market_efficiency
- overall_score

Rules:
1) Use ONLY facts present in the prompt.
2) Do not invent missing data (liquidity, model agreement, sources, or rules).
3) If data is missing, state uncertainty explicitly.
4) Keep reasoning concise and concrete.

Recommendation guidance:
- strong_execute / execute only when executable and edge is credible
- review for mixed evidence
- skip / strong_skip for non-executable, low-confidence, or structurally filtered trades

Respond with valid JSON matching the required schema.
"""

DIRECTIONAL_OPPORTUNITY_JUDGE_SYSTEM_PROMPT = """\
You are a risk-first directional trade judge for prediction markets.

The opportunity may be a statistical edge (not guaranteed arbitrage).
Do NOT reject solely because it is directional; instead evaluate whether
the expected edge is credible, executable, and sized appropriately.

Score:
- profit_viability
- resolution_safety
- execution_feasibility
- market_efficiency
- overall_score

Rules:
1) Use only evidence present in the prompt.
2) Explicitly account for uncertainty and edge fragility (slippage, stale moves, thin books).
3) Distinguish "not guaranteed" from "invalid" — directional can still be tradeable.
4) Penalize extreme/implausible ROI or weak data quality.

Recommendation guidance:
- strong_execute/execute when edge is credible and executable
- review for mixed evidence
- skip/strong_skip only when edge is likely noise, non-executable, or structurally unsafe

Respond with valid JSON matching the required schema.
"""

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

OPPORTUNITY_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "profit_viability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "resolution_safety": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "execution_feasibility": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "market_efficiency": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "recommendation": {
            "type": "string",
            "enum": ["strong_execute", "execute", "review", "skip", "strong_skip"],
        },
        "reasoning": {"type": "string"},
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "overall_score",
        "profit_viability",
        "resolution_safety",
        "execution_feasibility",
        "market_efficiency",
        "recommendation",
        "reasoning",
        "risk_factors",
    ],
}


# ---------------------------------------------------------------------------
# OpportunityJudge
# ---------------------------------------------------------------------------


class OpportunityJudge:
    """LLM-based opportunity scoring and evaluation.

    Provides a second opinion alongside the ML classifier for each
    arbitrage opportunity. Evaluates on four dimensions: profit
    viability, resolution safety, execution feasibility, and market
    efficiency. Results are persisted in the OpportunityJudgment table
    for auditability and calibration analysis.
    """

    def __init__(self) -> None:
        self._scratchpad = ScratchpadService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def judge_opportunity(
        self,
        opportunity: "ArbitrageOpportunity",
        resolution_analysis: dict | None = None,
        ml_prediction: dict | None = None,
        model: str | None = None,
        force_llm: bool = False,
    ) -> dict:
        """Judge an arbitrage opportunity.

        Parameters
        ----------
        opportunity:
            The detected arbitrage opportunity to evaluate.
        resolution_analysis:
            Pre-computed resolution analysis from ResolutionAnalyzer.
            If provided, it enriches the LLM's assessment.
        ml_prediction:
            The ML classifier's prediction dict (probability,
            recommendation, confidence). Used for agreement tracking.
        model:
            LLM model override. Defaults to gpt-4o-mini.
        force_llm:
            When True, bypass report-only heuristic and run full LLM judgment.

        Returns
        -------
        dict with overall_score, profit_viability, resolution_safety,
        execution_feasibility, market_efficiency, recommendation,
        reasoning, risk_factors, ml_agreement, session_id.
        """
        session_id = uuid.uuid4().hex[:16]
        started_at = utcnow()

        try:
            # Non-executable/report-only opportunities do not need LLM calls.
            if self._is_report_only_opportunity(opportunity) and not force_llm:
                judgment = self._report_only_judgment(opportunity)
                judgment = self._annotate_reject_codes(judgment)
                ml_agreement = self._compute_ml_agreement(judgment, ml_prediction)
                result = {
                    "opportunity_id": opportunity.id,
                    "overall_score": judgment["overall_score"],
                    "profit_viability": judgment["profit_viability"],
                    "resolution_safety": judgment["resolution_safety"],
                    "execution_feasibility": judgment["execution_feasibility"],
                    "market_efficiency": judgment["market_efficiency"],
                    "recommendation": judgment["recommendation"],
                    "reasoning": judgment["reasoning"],
                    "risk_factors": judgment.get("risk_factors", []),
                    "reject_codes": judgment.get("reject_codes", []),
                    "ml_agreement": ml_agreement,
                    "session_id": session_id,
                    "model_used": "heuristic_report_only",
                    "judged_at": started_at.isoformat(),
                }
                await self._store_judgment(
                    result=result,
                    opportunity=opportunity,
                    ml_prediction=ml_prediction,
                    session_id=session_id,
                    model_used="heuristic_report_only",
                    started_at=started_at,
                )
                return result

            # 1. Build context prompt with opportunity details
            if self._is_weather_opportunity(opportunity):
                user_prompt = self._build_weather_judgment_prompt(
                    opportunity=opportunity,
                    resolution_analysis=resolution_analysis,
                    ml_prediction=ml_prediction,
                )
            else:
                user_prompt = self._build_judgment_prompt(
                    opportunity=opportunity,
                    resolution_analysis=resolution_analysis,
                    ml_prediction=ml_prediction,
                )

            # 2. Call LLM with structured output
            manager = get_llm_manager()
            resolved_model = model or manager._default_model
            from services.ai.llm_provider import LLMMessage

            if self._is_weather_opportunity(opportunity):
                system_prompt = WEATHER_OPPORTUNITY_JUDGE_SYSTEM_PROMPT
            elif self._is_directional_opportunity(opportunity):
                system_prompt = DIRECTIONAL_OPPORTUNITY_JUDGE_SYSTEM_PROMPT
            else:
                system_prompt = OPPORTUNITY_JUDGE_SYSTEM_PROMPT
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]

            llm_response = await manager.structured_output(
                messages=messages,
                schema=OPPORTUNITY_JUDGMENT_SCHEMA,
                model=resolved_model,
                purpose="opportunity_judge",
            )

            # structured_output() returns a dict directly, but some
            # providers may return a list or other non-dict JSON.
            if llm_response and isinstance(llm_response, dict):
                judgment = llm_response
            elif llm_response and isinstance(llm_response, list):
                # Some LLM providers wrap structured output in a list.
                # Try to extract the first dict element before falling back.
                extracted = next(
                    (item for item in llm_response if isinstance(item, dict)),
                    None,
                )
                if extracted:
                    logger.warning(
                        "LLM returned list instead of dict, extracted first dict element",
                        opportunity_id=opportunity.id,
                    )
                    judgment = extracted
                else:
                    judgment = self._fallback_judgment("LLM returned list with no dict elements")
            elif llm_response:
                judgment = self._fallback_judgment(f"LLM returned {type(llm_response).__name__} instead of dict")
            else:
                judgment = self._fallback_judgment("LLM returned empty response")

            # Validate and fill defaults
            judgment = self._validate_judgment(judgment)
            judgment = self._annotate_reject_codes(judgment)

            # 3. Compare with ML prediction
            ml_agreement = self._compute_ml_agreement(judgment, ml_prediction)

            # 4. Build result
            result = {
                "opportunity_id": opportunity.id,
                "overall_score": judgment["overall_score"],
                "profit_viability": judgment["profit_viability"],
                "resolution_safety": judgment["resolution_safety"],
                "execution_feasibility": judgment["execution_feasibility"],
                "market_efficiency": judgment["market_efficiency"],
                "recommendation": judgment["recommendation"],
                "reasoning": judgment["reasoning"],
                "risk_factors": judgment.get("risk_factors", []),
                "reject_codes": judgment.get("reject_codes", []),
                "ml_agreement": ml_agreement,
                "session_id": session_id,
                "model_used": resolved_model,
                "judged_at": started_at.isoformat(),
            }

            # 5. Persist to database
            await self._store_judgment(
                result=result,
                opportunity=opportunity,
                ml_prediction=ml_prediction,
                session_id=session_id,
                model_used=resolved_model,
                started_at=started_at,
            )

            logger.info(
                "Opportunity judgment complete",
                opportunity_id=opportunity.id,
                overall_score=result["overall_score"],
                recommendation=result["recommendation"],
                ml_agreement=ml_agreement,
            )
            return result

        except Exception as exc:
            logger.error(
                "Opportunity judgment failed",
                opportunity_id=opportunity.id,
                error=str(exc),
            )
            await self._record_failed_session(
                session_id=session_id,
                opportunity_id=opportunity.id,
                error=str(exc),
                started_at=started_at,
            )
            fallback = self._fallback_judgment(str(exc))
            fallback = self._annotate_reject_codes(fallback)
            fallback["opportunity_id"] = opportunity.id
            fallback["session_id"] = session_id
            fallback["ml_agreement"] = None
            return fallback

    async def judge_batch(
        self,
        opportunities: list["ArbitrageOpportunity"],
        model: str | None = None,
    ) -> list[dict]:
        """Judge multiple opportunities concurrently.

        Uses a cheaper model for batch processing to manage costs.
        Opportunities are evaluated in parallel with asyncio.gather.

        Parameters
        ----------
        opportunities:
            List of arbitrage opportunities to evaluate.
        model:
            LLM model override. Defaults to gpt-4o-mini for cost
            efficiency in batch mode.

        Returns
        -------
        List of judgment dicts in the same order as the input.
        """
        batch_model = model

        tasks = [
            self.judge_opportunity(
                opportunity=opp,
                model=batch_model,
            )
            for opp in opportunities
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        judgments = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch judgment failed for opportunity",
                    opportunity_index=i,
                    opportunity_id=opportunities[i].id if i < len(opportunities) else "unknown",
                    error=str(result),
                )
                fallback = self._fallback_judgment(f"Batch judgment failed: {result}")
                fallback["opportunity_id"] = opportunities[i].id if i < len(opportunities) else "unknown"
                judgments.append(fallback)
            else:
                judgments.append(result)

        logger.info(
            "Batch judgment complete",
            total=len(opportunities),
            successful=sum(1 for r in results if not isinstance(r, Exception)),
        )
        return judgments

    async def get_judgment_history(
        self,
        opportunity_id: str | None = None,
        strategy_type: str | None = None,
        min_score: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get historical judgments with optional filtering.

        Parameters
        ----------
        opportunity_id:
            Filter to a specific opportunity.
        strategy_type:
            Filter by strategy type (e.g., "negrisk", "basic").
        min_score:
            Only return judgments with overall_score >= this value.
        limit:
            Maximum number of records to return.

        Returns
        -------
        List of judgment dicts, newest first.
        """
        try:
            async with AsyncSessionLocal() as session:
                query = select(OpportunityJudgment).order_by(desc(OpportunityJudgment.judged_at))

                if opportunity_id:
                    query = query.where(OpportunityJudgment.opportunity_id == opportunity_id)
                if strategy_type:
                    query = query.where(OpportunityJudgment.strategy_type == strategy_type)
                if min_score is not None:
                    query = query.where(OpportunityJudgment.overall_score >= min_score)

                query = query.limit(limit)
                result = await session.execute(query)
                rows = result.scalars().all()

                return [
                    {
                        "id": row.id,
                        "opportunity_id": row.opportunity_id,
                        "strategy_type": row.strategy_type,
                        "overall_score": row.overall_score,
                        "profit_viability": row.profit_viability,
                        "resolution_safety": row.resolution_safety,
                        "execution_feasibility": row.execution_feasibility,
                        "market_efficiency": row.market_efficiency,
                        "recommendation": row.recommendation,
                        "reasoning": row.reasoning,
                        "risk_factors": row.risk_factors or [],
                        "reject_codes": [
                            str(x).replace("CODE::", "")
                            for x in (row.risk_factors or [])
                            if str(x).startswith("CODE::")
                        ],
                        "ml_probability": row.ml_probability,
                        "ml_recommendation": row.ml_recommendation,
                        "agreement": row.agreement,
                        "model_used": row.model_used,
                        "judged_at": row.judged_at.isoformat() if row.judged_at else None,
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Failed to fetch judgment history", error=str(exc))
            return []

    async def get_agreement_stats(self) -> dict:
        """Get statistics on LLM vs ML classifier agreement.

        Returns
        -------
        dict with agreement rate, total judgments, counts by
        recommendation, and disagreement breakdown.
        """
        try:
            async with AsyncSessionLocal() as session:
                # Total judgments with ML comparison
                total_result = await session.execute(
                    select(func.count(OpportunityJudgment.id)).where(OpportunityJudgment.agreement.isnot(None))
                )
                total_with_ml = total_result.scalar() or 0

                # Agreement count
                agree_result = await session.execute(
                    select(func.count(OpportunityJudgment.id)).where(
                        OpportunityJudgment.agreement == True  # noqa: E712
                    )
                )
                agreements = agree_result.scalar() or 0

                # Disagreement count
                disagree_result = await session.execute(
                    select(func.count(OpportunityJudgment.id)).where(
                        OpportunityJudgment.agreement == False  # noqa: E712
                    )
                )
                disagreements = disagree_result.scalar() or 0

                # Total judgments overall
                all_result = await session.execute(select(func.count(OpportunityJudgment.id)))
                total_all = all_result.scalar() or 0

                # Average scores
                avg_result = await session.execute(select(func.avg(OpportunityJudgment.overall_score)))
                avg_score = avg_result.scalar()

                # Recommendation distribution
                rec_result = await session.execute(
                    select(
                        OpportunityJudgment.recommendation,
                        func.count(OpportunityJudgment.id),
                    ).group_by(OpportunityJudgment.recommendation)
                )
                recommendation_counts = {row[0]: row[1] for row in rec_result.all()}

                agreement_rate = round(agreements / total_with_ml, 4) if total_with_ml > 0 else None

                avg_overall = round(float(avg_score), 4) if avg_score else 0.0

                return {
                    "total_judgments": total_all,
                    "total_judged": total_all,
                    "total_with_ml_comparison": total_with_ml,
                    "agreements": agreements,
                    "disagreements": disagreements,
                    "ml_overrides": disagreements,
                    "agreement_rate": agreement_rate if agreement_rate is not None else 0.0,
                    "average_overall_score": avg_overall,
                    "avg_score": avg_overall,
                    "recommendation_distribution": recommendation_counts,
                }
        except Exception as exc:
            logger.error("Failed to compute agreement stats", error=str(exc))
            return {
                "total_judgments": 0,
                "total_judged": 0,
                "total_with_ml_comparison": 0,
                "agreements": 0,
                "disagreements": 0,
                "ml_overrides": 0,
                "agreement_rate": 0.0,
                "average_overall_score": 0.0,
                "avg_score": 0.0,
                "recommendation_distribution": {},
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_judgment_prompt(
        self,
        opportunity: "ArbitrageOpportunity",
        resolution_analysis: dict | None = None,
        ml_prediction: dict | None = None,
    ) -> str:
        """Build the user prompt with all opportunity details."""
        sections = ["## Opportunity to Judge\n"]

        # Core fields
        sections.append(f"**ID:** {opportunity.id}")
        sections.append(f"**Strategy:** {opportunity.strategy if opportunity.strategy else 'unknown'}")
        sections.append(f"**Title:** {opportunity.title}")
        sections.append(f"**Description:** {opportunity.description}")
        sections.append(
            f"**Guaranteed arbitrage:** {'yes' if bool(opportunity.is_guaranteed) else 'no'}"
        )
        if opportunity.roi_type:
            sections.append(f"**ROI type:** {opportunity.roi_type}")

        # Profit metrics
        sections.append("\n### Profit Metrics")
        sections.append(f"- Total cost: ${opportunity.total_cost:.4f}")
        sections.append(f"- Expected payout: ${opportunity.expected_payout:.4f}")
        sections.append(f"- Gross profit: ${opportunity.gross_profit:.4f}")
        sections.append(f"- Fee (2% on winnings): ${opportunity.fee:.4f}")
        sections.append(f"- Net profit: ${opportunity.net_profit:.4f}")
        sections.append(f"- ROI: {opportunity.roi_percent:.2f}%")

        # Risk
        sections.append("\n### Risk Assessment")
        sections.append(f"- Risk score: {opportunity.risk_score:.2f}")
        if opportunity.risk_factors:
            sections.append(f"- Risk factors: {', '.join(opportunity.risk_factors)}")

        # Liquidity
        sections.append("\n### Liquidity & Execution")
        sections.append(f"- Minimum liquidity: ${opportunity.min_liquidity:.2f}")
        sections.append(f"- Maximum position size: ${opportunity.max_position_size:.2f}")

        # Mispricing classification
        if opportunity.mispricing_type:
            sections.append(f"\n**Mispricing type:** {opportunity.mispricing_type.value}")
        if opportunity.guaranteed_profit is not None:
            sections.append(f"**Guaranteed profit (Frank-Wolfe):** ${opportunity.guaranteed_profit:.4f}")
        if opportunity.capture_ratio is not None:
            sections.append(f"**Capture ratio:** {opportunity.capture_ratio:.4f}")

        # Markets involved
        if opportunity.markets:
            sections.append("\n### Markets Involved")
            for i, mkt in enumerate(opportunity.markets):
                try:
                    if not isinstance(mkt, dict):
                        sections.append(f"\n**Market {i + 1}:** {mkt}")
                        continue
                    sections.append(f"\n**Market {i + 1}:**")
                    sections.append(f"  - ID: {mkt.get('id', mkt.get('condition_id', 'N/A'))}")
                    sections.append(f"  - Question: {mkt.get('question', 'N/A')}")
                    yes_price = mkt.get("yes_price", "N/A")
                    no_price = mkt.get("no_price", "N/A")
                    if yes_price == "N/A":
                        outcome_prices = mkt.get("outcome_prices")
                        if isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                            yes_price = outcome_prices[0]
                        if isinstance(outcome_prices, list) and len(outcome_prices) > 1:
                            no_price = outcome_prices[1]
                    sections.append(f"  - YES price: {yes_price}")
                    sections.append(f"  - NO price: {no_price}")
                    liquidity = mkt.get("liquidity")
                    if liquidity is not None:
                        try:
                            sections.append(f"  - Liquidity: ${float(liquidity):.2f}")
                        except (ValueError, TypeError):
                            sections.append(f"  - Liquidity: {liquidity}")
                except Exception as exc:
                    sections.append(f"\n**Market {i + 1}:** (error reading market data: {exc})")

        weather = self._extract_weather_context(opportunity)
        if weather:
            sections.append("\n### Weather Context")
            sections.append(
                f"- Report-only mode: {'yes' if self._is_report_only_opportunity(opportunity) else 'no'}"
            )
            if weather.get("location"):
                sections.append(f"- Location: {weather.get('location')}")
            if weather.get("target_time"):
                sections.append(f"- Target time: {weather.get('target_time')}")
            if isinstance(weather.get("source_count"), (int, float)):
                sections.append(f"- Forecast sources: {int(weather.get('source_count') or 0)}")
            if isinstance(weather.get("consensus_probability"), (int, float)):
                sections.append(
                    f"- Model probability (YES): {float(weather.get('consensus_probability')):.3f}"
                )
            if isinstance(weather.get("market_probability"), (int, float)):
                sections.append(
                    f"- Market probability (selected side): {float(weather.get('market_probability')):.3f}"
                )
            if isinstance(weather.get("consensus_temp_f"), (int, float)):
                sections.append(
                    f"- Consensus temperature: {float(weather.get('consensus_temp_f')):.1f}F"
                )
            if isinstance(weather.get("market_implied_temp_f"), (int, float)):
                sections.append(
                    f"- Market-implied temperature: {float(weather.get('market_implied_temp_f')):.1f}F"
                )
            if isinstance(weather.get("agreement"), (int, float)):
                sections.append(
                    f"- Model agreement: {float(weather.get('agreement')):.3f}"
                )

            raw_sources = weather.get("forecast_sources")
            if isinstance(raw_sources, list) and raw_sources:
                sections.append("- Forecast source snapshots:")
                for src in raw_sources[:3]:
                    if not isinstance(src, dict):
                        continue
                    source_id = src.get("source_id", "unknown")
                    src_prob = src.get("probability")
                    src_temp = src.get("value_f")
                    src_weight = src.get("weight")
                    prob_txt = (
                        f"{float(src_prob):.3f}" if isinstance(src_prob, (int, float)) else "n/a"
                    )
                    temp_txt = (
                        f"{float(src_temp):.1f}F"
                        if isinstance(src_temp, (int, float))
                        else "n/a"
                    )
                    weight_txt = (
                        f"{float(src_weight):.2f}"
                        if isinstance(src_weight, (int, float))
                        else "n/a"
                    )
                    sections.append(
                        f"  - {source_id}: prob={prob_txt}, temp={temp_txt}, weight={weight_txt}"
                    )

        # Event context
        if opportunity.event_title:
            sections.append(f"\n**Event:** {opportunity.event_title}")
        if opportunity.category:
            sections.append(f"**Category:** {opportunity.category}")

        # Timing
        if opportunity.resolution_date:
            sections.append(f"**Resolution date:** {opportunity.resolution_date.isoformat()}")
        sections.append(f"**Detected at:** {opportunity.detected_at.isoformat()}")

        # Positions to take
        if opportunity.positions_to_take:
            sections.append("\n### Positions to Take")
            for pos in opportunity.positions_to_take:
                side = pos.get("side", "unknown")
                outcome = pos.get("outcome", "N/A")
                price = pos.get("price", "N/A")
                size = pos.get("size", pos.get("amount", "N/A"))
                sections.append(f"  - {side} {outcome} @ {price} (size: {size})")

        # Resolution analysis if available
        if resolution_analysis:
            sections.append("\n### Pre-computed Resolution Analysis")
            sections.append(f"- Clarity score: {resolution_analysis.get('clarity_score', 'N/A')}")
            sections.append(f"- Risk score: {resolution_analysis.get('risk_score', 'N/A')}")
            sections.append(f"- Recommendation: {resolution_analysis.get('recommendation', 'N/A')}")
            ambiguities = resolution_analysis.get("ambiguities", [])
            if ambiguities:
                sections.append(f"- Ambiguities: {'; '.join(ambiguities[:5])}")
            edge_cases = resolution_analysis.get("edge_cases", [])
            if edge_cases:
                sections.append(f"- Edge cases: {'; '.join(edge_cases[:5])}")
            summary = resolution_analysis.get("summary", "")
            if summary:
                sections.append(f"- Summary: {summary}")

        # ML prediction for comparison
        if ml_prediction:
            sections.append("\n### ML Classifier Assessment")
            sections.append(f"- Probability of profitability: {ml_prediction.get('probability', 'N/A')}")
            sections.append(f"- Recommendation: {ml_prediction.get('recommendation', 'N/A')}")
            sections.append(f"- Confidence: {ml_prediction.get('confidence', 'N/A')}")

        sections.append(
            "\n---\n"
            "Score this opportunity on all dimensions. Be skeptical -- "
            "most detected prediction market arbitrage opportunities are "
            "false positives. Explain your recommendation briefly and concretely. "
            "Do not assert facts that are not explicitly present in the data above."
        )

        return "\n".join(sections)

    def _build_weather_judgment_prompt(
        self,
        opportunity: "ArbitrageOpportunity",
        resolution_analysis: dict | None = None,
        ml_prediction: dict | None = None,
    ) -> str:
        """Build a compact weather-specific prompt to reduce context pressure."""
        sections = ["## Weather Opportunity to Judge"]
        sections.append(f"- ID: {opportunity.id}")
        sections.append(f"- Title: {opportunity.title}")
        sections.append(f"- Description: {opportunity.description}")
        sections.append(
            f"- Report-only: {'yes' if self._is_report_only_opportunity(opportunity) else 'no'}"
        )
        sections.append(
            f"- Liquidity: min_liquidity=${opportunity.min_liquidity:.2f}, "
            f"max_position_size=${opportunity.max_position_size:.2f}"
        )
        sections.append(
            f"- Profit fields: total_cost=${opportunity.total_cost:.4f}, "
            f"expected_payout=${opportunity.expected_payout:.4f}, "
            f"net_profit=${opportunity.net_profit:.4f}, roi={opportunity.roi_percent:.2f}%"
        )
        if opportunity.risk_factors:
            sections.append(
                f"- Risk factors: {'; '.join(str(x) for x in opportunity.risk_factors[:6])}"
            )

        if opportunity.positions_to_take:
            pos = opportunity.positions_to_take[0]
            sections.append(
                f"- Proposed leg: {pos.get('action', 'BUY')} {pos.get('outcome', 'N/A')} "
                f"@ {pos.get('price', 'N/A')}"
            )

        weather = self._extract_weather_context(opportunity) or {}
        if weather:
            sections.append("### Weather Data")
            for label, key in [
                ("Location", "location"),
                ("Target time", "target_time"),
                ("Source count", "source_count"),
                ("Consensus YES probability", "consensus_probability"),
                ("Market selected-side probability", "market_probability"),
                ("Consensus temp F", "consensus_temp_f"),
                ("Market-implied temp F", "market_implied_temp_f"),
                ("Model agreement", "agreement"),
            ]:
                value = weather.get(key)
                if value is not None:
                    sections.append(f"- {label}: {value}")

            raw_sources = weather.get("forecast_sources")
            if isinstance(raw_sources, list) and raw_sources:
                sections.append("- Forecast sources:")
                for src in raw_sources[:4]:
                    if not isinstance(src, dict):
                        continue
                    sections.append(
                        f"  - {src.get('source_id', 'unknown')}: "
                        f"prob={src.get('probability', 'n/a')}, "
                        f"temp_f={src.get('value_f', 'n/a')}, "
                        f"weight={src.get('weight', 'n/a')}"
                    )

        if resolution_analysis:
            sections.append("### Resolution Analysis")
            sections.append(
                f"- clarity={resolution_analysis.get('clarity_score', 'N/A')}, "
                f"risk={resolution_analysis.get('risk_score', 'N/A')}, "
                f"rec={resolution_analysis.get('recommendation', 'N/A')}"
            )
            summary = resolution_analysis.get("summary")
            if summary:
                sections.append(f"- Summary: {summary}")

        if ml_prediction:
            sections.append("### ML Classifier")
            sections.append(
                f"- probability={ml_prediction.get('probability', 'N/A')}, "
                f"recommendation={ml_prediction.get('recommendation', 'N/A')}, "
                f"confidence={ml_prediction.get('confidence', 'N/A')}"
            )

        sections.append(
            "---\n"
            "Score strictly from the provided fields. "
            "If execution is blocked (report-only or max_position_size=0), "
            "execution_feasibility should be 0 and recommendation should be skip/strong_skip."
        )
        return "\n".join(sections)

    @staticmethod
    def _extract_weather_context(opportunity: "ArbitrageOpportunity") -> dict | None:
        for market in opportunity.markets or []:
            if not isinstance(market, dict):
                continue
            weather = market.get("weather")
            if isinstance(weather, dict):
                return weather
        return None

    def _is_weather_opportunity(self, opportunity: "ArbitrageOpportunity") -> bool:
        return (
            str(opportunity.strategy).lower() == "weather_edge"
            or self._extract_weather_context(opportunity) is not None
        )

    @staticmethod
    def _is_directional_opportunity(opportunity: "ArbitrageOpportunity") -> bool:
        return not bool(getattr(opportunity, "is_guaranteed", False))

    @staticmethod
    def _is_report_only_opportunity(opportunity: "ArbitrageOpportunity") -> bool:
        if (opportunity.max_position_size or 0.0) <= 0.0:
            return True
        desc = (opportunity.description or "").strip().lower()
        return desc.startswith("report only")

    def _report_only_judgment(self, opportunity: "ArbitrageOpportunity") -> dict:
        reason_bits = []
        for rf in (opportunity.risk_factors or []):
            text = str(rf).strip()
            if not text:
                continue
            reason_bits.append(text)
            if len(reason_bits) >= 4:
                break
        reason_tail = (
            " Key blockers: " + "; ".join(reason_bits)
            if reason_bits
            else ""
        )
        return {
            "overall_score": 0.12,
            "profit_viability": 0.30,
            "resolution_safety": 0.45,
            "execution_feasibility": 0.0,
            "market_efficiency": 0.20,
            "recommendation": "strong_skip",
            "reasoning": (
                "Opportunity is report-only/non-executable in current state "
                f"(max_position_size=${opportunity.max_position_size:.2f})."
                f"{reason_tail}"
            ),
            "risk_factors": [
                "CODE::NON_EXECUTABLE",
                "Report-only opportunity (non-executable)",
                *reason_bits,
            ],
            "reject_codes": ["NON_EXECUTABLE"],
        }

    @staticmethod
    def _derive_reject_codes(judgment: dict) -> list[str]:
        text_bits: list[str] = []
        reasoning = str(judgment.get("reasoning") or "").strip().lower()
        if reasoning:
            text_bits.append(reasoning)
        for rf in judgment.get("risk_factors", []) or []:
            t = str(rf or "").strip().lower()
            if t:
                text_bits.append(t)
        text = " | ".join(text_bits)

        codes: list[str] = []
        if (judgment.get("execution_feasibility") or 0.0) <= 0.05:
            codes.append("NON_EXECUTABLE")
        if "report-only" in text or "report only" in text:
            codes.append("NON_EXECUTABLE")
        if "directional bet" in text:
            codes.append("DIRECTIONAL_NON_ARB")
        if "rate-limited" in text or "source diversity" in text or "single-source" in text:
            codes.append("DATA_QUALITY")
        if "slippage" in text:
            codes.append("SLIPPAGE_RISK")
        if "liquidity" in text:
            codes.append("LIQUIDITY_RISK")
        if "edge" in text and "< min" in text:
            codes.append("WEAK_EDGE")
        if "confidence" in text and "< min" in text:
            codes.append("LOW_CONFIDENCE")

        recommendation = str(judgment.get("recommendation") or "").strip().lower()
        if recommendation in {"skip", "strong_skip"} and "NON_EXECUTABLE" not in codes:
            # Keep a default structured tag for skip-like outcomes.
            codes.append("RISK_REJECTED")

        # Preserve order while de-duplicating.
        seen: set[str] = set()
        deduped: list[str] = []
        for code in codes:
            if code in seen:
                continue
            seen.add(code)
            deduped.append(code)
        return deduped

    def _annotate_reject_codes(self, judgment: dict) -> dict:
        codes = self._derive_reject_codes(judgment)
        factors = [str(x) for x in (judgment.get("risk_factors") or []) if str(x).strip()]
        for code in codes:
            tag = f"CODE::{code}"
            if tag not in factors:
                factors.insert(0, tag)
        judgment["risk_factors"] = factors
        judgment["reject_codes"] = codes
        return judgment

    def _validate_judgment(self, judgment: dict) -> dict:
        """Ensure all required fields exist with sensible defaults."""
        # Guard: if the LLM returned a non-dict (e.g. a list), return
        # safe defaults instead of crashing with a TypeError.
        if not isinstance(judgment, dict):
            logger.warning(
                "Judgment validation received %s instead of dict, using defaults",
                type(judgment).__name__,
            )
            return {
                "overall_score": 0.2,
                "profit_viability": 0.2,
                "resolution_safety": 0.3,
                "execution_feasibility": 0.2,
                "market_efficiency": 0.2,
                "recommendation": "strong_skip",
                "reasoning": ("LLM returned a non-dict response. Defaulting to strong_skip as a safety measure."),
                "risk_factors": ["LLM response format error"],
            }

        defaults = {
            "overall_score": 0.3,
            "profit_viability": 0.3,
            "resolution_safety": 0.5,
            "execution_feasibility": 0.3,
            "market_efficiency": 0.3,
            "recommendation": "skip",
            "reasoning": "Judgment completed but some fields were missing.",
            "risk_factors": [],
        }
        for key, default in defaults.items():
            if key not in judgment or judgment[key] is None:
                judgment[key] = default

        # Clamp numeric scores to [0, 1]
        for score_key in (
            "overall_score",
            "profit_viability",
            "resolution_safety",
            "execution_feasibility",
            "market_efficiency",
        ):
            val = judgment[score_key]
            if isinstance(val, (int, float)):
                judgment[score_key] = max(0.0, min(1.0, float(val)))
            else:
                judgment[score_key] = defaults[score_key]

        # Validate recommendation enum
        valid_recs = {"strong_execute", "execute", "review", "skip", "strong_skip"}
        if judgment["recommendation"] not in valid_recs:
            judgment["recommendation"] = "skip"

        return judgment

    def _fallback_judgment(self, reason: str) -> dict:
        """Return a conservative fallback judgment when the LLM call fails."""
        return {
            "overall_score": 0.2,
            "profit_viability": 0.2,
            "resolution_safety": 0.3,
            "execution_feasibility": 0.2,
            "market_efficiency": 0.2,
            "recommendation": "strong_skip",
            "reasoning": (
                f"Automated judgment was unable to complete ({reason}). Defaulting to strong_skip as a safety measure."
            ),
            "risk_factors": [
                "CODE::JUDGMENT_FAILURE",
                f"Judgment failed: {reason}",
            ],
            "reject_codes": ["JUDGMENT_FAILURE"],
        }

    def _compute_ml_agreement(
        self,
        judgment: dict,
        ml_prediction: dict | None,
    ) -> bool | None:
        """Determine whether the LLM judge agrees with the ML classifier.

        Agreement is defined as both recommending the same directional
        action: both say execute/strong_execute, or both say skip/strong_skip.
        The 'review' recommendation is considered neutral and agrees with
        either side.

        Returns None if no ML prediction is available.
        """
        if not ml_prediction:
            return None

        ml_rec = ml_prediction.get("recommendation", "review")
        llm_rec = judgment.get("recommendation", "skip")

        # Map to directional signals
        execute_signals = {"strong_execute", "execute"}
        skip_signals = {"skip", "strong_skip"}

        ml_execute = ml_rec in execute_signals or ml_rec == "execute"
        ml_skip = ml_rec in skip_signals or ml_rec == "skip"

        llm_execute = llm_rec in execute_signals
        llm_skip = llm_rec in skip_signals

        # Agreement: both execute, both skip, or either is neutral ("review")
        if ml_rec == "review" or llm_rec == "review":
            return True
        return (ml_execute and llm_execute) or (ml_skip and llm_skip)

    async def _store_judgment(
        self,
        result: dict,
        opportunity: "ArbitrageOpportunity",
        ml_prediction: dict | None,
        session_id: str,
        model_used: str,
        started_at: datetime,
    ) -> None:
        """Persist the judgment to the database."""
        try:
            completed_at = utcnow()
            duration = (completed_at - started_at).total_seconds()

            async with AsyncSessionLocal() as session:
                # Research session record
                research_session = ResearchSession(
                    id=session_id,
                    session_type="opportunity_judge",
                    query=f"Judge opportunity {opportunity.id}",
                    opportunity_id=opportunity.id,
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

                # Opportunity judgment record
                judgment_row = OpportunityJudgment(
                    id=uuid.uuid4().hex[:16],
                    opportunity_id=opportunity.id,
                    strategy_type=opportunity.strategy if opportunity.strategy else "unknown",
                    overall_score=result["overall_score"],
                    profit_viability=result["profit_viability"],
                    resolution_safety=result["resolution_safety"],
                    execution_feasibility=result["execution_feasibility"],
                    market_efficiency=result["market_efficiency"],
                    reasoning=result["reasoning"],
                    recommendation=result["recommendation"],
                    risk_factors=result.get("risk_factors"),
                    ml_probability=ml_prediction.get("probability") if ml_prediction else None,
                    ml_recommendation=ml_prediction.get("recommendation") if ml_prediction else None,
                    agreement=result.get("ml_agreement"),
                    session_id=session_id,
                    model_used=model_used,
                    judged_at=started_at,
                )
                session.add(judgment_row)
                await session.commit()

            logger.info(
                "Opportunity judgment stored",
                opportunity_id=opportunity.id,
                session_id=session_id,
                duration_s=round(duration, 2),
            )
        except Exception as exc:
            logger.error(
                "Failed to store opportunity judgment",
                opportunity_id=opportunity.id,
                error=str(exc),
            )

    async def _record_failed_session(
        self,
        session_id: str,
        opportunity_id: str,
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
                    session_type="opportunity_judge",
                    query=f"Judge opportunity {opportunity_id}",
                    opportunity_id=opportunity_id,
                    status="failed",
                    error=error,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                )
                session.add(research_session)
                await session.commit()
        except Exception as exc:
            logger.error(
                "Failed to record failed session",
                session_id=session_id,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

opportunity_judge = OpportunityJudge()
