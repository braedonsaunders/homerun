"""
API routes for AI intelligence features.

Provides endpoints for:
- Resolution criteria analysis
- Opportunity judging
- Market analysis
- News sentiment
- Skill management
- Research session history
- LLM usage stats
- AI chat / copilot
- Market search (for smart autocomplete)
- Opportunity AI summaries
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# === Resolution Analysis ===


class ResolutionAnalysisRequest(BaseModel):
    market_id: str
    question: str
    description: str = ""
    resolution_source: str = ""
    end_date: str = ""
    outcomes: list[str] = []
    force_refresh: bool = False
    model: Optional[str] = None


@router.post("/ai/resolution/analyze")
async def analyze_resolution(request: ResolutionAnalysisRequest):
    """Analyze a market's resolution criteria."""
    try:
        from services.ai.resolution_analyzer import resolution_analyzer

        result = await resolution_analyzer.analyze_market(
            market_id=request.market_id,
            question=request.question,
            description=request.description,
            resolution_source=request.resolution_source,
            end_date=request.end_date,
            outcomes=request.outcomes,
            force_refresh=request.force_refresh,
            model=request.model,
        )
        return result
    except Exception as e:
        logger.error(f"Resolution analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/resolution/{market_id}")
async def get_resolution_analysis(market_id: str):
    """Get cached resolution analysis for a market."""
    from services.ai.resolution_analyzer import resolution_analyzer

    result = await resolution_analyzer.get_cached_analysis(market_id)
    if not result:
        raise HTTPException(status_code=404, detail="No analysis found for this market")
    return result


@router.get("/ai/resolution/history")
async def get_resolution_history(
    market_id: Optional[str] = None, limit: int = Query(20, le=100)
):
    """Get resolution analysis history."""
    from services.ai.resolution_analyzer import resolution_analyzer

    return await resolution_analyzer.get_analysis_history(
        market_id=market_id, limit=limit
    )


# === Opportunity Judging ===


class JudgeOpportunityRequest(BaseModel):
    opportunity_id: str
    model: Optional[str] = None


@router.post("/ai/judge/opportunity")
async def judge_opportunity(request: JudgeOpportunityRequest):
    """Judge a specific opportunity using LLM."""
    # Get opportunity from scanner
    from services import scanner

    opps = scanner.get_opportunities()
    opp = next((o for o in opps if o.id == request.opportunity_id), None)
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    from services.ai.opportunity_judge import opportunity_judge

    result = await opportunity_judge.judge_opportunity(opp, model=request.model)

    # Update the in-memory opportunity so subsequent API fetches include the
    # judgment without waiting for the next scan cycle.
    from models.opportunity import AIAnalysis
    from datetime import datetime

    opp.ai_analysis = AIAnalysis(
        overall_score=result.get("overall_score", 0.0),
        profit_viability=result.get("profit_viability", 0.0),
        resolution_safety=result.get("resolution_safety", 0.0),
        execution_feasibility=result.get("execution_feasibility", 0.0),
        market_efficiency=result.get("market_efficiency", 0.0),
        recommendation=result.get("recommendation", "review"),
        reasoning=result.get("reasoning"),
        risk_factors=result.get("risk_factors", []),
        judged_at=datetime.fromisoformat(result["judged_at"])
        if result.get("judged_at")
        else datetime.utcnow(),
    )

    return result


class JudgeBulkRequest(BaseModel):
    opportunity_ids: list[str] = []  # empty = all unjudged


@router.post("/ai/judge/opportunities/bulk")
async def judge_opportunities_bulk(request: JudgeBulkRequest):
    """Judge multiple opportunities sequentially. Returns results as they complete."""
    import asyncio
    from services import scanner
    from services.ai.opportunity_judge import opportunity_judge
    from models.opportunity import AIAnalysis
    from datetime import datetime

    opps = scanner.get_opportunities()

    if request.opportunity_ids:
        id_set = set(request.opportunity_ids)
        targets = [o for o in opps if o.id in id_set]
    else:
        # Judge all that don't already have a non-pending analysis
        targets = [
            o
            for o in opps
            if not o.ai_analysis or o.ai_analysis.recommendation == "pending"
        ]

    results = []
    errors = []

    for opp in targets:
        try:
            result = await opportunity_judge.judge_opportunity(opp)
            # Update in-memory opportunity
            opp.ai_analysis = AIAnalysis(
                overall_score=result.get("overall_score", 0.0),
                profit_viability=result.get("profit_viability", 0.0),
                resolution_safety=result.get("resolution_safety", 0.0),
                execution_feasibility=result.get("execution_feasibility", 0.0),
                market_efficiency=result.get("market_efficiency", 0.0),
                recommendation=result.get("recommendation", "review"),
                reasoning=result.get("reasoning"),
                risk_factors=result.get("risk_factors", []),
                judged_at=datetime.fromisoformat(result["judged_at"])
                if result.get("judged_at")
                else datetime.utcnow(),
            )
            results.append(result)
        except Exception as e:
            errors.append({"opportunity_id": opp.id, "error": str(e)})

    return {"judged": len(results), "errors": errors, "total_requested": len(targets)}


@router.get("/ai/judge/history")
async def get_judgment_history(
    opportunity_id: Optional[str] = None,
    strategy_type: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: int = Query(50, le=200),
):
    """Get opportunity judgment history."""
    from services.ai.opportunity_judge import opportunity_judge

    return await opportunity_judge.get_judgment_history(
        opportunity_id=opportunity_id,
        strategy_type=strategy_type,
        min_score=min_score,
        limit=limit,
    )


@router.get("/ai/judge/agreement-stats")
async def get_agreement_stats():
    """Get ML vs LLM agreement statistics."""
    from services.ai.opportunity_judge import opportunity_judge

    return await opportunity_judge.get_agreement_stats()


# === Market Analysis ===


class MarketAnalysisRequest(BaseModel):
    query: str
    market_id: Optional[str] = None
    market_question: Optional[str] = None
    model: Optional[str] = None


@router.post("/ai/market/analyze")
async def analyze_market(request: MarketAnalysisRequest):
    """Run an AI-powered market analysis."""
    from services.ai.market_analyzer import market_analyzer

    result = await market_analyzer.analyze(
        query=request.query,
        market_id=request.market_id,
        market_question=request.market_question,
        model=request.model,
    )
    return result


# === News Sentiment ===


class NewsSentimentRequest(BaseModel):
    query: str
    market_context: str = ""
    max_articles: int = 5
    model: Optional[str] = None


@router.post("/ai/news/sentiment")
async def analyze_news_sentiment(request: NewsSentimentRequest):
    """Search news and analyze sentiment."""
    from services.ai.news_sentiment import news_sentiment_analyzer

    result = await news_sentiment_analyzer.search_and_analyze(
        query=request.query,
        market_context=request.market_context,
        max_articles=request.max_articles,
        model=request.model,
    )
    return result


# === Skills ===


@router.get("/ai/skills")
async def list_skills():
    """List available AI skills."""
    from services.ai.skills.loader import skill_loader

    return skill_loader.list_skills()


class ExecuteSkillRequest(BaseModel):
    skill_name: str
    context: dict = {}
    model: Optional[str] = None


@router.post("/ai/skills/execute")
async def execute_skill(request: ExecuteSkillRequest):
    """Execute an AI skill."""
    from services.ai.skills.loader import skill_loader

    skill = skill_loader.get_skill(request.skill_name)
    if not skill:
        raise HTTPException(
            status_code=404, detail=f"Skill not found: {request.skill_name}"
        )

    result = await skill_loader.execute_skill(
        name=request.skill_name,
        context=request.context,
        model=request.model,
    )
    return result


# === Research Sessions ===


@router.get("/ai/sessions")
async def get_research_sessions(
    session_type: Optional[str] = None,
    limit: int = Query(20, le=100),
):
    """Get recent research sessions."""
    from services.ai.scratchpad import ScratchpadService

    scratchpad = ScratchpadService()
    return await scratchpad.get_recent_sessions(session_type=session_type, limit=limit)


@router.get("/ai/sessions/{session_id}")
async def get_research_session(session_id: str):
    """Get a specific research session with all entries."""
    from services.ai.scratchpad import ScratchpadService

    scratchpad = ScratchpadService()
    result = await scratchpad.get_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


# === Usage Stats ===


@router.get("/ai/usage")
async def get_ai_usage():
    """Get AI/LLM usage statistics."""
    from services.ai import get_llm_manager

    manager = get_llm_manager()
    return await manager.get_usage_stats()


# === AI Status ===


@router.get("/ai/status")
async def get_ai_status():
    """Get overall AI system status."""
    try:
        from services.ai import get_llm_manager
        from services.ai.skills.loader import skill_loader

        manager = get_llm_manager()
        return {
            "enabled": manager.is_available(),
            "providers_configured": list(manager._providers.keys())
            if hasattr(manager, "_providers")
            else [],
            "skills_available": len(skill_loader.list_skills()),
            "usage": await manager.get_usage_stats()
            if manager.is_available()
            else None,
        }
    except RuntimeError:
        return {
            "enabled": False,
            "providers_configured": [],
            "skills_available": 0,
            "usage": None,
        }


# === Market Search (for smart autocomplete) ===


@router.get("/ai/markets/search")
async def search_markets(
    q: str = Query(..., min_length=1, description="Search query for market titles"),
    limit: int = Query(10, le=50),
):
    """Search available markets by question text for autocomplete.

    Returns markets from the scanner's current market pool, allowing users
    to find markets without needing to manually look up IDs.
    """
    from services import scanner

    opportunities = scanner.get_opportunities()
    seen_ids = set()
    results = []
    q_lower = q.lower()

    # Collect unique markets from all opportunities
    for opp in opportunities:
        for m in opp.markets:
            mid = m.get("id", "")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            question = m.get("question", "")
            if q_lower in question.lower() or q_lower in mid.lower():
                results.append(
                    {
                        "market_id": mid,
                        "question": question,
                        "yes_price": m.get("yes_price"),
                        "no_price": m.get("no_price"),
                        "liquidity": m.get("liquidity"),
                        "event_title": opp.event_title,
                        "category": opp.category,
                    }
                )
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break

    return {"results": results, "total": len(results)}


# === Opportunity AI Summary ===


@router.get("/ai/opportunity/{opportunity_id}/summary")
async def get_opportunity_ai_summary(opportunity_id: str):
    """Get a quick AI intelligence summary for a specific opportunity.

    Returns cached judgment + resolution analysis if available,
    or triggers a quick analysis if not cached.
    """
    from services import scanner

    opps = scanner.get_opportunities()
    opp = next((o for o in opps if o.id == opportunity_id), None)
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    summary = {
        "opportunity_id": opportunity_id,
        "judgment": None,
        "resolution_analyses": [],
    }

    # Try to get cached judgment
    try:
        from services.ai.opportunity_judge import opportunity_judge

        history = await opportunity_judge.get_judgment_history(
            opportunity_id=opportunity_id, limit=1
        )
        if history and len(history) > 0:
            summary["judgment"] = history[0]
    except Exception as e:
        logger.debug(f"No cached judgment for {opportunity_id}: {e}")

    # Try to get cached resolution analyses for each market
    try:
        from services.ai.resolution_analyzer import resolution_analyzer

        for m in opp.markets:
            mid = m.get("id", "")
            if mid:
                cached = await resolution_analyzer.get_cached_analysis(mid)
                if cached:
                    summary["resolution_analyses"].append(cached)
    except Exception as e:
        logger.debug(f"No cached resolution for {opportunity_id}: {e}")

    return summary


# === AI Chat / Copilot ===


class AIChatRequest(BaseModel):
    message: str
    context_type: Optional[str] = None  # "opportunity", "market", "general"
    context_id: Optional[str] = None  # opportunity_id or market_id
    history: list[dict] = []  # prior messages [{role, content}]
    model: Optional[str] = None


@router.post("/ai/chat")
async def ai_chat(request: AIChatRequest):
    """Conversational AI copilot for the trading platform.

    Context-aware chat that understands the current page/opportunity
    the user is viewing and can answer questions, analyze markets,
    and provide trading recommendations.
    """
    try:
        from services.ai import get_llm_manager

        manager = get_llm_manager()
        if not manager.is_available():
            raise HTTPException(
                status_code=503,
                detail="No AI provider configured. Add an API key in Settings.",
            )

        # Build context
        context_parts = []

        if request.context_type == "opportunity" and request.context_id:
            from services import scanner

            opps = scanner.get_opportunities()
            opp = next((o for o in opps if o.id == request.context_id), None)
            if opp:
                context_parts.append(
                    f"The user is currently viewing this arbitrage opportunity:\n"
                    f"Title: {opp.title}\n"
                    f"Strategy: {opp.strategy}\n"
                    f"ROI: {opp.roi_percent:.2f}%\n"
                    f"Net Profit: ${opp.net_profit:.4f}\n"
                    f"Risk Score: {opp.risk_score:.2f}\n"
                    f"Risk Factors: {', '.join(opp.risk_factors)}\n"
                    f"Markets: {', '.join(m.get('question', '') for m in opp.markets)}\n"
                    f"Event: {opp.event_title or 'N/A'}\n"
                    f"Category: {opp.category or 'N/A'}\n"
                    f"Total Cost: ${opp.total_cost:.4f}\n"
                    f"Max Position Size: ${opp.max_position_size:.2f}\n"
                )

        if request.context_type == "market" and request.context_id:
            context_parts.append(f"The user is viewing market ID: {request.context_id}")

        system_prompt = (
            "You are the AI copilot for Homerun, a Polymarket prediction market "
            "arbitrage trading platform. You help traders understand opportunities, "
            "analyze resolution criteria, assess risk, and make trading decisions.\n\n"
            "Key knowledge:\n"
            "- Polymarket uses UMA's Optimistic Oracle for resolution\n"
            "- 2% fee on net winnings\n"
            "- Strategies: basic arb, NegRisk, mutually exclusive, contradiction, "
            "must-happen, settlement lag\n"
            "- Risk factors: resolution ambiguity, liquidity, correlation, timing\n\n"
            "Be concise, specific, and data-driven. When the user asks about a "
            "specific opportunity, reference its actual data. Flag risks clearly.\n"
        )

        if context_parts:
            system_prompt += "\nCurrent context:\n" + "\n".join(context_parts)

        # Build messages with system prompt as first message
        from services.ai.llm_provider import LLMMessage

        messages = [LLMMessage(role="system", content=system_prompt)]
        for msg in request.history[-10:]:  # Keep last 10 messages
            messages.append(
                LLMMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
            )
        messages.append(LLMMessage(role="user", content=request.message))

        response = await manager.chat(
            messages=messages,
            model=request.model,
            max_tokens=1024,
            purpose="ai_chat",
        )

        return {
            "response": response.content or "",
            "model": response.model or "",
            "tokens_used": {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
