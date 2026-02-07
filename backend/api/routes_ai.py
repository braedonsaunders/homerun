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
    return result


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
    return await scratchpad.get_recent_sessions(
        session_type=session_type, limit=limit
    )


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
