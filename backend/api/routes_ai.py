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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging

from models.database import get_db_session
from services import shared_state
from services.weather import shared_state as weather_shared_state
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)

router = APIRouter()


async def _find_opportunity_by_id(session: AsyncSession, opportunity_id: str) -> tuple[Any, Optional[str]]:
    """Find opportunity across scanner and weather snapshots."""
    scanner_opps = await shared_state.get_opportunities_from_db(session, None)
    scanner_hit = next((o for o in scanner_opps if o.id == opportunity_id), None)
    if scanner_hit:
        return scanner_hit, "scanner"

    # Weather opportunities panel can display both executable opportunities and
    # report-only findings; AI lookup must search the same visible set.
    weather_opps = await weather_shared_state.get_weather_opportunities_from_db(
        session, include_report_only=True
    )
    weather_hit = next((o for o in weather_opps if o.id == opportunity_id), None)
    if weather_hit:
        return weather_hit, "weather"

    return None, None


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
async def get_resolution_history(market_id: Optional[str] = None, limit: int = Query(20, le=100)):
    """Get resolution analysis history."""
    from services.ai.resolution_analyzer import resolution_analyzer

    return await resolution_analyzer.get_analysis_history(market_id=market_id, limit=limit)


# === Opportunity Judging ===


class JudgeOpportunityRequest(BaseModel):
    opportunity_id: str
    model: Optional[str] = None
    force_llm: bool = False


@router.post("/ai/judge/opportunity")
async def judge_opportunity(
    request: JudgeOpportunityRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Judge a specific opportunity using LLM."""
    opp, snapshot_source = await _find_opportunity_by_id(session, request.opportunity_id)
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    from services.ai.opportunity_judge import opportunity_judge

    result = await opportunity_judge.judge_opportunity(
        opp,
        model=request.model,
        force_llm=bool(request.force_llm),
    )

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
        judged_at=datetime.fromisoformat(result["judged_at"]) if result.get("judged_at") else utcnow(),
    )
    try:
        if snapshot_source == "weather":
            await weather_shared_state.update_weather_opportunity_ai_analysis_in_snapshot(
                session=session,
                opportunity_id=opp.id,
                stable_id=opp.stable_id,
                ai_analysis=opp.ai_analysis.model_dump(mode="json"),
            )
        else:
            await shared_state.update_opportunity_ai_analysis_in_snapshot(
                session=session,
                opportunity_id=opp.id,
                stable_id=opp.stable_id,
                ai_analysis=opp.ai_analysis.model_dump(mode="json"),
            )
    except Exception as e:
        logger.warning(
            "Failed to persist inline ai_analysis into %s snapshot for %s: %s",
            snapshot_source or "unknown",
            opp.id,
            e,
        )

    return result


class JudgeBulkRequest(BaseModel):
    opportunity_ids: list[str] = []  # empty = all unjudged
    force: bool = False  # if True, re-analyze even if already judged


@router.post("/ai/judge/opportunities/bulk")
async def judge_opportunities_bulk(
    request: JudgeBulkRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Judge multiple opportunities sequentially. Returns results as they complete."""
    from services.ai.opportunity_judge import opportunity_judge
    from models.opportunity import AIAnalysis
    from datetime import datetime

    opps = await shared_state.get_opportunities_from_db(session, None)

    if request.opportunity_ids:
        id_set = set(request.opportunity_ids)
        targets = [o for o in opps if o.id in id_set]
    elif request.force:
        # Re-analyze all opportunities regardless of existing analysis
        targets = list(opps)
    else:
        # Judge all that don't already have a non-pending analysis
        targets = [o for o in opps if not o.ai_analysis or o.ai_analysis.recommendation == "pending"]

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
                judged_at=datetime.fromisoformat(result["judged_at"]) if result.get("judged_at") else utcnow(),
            )
            try:
                await shared_state.update_opportunity_ai_analysis_in_snapshot(
                    session=session,
                    opportunity_id=opp.id,
                    stable_id=opp.stable_id,
                    ai_analysis=opp.ai_analysis.model_dump(mode="json"),
                )
            except Exception:
                pass
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
        raise HTTPException(status_code=404, detail=f"Skill not found: {request.skill_name}")

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
    import asyncio

    try:
        from services.ai import get_llm_manager
        from services.ai.skills.loader import skill_loader

        manager = get_llm_manager()
        # Run sync filesystem scan in thread pool so it doesn't block the event loop
        skills_list = await asyncio.to_thread(skill_loader.list_skills)
        return {
            "enabled": manager.is_available(),
            "providers_configured": list(manager._providers.keys()) if hasattr(manager, "_providers") else [],
            "skills_available": len(skills_list),
            "usage": await manager.get_usage_stats() if manager.is_available() else None,
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
    session: AsyncSession = Depends(get_db_session),
    q: str = Query(..., min_length=1, description="Search query for market titles"),
    limit: int = Query(10, le=50),
):
    """Search available markets by question text for autocomplete.

    Returns markets from the current opportunity snapshot (DB).
    """
    opportunities = await shared_state.get_opportunities_from_db(session, None)
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
async def get_opportunity_ai_summary(
    opportunity_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get a quick AI intelligence summary for a specific opportunity.

    Returns cached judgment + resolution analysis if available.
    """
    opp, _ = await _find_opportunity_by_id(session, opportunity_id)
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

        history = await opportunity_judge.get_judgment_history(opportunity_id=opportunity_id, limit=1)
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


async def _build_context_pack(
    session: AsyncSession,
    context_type: Optional[str],
    context_id: Optional[str],
    *,
    include_news: bool = True,
    news_limit: int = 5,
) -> dict[str, Any]:
    """Build a compact context pack for AI chat and UI context inspection."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import desc, select
    from models.database import (
        NewsTradeIntent,
        NewsWorkflowFinding,
        OpportunityJudgment,
        ResolutionAnalysis,
    )

    pack: dict[str, Any] = {
        "context_type": context_type,
        "context_id": context_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "opportunity": None,
        "trader_signal": None,
        "latest_judgment": None,
        "resolution_analyses": [],
        "news_findings": [],
        "news_intents": [],
        "market_ids": [],
    }

    market_ids: list[str] = []
    opportunity = None
    opps = await shared_state.get_opportunities_from_db(session, None)
    weather_opps = await weather_shared_state.get_weather_opportunities_from_db(session)

    if context_type == "opportunity" and context_id:
        opportunity = next((o for o in opps if o.id == context_id), None)
        if opportunity is None:
            opportunity = next((o for o in weather_opps if o.id == context_id), None)
    elif context_type == "market" and context_id:
        for o in opps:
            if any(m.get("id", "") == context_id for m in o.markets):
                opportunity = o
                break
        if opportunity is None:
            for o in weather_opps:
                if any(m.get("id", "") == context_id for m in o.markets):
                    opportunity = o
                    break
    elif context_type == "trader_signal" and context_id:
        # Context_id format: "<source>:<id>" where source is "confluence" or "insider".
        # If source is omitted, search both domains.
        source_hint: Optional[str] = None
        signal_id = str(context_id)
        if ":" in signal_id:
            maybe_source, maybe_id = signal_id.split(":", 1)
            source_hint = (maybe_source or "").strip().lower() or None
            signal_id = maybe_id

        trader_signal: Optional[dict[str, Any]] = None

        if source_hint in {None, "confluence"}:
            from services.smart_wallet_pool import smart_wallet_pool

            confluence_rows = await smart_wallet_pool.get_tracked_trader_opportunities(
                limit=250,
                min_tier="WATCH",
            )
            match = next(
                (row for row in confluence_rows if str(row.get("id") or "") == signal_id),
                None,
            )
            if match:
                trader_signal = {"source": "confluence", **match}

        if trader_signal is None and source_hint in {None, "insider"}:
            from services.insider_detector import insider_detector

            insider_payload = await insider_detector.list_opportunities(
                limit=250,
                offset=0,
                min_confidence=0.0,
                direction=None,
                max_age_minutes=24 * 60,
            )
            insider_rows = insider_payload.get("opportunities", []) if isinstance(insider_payload, dict) else []
            match = next(
                (row for row in insider_rows if str(row.get("id") or "") == signal_id),
                None,
            )
            if match:
                trader_signal = {"source": "insider", **match}

        if trader_signal:
            market_id = str(trader_signal.get("market_id") or "").strip()
            if market_id:
                market_ids = [market_id]

            pack["trader_signal"] = {
                "id": trader_signal.get("id"),
                "source": trader_signal.get("source"),
                "market_id": market_id,
                "market_question": trader_signal.get("market_question"),
                "market_slug": trader_signal.get("market_slug"),
                "direction": trader_signal.get("direction"),
                "tier": trader_signal.get("tier"),
                "confidence": trader_signal.get("confidence") or trader_signal.get("conviction_score"),
                "wallet_count": trader_signal.get("wallet_count"),
                "edge_percent": trader_signal.get("edge_percent"),
                "insider_score": trader_signal.get("insider_score"),
                "cluster_count": trader_signal.get("cluster_count"),
                "signal_type": trader_signal.get("signal_type"),
                "detected_at": trader_signal.get("detected_at"),
                "last_seen_at": trader_signal.get("last_seen_at"),
                "yes_price": trader_signal.get("yes_price"),
                "no_price": trader_signal.get("no_price"),
                "price_history": (
                    trader_signal.get("price_history", [])[-20:]
                    if isinstance(trader_signal.get("price_history"), list)
                    else []
                ),
            }

    if opportunity:
        market_ids = [m.get("id", "") for m in opportunity.markets if m.get("id", "")]
        pack["opportunity"] = {
            "id": opportunity.id,
            "stable_id": opportunity.stable_id,
            "title": opportunity.title,
            "strategy": opportunity.strategy,
            "roi_percent": opportunity.roi_percent,
            "net_profit": opportunity.net_profit,
            "risk_score": opportunity.risk_score,
            "risk_factors": opportunity.risk_factors,
            "event_title": opportunity.event_title,
            "category": opportunity.category,
            "markets": [
                {
                    "id": m.get("id"),
                    "question": m.get("question"),
                    "yes_price": m.get("yes_price"),
                    "no_price": m.get("no_price"),
                    "liquidity": m.get("liquidity"),
                    "price_history": m.get("price_history", [])[-20:],
                }
                for m in opportunity.markets
            ],
        }

        judgment_q = (
            select(OpportunityJudgment)
            .where(OpportunityJudgment.opportunity_id == opportunity.id)
            .order_by(desc(OpportunityJudgment.judged_at))
            .limit(1)
        )
        judgment_row = (await session.execute(judgment_q)).scalar_one_or_none()
        if judgment_row:
            pack["latest_judgment"] = {
                "overall_score": judgment_row.overall_score,
                "profit_viability": judgment_row.profit_viability,
                "resolution_safety": judgment_row.resolution_safety,
                "execution_feasibility": judgment_row.execution_feasibility,
                "market_efficiency": judgment_row.market_efficiency,
                "recommendation": judgment_row.recommendation,
                "risk_factors": judgment_row.risk_factors or [],
                "judged_at": judgment_row.judged_at.isoformat() if judgment_row.judged_at else None,
            }

    elif context_type == "market" and context_id:
        market_ids = [context_id]

    pack["market_ids"] = market_ids

    if market_ids:
        analysis_q = (
            select(ResolutionAnalysis)
            .where(ResolutionAnalysis.market_id.in_(market_ids))
            .order_by(desc(ResolutionAnalysis.analyzed_at))
            .limit(10)
        )
        analyses = (await session.execute(analysis_q)).scalars().all()
        pack["resolution_analyses"] = [
            {
                "market_id": a.market_id,
                "clarity_score": a.clarity_score,
                "risk_score": a.risk_score,
                "confidence": a.confidence,
                "recommendation": a.recommendation,
                "summary": a.summary,
                "ambiguities": a.ambiguities or [],
                "edge_cases": a.edge_cases or [],
                "analyzed_at": a.analyzed_at.isoformat() if a.analyzed_at else None,
            }
            for a in analyses
        ]

        if include_news:
            finding_q = (
                select(NewsWorkflowFinding)
                .where(
                    NewsWorkflowFinding.market_id.in_(market_ids),
                    NewsWorkflowFinding.created_at >= datetime.now(timezone.utc) - timedelta(hours=48),
                )
                .order_by(desc(NewsWorkflowFinding.created_at))
                .limit(max(1, min(news_limit, 20)))
            )
            findings = (await session.execute(finding_q)).scalars().all()
            pack["news_findings"] = [
                {
                    "id": f.id,
                    "market_id": f.market_id,
                    "article_title": f.article_title,
                    "article_source": f.article_source,
                    "edge_percent": f.edge_percent,
                    "direction": f.direction,
                    "confidence": f.confidence,
                    "actionable": f.actionable,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in findings
            ]

            intent_q = (
                select(NewsTradeIntent)
                .where(NewsTradeIntent.market_id.in_(market_ids))
                .order_by(desc(NewsTradeIntent.created_at))
                .limit(max(1, min(news_limit, 20)))
            )
            intents = (await session.execute(intent_q)).scalars().all()
            pack["news_intents"] = [
                {
                    "id": i.id,
                    "market_id": i.market_id,
                    "direction": i.direction,
                    "entry_price": i.entry_price,
                    "edge_percent": i.edge_percent,
                    "confidence": i.confidence,
                    "status": i.status,
                    "created_at": i.created_at.isoformat() if i.created_at else None,
                }
                for i in intents
            ]

    return pack


@router.get("/ai/context-pack")
async def get_ai_context_pack(
    session: AsyncSession = Depends(get_db_session),
    context_type: str = Query(
        "general",
        description="opportunity | trader_signal | market | general",
    ),
    context_id: Optional[str] = Query(None, description="opportunity_id or market_id for the context type"),
    include_news: bool = Query(True, description="Include workflow findings/intents"),
    news_limit: int = Query(5, ge=1, le=20),
):
    """Return a compact context bundle for AI-assisted workflows."""
    return await _build_context_pack(
        session,
        context_type=context_type,
        context_id=context_id,
        include_news=include_news,
        news_limit=news_limit,
    )


class AIChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context_type: Optional[str] = None  # "opportunity", "trader_signal", "market", "general"
    context_id: Optional[str] = None  # opportunity_id or market_id
    history: list[dict] = Field(default_factory=list)  # prior messages [{role, content}]
    model: Optional[str] = None


@router.get("/ai/chat/sessions")
async def list_ai_chat_sessions(
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """List recent persistent AI chat sessions."""
    from services.ai.chat_memory import chat_memory_service

    sessions = await chat_memory_service.list_sessions(context_type=context_type, context_id=context_id, limit=limit)
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/ai/chat/sessions/{session_id}")
async def get_ai_chat_session(session_id: str):
    """Get a persistent AI chat session including messages."""
    from services.ai.chat_memory import chat_memory_service

    result = await chat_memory_service.get_session(session_id, message_limit=500)
    if not result:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return result


@router.delete("/ai/chat/sessions/{session_id}")
async def archive_ai_chat_session(session_id: str):
    """Archive a persistent AI chat session."""
    from services.ai.chat_memory import chat_memory_service

    ok = await chat_memory_service.archive_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"status": "archived", "session_id": session_id}


@router.post("/ai/chat")
async def ai_chat(
    request: AIChatRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Conversational AI copilot for the trading platform.

    Context-aware chat that understands the current page/opportunity
    the user is viewing and can answer questions, analyze markets,
    and provide trading recommendations.
    """
    try:
        from services.ai import get_llm_manager
        from services.ai.chat_memory import chat_memory_service
        from services.ai.llm_provider import LLMMessage

        manager = get_llm_manager()
        if not manager.is_available():
            raise HTTPException(
                status_code=503,
                detail="No AI provider configured. Configure an LLM provider in Settings.",
            )

        chat_session = None
        if request.session_id:
            chat_session = await chat_memory_service.get_session(request.session_id, message_limit=1)
        if chat_session is None and request.context_type and request.context_id:
            chat_session = await chat_memory_service.find_latest_for_context(request.context_type, request.context_id)
        if chat_session is None:
            chat_session = await chat_memory_service.create_session(
                context_type=request.context_type or "general",
                context_id=request.context_id,
                title=(request.message or "").strip()[:120] or "AI chat",
            )
        session_id = chat_session["session_id"]

        persisted = await chat_memory_service.get_recent_messages(session_id, limit=20)

        context_pack = await _build_context_pack(
            session,
            context_type=request.context_type,
            context_id=request.context_id,
            include_news=True,
            news_limit=5,
        )

        system_prompt = (
            "You are the AI copilot for Homerun, a Polymarket prediction market "
            "arbitrage trading platform. You help traders understand opportunities, "
            "analyze resolution criteria, assess risk, and make trading decisions.\n\n"
            "Key knowledge:\n"
            "- Polymarket uses UMA's Optimistic Oracle for resolution\n"
            "- 2% fee on net winnings\n"
            "- Strategies include: basic arb, NegRisk, mutually exclusive, contradiction, must-happen, "
            "miracle, combinatorial, settlement lag, cross-platform, bayesian cascade, liquidity vacuum, "
            "entropy arb, event-driven, temporal decay, correlation arb, market making, stat arb, BTC/ETH highfreq\n"
            "- Risk factors: resolution ambiguity, liquidity, correlation, timing\n\n"
            "Be concise, specific, and data-driven. When the user asks about a "
            "specific opportunity (including trader-signal opportunities), "
            "reference its actual data. Flag risks clearly.\n"
        )

        compact_context = {
            "context_type": context_pack.get("context_type"),
            "context_id": context_pack.get("context_id"),
            "opportunity": context_pack.get("opportunity"),
            "trader_signal": context_pack.get("trader_signal"),
            "latest_judgment": context_pack.get("latest_judgment"),
            "resolution_analyses": context_pack.get("resolution_analyses", [])[:3],
            "news_findings": context_pack.get("news_findings", [])[:3],
            "news_intents": context_pack.get("news_intents", [])[:3],
        }
        system_prompt += "\nCurrent context pack (JSON):\n" + json.dumps(compact_context, ensure_ascii=True)

        messages = [LLMMessage(role="system", content=system_prompt)]
        history_source = persisted if persisted else request.history[-10:]
        for msg in history_source:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            messages.append(LLMMessage(role=role, content=msg.get("content", "")))
        messages.append(LLMMessage(role="user", content=request.message))

        await chat_memory_service.append_message(
            session_id=session_id,
            role="user",
            content=request.message,
        )

        response = await manager.chat(
            messages=messages,
            model=request.model,
            max_tokens=1024,
            purpose="ai_chat",
            session_id=session_id,
        )

        await chat_memory_service.append_message(
            session_id=session_id,
            role="assistant",
            content=response.content or "",
            model_used=response.model or request.model,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

        return {
            "session_id": session_id,
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
