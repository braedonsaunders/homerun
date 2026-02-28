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
import re

from models.database import get_db_session
from services import shared_state
from services.weather import shared_state as weather_shared_state
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)

router = APIRouter()

_IDENTIFIER_SLUG_RE = re.compile(r"[^a-z0-9_]+")
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def _slugify_identifier(raw_value: Any, default_prefix: str = "custom") -> str:
    text = str(raw_value or "").strip().lower()
    text = _IDENTIFIER_SLUG_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = default_prefix
    if text[0].isdigit():
        text = f"s_{text}"
    if len(text) < 3:
        text = f"{default_prefix}_{text}".strip("_")
    text = text[:50].rstrip("_")
    if not text:
        text = default_prefix
    if not text[0].isalpha():
        text = f"s{text}"
    if not text[-1].isalnum():
        text = f"{text[:-1]}x" if text[:-1] else f"{default_prefix}x"
    return text[:50]


def _coerce_json_object(value: Any, fallback: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return dict(fallback or {})


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = str(raw or "").strip()
        if not item:
            continue
        normalized = item.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(item)
    return out


def _extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    candidates: list[str] = [text]
    for match in _JSON_BLOCK_RE.findall(text):
        block = str(match or "").strip()
        if block:
            candidates.append(block)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start < 0 or end <= start:
            continue
        fragment = normalized[start : end + 1].strip()
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)
        try:
            parsed_fragment = json.loads(fragment)
            if isinstance(parsed_fragment, dict):
                return parsed_fragment
        except Exception:
            continue
    return None


def _compact_strategy_docs_for_prompt(docs: dict[str, Any]) -> dict[str, Any]:
    strategy_sdk = _coerce_json_object(docs.get("strategy_sdk"))
    overview = _coerce_json_object(docs.get("overview"))
    detect_phase = _coerce_json_object(docs.get("detect_phase"))
    evaluate_phase = _coerce_json_object(docs.get("evaluate_phase"))
    exit_phase = _coerce_json_object(docs.get("exit_phase"))
    imports = _coerce_json_object(docs.get("imports"))
    data_source_sdk = _coerce_json_object(docs.get("data_source_sdk"))
    return {
        "overview": {
            "summary": overview.get("summary"),
            "strategy_types": _coerce_json_object(overview.get("strategy_types")),
        },
        "detect_phase": {
            "methods": _coerce_json_object(detect_phase.get("methods")),
            "parameters": _coerce_json_object(detect_phase.get("parameters")),
        },
        "evaluate_phase": {
            "method": evaluate_phase.get("method"),
            "signal_object": _coerce_json_object(evaluate_phase.get("signal_object")).get("fields"),
            "context_object": _coerce_json_object(evaluate_phase.get("context_object")).get("fields"),
            "return_value": _coerce_json_object(evaluate_phase.get("return_value")).get("decision_values"),
        },
        "exit_phase": {
            "method": exit_phase.get("method"),
            "position_object": _coerce_json_object(exit_phase.get("position_object")).get("fields"),
            "market_state_object": _coerce_json_object(exit_phase.get("market_state_object")).get("fields"),
            "return_value": _coerce_json_object(exit_phase.get("return_value")).get("action_values"),
        },
        "strategy_sdk": {
            "configuration_helpers": _coerce_json_object(strategy_sdk.get("configuration_helpers")),
            "validation_helpers": _coerce_json_object(strategy_sdk.get("validation_helpers")),
            "market_and_execution_helpers": _coerce_json_object(strategy_sdk.get("market_and_execution_helpers")),
            "llm_and_news_helpers": _coerce_json_object(strategy_sdk.get("llm_and_news_helpers")),
            "trader_data_helpers": _coerce_json_object(strategy_sdk.get("trader_data_helpers")),
        },
        "imports": {
            "app_modules": _coerce_json_object(imports.get("app_modules")),
            "standard_library": imports.get("standard_library") if isinstance(imports.get("standard_library"), list) else [],
            "third_party": _coerce_json_object(imports.get("third_party")),
            "blocked": _coerce_json_object(imports.get("blocked")),
        },
        "data_source_sdk": {
            "read_methods": _coerce_json_object(data_source_sdk.get("read_methods")),
            "management_methods": _coerce_json_object(data_source_sdk.get("management_methods")),
            "strategy_sdk_wrappers": _coerce_json_object(data_source_sdk.get("strategy_sdk_wrappers")),
            "guidance": data_source_sdk.get("guidance") if isinstance(data_source_sdk.get("guidance"), list) else [],
        },
    }


def _compact_data_source_docs_for_prompt(docs: dict[str, Any]) -> dict[str, Any]:
    overview = _coerce_json_object(docs.get("overview"))
    base_data_source = _coerce_json_object(docs.get("base_data_source"))
    record_contract = _coerce_json_object(docs.get("record_contract"))
    sdk_reference = _coerce_json_object(docs.get("sdk_reference"))
    imports = _coerce_json_object(docs.get("imports"))
    return {
        "overview": {
            "summary": overview.get("summary"),
            "concepts": _coerce_json_object(overview.get("concepts")),
            "three_stage_lifecycle": _coerce_json_object(overview.get("three_stage_lifecycle")),
        },
        "base_data_source": {
            "class_attributes": _coerce_json_object(base_data_source.get("class_attributes")),
            "methods": _coerce_json_object(base_data_source.get("methods")),
            "helpers": _coerce_json_object(base_data_source.get("helpers")),
        },
        "record_contract": {
            "description": record_contract.get("description"),
            "fields": _coerce_json_object(record_contract.get("fields")),
            "normalization_rules": record_contract.get("normalization_rules")
            if isinstance(record_contract.get("normalization_rules"), list)
            else [],
        },
        "sdk_reference": {
            "read_methods": _coerce_json_object(sdk_reference.get("read_methods")),
            "management_methods": _coerce_json_object(sdk_reference.get("management_methods")),
            "strategy_sdk_access": _coerce_json_object(sdk_reference.get("strategy_sdk_access")),
        },
        "imports": {
            "app_modules": _coerce_json_object(imports.get("app_modules")),
            "standard_library": imports.get("standard_library") if isinstance(imports.get("standard_library"), list) else [],
            "third_party": _coerce_json_object(imports.get("third_party")),
        },
    }


def _strategy_class_name_from_slug(slug: str) -> str:
    tokens = [part for part in str(slug or "").strip().split("_") if part]
    base = "".join(token[:1].upper() + token[1:] for token in tokens)
    if not base:
        base = "Custom"
    if not re.match(r"^[A-Za-z_]", base):
        base = f"S{base}"
    if not base.endswith("Strategy"):
        base = f"{base}Strategy"
    return base


def _data_source_class_name_from_slug(slug: str) -> str:
    tokens = [part for part in str(slug or "").strip().split("_") if part]
    base = "".join(token[:1].upper() + token[1:] for token in tokens)
    if not base:
        base = "Custom"
    if not re.match(r"^[A-Za-z_]", base):
        base = f"S{base}"
    if not base.endswith("Source"):
        base = f"{base}Source"
    return base


def _python_string_literal(value: str) -> str:
    escaped = (
        str(value or "")
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _render_strategy_template_source(
    template: str,
    *,
    slug: str,
    name: str,
    description: str,
    source_key: str,
) -> str:
    class_name = _strategy_class_name_from_slug(slug)
    rendered = str(template or "")
    rendered = rendered.replace("MyCustomStrategy", class_name)
    rendered = re.sub(r'^\s*name\s*=\s*".*"$', f"    name = {_python_string_literal(name)}", rendered, flags=re.MULTILINE)
    rendered = re.sub(
        r'^\s*description\s*=\s*".*"$',
        f"    description = {_python_string_literal(description)}",
        rendered,
        flags=re.MULTILINE,
    )
    rendered = re.sub(
        r'^\s*source_key\s*=\s*".*"$',
        f"    source_key = {_python_string_literal(source_key)}",
        rendered,
        flags=re.MULTILINE,
    )
    rendered = re.sub(
        r'^\s*worker_affinity\s*=\s*".*"$',
        f"    worker_affinity = {_python_string_literal(source_key)}",
        rendered,
        flags=re.MULTILINE,
    )
    return rendered


def _render_data_source_template_source(
    template: str,
    *,
    slug: str,
    name: str,
    description: str,
) -> str:
    class_name = _data_source_class_name_from_slug(slug)
    rendered = str(template or "")
    rendered = rendered.replace("MyCustomSource", class_name)
    rendered = re.sub(r'^\s*name\s*=\s*".*"$', f"    name = {_python_string_literal(name)}", rendered, flags=re.MULTILINE)
    rendered = re.sub(
        r'^\s*description\s*=\s*".*"$',
        f"    description = {_python_string_literal(description)}",
        rendered,
        flags=re.MULTILINE,
    )
    return rendered


async def _find_opportunity_by_id(session: AsyncSession, opportunity_id: str) -> tuple[Any, Optional[str]]:
    """Find opportunity across scanner and weather snapshots."""
    scanner_opps = await shared_state.get_opportunities_from_db(session, None)
    scanner_hit = next((o for o in scanner_opps if o.id == opportunity_id), None)
    if scanner_hit:
        return scanner_hit, "scanner"

    # Weather opportunities panel can display both executable opportunities and
    # report-only findings; AI lookup must search the same visible set.
    weather_opps = await weather_shared_state.get_weather_opportunities_from_db(session, include_report_only=True)
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
    opportunity_ids: list[str] = Field(default_factory=list)  # empty = all unjudged
    force: bool = False  # if True, re-analyze even if already judged


@router.post("/ai/judge/opportunities/bulk")
async def judge_opportunities_bulk(
    request: JudgeBulkRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Judge multiple opportunities concurrently."""
    from services.ai.opportunity_judge import opportunity_judge
    from models.opportunity import AIAnalysis
    from datetime import datetime

    scanner_opportunities = await shared_state.get_opportunities_from_db(
        session,
        None,
        source="all",
    )
    weather_opportunities = await weather_shared_state.get_weather_opportunities_from_db(
        session,
        include_report_only=True,
    )
    scanner_by_id = {opp.id: opp for opp in scanner_opportunities}
    weather_by_id = {opp.id: opp for opp in weather_opportunities}

    combined_by_id: dict[str, Any] = {}
    combined_by_id.update(scanner_by_id)
    for opportunity_id, opp in weather_by_id.items():
        if opportunity_id not in combined_by_id:
            combined_by_id[opportunity_id] = opp

    targets: list[Any]
    if request.opportunity_ids:
        seen_ids: set[str] = set()
        requested_ids: list[str] = []
        for opportunity_id in request.opportunity_ids:
            if opportunity_id in seen_ids or opportunity_id not in combined_by_id:
                continue
            seen_ids.add(opportunity_id)
            requested_ids.append(opportunity_id)
        targets = [combined_by_id[opportunity_id] for opportunity_id in requested_ids]
    elif request.force:
        targets = list(combined_by_id.values())
    else:
        targets = [
            opp for opp in combined_by_id.values() if not opp.ai_analysis or opp.ai_analysis.recommendation == "pending"
        ]

    if not targets:
        return {"judged": 0, "errors": [], "total_requested": 0}

    results = await opportunity_judge.judge_batch(targets)
    errors = []
    judged = 0

    for opp, result in zip(targets, results):
        try:
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
            judged += 1
            try:
                if opp.id in scanner_by_id:
                    await shared_state.update_opportunity_ai_analysis_in_snapshot(
                        session=session,
                        opportunity_id=opp.id,
                        stable_id=opp.stable_id,
                        ai_analysis=opp.ai_analysis.model_dump(mode="json"),
                    )
                if opp.id in weather_by_id:
                    await weather_shared_state.update_weather_opportunity_ai_analysis_in_snapshot(
                        session=session,
                        opportunity_id=opp.id,
                        stable_id=opp.stable_id,
                        ai_analysis=opp.ai_analysis.model_dump(mode="json"),
                    )
            except Exception as snapshot_error:
                logger.warning(
                    "Failed to persist inline ai_analysis for %s: %s",
                    opp.id,
                    snapshot_error,
                )
        except Exception as e:
            errors.append({"opportunity_id": opp.id, "error": str(e)})

    return {"judged": judged, "errors": errors, "total_requested": len(targets)}


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
        DataSource,
        NewsTradeIntent,
        NewsWorkflowFinding,
        OpportunityJudgment,
        ResolutionAnalysis,
        Strategy,
        StrategyVersion,
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
        "strategy": None,
        "strategy_versions": [],
        "strategy_docs": {},
        "available_data_sources": [],
        "data_source": None,
        "data_source_docs": {},
        "available_source_keys": [],
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
        source_hint: Optional[str] = None
        signal_id = str(context_id)
        if ":" in signal_id:
            maybe_source, maybe_id = signal_id.split(":", 1)
            source_hint = (maybe_source or "").strip().lower() or None
            signal_id = maybe_id
        if source_hint in {None, "traders", "confluence"}:
            trader_opps = await shared_state.get_opportunities_from_db(
                session,
                None,
                source="traders",
            )
            match = next(
                (opp for opp in trader_opps if str(opp.id or "") == signal_id or str(opp.stable_id or "") == signal_id),
                None,
            )
        else:
            match = None

        if match:
            market = match.markets[0] if match.markets else {}
            strategy_context = match.strategy_context if isinstance(match.strategy_context, dict) else {}
            firehose = strategy_context.get("firehose") if isinstance(strategy_context.get("firehose"), dict) else {}
            market_id = str(market.get("id") or "").strip()
            if market_id:
                market_ids = [market_id]

            pack["trader_signal"] = {
                "id": match.id,
                "source": "traders",
                "market_id": market_id,
                "market_question": market.get("question"),
                "market_slug": market.get("slug"),
                "direction": strategy_context.get("side"),
                "tier": strategy_context.get("tier"),
                "confidence": strategy_context.get("confidence") or match.confidence,
                "wallet_count": strategy_context.get("wallet_count"),
                "edge_percent": strategy_context.get("edge_percent") or match.roi_percent,
                "cluster_count": strategy_context.get("wallet_count"),
                "signal_type": firehose.get("signal_type"),
                "detected_at": match.detected_at.isoformat() if match.detected_at else None,
                "last_seen_at": match.last_seen_at.isoformat() if match.last_seen_at else None,
                "yes_price": market.get("yes_price"),
                "no_price": market.get("no_price"),
                "price_history": (
                    market.get("price_history", [])[-20:] if isinstance(market.get("price_history"), list) else []
                ),
            }
    elif context_type == "strategy" and context_id:
        strategy_identifier = str(context_id or "").strip()
        strategy_row = await session.get(Strategy, strategy_identifier) if strategy_identifier else None
        if strategy_row is None and strategy_identifier:
            strategy_row = (
                await session.execute(
                    select(Strategy).where(Strategy.slug == strategy_identifier.lower()).limit(1)
                )
            ).scalar_one_or_none()

        if strategy_row is not None:
            strategy_source_key = str(strategy_row.source_key or "scanner").strip().lower() or "scanner"
            pack["strategy"] = {
                "id": strategy_row.id,
                "slug": strategy_row.slug,
                "source_key": strategy_source_key,
                "name": strategy_row.name,
                "description": strategy_row.description,
                "source_code": str(strategy_row.source_code or ""),
                "class_name": strategy_row.class_name,
                "enabled": bool(strategy_row.enabled),
                "status": strategy_row.status,
                "error_message": strategy_row.error_message,
                "version": int(strategy_row.version or 1),
                "config": dict(strategy_row.config or {}),
                "config_schema": dict(strategy_row.config_schema or {}),
                "aliases": list(strategy_row.aliases or []),
            }

            version_rows = (
                await session.execute(
                    select(StrategyVersion)
                    .where(StrategyVersion.strategy_id == strategy_row.id)
                    .order_by(StrategyVersion.version.desc())
                    .limit(25)
                )
            ).scalars().all()
            pack["strategy_versions"] = [
                {
                    "version": int(row.version or 0),
                    "is_latest": bool(row.is_latest),
                    "reason": row.reason,
                    "created_by": row.created_by,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in version_rows
            ]

            source_rows = (
                await session.execute(
                    select(DataSource)
                    .where(DataSource.enabled == True)  # noqa: E712
                    .order_by(DataSource.source_key.asc(), DataSource.sort_order.asc(), DataSource.slug.asc())
                    .limit(500)
                )
            ).scalars().all()
            pack["available_data_sources"] = [
                {
                    "id": row.id,
                    "slug": row.slug,
                    "source_key": row.source_key,
                    "source_kind": row.source_kind,
                    "name": row.name,
                    "description": row.description,
                    "enabled": bool(row.enabled),
                    "status": row.status,
                }
                for row in source_rows
            ]
            pack["available_source_keys"] = sorted(
                {str(row.source_key or "").strip().lower() for row in source_rows if str(row.source_key or "").strip()}
            )
            try:
                from api.routes_strategies import get_unified_docs

                raw_docs = await get_unified_docs()
                pack["strategy_docs"] = _compact_strategy_docs_for_prompt(_coerce_json_object(raw_docs))
            except Exception as exc:
                logger.warning("Failed to load strategy docs for AI context pack", exc_info=exc)
    elif context_type == "data_source" and context_id:
        source_identifier = str(context_id or "").strip()
        source_row = await session.get(DataSource, source_identifier) if source_identifier else None
        if source_row is None and source_identifier:
            source_row = (
                await session.execute(
                    select(DataSource).where(DataSource.slug == source_identifier.lower()).limit(1)
                )
            ).scalar_one_or_none()

        if source_row is not None:
            pack["data_source"] = {
                "id": source_row.id,
                "slug": source_row.slug,
                "source_key": source_row.source_key,
                "source_kind": source_row.source_kind,
                "name": source_row.name,
                "description": source_row.description,
                "source_code": str(source_row.source_code or ""),
                "class_name": source_row.class_name,
                "enabled": bool(source_row.enabled),
                "status": source_row.status,
                "error_message": source_row.error_message,
                "version": int(source_row.version or 1),
                "retention": dict(source_row.retention or {}),
                "config": dict(source_row.config or {}),
                "config_schema": dict(source_row.config_schema or {}),
            }
            source_rows = (
                await session.execute(
                    select(DataSource)
                    .order_by(DataSource.source_key.asc(), DataSource.sort_order.asc(), DataSource.slug.asc())
                    .limit(500)
                )
            ).scalars().all()
            pack["available_data_sources"] = [
                {
                    "id": row.id,
                    "slug": row.slug,
                    "source_key": row.source_key,
                    "source_kind": row.source_kind,
                    "name": row.name,
                    "enabled": bool(row.enabled),
                    "status": row.status,
                }
                for row in source_rows
            ]
            pack["available_source_keys"] = sorted(
                {str(row.source_key or "").strip().lower() for row in source_rows if str(row.source_key or "").strip()}
            )
            try:
                from api.routes_data_sources import get_data_source_docs

                raw_docs = await get_data_source_docs()
                pack["data_source_docs"] = _compact_data_source_docs_for_prompt(_coerce_json_object(raw_docs))
            except Exception as exc:
                logger.warning("Failed to load data source docs for AI context pack", exc_info=exc)

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
        description="opportunity | trader_signal | market | strategy | data_source | general",
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
    context_type: Optional[str] = None  # "opportunity", "trader_signal", "market", "strategy", "data_source", "general"
    context_id: Optional[str] = None  # context object id
    history: list[dict] = Field(default_factory=list)  # prior messages [{role, content}]
    model: Optional[str] = None
    allow_actions: bool = True


class AIGenerateStrategyDraftRequest(BaseModel):
    description: str = Field(..., min_length=8, max_length=12000)
    source_key: Optional[str] = Field(default=None, min_length=2, max_length=64)
    model: Optional[str] = None


class AIGenerateDataSourceDraftRequest(BaseModel):
    description: str = Field(..., min_length=8, max_length=12000)
    source_key: Optional[str] = Field(default=None, min_length=2, max_length=64)
    source_kind: Optional[str] = Field(default=None, min_length=2, max_length=32)
    model: Optional[str] = None


@router.post("/ai/generate/strategy-draft")
async def generate_strategy_draft_with_ai(
    request: AIGenerateStrategyDraftRequest,
    session: AsyncSession = Depends(get_db_session),
):
    from sqlalchemy import select

    from api.routes_strategies import (
        _detect_capabilities,
        _infer_strategy_type,
        _merge_config_schemas,
        _normalize_strategy_config_for_source,
        get_unified_docs,
    )
    from models.database import DataSource, Strategy
    from services.ai import get_llm_manager
    from services.ai.llm_provider import LLMMessage
    from services.strategy_loader import STRATEGY_TEMPLATE, validate_strategy_source
    from services.strategy_sdk import StrategySDK

    manager = get_llm_manager()
    if not manager.is_available():
        raise HTTPException(
            status_code=503,
            detail="No AI provider configured. Configure an LLM provider in Settings.",
        )

    strategy_rows = (
        await session.execute(
            select(Strategy).order_by(Strategy.source_key.asc(), Strategy.sort_order.asc(), Strategy.slug.asc())
        )
    ).scalars().all()
    source_rows = (
        await session.execute(
            select(DataSource).where(DataSource.enabled == True).order_by(DataSource.source_key.asc(), DataSource.slug.asc())  # noqa: E712
        )
    ).scalars().all()

    strategy_source_keys = sorted(
        {
            "scanner",
            "news",
            "weather",
            "crypto",
            "traders",
            "events",
            *[str(row.source_key or "").strip().lower() for row in strategy_rows if str(row.source_key or "").strip()],
        }
    )
    source_key_requested = str(request.source_key or "").strip().lower()
    if source_key_requested:
        strategy_source_keys = sorted({*strategy_source_keys, source_key_requested})

    docs_payload = _compact_strategy_docs_for_prompt(_coerce_json_object(await get_unified_docs()))
    generation_context = {
        "strategy_source_keys": strategy_source_keys,
        "existing_strategy_slugs": [str(row.slug or "").strip().lower() for row in strategy_rows[:400]],
        "available_data_sources": [
            {
                "slug": row.slug,
                "source_key": row.source_key,
                "source_kind": row.source_kind,
                "name": row.name,
            }
            for row in source_rows[:400]
        ],
        "strategy_template": STRATEGY_TEMPLATE,
        "strategy_docs": docs_payload,
    }

    system_prompt = (
        "You generate production-ready Homerun strategy drafts. "
        "Return STRICT JSON with keys: "
        "name, slug, source_key, description, source_code, config, config_schema, aliases. "
        "Rules: source_code must define a class extending BaseStrategy; include detect/detect_async and/or evaluate; "
        "only use allowed imports from docs; keep business logic complete; no placeholders, no TODOs, no pass stubs."
    )
    user_payload = {
        "task": "Generate a new strategy draft from user description.",
        "user_description": request.description,
        "preferred_source_key": source_key_requested or None,
        "context": generation_context,
    }

    response = await manager.chat(
        messages=[
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=json.dumps(user_payload, ensure_ascii=True)),
        ],
        model=request.model,
        max_tokens=3200,
        purpose="ai_strategy_draft_generation",
    )

    parsed = _extract_json_payload(response.content or "")
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=422,
            detail="AI returned a non-JSON strategy draft. Retry with a more specific description.",
        )

    name = str(parsed.get("name") or "").strip() or "AI Generated Strategy"
    slug = _slugify_identifier(parsed.get("slug") or name, default_prefix="strategy")
    source_key = source_key_requested or str(parsed.get("source_key") or "scanner").strip().lower() or "scanner"
    description = str(parsed.get("description") or "").strip() or str(request.description).strip()
    source_code = str(parsed.get("source_code") or "").strip()
    if not source_code:
        source_code = _render_strategy_template_source(
            STRATEGY_TEMPLATE,
            slug=slug,
            name=name,
            description=description,
            source_key=source_key,
        )

    parsed_config = _coerce_json_object(parsed.get("config"))
    config = _normalize_strategy_config_for_source(source_key, parsed_config)
    config_schema = _merge_config_schemas(
        _coerce_json_object(parsed.get("config_schema"), {"param_fields": []}),
        StrategySDK.strategy_retention_config_schema(),
    )
    aliases = _coerce_string_list(parsed.get("aliases"))

    validation = validate_strategy_source(source_code)
    repaired = False
    if not bool(validation.get("valid")):
        repair_payload = {
            "task": "Repair this strategy source to satisfy Homerun validation.",
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "draft": {
                "name": name,
                "slug": slug,
                "source_key": source_key,
                "description": description,
                "source_code": source_code,
            },
            "required_output": {"source_code": "valid Python strategy source extending BaseStrategy"},
        }
        repair_response = await manager.chat(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Return STRICT JSON with one key: source_code. "
                        "Produce complete valid strategy source code only."
                    ),
                ),
                LLMMessage(role="user", content=json.dumps(repair_payload, ensure_ascii=True)),
            ],
            model=request.model,
            max_tokens=3200,
            purpose="ai_strategy_draft_repair",
        )
        repaired_payload = _extract_json_payload(repair_response.content or "")
        repaired_source = str((repaired_payload or {}).get("source_code") or "").strip()
        if repaired_source:
            source_code = repaired_source
            validation = validate_strategy_source(source_code)
            repaired = True

    capabilities = validation.get("capabilities") if isinstance(validation.get("capabilities"), dict) else None
    if not capabilities:
        capabilities = _detect_capabilities(source_code)
    inferred_type = _infer_strategy_type(capabilities)
    class_name = str(validation.get("class_name") or "").strip() or _strategy_class_name_from_slug(slug)

    return {
        "name": name,
        "slug": slug,
        "source_key": source_key,
        "description": description,
        "source_code": source_code,
        "class_name": class_name,
        "config": config,
        "config_schema": config_schema,
        "aliases": aliases,
        "validation": {
            "valid": bool(validation.get("valid")),
            "class_name": validation.get("class_name"),
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "capabilities": capabilities,
            "inferred_type": inferred_type,
        },
        "used_repair_pass": repaired,
        "model": response.model or "",
        "tokens_used": {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
        },
    }


@router.post("/ai/generate/data-source-draft")
async def generate_data_source_draft_with_ai(
    request: AIGenerateDataSourceDraftRequest,
    session: AsyncSession = Depends(get_db_session),
):
    from sqlalchemy import select

    from api.routes_data_sources import (
        _normalize_source_kind,
        _normalize_source_key,
        _resolve_retention_policy,
        get_data_source_docs,
        get_data_source_template,
    )
    from models.database import DataSource
    from services.ai import get_llm_manager
    from services.ai.llm_provider import LLMMessage
    from services.data_source_catalog import build_data_source_source_code
    from services.data_source_loader import DATA_SOURCE_TEMPLATE, validate_data_source_source

    manager = get_llm_manager()
    if not manager.is_available():
        raise HTTPException(
            status_code=503,
            detail="No AI provider configured. Configure an LLM provider in Settings.",
        )

    template_payload = _coerce_json_object(await get_data_source_template())
    presets = template_payload.get("presets") if isinstance(template_payload.get("presets"), list) else []
    docs_payload = _compact_data_source_docs_for_prompt(_coerce_json_object(await get_data_source_docs()))
    source_rows = (
        await session.execute(
            select(DataSource).order_by(DataSource.source_key.asc(), DataSource.sort_order.asc(), DataSource.slug.asc())
        )
    ).scalars().all()

    source_key_requested = str(request.source_key or "").strip().lower()
    source_kind_requested = str(request.source_kind or "").strip().lower()
    generation_context = {
        "available_source_keys": sorted(
            {
                "custom",
                "stories",
                "events",
                *[str(row.source_key or "").strip().lower() for row in source_rows if str(row.source_key or "").strip()],
            }
        ),
        "available_source_kinds": ["python", "rss", "rest_api", "twitter"],
        "existing_source_slugs": [str(row.slug or "").strip().lower() for row in source_rows[:500]],
        "presets": presets,
        "template": template_payload.get("template") or DATA_SOURCE_TEMPLATE,
        "docs": docs_payload,
    }

    system_prompt = (
        "You generate production-ready Homerun data source drafts. "
        "Return STRICT JSON with keys: "
        "name, slug, source_key, source_kind, description, source_code, retention, config, config_schema. "
        "Rules: source_code must define a class extending BaseDataSource and implement fetch() or fetch_async(); "
        "no placeholders or TODO stubs; output normalized record contract-compatible logic."
    )
    user_payload = {
        "task": "Generate a new data source draft from user description.",
        "user_description": request.description,
        "preferred_source_key": source_key_requested or None,
        "preferred_source_kind": source_kind_requested or None,
        "context": generation_context,
    }
    response = await manager.chat(
        messages=[
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=json.dumps(user_payload, ensure_ascii=True)),
        ],
        model=request.model,
        max_tokens=3200,
        purpose="ai_data_source_draft_generation",
    )

    parsed = _extract_json_payload(response.content or "")
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=422,
            detail="AI returned a non-JSON data source draft. Retry with a more specific description.",
        )

    name = str(parsed.get("name") or "").strip() or "AI Generated Data Source"
    slug = _slugify_identifier(parsed.get("slug") or name, default_prefix="source")
    source_key = _normalize_source_key(source_key_requested or parsed.get("source_key") or "custom", "custom")
    source_kind = _normalize_source_kind(source_kind_requested or parsed.get("source_kind") or "python", "python")
    description = str(parsed.get("description") or "").strip() or str(request.description).strip()

    config = _coerce_json_object(parsed.get("config"))
    config_schema = _coerce_json_object(parsed.get("config_schema"), {"param_fields": []})
    source_code = str(parsed.get("source_code") or "").strip()

    if not source_code:
        if source_kind in {"rss", "rest_api", "twitter"}:
            source_code = build_data_source_source_code(
                source_kind=source_kind,
                slug=slug,
                name=name,
                description=description,
                config=config,
                include_seed_marker=False,
            )
        else:
            source_code = _render_data_source_template_source(
                template_payload.get("template") or DATA_SOURCE_TEMPLATE,
                slug=slug,
                name=name,
                description=description,
            )

    validation = validate_data_source_source(source_code)
    repaired = False
    if not bool(validation.get("valid")):
        repair_payload = {
            "task": "Repair this data source source code to satisfy Homerun validation.",
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "draft": {
                "name": name,
                "slug": slug,
                "source_key": source_key,
                "source_kind": source_kind,
                "description": description,
                "source_code": source_code,
                "config": config,
            },
            "required_output": {"source_code": "valid Python source extending BaseDataSource"},
        }
        repair_response = await manager.chat(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Return STRICT JSON with one key: source_code. "
                        "Produce complete valid data source code only."
                    ),
                ),
                LLMMessage(role="user", content=json.dumps(repair_payload, ensure_ascii=True)),
            ],
            model=request.model,
            max_tokens=3200,
            purpose="ai_data_source_draft_repair",
        )
        repaired_payload = _extract_json_payload(repair_response.content or "")
        repaired_source = str((repaired_payload or {}).get("source_code") or "").strip()
        if repaired_source:
            source_code = repaired_source
            validation = validate_data_source_source(source_code)
            repaired = True

    retention = _resolve_retention_policy(
        parsed.get("retention"),
        slug=slug,
        source_key=source_key,
        source_kind=source_kind,
        strict=False,
    )
    class_name = str(validation.get("class_name") or "").strip() or _data_source_class_name_from_slug(slug)

    return {
        "name": name,
        "slug": slug,
        "source_key": source_key,
        "source_kind": source_kind,
        "description": description,
        "source_code": source_code,
        "class_name": class_name,
        "retention": retention,
        "config": config,
        "config_schema": config_schema,
        "validation": {
            "valid": bool(validation.get("valid")),
            "class_name": validation.get("class_name"),
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "capabilities": validation.get("capabilities") or {},
        },
        "used_repair_pass": repaired,
        "model": response.model or "",
        "tokens_used": {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
        },
    }


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


_EDIT_INTENT_RE = re.compile(
    r"\b(apply|update|change|modify|edit|rewrite|patch|save|implement|refactor|add|remove|tweak|adjust)\b",
    re.IGNORECASE,
)


def _user_requests_direct_change(message: str) -> bool:
    return bool(_EDIT_INTENT_RE.search(str(message or "")))


def _parse_chat_content_and_actions(raw_content: str) -> tuple[str, list[dict[str, Any]]]:
    text = str(raw_content or "").strip()
    parsed = _extract_json_payload(text)
    if not isinstance(parsed, dict):
        return text, []
    message = str(parsed.get("message") or parsed.get("response") or "").strip()
    actions_raw = parsed.get("actions")
    actions = [item for item in actions_raw if isinstance(item, dict)] if isinstance(actions_raw, list) else []
    return message or text, actions


def _http_detail_to_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        if isinstance(detail.get("message"), str):
            return detail["message"]
        errors = detail.get("errors")
        if isinstance(errors, list):
            return "; ".join(str(item) for item in errors if str(item or "").strip())
        return json.dumps(detail, ensure_ascii=True)
    return str(detail)


async def _resolve_strategy_identifier_to_id(session: AsyncSession, identifier: Any) -> Optional[str]:
    from sqlalchemy import select
    from models.database import Strategy

    raw = str(identifier or "").strip()
    if not raw:
        return None
    by_id = await session.get(Strategy, raw)
    if by_id is not None:
        return str(by_id.id)
    by_slug = (
        await session.execute(
            select(Strategy).where(Strategy.slug == raw.lower()).limit(1)
        )
    ).scalar_one_or_none()
    if by_slug is not None:
        return str(by_slug.id)
    return None


async def _resolve_data_source_identifier_to_id(session: AsyncSession, identifier: Any) -> Optional[str]:
    from sqlalchemy import select
    from models.database import DataSource

    raw = str(identifier or "").strip()
    if not raw:
        return None
    by_id = await session.get(DataSource, raw)
    if by_id is not None:
        return str(by_id.id)
    by_slug = (
        await session.execute(
            select(DataSource).where(DataSource.slug == raw.lower()).limit(1)
        )
    ).scalar_one_or_none()
    if by_slug is not None:
        return str(by_slug.id)
    return None


async def _apply_ai_chat_actions(
    session: AsyncSession,
    actions: list[dict[str, Any]],
    *,
    context_type: Optional[str],
    context_id: Optional[str],
) -> dict[str, list[dict[str, Any]]]:
    from api.routes_data_sources import DataSourceUpdateRequest, update_data_source
    from api.routes_strategies import UnifiedStrategyUpdateRequest, update_strategy

    applied: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, action in enumerate(actions[:5]):
        action_type = str(action.get("type") or "").strip().lower()
        if not action_type:
            errors.append({"index": index, "error": "Action type is required."})
            continue

        try:
            if action_type in {"update_strategy", "apply_strategy_update"}:
                identifier = action.get("strategy_id") or action.get("id")
                if not identifier and context_type == "strategy":
                    identifier = context_id
                strategy_id = await _resolve_strategy_identifier_to_id(session, identifier)
                if not strategy_id:
                    raise RuntimeError("Strategy target not found for action.")

                changes = _coerce_json_object(action.get("changes"))
                payload: dict[str, Any] = {"unlock_system": True}
                if "slug" in changes:
                    payload["slug"] = str(changes.get("slug") or "").strip()
                if "source_key" in changes:
                    payload["source_key"] = str(changes.get("source_key") or "").strip().lower()
                if "name" in changes:
                    payload["name"] = str(changes.get("name") or "").strip()
                if "description" in changes:
                    payload["description"] = str(changes.get("description") or "").strip()
                if "source_code" in changes:
                    payload["source_code"] = str(changes.get("source_code") or "")
                if "config" in changes:
                    payload["config"] = _coerce_json_object(changes.get("config"))
                if "config_schema" in changes:
                    payload["config_schema"] = _coerce_json_object(changes.get("config_schema"))
                if "enabled" in changes:
                    payload["enabled"] = bool(changes.get("enabled"))

                if len(payload) <= 1:
                    raise RuntimeError("No valid strategy update fields were provided.")

                req = UnifiedStrategyUpdateRequest(**payload)
                updated = await update_strategy(strategy_id, req)
                applied.append(
                    {
                        "type": "update_strategy",
                        "strategy_id": strategy_id,
                        "slug": updated.get("slug"),
                        "version": updated.get("version"),
                        "status": updated.get("status"),
                    }
                )
                continue

            if action_type in {"update_data_source", "apply_data_source_update"}:
                identifier = action.get("data_source_id") or action.get("source_id") or action.get("id")
                if not identifier and context_type == "data_source":
                    identifier = context_id
                source_id = await _resolve_data_source_identifier_to_id(session, identifier)
                if not source_id:
                    raise RuntimeError("Data source target not found for action.")

                changes = _coerce_json_object(action.get("changes"))
                payload: dict[str, Any] = {"unlock_system": True}
                if "slug" in changes:
                    payload["slug"] = str(changes.get("slug") or "").strip()
                if "source_key" in changes:
                    payload["source_key"] = str(changes.get("source_key") or "").strip().lower()
                if "source_kind" in changes:
                    payload["source_kind"] = str(changes.get("source_kind") or "").strip().lower()
                if "name" in changes:
                    payload["name"] = str(changes.get("name") or "").strip()
                if "description" in changes:
                    payload["description"] = str(changes.get("description") or "").strip()
                if "source_code" in changes:
                    payload["source_code"] = str(changes.get("source_code") or "")
                if "retention" in changes:
                    payload["retention"] = _coerce_json_object(changes.get("retention"))
                if "config" in changes:
                    payload["config"] = _coerce_json_object(changes.get("config"))
                if "config_schema" in changes:
                    payload["config_schema"] = _coerce_json_object(changes.get("config_schema"))
                if "enabled" in changes:
                    payload["enabled"] = bool(changes.get("enabled"))

                if len(payload) <= 1:
                    raise RuntimeError("No valid data-source update fields were provided.")

                req = DataSourceUpdateRequest(**payload)
                updated = await update_data_source(source_id, req)
                applied.append(
                    {
                        "type": "update_data_source",
                        "data_source_id": source_id,
                        "slug": updated.get("slug"),
                        "version": updated.get("version"),
                        "status": updated.get("status"),
                    }
                )
                continue

            errors.append({"index": index, "type": action_type, "error": f"Unsupported action type '{action_type}'."})
        except HTTPException as exc:
            errors.append(
                {
                    "index": index,
                    "type": action_type,
                    "error": _http_detail_to_text(exc.detail),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "type": action_type,
                    "error": str(exc),
                }
            )

    return {"applied": applied, "errors": errors}


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

        context_type = str(request.context_type or "general").strip().lower() or "general"
        allow_actions = bool(request.allow_actions)
        can_apply_actions = (
            allow_actions
            and context_type in {"strategy", "data_source"}
            and _user_requests_direct_change(request.message)
        )

        system_prompt = (
            "You are the AI copilot for Homerun, a Polymarket prediction market "
            "arbitrage trading platform. You help traders understand opportunities, "
            "analyze resolution criteria, assess risk, and make trading decisions.\n\n"
            "General rules:\n"
            "- Be concise, specific, and data-driven.\n"
            "- Reference only context provided in the context pack.\n"
            "- If context is insufficient, state what is missing.\n"
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

        if context_type == "strategy":
            compact_context["strategy"] = context_pack.get("strategy")
            compact_context["strategy_versions"] = (context_pack.get("strategy_versions") or [])[:10]
            compact_context["strategy_docs"] = context_pack.get("strategy_docs") or {}
            compact_context["available_data_sources"] = (context_pack.get("available_data_sources") or [])[:100]
            system_prompt += (
                "\nYou are acting as a strategy engineering copilot. "
                "Use the strategy source code and SDK documentation in context to reason precisely.\n"
            )
            if can_apply_actions:
                system_prompt += (
                    "The user explicitly requested direct edits and action execution is enabled. "
                    "Return STRICT JSON with keys:\n"
                    "message: string\n"
                    "actions: array of action objects\n"
                    "Supported action object for strategy updates:\n"
                    "{type:'update_strategy', strategy_id:'<id-or-slug-optional>', changes:{source_code?,name?,description?,config?,config_schema?,source_key?,enabled?}}\n"
                    "Only include actions when you are confident in the code changes.\n"
                )
            else:
                system_prompt += (
                    "Action execution is disabled for this turn. Respond with plain text guidance only.\n"
                )
        elif context_type == "data_source":
            compact_context["data_source"] = context_pack.get("data_source")
            compact_context["data_source_docs"] = context_pack.get("data_source_docs") or {}
            compact_context["available_data_sources"] = (context_pack.get("available_data_sources") or [])[:100]
            system_prompt += (
                "\nYou are acting as a data-source engineering copilot. "
                "Use BaseDataSource docs and source registry context.\n"
            )
            if can_apply_actions:
                system_prompt += (
                    "The user explicitly requested direct edits and action execution is enabled. "
                    "Return STRICT JSON with keys:\n"
                    "message: string\n"
                    "actions: array of action objects\n"
                    "Supported action object for data-source updates:\n"
                    "{type:'update_data_source', data_source_id:'<id-or-slug-optional>', changes:{source_code?,name?,description?,config?,config_schema?,retention?,source_key?,source_kind?,enabled?}}\n"
                    "Only include actions when you are confident in the code changes.\n"
                )
            else:
                system_prompt += (
                    "Action execution is disabled for this turn. Respond with plain text guidance only.\n"
                )

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
            max_tokens=1600 if context_type in {"strategy", "data_source"} else 1024,
            purpose="ai_chat",
            session_id=session_id,
        )

        assistant_text = response.content or ""
        action_results: dict[str, list[dict[str, Any]]] = {"applied": [], "errors": []}
        if context_type in {"strategy", "data_source"}:
            parsed_text, parsed_actions = _parse_chat_content_and_actions(assistant_text)
            if parsed_text:
                assistant_text = parsed_text
            if can_apply_actions and parsed_actions:
                action_results = await _apply_ai_chat_actions(
                    session,
                    parsed_actions,
                    context_type=context_type,
                    context_id=request.context_id,
                )
                if action_results["applied"]:
                    applied_lines = [
                        f"- {item.get('type')}: {item.get('slug') or item.get('strategy_id') or item.get('data_source_id')}"
                        for item in action_results["applied"]
                    ]
                    assistant_text = (
                        assistant_text.rstrip()
                        + "\n\nApplied changes:\n"
                        + "\n".join(applied_lines)
                    )
                if action_results["errors"]:
                    error_lines = [f"- {item.get('error')}" for item in action_results["errors"]]
                    assistant_text = (
                        assistant_text.rstrip()
                        + "\n\nAction errors:\n"
                        + "\n".join(error_lines)
                    )

        await chat_memory_service.append_message(
            session_id=session_id,
            role="assistant",
            content=assistant_text,
            model_used=response.model or request.model,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

        return {
            "session_id": session_id,
            "response": assistant_text,
            "model": response.model or "",
            "tokens_used": {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
            "actions_applied": action_results["applied"],
            "action_errors": action_results["errors"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("AI chat failed", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))
