"""API routes for the worker-driven News Workflow pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    NewsTradeIntent,
    NewsWorkflowFinding,
    get_db_session,
)
from services.news import shared_state

logger = logging.getLogger(__name__)
router = APIRouter()


class WorkflowSettingsRequest(BaseModel):
    """Update workflow settings."""

    enabled: Optional[bool] = None
    auto_run: Optional[bool] = None
    scan_interval_seconds: Optional[int] = Field(None, ge=30, le=3600)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    rerank_top_n: Optional[int] = Field(None, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    event_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_edge_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    require_second_source: Optional[bool] = None
    auto_trader_enabled: Optional[bool] = None
    auto_trader_min_edge: Optional[float] = Field(None, ge=0.0, le=100.0)
    auto_trader_max_age_minutes: Optional[int] = Field(None, ge=1, le=1440)
    cycle_spend_cap_usd: Optional[float] = Field(None, ge=0.0, le=100.0)
    hourly_spend_cap_usd: Optional[float] = Field(None, ge=0.0, le=1000.0)
    cycle_llm_call_cap: Optional[int] = Field(None, ge=0, le=500)
    cache_ttl_minutes: Optional[int] = Field(None, ge=1, le=1440)
    max_edge_evals_per_article: Optional[int] = Field(None, ge=1, le=20)
    model: Optional[str] = None


async def _build_status_payload(session: AsyncSession) -> dict:
    status = await shared_state.read_news_snapshot(session)
    control = await shared_state.read_news_control(session)
    pending = await shared_state.count_pending_news_intents(session)
    stats = status.get("stats") or {}
    stats.setdefault("budget_skip_count", int(stats.get("llm_calls_skipped", 0) or 0))
    stats.setdefault("findings", int(stats.get("findings", 0) or 0))
    stats.setdefault("intents", int(stats.get("intents", 0) or 0))
    stats["pending_intents"] = pending

    return {
        "running": bool(status.get("running", False)),
        "enabled": bool(control.get("is_enabled", True)) and bool(
            status.get("enabled", True)
        ),
        "paused": bool(control.get("is_paused", False)),
        "interval_seconds": int(
            control.get("scan_interval_seconds")
            or status.get("interval_seconds")
            or 120
        ),
        "last_scan": status.get("last_scan"),
        "next_scan": status.get("next_scan"),
        "current_activity": status.get("current_activity"),
        "last_error": status.get("last_error"),
        "degraded_mode": bool(status.get("degraded_mode", False)),
        "budget_remaining": status.get("budget_remaining"),
        "pending_intents": pending,
        "requested_scan_at": (
            control.get("requested_scan_at").isoformat()
            if control.get("requested_scan_at")
            else None
        ),
        "stats": stats,
    }


@router.get("/news-workflow/status")
async def get_workflow_status(session: AsyncSession = Depends(get_db_session)):
    return await _build_status_payload(session)


@router.post("/news-workflow/run")
async def run_workflow_cycle(session: AsyncSession = Depends(get_db_session)):
    """Queue a one-time workflow cycle (non-blocking)."""
    await shared_state.request_one_news_scan(session)
    return {
        "status": "queued",
        "message": "News workflow cycle requested; worker will run it shortly.",
        **await _build_status_payload(session),
    }


@router.post("/news-workflow/start")
async def start_workflow(session: AsyncSession = Depends(get_db_session)):
    await shared_state.set_news_paused(session, False)
    return {"status": "started", **await _build_status_payload(session)}


@router.post("/news-workflow/pause")
async def pause_workflow(session: AsyncSession = Depends(get_db_session)):
    await shared_state.set_news_paused(session, True)
    return {"status": "paused", **await _build_status_payload(session)}


@router.post("/news-workflow/interval")
async def set_workflow_interval(
    interval_seconds: int = Query(..., ge=30, le=3600),
    session: AsyncSession = Depends(get_db_session),
):
    await shared_state.set_news_interval(session, interval_seconds)
    await shared_state.update_news_settings(
        session, {"scan_interval_seconds": interval_seconds}
    )
    return {"status": "updated", **await _build_status_payload(session)}


@router.get("/news-workflow/findings")
async def get_findings(
    min_edge: float = Query(0.0, ge=0, description="Minimum edge %"),
    actionable_only: bool = Query(False, description="Only actionable findings"),
    max_age_hours: int = Query(24, ge=1, le=336, description="Max age in hours"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session),
):
    """Get persisted workflow findings."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    query = select(NewsWorkflowFinding).where(NewsWorkflowFinding.created_at >= cutoff)

    if min_edge > 0:
        query = query.where(NewsWorkflowFinding.edge_percent >= min_edge)

    if actionable_only:
        query = query.where(NewsWorkflowFinding.actionable == True)  # noqa: E712

    query = query.order_by(desc(NewsWorkflowFinding.created_at))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar() or 0

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    rows = result.scalars().all()

    findings = [
        {
            "id": r.id,
            "article_id": r.article_id,
            "market_id": r.market_id,
            "article_title": r.article_title,
            "article_source": r.article_source,
            "article_url": r.article_url,
            "signal_key": r.signal_key,
            "cache_key": r.cache_key,
            "market_question": r.market_question,
            "market_price": r.market_price,
            "model_probability": r.model_probability,
            "edge_percent": r.edge_percent,
            "direction": r.direction,
            "confidence": r.confidence,
            "retrieval_score": r.retrieval_score,
            "semantic_score": r.semantic_score,
            "keyword_score": r.keyword_score,
            "event_score": r.event_score,
            "rerank_score": r.rerank_score,
            "event_graph": r.event_graph,
            "evidence": r.evidence,
            "reasoning": r.reasoning,
            "actionable": r.actionable,
            "consumed_by_auto_trader": r.consumed_by_auto_trader,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "findings": findings,
    }


@router.get("/news-workflow/intents")
async def get_intents(
    status_filter: Optional[str] = Query(
        None, description="Filter by status: pending, submitted, executed, skipped, expired"
    ),
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_db_session),
):
    """Get trade intents."""
    rows = await shared_state.list_news_intents(
        session, status_filter=status_filter, limit=limit
    )
    intents = [
        {
            "id": r.id,
            "signal_key": r.signal_key,
            "finding_id": r.finding_id,
            "market_id": r.market_id,
            "market_question": r.market_question,
            "direction": r.direction,
            "entry_price": r.entry_price,
            "model_probability": r.model_probability,
            "edge_percent": r.edge_percent,
            "confidence": r.confidence,
            "suggested_size_usd": r.suggested_size_usd,
            "metadata": r.metadata_json,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "consumed_at": r.consumed_at.isoformat() if r.consumed_at else None,
        }
        for r in rows
    ]

    return {"total": len(intents), "intents": intents}


@router.post("/news-workflow/intents/{intent_id}/skip")
async def skip_intent(intent_id: str, session: AsyncSession = Depends(get_db_session)):
    """Manually skip a pending intent."""
    intent_result = await session.execute(
        select(NewsTradeIntent).where(NewsTradeIntent.id == intent_id)
    )
    intent = intent_result.scalar_one_or_none()
    if intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    if intent.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot skip intent with status '{intent.status}'",
        )

    ok = await shared_state.mark_news_intent(session, intent_id, "skipped")
    if not ok:
        raise HTTPException(status_code=404, detail="Intent not found")
    return {"status": "skipped", "intent_id": intent_id}


@router.get("/news-workflow/settings")
async def get_workflow_settings(session: AsyncSession = Depends(get_db_session)):
    """Get current workflow settings."""
    try:
        return await shared_state.get_news_settings(session)
    except Exception as e:
        logger.error("Failed to get workflow settings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/news-workflow/settings")
async def update_workflow_settings(
    request: WorkflowSettingsRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Update workflow settings."""
    try:
        updates = request.model_dump(exclude_unset=True)
        settings_payload = await shared_state.update_news_settings(session, updates)

        if "scan_interval_seconds" in updates:
            await shared_state.set_news_interval(
                session, int(updates["scan_interval_seconds"])
            )

        return {"status": "success", "settings": settings_payload}
    except Exception as e:
        logger.error("Failed to update workflow settings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
