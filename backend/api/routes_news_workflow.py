"""
API routes for the independent News Workflow pipeline.

Completely separate from routes_news.py (legacy matching/edges).
Provides endpoints for:
- Pipeline status and manual trigger
- Persisted findings with filters
- Trade intents with status management
- Workflow-specific settings
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta, timezone
import logging

from sqlalchemy import select, update, desc, func

logger = logging.getLogger(__name__)
router = APIRouter()


# ======================================================================
# Request / Response Models
# ======================================================================


class WorkflowSettingsRequest(BaseModel):
    """Update workflow settings."""

    enabled: Optional[bool] = None
    auto_run: Optional[bool] = None
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
    model: Optional[str] = None


# ======================================================================
# Status
# ======================================================================


@router.get("/news-workflow/status")
async def get_workflow_status():
    """Get current status of the news workflow pipeline."""
    from services.news.workflow_orchestrator import workflow_orchestrator
    from services.news.market_watcher_index import market_watcher_index

    status = workflow_orchestrator.get_status()
    status["market_index"] = market_watcher_index.get_status()

    # Count pending intents
    try:
        from models.database import AsyncSessionLocal, NewsTradeIntent

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.count(NewsTradeIntent.id)).where(
                    NewsTradeIntent.status == "pending"
                )
            )
            status["pending_intents"] = result.scalar() or 0
    except Exception:
        status["pending_intents"] = 0

    return status


# ======================================================================
# Run Cycle
# ======================================================================


@router.post("/news-workflow/run")
async def run_workflow_cycle():
    """Manually trigger a workflow cycle."""
    from services.news.workflow_orchestrator import workflow_orchestrator

    try:
        result = await workflow_orchestrator.run_cycle()
        return result
    except Exception as e:
        logger.error("Workflow cycle failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Findings
# ======================================================================


@router.get("/news-workflow/findings")
async def get_findings(
    min_edge: float = Query(0.0, ge=0, description="Minimum edge %"),
    actionable_only: bool = Query(False, description="Only actionable findings"),
    max_age_hours: int = Query(24, ge=1, le=336, description="Max age in hours"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get persisted workflow findings."""
    from models.database import AsyncSessionLocal, NewsWorkflowFinding

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        async with AsyncSessionLocal() as session:
            query = select(NewsWorkflowFinding).where(
                NewsWorkflowFinding.created_at >= cutoff
            )

            if min_edge > 0:
                query = query.where(NewsWorkflowFinding.edge_percent >= min_edge)

            if actionable_only:
                query = query.where(NewsWorkflowFinding.actionable == True)  # noqa: E712

            query = query.order_by(desc(NewsWorkflowFinding.created_at))

            # Count total
            count_q = select(func.count()).select_from(query.subquery())
            total = (await session.execute(count_q)).scalar() or 0

            # Paginate
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
    except Exception as e:
        logger.error("Failed to get findings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Trade Intents
# ======================================================================


@router.get("/news-workflow/intents")
async def get_intents(
    status_filter: Optional[str] = Query(
        None, description="Filter by status: pending, submitted, executed, skipped, expired"
    ),
    limit: int = Query(50, ge=1, le=500),
):
    """Get trade intents."""
    from models.database import AsyncSessionLocal, NewsTradeIntent

    try:
        async with AsyncSessionLocal() as session:
            query = select(NewsTradeIntent).order_by(desc(NewsTradeIntent.created_at))

            if status_filter:
                query = query.where(NewsTradeIntent.status == status_filter)

            query = query.limit(limit)
            result = await session.execute(query)
            rows = result.scalars().all()

        intents = [
            {
                "id": r.id,
                "finding_id": r.finding_id,
                "market_id": r.market_id,
                "market_question": r.market_question,
                "direction": r.direction,
                "entry_price": r.entry_price,
                "model_probability": r.model_probability,
                "edge_percent": r.edge_percent,
                "confidence": r.confidence,
                "suggested_size_usd": r.suggested_size_usd,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "consumed_at": r.consumed_at.isoformat() if r.consumed_at else None,
            }
            for r in rows
        ]

        return {"total": len(intents), "intents": intents}
    except Exception as e:
        logger.error("Failed to get intents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/news-workflow/intents/{intent_id}/skip")
async def skip_intent(intent_id: str):
    """Manually skip a pending intent."""
    from models.database import AsyncSessionLocal, NewsTradeIntent

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(NewsTradeIntent).where(NewsTradeIntent.id == intent_id)
            )
            intent = result.scalar_one_or_none()

            if not intent:
                raise HTTPException(status_code=404, detail="Intent not found")

            if intent.status != "pending":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot skip intent with status '{intent.status}'",
                )

            intent.status = "skipped"
            intent.consumed_at = datetime.now(timezone.utc)
            await session.commit()

        return {"status": "skipped", "intent_id": intent_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to skip intent: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Settings
# ======================================================================


@router.get("/news-workflow/settings")
async def get_workflow_settings():
    """Get current workflow settings."""
    from models.database import AsyncSessionLocal, AppSettings

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.id == "default")
            )
            db = result.scalar_one_or_none()

        if not db:
            return _default_settings()

        return {
            "enabled": getattr(db, "news_workflow_enabled", True),
            "auto_run": getattr(db, "news_workflow_auto_run", True),
            "top_k": getattr(db, "news_workflow_top_k", 8),
            "rerank_top_n": getattr(db, "news_workflow_rerank_top_n", 5),
            "similarity_threshold": getattr(db, "news_workflow_similarity_threshold", 0.35),
            "keyword_weight": getattr(db, "news_workflow_keyword_weight", 0.25),
            "semantic_weight": getattr(db, "news_workflow_semantic_weight", 0.45),
            "event_weight": getattr(db, "news_workflow_event_weight", 0.30),
            "min_edge_percent": getattr(db, "news_workflow_min_edge_percent", 8.0),
            "min_confidence": getattr(db, "news_workflow_min_confidence", 0.6),
            "require_second_source": getattr(db, "news_workflow_require_second_source", False),
            "auto_trader_enabled": getattr(db, "news_workflow_auto_trader_enabled", True),
            "auto_trader_min_edge": getattr(db, "news_workflow_auto_trader_min_edge", 10.0),
            "auto_trader_max_age_minutes": getattr(db, "news_workflow_auto_trader_max_age_minutes", 120),
            "model": getattr(db, "news_workflow_model", None),
        }
    except Exception as e:
        logger.error("Failed to get workflow settings: %s", e)
        return _default_settings()


@router.put("/news-workflow/settings")
async def update_workflow_settings(request: WorkflowSettingsRequest):
    """Update workflow settings."""
    from models.database import AsyncSessionLocal, AppSettings

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.id == "default")
            )
            db = result.scalar_one_or_none()

            if not db:
                db = AppSettings(id="default")
                session.add(db)

            if request.enabled is not None:
                db.news_workflow_enabled = request.enabled
            if request.auto_run is not None:
                db.news_workflow_auto_run = request.auto_run
            if request.top_k is not None:
                db.news_workflow_top_k = request.top_k
            if request.rerank_top_n is not None:
                db.news_workflow_rerank_top_n = request.rerank_top_n
            if request.similarity_threshold is not None:
                db.news_workflow_similarity_threshold = request.similarity_threshold
            if request.keyword_weight is not None:
                db.news_workflow_keyword_weight = request.keyword_weight
            if request.semantic_weight is not None:
                db.news_workflow_semantic_weight = request.semantic_weight
            if request.event_weight is not None:
                db.news_workflow_event_weight = request.event_weight
            if request.min_edge_percent is not None:
                db.news_workflow_min_edge_percent = request.min_edge_percent
            if request.min_confidence is not None:
                db.news_workflow_min_confidence = request.min_confidence
            if request.require_second_source is not None:
                db.news_workflow_require_second_source = request.require_second_source
            if request.auto_trader_enabled is not None:
                db.news_workflow_auto_trader_enabled = request.auto_trader_enabled
            if request.auto_trader_min_edge is not None:
                db.news_workflow_auto_trader_min_edge = request.auto_trader_min_edge
            if request.auto_trader_max_age_minutes is not None:
                db.news_workflow_auto_trader_max_age_minutes = request.auto_trader_max_age_minutes
            if request.model is not None:
                db.news_workflow_model = request.model or None

            db.updated_at = datetime.utcnow()
            await session.commit()

        return {"status": "success", "message": "Workflow settings updated"}
    except Exception as e:
        logger.error("Failed to update workflow settings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _default_settings() -> dict:
    return {
        "enabled": True,
        "auto_run": True,
        "top_k": 8,
        "rerank_top_n": 5,
        "similarity_threshold": 0.35,
        "keyword_weight": 0.25,
        "semantic_weight": 0.45,
        "event_weight": 0.30,
        "min_edge_percent": 8.0,
        "min_confidence": 0.6,
        "require_second_source": False,
        "auto_trader_enabled": True,
        "auto_trader_min_edge": 10.0,
        "auto_trader_max_age_minutes": 120,
        "model": None,
    }
