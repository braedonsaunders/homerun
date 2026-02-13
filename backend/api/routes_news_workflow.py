"""API routes for the worker-driven News Workflow pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    NewsArticleCache,
    ScannerSnapshot,
    NewsTradeIntent,
    NewsWorkflowFinding,
    get_db_session,
)
from services.news import shared_state
from services.pause_state import global_pause_state

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_iso_utc_z(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None).isoformat() + "Z"


def _collect_cluster_article_ids(findings: list[NewsWorkflowFinding]) -> set[str]:
    article_ids: set[str] = set()
    for finding in findings:
        evidence = finding.evidence or {}
        cluster = evidence.get("cluster") if isinstance(evidence, dict) else None
        if not isinstance(cluster, dict):
            continue
        refs = cluster.get("article_refs")
        if isinstance(refs, list) and refs:
            continue
        raw_ids = cluster.get("article_ids")
        if not isinstance(raw_ids, list):
            continue
        for raw_id in raw_ids:
            article_id = str(raw_id or "").strip()
            if article_id:
                article_ids.add(article_id)
    return article_ids


def _build_supporting_articles_from_finding(
    finding: Optional[NewsWorkflowFinding],
    article_cache_by_id: Optional[dict[str, NewsArticleCache]] = None,
) -> list[dict]:
    if finding is None:
        return []

    cache = article_cache_by_id or {}
    evidence = finding.evidence or {}
    cluster = evidence.get("cluster") if isinstance(evidence, dict) else None
    refs: list[dict] = []

    if isinstance(cluster, dict):
        raw_refs = cluster.get("article_refs")
        if isinstance(raw_refs, list):
            for raw in raw_refs:
                if not isinstance(raw, dict):
                    continue
                title = str(raw.get("title") or "").strip()
                url = str(raw.get("url") or "").strip()
                if not title and not url:
                    continue
                refs.append(
                    {
                        "article_id": str(raw.get("article_id") or ""),
                        "title": title,
                        "url": url,
                        "source": str(raw.get("source") or ""),
                        "published": raw.get("published"),
                        "fetched_at": raw.get("fetched_at"),
                    }
                )

        if not refs:
            raw_ids = cluster.get("article_ids")
            if isinstance(raw_ids, list):
                for raw_id in raw_ids:
                    article_id = str(raw_id or "").strip()
                    if not article_id:
                        continue
                    row = cache.get(article_id)
                    title = ((row.title if row else "") or "").strip()
                    url = ((row.url if row else "") or "").strip()
                    if not title and not url:
                        continue
                    refs.append(
                        {
                            "article_id": article_id,
                            "title": title,
                            "url": url,
                            "source": (row.source if row else "") or "",
                            "published": _to_iso_utc_z(row.published) if row else None,
                            "fetched_at": _to_iso_utc_z(row.fetched_at) if row else None,
                        }
                    )

    if not refs:
        refs = [
            {
                "article_id": finding.article_id,
                "title": finding.article_title,
                "url": finding.article_url or "",
                "source": finding.article_source or "",
                "published": None,
                "fetched_at": _to_iso_utc_z(finding.created_at),
            }
        ]

    deduped: list[dict] = []
    seen: set[str] = set()
    for ref in refs:
        title = str(ref.get("title") or "").strip()
        url = str(ref.get("url") or "").strip()
        if not title and not url:
            continue
        key = (
            str(ref.get("article_id") or "").strip()
            or url
            or title.lower()
        )
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped[:8]


def _normalize_market_id(value: object) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _extract_yes_no_from_history(
    history: list[dict[str, float]],
) -> tuple[Optional[float], Optional[float]]:
    if not history:
        return None, None
    last = history[-1]
    yes_price = _safe_float(last.get("yes"))
    no_price = _safe_float(last.get("no"))
    return yes_price, no_price


def _normalize_history_points(raw_points: object) -> list[dict[str, float]]:
    if not isinstance(raw_points, list):
        return []

    normalized: list[dict[str, float]] = []
    for raw in raw_points:
        if not isinstance(raw, dict):
            continue

        yes_price = _safe_float(raw.get("yes"))
        no_price = _safe_float(raw.get("no"))

        if yes_price is None and no_price is not None and 0.0 <= no_price <= 1.0:
            yes_price = 1.0 - no_price
        if no_price is None and yes_price is not None and 0.0 <= yes_price <= 1.0:
            no_price = 1.0 - yes_price
        if yes_price is None or no_price is None:
            continue

        yes_price = float(max(0.0, min(1.0, yes_price)))
        no_price = float(max(0.0, min(1.0, no_price)))
        point: dict[str, float] = {"yes": yes_price, "no": no_price}

        ts = _safe_float(raw.get("t"))
        if ts is not None:
            point["t"] = ts

        normalized.append(point)

    return normalized


async def _load_scanner_market_history(
    session: AsyncSession,
) -> dict[str, list[dict[str, float]]]:
    result = await session.execute(
        select(ScannerSnapshot).where(ScannerSnapshot.id == "latest")
    )
    row = result.scalar_one_or_none()
    if row is None or not isinstance(row.market_history_json, dict):
        return {}

    history_map: dict[str, list[dict[str, float]]] = {}
    for raw_market_id, raw_points in row.market_history_json.items():
        market_id = _normalize_market_id(raw_market_id)
        if not market_id:
            continue
        points = _normalize_history_points(raw_points)
        if len(points) >= 2:
            history_map[market_id] = points[-20:]
    return history_map


def _build_finding_market_snapshot(
    finding: NewsWorkflowFinding,
    market_history: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    market_id = _normalize_market_id(finding.market_id)
    history = market_history.get(market_id, [])
    yes_from_history, no_from_history = _extract_yes_no_from_history(history)

    fallback_yes = _safe_float(finding.market_price)
    fallback_no = (
        float(1.0 - fallback_yes)
        if fallback_yes is not None and 0.0 <= fallback_yes <= 1.0
        else None
    )

    current_yes = yes_from_history if yes_from_history is not None else fallback_yes
    current_no = no_from_history if no_from_history is not None else fallback_no
    if current_yes is None and current_no is not None and 0.0 <= current_no <= 1.0:
        current_yes = float(1.0 - current_no)
    if current_no is None and current_yes is not None and 0.0 <= current_yes <= 1.0:
        current_no = float(1.0 - current_yes)

    return {
        "price_history": history,
        "yes_price": current_yes,
        "no_price": current_no,
        "current_yes_price": current_yes,
        "current_no_price": current_no,
    }


class CustomRssFeedRequest(BaseModel):
    id: Optional[str] = None
    name: str = ""
    url: str = ""
    enabled: bool = True
    category: Optional[str] = ""


class GovRssFeedRequest(BaseModel):
    id: Optional[str] = None
    agency: str = "government"
    name: str = ""
    url: str = ""
    priority: str = "medium"
    country_iso3: str = "USA"
    enabled: bool = True


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
    require_verifier: Optional[bool] = None
    market_min_liquidity: Optional[float] = Field(None, ge=0.0, le=1_000_000.0)
    market_max_days_to_resolution: Optional[int] = Field(None, ge=1, le=3650)
    min_keyword_signal: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_semantic_signal: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_edge_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    require_second_source: Optional[bool] = None
    orchestrator_enabled: Optional[bool] = None
    orchestrator_min_edge: Optional[float] = Field(None, ge=0.0, le=100.0)
    orchestrator_max_age_minutes: Optional[int] = Field(None, ge=1, le=1440)
    cycle_spend_cap_usd: Optional[float] = Field(None, ge=0.0, le=100.0)
    hourly_spend_cap_usd: Optional[float] = Field(None, ge=0.0, le=1000.0)
    cycle_llm_call_cap: Optional[int] = Field(None, ge=0, le=500)
    cache_ttl_minutes: Optional[int] = Field(None, ge=1, le=1440)
    max_edge_evals_per_article: Optional[int] = Field(None, ge=1, le=20)
    model: Optional[str] = None
    rss_feeds: Optional[list[CustomRssFeedRequest]] = None
    rss_enabled: Optional[bool] = None
    rss_sources: Optional[list[GovRssFeedRequest]] = None
    # Backward-compatible request aliases.
    gov_rss_enabled: Optional[bool] = None
    gov_rss_feeds: Optional[list[GovRssFeedRequest]] = None


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
            _to_iso_utc_z(control.get("requested_scan_at"))
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
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Resume all workers before queueing runs.",
        )
    await shared_state.request_one_news_scan(session)
    return {
        "status": "queued",
        "message": "News workflow cycle requested; worker will run it shortly.",
        **await _build_status_payload(session),
    }


@router.post("/news-workflow/start")
async def start_workflow(session: AsyncSession = Depends(get_db_session)):
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )
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
    include_debug_rejections: bool = Query(
        False, description="Include non-actionable debug rejection rows"
    ),
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
    elif not include_debug_rejections:
        query = query.where(
            (NewsWorkflowFinding.actionable == True)  # noqa: E712
            | (NewsWorkflowFinding.confidence > 0.0)
        )

    query = query.order_by(desc(NewsWorkflowFinding.created_at))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar() or 0

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    rows = result.scalars().all()
    article_ids_needed = _collect_cluster_article_ids(rows)
    article_cache_by_id: dict[str, NewsArticleCache] = {}
    if article_ids_needed:
        article_result = await session.execute(
            select(NewsArticleCache).where(
                NewsArticleCache.article_id.in_(list(article_ids_needed))
            )
        )
        cached_rows = article_result.scalars().all()
        article_cache_by_id = {
            row.article_id: row for row in cached_rows if row.article_id
        }
    market_history = await _load_scanner_market_history(session)

    findings = []
    for r in rows:
        supporting_articles = _build_supporting_articles_from_finding(
            r, article_cache_by_id=article_cache_by_id
        )
        market_snapshot = _build_finding_market_snapshot(r, market_history)
        findings.append(
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
                "consumed_by_orchestrator": r.consumed_by_orchestrator,
                "supporting_articles": supporting_articles,
                "supporting_article_count": int(len(supporting_articles)),
                **market_snapshot,
                "created_at": _to_iso_utc_z(r.created_at),
            }
        )

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

    finding_ids = [r.finding_id for r in rows if r.finding_id]
    finding_by_id: dict[str, NewsWorkflowFinding] = {}
    article_cache_by_id: dict[str, NewsArticleCache] = {}
    if finding_ids:
        finding_result = await session.execute(
            select(NewsWorkflowFinding).where(NewsWorkflowFinding.id.in_(finding_ids))
        )
        findings = finding_result.scalars().all()
        finding_by_id = {f.id: f for f in findings if f.id}

        article_ids_needed = _collect_cluster_article_ids(findings)
        if article_ids_needed:
            article_result = await session.execute(
                select(NewsArticleCache).where(
                    NewsArticleCache.article_id.in_(list(article_ids_needed))
                )
            )
            cached_rows = article_result.scalars().all()
            article_cache_by_id = {
                row.article_id: row for row in cached_rows if row.article_id
            }

    intents = []
    for r in rows:
        metadata = r.metadata_json if isinstance(r.metadata_json, dict) else {}
        supporting_articles = metadata.get(
            "supporting_articles"
        ) or _build_supporting_articles_from_finding(
            finding_by_id.get(r.finding_id),
            article_cache_by_id=article_cache_by_id,
        )
        intents.append(
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
                "metadata": {
                    **metadata,
                    "supporting_articles": supporting_articles,
                    "supporting_article_count": int(len(supporting_articles)),
                },
                "status": r.status,
                "created_at": _to_iso_utc_z(r.created_at),
                "consumed_at": _to_iso_utc_z(r.consumed_at),
            }
        )

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
        # Normalize legacy payload keys to the rebranded RSS keys.
        if "gov_rss_enabled" in updates and "rss_enabled" not in updates:
            updates["rss_enabled"] = updates.get("gov_rss_enabled")
        if "gov_rss_feeds" in updates and "rss_sources" not in updates:
            updates["rss_sources"] = updates.get("gov_rss_feeds")
        settings_payload = await shared_state.update_news_settings(session, updates)

        if "scan_interval_seconds" in updates:
            await shared_state.set_news_interval(
                session, int(updates["scan_interval_seconds"])
            )

        return {"status": "success", "settings": settings_payload}
    except Exception as e:
        logger.error("Failed to update workflow settings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
