"""Shared DB state for news workflow worker/API/orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings as app_settings
from models.database import (
    AppSettings,
    NewsTradeIntent,
    NewsWorkflowControl,
    NewsWorkflowFinding,
    NewsWorkflowSnapshot,
)
from services.market_tradability import get_market_tradability_map
from services.news.rss_config import (
    default_custom_rss_feeds,
    default_gov_rss_feeds,
    normalize_custom_rss_feeds,
    normalize_gov_rss_feeds,
)

NEWS_SNAPSHOT_ID = "latest"
NEWS_CONTROL_ID = "default"


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("+00:00+00:00"):
        text = text[:-6]
    if text.endswith("Z"):
        text = text[:-1]
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _format_iso_utc_z(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + "Z"


def _default_status() -> dict[str, Any]:
    return {
        "running": False,
        "enabled": True,
        "interval_seconds": 120,
        "last_scan": None,
        "next_scan": None,
        "current_activity": "Waiting for news worker.",
        "last_error": None,
        "degraded_mode": False,
        "budget_remaining": None,
        "stats": {},
    }


async def write_news_snapshot(
    session: AsyncSession,
    status: dict[str, Any],
    stats: Optional[dict[str, Any]] = None,
) -> None:
    has_last_scan = "last_scan" in status
    has_next_scan = "next_scan" in status
    has_last_error = "last_error" in status
    has_budget_remaining = "budget_remaining" in status

    last_scan = status.get("last_scan")
    if has_last_scan and isinstance(last_scan, str):
        try:
            last_scan = _parse_iso_datetime(last_scan)
        except Exception:
            last_scan = utcnow()

    next_scan = status.get("next_scan")
    if has_next_scan and isinstance(next_scan, str):
        try:
            next_scan = _parse_iso_datetime(next_scan)
        except Exception:
            next_scan = None

    result = await session.execute(select(NewsWorkflowSnapshot).where(NewsWorkflowSnapshot.id == NEWS_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = NewsWorkflowSnapshot(id=NEWS_SNAPSHOT_ID)
        session.add(row)

    row.updated_at = utcnow()
    if has_last_scan:
        row.last_scan_at = last_scan
    if has_next_scan:
        row.next_scan_at = next_scan
    row.running = bool(status.get("running", True))
    row.enabled = bool(status.get("enabled", True))
    row.current_activity = status.get("current_activity")
    row.interval_seconds = int(status.get("interval_seconds", 120))
    if has_last_error:
        row.last_error = status.get("last_error")
    row.degraded_mode = bool(status.get("degraded_mode", False))
    if has_budget_remaining:
        row.budget_remaining_usd = status.get("budget_remaining")
    row.stats_json = stats if stats is not None else (status.get("stats") or {})
    await session.commit()


async def read_news_snapshot(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(select(NewsWorkflowSnapshot).where(NewsWorkflowSnapshot.id == NEWS_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return _default_status()

    return {
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "interval_seconds": int(row.interval_seconds or 120),
        "last_scan": _format_iso_utc_z(row.last_scan_at),
        "next_scan": _format_iso_utc_z(row.next_scan_at),
        "current_activity": row.current_activity,
        "last_error": row.last_error,
        "degraded_mode": bool(row.degraded_mode),
        "budget_remaining": row.budget_remaining_usd,
        "stats": row.stats_json or {},
    }


async def get_news_status_from_db(session: AsyncSession) -> dict[str, Any]:
    status = await read_news_snapshot(session)
    pending = await count_pending_news_intents(session)
    control = await read_news_control(session)
    status["pending_intents"] = pending
    status["paused"] = bool(control.get("is_paused", False))
    status["requested_scan_at"] = (
        control.get("requested_scan_at").isoformat() if control.get("requested_scan_at") else None
    )
    return status


async def ensure_news_control(session: AsyncSession) -> NewsWorkflowControl:
    result = await session.execute(select(NewsWorkflowControl).where(NewsWorkflowControl.id == NEWS_CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = NewsWorkflowControl(id=NEWS_CONTROL_ID)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_news_control(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(select(NewsWorkflowControl).where(NewsWorkflowControl.id == NEWS_CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "scan_interval_seconds": 120,
            "requested_scan_at": None,
            "lease_owner": None,
            "lease_expires_at": None,
        }
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "scan_interval_seconds": int(row.scan_interval_seconds or 120),
        "requested_scan_at": row.requested_scan_at,
        "lease_owner": row.lease_owner,
        "lease_expires_at": row.lease_expires_at,
    }


async def set_news_paused(session: AsyncSession, paused: bool) -> None:
    row = await ensure_news_control(session)
    row.is_paused = paused
    row.updated_at = utcnow()
    await session.commit()


async def set_news_interval(session: AsyncSession, interval_seconds: int) -> None:
    row = await ensure_news_control(session)
    row.scan_interval_seconds = max(30, min(3600, int(interval_seconds)))
    row.updated_at = utcnow()
    await session.commit()


async def request_one_news_scan(session: AsyncSession) -> None:
    row = await ensure_news_control(session)
    row.requested_scan_at = utcnow()
    row.updated_at = utcnow()
    await session.commit()


async def clear_news_scan_request(session: AsyncSession) -> None:
    row = await ensure_news_control(session)
    row.requested_scan_at = None
    row.updated_at = utcnow()
    await session.commit()


async def try_acquire_news_lease(
    session: AsyncSession,
    owner: str,
    ttl_seconds: int,
) -> bool:
    """Try to acquire/renew the worker lease. Returns True if owned."""
    await ensure_news_control(session)
    now = utcnow()
    lease_until = now + timedelta(seconds=max(30, ttl_seconds))

    stmt = (
        update(NewsWorkflowControl)
        .where(NewsWorkflowControl.id == NEWS_CONTROL_ID)
        .where(
            (NewsWorkflowControl.lease_owner.is_(None))
            | (NewsWorkflowControl.lease_owner == owner)
            | (NewsWorkflowControl.lease_expires_at.is_(None))
            | (NewsWorkflowControl.lease_expires_at < now)
        )
        .values(
            lease_owner=owner,
            lease_expires_at=lease_until,
            updated_at=now,
        )
    )
    result = await session.execute(stmt)
    await session.commit()
    return (result.rowcount or 0) > 0


async def release_news_lease(session: AsyncSession, owner: str) -> None:
    await session.execute(
        update(NewsWorkflowControl)
        .where(NewsWorkflowControl.id == NEWS_CONTROL_ID)
        .where(NewsWorkflowControl.lease_owner == owner)
        .values(
            lease_owner=None,
            lease_expires_at=None,
            updated_at=utcnow(),
        )
    )
    await session.commit()


async def count_pending_news_intents(session: AsyncSession) -> int:
    rows = await list_news_intents(session, status_filter="pending", limit=5000)
    return len(rows)


async def list_news_intents(
    session: AsyncSession,
    status_filter: Optional[str] = None,
    limit: int = 100,
) -> list[NewsTradeIntent]:
    query = select(NewsTradeIntent).order_by(NewsTradeIntent.created_at.desc())
    if status_filter:
        query = query.where(NewsTradeIntent.status == status_filter)
    query = query.limit(limit)
    result = await session.execute(query)
    rows = list(result.scalars().all())

    actionable = [r for r in rows if r.status in {"pending", "submitted"} and r.market_id]
    if actionable:
        tradability = await get_market_tradability_map([str(r.market_id) for r in actionable])
        now = utcnow()
        changed = 0
        for row in actionable:
            if tradability.get(str(row.market_id).strip().lower(), True):
                continue
            row.status = "expired"
            row.consumed_at = now
            changed += 1
        if changed:
            await session.commit()
            if status_filter in {"pending", "submitted"}:
                rows = [r for r in rows if r.status == status_filter]

    return rows


async def mark_news_intent(
    session: AsyncSession,
    intent_id: str,
    status: str,
) -> bool:
    result = await session.execute(select(NewsTradeIntent).where(NewsTradeIntent.id == intent_id))
    row = result.scalar_one_or_none()
    if row is None:
        return False
    row.status = status
    row.consumed_at = datetime.now(timezone.utc)
    if row.finding_id and status != "pending":
        finding = await session.execute(select(NewsWorkflowFinding).where(NewsWorkflowFinding.id == row.finding_id))
        finding_row = finding.scalar_one_or_none()
        if finding_row is not None:
            finding_row.consumed_by_orchestrator = True
    await session.commit()
    return True


async def expire_stale_news_intents(
    session: AsyncSession,
    max_age_minutes: int,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    result = await session.execute(
        update(NewsTradeIntent)
        .where(NewsTradeIntent.status == "pending")
        .where(NewsTradeIntent.created_at < cutoff)
        .values(
            status="expired",
            consumed_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
    return int(result.rowcount or 0)


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    db = result.scalar_one_or_none()
    mutated = False
    if db is None:
        db = AppSettings(id="default")
        db.news_gov_rss_enabled = True
        db.news_gov_rss_feeds_json = default_gov_rss_feeds()
        session.add(db)
        mutated = True
    else:
        if getattr(db, "news_gov_rss_enabled", None) is None:
            db.news_gov_rss_enabled = True
            mutated = True
        # Lift legacy strict retrieval defaults that frequently produced zero
        # findings in live workflows.
        if getattr(db, "news_workflow_top_k", None) in (None, 8):
            db.news_workflow_top_k = 20
            mutated = True
        if getattr(db, "news_workflow_rerank_top_n", None) in (None, 5):
            db.news_workflow_rerank_top_n = 8
            mutated = True
        if getattr(db, "news_workflow_similarity_threshold", None) in (
            None,
            0.35,
            0.42,
        ):
            db.news_workflow_similarity_threshold = 0.20
            mutated = True
        if getattr(db, "news_workflow_min_semantic_signal", None) in (None, 0.22):
            db.news_workflow_min_semantic_signal = 0.05
            mutated = True
        if getattr(db, "news_workflow_max_edge_evals_per_article", None) in (
            None,
            3,
        ):
            db.news_workflow_max_edge_evals_per_article = 6
            mutated = True
        raw_gov_feeds = getattr(db, "news_gov_rss_feeds_json", None)
        normalized_gov_feeds = normalize_gov_rss_feeds(raw_gov_feeds)
        if not normalized_gov_feeds:
            db.news_gov_rss_feeds_json = default_gov_rss_feeds()
            mutated = True
    if mutated:
        await session.commit()
        await session.refresh(db)
    return db


async def get_news_settings(session: AsyncSession) -> dict[str, Any]:
    db = await _get_or_create_app_settings(session)
    raw_custom_feeds = getattr(db, "news_rss_feeds_json", None)
    custom_rss_feeds = normalize_custom_rss_feeds(raw_custom_feeds) if raw_custom_feeds else default_custom_rss_feeds()
    raw_gov_feeds = getattr(db, "news_gov_rss_feeds_json", None)
    rss_sources = normalize_gov_rss_feeds(raw_gov_feeds) if raw_gov_feeds else default_gov_rss_feeds()
    raw_gov_enabled = getattr(db, "news_gov_rss_enabled", None)
    rss_enabled = True if raw_gov_enabled is None else bool(raw_gov_enabled)

    return {
        "enabled": bool(getattr(db, "news_workflow_enabled", True)),
        "auto_run": bool(getattr(db, "news_workflow_auto_run", True)),
        "scan_interval_seconds": int(getattr(db, "news_workflow_scan_interval_seconds", 120) or 120),
        "top_k": int(getattr(db, "news_workflow_top_k", 20) or 20),
        "rerank_top_n": int(getattr(db, "news_workflow_rerank_top_n", 8) or 8),
        "similarity_threshold": float(getattr(db, "news_workflow_similarity_threshold", 0.20) or 0.20),
        "keyword_weight": float(getattr(db, "news_workflow_keyword_weight", 0.25) or 0.25),
        "semantic_weight": float(getattr(db, "news_workflow_semantic_weight", 0.45) or 0.45),
        "event_weight": float(getattr(db, "news_workflow_event_weight", 0.30) or 0.30),
        "require_verifier": bool(getattr(db, "news_workflow_require_verifier", True)),
        "market_min_liquidity": float(getattr(db, "news_workflow_market_min_liquidity", 500.0) or 500.0),
        "market_max_days_to_resolution": int(getattr(db, "news_workflow_market_max_days_to_resolution", 365) or 365),
        "min_keyword_signal": float(getattr(db, "news_workflow_min_keyword_signal", 0.04) or 0.04),
        "min_semantic_signal": float(getattr(db, "news_workflow_min_semantic_signal", 0.05) or 0.05),
        "min_edge_percent": float(getattr(db, "news_workflow_min_edge_percent", 8.0) or 8.0),
        "min_confidence": float(getattr(db, "news_workflow_min_confidence", 0.6) or 0.6),
        "require_second_source": bool(getattr(db, "news_workflow_require_second_source", False)),
        "orchestrator_enabled": bool(getattr(db, "news_workflow_orchestrator_enabled", True)),
        "orchestrator_min_edge": float(getattr(db, "news_workflow_orchestrator_min_edge", 10.0) or 10.0),
        "orchestrator_max_age_minutes": int(getattr(db, "news_workflow_orchestrator_max_age_minutes", 120) or 120),
        "model": getattr(db, "news_workflow_model", None),
        "article_max_age_hours": min(int(app_settings.NEWS_ARTICLE_TTL_HOURS), 48),
        "cycle_spend_cap_usd": float(getattr(db, "news_workflow_cycle_spend_cap_usd", 0.25) or 0.25),
        "hourly_spend_cap_usd": float(getattr(db, "news_workflow_hourly_spend_cap_usd", 2.0) or 2.0),
        "cycle_llm_call_cap": int(getattr(db, "news_workflow_cycle_llm_call_cap", 30) or 30),
        "cache_ttl_minutes": int(getattr(db, "news_workflow_cache_ttl_minutes", 30) or 30),
        "max_edge_evals_per_article": int(getattr(db, "news_workflow_max_edge_evals_per_article", 6) or 6),
        "rss_feeds": custom_rss_feeds,
        "rss_enabled": rss_enabled,
        "rss_sources": rss_sources,
        # Backward-compatible API aliases. Keep until older clients are removed.
        "gov_rss_enabled": rss_enabled,
        "gov_rss_feeds": rss_sources,
    }


async def update_news_settings(
    session: AsyncSession,
    updates: dict[str, Any],
) -> dict[str, Any]:
    db = await _get_or_create_app_settings(session)
    if "gov_rss_enabled" in updates and "rss_enabled" not in updates:
        updates["rss_enabled"] = updates.get("gov_rss_enabled")
    if "gov_rss_feeds" in updates and "rss_sources" not in updates:
        updates["rss_sources"] = updates.get("gov_rss_feeds")
    if "custom_rss_feeds" in updates and "rss_feeds" not in updates:
        updates["rss_feeds"] = updates.get("custom_rss_feeds")

    mapping = {
        "enabled": "news_workflow_enabled",
        "auto_run": "news_workflow_auto_run",
        "scan_interval_seconds": "news_workflow_scan_interval_seconds",
        "top_k": "news_workflow_top_k",
        "rerank_top_n": "news_workflow_rerank_top_n",
        "similarity_threshold": "news_workflow_similarity_threshold",
        "keyword_weight": "news_workflow_keyword_weight",
        "semantic_weight": "news_workflow_semantic_weight",
        "event_weight": "news_workflow_event_weight",
        "require_verifier": "news_workflow_require_verifier",
        "market_min_liquidity": "news_workflow_market_min_liquidity",
        "market_max_days_to_resolution": "news_workflow_market_max_days_to_resolution",
        "min_keyword_signal": "news_workflow_min_keyword_signal",
        "min_semantic_signal": "news_workflow_min_semantic_signal",
        "min_edge_percent": "news_workflow_min_edge_percent",
        "min_confidence": "news_workflow_min_confidence",
        "require_second_source": "news_workflow_require_second_source",
        "orchestrator_enabled": "news_workflow_orchestrator_enabled",
        "orchestrator_min_edge": "news_workflow_orchestrator_min_edge",
        "orchestrator_max_age_minutes": "news_workflow_orchestrator_max_age_minutes",
        "model": "news_workflow_model",
        "cycle_spend_cap_usd": "news_workflow_cycle_spend_cap_usd",
        "hourly_spend_cap_usd": "news_workflow_hourly_spend_cap_usd",
        "cycle_llm_call_cap": "news_workflow_cycle_llm_call_cap",
        "cache_ttl_minutes": "news_workflow_cache_ttl_minutes",
        "max_edge_evals_per_article": "news_workflow_max_edge_evals_per_article",
        "rss_enabled": "news_gov_rss_enabled",
        "gov_rss_enabled": "news_gov_rss_enabled",
    }
    for key, value in updates.items():
        col = mapping.get(key)
        if not col:
            continue
        setattr(db, col, value)

    if "rss_feeds" in updates:
        db.news_rss_feeds_json = normalize_custom_rss_feeds(updates.get("rss_feeds"))
    if "rss_sources" in updates:
        db.news_gov_rss_feeds_json = normalize_gov_rss_feeds(updates.get("rss_sources"))
    elif "gov_rss_feeds" in updates:
        db.news_gov_rss_feeds_json = normalize_gov_rss_feeds(updates.get("gov_rss_feeds"))

    db.updated_at = utcnow()
    await session.commit()
    return await get_news_settings(session)
