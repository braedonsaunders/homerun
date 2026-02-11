"""Shared DB state for news workflow worker/API/autotrader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings as app_settings
from models.database import (
    AppSettings,
    NewsTradeIntent,
    NewsWorkflowControl,
    NewsWorkflowFinding,
    NewsWorkflowSnapshot,
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
            last_scan = datetime.utcnow()

    next_scan = status.get("next_scan")
    if has_next_scan and isinstance(next_scan, str):
        try:
            next_scan = _parse_iso_datetime(next_scan)
        except Exception:
            next_scan = None

    result = await session.execute(
        select(NewsWorkflowSnapshot).where(NewsWorkflowSnapshot.id == NEWS_SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = NewsWorkflowSnapshot(id=NEWS_SNAPSHOT_ID)
        session.add(row)

    row.updated_at = datetime.utcnow()
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
    result = await session.execute(
        select(NewsWorkflowSnapshot).where(NewsWorkflowSnapshot.id == NEWS_SNAPSHOT_ID)
    )
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
        control.get("requested_scan_at").isoformat()
        if control.get("requested_scan_at")
        else None
    )
    return status


async def ensure_news_control(session: AsyncSession) -> NewsWorkflowControl:
    result = await session.execute(
        select(NewsWorkflowControl).where(NewsWorkflowControl.id == NEWS_CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = NewsWorkflowControl(id=NEWS_CONTROL_ID)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_news_control(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(
        select(NewsWorkflowControl).where(NewsWorkflowControl.id == NEWS_CONTROL_ID)
    )
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
    row.updated_at = datetime.utcnow()
    await session.commit()


async def set_news_interval(session: AsyncSession, interval_seconds: int) -> None:
    row = await ensure_news_control(session)
    row.scan_interval_seconds = max(30, min(3600, int(interval_seconds)))
    row.updated_at = datetime.utcnow()
    await session.commit()


async def request_one_news_scan(session: AsyncSession) -> None:
    row = await ensure_news_control(session)
    row.requested_scan_at = datetime.utcnow()
    row.updated_at = datetime.utcnow()
    await session.commit()


async def clear_news_scan_request(session: AsyncSession) -> None:
    row = await ensure_news_control(session)
    row.requested_scan_at = None
    row.updated_at = datetime.utcnow()
    await session.commit()


async def try_acquire_news_lease(
    session: AsyncSession,
    owner: str,
    ttl_seconds: int,
) -> bool:
    """Try to acquire/renew the worker lease. Returns True if owned."""
    await ensure_news_control(session)
    now = datetime.utcnow()
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
            updated_at=datetime.utcnow(),
        )
    )
    await session.commit()


async def count_pending_news_intents(session: AsyncSession) -> int:
    result = await session.execute(
        select(func.count(NewsTradeIntent.id)).where(NewsTradeIntent.status == "pending")
    )
    return int(result.scalar() or 0)


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
    return list(result.scalars().all())


async def mark_news_intent(
    session: AsyncSession,
    intent_id: str,
    status: str,
) -> bool:
    result = await session.execute(
        select(NewsTradeIntent).where(NewsTradeIntent.id == intent_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return False
    row.status = status
    row.consumed_at = datetime.now(timezone.utc)
    if row.finding_id and status != "pending":
        finding = await session.execute(
            select(NewsWorkflowFinding).where(NewsWorkflowFinding.id == row.finding_id)
        )
        finding_row = finding.scalar_one_or_none()
        if finding_row is not None:
            finding_row.consumed_by_auto_trader = True
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
    if db is None:
        db = AppSettings(id="default")
        session.add(db)
        await session.commit()
        await session.refresh(db)
    return db


async def get_news_settings(session: AsyncSession) -> dict[str, Any]:
    db = await _get_or_create_app_settings(session)
    return {
        "enabled": bool(getattr(db, "news_workflow_enabled", True)),
        "auto_run": bool(getattr(db, "news_workflow_auto_run", True)),
        "scan_interval_seconds": int(
            getattr(db, "news_workflow_scan_interval_seconds", 120) or 120
        ),
        "top_k": int(getattr(db, "news_workflow_top_k", 8) or 8),
        "rerank_top_n": int(getattr(db, "news_workflow_rerank_top_n", 5) or 5),
        "similarity_threshold": float(
            getattr(db, "news_workflow_similarity_threshold", 0.42) or 0.42
        ),
        "keyword_weight": float(getattr(db, "news_workflow_keyword_weight", 0.25) or 0.25),
        "semantic_weight": float(
            getattr(db, "news_workflow_semantic_weight", 0.45) or 0.45
        ),
        "event_weight": float(getattr(db, "news_workflow_event_weight", 0.30) or 0.30),
        "require_verifier": bool(
            getattr(db, "news_workflow_require_verifier", True)
        ),
        "market_min_liquidity": float(
            getattr(db, "news_workflow_market_min_liquidity", 500.0) or 500.0
        ),
        "market_max_days_to_resolution": int(
            getattr(db, "news_workflow_market_max_days_to_resolution", 365) or 365
        ),
        "min_keyword_signal": float(
            getattr(db, "news_workflow_min_keyword_signal", 0.04) or 0.04
        ),
        "min_semantic_signal": float(
            getattr(db, "news_workflow_min_semantic_signal", 0.22) or 0.22
        ),
        "min_edge_percent": float(
            getattr(db, "news_workflow_min_edge_percent", 8.0) or 8.0
        ),
        "min_confidence": float(getattr(db, "news_workflow_min_confidence", 0.6) or 0.6),
        "require_second_source": bool(
            getattr(db, "news_workflow_require_second_source", False)
        ),
        "auto_trader_enabled": bool(
            getattr(db, "news_workflow_auto_trader_enabled", True)
        ),
        "auto_trader_min_edge": float(
            getattr(db, "news_workflow_auto_trader_min_edge", 10.0) or 10.0
        ),
        "auto_trader_max_age_minutes": int(
            getattr(db, "news_workflow_auto_trader_max_age_minutes", 120) or 120
        ),
        "model": getattr(db, "news_workflow_model", None),
        "article_max_age_hours": min(int(app_settings.NEWS_ARTICLE_TTL_HOURS), 48),
        "cycle_spend_cap_usd": float(
            getattr(db, "news_workflow_cycle_spend_cap_usd", 0.25) or 0.25
        ),
        "hourly_spend_cap_usd": float(
            getattr(db, "news_workflow_hourly_spend_cap_usd", 2.0) or 2.0
        ),
        "cycle_llm_call_cap": int(
            getattr(db, "news_workflow_cycle_llm_call_cap", 30) or 30
        ),
        "cache_ttl_minutes": int(
            getattr(db, "news_workflow_cache_ttl_minutes", 30) or 30
        ),
        "max_edge_evals_per_article": int(
            getattr(db, "news_workflow_max_edge_evals_per_article", 3) or 3
        ),
    }


async def update_news_settings(
    session: AsyncSession,
    updates: dict[str, Any],
) -> dict[str, Any]:
    db = await _get_or_create_app_settings(session)
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
        "auto_trader_enabled": "news_workflow_auto_trader_enabled",
        "auto_trader_min_edge": "news_workflow_auto_trader_min_edge",
        "auto_trader_max_age_minutes": "news_workflow_auto_trader_max_age_minutes",
        "model": "news_workflow_model",
        "cycle_spend_cap_usd": "news_workflow_cycle_spend_cap_usd",
        "hourly_spend_cap_usd": "news_workflow_hourly_spend_cap_usd",
        "cycle_llm_call_cap": "news_workflow_cycle_llm_call_cap",
        "cache_ttl_minutes": "news_workflow_cache_ttl_minutes",
        "max_edge_evals_per_article": "news_workflow_max_edge_evals_per_article",
    }
    for key, value in updates.items():
        col = mapping.get(key)
        if not col:
            continue
        setattr(db, col, value)
    db.updated_at = datetime.utcnow()
    await session.commit()
    return await get_news_settings(session)
