"""DB-backed GDELT DOC source for world-intelligence news pulse."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AppSettings, AsyncSessionLocal
from .country_catalog import country_catalog

logger = logging.getLogger(__name__)

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/2.0)"

_DEFAULT_QUERY_ROWS = [
    {
        "name": "Global Conflict Escalation",
        "query": "(war OR conflict OR invasion OR ceasefire OR sanctions)",
        "priority": "high",
        "country_iso3": "",
        "enabled": True,
    },
    {
        "name": "Maritime Chokepoints",
        "query": '("Suez Canal" OR "Strait of Hormuz" OR "Malacca Strait" OR "Panama Canal")',
        "priority": "high",
        "country_iso3": "",
        "enabled": True,
    },
    {
        "name": "Energy Disruption",
        "query": "(pipeline attack OR oil supply disruption OR LNG outage OR refinery fire)",
        "priority": "medium",
        "country_iso3": "",
        "enabled": True,
    },
    {
        "name": "State Security Moves",
        "query": "(military mobilization OR emergency decree OR missile launch OR naval exercise)",
        "priority": "medium",
        "country_iso3": "",
        "enabled": True,
    },
]
_ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}


def default_world_intel_gdelt_queries() -> list[dict[str, Any]]:
    return deepcopy(_DEFAULT_QUERY_ROWS)


def normalize_world_intel_gdelt_queries(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        if isinstance(row, str):
            row = {
                "name": row,
                "query": row,
                "priority": "medium",
                "country_iso3": "",
                "enabled": True,
            }
        if not isinstance(row, dict):
            continue

        query = str(row.get("query") or "").strip()
        if not query:
            continue
        dedupe_key = query.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        name = str(row.get("name") or query).strip()[:120]
        priority = str(row.get("priority") or "medium").strip().lower()
        if priority not in _ALLOWED_PRIORITIES:
            priority = "medium"
        enabled = bool(row.get("enabled", True))
        country_iso3 = country_catalog.normalize_iso3(str(row.get("country_iso3") or ""))

        out.append(
            {
                "name": name,
                "query": query,
                "priority": priority,
                "country_iso3": country_iso3,
                "enabled": enabled,
            }
        )
    return out


@dataclass
class GDELTWorldArticle:
    article_id: str
    title: str
    url: str
    source: str
    query: str
    priority: str
    country_iso3: str = ""
    published: Optional[datetime] = None
    summary: str = ""
    tone: float = 0.0
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GDELTWorldNewsService:
    def __init__(self) -> None:
        self._articles: dict[str, GDELTWorldArticle] = {}
        self._seen_by_consumer: dict[str, set[str]] = {}
        self._last_fetch_at: Optional[datetime] = None
        self._last_errors: list[str] = []
        self._failed_queries: int = 0
        self._configured_queries_count: int = 0
        self._enabled: bool = bool(
            getattr(settings, "WORLD_INTEL_GDELT_NEWS_ENABLED", True)
        )
        self._timespan_hours: int = int(
            max(1, getattr(settings, "WORLD_INTEL_GDELT_NEWS_TIMESPAN_HOURS", 6) or 6)
        )
        self._max_records: int = int(
            max(10, getattr(settings, "WORLD_INTEL_GDELT_NEWS_MAX_RECORDS", 40) or 40)
        )
        self._request_timeout: float = float(
            max(
                5.0,
                getattr(
                    settings,
                    "WORLD_INTEL_GDELT_NEWS_REQUEST_TIMEOUT_SECONDS",
                    20.0,
                )
                or 20.0,
            )
        )
        self._cache_ttl_seconds: int = int(
            max(
                15,
                getattr(settings, "WORLD_INTEL_GDELT_NEWS_CACHE_SECONDS", 300) or 300,
            )
        )
        self._query_delay_seconds: float = float(
            max(
                1.0,
                getattr(settings, "WORLD_INTEL_GDELT_NEWS_QUERY_DELAY_SECONDS", 5.0)
                or 5.0,
            )
        )
        self._query_lock = asyncio.Lock()
        self._last_query_ts = 0.0
        self._cache_articles: list[GDELTWorldArticle] = []
        self._cache_refreshed_at: Optional[datetime] = None
        self._active_source: str = "gdelt_doc_api"

    async def _load_configuration(self) -> tuple[bool, list[dict[str, Any]], int, int, str]:
        default_enabled = bool(getattr(settings, "WORLD_INTEL_GDELT_NEWS_ENABLED", True))
        default_queries = default_world_intel_gdelt_queries()
        default_timespan = int(
            max(1, getattr(settings, "WORLD_INTEL_GDELT_NEWS_TIMESPAN_HOURS", 6) or 6)
        )
        default_max_records = int(
            max(10, getattr(settings, "WORLD_INTEL_GDELT_NEWS_MAX_RECORDS", 40) or 40)
        )

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AppSettings).where(AppSettings.id == "default")
                )
                row = result.scalar_one_or_none()
                if row is None:
                    row = AppSettings(
                        id="default",
                        world_intel_gdelt_news_enabled=default_enabled,
                        world_intel_gdelt_news_queries_json=default_queries,
                        world_intel_gdelt_news_timespan_hours=default_timespan,
                        world_intel_gdelt_news_max_records=default_max_records,
                        world_intel_gdelt_news_source="gdelt_doc_seed",
                    )
                    session.add(row)
                    await session.commit()
                    await session.refresh(row)

                raw_enabled = getattr(row, "world_intel_gdelt_news_enabled", None)
                enabled = default_enabled if raw_enabled is None else bool(raw_enabled)
                rows = normalize_world_intel_gdelt_queries(
                    getattr(row, "world_intel_gdelt_news_queries_json", None)
                )
                if not rows:
                    rows = default_queries
                    row.world_intel_gdelt_news_queries_json = rows
                    row.world_intel_gdelt_news_source = (
                        str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip()
                        or "gdelt_doc_seed"
                    )
                    await session.commit()

                source = (
                    str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip()
                    or "gdelt_doc_seed"
                )
                raw_timespan = getattr(row, "world_intel_gdelt_news_timespan_hours", None)
                raw_max_records = getattr(row, "world_intel_gdelt_news_max_records", None)
        except Exception as exc:
            logger.debug("World GDELT config DB read failed, using defaults: %s", exc)
            self._enabled = default_enabled
            self._configured_queries_count = len(default_queries)
            self._timespan_hours = default_timespan
            self._max_records = default_max_records
            self._active_source = "gdelt_doc_seed"
            return (
                default_enabled,
                default_queries,
                default_timespan,
                default_max_records,
                "gdelt_doc_seed",
            )

        try:
            timespan = int(raw_timespan)
        except Exception:
            timespan = default_timespan
        timespan = max(1, timespan)
        try:
            max_records = int(raw_max_records)
        except Exception:
            max_records = default_max_records
        max_records = max(10, max_records)

        self._enabled = enabled
        self._configured_queries_count = len(rows)
        self._timespan_hours = timespan
        self._max_records = max_records
        self._active_source = source
        return enabled, rows, timespan, max_records, source

    def _cache_fresh(self) -> bool:
        if not self._cache_articles or not isinstance(self._cache_refreshed_at, datetime):
            return False
        age = datetime.now(timezone.utc) - self._cache_refreshed_at
        return age.total_seconds() < self._cache_ttl_seconds

    async def _throttle_query(self) -> None:
        if self._query_delay_seconds <= 0:
            return
        async with self._query_lock:
            now = time.monotonic()
            wait = self._query_delay_seconds - (now - self._last_query_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_query_ts = time.monotonic()

    async def _fetch_query(
        self,
        row: dict[str, Any],
        *,
        timespan_hours: int,
        max_records: int,
    ) -> list[GDELTWorldArticle]:
        query = str(row.get("query") or "").strip()
        if not query:
            return []

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "timespan": f"{timespan_hours}h",
            "maxrecords": str(max_records),
        }
        headers = {"User-Agent": _USER_AGENT}

        try:
            await self._throttle_query()
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                resp = await client.get(_GDELT_DOC_API, params=params, headers=headers)
                if resp.status_code == 429:
                    self._last_errors.append(f"{row.get('name') or query}: HTTP 429")
                    return []
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            self._last_errors.append(f"{row.get('name') or query}: {exc}")
            return []

        raw_articles = payload.get("articles", []) if isinstance(payload, dict) else []
        if not isinstance(raw_articles, list):
            return []

        out: list[GDELTWorldArticle] = []
        for raw in raw_articles[:max_records]:
            article = self._parse_article(raw, row)
            if article is not None:
                out.append(article)
        return out

    def _parse_article(
        self,
        raw: Any,
        row: dict[str, Any],
    ) -> Optional[GDELTWorldArticle]:
        if not isinstance(raw, dict):
            return None

        url = str(raw.get("url") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not url or not title:
            return None

        source = str(raw.get("domain") or raw.get("source") or "GDELT").strip()
        published = _parse_gdelt_date(raw.get("seendate"))
        tone = _parse_tone(raw.get("tone"))
        base_priority = str(row.get("priority") or "medium").strip().lower()
        priority = _priority_from_tone(base_priority, tone)

        summary = str(raw.get("snippet") or raw.get("description") or "").strip()
        if summary and summary.lower().startswith("http"):
            summary = ""
        summary = summary[:500]

        article_country = ""
        for key in ("sourcecountry", "sourceCountry", "country"):
            article_country = country_catalog.normalize_iso3(str(raw.get(key) or ""))
            if article_country:
                break
        if not article_country:
            article_country = country_catalog.normalize_iso3(str(row.get("country_iso3") or ""))

        query_name = str(row.get("name") or row.get("query") or "").strip()
        article_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return GDELTWorldArticle(
            article_id=article_id,
            title=title,
            url=url,
            source=source,
            query=query_name,
            priority=priority,
            country_iso3=article_country,
            published=published,
            summary=summary,
            tone=tone,
        )

    async def fetch_all(
        self,
        *,
        consumer: str = "world_intelligence",
        force: bool = False,
    ) -> list[GDELTWorldArticle]:
        consumer_key = str(consumer or "world_intelligence").strip().lower() or "world_intelligence"
        seen = self._seen_by_consumer.setdefault(consumer_key, set())

        enabled, rows, timespan_hours, max_records, source = await self._load_configuration()
        self._active_source = source
        if not enabled:
            self._last_errors = ["disabled"]
            self._failed_queries = 0
            return []

        enabled_rows = [row for row in rows if bool(row.get("enabled", True))]
        if not enabled_rows:
            self._last_errors = ["No enabled GDELT world-intel queries"]
            self._failed_queries = 0
            return []

        self._last_errors = []
        self._failed_queries = 0

        articles: list[GDELTWorldArticle]
        if not force and self._cache_fresh():
            articles = list(self._cache_articles)
        else:
            merged: dict[str, GDELTWorldArticle] = {}
            for row in enabled_rows:
                try:
                    result = await self._fetch_query(
                        row,
                        timespan_hours=timespan_hours,
                        max_records=max_records,
                    )
                except Exception as exc:
                    self._failed_queries += 1
                    self._last_errors.append(str(exc))
                    continue
                for article in result:
                    if not isinstance(article, GDELTWorldArticle):
                        continue
                    existing = merged.get(article.article_id)
                    if existing is None:
                        merged[article.article_id] = article
                        continue
                    existing_ts = existing.published or existing.fetched_at
                    candidate_ts = article.published or article.fetched_at
                    if candidate_ts > existing_ts:
                        merged[article.article_id] = article

            articles = sorted(
                merged.values(),
                key=lambda item: item.published or item.fetched_at,
                reverse=True,
            )
            self._cache_articles = list(articles)
            self._cache_refreshed_at = datetime.now(timezone.utc)

        new_articles: list[GDELTWorldArticle] = []
        for article in articles:
            self._articles[article.article_id] = article
            if article.article_id not in seen:
                seen.add(article.article_id)
                new_articles.append(article)

        self._last_fetch_at = datetime.now(timezone.utc)
        self._prune_old_articles()
        return new_articles

    def get_health(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "configured_queries": self._configured_queries_count,
            "tracked_articles": len(self._articles),
            "consumer_count": len(self._seen_by_consumer),
            "failed_queries": self._failed_queries,
            "timespan_hours": self._timespan_hours,
            "max_records": self._max_records,
            "source": self._active_source,
            "last_fetch_at": self._last_fetch_at.isoformat() if self._last_fetch_at else None,
            "last_error": self._last_errors[0] if self._last_errors else None,
            "recent_errors": self._last_errors[:5],
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "cache_fresh": self._cache_fresh(),
        }

    def _prune_old_articles(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
        valid_ids = {
            article_id
            for article_id, article in self._articles.items()
            if article.fetched_at >= cutoff
        }
        self._articles = {
            article_id: article
            for article_id, article in self._articles.items()
            if article_id in valid_ids
        }
        stale_consumers: list[str] = []
        for consumer, seen in self._seen_by_consumer.items():
            seen.intersection_update(valid_ids)
            if not seen:
                stale_consumers.append(consumer)
        for consumer in stale_consumers:
            self._seen_by_consumer.pop(consumer, None)


def _parse_tone(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        text = str(value).strip()
        if "," in text:
            text = text.split(",", 1)[0].strip()
        return float(text)
    except Exception:
        return 0.0


def _priority_from_tone(base_priority: str, tone: float) -> str:
    base = str(base_priority or "medium").strip().lower()
    if base not in _ALLOWED_PRIORITIES:
        base = "medium"
    if tone <= -8.0:
        return "critical"
    if tone <= -4.0 and base in {"medium", "low"}:
        return "high"
    return base


def _parse_gdelt_date(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = text.replace("Z", "").replace("z", "")
    formats = ("%Y%m%dT%H%M%S", "%Y%m%d%H%M%S")
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        row = AppSettings(id="default")
        row.world_intel_gdelt_news_enabled = bool(
            getattr(settings, "WORLD_INTEL_GDELT_NEWS_ENABLED", True)
        )
        row.world_intel_gdelt_news_queries_json = default_world_intel_gdelt_queries()
        row.world_intel_gdelt_news_timespan_hours = int(
            max(1, getattr(settings, "WORLD_INTEL_GDELT_NEWS_TIMESPAN_HOURS", 6) or 6)
        )
        row.world_intel_gdelt_news_max_records = int(
            max(10, getattr(settings, "WORLD_INTEL_GDELT_NEWS_MAX_RECORDS", 40) or 40)
        )
        row.world_intel_gdelt_news_source = "gdelt_doc_seed"
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def load_gdelt_news_config_from_db(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    rows = normalize_world_intel_gdelt_queries(
        getattr(row, "world_intel_gdelt_news_queries_json", None)
    )
    if not rows:
        rows = default_world_intel_gdelt_queries()
        row.world_intel_gdelt_news_queries_json = rows
        row.world_intel_gdelt_news_source = (
            str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip()
            or "gdelt_doc_seed"
        )
        await session.commit()

    enabled = getattr(row, "world_intel_gdelt_news_enabled", None)
    if enabled is None:
        enabled = bool(getattr(settings, "WORLD_INTEL_GDELT_NEWS_ENABLED", True))
    source = str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip() or "gdelt_doc_seed"
    try:
        timespan = int(getattr(row, "world_intel_gdelt_news_timespan_hours", 6) or 6)
    except Exception:
        timespan = 6
    try:
        max_records = int(getattr(row, "world_intel_gdelt_news_max_records", 40) or 40)
    except Exception:
        max_records = 40

    gdelt_world_news_service._enabled = bool(enabled)
    gdelt_world_news_service._configured_queries_count = len(rows)
    gdelt_world_news_service._timespan_hours = max(1, timespan)
    gdelt_world_news_service._max_records = max(10, max_records)
    gdelt_world_news_service._active_source = source

    return {
        "source": source,
        "enabled": bool(enabled),
        "queries": len(rows),
        "timespan_hours": max(1, timespan),
        "max_records": max(10, max_records),
        "last_synced_at": (
            row.world_intel_gdelt_news_synced_at.isoformat()
            if isinstance(row.world_intel_gdelt_news_synced_at, datetime)
            else None
        ),
    }


async def sync_gdelt_news_from_source(
    session: AsyncSession,
    *,
    force: bool = False,
) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)

    enabled = bool(getattr(settings, "WORLD_INTEL_GDELT_NEWS_SYNC_ENABLED", True))
    interval_hours = int(
        max(1, getattr(settings, "WORLD_INTEL_GDELT_NEWS_SYNC_HOURS", 1) or 1)
    )
    last_sync = getattr(row, "world_intel_gdelt_news_synced_at", None)

    due = force or last_sync is None
    if not due and isinstance(last_sync, datetime):
        last_sync_utc = (
            last_sync.astimezone(timezone.utc).replace(tzinfo=None)
            if last_sync.tzinfo is not None
            else last_sync
        )
        due = (datetime.now(timezone.utc).replace(tzinfo=None) - last_sync_utc) >= timedelta(
            hours=interval_hours
        )

    if not due:
        health = gdelt_world_news_service.get_health()
        return {
            "updated": False,
            "reason": "fresh",
            "source": str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip()
            or "gdelt_doc_seed",
            "fetched_articles": 0,
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
            "health": health,
        }

    if not enabled and not force:
        health = gdelt_world_news_service.get_health()
        return {
            "updated": False,
            "reason": "disabled",
            "source": str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip()
            or "gdelt_doc_seed",
            "fetched_articles": 0,
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
            "health": health,
        }

    fetched = await gdelt_world_news_service.fetch_all(
        consumer="world_intel_sync",
        force=True,
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row.world_intel_gdelt_news_source = "gdelt_doc_api"
    row.world_intel_gdelt_news_synced_at = now
    await session.commit()
    gdelt_world_news_service._active_source = "gdelt_doc_api"

    health = gdelt_world_news_service.get_health()
    return {
        "updated": True,
        "reason": "synced",
        "source": "gdelt_doc_api",
        "fetched_articles": len(fetched),
        "last_synced_at": now.isoformat(),
        "health": health,
    }


async def get_gdelt_news_source_status(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    rows = normalize_world_intel_gdelt_queries(
        getattr(row, "world_intel_gdelt_news_queries_json", None)
    )
    if not rows:
        rows = default_world_intel_gdelt_queries()

    source = str(getattr(row, "world_intel_gdelt_news_source", "") or "").strip() or "gdelt_doc_seed"
    last_sync = getattr(row, "world_intel_gdelt_news_synced_at", None)
    health = gdelt_world_news_service.get_health()
    return {
        "source": source,
        "enabled": bool(
            getattr(row, "world_intel_gdelt_news_enabled", True)
        ),
        "queries": len(rows),
        "timespan_hours": int(
            max(1, getattr(row, "world_intel_gdelt_news_timespan_hours", 6) or 6)
        ),
        "max_records": int(
            max(10, getattr(row, "world_intel_gdelt_news_max_records", 40) or 40)
        ),
        "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        "sync_enabled": bool(getattr(settings, "WORLD_INTEL_GDELT_NEWS_SYNC_ENABLED", True)),
        "sync_interval_hours": int(
            max(1, getattr(settings, "WORLD_INTEL_GDELT_NEWS_SYNC_HOURS", 1) or 1)
        ),
        "health": health,
    }


gdelt_world_news_service = GDELTWorldNewsService()
