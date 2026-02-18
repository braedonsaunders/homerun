from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import DataSource, DataSourceTombstone
from utils.logger import get_logger

logger = get_logger(__name__)

_SEED_MARKER = "# System data source seed"
_LEGACY_SYSTEM_SOURCE_KEYS = {"news", "world_intelligence"}


@dataclass(frozen=True)
class SystemDataSourceSeed:
    slug: str
    source_key: str
    source_kind: str
    name: str
    description: str
    source_code: str
    sort_order: int
    config: dict
    config_schema: dict
    enabled: bool = True


def _now_or_default(now: datetime | None) -> datetime:
    return now or datetime.utcnow()


NEWS_FEED_BRIDGE_SOURCE_CODE = '''# System data source seed
from services.data_source_sdk import BaseDataSource


class NewsFeedBridgeSource(BaseDataSource):
    name = "News Feed Bridge"
    description = "Exports normalized records from the in-memory news feed cache"

    default_config = {
        "max_age_hours": 168,
        "limit": 500,
    }

    async def fetch_async(self):
        from services.news.feed_service import news_feed_service

        max_age_hours = int(self.config.get("max_age_hours") or 168)
        limit = int(self.config.get("limit") or 500)
        out = []

        for article in news_feed_service.get_articles(max_age_hours=max_age_hours)[:limit]:
            published = getattr(article, "published", None)
            fetched_at = getattr(article, "fetched_at", None)
            out.append(
                {
                    "external_id": str(getattr(article, "article_id", "") or "").strip(),
                    "title": str(getattr(article, "title", "") or "").strip(),
                    "summary": str(getattr(article, "summary", "") or "").strip(),
                    "category": str(getattr(article, "category", "news") or "news").strip().lower(),
                    "source": str(getattr(article, "source", "") or "").strip() or "news",
                    "url": str(getattr(article, "url", "") or "").strip() or None,
                    "observed_at": published or fetched_at,
                    "payload": {
                        "feed_source": str(getattr(article, "feed_source", "") or "").strip().lower(),
                        "published": published.isoformat() if published else None,
                        "fetched_at": fetched_at.isoformat() if fetched_at else None,
                    },
                    "tags": [
                        "news",
                        str(getattr(article, "feed_source", "") or "").strip().lower() or "unknown",
                    ],
                    "geotagged": False,
                }
            )

        return out
'''


RSS_NEWS_SOURCE_CODE = '''# System data source seed
from services.data_source_sdk import BaseDataSource


class RssNewsSource(BaseDataSource):
    name = "RSS News Source"
    description = "Exports RSS-only records from the shared news feed cache"

    default_config = {
        "feed_sources": ["rss", "custom_rss"],
        "source_names": [],
        "categories": [],
        "max_age_hours": 168,
        "limit": 500,
    }

    async def fetch_async(self):
        from services.news.feed_service import news_feed_service

        feed_sources = {
            str(v or "").strip().lower()
            for v in (self.config.get("feed_sources") or [])
            if str(v or "").strip()
        }
        source_names = {
            str(v or "").strip().lower()
            for v in (self.config.get("source_names") or [])
            if str(v or "").strip()
        }
        categories = {
            str(v or "").strip().lower()
            for v in (self.config.get("categories") or [])
            if str(v or "").strip()
        }
        max_age_hours = int(self.config.get("max_age_hours") or 168)
        limit = int(self.config.get("limit") or 500)

        rows = []
        for article in news_feed_service.get_articles(max_age_hours=max_age_hours):
            article_feed_source = str(getattr(article, "feed_source", "") or "").strip().lower()
            article_source = str(getattr(article, "source", "") or "").strip().lower()
            article_category = str(getattr(article, "category", "") or "").strip().lower()

            if feed_sources and article_feed_source not in feed_sources:
                continue
            if source_names and article_source not in source_names:
                continue
            if categories and article_category not in categories:
                continue

            published = getattr(article, "published", None)
            fetched_at = getattr(article, "fetched_at", None)
            rows.append(
                {
                    "external_id": str(getattr(article, "article_id", "") or "").strip(),
                    "title": str(getattr(article, "title", "") or "").strip(),
                    "summary": str(getattr(article, "summary", "") or "").strip(),
                    "category": str(getattr(article, "category", "news") or "news").strip().lower(),
                    "source": str(getattr(article, "source", "") or "").strip() or "news",
                    "url": str(getattr(article, "url", "") or "").strip() or None,
                    "observed_at": published or fetched_at,
                    "payload": {
                        "source_name": article_source or None,
                        "category": article_category or None,
                        "feed_source": article_feed_source,
                        "published": published.isoformat() if published else None,
                        "fetched_at": fetched_at.isoformat() if fetched_at else None,
                    },
                    "tags": ["news", "rss", article_feed_source or "rss"],
                    "geotagged": False,
                }
            )
            if len(rows) >= limit:
                break

        return rows
'''


GDELT_WORLD_NEWS_SOURCE_CODE = '''# System data source seed
from services.data_source_sdk import BaseDataSource


class GdeltWorldNewsSource(BaseDataSource):
    name = "GDELT World News Source"
    description = "Fetches world-news records directly from the GDELT DOC API service"

    default_config = {
        "consumer": "data_source_gdelt",
        "force_refresh": False,
        "limit": 200,
        "priorities": [],
        "country_iso3": [],
    }

    async def fetch_async(self):
        from services.world_intelligence.gdelt_news_source import gdelt_world_news_service

        consumer = str(self.config.get("consumer") or "data_source_gdelt").strip().lower() or "data_source_gdelt"
        force_refresh = bool(self.config.get("force_refresh", False))
        limit = int(self.config.get("limit") or 200)
        priorities = {
            str(v or "").strip().lower()
            for v in (self.config.get("priorities") or [])
            if str(v or "").strip()
        }
        country_iso3 = {
            str(v or "").strip().upper()
            for v in (self.config.get("country_iso3") or [])
            if str(v or "").strip()
        }

        rows = []
        articles = await gdelt_world_news_service.fetch_all(consumer=consumer, force=force_refresh)
        for article in articles:
            priority = str(getattr(article, "priority", "") or "").strip().lower()
            iso3 = str(getattr(article, "country_iso3", "") or "").strip().upper()
            if priorities and priority not in priorities:
                continue
            if country_iso3 and iso3 not in country_iso3:
                continue

            published = getattr(article, "published", None)
            fetched_at = getattr(article, "fetched_at", None)
            rows.append(
                {
                    "external_id": str(getattr(article, "article_id", "") or "").strip(),
                    "title": str(getattr(article, "title", "") or "").strip(),
                    "summary": str(getattr(article, "summary", "") or "").strip(),
                    "category": "news",
                    "source": str(getattr(article, "source", "") or "").strip() or "gdelt",
                    "url": str(getattr(article, "url", "") or "").strip() or None,
                    "observed_at": published or fetched_at,
                    "payload": {
                        "query": str(getattr(article, "query", "") or "").strip() or None,
                        "priority": priority or None,
                        "country_iso3": iso3 or None,
                        "tone": float(getattr(article, "tone", 0.0) or 0.0),
                        "published": published.isoformat() if published else None,
                        "fetched_at": fetched_at.isoformat() if fetched_at else None,
                    },
                    "tags": ["news", "gdelt", priority or "medium"],
                    "geotagged": False,
                }
            )
            if len(rows) >= limit:
                break

        return rows
'''


WORLD_SIGNALS_BRIDGE_SOURCE_CODE = '''# System data source seed
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, select

from models.database import AsyncSessionLocal, WorldIntelligenceSignal
from services.data_source_sdk import BaseDataSource


class WorldSignalsBridgeSource(BaseDataSource):
    name = "World Intelligence Signals"
    description = "Exports geotagged world-intelligence signals from the DB"

    default_config = {
        "hours": 72,
        "limit": 1500,
        "min_severity": 0.0,
        "source_names": [],
        "signal_types": [],
    }

    async def fetch_async(self):
        hours = int(self.config.get("hours") or 72)
        limit = int(self.config.get("limit") or 1500)
        min_severity = float(self.config.get("min_severity") or 0.0)
        source_names = [str(v).strip().lower() for v in (self.config.get("source_names") or []) if str(v).strip()]
        signal_types = [str(v).strip().lower() for v in (self.config.get("signal_types") or []) if str(v).strip()]

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
        async with AsyncSessionLocal() as session:
            query = (
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .where(WorldIntelligenceSignal.severity >= min_severity)
                .order_by(desc(WorldIntelligenceSignal.detected_at), desc(WorldIntelligenceSignal.severity))
                .limit(max(1, limit))
            )
            if source_names:
                query = query.where(WorldIntelligenceSignal.source.in_(source_names))
            if signal_types:
                query = query.where(WorldIntelligenceSignal.signal_type.in_(signal_types))

            rows = (await session.execute(query)).scalars().all()

        out = []
        for row in rows:
            metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
            out.append(
                {
                    "external_id": str(row.id or "").strip(),
                    "title": str(row.title or "").strip() or "World signal",
                    "summary": str(row.description or "").strip(),
                    "category": str(row.signal_type or "world").strip().lower(),
                    "source": str(row.source or "world").strip().lower(),
                    "url": str(metadata.get("url") or "").strip() or None,
                    "observed_at": row.detected_at,
                    "payload": {
                        "signal_type": row.signal_type,
                        "severity": float(row.severity or 0.0),
                        "related_market_ids": list(row.related_market_ids or []),
                        "market_relevance_score": row.market_relevance_score,
                        "metadata": metadata,
                    },
                    "geotagged": row.latitude is not None and row.longitude is not None,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "country_iso3": str(row.iso3 or "").strip().upper() or None,
                    "tags": ["world", str(row.signal_type or "signal").strip().lower()],
                }
            )

        return out
'''


BASE_SYSTEM_DATA_SOURCE_SEEDS: list[SystemDataSourceSeed] = [
    SystemDataSourceSeed(
        slug="events_all",
        source_key="events",
        source_kind="bridge",
        name="Events: All",
        description="Unified export of all world-intelligence events",
        source_code=WORLD_SIGNALS_BRIDGE_SOURCE_CODE,
        sort_order=10,
        config={
            "hours": 72,
            "limit": 1500,
            "min_severity": 0.0,
            "source_names": [],
            "signal_types": [],
        },
        config_schema={
            "param_fields": [
                {"key": "hours", "label": "Lookback (hours)", "type": "integer", "min": 1, "max": 720},
                {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                {"key": "min_severity", "label": "Min Severity", "type": "number", "min": 0, "max": 1},
                {"key": "source_names", "label": "Source Names", "type": "list"},
                {"key": "signal_types", "label": "Signal Types", "type": "list"},
            ]
        },
        enabled=True,
    ),
    SystemDataSourceSeed(
        slug="stories_all",
        source_key="stories",
        source_kind="bridge",
        name="Stories: All",
        description="Unified export of all cached stories",
        source_code=NEWS_FEED_BRIDGE_SOURCE_CODE,
        sort_order=20,
        config={
            "max_age_hours": 168,
            "limit": 500,
        },
        config_schema={
            "param_fields": [
                {"key": "max_age_hours", "label": "Max Age (hours)", "type": "integer", "min": 1, "max": 336},
                {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
            ]
        },
        enabled=True,
    ),
    SystemDataSourceSeed(
        slug="stories_rss",
        source_key="stories",
        source_kind="rss",
        name="Stories: RSS",
        description="Prebuilt RSS story source from the shared news cache",
        source_code=RSS_NEWS_SOURCE_CODE,
        sort_order=30,
        config={
            "feed_sources": ["rss", "custom_rss"],
            "source_names": [],
            "categories": [],
            "max_age_hours": 168,
            "limit": 500,
        },
        config_schema={
            "param_fields": [
                {"key": "feed_sources", "label": "Feed Sources", "type": "list"},
                {"key": "source_names", "label": "Source Names", "type": "list"},
                {"key": "categories", "label": "Categories", "type": "list"},
                {"key": "max_age_hours", "label": "Max Age (hours)", "type": "integer", "min": 1, "max": 336},
                {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
            ]
        },
        enabled=True,
    ),
    SystemDataSourceSeed(
        slug="stories_gdelt",
        source_key="stories",
        source_kind="gdelt",
        name="Stories: GDELT",
        description="Prebuilt GDELT story source",
        source_code=GDELT_WORLD_NEWS_SOURCE_CODE,
        sort_order=40,
        config={
            "consumer": "data_source_gdelt",
            "force_refresh": False,
            "limit": 200,
            "priorities": [],
            "country_iso3": [],
        },
        config_schema={
            "param_fields": [
                {"key": "consumer", "label": "Consumer Key", "type": "string"},
                {"key": "force_refresh", "label": "Force Refresh", "type": "boolean"},
                {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                {"key": "priorities", "label": "Priorities", "type": "list"},
                {"key": "country_iso3", "label": "Country ISO3", "type": "list"},
            ]
        },
        enabled=True,
    ),
]

_EVENT_SOURCE_DEFS: list[dict] = [
    {
        "slug": "events_acled",
        "name": "Events: ACLED",
        "description": "Conflict and instability events from ACLED",
        "source_names": ["acled"],
        "signal_types": ["conflict", "instability", "tension"],
    },
    {
        "slug": "events_gdelt_tensions",
        "name": "Events: GDELT Tensions",
        "description": "Diplomatic tension events from GDELT",
        "source_names": ["gdelt_tensions"],
        "signal_types": ["tension"],
    },
    {
        "slug": "events_military",
        "name": "Events: Military",
        "description": "Military movement and posture events",
        "source_names": ["military"],
        "signal_types": ["military"],
    },
    {
        "slug": "events_infrastructure",
        "name": "Events: Infrastructure",
        "description": "Infrastructure disruption events",
        "source_names": ["infrastructure"],
        "signal_types": ["infrastructure"],
    },
    {
        "slug": "events_gdelt_news",
        "name": "Events: GDELT News",
        "description": "News-derived world events from GDELT",
        "source_names": ["gdelt_news"],
        "signal_types": ["news"],
    },
    {
        "slug": "events_usgs",
        "name": "Events: USGS",
        "description": "Earthquake events from USGS",
        "source_names": ["usgs"],
        "signal_types": ["earthquake"],
    },
    {
        "slug": "events_rss_news",
        "name": "Events: RSS News",
        "description": "News-derived world events from RSS stories",
        "source_names": ["rss_news"],
        "signal_types": ["news"],
    },
    {
        "slug": "events_chokepoints",
        "name": "Events: Chokepoints",
        "description": "Chokepoint and route disruption events",
        "source_names": ["chokepoints"],
        "signal_types": ["infrastructure", "convergence"],
    },
    {
        "slug": "events_convergence",
        "name": "Events: Convergence",
        "description": "Cross-source convergence events",
        "source_names": ["convergence"],
        "signal_types": ["convergence"],
    },
    {
        "slug": "events_instability",
        "name": "Events: Instability",
        "description": "Instability-scored macro events",
        "source_names": ["instability"],
        "signal_types": ["instability"],
    },
    {
        "slug": "events_anomaly",
        "name": "Events: Anomaly",
        "description": "Anomaly-detected world events",
        "source_names": ["anomaly"],
        "signal_types": ["anomaly"],
    },
]

_STORY_SOURCE_DEFS: list[dict] = [
    {
        "slug": "stories_google_news",
        "name": "Stories: Google News",
        "description": "Stories from Google News RSS",
        "feed_sources": ["google_news"],
    },
    {
        "slug": "stories_custom_rss",
        "name": "Stories: Custom RSS",
        "description": "Stories from configured custom RSS feeds",
        "feed_sources": ["custom_rss"],
    },
    {
        "slug": "stories_gov_rss",
        "name": "Stories: Gov RSS",
        "description": "Stories from government RSS feeds",
        "feed_sources": ["rss"],
    },
]

for index, row in enumerate(_EVENT_SOURCE_DEFS):
    BASE_SYSTEM_DATA_SOURCE_SEEDS.append(
        SystemDataSourceSeed(
            slug=str(row["slug"]),
            source_key="events",
            source_kind="bridge",
            name=str(row["name"]),
            description=str(row["description"]),
            source_code=WORLD_SIGNALS_BRIDGE_SOURCE_CODE,
            sort_order=100 + index,
            config={
                "hours": 168,
                "limit": 1500,
                "min_severity": 0.0,
                "source_names": list(row.get("source_names") or []),
                "signal_types": list(row.get("signal_types") or []),
            },
            config_schema={
                "param_fields": [
                    {"key": "hours", "label": "Lookback (hours)", "type": "integer", "min": 1, "max": 720},
                    {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                    {"key": "min_severity", "label": "Min Severity", "type": "number", "min": 0, "max": 1},
                    {"key": "source_names", "label": "Source Names", "type": "list"},
                    {"key": "signal_types", "label": "Signal Types", "type": "list"},
                ]
            },
            enabled=True,
        )
    )

for index, row in enumerate(_STORY_SOURCE_DEFS):
    BASE_SYSTEM_DATA_SOURCE_SEEDS.append(
        SystemDataSourceSeed(
            slug=str(row["slug"]),
            source_key="stories",
            source_kind="rss",
            name=str(row["name"]),
            description=str(row["description"]),
            source_code=RSS_NEWS_SOURCE_CODE,
            sort_order=200 + index,
            config={
                "feed_sources": list(row.get("feed_sources") or []),
                "source_names": [],
                "categories": [],
                "max_age_hours": 168,
                "limit": 500,
            },
            config_schema={
                "param_fields": [
                    {"key": "feed_sources", "label": "Feed Sources", "type": "list"},
                    {"key": "source_names", "label": "Source Names", "type": "list"},
                    {"key": "categories", "label": "Categories", "type": "list"},
                    {"key": "max_age_hours", "label": "Max Age (hours)", "type": "integer", "min": 1, "max": 336},
                    {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                ]
            },
            enabled=True,
        )
    )


def _is_seed_source_code(source_code: str) -> bool:
    return _SEED_MARKER in str(source_code or "")


def _is_legacy_system_source_row(row: DataSource) -> bool:
    source_key = str(row.source_key or "").strip().lower()
    return source_key in _LEGACY_SYSTEM_SOURCE_KEYS


def _build_row_from_seed(seed: SystemDataSourceSeed, ts: datetime) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "slug": seed.slug,
        "source_key": seed.source_key,
        "source_kind": seed.source_kind,
        "name": seed.name,
        "description": seed.description,
        "source_code": seed.source_code,
        "class_name": None,
        "is_system": True,
        "enabled": bool(seed.enabled),
        "status": "unloaded",
        "error_message": None,
        "config": dict(seed.config or {}),
        "config_schema": dict(seed.config_schema or {}),
        "version": 1,
        "sort_order": int(seed.sort_order),
        "created_at": ts,
        "updated_at": ts,
    }


def build_system_data_source_rows(*, now: datetime | None = None) -> list[dict]:
    ts = _now_or_default(now)
    rows: list[dict] = []

    for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS:
        rows.append(_build_row_from_seed(seed, ts))

    return rows


async def ensure_system_data_sources_seeded(session: AsyncSession) -> int:
    rows = build_system_data_source_rows()
    seed_by_slug = {row["slug"]: row for row in rows}

    try:
        tombstoned_slugs = set(
            (
                await session.execute(
                    select(DataSourceTombstone.slug).where(DataSourceTombstone.slug.in_(list(seed_by_slug.keys())))
                )
            )
            .scalars()
            .all()
        )
    except (ProgrammingError, OperationalError):
        tombstoned_slugs = set()

    existing_rows = (
        (
            await session.execute(
                select(DataSource).where(DataSource.is_system == True)  # noqa: E712
            )
        )
        .scalars()
        .all()
    )
    existing = {str(row.slug or "").strip().lower(): row for row in existing_rows if str(row.slug or "").strip()}

    inserted = 0
    updated = 0
    deleted = 0

    for row in existing_rows:
        slug = str(row.slug or "").strip().lower()
        if not slug or slug in seed_by_slug:
            continue
        if _is_seed_source_code(str(row.source_code or "")) or _is_legacy_system_source_row(row):
            await session.delete(row)
            deleted += 1

    for slug, row in seed_by_slug.items():
        if slug in tombstoned_slugs:
            continue

        current = existing.get(slug)
        if current is None:
            session.add(DataSource(**row))
            inserted += 1
            continue

        if not bool(current.is_system):
            continue

        current.source_key = row["source_key"]
        current.source_kind = row["source_kind"]
        current.name = row["name"]
        current.description = row["description"]
        current.enabled = bool(row["enabled"])
        current.config = dict(row["config"] or {})
        current.config_schema = dict(row["config_schema"] or {})
        current.sort_order = int(row["sort_order"])
        current.updated_at = row["updated_at"]

        if (
            not str(current.source_code or "").strip()
            or _is_seed_source_code(str(current.source_code or ""))
        ):
            if str(current.source_code or "") != row["source_code"]:
                current.source_code = row["source_code"]
                current.version = int(current.version or 1) + 1
                current.status = "unloaded"
                current.error_message = None

        updated += 1

    if inserted == 0 and updated == 0 and deleted == 0:
        return 0

    await session.commit()
    return inserted + updated + deleted


async def ensure_all_data_sources_seeded(session: AsyncSession) -> dict:
    seeded = await ensure_system_data_sources_seeded(session)
    return {"seeded": seeded}


def list_system_data_source_slugs() -> list[str]:
    return sorted({seed.slug for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS})


def list_prebuilt_data_source_presets() -> list[dict]:
    presets: list[dict] = []
    for seed in sorted(BASE_SYSTEM_DATA_SOURCE_SEEDS, key=lambda item: item.sort_order):
        presets.append(
            {
                "id": seed.slug,
                "slug_prefix": seed.slug,
                "name": seed.name,
                "description": seed.description,
                "source_key": seed.source_key,
                "source_kind": seed.source_kind,
                "source_code": seed.source_code,
                "config": dict(seed.config or {}),
                "config_schema": dict(seed.config_schema or {}),
                "is_system_seed": True,
            }
        )
    return presets
