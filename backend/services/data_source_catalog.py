from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import uuid

from sqlalchemy import select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AppSettings, DataSource, DataSourceTombstone
from services.news.rss_config import (
    normalize_custom_rss_feeds,
    normalize_gov_rss_feeds,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_SEED_MARKER = "# System data source seed"


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


def _slugify(value: str, prefix: str = "source") -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower())
    slug = slug.strip("_")
    if not slug:
        slug = prefix
    if len(slug) > 46:
        slug = slug[:46].rstrip("_")
    return slug


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


LEGACY_RSS_BRIDGE_SOURCE_CODE = '''# System data source seed
from services.data_source_sdk import BaseDataSource


class LegacyRssBridgeSource(BaseDataSource):
    name = "Legacy RSS Source Bridge"
    description = "Exports records for one legacy RSS feed from the shared feed cache"

    default_config = {
        "feed_name": "",
        "agency": "",
        "feed_source": "custom_rss",
        "max_age_hours": 168,
        "limit": 500,
    }

    async def fetch_async(self):
        from services.news.feed_service import news_feed_service

        feed_name = str(self.config.get("feed_name") or "").strip().lower()
        agency = str(self.config.get("agency") or "").strip().lower()
        feed_source = str(self.config.get("feed_source") or "").strip().lower()
        max_age_hours = int(self.config.get("max_age_hours") or 168)
        limit = int(self.config.get("limit") or 500)

        rows = []
        for article in news_feed_service.get_articles(max_age_hours=max_age_hours):
            article_source = str(getattr(article, "source", "") or "").strip().lower()
            article_feed_source = str(getattr(article, "feed_source", "") or "").strip().lower()
            if feed_source and article_feed_source != feed_source:
                continue
            if feed_name and article_source != feed_name:
                continue
            if agency and agency not in article_source:
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
                        "feed_source": article_feed_source,
                        "published": published.isoformat() if published else None,
                        "fetched_at": fetched_at.isoformat() if fetched_at else None,
                    },
                    "tags": ["news", article_feed_source or "rss"],
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
        slug="world_intelligence_signals",
        source_key="world_intelligence",
        source_kind="bridge",
        name="World Intelligence Signals",
        description="Unified export of world intelligence signals",
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
        slug="news_feed_all",
        source_key="news",
        source_kind="bridge",
        name="News Feed (All Sources)",
        description="Unified export of all cached news feed records",
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
]


_WORLD_SOURCE_ROWS = [
    {
        "slug": "world_acled",
        "name": "World Source: ACLED",
        "description": "Conflict and instability source bridge",
        "source_names": ["acled"],
        "signal_types": ["conflict", "instability", "tension"],
        "enabled_key": "enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_gdelt_tensions",
        "name": "World Source: GDELT Tensions",
        "description": "Diplomatic tension signal source bridge",
        "source_names": ["gdelt_tensions"],
        "signal_types": ["tension"],
        "enabled_key": "enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_military",
        "name": "World Source: Military",
        "description": "Military activity signal source bridge",
        "source_names": ["military"],
        "signal_types": ["military"],
        "enabled_key": "military_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_infrastructure",
        "name": "World Source: Infrastructure",
        "description": "Infrastructure disruption signal bridge",
        "source_names": ["infrastructure"],
        "signal_types": ["infrastructure"],
        "enabled_key": "enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_gdelt_news",
        "name": "World Source: GDELT News",
        "description": "World-news signal source bridge",
        "source_names": ["gdelt_news"],
        "signal_types": ["news"],
        "enabled_key": "gdelt_news_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_usgs",
        "name": "World Source: USGS",
        "description": "Earthquake and seismic event bridge",
        "source_names": ["usgs"],
        "signal_types": ["earthquake"],
        "enabled_key": "usgs_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_chokepoints",
        "name": "World Source: Chokepoints",
        "description": "Chokepoint reference and disruptions bridge",
        "source_names": ["chokepoints"],
        "signal_types": ["infrastructure", "convergence"],
        "enabled_key": "chokepoints_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_country_reference",
        "name": "World Source: Country Reference",
        "description": "Country reference sync source metadata",
        "source_names": ["country_reference"],
        "signal_types": [],
        "enabled_key": "country_reference_sync_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_ucdp_conflicts",
        "name": "World Source: UCDP Conflicts",
        "description": "UCDP conflict list source metadata",
        "source_names": ["ucdp_conflicts"],
        "signal_types": ["conflict", "instability"],
        "enabled_key": "ucdp_sync_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_mid_reference",
        "name": "World Source: MID Reference",
        "description": "Maritime identity reference source metadata",
        "source_names": ["mid_reference"],
        "signal_types": [],
        "enabled_key": "mid_sync_enabled",
        "default_enabled": True,
    },
    {
        "slug": "world_trade_dependencies",
        "name": "World Source: Trade Dependencies",
        "description": "Trade dependency overlay source metadata",
        "source_names": ["trade_dependencies"],
        "signal_types": ["convergence", "instability"],
        "enabled_key": "trade_dependency_sync_enabled",
        "default_enabled": True,
    },
]


def _is_seed_source_code(source_code: str) -> bool:
    return _SEED_MARKER in str(source_code or "")


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


def _extract_world_config(row: AppSettings | None) -> dict:
    if row is None:
        return {}
    raw = getattr(row, "world_intel_settings_json", None)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _build_dynamic_news_rows(settings_row: AppSettings | None, ts: datetime) -> list[dict]:
    out: list[dict] = []
    raw_custom = getattr(settings_row, "news_rss_feeds_json", None) if settings_row is not None else None
    custom_feeds = normalize_custom_rss_feeds(raw_custom or [])

    raw_gov = getattr(settings_row, "news_gov_rss_feeds_json", None) if settings_row is not None else None
    gov_feeds = normalize_gov_rss_feeds(raw_gov or [])

    for index, feed in enumerate(custom_feeds):
        feed_id = _slugify(str(feed.get("id") or "custom"), prefix="custom")
        name = str(feed.get("name") or feed.get("url") or "Custom RSS").strip()
        slug = f"news_custom_{feed_id}"
        out.append(
            {
                "id": uuid.uuid4().hex,
                "slug": slug,
                "source_key": "news",
                "source_kind": "rss",
                "name": f"Custom RSS: {name}",
                "description": "Legacy custom RSS source imported from AppSettings",
                "source_code": LEGACY_RSS_BRIDGE_SOURCE_CODE,
                "class_name": None,
                "is_system": True,
                "enabled": bool(feed.get("enabled", True)),
                "status": "unloaded",
                "error_message": None,
                "config": {
                    "feed_id": str(feed.get("id") or "").strip(),
                    "feed_name": name,
                    "url": str(feed.get("url") or "").strip(),
                    "category": str(feed.get("category") or "").strip().lower(),
                    "feed_source": "custom_rss",
                    "max_age_hours": 168,
                    "limit": 500,
                },
                "config_schema": {
                    "param_fields": [
                        {"key": "max_age_hours", "label": "Max Age (hours)", "type": "integer", "min": 1, "max": 336},
                        {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                    ]
                },
                "version": 1,
                "sort_order": 200 + index,
                "created_at": ts,
                "updated_at": ts,
            }
        )

    for index, feed in enumerate(gov_feeds):
        feed_id = _slugify(str(feed.get("id") or "gov"), prefix="gov")
        name = str(feed.get("name") or feed.get("url") or "Government RSS").strip()
        slug = f"news_gov_{feed_id}"
        out.append(
            {
                "id": uuid.uuid4().hex,
                "slug": slug,
                "source_key": "news",
                "source_kind": "rss",
                "name": f"Gov RSS: {name}",
                "description": "Legacy government RSS source imported from AppSettings",
                "source_code": LEGACY_RSS_BRIDGE_SOURCE_CODE,
                "class_name": None,
                "is_system": True,
                "enabled": bool(feed.get("enabled", True)),
                "status": "unloaded",
                "error_message": None,
                "config": {
                    "feed_id": str(feed.get("id") or "").strip(),
                    "agency": str(feed.get("agency") or "government").strip().lower(),
                    "feed_name": name,
                    "url": str(feed.get("url") or "").strip(),
                    "priority": str(feed.get("priority") or "medium").strip().lower(),
                    "country_iso3": str(feed.get("country_iso3") or "USA").strip().upper(),
                    "feed_source": "rss",
                    "max_age_hours": 168,
                    "limit": 500,
                },
                "config_schema": {
                    "param_fields": [
                        {"key": "max_age_hours", "label": "Max Age (hours)", "type": "integer", "min": 1, "max": 336},
                        {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                    ]
                },
                "version": 1,
                "sort_order": 280 + index,
                "created_at": ts,
                "updated_at": ts,
            }
        )

    return out


def _build_dynamic_world_rows(settings_row: AppSettings | None, ts: datetime) -> list[dict]:
    world_config = _extract_world_config(settings_row)
    out: list[dict] = []
    for index, source_def in enumerate(_WORLD_SOURCE_ROWS):
        enabled_key = str(source_def["enabled_key"])
        default_enabled = bool(source_def["default_enabled"])
        raw_enabled = world_config.get(enabled_key)
        enabled = default_enabled if raw_enabled is None else bool(raw_enabled)

        out.append(
            {
                "id": uuid.uuid4().hex,
                "slug": str(source_def["slug"]),
                "source_key": "world_intelligence",
                "source_kind": "bridge",
                "name": str(source_def["name"]),
                "description": str(source_def["description"]),
                "source_code": WORLD_SIGNALS_BRIDGE_SOURCE_CODE,
                "class_name": None,
                "is_system": True,
                "enabled": enabled,
                "status": "unloaded",
                "error_message": None,
                "config": {
                    "hours": 168,
                    "limit": 1500,
                    "min_severity": 0.0,
                    "source_names": list(source_def.get("source_names") or []),
                    "signal_types": list(source_def.get("signal_types") or []),
                },
                "config_schema": {
                    "param_fields": [
                        {"key": "hours", "label": "Lookback (hours)", "type": "integer", "min": 1, "max": 720},
                        {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
                        {"key": "min_severity", "label": "Min Severity", "type": "number", "min": 0, "max": 1},
                    ]
                },
                "version": 1,
                "sort_order": 320 + index,
                "created_at": ts,
                "updated_at": ts,
            }
        )

    return out


def build_system_data_source_rows(settings_row: AppSettings | None, *, now: datetime | None = None) -> list[dict]:
    ts = _now_or_default(now)
    rows: list[dict] = []

    for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS:
        rows.append(_build_row_from_seed(seed, ts))

    rows.extend(_build_dynamic_news_rows(settings_row, ts))
    rows.extend(_build_dynamic_world_rows(settings_row, ts))
    return rows


async def ensure_system_data_sources_seeded(session: AsyncSession) -> int:
    settings_row = (
        (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
    )

    rows = build_system_data_source_rows(settings_row)
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

    existing = {
        row.slug: row
        for row in (
            (
                await session.execute(select(DataSource).where(DataSource.slug.in_(list(seed_by_slug.keys()))))
            )
            .scalars()
            .all()
        )
    }

    inserted = 0
    updated = 0

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

        if not str(current.source_code or "").strip() or _is_seed_source_code(str(current.source_code or "")):
            if str(current.source_code or "") != row["source_code"]:
                current.source_code = row["source_code"]
                current.version = int(current.version or 1) + 1
                current.status = "unloaded"
                current.error_message = None

        updated += 1

    if inserted == 0 and updated == 0:
        return 0

    await session.commit()
    return inserted + updated


async def ensure_all_data_sources_seeded(session: AsyncSession) -> dict:
    seeded = await ensure_system_data_sources_seeded(session)
    return {"seeded": seeded}


def list_system_data_source_slugs() -> list[str]:
    static = [seed.slug for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS]
    dynamic_world = [str(row["slug"]) for row in _WORLD_SOURCE_ROWS]
    return sorted(set(static + dynamic_world))
