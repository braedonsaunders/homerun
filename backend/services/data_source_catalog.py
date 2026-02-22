from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from urllib.parse import quote
import uuid

from sqlalchemy import select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import DataSource, DataSourceTombstone
from utils.logger import get_logger

logger = get_logger(__name__)

_SEED_MARKER = "# System data source seed"
_LEGACY_SYSTEM_SOURCE_KEYS = {"news"}
_LEGACY_SYSTEM_SOURCE_KINDS = {"bridge", "gdelt"}

_GOOGLE_NEWS_TOPICS: tuple[str, ...] = (
    "breaking news",
    "politics",
    "business",
    "markets",
    "economy",
    "inflation",
    "federal reserve",
    "supreme court",
    "technology",
    "artificial intelligence",
    "science",
    "sports",
    "world",
    "ukraine",
    "china us relations",
    "middle east",
    "energy prices",
    "cryptocurrency",
    "bitcoin etf",
)

_GDELT_QUERIES: tuple[str, ...] = (
    "prediction market",
    "polymarket",
    "kalshi",
    "election odds",
    "federal reserve policy",
    "inflation report",
    "jobs report",
    "supreme court decision",
    "ceasefire agreement",
    "military escalation",
    "trade sanctions",
    "cryptocurrency regulation",
    "bitcoin etf",
)

_DEFAULT_RETENTION_BY_KIND: dict[str, dict[str, int]] = {
    "python": {"max_records": 25_000, "max_age_days": 180},
    "rss": {"max_records": 7_500, "max_age_days": 30},
    "rest_api": {"max_records": 15_000, "max_age_days": 45},
}

_EVENTS_RETENTION_POLICY: dict[str, int] = {"max_records": 50_000, "max_age_days": 365}
_EVENTS_ALL_RETENTION_POLICY: dict[str, int] = {"max_records": 100_000, "max_age_days": 365}
_STORIES_RSS_RETENTION_POLICY: dict[str, int] = {"max_records": 7_500, "max_age_days": 30}
_STORIES_REST_RETENTION_POLICY: dict[str, int] = {"max_records": 15_000, "max_age_days": 45}
_FALLBACK_RETENTION_POLICY: dict[str, int] = {"max_records": 10_000, "max_age_days": 90}


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
    retention: dict[str, int] = field(default_factory=dict)
    enabled: bool = True


def _now_or_default(now: datetime | None) -> datetime:
    return now or datetime.utcnow()


def _slug_fragment(value: str, *, max_len: int = 20) -> str:
    out = []
    prev_us = False
    for ch in str(value or "").strip().lower():
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            out.append(ch)
            prev_us = False
            continue
        if not prev_us:
            out.append("_")
            prev_us = True
    token = "".join(out).strip("_")
    if not token:
        return "source"
    if token[0].isdigit():
        token = f"s_{token}"
    return token[:max_len].rstrip("_")


def _slug_with_hash(prefix: str, value: str) -> str:
    token = _slug_fragment(value, max_len=18)
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:6]
    slug = f"{prefix}_{token}_{digest}" if token else f"{prefix}_{digest}"
    if len(slug) <= 50:
        return slug
    trimmed = slug[:50].rstrip("_")
    if trimmed and trimmed[-1].isalnum():
        return trimmed
    return f"{trimmed[:-1]}x"


def _google_rss_url(topic: str) -> str:
    encoded = quote(topic)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def _gdelt_doc_url(query: str, *, max_records: int = 30, timespan: str = "6h") -> str:
    encoded = quote(query)
    return (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={encoded}&mode=artlist&maxrecords={int(max_records)}"
        f"&format=json&sort=datedesc&timespan={timespan}"
    )


def _rss_config_schema() -> dict:
    return {
        "param_fields": [
            {"key": "url", "label": "Feed URL", "type": "string"},
            {"key": "source_name", "label": "Source Name", "type": "string"},
            {"key": "category_filter", "label": "Category Filter", "type": "string"},
            {"key": "category_override", "label": "Category Override", "type": "string"},
            {"key": "feed_source", "label": "Feed Source", "type": "string"},
            {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 500},
        ]
    }


def _rest_api_config_schema() -> dict:
    return {
        "param_fields": [
            {"key": "url", "label": "API URL", "type": "string"},
            {"key": "json_path", "label": "JSONPath", "type": "string"},
            {"key": "headers", "label": "Headers", "type": "json"},
            {"key": "source_name", "label": "Source Name", "type": "string"},
            {"key": "category_override", "label": "Category Override", "type": "string"},
            {"key": "feed_source", "label": "Feed Source", "type": "string"},
            {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
        ]
    }


def _source_class_name_from_slug(slug: str, *, suffix: str = "Source") -> str:
    token = "".join(part.capitalize() for part in str(slug or "").strip().split("_") if part)
    token = "".join(ch for ch in token if ch.isalnum())
    if not token:
        token = "Custom"
    if token[0].isdigit():
        token = f"S{token}"
    return f"{token}{suffix}"


def default_data_source_retention_policy(
    *,
    slug: str | None = None,
    source_key: str | None = None,
    source_kind: str | None = None,
) -> dict[str, int]:
    normalized_slug = str(slug or "").strip().lower()
    normalized_source_key = str(source_key or "").strip().lower()
    normalized_source_kind = str(source_kind or "").strip().lower()

    if normalized_slug == "events_all":
        return dict(_EVENTS_ALL_RETENTION_POLICY)
    if normalized_source_key == "events" or normalized_slug.startswith("events_"):
        return dict(_EVENTS_RETENTION_POLICY)
    if normalized_source_key == "stories":
        if normalized_source_kind == "rss" or normalized_slug.startswith("stories_google_"):
            return dict(_STORIES_RSS_RETENTION_POLICY)
        return dict(_STORIES_REST_RETENTION_POLICY)

    by_kind = _DEFAULT_RETENTION_BY_KIND.get(normalized_source_kind)
    if by_kind is not None:
        return dict(by_kind)
    return dict(_FALLBACK_RETENTION_POLICY)


def _stories_rss_default_config(
    *,
    url: str,
    source_name: str,
    category: str,
    feed_source: str,
) -> dict:
    return {
        "url": str(url or "").strip(),
        "source_name": str(source_name or "").strip(),
        "category_filter": "",
        "category_override": str(category or "").strip().lower(),
        "feed_source": str(feed_source or "rss").strip().lower() or "rss",
        "limit": 40,
    }


def _stories_rest_api_default_config(
    *,
    url: str,
    source_name: str,
    category: str,
    feed_source: str,
) -> dict:
    return {
        "url": str(url or "").strip(),
        "json_path": "$.articles[*]",
        "headers": {},
        "source_name": str(source_name or "").strip(),
        "category_override": str(category or "").strip().lower(),
        "feed_source": str(feed_source or "rest_api").strip().lower() or "rest_api",
        "limit": 30,
    }


def _stories_rss_source_code(
    *,
    class_name: str,
    source_name: str,
    source_description: str,
    default_config: dict,
    include_seed_marker: bool,
) -> str:
    imports = [
        _SEED_MARKER if include_seed_marker else None,
        "from __future__ import annotations",
        "",
        "import hashlib",
        "from services.data_source_sdk import BaseDataSource",
    ]
    imports = [line for line in imports if line is not None]

    class_lines = [
        "",
        "",
        f"class {class_name}(BaseDataSource):",
        f'    name = "{_escape_py_string(source_name)}"',
        f'    description = "{_escape_py_string(source_description)}"',
        "",
        f"    default_config = {repr(default_config)}",
        "",
        "    async def fetch_async(self):",
        "        url = str(self.config.get('url') or '').strip()",
        "        if not url:",
        "            return []",
        "        limit = self._as_int(self.config.get('limit'), self.default_config.get('limit', 40), 1, 500)",
        "        source_name = str(self.config.get('source_name') or '').strip() or str(self.name or '').strip() or 'rss'",
        "        category_filter = str(self.config.get('category_filter') or '').strip().lower() or None",
        "        category_override = str(self.config.get('category_override') or '').strip().lower() or None",
        "        feed_source = str(self.config.get('feed_source') or 'rss').strip().lower() or 'rss'",
        "",
        "        feed_title, entries = await self._fetch_rss(url, timeout=30.0)",
        "        records = []",
        "        for entry in list(entries or [])[:limit]:",
        "            guid = str(entry.get('id') or entry.get('guid') or entry.get('link') or '').strip()",
        "            if not guid:",
        "                continue",
        "",
        "            tags = []",
        "            for tag in list(entry.get('tags') or []):",
        "                if isinstance(tag, dict):",
        "                    term = str(tag.get('term') or '').strip()",
        "                else:",
        "                    term = str(getattr(tag, 'term', '') or '').strip()",
        "                if term:",
        "                    tags.append(term)",
        "",
        "            if category_filter and not any(category_filter in str(tag or '').lower() for tag in tags):",
        "                continue",
        "",
        "            published = entry.get('published_parsed') or entry.get('updated_parsed')",
        "            observed = self._parse_datetime(published)",
        "            observed_at = observed.isoformat() if observed is not None else None",
        "",
        "            summary = str(entry.get('summary') or entry.get('description') or '').strip()",
        "            link = str(entry.get('link') or '').strip()",
        "            category = category_override or (str(tags[0]).strip().lower() if tags else None)",
        "            records.append(",
        "                {",
        "                    'external_id': hashlib.sha256(guid.encode('utf-8')).hexdigest()[:32],",
        "                    'title': str(entry.get('title') or '').strip() or None,",
        "                    'summary': summary[:2000] if summary else None,",
        "                    'url': link or None,",
        "                    'source': source_name or feed_title or url,",
        "                    'category': category,",
        "                    'observed_at': observed_at,",
        "                    'tags': tags,",
        "                    'feed_source': feed_source,",
        "                }",
        "            )",
        "",
        "        return records",
        "",
    ]

    return "\n".join([*imports, *class_lines])


def _stories_rest_api_source_code(
    *,
    class_name: str,
    source_name: str,
    source_description: str,
    default_config: dict,
    include_seed_marker: bool,
) -> str:
    imports = [
        _SEED_MARKER if include_seed_marker else None,
        "from __future__ import annotations",
        "",
        "import hashlib",
        "",
        "from services.data_source_sdk import BaseDataSource",
    ]
    imports = [line for line in imports if line is not None]

    class_lines = [
        "",
        "",
        f"class {class_name}(BaseDataSource):",
        f'    name = "{_escape_py_string(source_name)}"',
        f'    description = "{_escape_py_string(source_description)}"',
        "",
        f"    default_config = {repr(default_config)}",
        "",
        "    async def fetch_async(self):",
        "        url = str(self.config.get('url') or '').strip()",
        "        if not url:",
        "            return []",
        "",
        "        headers = dict(self.config.get('headers') or {})",
        "        json_path = str(self.config.get('json_path') or '$[*]').strip()",
        "        limit = self._as_int(self.config.get('limit'), self.default_config.get('limit', 200), 1, 5000)",
        "        source_name = str(self.config.get('source_name') or '').strip() or None",
        "        category_override = str(self.config.get('category_override') or '').strip().lower() or None",
        "        feed_source = str(self.config.get('feed_source') or 'rest_api').strip().lower() or 'rest_api'",
        "",
        "        payload = await self._http_get_json(url, headers=headers, timeout=30.0, default={})",
        "",
        "        raw_rows = self._extract_json_path(payload, json_path)",
        "        records = []",
        "        for item in raw_rows[:limit]:",
        "            if not isinstance(item, dict):",
        "                continue",
        "",
        "            external_id = str(item.get('id') or item.get('external_id') or item.get('guid') or '').strip()",
        "            if not external_id:",
        "                external_id = hashlib.sha256(str(item).encode('utf-8')).hexdigest()[:32]",
        "",
        "            summary = str(",
        "                item.get('summary')",
        "                or item.get('description')",
        "                or item.get('snippet')",
        "                or item.get('excerpt')",
        "                or item.get('content')",
        "                or item.get('body')",
        "                or ''",
        "            ).strip()",
        "            tags = item.get('tags') if isinstance(item.get('tags'), list) else []",
        "            source = source_name or str(item.get('source') or item.get('domain') or item.get('provider') or '').strip() or None",
        "            category = category_override or str(item.get('category') or item.get('type') or '').strip().lower() or None",
        "",
        "            record = {",
        "                'external_id': external_id,",
        "                'title': str(item.get('title') or item.get('name') or '').strip() or None,",
        "                'summary': summary[:2000] if summary else None,",
        "                'url': str(item.get('url') or item.get('link') or '').strip() or None,",
        "                'source': source,",
        "                'category': category,",
        "                'observed_at': item.get('observed_at') or item.get('created_at') or item.get('published_at') or item.get('seendate') or item.get('date') or None,",
        "                'tags': tags,",
        "                'feed_source': feed_source,",
        "            }",
        "",
        "            for key, value in item.items():",
        "                if key in {'id', 'external_id', 'title', 'name', 'summary', 'description', 'snippet', 'excerpt', 'content', 'body', 'url', 'link', 'source', 'domain', 'provider', 'category', 'type', 'observed_at', 'created_at', 'published_at', 'seendate', 'date', 'tags'}:",
        "                    continue",
        "                record[key] = value",
        "",
        "            records.append(record)",
        "",
        "        return records",
        "",
    ]

    return "\n".join([*imports, *class_lines])


def build_data_source_source_code(
    *,
    source_kind: str,
    slug: str,
    name: str,
    description: str,
    config: dict | None = None,
    include_seed_marker: bool = False,
) -> str:
    normalized_kind = str(source_kind or "").strip().lower()
    normalized_slug = str(slug or "").strip().lower()
    source_name = str(name or "").strip() or normalized_slug.replace("_", " ").title()
    source_description = str(description or "").strip()

    if normalized_kind == "rss":
        default_config = _stories_rss_default_config(
            url=str((config or {}).get("url") or "").strip(),
            source_name=str((config or {}).get("source_name") or source_name).strip(),
            category=str((config or {}).get("category_override") or "").strip().lower(),
            feed_source=str((config or {}).get("feed_source") or "rss").strip().lower(),
        )
        if config:
            default_config.update(dict(config))
        class_name = _source_class_name_from_slug(normalized_slug, suffix="RssSource")
        return _stories_rss_source_code(
            class_name=class_name,
            source_name=source_name,
            source_description=source_description,
            default_config=default_config,
            include_seed_marker=include_seed_marker,
        )

    if normalized_kind == "rest_api":
        default_config = _stories_rest_api_default_config(
            url=str((config or {}).get("url") or "").strip(),
            source_name=str((config or {}).get("source_name") or source_name).strip(),
            category=str((config or {}).get("category_override") or "").strip().lower(),
            feed_source=str((config or {}).get("feed_source") or "rest_api").strip().lower(),
        )
        if config:
            default_config.update(dict(config))
        class_name = _source_class_name_from_slug(normalized_slug, suffix="RestApiSource")
        return _stories_rest_api_source_code(
            class_name=class_name,
            source_name=source_name,
            source_description=source_description,
            default_config=default_config,
            include_seed_marker=include_seed_marker,
        )

    raise ValueError(f"Unsupported source_kind for source code generation: {normalized_kind or 'unknown'}")


def _make_story_rss_seed(
    *,
    slug: str,
    name: str,
    description: str,
    url: str,
    category: str,
    feed_source: str,
    sort_order: int,
) -> SystemDataSourceSeed:
    config = _stories_rss_default_config(
        url=url,
        source_name=name.replace("Stories: ", "").strip() or name,
        category=category,
        feed_source=feed_source,
    )
    source_code = build_data_source_source_code(
        source_kind="rss",
        slug=slug,
        name=name,
        description=description,
        config=config,
        include_seed_marker=True,
    )
    return SystemDataSourceSeed(
        slug=slug,
        source_key="stories",
        source_kind="rss",
        name=name,
        description=description,
        source_code=source_code,
        sort_order=sort_order,
        config=config,
        config_schema=_rss_config_schema(),
        retention=default_data_source_retention_policy(
            slug=slug,
            source_key="stories",
            source_kind="rss",
        ),
        enabled=True,
    )


def _make_story_rest_seed(
    *,
    slug: str,
    name: str,
    description: str,
    url: str,
    category: str,
    feed_source: str,
    sort_order: int,
) -> SystemDataSourceSeed:
    config = _stories_rest_api_default_config(
        url=url,
        source_name="GDELT",
        category=category,
        feed_source=feed_source,
    )
    source_code = build_data_source_source_code(
        source_kind="rest_api",
        slug=slug,
        name=name,
        description=description,
        config=config,
        include_seed_marker=True,
    )
    return SystemDataSourceSeed(
        slug=slug,
        source_key="stories",
        source_kind="rest_api",
        name=name,
        description=description,
        source_code=source_code,
        sort_order=sort_order,
        config=config,
        config_schema=_rest_api_config_schema(),
        retention=default_data_source_retention_policy(
            slug=slug,
            source_key="stories",
            source_kind="rest_api",
        ),
        enabled=True,
    )


_EVENT_SOURCE_DEFS: tuple[dict, ...] = (
    {
        "slug": "events_acled",
        "name": "Events: ACLED",
        "description": "Conflict and instability events from ACLED",
        "source_names": ["acled"],
        "signal_types": ["conflict", "instability", "tension"],
        "map_color": "#f87171",
    },
    {
        "slug": "events_gdelt_tensions",
        "name": "Events: GDELT Tensions",
        "description": "Diplomatic tension events from GDELT",
        "source_names": ["gdelt_tensions"],
        "signal_types": ["tension"],
        "map_color": "#fb923c",
    },
    {
        "slug": "events_military",
        "name": "Events: Military",
        "description": "Military aircraft tracking via OpenSky Network ADS-B data over geopolitical hotspots",
        "source_names": ["military"],
        "signal_types": ["military"],
        "map_color": "#60a5fa",
    },
    {
        "slug": "events_infrastructure",
        "name": "Events: Infrastructure",
        "description": "Infrastructure disruption news from GDELT article search",
        "source_names": ["infrastructure"],
        "signal_types": ["infrastructure"],
        "map_color": "#34d399",
    },
    {
        "slug": "events_gdelt_news",
        "name": "Events: GDELT News",
        "description": "News-derived world events from GDELT",
        "source_names": ["gdelt_news"],
        "signal_types": ["news"],
        "map_color": "#a78bfa",
    },
    {
        "slug": "events_usgs",
        "name": "Events: USGS",
        "description": "Earthquake events from USGS",
        "source_names": ["usgs"],
        "signal_types": ["earthquake"],
        "map_color": "#f59e0b",
        "map_layer_label": "Earthquakes",
        "map_layer_short": "EQ",
        "map_radius_min": 6,
        "map_radius_max": 22,
        "map_opacity": 0.88,
        "map_blur": 0.15,
        "map_stroke_width": 1.5,
        "map_glow": False,
        "map_dedicated_toggle": True,
    },
    {
        "slug": "events_ucdp_conflicts",
        "name": "Events: UCDP Conflicts",
        "description": "Armed conflict events from Uppsala Conflict Data Program (free, no key required)",
        "source_names": ["ucdp"],
        "signal_types": ["conflict"],
        "hours": 8760,
        "map_color": "#f87171",
    },
    {
        "slug": "events_trade_dependencies",
        "name": "Events: Trade Dependencies",
        "description": "Trade-to-GDP ratios by country from World Bank (free, no key required)",
        "source_names": ["world_bank"],
        "signal_types": ["trade"],
        "hours": 43800,
        "map_color": "#facc15",
    },
    {
        "slug": "events_chokepoint_reference",
        "name": "Events: Chokepoint Reference",
        "description": "Major global shipping chokepoints — static reference data",
        "source_names": ["chokepoints"],
        "signal_types": ["infrastructure"],
        "map_color": "#34d399",
    },
    {
        "slug": "events_country_instability",
        "name": "Events: Country Instability Index",
        "description": "Derived instability scores aggregated from conflict, tension, and military signals",
        "source_names": ["instability_index"],
        "signal_types": ["instability"],
        "map_color": "#facc15",
    },
    {
        "slug": "events_airplanes_live",
        "name": "Events: Airplanes.live Military",
        "description": "Real-time military aircraft positions from Airplanes.live ADS-B aggregator (free, no auth)",
        "source_names": ["airplanes_live"],
        "signal_types": ["military"],
        "map_color": "#60a5fa",
    },
    {
        "slug": "events_ais_ships",
        "name": "Events: AIS Ship Tracking",
        "description": "Vessel positions from Finnish Transport Agency Digitraffic AIS feed (free, no auth)",
        "source_names": ["ais_ships"],
        "signal_types": ["shipping"],
        "map_color": "#38bdf8",
    },
    {
        "slug": "events_nasa_firms",
        "name": "Events: NASA FIRMS Wildfires",
        "description": "Global active fire detections from NASA VIIRS satellite (free, no auth, updated every few hours)",
        "source_names": ["nasa_firms"],
        "signal_types": ["fire"],
        "map_color": "#ef4444",
        "map_layer_label": "Wildfires",
        "map_layer_short": "WF",
        "map_radius_min": 4,
        "map_radius_max": 16,
        "map_opacity": 0.92,
        "map_blur": 0.1,
        "map_stroke_width": 1.2,
        "map_glow": False,
        "map_dedicated_toggle": True,
    },
)


def _escape_py_string(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _events_source_id_from_slug(source_slug: str) -> str:
    normalized_slug = str(source_slug or "").strip().lower()
    if normalized_slug.startswith("events_"):
        suffix = normalized_slug[len("events_") :].strip()
        return suffix or "unknown"
    return "unknown"


def _events_source_class_name(source_slug: str) -> str:
    return _source_class_name_from_slug(source_slug, suffix="Source")


def _events_default_config(
    *,
    hours: int,
    limit: int,
    min_severity: float,
    source_names: list[str],
    signal_types: list[str],
    min_magnitude: float,
    map_color: str = "",
    map_layer_label: str = "",
    map_layer_short: str = "",
    map_radius_min: float = 4,
    map_radius_max: float = 9,
    map_opacity: float = 0.92,
    map_blur: float = 0.1,
    map_stroke_width: float = 1.0,
    map_glow: bool = True,
    map_dedicated_toggle: bool = False,
) -> dict:
    out: dict = {
        "hours": int(hours),
        "limit": int(limit),
        "min_severity": float(min_severity),
        "source_names": list(source_names or []),
        "signal_types": list(signal_types or []),
        "min_magnitude": float(min_magnitude),
    }
    if map_color:
        out["map_color"] = str(map_color)
    if map_layer_label:
        out["map_layer_label"] = str(map_layer_label)
    if map_layer_short:
        out["map_layer_short"] = str(map_layer_short)
    out["map_radius_min"] = float(map_radius_min)
    out["map_radius_max"] = float(map_radius_max)
    out["map_opacity"] = float(map_opacity)
    out["map_blur"] = float(map_blur)
    out["map_stroke_width"] = float(map_stroke_width)
    out["map_glow"] = bool(map_glow)
    out["map_dedicated_toggle"] = bool(map_dedicated_toggle)
    return out


def _events_wrapper_source_code(
    *,
    source_id: str,
    source_slug: str,
    class_name: str,
    source_name: str,
    source_description: str,
    default_config: dict,
) -> str:
    normalized_source_id = str(source_id or "unknown").strip().lower() or "unknown"
    imports = [
        _SEED_MARKER,
        "from __future__ import annotations",
        "",
        "import hashlib",
        "from datetime import datetime, timedelta, timezone",
        "",
        "import httpx",
        "from services.data_source_sdk import BaseDataSource",
    ]

    class_lines = [
        "",
        "",
        f"class {class_name}(BaseDataSource):",
        f'    name = "{_escape_py_string(source_name)}"',
        f'    description = "{_escape_py_string(source_description)}"',
        f'    slug = "{_escape_py_string(source_slug)}"',
        "",
        f"    default_config = {repr(default_config)}",
        "",
    ]

    if normalized_source_id == "acled":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://api.acleddata.com/acled/read/').strip()",
                "        api_key = str(self.config.get('api_key') or '').strip()",
                "        email = str(self.config.get('email') or '').strip()",
                "        country = str(self.config.get('country') or '').strip()",
                "        if not api_key or not email:",
                "            if not bool(getattr(self, '_missing_credentials_warned', False)):",
                "                import logging as _log",
                "                _log.getLogger('data_source.acled').warning('ACLED source requires api_key and email in config — skipping fetch')",
                "                self._missing_credentials_warned = True",
                "            return []",
                "        if not endpoint:",
                "            return []",
                "        since = (datetime.now(timezone.utc) - timedelta(hours=self._hours())).date().isoformat()",
                "        params = {",
                "            'key': api_key,",
                "            'email': email,",
                "            'event_date': since,",
                "            'event_date_where': '>=',",
                "            'limit': str(max(self._limit() * 2, self._limit())),",
                "        }",
                "        if country:",
                "            params['country'] = country",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            try:",
                "                response = await client.get(endpoint, params=params)",
                "            except Exception:",
                "                return []",
                "            if response.status_code == 429:",
                "                return []",
                "            try:",
                "                response.raise_for_status()",
                "                payload = response.json()",
                "            except Exception:",
                "                return []",
                "        events = payload.get('data') if isinstance(payload, dict) else []",
                "        if not isinstance(events, list):",
                "            events = []",
                "        out = []",
                "        for event in list(events or [])[: max(self._limit() * 2, self._limit())]:",
                "            if not isinstance(event, dict):",
                "                continue",
                "            fatalities = self._as_float(event.get('fatalities'), 0.0)",
                "            event_type = str(event.get('event_type') or '').strip().lower()",
                "            severity = self._clamp(max(0.2, fatalities / 20.0), 0.0, 1.0)",
                "            if event_type in {'battles', 'violence against civilians'}:",
                "                severity = self._clamp(severity + 0.15, 0.0, 1.0)",
                "            source = 'acled'",
                "            category = 'conflict'",
                "            if not self._keep(source, category, severity):",
                "                continue",
                "            event_type_name = str(event.get('event_type') or '').strip()",
                "            country_name = str(event.get('country') or '').strip()",
                "            sub_event_type = str(event.get('sub_event_type') or '').strip()",
                "            notes = str(event.get('notes') or '').strip()",
                "            observed = self._parse_datetime(event.get('event_date'))",
                "            observed_at = observed.isoformat() if observed is not None else None",
                "            event_id = str(event.get('event_id_cnty') or event.get('event_id_no_cnty') or event.get('event_id') or '').strip()",
                "            out.append(",
                "                {",
                "                    'external_id': event_id or hashlib.sha256(f\"{event_type_name}:{country_name}:{observed_at}\".encode('utf-8')).hexdigest()[:24],",
                "                    'title': (f\"{event_type_name.title()} in {country_name}\".strip() or 'ACLED event'),",
                "                    'summary': (notes or sub_event_type)[:500],",
                "                    'category': category,",
                "                    'source': source,",
                "                    'observed_at': observed_at,",
                "                    'payload': {",
                "                        'severity': severity,",
                "                        'event_type': event.get('event_type'),",
                "                        'sub_event_type': event.get('sub_event_type'),",
                "                        'fatalities': fatalities,",
                "                        'actor1': event.get('actor1'),",
                "                        'actor2': event.get('actor2'),",
                "                        'notes': notes,",
                "                    },",
                "                    'latitude': event.get('latitude'),",
                "                    'longitude': event.get('longitude'),",
                "                    'country_iso3': event.get('iso3'),",
                "                    'tags': ['events', 'conflict', 'acled'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "gdelt_tensions":
        class_lines.extend(
            [
                "    _COUNTRY_NAMES = {",
                "        'US': 'United States', 'CN': 'China', 'RU': 'Russia', 'TW': 'Taiwan',",
                "        'IR': 'Iran', 'UA': 'Ukraine', 'KP': 'North Korea', 'KR': 'South Korea',",
                "        'IL': 'Israel', 'PS': 'Palestine', 'SY': 'Syria', 'IN': 'India',",
                "        'PK': 'Pakistan', 'JP': 'Japan', 'GB': 'United Kingdom', 'FR': 'France',",
                "        'DE': 'Germany', 'SA': 'Saudi Arabia', 'TR': 'Turkey', 'EU': 'Europe',",
                "    }",
                "",
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://api.gdeltproject.org/api/v2/doc/doc').strip()",
                "        pairs = self.config.get('pairs') or ['US-CN', 'US-RU', 'CN-TW', 'US-IR', 'RU-UA']",
                "        timespan_hours = self._as_int(self.config.get('timespan_hours'), max(1, min(168, self._hours())), 1, 168)",
                "        max_per_pair = self._as_int(self.config.get('maxrecords_per_pair'), 250, 10, 250)",
                "        if not isinstance(pairs, list):",
                "            pairs = []",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            for raw_pair in pairs:",
                "                pair_text = str(raw_pair or '').strip().upper().replace(':', '-')",
                "                if '-' not in pair_text:",
                "                    continue",
                "                country_a, country_b = [part.strip() for part in pair_text.split('-', 1)]",
                "                if not country_a or not country_b:",
                "                    continue",
                "                name_a = self._COUNTRY_NAMES.get(country_a, country_a)",
                "                name_b = self._COUNTRY_NAMES.get(country_b, country_b)",
                "                params = {",
                "                    'query': f'\"{name_a}\" \"{name_b}\" (tensions OR conflict OR sanctions OR military)',",
                "                    'mode': 'artlist',",
                "                    'format': 'json',",
                "                    'sort': 'datedesc',",
                "                    'timespan': f'{timespan_hours}h',",
                "                    'maxrecords': str(max_per_pair),",
                "                }",
                "                try:",
                "                    response = await client.get(endpoint, params=params)",
                "                except Exception:",
                "                    continue",
                "                if response.status_code == 429:",
                "                    continue",
                "                try:",
                "                    response.raise_for_status()",
                "                except Exception:",
                "                    continue",
                "                try:",
                "                    payload = response.json()",
                "                except Exception:",
                "                    continue",
                "                articles = payload.get('articles') if isinstance(payload, dict) else []",
                "                if not isinstance(articles, list):",
                "                    articles = []",
                "                tones = []",
                "                theme_counts = {}",
                "                for article in articles:",
                "                    if not isinstance(article, dict):",
                "                        continue",
                "                    tones.append(self._as_float(article.get('tone'), 0.0))",
                "                    for token in str(article.get('themes') or '').split(';'):",
                "                        item = str(token or '').strip()",
                "                        if not item:",
                "                            continue",
                "                        theme_counts[item] = int(theme_counts.get(item, 0)) + 1",
                "                event_count = len(articles)",
                "                avg_tone = (sum(tones) / len(tones)) if tones else 0.0",
                "                tone_component = self._clamp((-avg_tone + 10.0) / 20.0, 0.0, 1.0)",
                "                volume_component = self._clamp(event_count / 120.0, 0.0, 1.0)",
                "                severity = self._clamp((tone_component * 0.65) + (volume_component * 0.35), 0.0, 1.0)",
                "                source = 'gdelt_tensions'",
                "                category = 'tension'",
                "                if not self._keep(source, category, severity):",
                "                    continue",
                "                top_themes = sorted(theme_counts.items(), key=lambda item: item[1], reverse=True)[:5]",
                "                top_event_types = [name for name, _ in top_themes]",
                "                first_article = articles[0] if articles else {}",
                "                observed = self._parse_datetime(first_article.get('seendate') if isinstance(first_article, dict) else None)",
                "                observed_at = observed.isoformat() if observed is not None else None",
                "                score = round((severity * 100.0), 2)",
                "                trend = 'rising' if event_count >= 60 else ('falling' if event_count <= 8 else 'stable')",
                '                pair = f"{country_a}-{country_b}"',
                "                out.append(",
                "                    {",
                "                        'external_id': hashlib.sha256(f\"{pair}:{timespan_hours}\".encode('utf-8')).hexdigest()[:24],",
                "                        'title': f\"Tension: {pair}\",",
                "                        'summary': f\"score={score:.1f}, trend={trend}, events={event_count}\",",
                "                        'category': category,",
                "                        'source': source,",
                "                        'observed_at': observed_at,",
                "                        'payload': {",
                "                            'severity': severity,",
                "                            'tension_score': score,",
                "                            'event_count': int(event_count),",
                "                            'avg_tone': avg_tone,",
                "                            'avg_goldstein_scale': round(avg_tone * -0.5, 2),",
                "                            'trend': trend,",
                "                            'top_event_types': top_event_types,",
                "                            'country_a': country_a,",
                "                            'country_b': country_b,",
                "                            'query': params.get('query'),",
                "                        },",
                "                        'country': pair,",
                "                        'country_iso3': country_a if len(country_a) == 3 else None,",
                "                        'tags': ['events', 'tension', 'gdelt'],",
                "                    }",
                "                )",
                "                if len(out) >= self._limit():",
                "                    break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "military":
        class_lines.extend(
            [
                "    _MILITARY_PREFIXES = (",
                "        'RCH', 'REACH', 'FORTE', 'HOMER', 'LAGR', 'NATO', 'DUKE',",
                "        'JAKE', 'KING', 'IRON', 'HAVOC', 'VADER', 'GHOST', 'REAPER',",
                "        'ETHYL', 'SNTRY', 'SENTRY', 'EVAC', 'TOPCAT', 'BOLT', 'VIPER',",
                "        'NCHO', 'ECHO', 'GRIM', 'DOOM', 'WRATH', 'NOBLE', 'CHAOS',",
                "        'STONE', 'SCARY', 'BISON', 'FURY', 'ATLAS', 'GIANT', 'MOOSE',",
                "        'MMF', 'RRR', 'CNV', 'CFC', 'IAM', 'GAF', 'BAF', 'FAF',",
                "        'SHF', 'PLF', 'HAF', 'HRZ', 'TUF', 'RFF', 'SVF', 'NRF',",
                "    )",
                "",
                "    _DEFAULT_HOTSPOTS = (",
                "        {'name': 'Ukraine/Black Sea', 'lamin': 44, 'lamax': 52.5, 'lomin': 22, 'lomax': 40},",
                "        {'name': 'Eastern Mediterranean', 'lamin': 32, 'lamax': 37.5, 'lomin': 27, 'lomax': 36},",
                "        {'name': 'South China Sea', 'lamin': 5, 'lamax': 23, 'lomin': 105, 'lomax': 121},",
                "        {'name': 'Taiwan Strait', 'lamin': 22, 'lamax': 26, 'lomin': 117, 'lomax': 123},",
                "        {'name': 'Korean Peninsula', 'lamin': 33, 'lamax': 43, 'lomin': 124, 'lomax': 132},",
                "        {'name': 'Persian Gulf', 'lamin': 23, 'lamax': 30, 'lomin': 48, 'lomax': 57},",
                "        {'name': 'Baltic Sea', 'lamin': 53.5, 'lamax': 60, 'lomin': 13, 'lomax': 30},",
                "        {'name': 'Arctic/Barents', 'lamin': 68, 'lamax': 80, 'lomin': 15, 'lomax': 60},",
                "    )",
                "",
                "    def _is_military_callsign(self, callsign):",
                "        if not callsign:",
                "            return False",
                "        cs = callsign.strip().upper()",
                "        if not cs:",
                "            return False",
                "        for prefix in self._MILITARY_PREFIXES:",
                "            if cs.startswith(prefix):",
                "                return True",
                "        return False",
                "",
                "    async def fetch_async(self):",
                "        hotspots = self.config.get('hotspots') or list(self._DEFAULT_HOTSPOTS)",
                "        if not isinstance(hotspots, list):",
                "            hotspots = list(self._DEFAULT_HOTSPOTS)",
                "        out = []",
                "        seen_icao = set()",
                "        now_iso = datetime.now(timezone.utc).isoformat()",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            for hotspot in hotspots:",
                "                if not isinstance(hotspot, dict):",
                "                    continue",
                "                region_name = str(hotspot.get('name') or 'unknown').strip()",
                "                params = {",
                "                    'lamin': str(hotspot.get('lamin', 0)),",
                "                    'lamax': str(hotspot.get('lamax', 0)),",
                "                    'lomin': str(hotspot.get('lomin', 0)),",
                "                    'lomax': str(hotspot.get('lomax', 0)),",
                "                }",
                "                try:",
                "                    response = await client.get('https://opensky-network.org/api/states/all', params=params)",
                "                except Exception:",
                "                    continue",
                "                if response.status_code == 429:",
                "                    continue",
                "                try:",
                "                    response.raise_for_status()",
                "                    data = response.json()",
                "                except Exception:",
                "                    continue",
                "                states = data.get('states') if isinstance(data, dict) else None",
                "                if not isinstance(states, list):",
                "                    continue",
                "                for state_vec in states:",
                "                    if not isinstance(state_vec, (list, tuple)) or len(state_vec) < 10:",
                "                        continue",
                "                    icao24 = str(state_vec[0] or '').strip().lower()",
                "                    callsign = str(state_vec[1] or '').strip().upper()",
                "                    origin_country = str(state_vec[2] or '').strip()",
                "                    lon = state_vec[5]",
                "                    lat = state_vec[6]",
                "                    baro_alt = state_vec[7]",
                "                    on_ground = bool(state_vec[8]) if state_vec[8] is not None else False",
                "                    velocity = state_vec[9]",
                "                    if not self._is_military_callsign(callsign):",
                "                        continue",
                "                    if icao24 in seen_icao:",
                "                        continue",
                "                    seen_icao.add(icao24)",
                "                    if on_ground:",
                "                        continue",
                "                    severity = 0.6",
                "                    high_interest = ('FORTE', 'REAPER', 'HOMER', 'LAGR', 'SENTRY', 'SNTRY')",
                "                    for hi in high_interest:",
                "                        if callsign.startswith(hi):",
                "                            severity = 0.85",
                "                            break",
                "                    source = 'military'",
                "                    category = 'military'",
                "                    if not self._keep(source, category, severity):",
                "                        continue",
                "                    alt_str = f\"{int(baro_alt)}m\" if baro_alt is not None else 'N/A'",
                "                    vel_str = f\"{int(velocity)}m/s\" if velocity is not None else 'N/A'",
                "                    out.append(",
                "                        {",
                "                            'external_id': hashlib.sha256(f\"opensky:{icao24}:{callsign}\".encode('utf-8')).hexdigest()[:24],",
                "                            'title': f\"Military flight: {callsign} ({origin_country})\",",
                "                            'summary': f\"region={region_name}, alt={alt_str}, speed={vel_str}\",",
                "                            'category': category,",
                "                            'source': source,",
                "                            'observed_at': now_iso,",
                "                            'payload': {",
                "                                'severity': severity,",
                "                                'activity_type': 'flight',",
                "                                'callsign': callsign,",
                "                                'transponder': icao24,",
                "                                'origin_country': origin_country,",
                "                                'altitude_m': baro_alt,",
                "                                'velocity_ms': velocity,",
                "                                'on_ground': on_ground,",
                "                                'region': region_name,",
                "                                'source_name': 'opensky',",
                "                            },",
                "                            'latitude': lat,",
                "                            'longitude': lon,",
                "                            'tags': ['events', 'military', 'flight'],",
                "                        }",
                "                    )",
                "                    if len(out) >= self._limit():",
                "                        return out",
                "            endpoints = self.config.get('endpoints') or []",
                "            if not isinstance(endpoints, list):",
                "                endpoints = []",
                "            for endpoint in endpoints:",
                "                if not isinstance(endpoint, dict):",
                "                    continue",
                "                url = str(endpoint.get('url') or '').strip()",
                "                if not url:",
                "                    continue",
                "                headers = endpoint.get('headers') if isinstance(endpoint.get('headers'), dict) else {}",
                "                json_path = str(endpoint.get('json_path') or '$[*]').strip()",
                "                source_name = str(endpoint.get('source') or 'military').strip().lower() or 'military'",
                "                activity_type_default = str(endpoint.get('activity_type') or 'activity').strip().lower() or 'activity'",
                "                try:",
                "                    response = await client.get(url, headers=headers)",
                "                except Exception:",
                "                    continue",
                "                if response.status_code == 429:",
                "                    continue",
                "                try:",
                "                    response.raise_for_status()",
                "                    payload = response.json()",
                "                except Exception:",
                "                    continue",
                "                rows = self._extract_json_path(payload, json_path)",
                "                for activity in rows:",
                "                    activity_type = str(activity.get('activity_type') or activity.get('type') or activity_type_default).strip().lower() or activity_type_default",
                "                    callsign = str(activity.get('callsign') or activity.get('name') or '').strip().upper()",
                "                    region = str(activity.get('region') or activity.get('area') or '').strip()",
                "                    is_unusual = bool(activity.get('is_unusual') or activity.get('anomaly') or False)",
                "                    severity = self._as_float(activity.get('severity'), 0.7 if is_unusual else 0.5)",
                "                    severity = self._clamp(severity, 0.0, 1.0)",
                "                    source = source_name",
                "                    category = 'military'",
                "                    if not self._keep(source, category, severity):",
                "                        continue",
                "                    observed = self._parse_datetime(activity.get('detected_at') or activity.get('timestamp') or activity.get('time'))",
                "                    observed_at = observed.isoformat() if observed is not None else None",
                "                    entity = str(activity.get('transponder') or activity.get('id') or callsign or region or activity_type).strip().lower()",
                "                    out.append(",
                "                        {",
                "                            'external_id': str(activity.get('id') or '').strip() or hashlib.sha256(f\"{activity_type}:{entity}:{observed_at}\".encode('utf-8')).hexdigest()[:24],",
                "                            'title': f\"Military {activity_type}: {callsign or entity}\",",
                "                            'summary': f\"region={region}, source={source}\",",
                "                            'category': category,",
                "                            'source': source,",
                "                            'observed_at': observed_at,",
                "                            'payload': {",
                "                                'severity': severity,",
                "                                'activity_type': activity_type,",
                "                                'callsign': callsign,",
                "                                'source_name': source,",
                "                                'is_unusual': is_unusual,",
                "                                'region': region,",
                "                                'raw': activity,",
                "                            },",
                "                            'latitude': activity.get('latitude') if activity.get('latitude') is not None else activity.get('lat'),",
                "                            'longitude': activity.get('longitude') if activity.get('longitude') is not None else activity.get('lon'),",
                "                            'country_iso3': activity.get('country_iso3') or activity.get('country'),",
                "                            'tags': ['events', 'military', activity_type],",
                "                        }",
                "                    )",
                "                    if len(out) >= self._limit():",
                "                        return out",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "infrastructure":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://api.gdeltproject.org/api/v2/doc/doc').strip()",
                "        timespan_hours = self._as_int(self.config.get('timespan_hours'), max(1, min(168, self._hours())), 1, 168)",
                "        max_per_query = self._as_int(self.config.get('maxrecords_per_query'), 30, 10, 250)",
                "        query_rows = self.config.get('queries')",
                "        if not isinstance(query_rows, list) or not query_rows:",
                "            query_rows = [",
                "                {'name': 'Pipeline Disruption', 'query': '(pipeline explosion OR pipeline shutdown OR pipeline sabotage)', 'priority': 'high', 'enabled': True},",
                "                {'name': 'Power Grid Failure', 'query': '(power outage OR grid failure OR blackout OR electricity crisis)', 'priority': 'high', 'enabled': True},",
                "                {'name': 'Undersea Cable', 'query': '(undersea cable cut OR submarine cable damaged OR internet cable severed)', 'priority': 'critical', 'enabled': True},",
                "                {'name': 'Port & Shipping', 'query': '(port closure OR port blockade OR shipping disruption OR canal blocked)', 'priority': 'high', 'enabled': True},",
                "                {'name': 'Telecom Outage', 'query': '(telecom outage OR network outage OR cell tower OR internet outage)', 'priority': 'medium', 'enabled': True},",
                "                {'name': 'Water & Dam', 'query': '(dam failure OR dam breach OR water contamination OR water crisis)', 'priority': 'high', 'enabled': True},",
                "            ]",
                "        priority_weight = {'critical': 0.75, 'high': 0.6, 'medium': 0.45}",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            for row in query_rows:",
                "                if isinstance(row, dict):",
                "                    if not bool(row.get('enabled', True)):",
                "                        continue",
                "                    query = str(row.get('query') or '').strip()",
                "                    query_name = str(row.get('name') or query).strip()",
                "                    priority = str(row.get('priority') or 'medium').strip().lower() or 'medium'",
                "                else:",
                "                    query = str(row or '').strip()",
                "                    query_name = query",
                "                    priority = 'medium'",
                "                if not query:",
                "                    continue",
                "                params = {",
                "                    'query': query,",
                "                    'mode': 'artlist',",
                "                    'format': 'json',",
                "                    'sort': 'datedesc',",
                "                    'timespan': f'{timespan_hours}h',",
                "                    'maxrecords': str(max_per_query),",
                "                }",
                "                try:",
                "                    response = await client.get(endpoint, params=params)",
                "                except Exception:",
                "                    continue",
                "                if response.status_code == 429:",
                "                    continue",
                "                try:",
                "                    response.raise_for_status()",
                "                except Exception:",
                "                    continue",
                "                try:",
                "                    payload = response.json()",
                "                except Exception:",
                "                    continue",
                "                articles = payload.get('articles') if isinstance(payload, dict) else []",
                "                if not isinstance(articles, list):",
                "                    articles = []",
                "                for item in list(articles):",
                "                    if not isinstance(item, dict):",
                "                        continue",
                "                    source = 'infrastructure'",
                "                    category = 'infrastructure'",
                "                    tone = self._as_float(item.get('tone'), 0.0)",
                "                    base = priority_weight.get(priority, 0.35)",
                "                    severity = self._clamp(base + max(0.0, (-tone / 25.0)), 0.0, 1.0)",
                "                    if not self._keep(source, category, severity):",
                "                        continue",
                "                    url = str(item.get('url') or '').strip()",
                "                    title = str(item.get('title') or '').strip() or 'Infrastructure disruption article'",
                "                    observed = self._parse_datetime(item.get('seendate') or item.get('published') or item.get('date'))",
                "                    observed_at = observed.isoformat() if observed is not None else None",
                "                    themes = str(item.get('socialimage') or item.get('domain') or '').strip()",
                '                    key = f"{query}:{url}:{observed_at}:{title}"',
                "                    out.append(",
                "                        {",
                "                            'external_id': hashlib.sha256(key.encode('utf-8')).hexdigest()[:24],",
                "                            'title': title,",
                "                            'summary': f'{query_name}: {title[:400]}'[:500],",
                "                            'category': category,",
                "                            'source': source,",
                "                            'observed_at': observed_at,",
                "                            'url': url or None,",
                "                            'payload': {",
                "                                'severity': severity,",
                "                                'priority': priority,",
                "                                'query': query,",
                "                                'query_name': query_name,",
                "                                'tone': tone,",
                "                                'domain': item.get('domain'),",
                "                                'language': item.get('language'),",
                "                                'sourcecountry': item.get('sourcecountry'),",
                "                            },",
                "                            'country_iso3': item.get('sourcecountry'),",
                "                            'tags': ['events', 'infrastructure', query_name.lower().replace(' ', '_')],",
                "                        }",
                "                    )",
                "                    if len(out) >= self._limit():",
                "                        return out",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "gdelt_news":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://api.gdeltproject.org/api/v2/doc/doc').strip()",
                "        timespan_hours = self._as_int(self.config.get('timespan_hours'), max(1, min(168, self._hours())), 1, 168)",
                "        max_per_query = self._as_int(self.config.get('maxrecords_per_query'), 40, 10, 250)",
                "        query_rows = self.config.get('queries')",
                "        if not isinstance(query_rows, list) or not query_rows:",
                "            query_rows = [",
                "                {'name': 'Global Conflict Escalation', 'query': '(war OR conflict OR invasion OR ceasefire OR sanctions)', 'priority': 'high', 'enabled': True},",
                "                {'name': 'Central Bank & Macro Policy', 'query': '(federal reserve OR ECB OR interest rates OR inflation)', 'priority': 'medium', 'enabled': True},",
                "                {'name': 'Energy Supply Disruption', 'query': '(oil supply disruption OR LNG outage OR pipeline explosion)', 'priority': 'high', 'enabled': True},",
                "            ]",
                "        priority_weight = {'critical': 0.75, 'high': 0.6, 'medium': 0.45}",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            for row in query_rows:",
                "                if isinstance(row, dict):",
                "                    if not bool(row.get('enabled', True)):",
                "                        continue",
                "                    query = str(row.get('query') or '').strip()",
                "                    query_name = str(row.get('name') or query).strip()",
                "                    priority = str(row.get('priority') or 'medium').strip().lower() or 'medium'",
                "                    country_iso3 = str(row.get('country_iso3') or '').strip().upper() or None",
                "                else:",
                "                    query = str(row or '').strip()",
                "                    query_name = query",
                "                    priority = 'medium'",
                "                    country_iso3 = None",
                "                if not query:",
                "                    continue",
                "                params = {",
                "                    'query': query,",
                "                    'mode': 'artlist',",
                "                    'format': 'json',",
                "                    'sort': 'datedesc',",
                "                    'timespan': f'{timespan_hours}h',",
                "                    'maxrecords': str(max_per_query),",
                "                }",
                "                try:",
                "                    response = await client.get(endpoint, params=params)",
                "                except Exception:",
                "                    continue",
                "                if response.status_code == 429:",
                "                    continue",
                "                try:",
                "                    response.raise_for_status()",
                "                except Exception:",
                "                    continue",
                "                try:",
                "                    payload = response.json()",
                "                except Exception:",
                "                    continue",
                "                articles = payload.get('articles') if isinstance(payload, dict) else []",
                "                if not isinstance(articles, list):",
                "                    articles = []",
                "                for item in list(articles):",
                "                    if not isinstance(item, dict):",
                "                        continue",
                "                    source = 'gdelt_news'",
                "                    category = 'news'",
                "                    tone = self._as_float(item.get('tone'), 0.0)",
                "                    base = priority_weight.get(priority, 0.35)",
                "                    severity = self._clamp(base + max(0.0, (-tone / 25.0)), 0.0, 1.0)",
                "                    if not self._keep(source, category, severity):",
                "                        continue",
                "                    url = str(item.get('url') or '').strip()",
                "                    title = str(item.get('title') or '').strip() or 'GDELT world article'",
                "                    observed = self._parse_datetime(item.get('seendate') or item.get('published') or item.get('date'))",
                "                    observed_at = observed.isoformat() if observed is not None else None",
                "                    summary = str(item.get('seendate') or item.get('domain') or item.get('sourcecountry') or '').strip()",
                '                    key = f"{query}:{url}:{observed_at}:{title}"',
                "                    out.append(",
                "                        {",
                "                            'external_id': hashlib.sha256(key.encode('utf-8')).hexdigest()[:24],",
                "                            'title': title,",
                "                            'summary': summary[:500],",
                "                            'category': category,",
                "                            'source': source,",
                "                            'observed_at': observed_at,",
                "                            'url': url or None,",
                "                            'payload': {",
                "                                'severity': severity,",
                "                                'priority': priority,",
                "                                'query': query,",
                "                                'query_name': query_name,",
                "                                'tone': tone,",
                "                                'domain': item.get('domain'),",
                "                                'language': item.get('language'),",
                "                                'sourcecountry': item.get('sourcecountry'),",
                "                            },",
                "                            'country_iso3': country_iso3 or item.get('sourcecountry'),",
                "                            'tags': ['events', 'news', 'gdelt'],",
                "                        }",
                "                    )",
                "                    if len(out) >= self._limit():",
                "                        return out",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "usgs":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        min_magnitude = self._as_float(self.config.get('min_magnitude'), self.default_config.get('min_magnitude', 4.5))",
                "        endpoint = str(self.config.get('endpoint') or 'https://earthquake.usgs.gov/fdsnws/event/1/query').strip()",
                "        now = datetime.now(timezone.utc)",
                "        start = now - timedelta(hours=self._hours())",
                "        params = {",
                "            'format': 'geojson',",
                "            'starttime': start.isoformat(),",
                "            'endtime': now.isoformat(),",
                "            'minmagnitude': f\"{min_magnitude:.1f}\",",
                "            'limit': str(min(max(self._limit() * 2, self._limit()), 2000)),",
                "            'orderby': 'time',",
                "        }",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            try:",
                "                response = await client.get(endpoint, params=params)",
                "            except Exception:",
                "                return []",
                "            if response.status_code == 429:",
                "                return []",
                "            try:",
                "                response.raise_for_status()",
                "                payload = response.json()",
                "            except Exception:",
                "                return []",
                "        quakes = payload.get('features') if isinstance(payload, dict) else []",
                "        if not isinstance(quakes, list):",
                "            quakes = []",
                "        out = []",
                "        for quake in list(quakes or []):",
                "            if not isinstance(quake, dict):",
                "                continue",
                "            properties = quake.get('properties') if isinstance(quake.get('properties'), dict) else {}",
                "            geometry = quake.get('geometry') if isinstance(quake.get('geometry'), dict) else {}",
                "            coordinates = geometry.get('coordinates') if isinstance(geometry.get('coordinates'), list) else []",
                "            source = 'usgs'",
                "            category = 'earthquake'",
                "            magnitude = self._as_float(properties.get('mag'), 0.0)",
                "            severity = self._clamp(max(0.1, magnitude / 10.0), 0.0, 1.0)",
                "            if not self._keep(source, category, severity):",
                "                continue",
                "            url = str(properties.get('url') or '').strip()",
                "            observed = self._parse_datetime(properties.get('time'))",
                "            observed_at = observed.isoformat() if observed is not None else None",
                "            latitude = coordinates[1] if len(coordinates) > 1 else None",
                "            longitude = coordinates[0] if len(coordinates) > 0 else None",
                "            depth = coordinates[2] if len(coordinates) > 2 else None",
                "            out.append(",
                "                {",
                "                    'external_id': str(quake.get('id') or '').strip() or hashlib.sha256(f\"{url}:{observed_at}\".encode('utf-8')).hexdigest()[:24],",
                "                    'title': f\"M{magnitude:.1f} Earthquake: {str(properties.get('place') or '').strip()}\",",
                "                    'summary': f\"depth_km={self._as_float(depth, 0.0):.1f}\",",
                "                    'category': category,",
                "                    'source': source,",
                "                    'observed_at': observed_at,",
                "                    'url': url or None,",
                "                    'payload': {",
                "                        'severity': severity,",
                "                        'magnitude': magnitude,",
                "                        'depth': self._as_float(depth, 0.0),",
                "                        'alert': properties.get('alert'),",
                "                        'tsunami': int(properties.get('tsunami') or 0),",
                "                        'status': properties.get('status'),",
                "                    },",
                "                    'latitude': latitude,",
                "                    'longitude': longitude,",
                "                    'tags': ['events', 'earthquake', 'usgs'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "ucdp_conflicts":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://ucdpapi.pcr.uu.se/api/gedevents/24.1').strip()",
                "        pagesize = self._as_int(self.config.get('pagesize'), 500, 50, 1000)",
                "        max_pages = self._as_int(self.config.get('max_pages'), 5, 1, 20)",
                "        since = datetime.now(timezone.utc) - timedelta(hours=self._hours())",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:",
                "            for page in range(1, max_pages + 1):",
                "                params = {",
                "                    'pagesize': str(pagesize),",
                "                    'page': str(page),",
                "                    'StartDate': since.strftime('%Y-%m-%d'),",
                "                }",
                "                try:",
                "                    response = await client.get(endpoint, params=params)",
                "                except Exception:",
                "                    break",
                "                if response.status_code == 429:",
                "                    break",
                "                try:",
                "                    response.raise_for_status()",
                "                    payload = response.json()",
                "                except Exception:",
                "                    break",
                "                results = payload.get('Result') if isinstance(payload, dict) else []",
                "                if not isinstance(results, list) or not results:",
                "                    break",
                "                for event in results:",
                "                    if not isinstance(event, dict):",
                "                        continue",
                "                    best_est = self._as_float(event.get('best'), 0.0)",
                "                    severity = self._clamp(max(0.15, best_est / 50.0), 0.0, 1.0)",
                "                    source = 'ucdp'",
                "                    category = 'conflict'",
                "                    if not self._keep(source, category, severity):",
                "                        continue",
                "                    country_name = str(event.get('country') or '').strip()",
                "                    where_desc = str(event.get('where_description') or '').strip()",
                "                    observed = self._parse_datetime(event.get('date_start') or event.get('date_end'))",
                "                    observed_at = observed.isoformat() if observed is not None else None",
                "                    event_id = str(event.get('id') or '').strip()",
                "                    out.append(",
                "                        {",
                "                            'external_id': event_id or hashlib.sha256(f\"{country_name}:{where_desc}:{observed_at}\".encode('utf-8')).hexdigest()[:24],",
                "                            'title': f\"UCDP Conflict: {where_desc or country_name}\",",
                "                            'summary': f\"fatalities_best={int(best_est)}, country={country_name}\",",
                "                            'category': category,",
                "                            'source': source,",
                "                            'observed_at': observed_at,",
                "                            'payload': {",
                "                                'severity': severity,",
                "                                'best_estimate': int(best_est),",
                "                                'high_estimate': self._as_float(event.get('high'), 0.0),",
                "                                'low_estimate': self._as_float(event.get('low'), 0.0),",
                "                                'type_of_violence': event.get('type_of_violence'),",
                "                                'side_a': event.get('side_a'),",
                "                                'side_b': event.get('side_b'),",
                "                                'where_description': where_desc,",
                "                            },",
                "                            'latitude': event.get('latitude'),",
                "                            'longitude': event.get('longitude'),",
                "                            'country_iso3': event.get('country_id'),",
                "                            'tags': ['events', 'conflict', 'ucdp'],",
                "                        }",
                "                    )",
                "                    if len(out) >= self._limit():",
                "                        return out",
                "                if len(results) < pagesize:",
                "                    break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "trade_dependencies":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://api.worldbank.org/v2/country/all/indicator/TG.VAL.TOTL.GD.ZS').strip()",
                "        now = datetime.now(timezone.utc)",
                "        year_end = now.year",
                "        year_start = year_end - 5",
                "        params = {",
                "            'format': 'json',",
                "            'per_page': '500',",
                "            'date': f'{year_start}:{year_end}',",
                "        }",
                "        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:",
                "            try:",
                "                response = await client.get(endpoint, params=params)",
                "            except Exception:",
                "                return []",
                "            if response.status_code == 429:",
                "                return []",
                "            try:",
                "                response.raise_for_status()",
                "                payload = response.json()",
                "            except Exception:",
                "                return []",
                "        if not isinstance(payload, list) or len(payload) < 2:",
                "            return []",
                "        data = payload[1] if isinstance(payload[1], list) else []",
                "        out = []",
                "        for entry in data:",
                "            if not isinstance(entry, dict):",
                "                continue",
                "            value = entry.get('value')",
                "            if value is None:",
                "                continue",
                "            trade_pct = self._as_float(value, 0.0)",
                "            iso3 = str(entry.get('countryiso3code') or '').strip().upper()",
                "            if not iso3 or len(iso3) != 3:",
                "                continue",
                "            year = str(entry.get('date') or '').strip()",
                "            severity = self._clamp(trade_pct / 200.0, 0.0, 1.0)",
                "            source = 'world_bank'",
                "            category = 'trade'",
                "            if not self._keep(source, category, severity):",
                "                continue",
                "            country_name = str((entry.get('country') or {}).get('value') or iso3).strip()",
                "            out.append(",
                "                {",
                "                    'external_id': f'{iso3}:{year}',",
                "                    'title': f'Trade Dependency: {country_name} ({year})',",
                "                    'summary': f'trade_to_gdp={trade_pct:.1f}%',",
                "                    'category': category,",
                "                    'source': source,",
                "                    'observed_at': datetime.now(timezone.utc).isoformat(),",
                "                    'payload': {",
                "                        'severity': severity,",
                "                        'trade_to_gdp_pct': trade_pct,",
                "                        'year': year,",
                "                        'indicator': 'TG.VAL.TOTL.GD.ZS',",
                "                    },",
                "                    'country_iso3': iso3,",
                "                    'tags': ['events', 'trade', 'world_bank'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "chokepoint_reference":
        class_lines.extend(
            [
                "    _FALLBACK_CHOKEPOINTS = [",
                "        {'name': 'Strait of Hormuz', 'lat': 26.56, 'lon': 56.25, 'daily_vessels': 80, 'risk': 0.72, 'region': 'Middle East'},",
                "        {'name': 'Suez Canal', 'lat': 30.46, 'lon': 32.34, 'daily_vessels': 55, 'risk': 0.55, 'region': 'Middle East'},",
                "        {'name': 'Panama Canal', 'lat': 9.08, 'lon': -79.68, 'daily_vessels': 40, 'risk': 0.40, 'region': 'Americas'},",
                "        {'name': 'Strait of Malacca', 'lat': 2.50, 'lon': 101.80, 'daily_vessels': 85, 'risk': 0.48, 'region': 'Asia'},",
                "        {'name': 'Bab el-Mandeb', 'lat': 12.58, 'lon': 43.33, 'daily_vessels': 30, 'risk': 0.68, 'region': 'Middle East'},",
                "        {'name': 'Bosporus', 'lat': 41.12, 'lon': 29.05, 'daily_vessels': 45, 'risk': 0.38, 'region': 'Europe'},",
                "        {'name': 'GIUK Gap', 'lat': 63.00, 'lon': -20.00, 'daily_vessels': 25, 'risk': 0.35, 'region': 'North Atlantic'},",
                "        {'name': 'Strait of Gibraltar', 'lat': 35.96, 'lon': -5.50, 'daily_vessels': 60, 'risk': 0.30, 'region': 'Europe'},",
                "        {'name': 'Cape of Good Hope', 'lat': -34.36, 'lon': 18.47, 'daily_vessels': 30, 'risk': 0.28, 'region': 'Africa'},",
                "        {'name': 'Taiwan Strait', 'lat': 24.00, 'lon': 119.50, 'daily_vessels': 50, 'risk': 0.65, 'region': 'Asia'},",
                "        {'name': 'Danish Straits', 'lat': 55.70, 'lon': 12.60, 'daily_vessels': 35, 'risk': 0.22, 'region': 'Europe'},",
                "        {'name': 'Lombok Strait', 'lat': -8.47, 'lon': 115.72, 'daily_vessels': 20, 'risk': 0.20, 'region': 'Asia'},",
                "        {'name': 'Tsugaru Strait', 'lat': 41.65, 'lon': 140.80, 'daily_vessels': 15, 'risk': 0.18, 'region': 'Asia'},",
                "        {'name': 'Mozambique Channel', 'lat': -17.00, 'lon': 41.00, 'daily_vessels': 12, 'risk': 0.25, 'region': 'Africa'},",
                "        {'name': 'Korea Strait', 'lat': 34.00, 'lon': 129.00, 'daily_vessels': 40, 'risk': 0.32, 'region': 'Asia'},",
                "    ]",
                "",
                "    def _build_record(self, cp, now_iso):",
                "        name = str(cp.get('name') or '').strip()",
                "        risk = self._clamp(self._as_float(cp.get('risk'), 0.3), 0.0, 1.0)",
                "        source = 'chokepoints'",
                "        category = 'infrastructure'",
                "        if not self._keep(source, category, risk):",
                "            return None",
                "        return {",
                "            'external_id': hashlib.sha256(name.encode('utf-8')).hexdigest()[:24],",
                "            'title': f'Chokepoint: {name}',",
                "            'summary': f\"region={cp.get('region')}, daily_vessels={cp.get('daily_vessels')}\",",
                "            'category': category,",
                "            'source': source,",
                "            'observed_at': now_iso,",
                "            'payload': {",
                "                'severity': risk,",
                "                'risk_score': risk,",
                "                'daily_vessel_count': cp.get('daily_vessels'),",
                "                'region': cp.get('region'),",
                "                'chokepoint_name': name,",
                "            },",
                "            'latitude': cp.get('lat'),",
                "            'longitude': cp.get('lon'),",
                "            'tags': ['events', 'infrastructure', 'chokepoint'],",
                "        }",
                "",
                "    async def fetch_async(self):",
                "        portwatch_url = str(self.config.get('portwatch_points_url') or (",
                "            'https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/'",
                "            'PortWatch_chokepoints_database/FeatureServer/0/query'",
                "        )).strip()",
                "        now_iso = datetime.now(timezone.utc).isoformat()",
                "        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:",
                "            try:",
                "                params = {'where': '1=1', 'outFields': '*', 'f': 'json', 'resultRecordCount': '200'}",
                "                response = await client.get(portwatch_url, params=params)",
                "                response.raise_for_status()",
                "                data = response.json()",
                "            except Exception:",
                "                data = None",
                "        if isinstance(data, dict) and isinstance(data.get('features'), list) and data['features']:",
                "            out = []",
                "            for feature in data['features']:",
                "                if not isinstance(feature, dict):",
                "                    continue",
                "                attrs = feature.get('attributes') or {}",
                "                geom = feature.get('geometry') or {}",
                "                name = str(attrs.get('Name') or attrs.get('name') or attrs.get('chokepoint_name') or '').strip()",
                "                if not name:",
                "                    continue",
                "                lat = self._as_float(geom.get('y') or attrs.get('latitude') or attrs.get('lat'), None)",
                "                lon = self._as_float(geom.get('x') or attrs.get('longitude') or attrs.get('lon'), None)",
                "                daily_vessels = self._as_int(attrs.get('daily_vessels') or attrs.get('DailyVessels') or attrs.get('vessel_count'), 0, 0, 99999)",
                "                region = str(attrs.get('Region') or attrs.get('region') or '').strip()",
                "                risk = self._clamp(self._as_float(attrs.get('risk_score') or attrs.get('Risk') or attrs.get('risk'), 0.3), 0.0, 1.0)",
                "                cp = {'name': name, 'lat': lat, 'lon': lon, 'daily_vessels': daily_vessels, 'risk': risk, 'region': region}",
                "                record = self._build_record(cp, now_iso)",
                "                if record is not None:",
                "                    out.append(record)",
                "                if len(out) >= self._limit():",
                "                    break",
                "            if out:",
                "                return out",
                "        out = []",
                "        for cp in self._FALLBACK_CHOKEPOINTS:",
                "            record = self._build_record(cp, now_iso)",
                "            if record is not None:",
                "                out.append(record)",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "country_instability":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        lookback_hours = self._as_int(self.config.get('lookback_hours'), 72, 12, 720)",
                "        min_signals = self._as_int(self.config.get('min_signals'), 2, 1, 50)",
                "        conflict_weight = self._as_float(self.config.get('conflict_weight'), 0.40)",
                "        tension_weight = self._as_float(self.config.get('tension_weight'), 0.35)",
                "        military_weight = self._as_float(self.config.get('military_weight'), 0.25)",
                "        from sqlalchemy import select as sa_select",
                "        from models.database import EventsSignal as _EventsSignal, AsyncSessionLocal as _AsyncSessionLocal",
                "        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=lookback_hours)",
                "        async with _AsyncSessionLocal() as session:",
                "            rows = (",
                "                (",
                "                    await session.execute(",
                "                        sa_select(_EventsSignal)",
                "                        .where(_EventsSignal.detected_at >= cutoff)",
                "                        .where(_EventsSignal.signal_type.in_(['conflict', 'tension', 'military', 'instability']))",
                "                        .order_by(_EventsSignal.detected_at.desc())",
                "                        .limit(5000)",
                "                    )",
                "                )",
                "                .scalars()",
                "                .all()",
                "            )",
                "        by_country = {}",
                "        for sig in rows:",
                "            iso3 = str(sig.iso3 or sig.country or '').strip().upper()",
                "            if not iso3 or len(iso3) != 3:",
                "                continue",
                "            if iso3 not in by_country:",
                "                by_country[iso3] = {'conflict': [], 'tension': [], 'military': [], 'other': []}",
                "            stype = str(sig.signal_type or '').strip().lower()",
                "            bucket = by_country[iso3].get(stype, by_country[iso3]['other'])",
                "            bucket.append(float(sig.severity or 0.0))",
                "        out = []",
                "        for iso3, buckets in by_country.items():",
                "            total_signals = sum(len(v) for v in buckets.values())",
                "            if total_signals < min_signals:",
                "                continue",
                "            avg_conflict = (sum(buckets['conflict']) / len(buckets['conflict'])) if buckets['conflict'] else 0.0",
                "            avg_tension = (sum(buckets['tension']) / len(buckets['tension'])) if buckets['tension'] else 0.0",
                "            avg_military = (sum(buckets['military']) / len(buckets['military'])) if buckets['military'] else 0.0",
                "            composite = self._clamp(",
                "                (avg_conflict * conflict_weight) + (avg_tension * tension_weight) + (avg_military * military_weight),",
                "                0.0, 1.0",
                "            )",
                "            score = round(composite * 100.0, 1)",
                "            trend = 'rising' if total_signals >= 8 else ('falling' if total_signals <= 2 else 'stable')",
                "            source = 'instability_index'",
                "            category = 'instability'",
                "            if not self._keep(source, category, composite):",
                "                continue",
                "            out.append(",
                "                {",
                "                    'external_id': f'instability:{iso3}:{lookback_hours}h',",
                "                    'title': f'Instability: {iso3}',",
                "                    'summary': f'score={score:.1f}, trend={trend}, signals={total_signals}',",
                "                    'category': category,",
                "                    'source': source,",
                "                    'observed_at': datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),",
                "                    'payload': {",
                "                        'severity': composite,",
                "                        'score': score,",
                "                        'trend': trend,",
                "                        'total_signals': total_signals,",
                "                        'components': {",
                "                            'conflict': round(avg_conflict * 100.0, 1),",
                "                            'tension': round(avg_tension * 100.0, 1),",
                "                            'military': round(avg_military * 100.0, 1),",
                "                        },",
                "                        'contributing_signals': [",
                "                            {'type': 'conflict', 'count': len(buckets['conflict'])},",
                "                            {'type': 'tension', 'count': len(buckets['tension'])},",
                "                            {'type': 'military', 'count': len(buckets['military'])},",
                "                        ],",
                "                    },",
                "                    'country_iso3': iso3,",
                "                    'tags': ['events', 'instability', 'derived'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "airplanes_live":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://api.airplanes.live/v2/mil').strip()",
                "        timeout = self._as_float(self.config.get('timeout'), 20.0)",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:",
                "            try:",
                "                resp = await client.get(endpoint)",
                "                resp.raise_for_status()",
                "                data = resp.json()",
                "            except Exception:",
                "                return []",
                "        aircraft_list = data.get('ac') if isinstance(data, dict) else []",
                "        if not isinstance(aircraft_list, list):",
                "            aircraft_list = []",
                "        seen_icao = set()",
                "        source = 'airplanes_live'",
                "        category = 'military'",
                "        for ac in aircraft_list:",
                "            if not isinstance(ac, dict):",
                "                continue",
                "            icao = str(ac.get('hex') or '').strip().upper()",
                "            if not icao or icao in seen_icao:",
                "                continue",
                "            seen_icao.add(icao)",
                "            lat = self._as_float(ac.get('lat'))",
                "            lon = self._as_float(ac.get('lon'))",
                "            if lat is None or lon is None:",
                "                continue",
                "            alt = self._as_float(ac.get('alt_baro'))",
                "            if alt is not None and alt <= 0:",
                "                continue",
                "            callsign = str(ac.get('flight') or '').strip()",
                "            operator = str(ac.get('ownOp') or '').strip()",
                "            ac_type = str(ac.get('t') or '').strip()",
                "            reg = str(ac.get('r') or '').strip()",
                "            speed = self._as_float(ac.get('gs'))",
                "            severity = 0.5",
                "            high_interest = ('FORTE', 'REAPER', 'HOMER', 'LAGR', 'SENTRY', 'SNTRY', 'RCH', 'DUKE')",
                "            for hi in high_interest:",
                "                if callsign.upper().startswith(hi):",
                "                    severity = 0.85",
                "                    break",
                "            if not self._keep(source, category, severity):",
                "                continue",
                "            alt_str = f\"{int(alt)}ft\" if alt is not None else 'N/A'",
                "            spd_str = f\"{int(speed)}kn\" if speed is not None else 'N/A'",
                "            summary_parts = [f'alt={alt_str}', f'spd={spd_str}']",
                "            if operator:",
                "                summary_parts.insert(0, operator)",
                "            out.append(",
                "                {",
                "                    'external_id': hashlib.sha256(f\"apl:{icao}\".encode('utf-8')).hexdigest()[:24],",
                "                    'title': f\"Military aircraft: {callsign or icao}\",",
                "                    'summary': ', '.join(summary_parts),",
                "                    'category': category,",
                "                    'source': source,",
                "                    'latitude': lat,",
                "                    'longitude': lon,",
                "                    'observed_at': datetime.now(timezone.utc).isoformat(),",
                "                    'payload': {",
                "                        'severity': severity,",
                "                        'icao24': icao,",
                "                        'callsign': callsign,",
                "                        'operator': operator,",
                "                        'aircraft_type': ac_type,",
                "                        'registration': reg,",
                "                        'altitude_ft': alt,",
                "                        'ground_speed_kn': speed,",
                "                        'activity_type': 'surveillance' if severity >= 0.8 else 'patrol',",
                "                    },",
                "                    'tags': ['events', 'military', 'aircraft', 'adsb'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "ais_ships":
        class_lines.extend(
            [
                "    _CHOKEPOINT_BBOXES = {",
                "        'Strait of Hormuz': (24.0, 54.0, 27.5, 57.5),",
                "        'Bab el-Mandeb': (11.5, 42.5, 13.5, 44.0),",
                "        'Suez Canal': (29.5, 32.0, 31.5, 33.0),",
                "        'Strait of Malacca': (-1.5, 99.5, 4.5, 104.5),",
                "        'Taiwan Strait': (22.5, 117.0, 26.0, 121.0),",
                "        'Danish Straits': (54.5, 9.5, 58.0, 13.0),",
                "        'English Channel': (49.5, -2.0, 51.5, 2.0),",
                "        'Strait of Gibraltar': (35.5, -6.5, 36.5, -5.0),",
                "        'Cape of Good Hope': (-35.5, 17.0, -33.0, 20.5),",
                "        'Panama Canal': (8.5, -80.0, 9.5, -79.0),",
                "    }",
                "",
                "    async def fetch_async(self):",
                "        endpoint = str(self.config.get('endpoint') or 'https://meri.digitraffic.fi/api/ais/v1/locations').strip()",
                "        timeout = self._as_float(self.config.get('timeout'), 30.0)",
                "        out = []",
                "        headers = {'Accept-Encoding': 'gzip'}",
                "        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:",
                "            try:",
                "                resp = await client.get(endpoint, headers=headers)",
                "                resp.raise_for_status()",
                "                data = resp.json()",
                "            except Exception:",
                "                return []",
                "        features = data.get('features') if isinstance(data, dict) else []",
                "        if not isinstance(features, list):",
                "            features = []",
                "        source = 'ais_ships'",
                "        category = 'shipping'",
                "        seen_mmsi = set()",
                "        for feat in features:",
                "            if not isinstance(feat, dict):",
                "                continue",
                "            mmsi = feat.get('mmsi')",
                "            props = feat.get('properties') if isinstance(feat.get('properties'), dict) else {}",
                "            if not mmsi:",
                "                mmsi = props.get('mmsi')",
                "            if not mmsi or mmsi in seen_mmsi:",
                "                continue",
                "            seen_mmsi.add(mmsi)",
                "            geom = feat.get('geometry') if isinstance(feat.get('geometry'), dict) else {}",
                "            coords = geom.get('coordinates') if isinstance(geom.get('coordinates'), list) else []",
                "            if len(coords) < 2:",
                "                continue",
                "            lon = self._as_float(coords[0])",
                "            lat = self._as_float(coords[1])",
                "            if lat is None or lon is None:",
                "                continue",
                "            sog = self._as_float(props.get('sog'), 0.0)",
                "            cog = self._as_float(props.get('cog'), 0.0)",
                "            heading = self._as_float(props.get('heading'), 0.0)",
                "            nav_status = self._as_int(props.get('navStat'), -1, -1, 99)",
                "            if nav_status in (1, 5, 6):",
                "                continue",
                "            chokepoint_name = None",
                "            for cp_name, bbox in self._CHOKEPOINT_BBOXES.items():",
                "                min_lat, min_lon, max_lat, max_lon = bbox",
                "                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:",
                "                    chokepoint_name = cp_name",
                "                    break",
                "            if not chokepoint_name:",
                "                continue",
                "            severity = 0.5",
                "            if not self._keep(source, category, severity):",
                "                continue",
                "            out.append(",
                "                {",
                "                    'external_id': hashlib.sha256(f\"ais:{mmsi}\".encode('utf-8')).hexdigest()[:24],",
                "                    'title': f\"Vessel {mmsi} near {chokepoint_name}\",",
                "                    'summary': f\"sog={sog:.1f}kn, cog={cog:.0f}\\u00b0, heading={heading:.0f}\\u00b0\",",
                "                    'category': category,",
                "                    'source': source,",
                "                    'latitude': lat,",
                "                    'longitude': lon,",
                "                    'observed_at': datetime.now(timezone.utc).isoformat(),",
                "                    'payload': {",
                "                        'severity': severity,",
                "                        'mmsi': mmsi,",
                "                        'speed_over_ground': sog,",
                "                        'course_over_ground': cog,",
                "                        'heading': heading,",
                "                        'chokepoint': chokepoint_name,",
                "                        'nav_status': nav_status,",
                "                    },",
                "                    'tags': ['events', 'shipping', 'ais', 'vessel'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    elif normalized_source_id == "nasa_firms":
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        import csv",
                "        import io",
                "        url = str(self.config.get('endpoint') or 'https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv').strip()",
                "        min_confidence = str(self.config.get('min_confidence') or 'high').strip().lower()",
                "        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:",
                "            try:",
                "                response = await client.get(url)",
                "            except Exception:",
                "                return []",
                "            if response.status_code == 429:",
                "                return []",
                "            try:",
                "                response.raise_for_status()",
                "            except Exception:",
                "                return []",
                "        reader = csv.DictReader(io.StringIO(response.text))",
                "        raw = []",
                "        for row in reader:",
                "            confidence = str(row.get('confidence') or '').strip().lower()",
                "            if min_confidence == 'high' and confidence not in ('high', 'h'):",
                "                continue",
                "            if min_confidence == 'nominal' and confidence not in ('high', 'h', 'nominal', 'n'):",
                "                continue",
                "            lat = self._as_float(row.get('latitude'), 0.0)",
                "            lon = self._as_float(row.get('longitude'), 0.0)",
                "            if lat == 0.0 and lon == 0.0:",
                "                continue",
                "            frp = self._as_float(row.get('frp'), 0.0)",
                "            raw.append((frp, lat, lon, row))",
                "        raw.sort(key=lambda x: x[0], reverse=True)",
                "        seen = set()",
                "        out = []",
                "        for frp, lat, lon, row in raw:",
                '            grid_key = f"{lat:.2f},{lon:.2f}"',
                "            if grid_key in seen:",
                "                continue",
                "            seen.add(grid_key)",
                "            source = 'nasa_firms'",
                "            category = 'fire'",
                "            severity = self._clamp(frp / 100.0, 0.1, 1.0) if frp > 0 else 0.1",
                "            if not self._keep(source, category, severity):",
                "                continue",
                "            acq_date = str(row.get('acq_date') or '').strip()",
                "            acq_time = str(row.get('acq_time') or '').strip().zfill(4)",
                "            observed_at = None",
                "            if acq_date:",
                "                try:",
                "                    observed_at = datetime.strptime(f\"{acq_date} {acq_time}\", '%Y-%m-%d %H%M').replace(tzinfo=timezone.utc).isoformat()",
                "                except Exception:",
                "                    observed_at = datetime.now(timezone.utc).isoformat()",
                "            bright_ti4 = self._as_float(row.get('bright_ti4'), 0.0)",
                "            bright_ti5 = self._as_float(row.get('bright_ti5'), 0.0)",
                "            satellite = str(row.get('satellite') or '').strip()",
                "            daynight = str(row.get('daynight') or '').strip()",
                "            ext_id = hashlib.sha256(f\"{lat:.4f},{lon:.4f},{acq_date},{acq_time}\".encode('utf-8')).hexdigest()[:24]",
                "            out.append(",
                "                {",
                "                    'external_id': ext_id,",
                "                    'title': f\"VIIRS Fire: {lat:.2f}, {lon:.2f} (FRP={frp:.1f}MW)\",",
                "                    'summary': f\"confidence={row.get('confidence','')} satellite={satellite} {daynight}\",",
                "                    'category': category,",
                "                    'source': source,",
                "                    'observed_at': observed_at,",
                "                    'url': None,",
                "                    'payload': {",
                "                        'severity': round(severity, 3),",
                "                        'frp': frp,",
                "                        'bright_ti4': bright_ti4,",
                "                        'bright_ti5': bright_ti5,",
                "                        'confidence': str(row.get('confidence') or '').strip(),",
                "                        'satellite': satellite,",
                "                        'daynight': daynight,",
                "                        'scan': self._as_float(row.get('scan'), 0.0),",
                "                        'track': self._as_float(row.get('track'), 0.0),",
                "                    },",
                "                    'latitude': lat,",
                "                    'longitude': lon,",
                "                    'tags': ['events', 'fire', 'nasa_firms', 'viirs'],",
                "                }",
                "            )",
                "            if len(out) >= self._limit():",
                "                break",
                "        return out",
                "",
            ]
        )
    else:
        class_lines.extend(
            [
                "    async def fetch_async(self):",
                "        return []",
                "",
            ]
        )

    return "\n".join([*imports, *class_lines])


def _events_config_schema() -> dict:
    fields: list[dict] = [
        {"key": "hours", "label": "Lookback (hours)", "type": "integer", "min": 1, "max": 87600},
        {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
        {"key": "min_severity", "label": "Min Severity", "type": "number", "min": 0, "max": 1},
        {"key": "source_names", "label": "Source Names", "type": "list"},
        {"key": "signal_types", "label": "Signal Types", "type": "list"},
        {"key": "min_magnitude", "label": "USGS Min Magnitude", "type": "number", "min": 0, "max": 10},
        {"key": "pairs", "label": "Country Pairs", "type": "list"},
        {
            "key": "queries",
            "label": "Query Rows",
            "type": "json",
            "item_schema": {
                "name": "string",
                "query": "string",
                "priority": "string",
                "enabled": "boolean",
                "country_iso3": "string",
            },
        },
        {
            "key": "endpoints",
            "label": "API Endpoints",
            "type": "json",
            "item_schema": {
                "url": "string",
                "headers": "json",
                "json_path": "string",
                "source": "string",
                "activity_type": "string",
            },
        },
        {"key": "timespan_hours", "label": "Timespan (hours)", "type": "integer", "min": 1, "max": 168},
        {"key": "maxrecords_per_pair", "label": "Max Records / Pair", "type": "integer", "min": 10, "max": 250},
        {"key": "maxrecords_per_query", "label": "Max Records / Query", "type": "integer", "min": 10, "max": 250},
        {"key": "api_key", "label": "API Key", "type": "string"},
        {"key": "email", "label": "Email", "type": "string"},
        {"key": "endpoint", "label": "Endpoint", "type": "string"},
        {"key": "country", "label": "Country Filter", "type": "string"},
        {"key": "pagesize", "label": "Page Size (UCDP)", "type": "integer", "min": 50, "max": 1000},
        {"key": "max_pages", "label": "Max Pages (UCDP)", "type": "integer", "min": 1, "max": 20},
        {"key": "lookback_hours", "label": "Lookback Hours (Instability)", "type": "integer", "min": 12, "max": 720},
        {"key": "min_signals", "label": "Min Signals (Instability)", "type": "integer", "min": 1, "max": 50},
        {"key": "conflict_weight", "label": "Conflict Weight", "type": "number", "min": 0, "max": 1},
        {"key": "tension_weight", "label": "Tension Weight", "type": "number", "min": 0, "max": 1},
        {"key": "military_weight", "label": "Military Weight", "type": "number", "min": 0, "max": 1},
        {"key": "map_color", "label": "Map Color (hex)", "type": "string"},
        {"key": "map_layer_label", "label": "Map Layer Label", "type": "string"},
        {"key": "map_layer_short", "label": "Map Layer Short (2-3 chars)", "type": "string"},
        {"key": "map_radius_min", "label": "Map Dot Min Radius", "type": "number", "min": 1, "max": 30},
        {"key": "map_radius_max", "label": "Map Dot Max Radius", "type": "number", "min": 2, "max": 50},
        {"key": "map_opacity", "label": "Map Dot Opacity", "type": "number", "min": 0, "max": 1},
        {"key": "map_blur", "label": "Map Dot Blur", "type": "number", "min": 0, "max": 2},
        {"key": "map_stroke_width", "label": "Map Stroke Width", "type": "number", "min": 0, "max": 5},
        {"key": "map_glow", "label": "Map Glow Effect", "type": "boolean"},
        {"key": "map_dedicated_toggle", "label": "Dedicated Layer Toggle", "type": "boolean"},
    ]
    return {"param_fields": fields}


def _build_events_seeds() -> list[SystemDataSourceSeed]:
    seeds: list[SystemDataSourceSeed] = []

    for index, row in enumerate(_EVENT_SOURCE_DEFS):
        source_slug = str(row["slug"])
        source_id = _events_source_id_from_slug(source_slug)
        default_config = _events_default_config(
            hours=int(row.get("hours") or 168),
            limit=1500,
            min_severity=0.0,
            source_names=list(row.get("source_names") or []),
            signal_types=list(row.get("signal_types") or []),
            min_magnitude=4.5,
            map_color=str(row.get("map_color") or ""),
            map_layer_label=str(row.get("map_layer_label") or ""),
            map_layer_short=str(row.get("map_layer_short") or ""),
            map_radius_min=float(row.get("map_radius_min") or 4),
            map_radius_max=float(row.get("map_radius_max") or 9),
            map_opacity=float(row.get("map_opacity") or 0.92),
            map_blur=float(row.get("map_blur") or 0.1),
            map_stroke_width=float(row.get("map_stroke_width") or 1.0),
            map_glow=bool(row.get("map_glow", True)),
            map_dedicated_toggle=bool(row.get("map_dedicated_toggle", False)),
        )
        if row.get("default_endpoints"):
            default_config["endpoints"] = list(row["default_endpoints"])
        seeds.append(
            SystemDataSourceSeed(
                slug=source_slug,
                source_key="events",
                source_kind="python",
                name=str(row["name"]),
                description=str(row["description"]),
                source_code=_events_wrapper_source_code(
                    source_id=source_id,
                    source_slug=source_slug,
                    class_name=_events_source_class_name(source_slug),
                    source_name=str(row["name"]),
                    source_description=str(row["description"]),
                    default_config=default_config,
                ),
                sort_order=100 + index,
                config=default_config,
                config_schema=_events_config_schema(),
                retention=default_data_source_retention_policy(
                    slug=source_slug,
                    source_key="events",
                    source_kind="python",
                ),
                enabled=bool(row.get("enabled", True)),
            )
        )

    return seeds


def _build_story_seeds() -> list[SystemDataSourceSeed]:
    seeds: list[SystemDataSourceSeed] = []
    sort_order = 400

    for topic in _GOOGLE_NEWS_TOPICS:
        topic_text = str(topic).strip()
        if not topic_text:
            continue
        slug = _slug_with_hash("stories_google", topic_text)
        seeds.append(
            _make_story_rss_seed(
                slug=slug,
                name=f"Stories: Google - {topic_text.title()}",
                description=f"Google News RSS feed for '{topic_text}'",
                url=_google_rss_url(topic_text),
                category=topic_text,
                feed_source="google_news",
                sort_order=sort_order,
            )
        )
        sort_order += 1

    for query in _GDELT_QUERIES:
        query_text = str(query).strip()
        if not query_text:
            continue
        slug = _slug_with_hash("stories_gdelt", query_text)
        seeds.append(
            _make_story_rest_seed(
                slug=slug,
                name=f"Stories: GDELT - {query_text.title()}",
                description=f"GDELT DOC API query source for '{query_text}'",
                url=_gdelt_doc_url(query_text),
                category=query_text,
                feed_source="gdelt",
                sort_order=sort_order,
            )
        )
        sort_order += 1

    return seeds


BASE_SYSTEM_DATA_SOURCE_SEEDS: list[SystemDataSourceSeed] = [
    *_build_events_seeds(),
    *_build_story_seeds(),
]


def _is_seed_source_code(source_code: str) -> bool:
    return _SEED_MARKER in str(source_code or "")


def _is_legacy_system_source_row(row: DataSource) -> bool:
    source_key = str(row.source_key or "").strip().lower()
    source_kind = str(row.source_kind or "").strip().lower()
    return source_key in _LEGACY_SYSTEM_SOURCE_KEYS or source_kind in _LEGACY_SYSTEM_SOURCE_KINDS


def _is_legacy_imported_story_source_row(row: DataSource) -> bool:
    source_key = str(row.source_key or "").strip().lower()
    if source_key != "stories":
        return False
    slug = str(row.slug or "").strip().lower()
    if not slug.startswith("stories_custom_") and not slug.startswith("stories_gov_"):
        return False
    description = str(row.description or "").strip().lower()
    return "from legacy app settings" in description


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
        "retention": dict(seed.retention or {}),
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

    all_rows = (await session.execute(select(DataSource))).scalars().all()
    existing_rows = [row for row in all_rows if bool(row.is_system)]
    existing = {str(row.slug or "").strip().lower(): row for row in existing_rows if str(row.slug or "").strip()}

    inserted = 0
    updated = 0
    deleted = 0

    for row in all_rows:
        slug = str(row.slug or "").strip().lower()
        if not slug or slug in seed_by_slug:
            continue
        if _is_legacy_imported_story_source_row(row):
            await session.delete(row)
            deleted += 1
            continue
        if bool(row.is_system) and (
            _is_seed_source_code(str(row.source_code or "")) or _is_legacy_system_source_row(row)
        ):
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
        current.class_name = None
        current.enabled = bool(row["enabled"])
        current.retention = dict(row.get("retention") or {})
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
                "retention": dict(seed.retention or {}),
                "config": dict(seed.config or {}),
                "config_schema": dict(seed.config_schema or {}),
                "is_system_seed": True,
            }
        )
    return presets
