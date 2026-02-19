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
) -> dict:
    out = {
        "hours": int(hours),
        "limit": int(limit),
        "min_severity": float(min_severity),
        "source_names": list(source_names or []),
        "signal_types": list(signal_types or []),
        "min_magnitude": float(min_magnitude),
    }
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
                "        if not endpoint or not api_key or not email:",
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
                "                params = {",
                "                    'query': f'\"{country_a}\" \"{country_b}\"',",
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
                "                            'trend': trend,",
                "                            'top_event_types': top_event_types,",
                "                            'query': params.get('query'),",
                "                        },",
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
                "    async def fetch_async(self):",
                "        endpoints = self.config.get('endpoints') or []",
                "        if not isinstance(endpoints, list):",
                "            endpoints = []",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
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
                "        endpoints = self.config.get('endpoints') or []",
                "        if not isinstance(endpoints, list):",
                "            endpoints = []",
                "        out = []",
                "        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:",
                "            for endpoint in endpoints:",
                "                if not isinstance(endpoint, dict):",
                "                    continue",
                "                url = str(endpoint.get('url') or '').strip()",
                "                if not url:",
                "                    continue",
                "                headers = endpoint.get('headers') if isinstance(endpoint.get('headers'), dict) else {}",
                "                json_path = str(endpoint.get('json_path') or '$[*]').strip()",
                "                source_name = str(endpoint.get('source') or 'infrastructure').strip().lower() or 'infrastructure'",
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
                "                for item in rows:",
                "                    event_type = str(item.get('event_type') or item.get('type') or 'infrastructure').strip().lower() or 'infrastructure'",
                "                    country = str(item.get('country_iso3') or item.get('country') or '').strip().upper()",
                "                    summary = str(item.get('description') or item.get('summary') or '').strip()",
                "                    severity = self._clamp(self._as_float(item.get('severity'), 0.45), 0.0, 1.0)",
                "                    source = source_name",
                "                    category = 'infrastructure'",
                "                    if not self._keep(source, category, severity):",
                "                        continue",
                "                    observed = self._parse_datetime(item.get('started_at') or item.get('detected_at') or item.get('timestamp') or item.get('time'))",
                "                    observed_at = observed.isoformat() if observed is not None else None",
                '                    key = f"{event_type}:{country}:{summary[:120]}:{observed_at}"',
                "                    out.append(",
                "                        {",
                "                            'external_id': str(item.get('id') or '').strip() or hashlib.sha256(key.encode('utf-8')).hexdigest()[:24],",
                "                            'title': f\"Infrastructure: {event_type} in {country or 'region'}\",",
                "                            'summary': summary[:500],",
                "                            'category': category,",
                "                            'source': source,",
                "                            'observed_at': observed_at,",
                "                            'payload': {",
                "                                'severity': severity,",
                "                                'event_type': event_type,",
                "                                'affected_services': list(item.get('affected_services') or []),",
                "                                'cascade_risk_score': self._as_float(item.get('cascade_risk_score'), 0.0),",
                "                                'raw': item,",
                "                            },",
                "                            'latitude': item.get('latitude') if item.get('latitude') is not None else item.get('lat'),",
                "                            'longitude': item.get('longitude') if item.get('longitude') is not None else item.get('lon'),",
                "                            'country_iso3': country if len(country) == 3 else None,",
                "                            'tags': ['events', 'infrastructure', event_type],",
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
        {"key": "hours", "label": "Lookback (hours)", "type": "integer", "min": 1, "max": 720},
        {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 5000},
        {"key": "min_severity", "label": "Min Severity", "type": "number", "min": 0, "max": 1},
        {"key": "source_names", "label": "Source Names", "type": "list"},
        {"key": "signal_types", "label": "Signal Types", "type": "list"},
        {"key": "min_magnitude", "label": "USGS Min Magnitude", "type": "number", "min": 0, "max": 10},
        {"key": "pairs", "label": "Country Pairs", "type": "list"},
        {"key": "queries", "label": "Query Rows", "type": "json"},
        {"key": "endpoints", "label": "Endpoint Rows", "type": "json"},
        {"key": "timespan_hours", "label": "Timespan (hours)", "type": "integer", "min": 1, "max": 168},
        {"key": "maxrecords_per_pair", "label": "Max Records / Pair", "type": "integer", "min": 10, "max": 250},
        {"key": "maxrecords_per_query", "label": "Max Records / Query", "type": "integer", "min": 10, "max": 250},
        {"key": "api_key", "label": "API Key", "type": "string"},
        {"key": "email", "label": "Email", "type": "string"},
        {"key": "endpoint", "label": "Endpoint", "type": "string"},
        {"key": "country", "label": "Country Filter", "type": "string"},
    ]
    return {"param_fields": fields}


def _build_events_seeds() -> list[SystemDataSourceSeed]:
    seeds: list[SystemDataSourceSeed] = []

    for index, row in enumerate(_EVENT_SOURCE_DEFS):
        source_slug = str(row["slug"])
        source_id = _events_source_id_from_slug(source_slug)
        default_config = _events_default_config(
            hours=168,
            limit=1500,
            min_severity=0.0,
            source_names=list(row.get("source_names") or []),
            signal_types=list(row.get("signal_types") or []),
            min_magnitude=4.5,
        )
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
                enabled=True,
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
