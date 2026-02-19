"""Move legacy news RSS settings into data_sources and normalize source kinds.

Revision ID: 202602190001
Revises: 202602180010
Create Date: 2026-02-19 10:00:00.000000
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import uuid
from urllib.parse import urlparse

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


# revision identifiers, used by Alembic.
revision = "202602190001"
down_revision = "202602180010"
branch_labels = None
depends_on = None


_RSS_CONFIG_SCHEMA = {
    "param_fields": [
        {"key": "url", "label": "Feed URL", "type": "string"},
        {"key": "source_name", "label": "Source Name", "type": "string"},
        {"key": "category_filter", "label": "Category Filter", "type": "string"},
        {"key": "category_override", "label": "Category Override", "type": "string"},
        {"key": "feed_source", "label": "Feed Source", "type": "string"},
        {"key": "priority", "label": "Priority", "type": "string"},
        {"key": "country_iso3", "label": "Country ISO3", "type": "string"},
        {"key": "limit", "label": "Max Records", "type": "integer", "min": 1, "max": 500},
    ]
}


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_json_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except Exception:
            return []
    return []


def _parse_json_dict(value: object) -> dict:
    parsed = _parse_json_value(value)
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def _parse_json_list(value: object) -> list:
    parsed = _parse_json_value(value)
    if isinstance(parsed, list):
        return list(parsed)
    return []


def _clean_http_url(value: object) -> str:
    url = str(value or "").strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return ""


def _host_or_fallback(url: str, fallback: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc
    except Exception:
        pass
    return fallback


def _slug_fragment(value: str, *, max_len: int = 18) -> str:
    out: list[str] = []
    prev_us = False
    for ch in str(value or "").strip().lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out.append(ch)
            prev_us = False
            continue
        if not prev_us:
            out.append("_")
            prev_us = True
    token = "".join(out).strip("_")
    if not token:
        token = "source"
    if token[0].isdigit():
        token = f"s_{token}"
    return token[:max_len].rstrip("_")


def _slug_for_feed(prefix: str, stable_value: str) -> str:
    token = _slug_fragment(stable_value, max_len=18)
    digest = hashlib.sha256(stable_value.encode("utf-8")).hexdigest()[:6]
    slug = f"{prefix}_{token}_{digest}"
    if len(slug) <= 50:
        return slug
    trimmed = slug[:50].rstrip("_")
    if trimmed and trimmed[-1].isalnum():
        return trimmed
    return f"{trimmed[:-1]}x"


def _next_unique_slug(base_slug: str, used_slugs: set[str]) -> str:
    if base_slug not in used_slugs:
        used_slugs.add(base_slug)
        return base_slug
    counter = 2
    while True:
        suffix = f"_{counter}"
        max_prefix = max(1, 50 - len(suffix))
        candidate = f"{base_slug[:max_prefix].rstrip('_')}{suffix}"
        if candidate not in used_slugs:
            used_slugs.add(candidate)
            return candidate
        counter += 1


def _normalize_custom_rows(rows: object) -> list[dict]:
    normalized: list[dict] = []
    seen_urls: set[str] = set()
    for raw in _parse_json_list(rows):
        if isinstance(raw, str):
            url = _clean_http_url(raw)
            if not url:
                continue
            lower_url = url.lower()
            if lower_url in seen_urls:
                continue
            seen_urls.add(lower_url)
            normalized.append(
                {
                    "url": url,
                    "name": _host_or_fallback(url, "Custom RSS"),
                    "enabled": True,
                    "category": "",
                }
            )
            continue

        if not isinstance(raw, dict):
            continue

        url = _clean_http_url(raw.get("url"))
        if not url:
            continue
        lower_url = url.lower()
        if lower_url in seen_urls:
            continue
        seen_urls.add(lower_url)

        name = str(raw.get("name") or "").strip() or _host_or_fallback(url, "Custom RSS")
        category = str(raw.get("category") or "").strip().lower()
        normalized.append(
            {
                "url": url,
                "name": name,
                "enabled": bool(raw.get("enabled", True)),
                "category": category,
            }
        )
    return normalized


def _normalize_gov_rows(rows: object, default_enabled: bool) -> list[dict]:
    normalized: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for raw in _parse_json_list(rows):
        if not isinstance(raw, dict):
            continue
        url = _clean_http_url(raw.get("url"))
        if not url:
            continue
        agency = str(raw.get("agency") or "government").strip().lower() or "government"
        dedupe_key = (agency, url.lower())
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        priority = str(raw.get("priority") or "medium").strip().lower()
        if priority not in {"critical", "high", "medium", "low"}:
            priority = "medium"
        country_iso3 = str(raw.get("country_iso3") or "USA").strip().upper()[:3]
        if len(country_iso3) != 3:
            country_iso3 = "USA"
        name = str(raw.get("name") or "").strip() or _host_or_fallback(url, "Government RSS")
        normalized.append(
            {
                "url": url,
                "name": name,
                "agency": agency,
                "priority": priority,
                "country_iso3": country_iso3,
                "enabled": bool(raw.get("enabled", default_enabled)),
            }
        )
    return normalized


def _infer_feed_source(slug: str, explicit: object, fallback: str) -> str:
    parsed = str(explicit or "").strip().lower()
    if parsed:
        return parsed
    lowered = str(slug or "").strip().lower()
    if "google" in lowered:
        return "google_news"
    if "gdelt" in lowered:
        return "gdelt"
    if "gov" in lowered or "government" in lowered:
        return "government"
    if "custom" in lowered:
        return "custom_rss"
    return fallback


def upgrade() -> None:
    existing_tables = table_names()
    if "data_sources" not in existing_tables:
        return

    data_source_cols = column_names("data_sources")
    required_data_source_cols = {"id", "slug", "source_key", "source_kind", "name", "config"}
    if not required_data_source_cols.issubset(data_source_cols):
        return

    bind = op.get_bind()
    now = _utcnow_naive()

    has_description_col = "description" in data_source_cols

    data_sources = sa.table(
        "data_sources",
        sa.column("id", sa.String()),
        sa.column("slug", sa.String()),
        sa.column("source_key", sa.String()),
        sa.column("source_kind", sa.String()),
        sa.column("name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("source_code", sa.Text()),
        sa.column("class_name", sa.String()),
        sa.column("is_system", sa.Boolean()),
        sa.column("enabled", sa.Boolean()),
        sa.column("status", sa.String()),
        sa.column("error_message", sa.Text()),
        sa.column("config", sa.JSON()),
        sa.column("config_schema", sa.JSON()),
        sa.column("version", sa.Integer()),
        sa.column("sort_order", sa.Integer()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )

    bind.execute(
        data_sources.update().where(sa.func.lower(data_sources.c.source_key) == "news").values(source_key="stories")
    )
    bind.execute(
        data_sources.update().where(sa.func.lower(data_sources.c.source_kind) == "bridge").values(source_kind="python")
    )
    bind.execute(
        data_sources.update().where(sa.func.lower(data_sources.c.source_kind) == "gdelt").values(source_kind="rest_api")
    )

    rows = (
        bind.execute(
            sa.select(
                data_sources.c.id,
                data_sources.c.slug,
                data_sources.c.source_key,
                data_sources.c.source_kind,
                data_sources.c.name,
                data_sources.c.config,
            )
        )
        .mappings()
        .all()
    )

    used_slugs: set[str] = set()
    existing_story_urls: set[str] = set()

    for row in rows:
        slug = str(row.get("slug") or "").strip().lower()
        if slug:
            used_slugs.add(slug)

        source_key = str(row.get("source_key") or "").strip().lower()
        source_kind = str(row.get("source_kind") or "").strip().lower()
        source_name = str(row.get("name") or "").strip()
        config = _parse_json_dict(row.get("config"))
        next_config = dict(config)
        changed = False

        if source_kind == "rss":
            if not str(next_config.get("url") or "").strip():
                legacy_url = str(next_config.get("feed_url") or "").strip()
                if legacy_url:
                    next_config["url"] = legacy_url
                    changed = True
            if "feed_url" in next_config:
                next_config.pop("feed_url", None)
                changed = True
            if not str(next_config.get("feed_source") or "").strip():
                next_config["feed_source"] = _infer_feed_source(slug, next_config.get("feed_source"), "rss")
                changed = True
            if not str(next_config.get("source_name") or "").strip() and source_name:
                next_config["source_name"] = source_name
                changed = True
            if not str(next_config.get("category_override") or "").strip():
                legacy_category = str(next_config.get("category") or "").strip().lower()
                if legacy_category:
                    next_config["category_override"] = legacy_category
                    changed = True
            if "category" in next_config:
                next_config.pop("category", None)
                changed = True

        if source_kind == "rest_api":
            if not str(next_config.get("url") or "").strip():
                legacy_url = str(next_config.get("api_url") or "").strip()
                if legacy_url:
                    next_config["url"] = legacy_url
                    changed = True
            if "api_url" in next_config:
                next_config.pop("api_url", None)
                changed = True
            if not str(next_config.get("feed_source") or "").strip():
                next_config["feed_source"] = _infer_feed_source(slug, next_config.get("feed_source"), "rest_api")
                changed = True
            if not str(next_config.get("source_name") or "").strip() and source_name:
                next_config["source_name"] = source_name
                changed = True

        url = _clean_http_url(next_config.get("url"))
        if source_key == "stories" and source_kind in {"rss", "rest_api"} and url:
            existing_story_urls.add(url.lower())

        if changed:
            bind.execute(
                data_sources.update()
                .where(data_sources.c.id == row.get("id"))
                .values(config=next_config, updated_at=now)
            )

    if "app_settings" not in existing_tables:
        return
    app_settings_cols = column_names("app_settings")
    needed_app_settings_cols = {"news_rss_feeds_json", "news_gov_rss_feeds_json", "news_gov_rss_enabled"}
    if not needed_app_settings_cols.issubset(app_settings_cols):
        return

    app_settings = sa.table(
        "app_settings",
        sa.column("id", sa.String()),
        sa.column("news_rss_feeds_json", sa.JSON()),
        sa.column("news_gov_rss_enabled", sa.Boolean()),
        sa.column("news_gov_rss_feeds_json", sa.JSON()),
    )

    legacy_rows = (
        bind.execute(
            sa.select(
                app_settings.c.id,
                app_settings.c.news_rss_feeds_json,
                app_settings.c.news_gov_rss_enabled,
                app_settings.c.news_gov_rss_feeds_json,
            )
        )
        .mappings()
        .all()
    )

    sort_order = 900

    for legacy in legacy_rows:
        custom_rows = _normalize_custom_rows(legacy.get("news_rss_feeds_json"))
        gov_rows = _normalize_gov_rows(
            legacy.get("news_gov_rss_feeds_json"),
            default_enabled=bool(legacy.get("news_gov_rss_enabled", True)),
        )

        for item in custom_rows:
            url = item["url"]
            url_key = url.lower()
            if url_key in existing_story_urls:
                continue

            stable_value = f"custom|{url}"
            slug = _next_unique_slug(_slug_for_feed("stories_custom", stable_value), used_slugs)
            name = str(item.get("name") or "").strip() or _host_or_fallback(url, "Custom RSS")
            category_override = str(item.get("category") or "").strip().lower()

            values = {
                "id": uuid.uuid4().hex,
                "slug": slug,
                "source_key": "stories",
                "source_kind": "rss",
                "name": f"Stories: Custom - {name}",
                "source_code": "",
                "class_name": None,
                "is_system": False,
                "enabled": bool(item.get("enabled", True)),
                "status": "loaded",
                "error_message": None,
                "config": {
                    "url": url,
                    "source_name": name,
                    "category_override": category_override,
                    "feed_source": "custom_rss",
                    "limit": 40,
                },
                "config_schema": _RSS_CONFIG_SCHEMA,
                "version": 1,
                "sort_order": sort_order,
                "created_at": now,
                "updated_at": now,
            }
            if has_description_col:
                values["description"] = f"Imported custom RSS source ({name}) from legacy app settings"
            bind.execute(data_sources.insert().values(**values))
            sort_order += 1
            existing_story_urls.add(url_key)

        for item in gov_rows:
            url = item["url"]
            url_key = url.lower()
            if url_key in existing_story_urls:
                continue

            stable_value = f"gov|{item.get('agency')}|{url}"
            slug = _next_unique_slug(_slug_for_feed("stories_gov", stable_value), used_slugs)
            agency = str(item.get("agency") or "government").strip().lower() or "government"
            name = str(item.get("name") or "").strip() or _host_or_fallback(url, "Government RSS")

            values = {
                "id": uuid.uuid4().hex,
                "slug": slug,
                "source_key": "stories",
                "source_kind": "rss",
                "name": f"Stories: Gov - {name}",
                "source_code": "",
                "class_name": None,
                "is_system": False,
                "enabled": bool(item.get("enabled", True)),
                "status": "loaded",
                "error_message": None,
                "config": {
                    "url": url,
                    "source_name": name,
                    "category_override": agency,
                    "feed_source": "government",
                    "priority": str(item.get("priority") or "medium").strip().lower(),
                    "country_iso3": str(item.get("country_iso3") or "USA").strip().upper()[:3],
                    "limit": 40,
                },
                "config_schema": _RSS_CONFIG_SCHEMA,
                "version": 1,
                "sort_order": sort_order,
                "created_at": now,
                "updated_at": now,
            }
            if has_description_col:
                values["description"] = f"Imported government RSS source ({agency}) from legacy app settings"
            bind.execute(data_sources.insert().values(**values))
            sort_order += 1
            existing_story_urls.add(url_key)


def downgrade() -> None:
    return
