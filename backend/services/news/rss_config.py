"""Shared RSS feed configuration helpers for the news domain."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from config import settings

logger = logging.getLogger(__name__)

_NEWS_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "news"
_DEFAULT_GOV_FEED_FILE = _NEWS_DATA_DIR / "gov_rss_feeds.json"
_PRIORITY_VALUES = {"critical", "high", "medium", "low"}


def _stable_id(prefix: str, *parts: str) -> str:
    packed = "|".join(parts)
    digest = hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _clean_http_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not (text.startswith("http://") or text.startswith("https://")):
        return ""
    return text


def _host_or_url_name(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc
    except Exception:
        pass
    return url


def normalize_custom_rss_feeds(rows: Any) -> list[dict[str, Any]]:
    """Return normalized custom RSS feeds."""
    raw_rows: list[Any]
    if isinstance(rows, list):
        raw_rows = rows
    else:
        raw_rows = []

    normalized: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for row in raw_rows:
        if isinstance(row, str):
            url = _clean_http_url(row)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            normalized.append(
                {
                    "id": _stable_id("custom", url),
                    "name": _host_or_url_name(url),
                    "url": url,
                    "enabled": True,
                    "category": "",
                }
            )
            continue

        if not isinstance(row, dict):
            continue

        url = _clean_http_url(row.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        name = str(row.get("name") or "").strip() or _host_or_url_name(url)
        category = str(row.get("category") or "").strip().lower()
        normalized.append(
            {
                "id": str(row.get("id") or _stable_id("custom", url)),
                "name": name,
                "url": url,
                "enabled": bool(row.get("enabled", True)),
                "category": category,
            }
        )

    return normalized


def default_custom_rss_feeds() -> list[dict[str, Any]]:
    """Derive custom RSS defaults from legacy env config."""
    return normalize_custom_rss_feeds(list(getattr(settings, "NEWS_RSS_FEEDS", []) or []))


def _gov_rows_from_default_file() -> list[dict[str, Any]]:
    if not _DEFAULT_GOV_FEED_FILE.exists():
        return []
    try:
        payload = json.loads(_DEFAULT_GOV_FEED_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed reading default gov RSS feed file %s: %s", _DEFAULT_GOV_FEED_FILE, exc)
        return []

    feeds = payload.get("feeds")
    if not isinstance(feeds, dict):
        return []

    flattened: list[dict[str, Any]] = []
    for agency, rows in feeds.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_copy = dict(row)
            row_copy["agency"] = str(row_copy.get("agency") or agency).strip().lower()
            flattened.append(row_copy)
    return flattened


def normalize_gov_rss_feeds(rows: Any) -> list[dict[str, Any]]:
    """Return normalized government RSS feed rows."""
    raw_rows: list[Any]
    if isinstance(rows, list):
        raw_rows = rows
    else:
        raw_rows = []

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        url = _clean_http_url(row.get("url"))
        if not url:
            continue

        agency = str(row.get("agency") or "government").strip().lower() or "government"
        dedupe_key = (agency, url)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        name = str(row.get("name") or "").strip() or _host_or_url_name(url)
        priority = str(row.get("priority") or "medium").strip().lower()
        if priority not in _PRIORITY_VALUES:
            priority = "medium"
        country_iso3 = str(row.get("country_iso3") or "USA").strip().upper()[:3]
        if len(country_iso3) != 3:
            country_iso3 = "USA"

        normalized.append(
            {
                "id": str(row.get("id") or _stable_id("gov", agency, url)),
                "agency": agency,
                "name": name,
                "url": url,
                "priority": priority,
                "country_iso3": country_iso3,
                "enabled": bool(row.get("enabled", True)),
            }
        )

    return normalized


def default_gov_rss_feeds() -> list[dict[str, Any]]:
    return normalize_gov_rss_feeds(_gov_rows_from_default_file())

