"""Shared RSS feed configuration helpers for the news domain."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urlparse

from config import settings

_PRIORITY_VALUES = {"critical", "high", "medium", "low"}
_DEFAULT_GOV_RSS_FEEDS: tuple[dict[str, str], ...] = (
    {
        "agency": "white_house",
        "url": "https://www.whitehouse.gov/feed/",
        "name": "White House Blog",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "white_house",
        "url": "https://www.whitehouse.gov/briefing-room/statements-releases/feed/",
        "name": "WH Statements",
        "priority": "critical",
        "country_iso3": "USA",
    },
    {
        "agency": "white_house",
        "url": "https://www.whitehouse.gov/briefing-room/presidential-actions/feed/",
        "name": "Presidential Actions",
        "priority": "critical",
        "country_iso3": "USA",
    },
    {
        "agency": "state_department",
        "url": "https://www.state.gov/rss-feed/press-releases/feed/",
        "name": "State Dept Press",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "state_department",
        "url": "https://www.state.gov/rss-feed/travel-advisories/feed/",
        "name": "Travel Advisories",
        "priority": "medium",
        "country_iso3": "USA",
    },
    {
        "agency": "defense",
        "url": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945",
        "name": "DoD News",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "treasury",
        "url": "https://home.treasury.gov/system/files/136/treasury-rss.xml",
        "name": "Treasury",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "federal_reserve",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "name": "Fed Press Releases",
        "priority": "critical",
        "country_iso3": "USA",
    },
    {
        "agency": "federal_reserve",
        "url": "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "name": "Fed Monetary Policy",
        "priority": "critical",
        "country_iso3": "USA",
    },
    {
        "agency": "sec",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=20&search_text=&start=0&output=atom",
        "name": "SEC EDGAR Filings",
        "priority": "medium",
        "country_iso3": "USA",
    },
    {
        "agency": "sec",
        "url": "https://www.sec.gov/news/pressreleases.rss",
        "name": "SEC Press Releases",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "justice",
        "url": "https://www.justice.gov/feeds/opa/justice-news.xml",
        "name": "DOJ News",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "cdc",
        "url": "https://tools.cdc.gov/api/v2/resources/media/rss?topic=outbreaks",
        "name": "CDC Outbreaks",
        "priority": "high",
        "country_iso3": "USA",
    },
    {
        "agency": "faa",
        "url": "https://www.faa.gov/rss/all",
        "name": "FAA Notices",
        "priority": "medium",
        "country_iso3": "USA",
    },
)


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
    return normalize_gov_rss_feeds(list(_DEFAULT_GOV_RSS_FEEDS))
