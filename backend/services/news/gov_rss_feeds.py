"""Government RSS aggregation owned by the news domain."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from xml.etree import ElementTree

import httpx
from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal
from services.news.rss_config import (
    default_gov_rss_feeds,
    merge_gov_rss_feeds,
    normalize_gov_rss_feeds,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/2.0)"


@dataclass
class GovArticle:
    """Article from a government RSS feed."""

    article_id: str
    title: str
    url: str
    source: str
    agency: str
    priority: str
    country_iso3: str = "USA"
    published: Optional[datetime] = None
    summary: str = ""
    feed_source: str = "rss"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GovRSSFeedService:
    """Fetches and aggregates government RSS feeds."""

    def __init__(self) -> None:
        self._articles: dict[str, GovArticle] = {}
        # Track seen article ids per consumer so multiple pipelines can
        # independently consume the same live feed stream.
        self._seen_by_consumer: dict[str, set[str]] = {}
        self._last_fetch_at: Optional[datetime] = None
        self._last_errors: list[str] = []
        self._failed_feeds: int = 0
        self._configured_feeds_count: int = 0
        self._enabled: bool = True

    async def _load_configuration(self) -> tuple[bool, list[dict[str, Any]]]:
        """Load enabled flag + feed rows from DB-backed app settings."""
        default_enabled = True
        default_feeds = default_gov_rss_feeds()

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                row = result.scalar_one_or_none()
                if row is None:
                    row = AppSettings(
                        id="default",
                        news_gov_rss_enabled=default_enabled,
                        news_gov_rss_feeds_json=default_feeds,
                    )
                    session.add(row)
                    await session.commit()
                    await session.refresh(row)

                raw_enabled = getattr(row, "news_gov_rss_enabled", None)
                enabled = default_enabled if raw_enabled is None else bool(raw_enabled)
                raw_rows = getattr(row, "news_gov_rss_feeds_json", None)
                rows = normalize_gov_rss_feeds(raw_rows) if raw_rows else []
                merged_rows = merge_gov_rss_feeds(rows)
                if not merged_rows:
                    merged_rows = default_feeds
                if merged_rows != rows:
                    rows = merged_rows
                    row.news_gov_rss_feeds_json = rows
                    await session.commit()
        except Exception as exc:
            logger.debug("RSS config DB read failed, using defaults: %s", exc)
            self._enabled = default_enabled
            self._configured_feeds_count = len(default_feeds)
            return self._enabled, default_feeds

        self._enabled = enabled
        self._configured_feeds_count = len(rows)
        return enabled, rows

    async def fetch_all(self, consumer: str = "default") -> list[GovArticle]:
        """Fetch from all enabled government RSS feeds. Returns NEW articles only."""
        consumer_key = str(consumer or "default").strip().lower() or "default"
        seen = self._seen_by_consumer.setdefault(consumer_key, set())

        enabled, feed_rows = await self._load_configuration()
        if not enabled:
            self._last_errors = ["disabled"]
            self._failed_feeds = 0
            return []

        self._last_errors = []
        self._failed_feeds = 0
        if not feed_rows:
            self._last_errors.append("No RSS feeds configured")
            return []

        tasks = [self._fetch_single_feed(feed_info) for feed_info in feed_rows if bool(feed_info.get("enabled", True))]
        if not tasks:
            self._last_errors.append("All RSS feeds are disabled")
            return []

        new_articles: list[GovArticle] = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug("RSS fetch error: %s", result)
                self._failed_feeds += 1
                self._last_errors.append(str(result))
                continue
            if isinstance(result, list):
                for article in result:
                    if article.article_id not in self._articles:
                        self._articles[article.article_id] = article
                    if article.article_id not in seen:
                        seen.add(article.article_id)
                        new_articles.append(article)

        self._last_fetch_at = datetime.now(timezone.utc)
        self._prune_old_articles()

        if new_articles:
            logger.info(
                "RSS: %d new articles from %d agencies",
                len(new_articles),
                len({a.agency for a in new_articles}),
            )
        return new_articles

    async def _fetch_single_feed(self, feed_info: dict[str, Any]) -> list[GovArticle]:
        """Fetch articles from a single government RSS feed."""
        url = str(feed_info.get("url") or "").strip()
        if not url:
            return []

        name = str(feed_info.get("name") or url).strip()
        agency = str(feed_info.get("agency") or "government").strip().lower()
        priority = str(feed_info.get("priority") or "medium").strip().lower()
        country_iso3 = str(feed_info.get("country_iso3") or "USA").strip().upper()

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                    follow_redirects=True,
                )
                if resp.status_code != 200:
                    self._failed_feeds += 1
                    self._last_errors.append(f"{name}: HTTP {resp.status_code}")
                    return []

                root = ElementTree.fromstring(resp.text)
                articles: list[GovArticle] = []

                items = root.findall(".//item")
                if not items:
                    items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

                for item in items[:20]:
                    title = (
                        item.findtext("title", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
                    )
                    link = item.findtext("link", "").strip()
                    if not link:
                        atom_link = item.find("{http://www.w3.org/2005/Atom}link")
                        if atom_link is not None:
                            link = atom_link.get("href", "")
                    pub_date = (
                        item.findtext("pubDate", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}published", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}updated", "").strip()
                    )
                    description = (
                        item.findtext("description", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}summary", "").strip()
                    )

                    if not link or not title:
                        continue

                    article_id = hashlib.sha256(link.encode()).hexdigest()[:16]
                    articles.append(
                        GovArticle(
                            article_id=article_id,
                            title=title,
                            url=link,
                            source=name,
                            agency=agency,
                            priority=priority,
                            country_iso3=country_iso3,
                            published=_parse_date(pub_date),
                            summary=_strip_html(description)[:500] if description else "",
                        )
                    )

                return articles

        except Exception as exc:
            logger.debug("RSS fetch failed for '%s' (%s): %s", name, url, exc)
            self._failed_feeds += 1
            self._last_errors.append(f"{name}: {exc}")
            return []

    def get_health(self) -> dict[str, object]:
        return {
            "enabled": self._enabled,
            "configured_feeds": self._configured_feeds_count,
            "tracked_articles": len(self._articles),
            "consumer_count": len(self._seen_by_consumer),
            "failed_feeds": self._failed_feeds,
            "last_fetch_at": self._last_fetch_at.isoformat() if self._last_fetch_at else None,
            "last_error": self._last_errors[0] if self._last_errors else None,
            "recent_errors": self._last_errors[:5],
        }

    def get_articles(self, max_age_hours: int = 48) -> list[GovArticle]:
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        articles = [a for a in self._articles.values() if a.fetched_at.timestamp() > cutoff]
        articles.sort(key=lambda a: a.fetched_at.timestamp(), reverse=True)
        return articles

    def get_critical_articles(self, max_age_hours: int = 4) -> list[GovArticle]:
        articles = self.get_articles(max_age_hours=max_age_hours)
        return [a for a in articles if a.priority in ("critical", "high")]

    def _prune_old_articles(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - (72 * 3600)
        to_remove = [aid for aid, article in self._articles.items() if article.fetched_at.timestamp() < cutoff]
        for aid in to_remove:
            del self._articles[aid]
        valid_ids = set(self._articles.keys())
        stale_consumers: list[str] = []
        for consumer, seen in self._seen_by_consumer.items():
            seen.intersection_update(valid_ids)
            if not seen:
                stale_consumers.append(consumer)
        for consumer in stale_consumers:
            self._seen_by_consumer.pop(consumer, None)


def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


gov_rss_service = GovRSSFeedService()
# Rebranded alias. Keep legacy name for compatibility.
rss_service = gov_rss_service
