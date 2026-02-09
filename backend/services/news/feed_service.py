"""
Multi-source news feed ingestion service.

Aggregates articles from:
1. Google News RSS (free, broad, 5-30 min delay)
2. GDELT DOC 2.0 API (free, global events, 15-min updates)
3. Custom RSS feeds (user-configurable)

Articles are deduplicated, stored in a rolling window, and
made available for semantic matching against active markets.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree

import httpx

from config import settings

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 12  # seconds
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/2.0)"

# Topic feeds for broad coverage
_GOOGLE_NEWS_TOPICS = [
    "politics",
    "business",
    "technology",
    "science",
    "sports",
    "world",
    "cryptocurrency",
]


@dataclass
class NewsArticle:
    """A single news article from any source."""

    article_id: str  # SHA-256 of URL for dedup
    title: str
    url: str
    source: str
    published: Optional[datetime] = None
    summary: str = ""
    feed_source: str = ""  # "google_news", "gdelt", "custom_rss"
    category: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Set after embedding
    embedding: Optional[list[float]] = None


class NewsFeedService:
    """
    Aggregates news from multiple sources into a rolling article store.

    Usage:
        service = NewsFeedService()
        new_articles = await service.fetch_all()
        all_articles = service.get_articles()
    """

    def __init__(self) -> None:
        self._articles: dict[str, NewsArticle] = {}  # article_id -> article
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_all(self) -> list[NewsArticle]:
        """Fetch from all enabled sources. Returns only NEW articles."""
        new_articles: list[NewsArticle] = []

        tasks = [self._fetch_google_news_topics()]

        if settings.NEWS_GDELT_ENABLED:
            tasks.append(self._fetch_gdelt())

        if settings.NEWS_RSS_FEEDS:
            tasks.append(self._fetch_custom_rss_feeds())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("News fetch error: %s", result)
                continue
            if isinstance(result, list):
                for article in result:
                    if article.article_id not in self._articles:
                        self._articles[article.article_id] = article
                        new_articles.append(article)

        # Prune old articles
        self._prune_old_articles()

        logger.info(
            "News fetch complete: %d new, %d total in store",
            len(new_articles),
            len(self._articles),
        )
        return new_articles

    def get_articles(self, max_age_hours: Optional[int] = None) -> list[NewsArticle]:
        """Get all articles in the store, optionally filtered by age."""
        articles = list(self._articles.values())
        if max_age_hours:
            cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
            articles = [a for a in articles if a.fetched_at.timestamp() > cutoff]
        return articles

    def get_unembedded_articles(self) -> list[NewsArticle]:
        """Get articles that haven't been embedded yet."""
        return [a for a in self._articles.values() if a.embedding is None]

    def clear(self) -> int:
        """Clear all articles from the store."""
        count = len(self._articles)
        self._articles.clear()
        return count

    @property
    def article_count(self) -> int:
        return len(self._articles)

    # ------------------------------------------------------------------
    # Background scanning
    # ------------------------------------------------------------------

    async def start(self, interval_seconds: Optional[int] = None) -> None:
        """Start continuous background news fetching."""
        if self._running:
            return
        self._running = True
        interval = interval_seconds or settings.NEWS_SCAN_INTERVAL_SECONDS
        self._task = asyncio.create_task(self._scan_loop(interval))
        logger.info("News feed service started (interval=%ds)", interval)

    def stop(self) -> None:
        """Stop background fetching."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("News feed service stopped")

    async def _scan_loop(self, interval: int) -> None:
        while self._running:
            try:
                await self.fetch_all()
            except Exception as e:
                logger.error("News scan loop error: %s", e)
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Google News RSS
    # ------------------------------------------------------------------

    async def _fetch_google_news_topics(self) -> list[NewsArticle]:
        """Fetch from Google News RSS for multiple topic feeds."""
        all_articles: list[NewsArticle] = []
        tasks = []
        for topic in _GOOGLE_NEWS_TOPICS:
            tasks.append(self._fetch_google_news_rss(topic))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        return all_articles

    async def _fetch_google_news_rss(
        self, query: str, max_results: int = 15
    ) -> list[NewsArticle]:
        """Fetch articles from Google News RSS."""
        try:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            )
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code != 200:
                    return []

                root = ElementTree.fromstring(resp.text)
                articles: list[NewsArticle] = []

                for item in root.findall(".//item")[:max_results]:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    source = item.findtext("source", "").strip()

                    if not link:
                        continue

                    # Extract source from title if not present
                    if " - " in title and not source:
                        parts = title.rsplit(" - ", 1)
                        if len(parts) == 2:
                            title = parts[0].strip()
                            source = parts[1].strip()

                    published = _parse_rss_date(pub_date)
                    article_id = _make_article_id(link)

                    articles.append(
                        NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=link,
                            source=source or "Google News",
                            published=published,
                            feed_source="google_news",
                            category=query,
                        )
                    )
                return articles

        except Exception as e:
            logger.debug("Google News RSS fetch failed for '%s': %s", query, e)
            return []

    # ------------------------------------------------------------------
    # GDELT DOC 2.0 API
    # ------------------------------------------------------------------

    async def _fetch_gdelt(self) -> list[NewsArticle]:
        """Fetch recent articles from GDELT DOC 2.0 API.

        GDELT monitors global news in 100+ languages and updates every 15 minutes.
        The DOC API returns articles matching a query with metadata.
        """
        all_articles: list[NewsArticle] = []
        queries = [
            "prediction market",
            "polymarket",
            "election odds",
            "geopolitical crisis",
            "policy announcement",
            "cryptocurrency regulation",
        ]
        tasks = [self._fetch_gdelt_query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        return all_articles

    async def _fetch_gdelt_query(
        self, query: str, max_results: int = 10
    ) -> list[NewsArticle]:
        """Fetch from GDELT DOC 2.0 API for a single query."""
        try:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query={encoded}&mode=artlist&maxrecords={max_results}"
                f"&format=json&sort=datedesc&timespan=1h"
            )
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code != 200:
                    return []

                data = resp.json()
                raw_articles = data.get("articles", [])
                articles: list[NewsArticle] = []

                for raw in raw_articles[:max_results]:
                    art_url = raw.get("url", "")
                    title = raw.get("title", "")
                    source = raw.get("domain", raw.get("source", ""))
                    date_str = raw.get("seendate", "")

                    if not art_url or not title:
                        continue

                    published = _parse_gdelt_date(date_str)
                    article_id = _make_article_id(art_url)

                    articles.append(
                        NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=art_url,
                            source=source,
                            published=published,
                            summary=raw.get("socialimage", ""),
                            feed_source="gdelt",
                            category=query,
                        )
                    )
                return articles

        except Exception as e:
            logger.debug("GDELT fetch failed for '%s': %s", query, e)
            return []

    # ------------------------------------------------------------------
    # Custom RSS feeds
    # ------------------------------------------------------------------

    async def _fetch_custom_rss_feeds(self) -> list[NewsArticle]:
        """Fetch from user-configured RSS feed URLs."""
        all_articles: list[NewsArticle] = []
        tasks = [
            self._fetch_single_rss(feed_url) for feed_url in settings.NEWS_RSS_FEEDS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        return all_articles

    async def _fetch_single_rss(
        self, feed_url: str, max_results: int = 20
    ) -> list[NewsArticle]:
        """Fetch from a single RSS feed URL."""
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(feed_url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code != 200:
                    return []

                root = ElementTree.fromstring(resp.text)
                articles: list[NewsArticle] = []

                # Try both RSS and Atom formats
                items = root.findall(".//item") or root.findall(
                    ".//{http://www.w3.org/2005/Atom}entry"
                )

                for item in items[:max_results]:
                    # RSS format
                    title = (
                        item.findtext("title", "").strip()
                        or item.findtext(
                            "{http://www.w3.org/2005/Atom}title", ""
                        ).strip()
                    )
                    link = item.findtext("link", "").strip()
                    if not link:
                        atom_link = item.find("{http://www.w3.org/2005/Atom}link")
                        if atom_link is not None:
                            link = atom_link.get("href", "")
                    pub_date = (
                        item.findtext("pubDate", "").strip()
                        or item.findtext(
                            "{http://www.w3.org/2005/Atom}published", ""
                        ).strip()
                    )
                    description = (
                        item.findtext("description", "").strip()
                        or item.findtext(
                            "{http://www.w3.org/2005/Atom}summary", ""
                        ).strip()
                    )

                    if not link or not title:
                        continue

                    article_id = _make_article_id(link)
                    published = _parse_rss_date(pub_date)

                    articles.append(
                        NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=link,
                            source=feed_url,
                            published=published,
                            summary=_strip_html(description)[:500],
                            feed_source="custom_rss",
                        )
                    )
                return articles

        except Exception as e:
            logger.debug("Custom RSS fetch failed for '%s': %s", feed_url, e)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_articles(self) -> None:
        """Remove articles older than the configured TTL."""
        ttl_seconds = settings.NEWS_ARTICLE_TTL_HOURS * 3600
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
        to_remove = [
            aid
            for aid, article in self._articles.items()
            if article.fetched_at.timestamp() < cutoff
        ]
        for aid in to_remove:
            del self._articles[aid]
        if to_remove:
            logger.debug("Pruned %d old articles", len(to_remove))


# ======================================================================
# Module-level helpers
# ======================================================================


def _make_article_id(url: str) -> str:
    """Create a deterministic article ID from URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse RFC 2822 date from RSS feeds."""
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_gdelt_date(date_str: str) -> Optional[datetime]:
    """Parse GDELT date format (YYYYMMDDTHHMMSSz or similar)."""
    if not date_str:
        return None
    try:
        # GDELT uses YYYYMMDDTHHMMSSZ format
        cleaned = date_str.replace("Z", "").replace("z", "")
        return datetime.strptime(cleaned, "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        pass
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


# ======================================================================
# Singleton
# ======================================================================

news_feed_service = NewsFeedService()
