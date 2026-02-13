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
import html
import hashlib
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from xml.etree import ElementTree

import httpx
from sqlalchemy import select

from config import settings
from models.database import AppSettings, AsyncSessionLocal
from services.news.rss_config import (
    default_custom_rss_feeds,
    normalize_custom_rss_feeds,
)

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
        self._ingest_stats: dict[str, int] = {
            "articles_dropped_low_text_quality": 0,
            "gdelt_summary_url_filtered": 0,
            "google_summary_parsed": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_all(self) -> list[NewsArticle]:
        """Fetch from all enabled sources. Returns only NEW articles."""
        self._reset_ingest_stats()
        new_articles: list[NewsArticle] = []
        rss_config = await self._load_rss_configuration()
        custom_rss_feeds = list(rss_config.get("custom_rss_feeds") or [])
        rss_enabled = bool(rss_config.get("rss_enabled", True))

        tasks = [self._fetch_google_news_topics()]

        if settings.NEWS_GDELT_ENABLED:
            tasks.append(self._fetch_gdelt())

        if custom_rss_feeds:
            tasks.append(self._fetch_custom_rss_feeds(custom_rss_feeds))

        if rss_enabled:
            tasks.append(self._fetch_rss())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("News fetch error: %s", result)
                continue
            if isinstance(result, list):
                for article in result:
                    if not _has_min_text_quality(article.title, article.summary):
                        self._inc_ingest_stat("articles_dropped_low_text_quality")
                        continue
                    if article.article_id not in self._articles:
                        self._articles[article.article_id] = article
                        new_articles.append(article)

        # Prune old articles
        self._prune_old_articles()

        logger.info(
            (
                "News fetch complete: %d new, %d total in store "
                "(articles_dropped_low_text_quality=%d, gdelt_summary_url_filtered=%d, "
                "google_summary_parsed=%d)"
            ),
            len(new_articles),
            len(self._articles),
            self._ingest_stats["articles_dropped_low_text_quality"],
            self._ingest_stats["gdelt_summary_url_filtered"],
            self._ingest_stats["google_summary_parsed"],
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

    async def fetch_for_topics(self, topics: list[str]) -> list[NewsArticle]:
        """On-demand fetch for specific topics (e.g. triggered by scanner).

        Useful for reactive news fetching when the scanner detects new markets
        or price movements.  Only hits Google News RSS (fastest free source).
        """
        if not topics:
            return []
        new_articles: list[NewsArticle] = []
        tasks = [self._fetch_google_news_rss(t, max_results=10) for t in topics[:5]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                for article in result:
                    if article.article_id not in self._articles:
                        self._articles[article.article_id] = article
                        new_articles.append(article)
        if new_articles:
            logger.info(
                "Reactive news fetch: %d new articles for %d topics",
                len(new_articles),
                len(topics),
            )
        return new_articles

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
                new_articles = await self.fetch_all()
                # Persist to DB + prune expired rows periodically
                if new_articles:
                    await self.persist_to_db()
                    await self.prune_db()
                    # Push new article count to frontend via WS
                    try:
                        from api.websocket import broadcast_news_update

                        await broadcast_news_update(len(new_articles))
                    except Exception:
                        pass  # WS not available yet during startup
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

    async def _fetch_google_news_rss(self, query: str, max_results: int = 40) -> list[NewsArticle]:
        """Fetch articles from Google News RSS."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
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
                    description = item.findtext("description", "").strip()

                    if not link:
                        continue

                    # Extract source from title if not present
                    if " - " in title and not source:
                        parts = title.rsplit(" - ", 1)
                        if len(parts) == 2:
                            title = parts[0].strip()
                            source = parts[1].strip()
                    title = _strip_title_source_suffix(title, source)

                    published = _parse_rss_date(pub_date)
                    article_id = _make_article_id(link)
                    summary = _clean_summary_text(description, max_len=500)
                    if summary:
                        self._inc_ingest_stat("google_summary_parsed")

                    articles.append(
                        NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=link,
                            source=source or "Google News",
                            published=published,
                            summary=summary,
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

    async def _fetch_gdelt_query(self, query: str, max_results: int = 30) -> list[NewsArticle]:
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
                    summary = self._pick_gdelt_summary(raw)

                    articles.append(
                        NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=art_url,
                            source=source,
                            published=published,
                            summary=summary,
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

    async def _load_rss_configuration(self) -> dict[str, Any]:
        """Read RSS feed config from DB app settings."""
        default_custom = default_custom_rss_feeds()

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                row = result.scalar_one_or_none()
        except Exception as exc:
            logger.debug("RSS config DB read failed, using defaults: %s", exc)
            return {
                "custom_rss_feeds": default_custom,
                "rss_enabled": True,
            }

        if row is None:
            return {
                "custom_rss_feeds": default_custom,
                "rss_enabled": True,
            }

        raw_custom = getattr(row, "news_rss_feeds_json", None)
        custom_rows = normalize_custom_rss_feeds(raw_custom) if raw_custom else default_custom
        raw_gov_enabled = getattr(row, "news_gov_rss_enabled", None)
        rss_enabled = True if raw_gov_enabled is None else bool(raw_gov_enabled)
        return {
            "custom_rss_feeds": custom_rows,
            "rss_enabled": rss_enabled,
        }

    async def _fetch_custom_rss_feeds(self, feed_rows: list[dict[str, Any]]) -> list[NewsArticle]:
        """Fetch from user-configured RSS feed URLs."""
        all_articles: list[NewsArticle] = []
        tasks = [self._fetch_single_rss(feed_info) for feed_info in feed_rows if bool(feed_info.get("enabled", True))]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        return all_articles

    async def _fetch_single_rss(self, feed_info: dict[str, Any], max_results: int = 20) -> list[NewsArticle]:
        """Fetch from a single RSS feed URL."""
        feed_url = str(feed_info.get("url") or "").strip()
        if not feed_url:
            return []
        feed_name = str(feed_info.get("name") or "").strip() or feed_url
        feed_category = str(feed_info.get("category") or "").strip().lower()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(feed_url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code != 200:
                    return []

                root = ElementTree.fromstring(resp.text)
                articles: list[NewsArticle] = []

                # Try both RSS and Atom formats
                items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")

                for item in items[:max_results]:
                    # RSS format
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
                    )
                    description = (
                        item.findtext("description", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}summary", "").strip()
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
                            source=feed_name,
                            published=published,
                            summary=_strip_html(description)[:500],
                            feed_source="custom_rss",
                            category=feed_category,
                        )
                    )
                return articles

        except Exception as e:
            logger.debug("Custom RSS fetch failed for '%s': %s", feed_url, e)
            return []

    # ------------------------------------------------------------------
    # Government RSS feeds
    # ------------------------------------------------------------------

    async def _fetch_rss(self) -> list[NewsArticle]:
        """Fetch from official RSS feeds owned by the news domain."""
        try:
            from services.news.gov_rss_feeds import gov_rss_service

            gov_articles = await gov_rss_service.fetch_all(consumer="news")
            articles: list[NewsArticle] = []
            for ga in gov_articles:
                articles.append(
                    NewsArticle(
                        article_id=ga.article_id,
                        title=ga.title,
                        url=ga.url,
                        source=ga.source,
                        published=ga.published,
                        summary=ga.summary,
                        feed_source="rss",
                        category=ga.agency,
                    )
                )
            return articles
        except Exception as e:
            logger.debug("RSS integration failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    async def persist_to_db(self) -> int:
        """Persist in-memory articles to the database for long-term retention."""
        try:
            from models.database import AsyncSessionLocal, NewsArticleCache
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            articles = list(self._articles.values())
            if not articles:
                return 0

            persisted = 0
            async with AsyncSessionLocal() as session:
                for a in articles:
                    stmt = (
                        sqlite_insert(NewsArticleCache)
                        .values(
                            article_id=a.article_id,
                            url=a.url,
                            title=a.title,
                            source=a.source,
                            feed_source=a.feed_source,
                            category=a.category,
                            summary=a.summary or "",
                            published=a.published,
                            fetched_at=a.fetched_at,
                            embedding=a.embedding,
                        )
                        .on_conflict_do_update(
                            index_elements=["article_id"],
                            set_={
                                "embedding": a.embedding,
                            },
                        )
                    )
                    await session.execute(stmt)
                    persisted += 1
                await session.commit()

            logger.debug("Persisted %d articles to DB", persisted)
            return persisted
        except Exception as e:
            logger.warning("Failed to persist articles to DB: %s", e)
            return 0

    async def load_from_db(self) -> int:
        """Load articles from DB into in-memory store on startup."""
        try:
            from models.database import AsyncSessionLocal, NewsArticleCache
            from sqlalchemy import select

            cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.NEWS_ARTICLE_TTL_HOURS)
            loaded = 0
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(NewsArticleCache).where(NewsArticleCache.fetched_at >= cutoff))
                rows = result.scalars().all()
                for row in rows:
                    if row.article_id in self._articles:
                        continue
                    self._articles[row.article_id] = NewsArticle(
                        article_id=row.article_id,
                        title=row.title,
                        url=row.url,
                        source=row.source or "",
                        published=row.published,
                        summary=row.summary or "",
                        feed_source=row.feed_source or "",
                        category=row.category or "",
                        fetched_at=row.fetched_at,
                        embedding=row.embedding,
                    )
                    loaded += 1
            logger.info("Loaded %d articles from DB", loaded)
            return loaded
        except Exception as e:
            logger.warning("Failed to load articles from DB: %s", e)
            return 0

    async def prune_db(self) -> int:
        """Remove articles older than TTL from the database."""
        try:
            from models.database import AsyncSessionLocal, NewsArticleCache
            from sqlalchemy import delete

            cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.NEWS_ARTICLE_TTL_HOURS)
            async with AsyncSessionLocal() as session:
                result = await session.execute(delete(NewsArticleCache).where(NewsArticleCache.fetched_at < cutoff))
                await session.commit()
                count = result.rowcount
            if count:
                logger.info("Pruned %d expired articles from DB", count)
            return count
        except Exception as e:
            logger.warning("Failed to prune DB articles: %s", e)
            return 0

    def search_articles(self, query: str, max_age_hours: int = 168, limit: int = 50) -> list[NewsArticle]:
        """Search articles by keyword in title / summary / category."""
        q = query.lower().strip()
        if not q:
            return []
        articles = self.get_articles(max_age_hours=max_age_hours)
        matches = [
            a
            for a in articles
            if q in a.title.lower()
            or q in (a.summary or "").lower()
            or q in (a.category or "").lower()
            or q in (a.source or "").lower()
        ]
        matches.sort(key=lambda a: a.fetched_at.timestamp(), reverse=True)
        return matches[:limit]

    def _reset_ingest_stats(self) -> None:
        self._ingest_stats = {
            "articles_dropped_low_text_quality": 0,
            "gdelt_summary_url_filtered": 0,
            "google_summary_parsed": 0,
        }

    def _inc_ingest_stat(self, key: str, delta: int = 1) -> None:
        self._ingest_stats[key] = int(self._ingest_stats.get(key, 0)) + int(delta)

    def _pick_gdelt_summary(self, raw: dict) -> str:
        for key in ("snippet", "description", "excerpt", "summary", "content"):
            value = raw.get(key)
            cleaned = _clean_summary_text(value, max_len=500)
            if not cleaned:
                continue
            if _looks_like_url(cleaned):
                self._inc_ingest_stat("gdelt_summary_url_filtered")
                continue
            return cleaned
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_articles(self) -> None:
        """Remove articles older than the configured TTL."""
        ttl_seconds = settings.NEWS_ARTICLE_TTL_HOURS * 3600
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
        to_remove = [aid for aid, article in self._articles.items() if article.fetched_at.timestamp() < cutoff]
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
        return datetime.strptime(cleaned, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _looks_like_url(value: str) -> bool:
    if not value:
        return False
    return bool(re.match(r"^https?://", value.strip(), flags=re.IGNORECASE))


def _clean_summary_text(value: object, max_len: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    text = html.unescape(_strip_html(value))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text[:max_len]


def _strip_title_source_suffix(title: str, source: str) -> str:
    if not title or not source:
        return title
    source_norm = source.strip()
    if not source_norm:
        return title
    suffix = f" - {source_norm}"
    if title.endswith(suffix):
        return title[: -len(suffix)].strip()
    return title


def _has_min_text_quality(title: str, summary: str) -> bool:
    full = f"{title or ''} {summary or ''}".strip()
    alnum_chars = len(re.sub(r"[^A-Za-z0-9]", "", full))
    if alnum_chars < 80:
        return False
    if not (summary or "").strip() and len((title or "").strip()) < 35:
        return False
    return True


# ======================================================================
# Singleton
# ======================================================================

news_feed_service = NewsFeedService()
