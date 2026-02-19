"""
News feed ingestion service.

All story inputs are sourced from DB-managed data sources:
- data_sources.source_key = "stories"
- source_kind in {"rss", "rest_api", "python"}

No feed URLs or topic/query providers are configured outside the Data Sources registry.
"""

from __future__ import annotations

import asyncio
import html
import hashlib
import logging
import inspect
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import desc, select
from services.shared_state import _commit_with_retry

from config import settings
from models.database import AsyncSessionLocal, DataSource, DataSourceRecord
from services.data_source_runner import run_data_source

logger = logging.getLogger(__name__)

_MAX_SOURCE_FETCH_CONCURRENCY = 12


@dataclass
class NewsArticle:
    """A single news article from any source."""

    article_id: str
    title: str
    url: str
    source: str
    published: Optional[datetime] = None
    summary: str = ""
    feed_source: str = ""
    category: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    embedding: Optional[list[float]] = None


class NewsFeedService:
    """Aggregates stories from enabled DB-defined story sources."""

    def __init__(self) -> None:
        self._articles: dict[str, NewsArticle] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ingest_stats: dict[str, int] = {
            "articles_dropped_low_text_quality": 0,
            "gdelt_summary_url_filtered": 0,
            "source_fetch_failures": 0,
            "source_rows_skipped": 0,
        }

    async def fetch_all(self) -> list[NewsArticle]:
        """Fetch from all enabled story sources. Returns only NEW articles."""
        self._reset_ingest_stats()
        new_articles: list[NewsArticle] = []

        try:
            sources = await self._load_enabled_story_sources()
        except Exception as exc:
            logger.warning("Failed loading story sources: %s", exc)
            return []

        if not sources:
            logger.debug("No enabled story data sources configured")
            return []

        fetched_articles = await self._fetch_articles_from_sources(sources)
        for article in fetched_articles:
            if not _has_min_text_quality(article.title, article.summary):
                self._inc_ingest_stat("articles_dropped_low_text_quality")
                continue

            existing = self._articles.get(article.article_id)
            if existing is None:
                self._articles[article.article_id] = article
                new_articles.append(article)
            else:
                # Keep freshest metadata if we already know the article.
                if article.fetched_at >= existing.fetched_at:
                    self._articles[article.article_id] = article

        self._prune_old_articles()

        logger.info(
            (
                "News fetch complete: %d new, %d total, %d sources "
                "(dropped_low_quality=%d, source_fetch_failures=%d, source_rows_skipped=%d)"
            ),
            len(new_articles),
            len(self._articles),
            len(sources),
            self._ingest_stats["articles_dropped_low_text_quality"],
            self._ingest_stats["source_fetch_failures"],
            self._ingest_stats["source_rows_skipped"],
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
        return [a for a in self._articles.values() if a.embedding is None]

    def clear(self) -> int:
        count = len(self._articles)
        self._articles.clear()
        return count

    @property
    def article_count(self) -> int:
        return len(self._articles)

    async def fetch_for_topics(self, topics: list[str]) -> list[NewsArticle]:
        """Topic-targeted fetch from configured story sources.

        Topics are matched against source category overrides, names, descriptions,
        and slugs; only matching sources are polled.
        """
        normalized_topics = {
            str(topic or "").strip().lower()
            for topic in topics
            if str(topic or "").strip()
        }
        if not normalized_topics:
            return []

        sources = await self._load_enabled_story_sources()
        targeted: list[DataSource] = []
        for source in sources:
            config = dict(source.config or {})
            haystack_parts = [
                str(source.slug or ""),
                str(source.name or ""),
                str(source.description or ""),
                str(config.get("category_override") or ""),
                str(config.get("category_filter") or ""),
            ]
            haystack = " ".join(haystack_parts).strip().lower()
            if any(topic in haystack for topic in normalized_topics):
                targeted.append(source)

        if not targeted:
            return []

        fetched_articles = await self._fetch_articles_from_sources(targeted)
        new_articles: list[NewsArticle] = []
        for article in fetched_articles:
            if article.article_id not in self._articles:
                self._articles[article.article_id] = article
                new_articles.append(article)

        if new_articles:
            logger.info(
                "Reactive news fetch: %d new articles from %d targeted sources",
                len(new_articles),
                len(targeted),
            )
        return new_articles

    async def start(self, interval_seconds: Optional[int] = None) -> None:
        if self._running:
            return
        self._running = True
        interval = interval_seconds or settings.NEWS_SCAN_INTERVAL_SECONDS
        self._task = asyncio.create_task(self._scan_loop(interval))
        logger.info("News feed service started (interval=%ds)", interval)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("News feed service stopped")

    async def _scan_loop(self, interval: int) -> None:
        while self._running:
            try:
                new_articles = await self.fetch_all()
                if new_articles:
                    await self.persist_to_db()
                    await self.prune_db()
                    try:
                        from api.websocket import broadcast_news_update

                        await broadcast_news_update(len(new_articles))
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("News scan loop error: %s", exc)
            await asyncio.sleep(interval)

    async def _load_enabled_story_sources(self) -> list[DataSource]:
        async with AsyncSessionLocal() as session:
            query = (
                select(DataSource)
                .where(DataSource.source_key == "stories")
                .where(DataSource.enabled == True)  # noqa: E712
                .order_by(DataSource.sort_order.asc(), DataSource.slug.asc())
            )
            return list((await session.execute(query)).scalars().all())

    async def _fetch_articles_from_sources(self, sources: list[DataSource]) -> list[NewsArticle]:
        sem = asyncio.Semaphore(_MAX_SOURCE_FETCH_CONCURRENCY)

        async def _run(source: DataSource) -> list[NewsArticle]:
            async with sem:
                try:
                    rows = await self._fetch_source_rows(source)
                except Exception as exc:
                    self._inc_ingest_stat("source_fetch_failures")
                    logger.debug("Story source fetch failed for '%s': %s", source.slug, exc)
                    return []

                out: list[NewsArticle] = []
                for row in rows:
                    article = self._row_to_article(source, row)
                    if article is None:
                        self._inc_ingest_stat("source_rows_skipped")
                        continue
                    out.append(article)
                return out

        results = await asyncio.gather(*[_run(source) for source in sources], return_exceptions=True)
        flattened: list[NewsArticle] = []
        for result in results:
            if isinstance(result, Exception):
                self._inc_ingest_stat("source_fetch_failures")
                continue
            flattened.extend(result)
        return flattened

    async def _fetch_source_rows(self, source: DataSource) -> list[dict[str, Any]]:
        config = dict(source.config or {})
        limit = int(max(1, min(5000, _as_float(config.get("limit"), 200.0))))
        slug = str(source.slug or "").strip().lower()
        if not slug:
            return []

        run_started_at = datetime.now(timezone.utc)

        async with AsyncSessionLocal() as session:
            row = await session.get(DataSource, source.id)
            if row is None or not bool(row.enabled):
                return []

            run_result = await run_data_source(session, row, max_records=limit, commit=True)
            if str(run_result.get("status") or "").strip().lower() != "success":
                return []

            records = (
                (
                    await session.execute(
                        select(DataSourceRecord)
                        .where(DataSourceRecord.source_slug == slug)
                        .where(DataSourceRecord.ingested_at >= run_started_at)
                        .order_by(desc(DataSourceRecord.observed_at), desc(DataSourceRecord.ingested_at))
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

        out: list[dict[str, Any]] = []
        for record in records:
            payload = dict(record.payload_json or {}) if isinstance(record.payload_json, dict) else {}
            transformed = dict(record.transformed_json or {}) if isinstance(record.transformed_json, dict) else {}
            merged: dict[str, Any] = {}
            merged.update(payload)
            merged.update(transformed)
            out.append(
                {
                    "external_id": record.external_id,
                    "title": record.title or merged.get("title"),
                    "summary": record.summary or merged.get("summary"),
                    "category": record.category or merged.get("category"),
                    "source": record.source or merged.get("source"),
                    "url": record.url or merged.get("url"),
                    "observed_at": record.observed_at or record.ingested_at,
                    "payload": payload,
                    "feed_source": merged.get("feed_source"),
                    "tags": list(record.tags_json or []),
                }
            )
        return out

    def _row_to_article(self, source: DataSource, row: dict[str, Any]) -> NewsArticle | None:
        config = dict(source.config or {})
        url = _as_text(row.get("url"))
        title = _as_text(row.get("title"))
        if not url or not title:
            return None

        external_id = _as_text(row.get("external_id"))
        if not external_id:
            external_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

        source_name = (
            _as_text(row.get("source"))
            or _as_text(config.get("source_name"))
            or _as_text(source.name)
            or "news"
        )
        category = (
            _as_text(row.get("category"))
            or _as_text(config.get("category_override"))
            or ""
        ).lower()

        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        feed_source = (
            _as_text(row.get("feed_source"))
            or _as_text(payload.get("feed_source"))
            or _as_text(config.get("feed_source"))
            or _infer_feed_source_from_slug(str(source.slug or ""))
        ).lower()

        observed_at = (
            _parse_datetime(row.get("observed_at"))
            or _parse_datetime(row.get("published"))
            or _parse_datetime(row.get("published_at"))
            or _parse_datetime(payload.get("published"))
        )
        fetched_at = (
            _parse_datetime(row.get("fetched_at"))
            or _parse_datetime(payload.get("fetched_at"))
            or datetime.now(timezone.utc)
        )

        if feed_source == "gdelt":
            summary = self._pick_gdelt_summary(row)
        else:
            summary = _clean_summary_text(
                row.get("summary")
                or row.get("description")
                or payload.get("summary")
                or payload.get("description")
                or "",
                max_len=2000,
            )

        return NewsArticle(
            article_id=external_id,
            title=title,
            url=url,
            source=source_name,
            published=observed_at,
            summary=summary,
            feed_source=feed_source,
            category=category,
            fetched_at=fetched_at,
        )

    async def persist_to_db(self) -> int:
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
                await _commit_with_retry(session)

            logger.debug("Persisted %d articles to DB", persisted)
            return persisted
        except Exception as exc:
            logger.warning("Failed to persist articles to DB: %s", exc)
            return 0

    async def load_from_db(self) -> int:
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
                    published = _parse_datetime(row.published)
                    fetched_at = _parse_datetime(row.fetched_at)
                    self._articles[row.article_id] = NewsArticle(
                        article_id=row.article_id,
                        title=row.title,
                        url=row.url,
                        source=row.source or "",
                        published=published,
                        summary=row.summary or "",
                        feed_source=row.feed_source or "",
                        category=row.category or "",
                        fetched_at=fetched_at or datetime.now(timezone.utc),
                        embedding=row.embedding,
                    )
                    loaded += 1
            logger.info("Loaded %d articles from DB", loaded)
            return loaded
        except Exception as exc:
            logger.warning("Failed to load articles from DB: %s", exc)
            return 0

    async def prune_db(self) -> int:
        try:
            from models.database import AsyncSessionLocal, NewsArticleCache
            from sqlalchemy import delete

            cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.NEWS_ARTICLE_TTL_HOURS)
            async with AsyncSessionLocal() as session:
                result = await session.execute(delete(NewsArticleCache).where(NewsArticleCache.fetched_at < cutoff))
                await _commit_with_retry(session)
                count = result.rowcount
            if count:
                logger.info("Pruned %d expired articles from DB", count)
            return count
        except Exception as exc:
            logger.warning("Failed to prune DB articles: %s", exc)
            return 0

    def search_articles(self, query: str, max_age_hours: int = 168, limit: int = 50) -> list[NewsArticle]:
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
            "source_fetch_failures": 0,
            "source_rows_skipped": 0,
        }

    def _inc_ingest_stat(self, key: str, delta: int = 1) -> None:
        self._ingest_stats[key] = int(self._ingest_stats.get(key, 0)) + int(delta)

    def _pick_gdelt_summary(self, raw: dict[str, Any]) -> str:
        for key in ("snippet", "summary", "description", "excerpt", "content", "body"):
            text = _clean_summary_text(raw.get(key), max_len=2000)
            if not text:
                continue
            if _is_url_like_summary(text):
                self._inc_ingest_stat("gdelt_summary_url_filtered")
                continue
            return text
        return ""

    def _prune_old_articles(self) -> None:
        ttl_seconds = settings.NEWS_ARTICLE_TTL_HOURS * 3600
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
        to_remove = [aid for aid, article in self._articles.items() if article.fetched_at.timestamp() < cutoff]
        for aid in to_remove:
            del self._articles[aid]
        if to_remove:
            logger.debug("Pruned %d old articles", len(to_remove))


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if parsed != parsed:
        return float(default)
    return parsed


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc)
        return value.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y%m%dT%H%M%S",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _infer_feed_source_from_slug(slug: str) -> str:
    lowered = str(slug or "").strip().lower()
    if "google" in lowered:
        return "google_news"
    if "gdelt" in lowered:
        return "gdelt"
    if "custom" in lowered:
        return "custom_rss"
    return "rss"


def _merge_record_payloads(raw: dict[str, Any], transformed: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    merged = dict(raw)
    merged.update(dict(transformed or {}))

    payload_from_row = merged.get("payload")
    if isinstance(payload_from_row, dict):
        payload.update(payload_from_row)

    merged["payload"] = payload
    return merged


async def _invoke_callable(func: Any, *args: Any, **kwargs: Any) -> Any:
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _has_min_text_quality(title: str, summary: str) -> bool:
    full = f"{title or ''} {summary or ''}".strip()
    alnum_chars = len(re.sub(r"[^A-Za-z0-9]", "", full))
    if alnum_chars < 80:
        return False
    if not (summary or "").strip() and len((title or "").strip()) < 35:
        return False
    return True


def _clean_summary_text(value: Any, *, max_len: int = 2000) -> str:
    text = _as_text(value) or ""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip()


def _is_url_like_summary(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text.startswith(("http://", "https://", "www.")) and " " not in text:
        return True
    if re.fullmatch(r"[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+", text) and "http" in text:
        return True
    return False


news_feed_service = NewsFeedService()
