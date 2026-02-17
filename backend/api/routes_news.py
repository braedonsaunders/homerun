"""API routes for News feed ingestion and article browsing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from utils.converters import to_iso

router = APIRouter()


def _article_recency_timestamp(article) -> float:
    published = getattr(article, "published", None)
    if published is not None:
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return published.timestamp()
    fetched = getattr(article, "fetched_at", None)
    if fetched is not None:
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return fetched.timestamp()
    return 0.0


def _normalize_feed_source(raw: Optional[str]) -> str:
    source = str(raw or "").strip().lower()
    if source == "gov_rss":
        return "rss"
    return source or "unknown"


@router.get("/news/feed/status")
async def get_feed_status():
    """Get current feed status and source distribution."""
    from services.news.feed_service import news_feed_service

    articles = news_feed_service.get_articles()
    sources: dict[str, int] = {}
    for article in articles:
        key = _normalize_feed_source(getattr(article, "feed_source", None))
        sources[key] = sources.get(key, 0) + 1

    return {
        "article_count": len(articles),
        "sources": sources,
        "running": bool(news_feed_service._running),
    }


@router.post("/news/feed/fetch")
async def trigger_news_fetch():
    """Trigger one manual fetch cycle."""
    from services.news.feed_service import news_feed_service

    try:
        new_articles = await news_feed_service.fetch_all()
        return {
            "new_articles": len(new_articles),
            "total_articles": news_feed_service.article_count,
            "articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "feed_source": _normalize_feed_source(a.feed_source),
                    "url": a.url,
                    "published": to_iso(a.published),
                    "category": a.category,
                }
                for a in new_articles[:20]
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/news/feed/articles")
async def get_articles(
    max_age_hours: int = Query(168, ge=1, le=336),
    source: Optional[str] = Query(None, description="Filter by feed source"),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List cached articles in recency order."""
    from services.news.feed_service import news_feed_service

    articles = news_feed_service.get_articles(max_age_hours=max_age_hours)

    if source:
        filter_key = _normalize_feed_source(source)
        articles = [
            article
            for article in articles
            if _normalize_feed_source(getattr(article, "feed_source", None)) == filter_key
        ]

    articles.sort(key=_article_recency_timestamp, reverse=True)
    total = len(articles)
    page = articles[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
        "articles": [
            {
                "article_id": a.article_id,
                "title": a.title,
                "source": a.source,
                "feed_source": _normalize_feed_source(a.feed_source),
                "url": a.url,
                "published": to_iso(a.published),
                "category": a.category,
                "summary": a.summary[:200] if a.summary else "",
                "has_embedding": a.embedding is not None,
                "fetched_at": to_iso(a.fetched_at),
            }
            for a in page
        ],
    }


@router.get("/news/feed/search")
async def search_articles(
    q: str = Query(..., min_length=1, description="Keyword to search"),
    max_age_hours: int = Query(168, ge=1, le=336),
    limit: int = Query(50, ge=1, le=200),
):
    """Search cached articles by keyword."""
    from services.news.feed_service import news_feed_service

    results = news_feed_service.search_articles(
        query=q,
        max_age_hours=max_age_hours,
        limit=limit,
    )
    return {
        "query": q,
        "total": len(results),
        "articles": [
            {
                "article_id": a.article_id,
                "title": a.title,
                "source": a.source,
                "feed_source": _normalize_feed_source(a.feed_source),
                "url": a.url,
                "published": to_iso(a.published),
                "category": a.category,
                "summary": a.summary[:200] if a.summary else "",
                "has_embedding": a.embedding is not None,
                "fetched_at": to_iso(a.fetched_at),
            }
            for a in results
        ],
    }


@router.delete("/news/feed/clear")
async def clear_articles():
    """Clear in-memory feed cache."""
    from services.news.feed_service import news_feed_service

    count = news_feed_service.clear()
    return {"cleared": count}
