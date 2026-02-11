import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_news


def _article(
    *,
    article_id: str,
    title: str,
    published: Optional[datetime],
    fetched_at: datetime,
    feed_source: str = "google_news",
) -> SimpleNamespace:
    return SimpleNamespace(
        article_id=article_id,
        title=title,
        source="Unit Test Source",
        feed_source=feed_source,
        url=f"https://example.com/{article_id}",
        published=published,
        category="test",
        summary="summary",
        embedding=None,
        fetched_at=fetched_at,
    )


@pytest.mark.asyncio
async def test_get_articles_sorts_by_published_then_fetched_and_serializes_utc(
    monkeypatch,
):
    # Ordered intentionally out of recency order.
    articles = [
        _article(
            article_id="old-published",
            title="Old published",
            published=datetime(2026, 2, 10, 9, 0, 0),  # naive UTC
            fetched_at=datetime(2026, 2, 11, 5, 0, 0, tzinfo=timezone.utc),
        ),
        _article(
            article_id="recent-published",
            title="Recent published",
            published=datetime(2026, 2, 11, 4, 0, 0, tzinfo=timezone.utc),
            fetched_at=datetime(2026, 2, 11, 4, 5, 0, tzinfo=timezone.utc),
        ),
        _article(
            article_id="no-published",
            title="No published date",
            published=None,
            fetched_at=datetime(2026, 2, 11, 6, 0, 0, tzinfo=timezone.utc),
        ),
    ]
    fake_service = SimpleNamespace(get_articles=lambda max_age_hours=None: list(articles))
    monkeypatch.setitem(
        sys.modules,
        "services.news.feed_service",
        SimpleNamespace(news_feed_service=fake_service),
    )

    payload = await routes_news.get_articles(
        max_age_hours=168,
        source=None,
        limit=10,
        offset=0,
    )

    assert [a["article_id"] for a in payload["articles"]] == [
        "no-published",
        "recent-published",
        "old-published",
    ]

    # Naive datetimes should be normalized to explicit UTC in API payloads.
    old_row = next(a for a in payload["articles"] if a["article_id"] == "old-published")
    assert old_row["published"].endswith("Z")
    assert old_row["fetched_at"].endswith("Z")
