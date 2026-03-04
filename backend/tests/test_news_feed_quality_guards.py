import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.news.feed_service import (
    NewsArticle,
    NewsFeedService,
    _parse_datetime,
    _clean_summary_text,
    _has_min_text_quality,
)
import services.news.feed_service as feed_service_module


def test_google_description_is_normalized_for_summary():
    raw_description = "<p><b>Breaking:</b> Candidate gains momentum in latest poll.</p>"
    summary = _clean_summary_text(raw_description, max_len=500)
    assert summary == "Breaking: Candidate gains momentum in latest poll."


def test_gdelt_summary_ignores_url_like_snippet():
    service = NewsFeedService()
    raw = {
        "snippet": "https://cdn.example.com/image.jpg",
        "description": "Analysts expect tighter governor primary race after new filing.",
    }
    summary = service._pick_gdelt_summary(raw)
    assert summary == "Analysts expect tighter governor primary race after new filing."
    assert service._ingest_stats["gdelt_summary_url_filtered"] == 1


def test_min_text_quality_rejects_short_title_without_summary():
    assert _has_min_text_quality("Quick update", "") is False
    assert (
        _has_min_text_quality(
            "South Carolina gubernatorial primary shifts after endorsement news",
            "A longer summary gives enough content for meaningful matching.",
        )
        is True
    )


@pytest.mark.asyncio
async def test_fetch_source_rows_handles_string_limit_without_name_error(monkeypatch):
    service = NewsFeedService()

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(feed_service_module, "AsyncSessionLocal", lambda: DummySession())

    source = SimpleNamespace(
        id="stories_test_id",
        slug="stories_test",
        config={"limit": "25"},
    )

    rows = await service._fetch_source_rows(source)
    assert rows == []


def test_parse_datetime_normalizes_to_naive_utc():
    parsed = _parse_datetime("2026-02-21T00:29:13+00:00")
    assert parsed is not None
    assert parsed.tzinfo is None
    assert parsed == datetime(2026, 2, 21, 0, 29, 13)


@pytest.mark.asyncio
async def test_fetch_all_handles_mixed_timezone_article_timestamps(monkeypatch):
    service = NewsFeedService()

    article_id = "mixed-timezone-id"
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    service._articles[article_id] = NewsArticle(
        article_id=article_id,
        title="Existing article with enough alphanumeric content for quality checks to pass reliably",
        url="https://example.com/existing",
        source="existing",
        summary="Existing summary content that easily exceeds the minimum text quality threshold requirements.",
        fetched_at=now_utc - timedelta(minutes=1),
    )

    incoming = NewsArticle(
        article_id=article_id,
        title="Incoming article with enough alphanumeric content for quality checks to pass reliably",
        url="https://example.com/incoming",
        source="incoming",
        summary="Incoming summary content that also exceeds the text quality threshold by a wide margin.",
        fetched_at=now_utc.replace(tzinfo=None),
    )

    async def _fake_load_sources():
        return [SimpleNamespace()]

    async def _fake_fetch_articles(_sources):
        return [incoming]

    monkeypatch.setattr(service, "_load_enabled_story_sources", _fake_load_sources)
    monkeypatch.setattr(service, "_fetch_articles_from_sources", _fake_fetch_articles)

    new_articles = await service.fetch_all()
    assert new_articles == []
    assert service._articles[article_id].source == "incoming"
    assert service._articles[article_id].fetched_at == now_utc.replace(tzinfo=None)
    assert service._articles[article_id].fetched_at.tzinfo is None
