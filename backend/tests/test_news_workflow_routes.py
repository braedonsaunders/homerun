import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_news_workflow


def _finding(*, evidence=None):
    return SimpleNamespace(
        article_id="cluster:abc123",
        article_title="Primary article",
        article_url="https://example.com/primary",
        article_source="Primary Source",
        created_at=datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc),
        evidence=evidence or {},
    )


def test_build_supporting_articles_prefers_cluster_refs_and_dedupes():
    finding = _finding(
        evidence={
            "cluster": {
                "article_refs": [
                    {
                        "article_id": "a1",
                        "title": "First source",
                        "url": "https://example.com/a1",
                        "source": "Reuters",
                    },
                    {
                        "article_id": "a1",
                        "title": "First source duplicate",
                        "url": "https://example.com/a1",
                        "source": "Reuters",
                    },
                    {
                        "article_id": "a2",
                        "title": "Second source",
                        "url": "https://example.com/a2",
                        "source": "AP",
                    },
                ]
            }
        }
    )

    refs = routes_news_workflow._build_supporting_articles_from_finding(finding)

    assert [r["article_id"] for r in refs] == ["a1", "a2"]
    assert [r["title"] for r in refs] == ["First source", "Second source"]


def test_build_supporting_articles_uses_cache_for_cluster_article_ids():
    finding = _finding(
        evidence={
            "cluster": {
                "article_ids": ["a1", "a2"],
            }
        }
    )
    cache = {
        "a1": SimpleNamespace(
            article_id="a1",
            title="Cache one",
            url="https://example.com/cache-a1",
            source="BBC",
            published=datetime(2026, 2, 11, 9, 0, 0, tzinfo=timezone.utc),
            fetched_at=datetime(2026, 2, 11, 9, 5, 0, tzinfo=timezone.utc),
        ),
        "a2": SimpleNamespace(
            article_id="a2",
            title="Cache two",
            url="https://example.com/cache-a2",
            source="AP",
            published=None,
            fetched_at=datetime(2026, 2, 11, 10, 5, 0, tzinfo=timezone.utc),
        ),
    }

    refs = routes_news_workflow._build_supporting_articles_from_finding(finding, article_cache_by_id=cache)

    assert [r["article_id"] for r in refs] == ["a1", "a2"]
    assert refs[0]["published"].endswith("Z")
    assert refs[1]["fetched_at"].endswith("Z")


def test_build_supporting_articles_falls_back_to_primary_article():
    finding = _finding(evidence={})

    refs = routes_news_workflow._build_supporting_articles_from_finding(finding)

    assert len(refs) == 1
    assert refs[0]["article_id"] == "cluster:abc123"
    assert refs[0]["title"] == "Primary article"
    assert refs[0]["url"] == "https://example.com/primary"


def test_build_market_link_payload_prefers_polymarket_slug():
    payload = routes_news_workflow._build_market_link_payload(
        market_id="0xdeadbeef",
        market_question="Will this resolve yes?",
        market_context={
            "slug": "will-trump-deport-less-than-250000",
            "event_slug": "how-many-people-will-trump-deport-in-2025",
        },
    )

    assert payload["market_platform"] == "polymarket"
    assert payload["market_url"] == (
        "https://polymarket.com/event/"
        "how-many-people-will-trump-deport-in-2025/"
        "will-trump-deport-less-than-250000"
    )
    assert payload["polymarket_url"] == payload["market_url"]
    assert payload["kalshi_url"] is None


def test_build_market_link_payload_handles_kalshi_ticker():
    payload = routes_news_workflow._build_market_link_payload(
        market_id="KXELONMARS-99",
        market_question="Will this resolve yes?",
        market_context={},
    )

    assert payload["market_platform"] == "kalshi"
    assert payload["market_url"] == "https://kalshi.com/markets/KXELONMARS-99"
    assert payload["polymarket_url"] is None
    assert payload["kalshi_url"] == payload["market_url"]
