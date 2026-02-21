import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_news_workflow


def _finding(*, evidence=None, market_id="0xmarket", market_question="Will this resolve yes?", market_price=0.42, confidence=0.7):
    return SimpleNamespace(
        article_id="cluster:abc123",
        market_id=market_id,
        market_question=market_question,
        market_price=market_price,
        confidence=confidence,
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
        "https://polymarket.com/event/how-many-people-will-trump-deport-in-2025/will-trump-deport-less-than-250000"
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
    assert payload["market_url"] == "https://kalshi.com/markets/kxelonmars/kxelonmars-99"
    assert payload["polymarket_url"] is None
    assert payload["kalshi_url"] == payload["market_url"]


@pytest.mark.asyncio
async def test_load_shared_backfill_market_history_uses_scanner_attach(monkeypatch):
    yes_token = "123456789012345678901"
    no_token = "123456789012345678902"
    finding = _finding(
        evidence={
            "market": {
                "condition_id": "0xmarket",
                "token_ids": [yes_token, no_token],
                "platform": "polymarket",
            }
        }
    )

    async def _attach_side_effect(opportunities, **_kwargs):
        opportunities[0].markets[0]["price_history"] = [
            {"t": 1.0, "yes": 0.41, "no": 0.59},
            {"t": 2.0, "yes": 0.43, "no": 0.57},
        ]
        return 1

    attach_mock = AsyncMock(side_effect=_attach_side_effect)
    monkeypatch.setattr(
        "services.scanner.scanner.attach_price_history_to_opportunities",
        attach_mock,
    )

    history_map = await routes_news_workflow._load_shared_backfill_market_history([finding])

    assert attach_mock.await_count == 1
    args = attach_mock.await_args
    assert args is not None
    assert len(args.args) == 1
    assert len(args.args[0]) == 1
    assert args.kwargs.get("timeout_seconds") == 12.0
    assert "now" in args.kwargs
    assert args.args[0][0].strategy == "news_edge"
    assert args.args[0][0].markets[0]["id"] == "0xmarket"
    assert args.args[0][0].markets[0]["clob_token_ids"] == [yes_token, no_token]
    assert "0xmarket" in history_map
    assert len(history_map["0xmarket"]) == 2


@pytest.mark.asyncio
async def test_load_shared_backfill_market_history_tolerates_attach_failure(monkeypatch):
    finding = _finding()

    attach_mock = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        "services.scanner.scanner.attach_price_history_to_opportunities",
        attach_mock,
    )

    history_map = await routes_news_workflow._load_shared_backfill_market_history([finding])

    assert attach_mock.await_count == 1
    assert history_map == {}
