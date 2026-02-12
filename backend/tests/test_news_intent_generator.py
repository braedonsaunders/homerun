import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.news.edge_estimator import WorkflowFinding
from services.news.intent_generator import IntentGenerator


@pytest.mark.asyncio
async def test_intent_generator_carries_reasoning_and_evidence():
    generator = IntentGenerator()
    finding = WorkflowFinding(
        id="finding-1",
        article_id="article-1",
        market_id="mkt-1",
        market_question="Will event happen?",
        market_price=0.43,
        model_probability=0.62,
        edge_percent=19.0,
        direction="buy_yes",
        confidence=0.74,
        actionable=True,
        reasoning="Reasoning text",
        evidence={"retrieval": {"score": 0.81}},
    )

    intents = await generator.generate(
        findings=[finding],
        min_edge=10.0,
        min_confidence=0.6,
        market_metadata_by_id={"mkt-1": {"liquidity": 10000.0}},
    )

    assert len(intents) == 1
    metadata = intents[0]["metadata_json"]
    assert metadata["finding"]["reasoning"] == "Reasoning text"
    assert metadata["finding"]["evidence"] == {"retrieval": {"score": 0.81}}
    assert metadata["supporting_article_count"] == 1
    assert len(metadata["supporting_articles"]) == 1
    assert metadata["supporting_articles"][0]["article_id"] == "article-1"


@pytest.mark.asyncio
async def test_intent_generator_includes_cluster_supporting_articles():
    generator = IntentGenerator()
    finding = WorkflowFinding(
        id="finding-2",
        article_id="cluster:abc123",
        market_id="mkt-2",
        market_question="Will candidate X win?",
        market_price=0.48,
        model_probability=0.63,
        edge_percent=15.0,
        direction="buy_yes",
        confidence=0.71,
        actionable=True,
        reasoning="Cluster-backed signal",
        evidence={
            "cluster": {
                "article_refs": [
                    {
                        "article_id": "a1",
                        "title": "Article one",
                        "url": "https://example.com/a1",
                        "source": "Reuters",
                    },
                    {
                        "article_id": "a2",
                        "title": "Article two",
                        "url": "https://example.com/a2",
                        "source": "AP",
                    },
                ]
            }
        },
    )

    intents = await generator.generate(
        findings=[finding],
        min_edge=10.0,
        min_confidence=0.6,
        market_metadata_by_id={"mkt-2": {"liquidity": 10000.0}},
    )

    assert len(intents) == 1
    metadata = intents[0]["metadata_json"]
    assert metadata["supporting_article_count"] == 2
    assert [a["article_id"] for a in metadata["supporting_articles"]] == ["a1", "a2"]
