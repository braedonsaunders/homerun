import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.data_events import DataEvent
from services.strategies.news_edge import NewsEdgeStrategy


@pytest.mark.asyncio
async def test_news_edge_on_event_applies_strategy_thresholds():
    strategy = NewsEdgeStrategy()
    strategy.configure(
        {
            "min_edge_percent": 5.0,
            "min_confidence": 0.45,
            "min_supporting_articles": 1,
            "min_supporting_sources": 1,
            "require_verifier": False,
        }
    )
    event = DataEvent(
        event_type="news_update",
        source="test",
        timestamp=datetime.now(timezone.utc),
        payload={
            "findings": [
                {
                    "id": "finding-low-edge",
                    "market_id": "mkt-1",
                    "market_question": "Will X happen?",
                    "direction": "buy_yes",
                    "market_price": 0.42,
                    "model_probability": 0.44,
                    "edge_percent": 2.0,
                    "confidence": 0.9,
                    "article_id": "a-1",
                    "article_source": "Reuters",
                },
                {
                    "id": "finding-pass",
                    "market_id": "mkt-2",
                    "market_question": "Will Y happen?",
                    "direction": "buy_yes",
                    "market_price": 0.42,
                    "model_probability": 0.57,
                    "edge_percent": 15.0,
                    "confidence": 0.72,
                    "article_id": "a-2",
                    "article_source": "AP",
                },
            ]
        },
    )

    opportunities = await strategy.on_event(event)

    assert len(opportunities) == 1
    assert opportunities[0].strategy == "news_edge"
    assert opportunities[0].markets[0]["id"] == "mkt-2"


@pytest.mark.asyncio
async def test_news_edge_on_event_respects_verifier_requirement():
    strategy = NewsEdgeStrategy()
    strategy.configure(
        {
            "min_edge_percent": 5.0,
            "min_confidence": 0.45,
            "min_supporting_articles": 1,
            "min_supporting_sources": 1,
            "require_verifier": True,
        }
    )
    event = DataEvent(
        event_type="news_update",
        source="test",
        timestamp=datetime.now(timezone.utc),
        payload={
            "findings": [
                {
                    "id": "finding-verifier-failed",
                    "market_id": "mkt-3",
                    "market_question": "Will Z happen?",
                    "direction": "buy_no",
                    "market_price": 0.61,
                    "model_probability": 0.31,
                    "edge_percent": 8.0,
                    "confidence": 0.66,
                    "article_id": "a-3",
                    "article_source": "Bloomberg",
                    "evidence": {
                        "rejection_reasons": ["verifier_failed"],
                        "cluster": {"article_refs": [{"source": "Bloomberg"}]},
                    },
                }
            ]
        },
    )

    opportunities = await strategy.on_event(event)

    assert opportunities == []


@pytest.mark.asyncio
async def test_news_edge_on_event_populates_uncertainty_context_fields():
    strategy = NewsEdgeStrategy()
    strategy.configure(
        {
            "min_edge_percent": 5.0,
            "min_confidence": 0.45,
            "min_supporting_articles": 1,
            "min_supporting_sources": 1,
            "require_verifier": False,
        }
    )
    event = DataEvent(
        event_type="news_update",
        source="test",
        timestamp=datetime.now(timezone.utc),
        payload={
            "findings": [
                {
                    "id": "finding-ci",
                    "market_id": "mkt-ci",
                    "market_question": "Will policy X pass?",
                    "direction": "buy_yes",
                    "market_price": 0.52,
                    "model_probability": 0.69,
                    "edge_percent": 17.0,
                    "confidence": 0.61,
                    "article_id": "a-ci",
                    "article_source": "Reuters",
                }
            ]
        },
    )

    opportunities = await strategy.on_event(event)
    assert len(opportunities) == 1
    context = opportunities[0].strategy_context
    assert context["model_probability_ci_low"] <= context["model_probability"] <= context["model_probability_ci_high"]
    assert context["model_probability_ci_width"] > 0.0
    assert context["confidence_uncertainty"] == pytest.approx(0.39, rel=1e-6)


def test_news_edge_evaluate_includes_uncertainty_payload():
    strategy = NewsEdgeStrategy()
    signal = SimpleNamespace(
        edge_percent=12.5,
        confidence=0.74,
        source="news",
        payload_json={
            "risk_score": 0.35,
            "markets": [{"id": "mkt-1"}],
            "strategy_context": {
                "model_probability": 0.67,
                "model_probability_ci_low": 0.55,
                "model_probability_ci_high": 0.76,
            },
            "positions_to_take": [{"_news_edge": {"model_probability": 0.67}}],
        },
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 5.0,
                "min_confidence": 0.45,
                "max_risk_score": 0.8,
                "base_size_usd": 20.0,
                "max_size_usd": 120.0,
            }
        },
    )

    assert decision.decision == "selected"
    assert decision.payload["model_probability"] == pytest.approx(0.67, rel=1e-6)
    assert decision.payload["model_probability_ci_low"] == pytest.approx(0.55, rel=1e-6)
    assert decision.payload["model_probability_ci_high"] == pytest.approx(0.76, rel=1e-6)
    assert decision.payload["model_probability_ci_width"] == pytest.approx(0.21, rel=1e-6)
    assert decision.payload["confidence_uncertainty"] == pytest.approx(0.26, rel=1e-6)
