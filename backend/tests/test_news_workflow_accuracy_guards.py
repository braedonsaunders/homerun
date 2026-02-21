import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.news.edge_estimator import EdgeEstimator
from services.news.event_extractor import ExtractedEvent
from services.news.hybrid_retriever import HybridRetriever
from services.news.market_watcher_index import IndexedMarket, SearchResult
from services.news.reranker import RerankedCandidate
from services.news.workflow_orchestrator import WorkflowOrchestrator


def test_alignment_gate_requires_entity_overlap():
    orchestrator = WorkflowOrchestrator()
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Nancy Mace"],
        actors=[],
        action="wins Republican primary",
        confidence=0.8,
    )

    aligned_candidate = type(
        "Candidate",
        (),
        {
            "question": "Will Nancy Mace win the 2026 South Carolina Governor Republican primary election?",
            "event_title": "South Carolina Governor Primary",
            "slug": "sc-governor-republican-primary",
        },
    )()
    unaligned_candidate = type(
        "Candidate",
        (),
        {
            "question": "Will Bitcoin trade above $120k by year end?",
            "event_title": "Crypto Prices",
            "slug": "bitcoin-120k",
        },
    )()

    assert orchestrator._has_event_market_alignment(event, aligned_candidate) is True
    assert orchestrator._has_event_market_alignment(event, unaligned_candidate) is False


def test_alignment_gate_ignores_source_like_entities():
    orchestrator = WorkflowOrchestrator()
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Houston Public Media"],
        actors=[],
        action="local story update",
        confidence=0.6,
    )
    candidate = type(
        "Candidate",
        (),
        {
            "question": "Will Nancy Mace win the 2026 South Carolina Governor Republican primary election?",
            "event_title": "South Carolina Governor Primary",
            "slug": "sc-governor-republican-primary",
            "tags": [],
        },
    )()
    assert orchestrator._has_event_market_alignment(event, candidate) is False


def test_hybrid_retriever_filters_category_only_false_positives():
    class _FakeIndex:
        is_ml_mode = False

        def search(
            self,
            query_terms,
            query_embedding,
            category_filter,
            min_liquidity,
            top_k,
            keyword_weight,
            semantic_weight,
        ):
            market = IndexedMarket(
                market_id="mkt_1",
                question="Will Nancy Mace win the 2026 South Carolina Governor Republican primary election?",
                event_title="SC Governor Primary",
                category="Politics",
                yes_price=0.42,
                no_price=0.58,
                liquidity=10000.0,
                slug="sc-governor-primary",
            )
            weak = SearchResult(
                market=market,
                keyword_score=0.0,
                semantic_score=0.12,
                combined_score=0.12,
            )
            strong = SearchResult(
                market=market,
                keyword_score=0.04,
                semantic_score=0.21,
                combined_score=0.21,
            )
            return [weak, strong]

    retriever = HybridRetriever(_FakeIndex())
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Nancy Mace"],
        keywords=["nancy", "mace"],
        confidence=0.8,
    )
    out = retriever.retrieve(
        event=event,
        article_text="Nancy Mace launches campaign rally in South Carolina",
        top_k=5,
        similarity_threshold=0.1,
    )

    # Weak semantic/category-only candidate should be removed.
    assert len(out) == 1
    assert out[0].semantic_score >= 0.2


def test_temporal_guard_rejects_market_that_ended_before_article():
    orchestrator = WorkflowOrchestrator()
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Nancy Mace"],
        action="wins primary",
        confidence=0.8,
    )
    article = type(
        "Article",
        (),
        {
            "published": datetime.now(timezone.utc),
            "fetched_at": datetime.now(timezone.utc),
        },
    )()
    candidate = type(
        "Candidate",
        (),
        {
            "end_date": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
        },
    )()

    assert orchestrator._is_temporally_compatible(article, event, candidate) is False


def test_require_verifier_rejects_non_llm_rerank_candidates():
    orchestrator = WorkflowOrchestrator()
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Nancy Mace"],
        action="wins primary",
        confidence=0.8,
    )
    article = type(
        "Article",
        (),
        {
            "article_id": "art_1",
            "title": "Abbott breaks with Trump over election changes",
            "source": "Houston Public Media",
            "url": "https://example.com/news",
        },
    )()
    candidate = type(
        "Candidate",
        (),
        {
            "market_id": "mkt_1",
            "question": "Will Nancy Mace win the 2026 South Carolina Governor Republican primary election?",
            "event_title": "SC Governor Primary",
            "category": "Politics",
            "yes_price": 0.42,
            "no_price": 0.58,
            "combined_score": 0.36,
            "semantic_score": 0.19,
            "keyword_score": 0.02,
            "event_score": 1.0,
            "slug": "sc-governor-primary",
            "liquidity": 10000.0,
        },
    )()
    reranked = RerankedCandidate(
        candidate=candidate,
        relevance=0.36,
        rationale="Retrieval score (LLM unavailable)",
        rerank_score=0.36,
        used_llm=False,
    )

    # Case 1: LLM was NOT requested (budget skip) -> penalized, not rejected.
    verified, penalized, rejected = orchestrator._split_verified_candidates(
        article=article,
        event=event,
        reranked=[reranked],
        llm_was_requested=False,
    )
    assert verified == []
    assert len(penalized) == 1
    assert rejected == []
    # Penalized candidates get a 0.7x confidence penalty and a tag.
    assert penalized[0].rerank_score == reranked.rerank_score  # Already applied in-place
    assert "unverified_budget_skip" in penalized[0].rationale

    # Case 2: LLM WAS requested but candidate not scored -> hard rejection.
    reranked2 = RerankedCandidate(
        candidate=candidate,
        relevance=0.36,
        rationale="Retrieval score (LLM unavailable)",
        rerank_score=0.36,
        used_llm=False,
    )
    verified2, penalized2, rejected2 = orchestrator._split_verified_candidates(
        article=article,
        event=event,
        reranked=[reranked2],
        llm_was_requested=True,
    )
    assert verified2 == []
    assert penalized2 == []
    assert len(rejected2) == 1
    reasons = rejected2[0].evidence.get("rejection_reasons", [])
    assert reasons == ["verifier_failed"]


def test_edge_estimator_records_rejection_when_llm_not_used():
    estimator = EdgeEstimator()
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Nancy Mace"],
        action="wins primary",
        confidence=0.8,
    )
    candidate = type(
        "Candidate",
        (),
        {
            "market_id": "mkt_1",
            "question": "Will Nancy Mace win the 2026 South Carolina Governor Republican primary election?",
            "event_title": "SC Governor Primary",
            "category": "Politics",
            "yes_price": 0.42,
            "no_price": 0.58,
            "combined_score": 0.51,
            "semantic_score": 0.34,
            "keyword_score": 0.08,
            "event_score": 1.0,
            "slug": "sc-governor-primary",
            "liquidity": 10000.0,
        },
    )()
    reranked = RerankedCandidate(
        candidate=candidate,
        relevance=0.8,
        rationale="Directly about candidate election odds.",
        rerank_score=0.72,
    )

    finding = asyncio.run(
        estimator._estimate_one(
            article_title="Nancy Mace surges in latest polling",
            article_summary="New poll places Mace ahead in GOP primary.",
            article_source="Test Source",
            article_url="https://example.com/a",
            article_id="art_1",
            event=event,
            rc=reranked,
            model=None,
            allow_llm=False,
        )
    )

    assert finding is not None
    assert finding.actionable is False
    rejection_reasons = finding.evidence.get("rejection_reasons", [])
    assert "llm_budget_exhausted" in rejection_reasons


def test_edge_estimator_does_not_mark_rejected_findings_actionable_even_with_zero_thresholds():
    estimator = EdgeEstimator()
    event = ExtractedEvent(
        event_type="election",
        key_entities=["Nancy Mace"],
        action="wins primary",
        confidence=0.8,
    )
    candidate = type(
        "Candidate",
        (),
        {
            "market_id": "mkt_1",
            "question": "Will Nancy Mace win the 2026 South Carolina Governor Republican primary election?",
            "event_title": "SC Governor Primary",
            "category": "Politics",
            "yes_price": 0.42,
            "no_price": 0.58,
            "combined_score": 0.51,
            "semantic_score": 0.34,
            "keyword_score": 0.08,
            "event_score": 1.0,
            "slug": "sc-governor-primary",
            "liquidity": 10000.0,
        },
    )()
    reranked = RerankedCandidate(
        candidate=candidate,
        relevance=0.8,
        rationale="Directly about candidate election odds.",
        rerank_score=0.72,
    )

    findings = asyncio.run(
        estimator.estimate_batch(
            article_title="Nancy Mace surges in latest polling",
            article_summary="New poll places Mace ahead in GOP primary.",
            article_source="Test Source",
            article_url="https://example.com/a",
            article_id="art_1",
            event=event,
            reranked=[reranked],
            min_edge_percent=0.0,
            min_confidence=0.0,
            allow_llm=False,
            max_llm_calls=0,
        )
    )

    assert len(findings) == 1
    assert findings[0].actionable is False
    rejection_reasons = findings[0].evidence.get("rejection_reasons", [])
    assert "llm_budget_exhausted" in rejection_reasons


def test_cached_row_to_finding_drops_actionable_for_rejected_zero_edge_rows():
    row = SimpleNamespace(
        id="finding_1",
        article_id="article_1",
        market_id="market_1",
        article_title="Title",
        article_source="Source",
        article_url="https://example.com/a",
        market_question="Will this resolve yes?",
        market_price=0.42,
        model_probability=0.42,
        edge_percent=0.0,
        direction="buy_yes",
        confidence=0.0,
        retrieval_score=0.4,
        semantic_score=0.3,
        keyword_score=0.1,
        event_score=0.2,
        rerank_score=0.5,
        event_graph={},
        evidence={"rejection_reasons": ["llm_unavailable_or_failed"]},
        reasoning="Rejected before actionable edge.",
        actionable=True,
        created_at=datetime.now(timezone.utc),
        signal_key="signal_1",
        cache_key="cache_1",
    )

    finding = WorkflowOrchestrator._row_to_finding(row)

    assert finding.actionable is False
    assert finding.edge_percent == 0.0
    assert finding.confidence == 0.0


def test_local_model_mode_detection():
    orchestrator = WorkflowOrchestrator()

    assert orchestrator._is_local_model_mode("ollama/mistral:7b", {}) is True
    assert (
        orchestrator._is_local_model_mode(
            "mistral-7b-instruct",
            {"configured_providers": ["ollama"]},
        )
        is True
    )
    assert (
        orchestrator._is_local_model_mode(
            "gpt-4o-mini",
            {"configured_providers": ["openai"]},
        )
        is False
    )
