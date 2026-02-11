"""
News Intelligence Layer for Homerun.

Provides:
- Multi-source news ingestion (Google News RSS, GDELT, custom RSS)
- Semantic matching of news articles to prediction markets
- News-driven edge detection strategy
- On-demand forecaster committee for deep analysis
- Independent news workflow pipeline (Options B/C/D):
  - Event extraction (event_extractor)
  - Market watcher reverse index (market_watcher_index)
  - Hybrid retrieval (hybrid_retriever)
  - LLM reranking (reranker)
  - Edge estimation with evidence chain (edge_estimator)
  - Trade intent generation (intent_generator)
  - Workflow orchestration (workflow_orchestrator)
"""

from services.news.feed_service import news_feed_service, NewsFeedService
from services.news.semantic_matcher import semantic_matcher, SemanticMatcher

__all__ = [
    "news_feed_service",
    "NewsFeedService",
    "semantic_matcher",
    "SemanticMatcher",
]
