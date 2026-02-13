"""
News Intelligence Layer for Homerun.

Provides:
- Multi-source news ingestion (Google News RSS, GDELT, custom RSS)
- Cluster-first topic grouping before retrieval
- Independent news workflow pipeline (Options B/C/D):
  - Article clustering (article_clusterer)
  - Event extraction (event_extractor)
  - Market watcher reverse index (market_watcher_index)
  - Hybrid retrieval (hybrid_retriever)
  - LLM reranking (reranker)
  - Edge estimation with evidence chain (edge_estimator)
  - Trade intent generation (intent_generator)
  - Workflow orchestration (workflow_orchestrator)
"""

from services.news.feed_service import news_feed_service, NewsFeedService
from services.news.gov_rss_feeds import GovRSSFeedService, gov_rss_service, rss_service
from services.news.semantic_matcher import semantic_matcher, SemanticMatcher

__all__ = [
    "news_feed_service",
    "NewsFeedService",
    "rss_service",
    "gov_rss_service",
    "GovRSSFeedService",
    "semantic_matcher",
    "SemanticMatcher",
]
