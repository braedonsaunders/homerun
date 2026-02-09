"""
News Intelligence Layer for Homerun.

Provides:
- Multi-source news ingestion (Google News RSS, GDELT, custom RSS)
- Semantic matching of news articles to prediction markets
- News-driven edge detection strategy
- On-demand forecaster committee for deep analysis
"""

from services.news.feed_service import news_feed_service, NewsFeedService
from services.news.semantic_matcher import semantic_matcher, SemanticMatcher

__all__ = [
    "news_feed_service",
    "NewsFeedService",
    "semantic_matcher",
    "SemanticMatcher",
]
