"""Backward-compat import shim for gov RSS service.

RSS ownership now lives under services.news.
"""

from services.news.gov_rss_feeds import (
    GovArticle,
    GovRSSFeedService,
    gov_rss_service,
    rss_service,
)

__all__ = ["GovArticle", "GovRSSFeedService", "gov_rss_service", "rss_service"]
