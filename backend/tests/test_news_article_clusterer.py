import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.news.article_clusterer import ArticleClusterer
from services.news.feed_service import NewsArticle


def _article(
    article_id: str,
    title: str,
    summary: str,
    source: str,
    minutes_ago: int,
) -> NewsArticle:
    now = datetime.now(timezone.utc)
    ts = now - timedelta(minutes=minutes_ago)
    return NewsArticle(
        article_id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        source=source,
        published=ts,
        summary=summary,
        feed_source="custom_rss",
        category="politics",
        fetched_at=ts,
    )


def test_clusterer_groups_related_articles():
    clusterer = ArticleClusterer(similarity_threshold=0.2, recency_window_hours=72)
    articles = [
        _article(
            "a1",
            "Nancy Mace gains ground in South Carolina GOP primary poll",
            "A new statewide poll shows Nancy Mace widening her lead.",
            "Reuters",
            5,
        ),
        _article(
            "a2",
            "New survey gives Nancy Mace lead in SC Republican governor race",
            "Latest polling data points to Mace momentum before primary voting.",
            "AP",
            9,
        ),
        _article(
            "a3",
            "Bitcoin ETF inflows hit highest daily total this month",
            "Fund managers report strong demand after macro data release.",
            "Bloomberg",
            12,
        ),
    ]

    clusters = clusterer.cluster(articles, max_clusters=10)
    sizes = sorted([cluster.article_count for cluster in clusters], reverse=True)
    assert sizes == [2, 1]

    grouped_ids = [set(cluster.article_ids) for cluster in clusters]
    assert {"a1", "a2"} in grouped_ids


def test_clusterer_sets_cluster_identity_metadata():
    clusterer = ArticleClusterer(similarity_threshold=0.2, recency_window_hours=72)
    article = _article(
        "single-1",
        "US Supreme Court to hear election ballot case next week",
        "Court schedules expedited hearing tied to state election dispute.",
        "Reuters",
        3,
    )

    clusters = clusterer.cluster([article], max_clusters=10)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.article_key.startswith("cluster:")
    assert cluster.cluster_id
    assert cluster.article_count == 1
    assert cluster.primary_source.lower() == "reuters"
    assert cluster.summary
