import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.news.rss_config import merge_custom_rss_feeds, merge_gov_rss_feeds


def test_merge_custom_rss_feeds_seeds_defaults():
    merged = merge_custom_rss_feeds([])
    urls = {row["url"] for row in merged}
    assert "https://feeds.reuters.com/reuters/worldNews" in urls
    assert "https://feeds.bbci.co.uk/news/world/rss.xml" in urls


def test_merge_custom_rss_feeds_preserves_existing_rows():
    existing = [
        {
            "id": "custom_1",
            "name": "Reuters World",
            "url": "https://feeds.reuters.com/reuters/worldNews",
            "enabled": False,
            "category": "world",
        }
    ]
    merged = merge_custom_rss_feeds(existing)
    matched = [row for row in merged if row["url"] == "https://feeds.reuters.com/reuters/worldNews"]
    assert len(matched) == 1
    assert matched[0]["enabled"] is False


def test_merge_gov_rss_feeds_appends_missing_defaults_without_duplicates():
    existing = [
        {
            "id": "gov_custom_doj",
            "agency": "justice",
            "name": "DOJ News",
            "url": "https://www.justice.gov/feeds/opa/justice-news.xml",
            "priority": "high",
            "country_iso3": "USA",
            "enabled": False,
        }
    ]
    merged = merge_gov_rss_feeds(existing)
    doj = [
        row
        for row in merged
        if row["agency"] == "justice"
        and row["url"] == "https://www.justice.gov/feeds/opa/justice-news.xml"
    ]
    assert len(doj) == 1
    assert doj[0]["enabled"] is False
    # Ensure at least one different default feed was appended.
    assert any(row["agency"] == "white_house" for row in merged)
