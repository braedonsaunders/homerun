import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.news.feed_service import (
    NewsFeedService,
    _clean_summary_text,
    _has_min_text_quality,
)


def test_google_description_is_normalized_for_summary():
    raw_description = "<p><b>Breaking:</b> Candidate gains momentum in latest poll.</p>"
    summary = _clean_summary_text(raw_description, max_len=500)
    assert summary == "Breaking: Candidate gains momentum in latest poll."


def test_gdelt_summary_ignores_url_like_snippet():
    service = NewsFeedService()
    raw = {
        "snippet": "https://cdn.example.com/image.jpg",
        "description": "Analysts expect tighter governor primary race after new filing.",
    }
    summary = service._pick_gdelt_summary(raw)
    assert summary == "Analysts expect tighter governor primary race after new filing."
    assert service._ingest_stats["gdelt_summary_url_filtered"] == 1


def test_min_text_quality_rejects_short_title_without_summary():
    assert _has_min_text_quality("Quick update", "") is False
    assert _has_min_text_quality(
        "South Carolina gubernatorial primary shifts after endorsement news",
        "A longer summary gives enough content for meaningful matching.",
    ) is True
