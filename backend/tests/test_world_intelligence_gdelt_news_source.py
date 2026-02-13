import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.world_intelligence.gdelt_news_source import (
    normalize_world_intel_gdelt_queries,
    _priority_from_tone,
)


def test_normalize_world_intel_gdelt_queries_dedupes_and_sanitizes():
    rows = [
        {
            "name": "Conflict",
            "query": "war OR sanctions",
            "priority": "HIGH",
            "country_iso3": "us",
            "enabled": True,
        },
        {
            "name": "Duplicate Different Case",
            "query": "War OR Sanctions",
            "priority": "low",
            "country_iso3": "",
            "enabled": True,
        },
        {
            "name": "Invalid Priority",
            "query": "energy outage",
            "priority": "not_valid",
            "country_iso3": "france",
            "enabled": 1,
        },
    ]

    parsed = normalize_world_intel_gdelt_queries(rows)
    assert len(parsed) == 2

    first = parsed[0]
    assert first["priority"] == "high"
    assert first["country_iso3"] == "USA"

    second = parsed[1]
    assert second["priority"] == "medium"
    assert second["country_iso3"] == "FRA"


def test_priority_from_tone_escalates_negative_tone():
    assert _priority_from_tone("medium", -1.0) == "medium"
    assert _priority_from_tone("low", -5.0) == "high"
    assert _priority_from_tone("high", -9.0) == "critical"

