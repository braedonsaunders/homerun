import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.world_intelligence.chokepoint_reference_source import _clean_chokepoint_rows


def test_clean_chokepoint_rows_filters_invalid_and_dedupes():
    rows = [
        {
            "id": "suez_canal",
            "name": "Suez",
            "latitude": "30.4",
            "longitude": "32.3",
            "source": "imf_portwatch",
            "last_updated": datetime(2026, 2, 13, 10, 0, 0),
        },
        {
            "id": "suez_canal",
            "name": "Duplicate",
            "latitude": "30.4",
            "longitude": "32.3",
        },
        {
            "id": "bad_missing_coord",
            "name": "Bad",
            "latitude": "30.0",
        },
    ]

    cleaned = _clean_chokepoint_rows(rows)
    assert len(cleaned) == 1

    item = cleaned[0]
    assert item["id"] == "suez_canal"
    assert item["latitude"] == 30.4
    assert item["longitude"] == 32.3
    assert item["source"] == "imf_portwatch"
    assert "2026-02-13T10:00:00" in str(item["last_updated"])
