import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.data_events import DataEvent, EventType


def test_data_event_rejects_unknown_event_type():
    with pytest.raises(ValueError):
        DataEvent(
            event_type="unknown_event_type",
            source="unit_test",
            timestamp=datetime.now(timezone.utc),
        )


def test_data_event_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        DataEvent(
            event_type=EventType.PRICE_CHANGE,
            source="unit_test",
            timestamp=datetime.utcnow(),
        )


def test_data_event_normalizes_timestamp_to_utc():
    eastern = timezone(timedelta(hours=-5))
    event = DataEvent(
        event_type=EventType.MARKET_DATA_REFRESH,
        source="unit_test",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=eastern),
    )
    assert event.timestamp.tzinfo == timezone.utc
    assert event.timestamp.hour == 17
