import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import UTCDateTime
from services.strategies.base import make_aware, utcnow as strategy_utcnow
from utils.utcnow import utcfromtimestamp, utcnow


def test_utc_helpers_return_timezone_aware_values():
    now = utcnow()
    epoch = utcfromtimestamp(0)
    assert now.tzinfo == timezone.utc
    assert epoch.tzinfo == timezone.utc
    assert epoch == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_make_aware_normalizes_naive_and_non_utc_datetimes():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    eastern = datetime(2026, 1, 1, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert make_aware(naive) == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert make_aware(eastern) == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_database_and_strategy_datetime_paths_are_comparable():
    codec = UTCDateTime()
    bound = codec.process_bind_param(datetime(2026, 1, 10, 15, 30, 0), dialect=None)
    assert bound.tzinfo is None

    db_value = codec.process_result_value(bound, dialect=None)
    assert db_value.tzinfo == timezone.utc

    strategy_now = strategy_utcnow()
    assert strategy_now.tzinfo == timezone.utc
    assert isinstance(strategy_now > db_value, bool)
