import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_world_intelligence as routes


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)

    async def execute(self, _query):
        if not self._responses:
            raise AssertionError("unexpected execute call")
        return _FakeResult(self._responses.pop(0))


@pytest.mark.asyncio
async def test_latest_instability_by_country_dedupes_aliases():
    now = datetime(2026, 2, 12, 5, 0, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(
            iso3="USA",
            country="USA",
            score=8.3,
            trend="stable",
            computed_at=now,
            components={},
        ),
        SimpleNamespace(
            iso3="US",
            country="United States",
            score=5.5,
            trend="falling",
            computed_at=now - timedelta(hours=2),
            components={},
        ),
        SimpleNamespace(
            iso3="FRA",
            country="France",
            score=31.2,
            trend="rising",
            computed_at=now - timedelta(hours=1),
            components={},
        ),
    ]
    session = _FakeSession([rows])

    latest = await routes._latest_instability_by_country(session)

    assert set(latest.keys()) == {"USA", "FRA"}
    usa_current, usa_prev = latest["USA"]
    assert usa_current.iso3 == "USA"
    assert usa_prev is not None
    assert float(usa_prev.score) == pytest.approx(5.5)


@pytest.mark.asyncio
async def test_latest_tension_pairs_dedupes_normalized_pairs():
    now = datetime(2026, 2, 12, 5, 0, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(
            country_a="US",
            country_b="CN",
            tension_score=61.2,
            event_count=12,
            avg_goldstein_scale=-4.2,
            trend="rising",
            computed_at=now,
        ),
        SimpleNamespace(
            country_a="USA",
            country_b="CHN",
            tension_score=47.0,
            event_count=8,
            avg_goldstein_scale=-3.7,
            trend="stable",
            computed_at=now - timedelta(hours=1),
        ),
        SimpleNamespace(
            country_a="RU",
            country_b="UA",
            tension_score=70.0,
            event_count=15,
            avg_goldstein_scale=-5.1,
            trend="rising",
            computed_at=now - timedelta(minutes=30),
        ),
    ]
    session = _FakeSession([rows])

    latest = await routes._latest_tension_pairs(session)

    assert set(latest.keys()) == {"CHN-USA", "RUS-UKR"}
    current, prev = latest["CHN-USA"]
    assert current.country_a == "US"
    assert current.country_b == "CN"
    assert prev is not None
    assert float(prev.tension_score) == pytest.approx(47.0)
