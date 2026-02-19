import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_events as routes


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


class _FakeScalarExecuteResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeAllExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _FakeOneExecuteResult:
    def __init__(self, row):
        self._row = row

    def one(self):
        return self._row


class _FakeSummarySession:
    def __init__(self, execute_results):
        self._execute_results = list(execute_results)

    async def execute(self, _query):
        if not self._execute_results:
            raise AssertionError("unexpected execute call")
        return self._execute_results.pop(0)


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


@pytest.mark.asyncio
async def test_world_summary_uses_true_window_totals(monkeypatch):
    session = _FakeSummarySession(
        [
            _FakeScalarExecuteResult(6200),  # total_signals
            _FakeAllExecuteResult(
                [
                    ("conflict", 3800),
                    ("tension", 1200),
                    ("anomaly", 800),
                    ("convergence", 400),
                ]
            ),
            _FakeOneExecuteResult(
                SimpleNamespace(
                    low=900,
                    medium=2100,
                    high=2300,
                    critical=900,
                )
            ),
            _FakeScalarExecuteResult(250),  # critical_anomalies
            _FakeScalarExecuteResult(400),  # active_convergences
        ]
    )

    async def _fake_latest_instability(_session):
        return {}

    async def _fake_latest_tensions(_session):
        return {}

    async def _fake_snapshot(_session):
        return None

    async def _fake_worker_snapshot(_session, _name):
        return {"last_run_at": "2026-02-18T00:00:00Z"}

    monkeypatch.setattr(routes, "_latest_instability_by_country", _fake_latest_instability)
    monkeypatch.setattr(routes, "_latest_tension_pairs", _fake_latest_tensions)
    monkeypatch.setattr(routes, "_get_world_snapshot", _fake_snapshot)
    monkeypatch.setattr(routes, "read_worker_snapshot", _fake_worker_snapshot)

    summary = await routes.get_events_summary(session=session)

    assert int(summary["signal_summary"]["total"]) == 6200
    assert int(summary["signal_summary"]["critical"]) == 900
    assert int(summary["signal_summary"]["high"]) == 2300
    assert int(summary["signal_summary"]["medium"]) == 2100
    assert int(summary["signal_summary"]["low"]) == 900
    assert int(summary["critical_anomalies"]) == 250
    assert int(summary["active_convergences"]) == 400
