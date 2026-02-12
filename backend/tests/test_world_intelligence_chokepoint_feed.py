import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.world_intelligence.chokepoint_feed import ChokepointFeed


def test_refresh_merges_points_and_daily_metrics(monkeypatch):
    feed = ChokepointFeed()

    async def _fake_points():
        return [
            {
                "id": "chokepoint1",
                "portid": "chokepoint1",
                "name": "Suez Canal",
                "latitude": 30.59,
                "longitude": 32.43,
                "source": "imf_portwatch",
                "baseline_vessel_count_total": 100,
            },
            {
                "id": "chokepoint2",
                "portid": "chokepoint2",
                "name": "Panama Canal",
                "latitude": 9.12,
                "longitude": -79.76,
                "source": "imf_portwatch",
                "baseline_vessel_count_total": 80,
            },
        ]

    async def _fake_daily():
        return (
            {
                "chokepoint1": {
                    "daily_transit_total": 55,
                    "daily_capacity_estimate": 1100,
                    "daily_metrics_date": "2026-02-11T00:00:00Z",
                },
                "chokepoint2": {
                    "daily_transit_total": 25,
                    "daily_capacity_estimate": 700,
                    "daily_metrics_date": "2026-02-11T00:00:00Z",
                },
            },
            datetime(2026, 2, 11, tzinfo=timezone.utc),
        )

    monkeypatch.setattr(feed, "_fetch_portwatch_points", _fake_points)
    monkeypatch.setattr(feed, "_fetch_portwatch_daily_metrics", _fake_daily)

    rows = asyncio.run(feed.refresh(force=True))
    assert len(rows) == 2
    assert rows[0]["id"] == "chokepoint1"
    assert rows[0]["daily_transit_total"] == 55
    assert rows[0]["daily_dataset_updated_at"] == "2026-02-11T00:00:00Z"

    health = feed.get_health()
    assert health["ok"] is True
    assert health["source"] == "imf_portwatch"
    assert health["count"] == 2


def test_refresh_falls_back_to_static_when_remote_fails(monkeypatch):
    feed = ChokepointFeed()

    async def _fail_points():
        raise RuntimeError("portwatch unavailable")

    async def _fail_daily():
        raise RuntimeError("daily unavailable")

    monkeypatch.setattr(feed, "_fetch_portwatch_points", _fail_points)
    monkeypatch.setattr(feed, "_fetch_portwatch_daily_metrics", _fail_daily)
    monkeypatch.setattr(
        feed,
        "_load_static_fallback",
        lambda: [
            {
                "id": "fallback_1",
                "name": "Fallback",
                "latitude": 0.0,
                "longitude": 0.0,
                "source": "static_catalog",
            }
        ],
    )

    rows = asyncio.run(feed.refresh(force=True))
    assert len(rows) == 1
    assert rows[0]["id"] == "fallback_1"
    assert rows[0]["source"] == "static_catalog"

    health = feed.get_health()
    assert health["source"] == "static_fallback"
    assert "portwatch unavailable" in str(health["last_error"])
