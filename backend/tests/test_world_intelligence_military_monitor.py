import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.world_intelligence.military_monitor import MilitaryActivity, MilitaryMonitor


def _flight(
    *,
    provider: str,
    transponder: str,
    callsign: str = "RCH123",
    lat: float = 10.0,
    lon: float = 20.0,
) -> MilitaryActivity:
    return MilitaryActivity(
        activity_type="flight",
        callsign=callsign,
        country="USA",
        latitude=lat,
        longitude=lon,
        altitude=9000.0,
        heading=45.0,
        speed=210.0,
        aircraft_type="transport",
        timestamp=datetime.now(timezone.utc),
        region="other",
        provider=provider,
        transponder=transponder,
        providers=[provider],
    )


def test_fetch_military_flights_dedupes_cross_provider(monkeypatch):
    monitor = MilitaryMonitor()
    opensky = [_flight(provider="opensky", transponder="abc123", lat=12.0, lon=22.0)]
    airplanes = [_flight(provider="airplanes_live", transponder="abc123", lat=12.1, lon=22.1)]

    async def _fake_collect_opensky(region):
        assert region is None
        return opensky, 900

    async def _fake_collect_airplanes(region):
        assert region is None
        return airplanes

    monkeypatch.setattr(monitor, "_collect_opensky_flights", _fake_collect_opensky)
    monkeypatch.setattr(monitor, "_fetch_airplanes_live_flights", _fake_collect_airplanes)
    monkeypatch.setattr(monitor, "get_surge_regions", lambda: [])

    flights = asyncio.run(monitor.fetch_military_flights())
    assert len(flights) == 1
    assert flights[0].transponder == "abc123"
    assert sorted(flights[0].providers) == ["airplanes_live", "opensky"]

    health = monitor.get_health()
    assert health["last_total_states_seen"] == 900
    assert health["last_identified_flights"] == 1
    assert health["last_deduped_flights"] == 1
    assert health["last_identified_flights_by_provider"] == {
        "opensky": 1,
        "airplanes_live": 1,
    }


def test_parse_airplanes_live_record_normalizes_units():
    monitor = MilitaryMonitor()
    activity = monitor._parse_airplanes_live_record(
        {
            "hex": "aabbcc",
            "flight": "RCH456 ",
            "lat": 35.2,
            "lon": -77.8,
            "alt_baro": 10000,  # feet
            "gs": 200,  # knots
            "track": 120.0,
            "t": "C17",
            "desc": "Boeing C-17A Globemaster III",
        }
    )
    assert activity is not None
    assert activity.provider == "airplanes_live"
    assert activity.transponder == "aabbcc"
    assert activity.altitude == pytest.approx(3048.0, rel=1e-3)
    assert activity.speed == pytest.approx(102.8888, rel=1e-3)
    assert activity.aircraft_type == "C17"
