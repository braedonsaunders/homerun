import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.weather.adapters import open_meteo


def _period(start_time: str, temp: float, unit: str = "F") -> dict:
    return {
        "startTime": start_time,
        "temperature": temp,
        "temperatureUnit": unit,
    }


def test_select_nws_temperature_max_uses_daily_peak_not_nearest_hour():
    target = datetime(2026, 2, 13, 0, 0, tzinfo=timezone.utc)
    periods = [
        _period("2026-02-13T00:00:00Z", 31.0),
        _period("2026-02-13T18:00:00Z", 52.0),
        _period("2026-02-14T00:00:00Z", 28.0),
    ]

    out = open_meteo._select_nws_temperature_c(
        periods=periods,
        target_time=target,
        metric="temp_max_threshold",
    )

    assert out == pytest.approx((52.0 - 32.0) * (5.0 / 9.0))


def test_select_nws_temperature_min_uses_daily_low():
    target = datetime(2026, 2, 13, 12, 0, tzinfo=timezone.utc)
    periods = [
        _period("2026-02-13T01:00:00Z", 41.0),
        _period("2026-02-13T10:00:00Z", 26.0),
        _period("2026-02-13T18:00:00Z", 45.0),
    ]

    out = open_meteo._select_nws_temperature_c(
        periods=periods,
        target_time=target,
        metric="temp_min_threshold",
    )

    assert out == pytest.approx((26.0 - 32.0) * (5.0 / 9.0))


def test_select_nws_temperature_default_uses_nearest_hour():
    target = datetime(2026, 2, 13, 9, 0, tzinfo=timezone.utc)
    periods = [
        _period("2026-02-13T03:00:00Z", 20.0),
        _period("2026-02-13T09:30:00Z", 40.0),
        _period("2026-02-13T18:00:00Z", 60.0),
    ]

    out = open_meteo._select_nws_temperature_c(
        periods=periods,
        target_time=target,
        metric="temp_threshold",
    )

    assert out == pytest.approx((40.0 - 32.0) * (5.0 / 9.0))
