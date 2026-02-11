import sys
from pathlib import Path
from datetime import datetime, timezone

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.weather.contract_parser import parse_weather_contract


def test_parse_temperature_contract_with_location_and_threshold():
    parsed = parse_weather_contract(
        "Will the high in New York, NY on Jan 29 be above 18F?"
    )
    assert parsed is not None
    assert parsed.metric == "temp_threshold"
    assert parsed.operator == "gt"
    assert parsed.location == "New York, NY"
    assert parsed.raw_threshold == 18.0
    assert parsed.raw_unit == "F"
    assert parsed.threshold_c is not None


def test_parse_precipitation_contract():
    parsed = parse_weather_contract(
        "Will it rain in Seattle on February 14, 2026?"
    )
    assert parsed is not None
    assert parsed.metric == "precip_occurrence"
    assert parsed.operator == "gt"
    assert parsed.location == "Seattle"


def test_parse_unsupported_contract_returns_none():
    parsed = parse_weather_contract("Will Bitcoin be above $100k by Friday?")
    assert parsed is None


def test_parse_uses_resolution_date_when_question_has_no_date():
    resolution = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    parsed = parse_weather_contract(
        "Will the low in Chicago be below 0C?",
        resolution_date=resolution,
    )
    assert parsed is not None
    assert parsed.target_time == resolution
