import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.weather.adapters.base import WeatherForecastResult
from services.weather.signal_engine import build_weather_signal


def test_signal_tradable_when_thresholds_met():
    forecast = WeatherForecastResult(gfs_probability=0.72, ecmwf_probability=0.68)
    signal = build_weather_signal(
        market_id="mkt1",
        yes_price=0.25,
        no_price=0.75,
        forecast=forecast,
        entry_max_price=0.30,
        min_edge_percent=10.0,
        min_confidence=0.60,
        min_model_agreement=0.80,
    )
    assert signal.should_trade is True
    assert signal.direction == "buy_yes"
    assert signal.edge_percent > 10.0
    assert signal.market_price <= 0.30
    assert signal.reasons == []


def test_signal_soft_caps_entry_price_without_auto_reject():
    forecast = WeatherForecastResult(gfs_probability=0.80, ecmwf_probability=0.82)
    signal = build_weather_signal(
        market_id="mkt2",
        yes_price=0.42,
        no_price=0.58,
        forecast=forecast,
        entry_max_price=0.25,
        min_edge_percent=5.0,
        min_confidence=0.30,
        min_model_agreement=0.50,
    )
    assert signal.should_trade is True
    assert signal.reasons == []
    assert any("entry_price" in n for n in signal.notes)


def test_signal_rejected_when_entry_price_exceeds_hard_cap():
    forecast = WeatherForecastResult(gfs_probability=0.90, ecmwf_probability=0.88)
    signal = build_weather_signal(
        market_id="mkt2b",
        yes_price=0.78,
        no_price=0.22,
        forecast=forecast,
        entry_max_price=0.25,
        min_edge_percent=5.0,
        min_confidence=0.20,
        min_model_agreement=0.50,
    )
    assert signal.should_trade is False
    assert any("hard cap" in r for r in signal.reasons)


def test_signal_rejected_when_model_agreement_too_low():
    forecast = WeatherForecastResult(gfs_probability=0.95, ecmwf_probability=0.55)
    signal = build_weather_signal(
        market_id="mkt3",
        yes_price=0.20,
        no_price=0.80,
        forecast=forecast,
        entry_max_price=0.30,
        min_edge_percent=1.0,
        min_confidence=0.10,
        min_model_agreement=0.85,
    )
    assert signal.should_trade is False
    assert any("agreement" in r for r in signal.reasons)


def test_signal_uses_weighted_multi_source_probabilities():
    forecast = WeatherForecastResult(
        gfs_probability=0.52,
        ecmwf_probability=0.51,
        metadata={
            "source_probabilities": {
                "open_meteo:gfs_seamless": 0.90,
                "open_meteo:ecmwf_ifs04": 0.10,
            },
            "source_weights": {
                "open_meteo:gfs_seamless": 0.80,
                "open_meteo:ecmwf_ifs04": 0.20,
            },
        },
    )
    signal = build_weather_signal(
        market_id="mkt4",
        yes_price=0.60,
        no_price=0.40,
        forecast=forecast,
        entry_max_price=0.95,
        min_edge_percent=0.0,
        min_confidence=0.0,
        min_model_agreement=0.0,
    )
    assert signal.direction == "buy_yes"
    assert signal.model_probability > 0.70
    assert signal.source_count == 2


def test_market_implied_temp_is_estimated_for_threshold_contracts():
    forecast = WeatherForecastResult(gfs_probability=0.70, ecmwf_probability=0.68)
    signal = build_weather_signal(
        market_id="mkt5",
        yes_price=0.73,
        no_price=0.27,
        forecast=forecast,
        entry_max_price=0.95,
        min_edge_percent=0.0,
        min_confidence=0.0,
        min_model_agreement=0.0,
        operator="gt",
        threshold_c=10.0,
    )
    assert signal.market_implied_temperature_c is not None
    assert signal.market_implied_temperature_c > 10.0


def test_signal_fallback_without_source_data_is_neutral_and_report_only():
    forecast = WeatherForecastResult(
        gfs_probability=0.5,
        ecmwf_probability=0.5,
        metadata={"fallback": True},
    )
    signal = build_weather_signal(
        market_id="mkt6",
        yes_price=0.01,
        no_price=0.99,
        forecast=forecast,
        entry_max_price=0.95,
        min_edge_percent=0.0,
        min_confidence=0.0,
        min_model_agreement=0.0,
    )
    assert signal.source_count == 0
    assert signal.should_trade is False
    assert signal.edge_percent == 0.0
    assert signal.model_probability == 0.99
    assert any("forecast unavailable" in r for r in signal.reasons)
