import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_sdk import StrategySDK


def test_validate_trader_risk_config_normalizes_portfolio_allocator_fields():
    cfg = StrategySDK.validate_trader_risk_config(
        {
            "portfolio": {
                "enabled": "true",
                "target_utilization_pct": 150,
                "max_source_exposure_pct": -5,
                "min_order_notional_usd": "2.5",
            }
        }
    )

    portfolio = cfg["portfolio"]
    assert portfolio["enabled"] is True
    assert portfolio["target_utilization_pct"] == 100.0
    assert portfolio["max_source_exposure_pct"] == 1.0
    assert portfolio["min_order_notional_usd"] == 2.5


def test_validate_trader_filter_config_normalizes_min_order_size_fields():
    cfg = StrategySDK.validate_trader_filter_config(
        {
            "min_order_size_usd": "3.5",
            "shadow_min_order_size_usd": "2.0",
            "live_min_order_size_usd": "4.25",
        }
    )

    assert cfg["min_order_size_usd"] == 3.5
    assert cfg["shadow_min_order_size_usd"] == 2.0
    assert cfg["live_min_order_size_usd"] == 4.25


def test_validate_trader_risk_config_normalizes_new_execution_policy_fields():
    cfg = StrategySDK.validate_trader_risk_config(
        {
            "max_entry_drift_pct": "8.5",
            "max_market_data_age_ms": "20000",
            "allow_taker_limit_buy_above_signal": "true",
        }
    )

    assert cfg["max_entry_drift_pct"] == 8.5
    assert cfg["max_market_data_age_ms"] == 20000
    assert cfg["allow_taker_limit_buy_above_signal"] is True


def test_validate_trader_risk_config_clamps_max_entry_drift_and_md_age():
    cfg = StrategySDK.validate_trader_risk_config(
        {
            "max_entry_drift_pct": 999.0,
            "max_market_data_age_ms": 5,
            "allow_taker_limit_buy_above_signal": "no",
        }
    )

    assert cfg["max_entry_drift_pct"] == 100.0
    assert cfg["max_market_data_age_ms"] == 50
    assert cfg["allow_taker_limit_buy_above_signal"] is False


def test_validate_trader_risk_config_max_market_data_age_empty_means_fallback():
    cfg_none = StrategySDK.validate_trader_risk_config({"max_market_data_age_ms": None})
    cfg_empty = StrategySDK.validate_trader_risk_config({"max_market_data_age_ms": "  "})
    cfg_missing = StrategySDK.validate_trader_risk_config({})

    assert cfg_none["max_market_data_age_ms"] is None
    assert cfg_empty["max_market_data_age_ms"] is None
    assert cfg_missing["max_market_data_age_ms"] is None


def test_trader_risk_fields_schema_exposes_new_execution_policy_fields():
    schema_keys = {field["key"] for field in StrategySDK.trader_risk_fields_schema()}

    assert "max_entry_drift_pct" in schema_keys
    assert "max_market_data_age_ms" in schema_keys
    assert "allow_taker_limit_buy_above_signal" in schema_keys


def test_resolve_min_order_size_prefers_mode_specific_then_base_then_portfolio():
    assert (
        StrategySDK.resolve_min_order_size_usd(
            {"min_order_size_usd": 2.0, "live_min_order_size_usd": 5.0},
            mode="live",
            fallback=1.0,
        )
        == 5.0
    )
    assert (
        StrategySDK.resolve_min_order_size_usd(
            {"min_order_size_usd": 2.0},
            mode="live",
            fallback=1.0,
        )
        == 2.0
    )
    assert (
        StrategySDK.resolve_min_order_size_usd(
            {"portfolio": {"min_order_notional_usd": 6.0}},
            mode="live",
            fallback=1.0,
        )
        == 6.0
    )

