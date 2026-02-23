import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_sdk import StrategySDK


def test_parse_duration_minutes_supports_compact_and_words():
    assert StrategySDK.parse_duration_minutes("15m") == 15
    assert StrategySDK.parse_duration_minutes("2d") == 2880
    assert StrategySDK.parse_duration_minutes("2 days") == 2880
    assert StrategySDK.parse_duration_minutes("90") == 90
    assert StrategySDK.parse_duration_minutes(45) == 45


def test_parse_duration_minutes_rejects_invalid_values():
    assert StrategySDK.parse_duration_minutes(None) is None
    assert StrategySDK.parse_duration_minutes("-2h") is None
    assert StrategySDK.parse_duration_minutes("nonsense") is None


def test_normalize_strategy_retention_config_sets_canonical_ttl():
    cfg = StrategySDK.normalize_strategy_retention_config(
        {
            "max_opportunities": "250",
            "retention_window": "2d",
        }
    )
    assert cfg["max_opportunities"] == 250
    assert cfg["retention_max_age_minutes"] == 2880
    assert cfg["retention_window"] == "2d"


def test_validate_news_filter_config_keeps_and_normalizes_retention():
    cfg = StrategySDK.validate_news_filter_config(
        {
            "retention_window": "15 min",
            "max_opportunities": "80",
        }
    )
    assert cfg["retention_max_age_minutes"] == 15
    assert cfg["max_opportunities"] == 80


def test_strategy_retention_schema_contains_expected_fields():
    schema = StrategySDK.strategy_retention_config_schema()
    keys = {field.get("key") for field in schema.get("param_fields", []) if isinstance(field, dict)}
    assert "max_opportunities" in keys
    assert "retention_window" in keys


def test_crypto_highfreq_scope_schema_contains_include_exclude_fields():
    schema = StrategySDK.crypto_highfreq_scope_config_schema()
    keys = {field.get("key") for field in schema.get("param_fields", []) if isinstance(field, dict)}
    assert "include_assets" in keys
    assert "exclude_assets" in keys
    assert "include_timeframes" in keys
    assert "exclude_timeframes" in keys
    assert "live_window_required" in keys
    assert "min_liquidity_usd" in keys
    assert "max_spread_pct" in keys
    assert "max_signal_age_seconds" in keys
    assert "max_live_context_age_seconds" in keys
    assert "max_oracle_age_seconds" in keys
    assert "require_oracle_for_directional" in keys
    assert "min_seconds_left_for_entry_5m" in keys
    assert "take_profit_pct" in keys
    assert "stop_loss_pct" in keys
    assert "stop_loss_policy" in keys
    assert "stop_loss_activation_seconds" in keys
    assert "stop_loss_activation_seconds_5m" in keys
    assert "trailing_stop_pct" in keys
    assert "trailing_stop_activation_profit_pct" in keys
    assert "trailing_stop_activation_profit_pct_5m" in keys
    assert "min_hold_minutes" in keys
    assert "max_hold_minutes" in keys
    assert "resolve_only" in keys
    assert "close_on_inactive_market" in keys
    assert "preplace_take_profit_exit" in keys
    assert "enforce_min_exit_notional" in keys


def test_crypto_highfreq_scope_defaults_include_stop_loss_policy():
    defaults = StrategySDK.crypto_highfreq_scope_defaults()
    assert defaults["stop_loss_policy"] == "near_close_only"
    assert defaults["stop_loss_activation_seconds"] == 90
    assert defaults["min_liquidity_usd"] == 250.0
    assert defaults["max_spread_pct"] == 0.08
    assert defaults["max_signal_age_seconds"] == 35.0
    assert defaults["max_live_context_age_seconds"] == 5.0
    assert defaults["max_oracle_age_seconds"] == 20.0
    assert defaults["require_oracle_for_directional"] is True
    assert defaults["min_seconds_left_for_entry_5m"] == 35.0
    assert defaults["trailing_stop_activation_profit_pct"] == 4.0
    assert defaults["trailing_stop_activation_profit_pct_5m"] == 4.0
    assert defaults["enforce_min_exit_notional"] is False


def test_strategy_entry_take_profit_exit_allowlist_and_param_resolution():
    assert StrategySDK.strategy_supports_entry_take_profit_exit("btc_eth_highfreq") is True
    assert StrategySDK.strategy_supports_entry_take_profit_exit("news_edge") is False

    assert StrategySDK.should_preplace_take_profit_exit("btc_eth_highfreq", {"preplace_take_profit_exit": True}) is True
    assert StrategySDK.should_preplace_take_profit_exit("btc_eth_highfreq", {"preplace_take_profit_exit": False}) is False
    assert (
        StrategySDK.should_preplace_take_profit_exit(
            "btc_eth_highfreq",
            {"preplace_take_profit_exit": True, "live_preplace_take_profit_exit": False},
        )
        is False
    )
    assert (
        StrategySDK.should_preplace_take_profit_exit(
            "btc_eth_highfreq",
            {"preplace_take_profit_exit": False, "live_preplace_take_profit_exit": True},
        )
        is True
    )
    assert StrategySDK.should_preplace_take_profit_exit("news_edge", {"preplace_take_profit_exit": True}) is False
