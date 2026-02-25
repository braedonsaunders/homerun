import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_sdk import StrategySDK
from services.strategies.btc_eth_highfreq import (
    crypto_highfreq_direction_allowed,
    crypto_highfreq_should_flatten_resolution_risk,
)


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
    assert "min_edge_percent" in keys
    assert "min_confidence" in keys
    assert "base_size_usd" in keys
    assert "max_size_usd" in keys
    assert "include_assets" in keys
    assert "exclude_assets" in keys
    assert "include_timeframes" in keys
    assert "exclude_timeframes" in keys
    assert "live_window_required" in keys
    assert "min_liquidity_usd" in keys
    assert "min_liquidity_usd_opening" in keys
    assert "opening_directional_buy_yes_enabled" in keys
    assert "opening_directional_buy_yes_block_elapsed_pct" in keys
    assert "opening_directional_buy_yes_block_elapsed_pct_5m" in keys
    assert "opening_directional_buy_yes_block_elapsed_pct_15m" in keys
    assert "opening_directional_buy_yes_block_elapsed_pct_1h" in keys
    assert "opening_directional_buy_yes_block_elapsed_pct_4h" in keys
    assert "entry_executable_exit_ratio_floor" in keys
    assert "max_spread_pct" in keys
    assert "max_signal_age_seconds" in keys
    assert "max_open_order_seconds" in keys
    assert "max_live_context_age_seconds" in keys
    assert "max_oracle_age_seconds" in keys
    assert "max_oracle_age_ms" in keys
    assert "require_oracle_for_directional" in keys
    assert "oracle_source_policy" in keys
    assert "oracle_fallback_degrade_edge_multiplier" in keys
    assert "oracle_fallback_degrade_confidence_multiplier" in keys
    assert "oracle_fallback_degrade_size_multiplier" in keys
    assert "min_seconds_left_for_entry_5m" in keys
    assert "rapid_take_profit_pct" in keys
    assert "rapid_take_profit_pct_5m" in keys
    assert "rapid_take_profit_pct_15m" in keys
    assert "rapid_take_profit_pct_1h" in keys
    assert "rapid_take_profit_pct_4h" in keys
    assert "rapid_exit_window_minutes" in keys
    assert "rapid_exit_window_minutes_5m" in keys
    assert "rapid_exit_window_minutes_15m" in keys
    assert "rapid_exit_window_minutes_1h" in keys
    assert "rapid_exit_window_minutes_4h" in keys
    assert "rapid_exit_min_increase_pct" in keys
    assert "rapid_exit_breakeven_buffer_pct" in keys
    assert "reverse_on_adverse_velocity_enabled" in keys
    assert "reverse_min_loss_pct" in keys
    assert "reverse_min_adverse_velocity_score" in keys
    assert "reverse_flow_imbalance_threshold" in keys
    assert "reverse_momentum_short_pct_threshold" in keys
    assert "reverse_min_seconds_left" in keys
    assert "reverse_min_price_headroom" in keys
    assert "reverse_min_edge_percent" in keys
    assert "reverse_confidence" in keys
    assert "reverse_size_multiplier" in keys
    assert "reverse_signal_ttl_seconds" in keys
    assert "reverse_cooldown_seconds" in keys
    assert "reverse_max_reentries_per_position" in keys
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
    assert "force_flatten_seconds_left" in keys
    assert "force_flatten_seconds_left_5m" in keys
    assert "force_flatten_max_profit_pct" in keys
    assert "force_flatten_headroom_floor" in keys
    assert "resolution_risk_flatten_enabled" in keys
    assert "resolution_risk_seconds_left" in keys
    assert "resolution_risk_max_profit_pct" in keys
    assert "resolution_risk_min_loss_pct" in keys
    assert "resolution_risk_min_headroom_ratio" in keys
    assert "resolve_only" in keys
    assert "close_on_inactive_market" in keys
    assert "preplace_take_profit_exit" in keys
    assert "enforce_min_exit_notional" in keys


def test_crypto_highfreq_scope_defaults_include_stop_loss_policy():
    defaults = StrategySDK.crypto_highfreq_scope_defaults()
    assert defaults["stop_loss_policy"] == "near_close_only"
    assert defaults["stop_loss_activation_seconds"] == 90
    assert defaults["min_liquidity_usd"] == 250.0
    assert defaults["min_liquidity_usd_opening"] == 4000.0
    assert defaults["max_spread_pct"] == 0.08
    assert defaults["max_signal_age_seconds"] == 35.0
    assert defaults["max_open_order_seconds"] == 14.0
    assert defaults["max_live_context_age_seconds"] == 3.0
    assert defaults["max_oracle_age_seconds"] == 20.0
    assert defaults["max_oracle_age_ms"] == 20000.0
    assert defaults["require_oracle_for_directional"] is True
    assert defaults["oracle_source_policy"] == "degrade"
    assert defaults["min_seconds_left_for_entry_5m"] == 60.0
    assert defaults["min_seconds_left_for_entry_15m"] == 180.0
    assert defaults["min_seconds_left_for_entry_1h"] == 360.0
    assert defaults["opening_directional_buy_yes_enabled"] is False
    assert defaults["opening_directional_buy_yes_block_elapsed_pct"] == 0.10
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_5m"] == 0.45
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_15m"] == 0.25
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_1h"] == 0.10
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_4h"] == 0.05
    assert defaults["entry_executable_exit_ratio_floor"] == 0.28
    assert defaults["rapid_take_profit_pct"] == 10.0
    assert defaults["rapid_take_profit_pct_5m"] == 10.0
    assert defaults["rapid_take_profit_pct_15m"] == 10.0
    assert defaults["rapid_take_profit_pct_1h"] == 12.0
    assert defaults["rapid_take_profit_pct_4h"] == 15.0
    assert defaults["rapid_exit_window_minutes"] == 2.0
    assert defaults["rapid_exit_window_minutes_5m"] == 1.0
    assert defaults["rapid_exit_window_minutes_15m"] == 2.0
    assert defaults["rapid_exit_window_minutes_1h"] == 6.0
    assert defaults["rapid_exit_window_minutes_4h"] == 15.0
    assert defaults["rapid_exit_min_increase_pct"] == 0.0
    assert defaults["rapid_exit_breakeven_buffer_pct"] == 0.0
    assert defaults["reverse_on_adverse_velocity_enabled"] is False
    assert defaults["reverse_min_loss_pct"] == 2.0
    assert defaults["reverse_min_adverse_velocity_score"] == 0.55
    assert defaults["reverse_min_seconds_left"] == 90.0
    assert defaults["reverse_signal_ttl_seconds"] == 45.0
    assert defaults["reverse_max_reentries_per_position"] == 1
    assert defaults["trailing_stop_activation_profit_pct"] == 4.0
    assert defaults["trailing_stop_activation_profit_pct_5m"] == 4.0
    assert defaults["force_flatten_seconds_left_5m"] == 90.0
    assert defaults["force_flatten_max_profit_pct"] == 3.0
    assert defaults["resolution_risk_flatten_enabled"] is True
    assert defaults["resolution_risk_seconds_left_5m"] == 105.0
    assert defaults["resolution_risk_max_profit_pct"] == 6.0
    assert defaults["enforce_min_exit_notional"] is True


def test_strategy_entry_take_profit_exit_param_resolution():
    assert StrategySDK.should_preplace_take_profit_exit({"preplace_take_profit_exit": True}) is True
    assert StrategySDK.should_preplace_take_profit_exit({"preplace_take_profit_exit": False}) is False
    assert (
        StrategySDK.should_preplace_take_profit_exit(
            {"preplace_take_profit_exit": True, "live_preplace_take_profit_exit": False}
        )
        is False
    )
    assert (
        StrategySDK.should_preplace_take_profit_exit(
            {"preplace_take_profit_exit": False, "live_preplace_take_profit_exit": True}
        )
        is True
    )
    assert StrategySDK.should_preplace_take_profit_exit({}, default_enabled=False) is False
    assert StrategySDK.should_preplace_take_profit_exit({}, default_enabled=True) is True


def test_reverse_intent_helpers_normalize_and_build_payload():
    assert StrategySDK.opposite_direction("buy_yes") == "buy_no"
    assert StrategySDK.opposite_direction("buy_no") == "buy_yes"

    raw = StrategySDK.normalize_reverse_intent(
        {
            "enabled": True,
            "direction": "buy_no",
            "entry_price": 0.73,
            "size_multiplier": 1.2,
            "confidence": 0.67,
            "edge_percent": 3.4,
            "signal_type": "crypto_worker_reverse",
            "max_reentries_per_position": 2,
        },
        fallback_direction="buy_yes",
    )
    assert isinstance(raw, dict)
    assert raw["direction"] == "buy_no"
    assert raw["entry_price"] == 0.73
    assert raw["size_multiplier"] == 1.2
    assert raw["signal_type"] == "crypto_worker_reverse"
    assert raw["max_reentries_per_position"] == 2

    built = StrategySDK.build_reverse_intent(
        direction="buy_yes",
        entry_price=0.41,
        confidence=0.66,
        edge_percent=2.1,
        size_multiplier=1.15,
        signal_type="crypto_worker_reverse",
    )
    assert built["direction"] == "buy_yes"
    assert built["entry_price"] == 0.41
    assert built["confidence"] == 0.66
    assert built["edge_percent"] == 2.1
    assert built["size_multiplier"] == 1.15


def test_resolve_open_order_timeout_seconds_prefers_strategy_param():
    timeout = StrategySDK.resolve_open_order_timeout_seconds(
        {
            "max_open_order_seconds": 20,
            "order_ttl_seconds": 1200,
        }
    )
    assert timeout == 20.0


def test_resolve_open_order_timeout_seconds_supports_alias_and_default_fallback():
    timeout = StrategySDK.resolve_open_order_timeout_seconds(
        {"open_order_timeout_seconds": "45"},
    )
    assert timeout == 45.0

    fallback = StrategySDK.resolve_open_order_timeout_seconds(
        {"max_open_order_seconds": "invalid"},
        default_seconds=1200.0,
    )
    assert fallback == 1200.0


def test_crypto_highfreq_direction_policy_blocks_opening_directional_buy_yes_by_default():
    allowed, detail = crypto_highfreq_direction_allowed(
        {},
        regime="opening",
        active_mode="directional",
        direction="buy_yes",
    )
    assert allowed is False
    assert "opening_directional_buy_yes_enabled=False" in detail

    allowed_no, detail_no = crypto_highfreq_direction_allowed(
        {},
        regime="opening",
        active_mode="directional",
        direction="buy_no",
    )
    assert allowed_no is True
    assert "opening_directional_buy_no_enabled=True" in detail_no


def test_crypto_highfreq_direction_policy_uses_elapsed_progress_window_by_timeframe():
    blocked_1h, blocked_detail_1h = crypto_highfreq_direction_allowed(
        {},
        regime="opening",
        active_mode="directional",
        direction="buy_yes",
        timeframe="1h",
        seconds_left=3300.0,
    )
    assert blocked_1h is False
    assert "min_elapsed=0.100" in blocked_detail_1h
    assert "timeframe=1h" in blocked_detail_1h

    allowed_1h, allowed_detail_1h = crypto_highfreq_direction_allowed(
        {},
        regime="opening",
        active_mode="directional",
        direction="buy_yes",
        timeframe="1h",
        seconds_left=3000.0,
    )
    assert allowed_1h is True
    assert "min_elapsed=0.100" in allowed_detail_1h
    assert "timeframe=1h" in allowed_detail_1h

    blocked_15m, blocked_detail_15m = crypto_highfreq_direction_allowed(
        {},
        regime="opening",
        active_mode="directional",
        direction="buy_yes",
        timeframe="15m",
        seconds_left=700.0,
    )
    assert blocked_15m is False
    assert "min_elapsed=0.250" in blocked_detail_15m
    assert "timeframe=15m" in blocked_detail_15m

    blocked_5m_mid, blocked_detail_5m_mid = crypto_highfreq_direction_allowed(
        {},
        regime="mid",
        active_mode="directional",
        direction="buy_yes",
        timeframe="5m",
        seconds_left=180.0,
    )
    assert blocked_5m_mid is False
    assert "min_elapsed=0.450" in blocked_detail_5m_mid
    assert "timeframe=5m" in blocked_detail_5m_mid

    allowed_5m_mid, allowed_detail_5m_mid = crypto_highfreq_direction_allowed(
        {},
        regime="mid",
        active_mode="directional",
        direction="buy_yes",
        timeframe="5m",
        seconds_left=150.0,
    )
    assert allowed_5m_mid is True
    assert "min_elapsed=0.450" in allowed_detail_5m_mid
    assert "timeframe=5m" in allowed_detail_5m_mid


def test_crypto_highfreq_resolution_risk_flatten_policy():
    should_flatten, detail = crypto_highfreq_should_flatten_resolution_risk(
        {
            "resolution_risk_flatten_enabled": True,
            "resolution_risk_seconds_left_5m": 105,
            "resolution_risk_max_profit_pct_5m": 4,
            "resolution_risk_min_loss_pct": 2,
            "resolution_risk_min_headroom_ratio": 0.9,
        },
        timeframe="5m",
        seconds_left=90.0,
        pnl_percent=1.2,
        exit_headroom_ratio=1.1,
        take_profit_armed=False,
    )
    assert should_flatten is True
    assert "seconds_left=90.0s" in detail

    should_skip_armed, detail_armed = crypto_highfreq_should_flatten_resolution_risk(
        {},
        timeframe="5m",
        seconds_left=90.0,
        pnl_percent=1.2,
        exit_headroom_ratio=1.1,
        take_profit_armed=True,
    )
    assert should_skip_armed is False
    assert detail_armed == "take_profit_armed"
