import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.market import Market
from services.strategy_sdk import StrategySDK
from services.strategies.btc_eth_highfreq import (
    crypto_highfreq_direction_allowed,
    crypto_highfreq_scope_config_schema,
    crypto_highfreq_scope_defaults,
    crypto_highfreq_should_flatten_resolution_risk,
)
from services.strategies.late_favorite_alpha import (
    LateFavoriteAlphaStrategy,
    late_favorite_alpha_config_schema,
    late_favorite_alpha_defaults,
    validate_late_favorite_alpha_config,
)
from services.strategies.news_edge import validate_news_edge_config
from services.strategies.traders_copy_trade import (
    traders_copy_trade_defaults,
    validate_traders_copy_trade_config,
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
    cfg = validate_news_edge_config(
        {
            "retention_window": "15 min",
            "max_opportunities": "80",
        }
    )
    assert cfg["retention_max_age_minutes"] == 15
    assert cfg["max_opportunities"] == 80


def test_traders_copy_trade_defaults_and_validation_delegate_to_strategy_module():
    defaults = traders_copy_trade_defaults()
    assert defaults["min_confidence"] == 0.45
    assert defaults["max_signal_age_seconds"] == 5
    assert defaults["copy_buys"] is True
    assert defaults["default_leader_weight"] == 1.0
    assert defaults["copy_existing_positions_on_start"] is False
    assert defaults["require_inventory_for_sells"] is True

    validated = validate_traders_copy_trade_config(
        {
            "min_confidence": "0.7",
            "max_signal_age_seconds": "99",
            "base_size_usd": "100",
            "max_size_usd": "10",
            "retention_window": "2h",
        }
    )
    assert validated["min_confidence"] == 0.7
    assert validated["max_signal_age_seconds"] == 5
    assert validated["max_size_usd"] >= validated["base_size_usd"]
    assert validated["retention_max_age_minutes"] == 120


def test_late_favorite_alpha_defaults_and_schema_are_exposed():
    defaults = late_favorite_alpha_defaults()
    assert defaults["min_favorite_prob"] == 0.58
    assert defaults["max_favorite_prob"] == 0.88
    assert defaults["max_hours_to_resolution"] == 6.0
    assert defaults["min_alpha_prob"] == 0.014

    schema = late_favorite_alpha_config_schema()
    keys = {field.get("key") for field in schema.get("param_fields", []) if isinstance(field, dict)}
    assert "min_favorite_prob" in keys
    assert "max_favorite_prob" in keys
    assert "min_hours_to_resolution" in keys
    assert "max_hours_to_resolution" in keys
    assert "min_alpha_prob" in keys
    assert "max_target_exit_price" in keys


def test_validate_late_favorite_alpha_config_clamps_ranges_and_retention():
    cfg = validate_late_favorite_alpha_config(
        {
            "min_favorite_prob": "0.74",
            "max_favorite_prob": "0.70",
            "min_hours_to_resolution": "4",
            "max_hours_to_resolution": "2",
            "base_size_usd": "200",
            "max_size_usd": "100",
            "retention_window": "90m",
            "max_opportunities": "120",
        }
    )
    assert cfg["min_favorite_prob"] == 0.74
    assert cfg["max_favorite_prob"] > cfg["min_favorite_prob"]
    assert cfg["max_hours_to_resolution"] > cfg["min_hours_to_resolution"]
    assert cfg["max_size_usd"] >= cfg["base_size_usd"]
    assert cfg["retention_max_age_minutes"] == 90
    assert cfg["max_opportunities"] == 120


def test_strategy_retention_schema_contains_expected_fields():
    schema = StrategySDK.strategy_retention_config_schema()
    keys = {field.get("key") for field in schema.get("param_fields", []) if isinstance(field, dict)}
    assert "max_opportunities" in keys
    assert "retention_window" in keys


def test_crypto_highfreq_scope_schema_contains_include_exclude_fields():
    schema = crypto_highfreq_scope_config_schema()
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
    defaults = crypto_highfreq_scope_defaults()
    assert defaults["stop_loss_policy"] == "always"
    assert defaults["stop_loss_activation_seconds"] == 90
    assert defaults["min_liquidity_usd"] == 250.0
    assert defaults["min_liquidity_usd_opening"] == 4000.0
    assert defaults["max_spread_pct"] == 0.08
    assert defaults["max_signal_age_seconds"] == 35.0
    assert defaults["max_open_order_seconds"] == 45.0
    assert defaults["enable_live_market_context"] is True
    assert defaults["require_live_market_revalidation"] is True
    assert defaults["require_live_revalidation_for_sources"] == ["crypto"]
    assert defaults["max_live_context_age_seconds"] == 5.0
    assert defaults["max_oracle_age_seconds"] == 20.0
    assert defaults["max_oracle_age_ms"] == 20000.0
    assert defaults["require_oracle_for_directional"] is True
    assert defaults["oracle_source_policy"] == "degrade"
    assert defaults["min_seconds_left_for_entry_5m"] == 35.0
    assert defaults["min_seconds_left_for_entry_15m"] == 60.0
    assert defaults["min_seconds_left_for_entry_1h"] == 240.0
    assert defaults["opening_directional_buy_yes_enabled"] is False
    assert defaults["opening_directional_buy_yes_block_elapsed_pct"] == 0.10
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_5m"] == 0.45
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_15m"] == 0.25
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_1h"] == 0.10
    assert defaults["opening_directional_buy_yes_block_elapsed_pct_4h"] == 0.05
    assert defaults["entry_executable_exit_ratio_floor"] == 0.28
    assert defaults["rapid_take_profit_pct"] == 10.0
    assert defaults["rapid_take_profit_pct_5m"] == 85.0
    assert defaults["rapid_take_profit_pct_15m"] == 70.0
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
    assert defaults["force_flatten_max_profit_pct"] == 100.0
    assert defaults["force_flatten_min_loss_pct"] == 0.0
    assert defaults["resolution_risk_flatten_enabled"] is True
    assert defaults["resolution_risk_seconds_left_5m"] == 105.0
    assert defaults["resolution_risk_max_profit_pct"] == 6.0
    assert defaults["enforce_min_exit_notional"] is True
    assert defaults["hard_stop_loss_enabled"] is True
    assert defaults["hard_stop_loss_pct"] == 20.0
    assert defaults["hard_stop_loss_pct_5m"] == 15.0
    assert defaults["hard_stop_loss_pct_15m"] == 18.0
    assert defaults["hard_stop_loss_pct_1h"] == 22.0
    assert defaults["hard_stop_loss_pct_4h"] == 25.0
    assert defaults["immediate_stop_loss_requires_time_pressure"] is False
    assert defaults["stop_loss_activation_seconds_5m"] == 0
    assert defaults["min_edge_percent"] == 0.3
    assert defaults["min_confidence"] == 0.42
    assert defaults["directional_max_entry_price_ceiling"] == 0.80
    assert defaults["maker_max_entry_price_ceiling"] == 0.80


def _late_favorite_test_market(entry_yes: float = 0.70, entry_no: float = 0.30) -> Market:
    return Market(
        id="m1",
        condition_id="m1",
        question="Will this resolve soon?",
        slug="m1-resolve-soon",
        clob_token_ids=["token-yes", "token-no"],
        outcome_prices=[entry_yes, entry_no],
        active=True,
        closed=False,
        liquidity=5000.0,
        end_date=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_get_spread_bps_accepts_bid_ask_keys():
    market = SimpleNamespace(clob_token_ids=["token-yes", "token-no"])
    prices = {"token-yes": {"mid": 0.80, "bid": 0.79, "ask": 0.81}}
    spread = StrategySDK.get_spread_bps(market, prices, side="YES")
    assert spread is not None
    assert abs(spread - 250.0) < 1e-6


def test_get_spread_bps_infers_mid_when_missing():
    market = SimpleNamespace(clob_token_ids=["token-yes", "token-no"])
    prices = {"token-yes": {"bid": 0.79, "ask": 0.81}}
    spread = StrategySDK.get_spread_bps(market, prices, side="YES")
    assert spread is not None
    assert abs(spread - 250.0) < 1e-6


def test_late_favorite_alpha_detect_falls_back_when_trade_tape_unavailable(monkeypatch):
    monkeypatch.setattr(StrategySDK, "is_ws_feed_started", staticmethod(lambda: False))

    strategy = LateFavoriteAlphaStrategy()
    opportunities = strategy.detect(
        events=[],
        markets=[_late_favorite_test_market(entry_yes=0.70, entry_no=0.30)],
        prices={},
    )

    assert len(opportunities) == 1
    context = opportunities[0].strategy_context or {}
    assert context.get("trade_tape_available") is False
    assert context.get("flow_data_available") is False


def test_late_favorite_alpha_requires_flow_volume_when_trade_tape_available(monkeypatch):
    monkeypatch.setattr(StrategySDK, "is_ws_feed_started", staticmethod(lambda: True))
    monkeypatch.setattr(
        StrategySDK,
        "get_trade_volume",
        staticmethod(lambda token_id, lookback_seconds=300.0: {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}),
    )
    monkeypatch.setattr(
        StrategySDK,
        "get_buy_sell_imbalance",
        staticmethod(lambda token_id, lookback_seconds=300.0: 0.0),
    )
    monkeypatch.setattr(
        StrategySDK,
        "get_price_change",
        staticmethod(lambda token_id, lookback_seconds=300: None),
    )

    strategy = LateFavoriteAlphaStrategy()
    opportunities = strategy.detect(
        events=[],
        markets=[_late_favorite_test_market(entry_yes=0.70, entry_no=0.30)],
        prices={"token-yes": {"mid": 0.70, "bid": 0.695, "ask": 0.705}},
    )
    assert opportunities == []


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


def test_crypto_highfreq_resolution_risk_flatten_keeps_loss_pressure_near_close():
    should_flatten, detail = crypto_highfreq_should_flatten_resolution_risk(
        {
            "resolution_risk_flatten_enabled": True,
            "resolution_risk_seconds_left_15m": 240,
            "resolution_risk_min_loss_pct": 2,
        },
        timeframe="15m",
        seconds_left=180.0,
        pnl_percent=-7.5,
        exit_headroom_ratio=1.3,
        take_profit_armed=False,
    )

    assert should_flatten is True
    assert "loss_pressure=True" in detail
