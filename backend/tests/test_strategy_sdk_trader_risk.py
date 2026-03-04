import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_sdk import StrategySDK
from services.strategies.traders_copy_trade import validate_traders_copy_trade_config


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
            "paper_min_order_size_usd": "2.0",
            "live_min_order_size_usd": "4.25",
        }
    )

    assert cfg["min_order_size_usd"] == 3.5
    assert cfg["paper_min_order_size_usd"] == 2.0
    assert cfg["live_min_order_size_usd"] == 4.25


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


def test_validate_traders_copy_trade_config_normalizes_and_clamps_fields():
    cfg = validate_traders_copy_trade_config(
        {
            "min_confidence": "0.6",
            "min_source_notional_usd": "12.5",
            "max_signal_age_seconds": "120",
            "min_live_liquidity_usd": "275.5",
            "max_adverse_entry_drift_pct": "4.5",
            "copy_delay_seconds": "7",
            "copy_existing_positions_on_start": "true",
            "copy_buys": "true",
            "copy_sells": "false",
            "max_position_size": "2500",
            "proportional_sizing": "1",
            "proportional_multiplier": "1.75",
            "base_size_usd": "15",
            "max_size_usd": "10",
            "max_copy_drawdown_pct": "35",
            "max_copy_daily_loss_usd": "210.5",
            "max_copy_source_exposure_usd": "5000",
            "leader_weights": {"0xABC": "1.5", "0xDEF": "not-a-number"},
            "default_leader_weight": "0.8",
            "max_leader_exposure_usd": "1200",
            "leader_allocation_cap_pct": "65",
            "require_inventory_for_sells": "true",
            "allow_partial_inventory_sells": "false",
            "min_inventory_fraction": "0.4",
            "traders_scope": {"modes": ["individual"], "individual_wallets": ["0xabc"], "group_ids": []},
            "firehose_require_active_signal": False,
        }
    )

    assert cfg["min_confidence"] == 0.6
    assert cfg["min_source_notional_usd"] == 12.5
    assert cfg["max_signal_age_seconds"] == 5
    assert cfg["min_live_liquidity_usd"] == 275.5
    assert cfg["max_adverse_entry_drift_pct"] == 4.5
    assert cfg["copy_delay_seconds"] == 7
    assert cfg["copy_existing_positions_on_start"] is True
    assert cfg["copy_buys"] is True
    assert cfg["copy_sells"] is False
    assert cfg["max_position_size"] == 2500.0
    assert cfg["proportional_sizing"] is True
    assert cfg["proportional_multiplier"] == 1.75
    assert cfg["base_size_usd"] == 15.0
    assert cfg["max_size_usd"] == 15.0
    assert cfg["max_copy_drawdown_pct"] == 35.0
    assert cfg["max_copy_daily_loss_usd"] == 210.5
    assert cfg["max_copy_source_exposure_usd"] == 5000.0
    assert cfg["leader_weights"] == {"0xabc": 1.5}
    assert cfg["default_leader_weight"] == 0.8
    assert cfg["max_leader_exposure_usd"] == 1200.0
    assert cfg["leader_allocation_cap_pct"] == 65.0
    assert cfg["require_inventory_for_sells"] is True
    assert cfg["allow_partial_inventory_sells"] is False
    assert cfg["min_inventory_fraction"] == 0.4
    assert cfg["traders_scope"]["modes"] == ["individual"]
    assert "firehose_require_active_signal" not in cfg
