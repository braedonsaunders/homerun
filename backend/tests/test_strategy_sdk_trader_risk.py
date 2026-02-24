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
