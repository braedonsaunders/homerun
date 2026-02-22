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
