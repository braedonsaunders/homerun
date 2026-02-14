import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.risk_manager import evaluate_risk


def test_risk_blocks_on_open_position_limit_with_new_key():
    result = evaluate_risk(
        size_usd=25.0,
        gross_exposure_usd=100.0,
        trader_open_positions=2,
        market_exposure_usd=50.0,
        global_limits={
            "max_gross_exposure_usd": 10_000.0,
            "max_daily_loss_usd": 10_000.0,
        },
        trader_limits={
            "max_open_positions": 2,
            "max_orders_per_cycle": 10,
            "max_trade_notional_usd": 500.0,
            "max_per_market_exposure_usd": 500.0,
            "max_daily_loss_usd": 10_000.0,
        },
        global_daily_realized_pnl_usd=0.0,
        trader_daily_realized_pnl_usd=0.0,
        trader_consecutive_losses=0,
        cycle_orders_placed=0,
        cooldown_active=False,
    )

    assert result.allowed is False
    assert result.reason == "Risk blocked: trader_open_positions"
    assert any(check.key == "trader_open_positions" and not check.passed for check in result.checks)
