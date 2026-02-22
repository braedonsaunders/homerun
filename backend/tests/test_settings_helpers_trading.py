import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.settings_helpers import apply_update_request
from models.database import AppSettings


def _make_request(trading_section):
    return SimpleNamespace(
        polymarket=None,
        kalshi=None,
        llm=None,
        notifications=None,
        scanner=None,
        trading=trading_section,
        maintenance=None,
        discovery=None,
        search_filters=None,
        trading_proxy=None,
        events=None,
        ui_lock=None,
    )


def test_apply_update_request_trading_updates_safety_fields_only():
    settings = AppSettings(id="default")
    settings.btc_eth_hf_series_btc_15m = "persist-me"
    settings.max_trade_size_usd = 100.0
    settings.max_daily_trade_volume = 1000.0
    settings.max_open_positions = 10
    settings.max_slippage_percent = 2.0
    settings.trading_enabled = False

    request = _make_request(
        SimpleNamespace(
            trading_enabled=True,
            max_trade_size_usd=250.0,
            max_daily_trade_volume=5000.0,
            max_open_positions=20,
            max_slippage_percent=1.5,
        )
    )

    apply_update_request(settings, request)

    assert settings.trading_enabled is True
    assert settings.max_trade_size_usd == 250.0
    assert settings.max_daily_trade_volume == 5000.0
    assert settings.max_open_positions == 20
    assert settings.max_slippage_percent == 1.5
    assert settings.btc_eth_hf_series_btc_15m == "persist-me"
