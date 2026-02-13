import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.settings_helpers import apply_update_request
from models.database import AppSettings


def _make_request(search_filters_section):
    return SimpleNamespace(
        polymarket=None,
        kalshi=None,
        llm=None,
        notifications=None,
        scanner=None,
        trading=None,
        maintenance=None,
        discovery=None,
        trading_proxy=None,
        world_intelligence=None,
        search_filters=search_filters_section,
    )


def test_search_filter_partial_update_only_changes_explicit_fields():
    settings = AppSettings(id="default")
    settings.min_liquidity_hard = 320.0
    settings.max_trade_legs = 12
    settings.btc_eth_hf_enabled = False

    search_filters = SimpleNamespace(
        min_liquidity_hard=180.0,
        model_fields_set={"min_liquidity_hard"},
    )
    request = _make_request(search_filters)

    flags = apply_update_request(settings, request)

    assert settings.min_liquidity_hard == 180.0
    assert settings.max_trade_legs == 12
    assert settings.btc_eth_hf_enabled is False
    assert flags["needs_filter_reload"] is True


def test_search_filter_partial_update_supports_new_runtime_tunables():
    settings = AppSettings(id="default")
    settings.min_liquidity_per_leg = 700.0
    settings.btc_eth_hf_maker_mode = False

    search_filters = SimpleNamespace(
        min_liquidity_per_leg=250.0,
        btc_eth_hf_maker_mode=True,
        model_fields_set={"min_liquidity_per_leg", "btc_eth_hf_maker_mode"},
    )
    request = _make_request(search_filters)

    apply_update_request(settings, request)

    assert settings.min_liquidity_per_leg == 250.0
    assert settings.btc_eth_hf_maker_mode is True
