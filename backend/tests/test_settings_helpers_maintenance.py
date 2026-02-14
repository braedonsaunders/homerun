import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.settings_helpers import apply_update_request, maintenance_payload
from models.database import AppSettings


def _make_request(maintenance_section):
    return SimpleNamespace(
        polymarket=None,
        kalshi=None,
        llm=None,
        notifications=None,
        scanner=None,
        trading=None,
        maintenance=maintenance_section,
        discovery=None,
        search_filters=None,
        trading_proxy=None,
    )


def test_maintenance_payload_defaults_llm_usage_retention_days():
    settings = AppSettings(id="default")
    settings.llm_usage_retention_days = None

    payload = maintenance_payload(settings)

    assert payload["llm_usage_retention_days"] == 30


def test_apply_update_request_sets_llm_usage_retention_days():
    settings = AppSettings(id="default")
    maintenance = SimpleNamespace(
        auto_cleanup_enabled=True,
        cleanup_interval_hours=12,
        cleanup_resolved_trade_days=20,
        llm_usage_retention_days=14,
        market_cache_hygiene_enabled=True,
        market_cache_hygiene_interval_hours=4,
        market_cache_retention_days=120,
        market_cache_reference_lookback_days=45,
        market_cache_weak_entry_grace_days=7,
        market_cache_max_entries_per_slug=3,
    )
    request = _make_request(maintenance)

    apply_update_request(settings, request)

    assert settings.llm_usage_retention_days == 14
