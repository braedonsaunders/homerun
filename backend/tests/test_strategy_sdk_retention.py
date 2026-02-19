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
