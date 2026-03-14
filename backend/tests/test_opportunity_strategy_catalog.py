import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.opportunity_strategy_catalog import build_system_opportunity_strategy_rows


def test_tail_end_carry_exposes_market_name_exclusion_list_field():
    rows = build_system_opportunity_strategy_rows()
    tail_end_carry_row = next(
        row for row in rows if str(row.get("slug") or "").strip().lower() == "tail_end_carry"
    )
    config_schema = dict(tail_end_carry_row.get("config_schema") or {})
    param_fields = {
        str(field.get("key") or "").strip(): field
        for field in list(config_schema.get("param_fields") or [])
        if isinstance(field, dict)
    }

    assert "exclude_market_keywords" in param_fields
    exclude_field = param_fields["exclude_market_keywords"]
    assert str(exclude_field.get("type") or "").strip().lower() == "list"
    assert str(exclude_field.get("label") or "").strip() == "Exclude Market Name Contains"


def test_btc_5m_reversal_sniper_seed_is_present_with_expected_schema():
    rows = build_system_opportunity_strategy_rows()
    reversal_row = next(
        row for row in rows if str(row.get("slug") or "").strip().lower() == "btc_5m_reversal_sniper"
    )
    config_schema = dict(reversal_row.get("config_schema") or {})
    param_fields = {
        str(field.get("key") or "").strip(): field
        for field in list(config_schema.get("param_fields") or [])
        if isinstance(field, dict)
    }

    assert reversal_row["source_key"] == "crypto"
    assert reversal_row["class_name"] == "BTC5mReversalSniperStrategy"
    assert reversal_row["sort_order"] == 198
    assert param_fields["timeframe"]["options"] == ["5min"]
    assert param_fields["entry_window_start_seconds"]["max"] == 60
    assert param_fields["min_velocity_for_reversal"]["type"] == "number"
