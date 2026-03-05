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
