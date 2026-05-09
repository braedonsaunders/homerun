"""Backfill tail_end_carry time-weighted exit defaults onto existing traders.

Adds three exit-shaping mechanisms whose defaults live in the strategy's
``default_config`` but won't apply to in-flight traders unless we backfill
their ``source_configs_json`` entries:

  * ``time_weighted_stop_*``  – #1 time-weighted trailing stop schedule
  * ``scale_out_*``           – #2 resolution-window scale-out ladder
  * ``velocity_stop_*``       – #3 cliff-drop velocity stop

We only fill keys that are missing — never overwrite what an operator has
already tuned. This makes the migration a no-op on second run.

Revision ID: 202605090001
Revises: 202605070002
Create Date: 2026-05-09
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op
from alembic_helpers import table_names


revision = "202605090001"
down_revision = "202605070002"
branch_labels = None
depends_on = None


_NEW_DEFAULTS: dict[str, Any] = {
    "time_weighted_stop_enabled": True,
    "time_weighted_stop_schedule": [
        [30,    2.0,  0.85],
        [120,   5.0,  0.72],
        [360,  10.0,  0.60],
        [1440, 18.0,  0.50],
        [99999, 30.0, 0.40],
    ],
    "sports_time_weighted_stop_schedule": [
        [30,    5.0,  0.70],
        [120,  10.0,  0.55],
        [360,  20.0,  0.45],
        [1440, 35.0,  0.35],
        [99999, 50.0, 0.30],
    ],
    "scale_out_enabled": True,
    "scale_out_tiers": [
        {"minutes_left_max": 120, "min_price": 0.95, "exit_fraction": 0.25, "tighten_trailing_floor_pct": 4.0},
        {"minutes_left_max": 30,  "min_price": 0.97, "exit_fraction": 0.50, "tighten_trailing_floor_pct": 2.0},
        {"minutes_left_max": 5,   "min_price": 0.0,  "exit_fraction": 1.0,  "tighten_trailing_floor_pct": 0.0},
    ],
    "velocity_stop_enabled": True,
    "velocity_window_seconds": 300.0,
    "velocity_drop_threshold_pct": 3.0,
    "velocity_active_minutes_left": 60.0,
    "velocity_min_samples": 3,
}


def _coerce_configs(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def upgrade() -> None:
    if "traders" not in table_names():
        return

    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT id, source_configs_json FROM traders "
            "WHERE source_configs_json::text ILIKE '%tail_end_carry%'"
        )
    ).fetchall()

    for row in rows:
        trader_id = row[0]
        configs = _coerce_configs(row[1])
        changed = False
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            if cfg.get("strategy_key") != "tail_end_carry":
                continue
            params = cfg.setdefault("strategy_params", {})
            if not isinstance(params, dict):
                params = {}
                cfg["strategy_params"] = params
            for key, value in _NEW_DEFAULTS.items():
                if key not in params:
                    params[key] = value
                    changed = True

        if changed:
            conn.execute(
                sa.text("UPDATE traders SET source_configs_json = :cfg WHERE id = :tid"),
                {"cfg": json.dumps(configs), "tid": trader_id},
            )


def downgrade() -> None:
    if "traders" not in table_names():
        return

    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT id, source_configs_json FROM traders "
            "WHERE source_configs_json::text ILIKE '%tail_end_carry%'"
        )
    ).fetchall()

    for row in rows:
        trader_id = row[0]
        configs = _coerce_configs(row[1])
        changed = False
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            if cfg.get("strategy_key") != "tail_end_carry":
                continue
            params = cfg.get("strategy_params")
            if not isinstance(params, dict):
                continue
            for key in _NEW_DEFAULTS.keys():
                if key in params:
                    params.pop(key, None)
                    changed = True

        if changed:
            conn.execute(
                sa.text("UPDATE traders SET source_configs_json = :cfg WHERE id = :tid"),
                {"cfg": json.dumps(configs), "tid": trader_id},
            )
