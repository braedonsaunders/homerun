"""Backfill tail_end_carry gap-risk exclusions.

Revision ID: 202606130001
Revises: 202606120001
Create Date: 2026-06-13
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op


revision = "202606130001"
down_revision = "202606120001"
branch_labels = None
depends_on = None


_EXCLUDED_KEYWORDS = [
    "exact score",
    "exact-score",
    "correct score",
    "o/u",
    "over/under",
    "over under",
    "total goals",
    "total points",
    "bitcoin",
    "ethereum",
    "xrp",
    "ripple",
    "lol:",
    "counter-strike",
    "tweets",
    "league of legends",
    "esports",
    "rift legends",
    "dota",
    "valorant",
    "cs2",
    "cs:",
    "esl pro",
]


def _decode_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _merge_keywords(existing: Any) -> list[str]:
    if isinstance(existing, str):
        values = [part.strip() for part in existing.split(",") if part.strip()]
    elif isinstance(existing, list):
        values = [str(part).strip() for part in existing if str(part).strip()]
    else:
        values = []

    seen = {value.lower() for value in values}
    for keyword in _EXCLUDED_KEYWORDS:
        if keyword.lower() not in seen:
            values.append(keyword)
            seen.add(keyword.lower())
    return values


def _table_names(bind: sa.Connection) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    tables = _table_names(bind)

    if "strategies" in tables:
        rows = bind.execute(
            sa.text("SELECT id, config FROM strategies WHERE slug = 'tail_end_carry'")
        ).mappings()
        for row in rows:
            config = _decode_json(row["config"], {})
            if not isinstance(config, dict):
                config = {}
            config["exclude_market_keywords"] = _merge_keywords(config.get("exclude_market_keywords"))
            bind.execute(
                sa.text("UPDATE strategies SET config = :config WHERE id = :id"),
                {"id": row["id"], "config": json.dumps(config)},
            )

    if "strategy_versions" in tables:
        rows = bind.execute(
            sa.text(
                "SELECT id, config FROM strategy_versions "
                "WHERE strategy_slug = 'tail_end_carry' AND is_latest = true"
            )
        ).mappings()
        for row in rows:
            config = _decode_json(row["config"], {})
            if not isinstance(config, dict):
                config = {}
            config["exclude_market_keywords"] = _merge_keywords(config.get("exclude_market_keywords"))
            bind.execute(
                sa.text("UPDATE strategy_versions SET config = :config WHERE id = :id"),
                {"id": row["id"], "config": json.dumps(config)},
            )

    if "traders" in tables:
        rows = bind.execute(
            sa.text(
                "SELECT id, source_configs_json FROM traders "
                "WHERE source_configs_json::text ILIKE '%tail_end_carry%'"
            )
        ).mappings()
        for row in rows:
            source_configs = _decode_json(row["source_configs_json"], [])
            if not isinstance(source_configs, list):
                continue
            changed = False
            for source_config in source_configs:
                if not isinstance(source_config, dict):
                    continue
                if str(source_config.get("strategy_key") or "").strip().lower() != "tail_end_carry":
                    continue
                params = source_config.setdefault("strategy_params", {})
                if not isinstance(params, dict):
                    params = {}
                    source_config["strategy_params"] = params
                merged = _merge_keywords(params.get("exclude_market_keywords"))
                if params.get("exclude_market_keywords") != merged:
                    params["exclude_market_keywords"] = merged
                    changed = True
            if changed:
                bind.execute(
                    sa.text("UPDATE traders SET source_configs_json = :configs WHERE id = :id"),
                    {"id": row["id"], "configs": json.dumps(source_configs)},
                )


def downgrade() -> None:
    pass
