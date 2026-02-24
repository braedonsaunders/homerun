"""Move trader source scope settings into strategy_params and drop sidecar field.

Revision ID: 202602230003
Revises: 202602230002
Create Date: 2026-02-23 00:00:00.000000
"""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names, table_names


revision = "202602230003"
down_revision = "202602230002"
branch_labels = None
depends_on = None


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def upgrade() -> None:
    existing_tables = table_names()
    if "traders" not in existing_tables:
        return

    trader_cols = column_names("traders")
    if "source_configs_json" not in trader_cols:
        return

    bind = op.get_bind()
    traders = sa.table(
        "traders",
        sa.column("id", sa.String()),
        sa.column("source_configs_json", sa.JSON()),
        sa.column("updated_at", sa.DateTime()),
    )

    rows = list(bind.execute(sa.select(traders.c.id, traders.c.source_configs_json)).mappings())
    if not rows:
        return

    now = _utcnow_naive()
    has_updated_at = "updated_at" in trader_cols

    for row in rows:
        source_configs = row.get("source_configs_json")
        if not isinstance(source_configs, list):
            continue

        next_source_configs: list[dict] = []
        changed = False

        for source_config in source_configs:
            if not isinstance(source_config, dict):
                next_source_configs.append(source_config)
                continue

            next_config = dict(source_config)
            source_key = str(next_config.get("source_key") or "").strip().lower()
            strategy_params = (
                dict(next_config.get("strategy_params") or {})
                if isinstance(next_config.get("strategy_params"), dict)
                else {}
            )

            legacy_scope = next_config.pop("traders_scope", None)
            if legacy_scope is not None:
                changed = True

            if source_key == "traders" and "traders_scope" not in strategy_params and isinstance(legacy_scope, dict):
                strategy_params["traders_scope"] = legacy_scope
                changed = True

            next_config["strategy_params"] = strategy_params
            next_source_configs.append(next_config)

        if not changed:
            continue

        values: dict[str, object] = {"source_configs_json": next_source_configs}
        if has_updated_at:
            values["updated_at"] = now
        bind.execute(traders.update().where(traders.c.id == row.get("id")).values(**values))


def downgrade() -> None:
    pass
