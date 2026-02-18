"""Upgrade system trader strategy seeds to full source implementations.

Revision ID: 202602170001
Revises: 202602160002
Create Date: 2026-02-17 00:00:01.000000
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from alembic import op
import sqlalchemy as sa
from alembic_helpers import table_names

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.opportunity_strategy_catalog import build_system_opportunity_strategy_rows  # noqa: E402


# revision identifiers, used by Alembic.
revision = "202602170001"
down_revision = "202602160002"
branch_labels = None
depends_on = None

_LEGACY_WRAPPER_MARKERS = (
    "System strategy seed wrapper loaded from DB",
    "delegates runtime behavior to the built-in strategy implementation",
    "as _SeedStrategy",
)


def _is_legacy_wrapper_source(source_code: str | None) -> bool:
    text = str(source_code or "")
    return any(marker in text for marker in _LEGACY_WRAPPER_MARKERS)


def upgrade() -> None:
    if "trader_strategy_definitions" not in table_names():
        return

    bind = op.get_bind()
    table = sa.table(
        "trader_strategy_definitions",
        sa.column("id", sa.String()),
        sa.column("strategy_key", sa.String()),
        sa.column("source_key", sa.String()),
        sa.column("label", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("class_name", sa.String()),
        sa.column("source_code", sa.Text()),
        sa.column("default_params_json", sa.JSON()),
        sa.column("param_schema_json", sa.JSON()),
        sa.column("aliases_json", sa.JSON()),
        sa.column("is_system", sa.Boolean()),
        sa.column("enabled", sa.Boolean()),
        sa.column("status", sa.String()),
        sa.column("error_message", sa.Text()),
        sa.column("version", sa.Integer()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )

    now = datetime.utcnow()
    rows = build_system_opportunity_strategy_rows()
    for row in rows:
        existing = (
            bind.execute(
                sa.select(
                    table.c.id,
                    table.c.is_system,
                    table.c.source_code,
                    table.c.version,
                ).where(table.c.strategy_key == row["strategy_key"])
            )
            .mappings()
            .first()
        )

        if existing is None:
            bind.execute(
                table.insert().values(
                    id=row["id"],
                    strategy_key=row["strategy_key"],
                    source_key=row["source_key"],
                    label=row["label"],
                    description=row["description"],
                    class_name=row["class_name"],
                    source_code=row["source_code"],
                    default_params_json=row["default_params_json"],
                    param_schema_json=row["param_schema_json"],
                    aliases_json=row["aliases_json"],
                    is_system=True,
                    enabled=True,
                    status="unloaded",
                    error_message=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            continue

        if not bool(existing.get("is_system")):
            continue
        if not _is_legacy_wrapper_source(existing.get("source_code")):
            continue

        bind.execute(
            table.update()
            .where(table.c.strategy_key == row["strategy_key"])
            .values(
                source_key=row["source_key"],
                label=row["label"],
                description=row["description"],
                class_name=row["class_name"],
                source_code=row["source_code"],
                default_params_json=row["default_params_json"],
                param_schema_json=row["param_schema_json"],
                aliases_json=row["aliases_json"],
                status="unloaded",
                error_message=None,
                version=int(existing.get("version") or 0) + 1,
                updated_at=now,
            )
        )


def downgrade() -> None:
    # Not reversible: previous wrapper source stubs are intentionally not restored.
    return
