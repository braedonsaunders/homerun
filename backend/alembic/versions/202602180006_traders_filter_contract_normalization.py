"""Normalize traders strategy filter config to canonical StrategySDK contract.

Revision ID: 202602180006
Revises: 202602180005
Create Date: 2026-02-18 23:59:00.000000
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import op
import sqlalchemy as sa
from alembic_helpers import table_names

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_sdk import StrategySDK  # noqa: E402


# revision identifiers, used by Alembic.
revision = "202602180006"
down_revision = "202602180005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "strategies" not in table_names():
        return

    bind = op.get_bind()
    strategies = sa.table(
        "strategies",
        sa.column("id", sa.String()),
        sa.column("slug", sa.String()),
        sa.column("config", sa.JSON()),
    )

    row = (
        bind.execute(sa.select(strategies.c.id, strategies.c.config).where(strategies.c.slug == "traders_confluence"))
        .mappings()
        .first()
    )
    if row is None:
        return

    raw_config = row.get("config")
    if not isinstance(raw_config, dict):
        raw_config = {}

    normalized = StrategySDK.validate_trader_filter_config(raw_config)
    if normalized == raw_config:
        return

    bind.execute(strategies.update().where(strategies.c.id == row["id"]).values(config=normalized))


def downgrade() -> None:
    return
