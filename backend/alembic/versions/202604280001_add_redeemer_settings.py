"""add redeemer policy columns to app_settings

Revision ID: 202604280001
Revises: 202604270001
Create Date: 2026-04-28

The CTF redeemer worker needs operator-tunable policy:

  * ``redeemer_min_payout_usd`` — skip redemption when expected payout
    is below this floor. This is the guard against burning
    gas to redeem $0-payout positions (lost binary outcomes).
  * ``redeemer_max_gas_price_gwei`` — skip redemption when network gas
    is elevated above this ceiling. Defers cleanup to a cheaper window.
  * ``redeemer_force_including_losers`` — operator-only override that
    redeems every resolved condition regardless of payout, useful for
    one-off wallet hygiene sweeps. Costs gas with no return for losing
    outcomes; only flip when intentionally cleaning up dust.

The columns are nullable so existing rows fall through to code defaults
in ``config.Settings`` and ``apply_settings_overrides`` without a
backfill. Defaults are tuned for Polygon: ~120k gas × ~30 gwei × MATIC
≈ $0.001 per redemption tx, so a $0.10 floor leaves a comfortable 100×
safety margin.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column


# revision identifiers, used by Alembic.
revision = "202604280001"
down_revision = "202604270001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "app_settings",
        sa.Column(
            "redeemer_min_payout_usd",
            sa.Float(),
            nullable=True,
        ),
    )
    safe_add_column(
        "app_settings",
        sa.Column(
            "redeemer_max_gas_price_gwei",
            sa.Float(),
            nullable=True,
        ),
    )
    safe_add_column(
        "app_settings",
        sa.Column(
            "redeemer_force_including_losers",
            sa.Boolean(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "redeemer_force_including_losers")
    op.drop_column("app_settings", "redeemer_max_gas_price_gwei")
    op.drop_column("app_settings", "redeemer_min_payout_usd")
