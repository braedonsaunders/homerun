"""add polymarket_default_collateral to app_settings

Revision ID: 202604300004
Revises: 202604300003
Create Date: 2026-04-30

The CTF split/merge paths take an explicit ``collateral_address`` arg
since the 2026-04 Polymarket pUSD migration; absent an explicit value,
they fall back to ``settings.POLYMARKET_DEFAULT_COLLATERAL``.  This
column lets operators flip the default through the existing settings
UI without a code redeploy — useful for keeping legacy USDC.e-only
deployments stable while pUSD becomes the platform-wide canonical.

Recognized values stored as plain strings (case-normalized at apply
time): ``pusd``, ``usdc.e``, ``usdc_native``.  NULL means use the
``Settings`` class default (currently ``pusd``).

Redemption is unaffected: the redeemer derives per-position collateral
from chain math (see ``services.polymarket_collateral``) and dispatches
to the matching CTF / NegRiskAdapter path automatically.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column


# revision identifiers, used by Alembic.
revision = "202604300004"
down_revision = "202604300003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "app_settings",
        sa.Column("polymarket_default_collateral", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "polymarket_default_collateral")
