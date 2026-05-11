"""App settings: cached Telonex download quota.

Revision ID: 202605110100
Revises: 202605100300
Create Date: 2026-05-11

Adds a pair of columns that cache the last X-Downloads-Remaining
header we saw on a Telonex response.  The UI uses these to render the
operator's free-tier quota without making a fresh request for every
panel render.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op  # noqa: F401 (kept for downgrade)
from alembic_helpers import safe_add_column


# revision identifiers, used by Alembic.
revision = "202605110100"
down_revision = "202605100300"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "app_settings",
        sa.Column("telonex_downloads_remaining", sa.Integer(), nullable=True),
    )
    safe_add_column(
        "app_settings",
        sa.Column("telonex_downloads_remaining_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "telonex_downloads_remaining_at")
    op.drop_column("app_settings", "telonex_downloads_remaining")
