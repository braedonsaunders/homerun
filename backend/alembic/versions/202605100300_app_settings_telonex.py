"""App settings: telonex_api_key + telonex_base_url.

Revision ID: 202605100300
Revises: 202605100200
Create Date: 2026-05-10

Adds Telonex as a Data Lab provider.  API key + optional base URL
override are persisted on ``app_settings`` alongside the other
provider credentials (polybacktest etc.) so the operator can plug
their key in from Data Lab → Providers → Telonex.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op  # noqa: F401  (kept for downgrade)
from alembic_helpers import safe_add_column


# revision identifiers, used by Alembic.
revision = "202605100300"
down_revision = "202605100200"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "app_settings",
        sa.Column("telonex_api_key", sa.String(), nullable=True),
    )
    safe_add_column(
        "app_settings",
        sa.Column("telonex_base_url", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "telonex_base_url")
    op.drop_column("app_settings", "telonex_api_key")
