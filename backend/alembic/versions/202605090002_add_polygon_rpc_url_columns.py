"""Add polygon_rpc_url and polygon_ws_url columns to app_settings.

Lets the operator configure an authenticated Polygon RPC provider
(Ankr / Alchemy / QuickNode) through the Settings UI instead of
having to populate ``POLYGON_RPC_URL`` via env-var. URL is stored
encrypted because Ankr-style provider URLs embed the API key in the
path (``https://rpc.ankr.com/polygon/<key>``) — same secrecy class
as ``trading_proxy_url``. ``polygon_ws_url`` follows the same
pattern for the WebSocket subscription endpoint.

Revision ID: 202605090002
Revises: 202605090001
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op


revision = "202605090002"
down_revision = "202605090001"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    existing = _column_names("app_settings")
    columns = [
        sa.Column("polygon_rpc_url", sa.String(), nullable=True),
        sa.Column("polygon_ws_url", sa.String(), nullable=True),
    ]
    for col in columns:
        if col.name not in existing:
            op.add_column("app_settings", col)


def downgrade() -> None:
    pass
