"""Add the global recording master switch (app_settings.recording_enabled).

After moving all market-data recording off Postgres onto parquet, the
operator asked for a single control to "turn recording off completely".
This column is that switch: when False, the live book/delta ingestor drops
its flush batches, the proactive subscription loop unsubscribes and records
nothing, and the crypto.update.dispatch bus tee is skipped.  Default True.

Read live (short-TTL cache) via ``services.recording_control`` so the toggle
takes effect without an app restart.

Idempotent + boot-safe: the column is added only if absent (cheap
information_schema read; no lock on a missing column), mirroring the rest of
the alembic chain which must survive ``init_database`` retry loops.
"""

import sqlalchemy as sa
from alembic import op


revision = "202605280001"
down_revision = "202605260001"
branch_labels = None
depends_on = None


_COLUMN = "recording_enabled"


def _existing_columns(bind) -> set[str]:
    rows = bind.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'app_settings'"
        )
    ).scalars().all()
    return {str(r) for r in rows}


def upgrade() -> None:
    bind = op.get_bind()
    if _COLUMN in _existing_columns(bind):
        return
    op.add_column(
        "app_settings",
        sa.Column(
            _COLUMN,
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    if _COLUMN in _existing_columns(bind):
        op.drop_column("app_settings", _COLUMN)
