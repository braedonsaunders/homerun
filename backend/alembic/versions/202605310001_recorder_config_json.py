"""Add the operator recording-config blob (app_settings.recorder_config_json).

The recording master switch (``recording_enabled``) is a single on/off.
Operators also need to tune *how* recording behaves without an app restart:
how many L2 levels per side to persist, the proactive-subscription token cap
and liquidity floor, and whether to capture books / trades / the catalog
stream.  Rather than add a column per knob (and a migration per future knob),
those settings live in one nullable JSON blob, read / written via
``services.recording_control.get_recorder_config`` / ``set_recorder_config``
with a short-TTL cache (same shape as the master switch).

Null = use all service-level defaults, so existing rows need no backfill.

Idempotent + boot-safe: the column is added only if absent (cheap
information_schema read; no lock on a missing column), mirroring the rest of
the alembic chain which must survive ``init_database`` retry loops.
"""

import sqlalchemy as sa
from alembic import op


revision = "202605310001"
down_revision = "202605290001"
branch_labels = None
depends_on = None


_COLUMN = "recorder_config_json"


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
        sa.Column(_COLUMN, sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    bind = op.get_bind()
    if _COLUMN in _existing_columns(bind):
        op.drop_column("app_settings", _COLUMN)
