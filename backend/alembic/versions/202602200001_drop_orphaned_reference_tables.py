"""Drop orphaned reference tables that were never populated.

CountryInstabilityRecord, TensionPairRecord, and ConflictEventRecord
were planned but never implemented. All data now flows through the
DataSourceSDK pipeline (DataSourceRecord -> EventsSignal).

Revision ID: 202602200001
Revises: 202602190007
"""

from alembic import op
import sqlalchemy as sa

revision = "202602200001"
down_revision = "202602190007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for name in ("country_instability_records", "tension_pair_records", "conflict_event_records"):
        if name in existing:
            op.drop_table(name)


def downgrade() -> None:
    pass  # These tables were never populated; no data to restore
