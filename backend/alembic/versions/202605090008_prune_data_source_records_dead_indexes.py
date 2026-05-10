"""Prune four dead indexes on data_source_records, set fillfactor=85, tune autovacuum.

The 2026-05-09 23:00 SLOW COMMIT DIAGNOSTIC captured ``data_source_records``
as the WAL-pressure dominator that's been driving the system-wide commit
slowness even after the trade_signals HOT fix landed:

    n_tup_upd       : 1,344,828
    n_tup_hot_upd   :         0  (0.00% HOT)
    indexes_size    : 569 MB
    table_size      : 2,244 MB

pg_stat_user_indexes shows four indexes have been scanned ZERO times
across the full pg_stat history window:

    indexrelname                              | idx_scan |  size
    -----------------------------------------+----------+--------
    idx_data_source_records_observed_at      |     0    |  59 MB
    idx_data_source_records_ingested_at      |     0    |  62 MB
    idx_data_source_records_data_source_id   |     0    |  12 MB  (duplicate of ix_*)
    idx_data_source_records_geotagged        |     0    |  12 MB

That's 145 MB of pure-write-amplification storage. Every UPDATE writes
four new entries to indexes nothing reads. With 1.34M UPDATEs that's
~5.4M wasted index writes — directly translating to WAL volume that
back-pressures every other table's COMMIT throughput.

Specifically, ``observed_at`` and ``ingested_at`` being indexed is what
disqualifies HOT on every UPDATE (the producer SETs both columns each
upsert: ``UPDATE data_source_records SET observed_at=$1, ingested_at=$2,
payload_json=$3, transformed_json=...``). Dropping the standalone
indexes plus the duplicate ``idx_*`` redundancy of the SQLAlchemy-named
``ix_*`` cuts index churn to 6 indexes, of which only the composite
``idx_data_source_records_source_ordering`` still references the
timestamps — so HOT still won't fire on those columns alone, but the
write-amplification drops by ~30% per UPDATE and storage frees up
145 MB.

Fillfactor=85 + aggressive autovacuum tuning matches the pattern we
applied to trade_signals (migration 202605090003). With 339k live rows
in 2.2 GB, the table is ~6.5 KB/row average — fillfactor=85 leaves
~1 KB headroom per page for in-place updates, which combined with the
reduced index footprint should improve autovacuum throughput.

Verification command after deploy:
    SELECT relname, n_tup_upd, n_tup_hot_upd,
           ROUND(100.0*n_tup_hot_upd::numeric/NULLIF(n_tup_upd,0), 1) AS hot_pct
    FROM pg_stat_user_tables WHERE relname='data_source_records';
The cumulative ``hot_pct`` will only climb if the application stops
SETting timestamp columns on UPDATE; this migration's win is purely
the write-amplification reduction. A follow-up app-side change can
attempt true HOT eligibility if needed.

Revision ID: 202605090008
Revises: 202605090007
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op


revision = "202605090008"
down_revision = "202605090007"
branch_labels = None
depends_on = None


_DEAD_INDEX_NAMES = (
    "idx_data_source_records_observed_at",
    "idx_data_source_records_ingested_at",
    "idx_data_source_records_data_source_id",
    "idx_data_source_records_geotagged",
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "data_source_records" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("data_source_records")}

    for name in _DEAD_INDEX_NAMES:
        if name in existing:
            op.drop_index(name, table_name="data_source_records")

    # Match trade_signals' tuning. fillfactor leaves headroom for
    # in-place updates; aggressive autovacuum keeps dead-tuple bloat
    # bounded under the high UPDATE rate.
    op.execute(
        sa.text(
            """
            ALTER TABLE data_source_records SET (
                fillfactor = 85,
                autovacuum_vacuum_scale_factor = 0.05,
                autovacuum_analyze_scale_factor = 0.02,
                autovacuum_vacuum_cost_delay = 0
            )
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "data_source_records" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("data_source_records")}

    if "idx_data_source_records_observed_at" not in existing:
        op.create_index(
            "idx_data_source_records_observed_at",
            "data_source_records",
            ["observed_at"],
        )
    if "idx_data_source_records_ingested_at" not in existing:
        op.create_index(
            "idx_data_source_records_ingested_at",
            "data_source_records",
            ["ingested_at"],
        )
    if "idx_data_source_records_data_source_id" not in existing:
        op.create_index(
            "idx_data_source_records_data_source_id",
            "data_source_records",
            ["data_source_id"],
        )
    if "idx_data_source_records_geotagged" not in existing:
        op.create_index(
            "idx_data_source_records_geotagged",
            "data_source_records",
            ["geotagged"],
        )

    op.execute(
        sa.text(
            """
            ALTER TABLE data_source_records RESET (
                fillfactor,
                autovacuum_vacuum_scale_factor,
                autovacuum_analyze_scale_factor,
                autovacuum_vacuum_cost_delay
            )
            """
        )
    )
