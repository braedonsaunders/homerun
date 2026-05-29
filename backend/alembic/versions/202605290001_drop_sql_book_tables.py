"""Drop the SQL book/delta tables (market-data clean cut).

Book + delta recording moved fully onto the canonical parquet plane
(``live_ingestor`` / polybacktest / telonex providers), read through the
unified ``services.marketdata`` access layer. The
``market_microstructure_snapshots`` and ``book_delta_events`` tables are no
longer written or read by any code path — this migration drops them.

Also drops the two AppSettings retention knobs that only governed those SQL
tables (``cleanup_market_microstructure_days`` / ``cleanup_book_delta_events_days``):
book/delta retention is now handled by the parquet pruners
(``services.external_data.book_parquet_sink``), so the SQL-era knobs are dead
config — removed here as part of the same clean cut.

Idempotent + boot-safe: uses ``DROP TABLE IF EXISTS`` / ``DROP COLUMN IF EXISTS``
so the alembic chain survives ``init_database`` retry loops. The data was
derived/replayable market microstructure (no irreplaceable content);
``downgrade`` recreates the empty table structures and re-adds the columns so
the migration is reversible.
"""

import sqlalchemy as sa
from alembic import op


revision = "202605290001"
down_revision = "202605280001"
branch_labels = None
depends_on = None


_TABLES = ("book_delta_events", "market_microstructure_snapshots")
_DEAD_SETTINGS_COLUMNS = (
    "cleanup_market_microstructure_days",
    "cleanup_book_delta_events_days",
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        bind.execute(sa.text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
    for column in _DEAD_SETTINGS_COLUMNS:
        bind.execute(sa.text(f'ALTER TABLE app_settings DROP COLUMN IF EXISTS "{column}"'))


def downgrade() -> None:
    # Re-add the SQL-era retention knobs (defaults matched the originals).
    op.add_column("app_settings", sa.Column("cleanup_market_microstructure_days", sa.Integer(), server_default="7"))
    op.add_column("app_settings", sa.Column("cleanup_book_delta_events_days", sa.Integer(), server_default="7"))
    # Recreate the table structures (empty) so the migration is reversible.
    op.create_table(
        "market_microstructure_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False, server_default="polymarket"),
        sa.Column("token_id", sa.String(), nullable=False),
        sa.Column("snapshot_type", sa.String(), nullable=False),
        sa.Column("observed_at", sa.DateTime(), nullable=False),
        sa.Column("exchange_ts_ms", sa.BigInteger(), nullable=True),
        sa.Column("sequence", sa.BigInteger(), nullable=True),
        sa.Column("best_bid", sa.Float(), nullable=True),
        sa.Column("best_ask", sa.Float(), nullable=True),
        sa.Column("spread_bps", sa.Float(), nullable=True),
        sa.Column("bids_json", sa.JSON(), nullable=True),
        sa.Column("asks_json", sa.JSON(), nullable=True),
        sa.Column("trade_price", sa.Float(), nullable=True),
        sa.Column("trade_size", sa.Float(), nullable=True),
        sa.Column("trade_side", sa.String(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_mms_token_observed", "market_microstructure_snapshots", ["token_id", "observed_at"])
    op.create_index("idx_mms_token_type_observed", "market_microstructure_snapshots", ["token_id", "snapshot_type", "observed_at"])
    op.create_index("ix_market_microstructure_snapshots_provider", "market_microstructure_snapshots", ["provider"])
    op.create_index("ix_market_microstructure_snapshots_token_id", "market_microstructure_snapshots", ["token_id"])
    op.create_index("ix_market_microstructure_snapshots_snapshot_type", "market_microstructure_snapshots", ["snapshot_type"])
    op.create_index("ix_market_microstructure_snapshots_observed_at", "market_microstructure_snapshots", ["observed_at"])

    op.create_table(
        "book_delta_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False, server_default="polymarket"),
        sa.Column("token_id", sa.String(), nullable=False),
        sa.Column("observed_at", sa.DateTime(), nullable=False),
        sa.Column("exchange_ts_ms", sa.BigInteger(), nullable=True),
        sa.Column("sequence", sa.BigInteger(), nullable=True),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("side", sa.String(), nullable=True),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("trade_size", sa.Float(), nullable=True),
        sa.Column("cancel_size", sa.Float(), nullable=True),
        sa.Column("queue_depth_before", sa.Float(), nullable=True),
        sa.Column("queue_depth_after", sa.Float(), nullable=True),
        sa.Column("spread_bps_at_event", sa.Float(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_bde_token_observed", "book_delta_events", ["token_id", "observed_at"])
    op.create_index("idx_bde_token_type_observed", "book_delta_events", ["token_id", "event_type", "observed_at"])
    op.create_index("ix_book_delta_events_provider", "book_delta_events", ["provider"])
    op.create_index("ix_book_delta_events_token_id", "book_delta_events", ["token_id"])
    op.create_index("ix_book_delta_events_event_type", "book_delta_events", ["event_type"])
    op.create_index("ix_book_delta_events_observed_at", "book_delta_events", ["observed_at"])
