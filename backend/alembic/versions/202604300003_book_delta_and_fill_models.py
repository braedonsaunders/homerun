"""book_delta_events + fill_probability_models tables

Revision ID: 202604300003
Revises: 202604300002
Create Date: 2026-04-30

Two new tables backing the world-class shadow simulator.

``book_delta_events``: tick-by-tick decomposition of order book updates
into ``trade`` events (depth cleared with a matching trade print) and
``cancel`` events (depth disappeared with no trade — interpreted as
cancellation).  This split is the unlock that lets a fill model
distinguish queue-advancing fills from queue-advancing cancels — they
have very different implications for adverse selection.

``fill_probability_models``: versioned Cox proportional hazards (or
Kaplan-Meier nonparametric fallback) fill model artifacts, including
coefficients, baseline survival, validation scores (C-index, Brier
score), and the ``active`` flag inference reads from.

Both tables are append-mostly with retention managed by separate
worker maintenance — bounded indices on ``(token_id, observed_at)``
and ``(token_id, event_type, observed_at)`` keep the typical "fetch
last N seconds for token X" query path fast.
"""
from __future__ import annotations

from alembic import op
from alembic_helpers import safe_create_table, safe_create_index
import sqlalchemy as sa


revision = "202604300003"
down_revision = "202604300002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_create_table(
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
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    safe_create_index("idx_bde_provider", "book_delta_events", ["provider"])
    safe_create_index("idx_bde_token_id", "book_delta_events", ["token_id"])
    safe_create_index("idx_bde_observed_at", "book_delta_events", ["observed_at"])
    safe_create_index("idx_bde_event_type", "book_delta_events", ["event_type"])
    safe_create_index("idx_bde_token_observed", "book_delta_events", ["token_id", "observed_at"])
    safe_create_index(
        "idx_bde_token_type_observed",
        "book_delta_events",
        ["token_id", "event_type", "observed_at"],
    )

    safe_create_table(
        "fill_probability_models",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("family", sa.String(), nullable=False, server_default="cox_ph"),
        sa.Column("strata_key", sa.String(), nullable=False, server_default="pooled"),
        sa.Column("trained_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("training_window_start", sa.DateTime(), nullable=True),
        sa.Column("training_window_end", sa.DateTime(), nullable=True),
        sa.Column("n_events", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_observations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("concordance_index", sa.Float(), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column("log_likelihood", sa.Float(), nullable=True),
        sa.Column("coefficients_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("baseline_survival_json", sa.JSON(), nullable=True),
        sa.Column("feature_means_json", sa.JSON(), nullable=True),
        sa.Column("feature_stds_json", sa.JSON(), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.String(), nullable=True),
    )
    safe_create_index("idx_fpm_family", "fill_probability_models", ["family"])
    safe_create_index("idx_fpm_strata_key", "fill_probability_models", ["strata_key"])
    safe_create_index("idx_fpm_trained_at", "fill_probability_models", ["trained_at"])
    safe_create_index("idx_fpm_active", "fill_probability_models", ["active"])
    safe_create_index(
        "idx_fpm_family_strata_active",
        "fill_probability_models",
        ["family", "strata_key", "active"],
    )
    safe_create_index(
        "idx_fpm_active_trained",
        "fill_probability_models",
        ["active", "trained_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_fpm_active_trained", table_name="fill_probability_models")
    op.drop_index("idx_fpm_family_strata_active", table_name="fill_probability_models")
    op.drop_index("idx_fpm_active", table_name="fill_probability_models")
    op.drop_index("idx_fpm_trained_at", table_name="fill_probability_models")
    op.drop_index("idx_fpm_strata_key", table_name="fill_probability_models")
    op.drop_index("idx_fpm_family", table_name="fill_probability_models")
    op.drop_table("fill_probability_models")

    op.drop_index("idx_bde_token_type_observed", table_name="book_delta_events")
    op.drop_index("idx_bde_token_observed", table_name="book_delta_events")
    op.drop_index("idx_bde_event_type", table_name="book_delta_events")
    op.drop_index("idx_bde_observed_at", table_name="book_delta_events")
    op.drop_index("idx_bde_token_id", table_name="book_delta_events")
    op.drop_index("idx_bde_provider", table_name="book_delta_events")
    op.drop_table("book_delta_events")
