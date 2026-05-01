"""recording_sessions table for on-demand market data captures

Revision ID: 202605010002
Revises: 202605010001
Create Date: 2026-05-01

A "recording session" is a user-defined, scoped capture of market data:
pick markets (token_ids, condition_ids, or event slugs), set the tick
interval, choose what to capture (book / trade / deltas), and either
run now or schedule a window.  Sessions feed the unified backtester
via ``session_id`` — the backtester scopes its replay to the session's
target tokens + time window.

The session row is metadata only.  The actual captured rows continue
to live in ``MarketMicrostructureSnapshot`` / ``BookDeltaEvent`` with
no schema change there — the session's ``target_token_ids_json`` +
``started_at..ended_at`` window pins the rows to a session implicitly.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "202605010002"
down_revision = "202605010001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recording_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # status: pending | scheduled | running | paused | completed | failed | cancelled
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        # capture targeting
        sa.Column("platform", sa.String(), nullable=False, server_default="polymarket"),
        sa.Column("target_kind", sa.String(), nullable=False, server_default="token"),  # token|condition|event
        sa.Column("target_values_json", sa.JSON(), nullable=False),  # raw operator input
        sa.Column("target_token_ids_json", sa.JSON(), nullable=True),  # resolved tokens (post-resolve)
        # capture types: subset of {book, trade, delta}
        sa.Column("capture_types_json", sa.JSON(), nullable=False),
        sa.Column("tick_interval_ms", sa.Integer(), nullable=False, server_default="500"),
        sa.Column("retention_days", sa.Integer(), nullable=True),
        # scheduling
        sa.Column("scheduled_start_at", sa.DateTime(), nullable=True),
        sa.Column("scheduled_end_at", sa.DateTime(), nullable=True),
        sa.Column("max_duration_seconds", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        # progress
        sa.Column("rows_captured", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_capture_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        # arbitrary extra config (per-platform tweaks, depth limits, etc.)
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_recording_sessions_status",
        "recording_sessions",
        ["status"],
    )
    op.create_index(
        "idx_recording_sessions_started_at",
        "recording_sessions",
        ["started_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_recording_sessions_started_at", table_name="recording_sessions")
    op.drop_index("idx_recording_sessions_status", table_name="recording_sessions")
    op.drop_table("recording_sessions")
