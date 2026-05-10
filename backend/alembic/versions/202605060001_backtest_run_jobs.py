"""Add job-queue lifecycle columns to backtest_runs.

Extends ``backtest_runs`` so it doubles as the queue table for the
dedicated backtest worker process.  The worker polls
``status='queued'`` rows, claims them with ``FOR UPDATE SKIP LOCKED``,
runs the engine, and writes back ``progress`` + ``message`` updates
the UI can render as a live progress bar.

New columns are all NULL-tolerant or have safe defaults so existing
rows continue to work; the legacy sync ``POST /backtest/run`` path
still writes ``status='ok'`` directly without going through the
queue.

Index ``idx_btr_status_created`` makes the worker claim path
selective ("queued rows ordered by created_at ASC") fast.

Revision ID: 202605060001
Revises: 202605050001
Create Date: 2026-05-06
"""

from __future__ import annotations

from alembic import op
from alembic_helpers import safe_add_column, safe_create_index
import sqlalchemy as sa


revision = "202605060001"
down_revision = "202605050001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "backtest_runs",
        sa.Column("payload_json", sa.JSON(), nullable=True),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column("message", sa.String(), nullable=True),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column("worker_id", sa.String(), nullable=True),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column("claimed_at", sa.DateTime(), nullable=True),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column("error", sa.Text(), nullable=True),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column(
            "snapshots_processed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    safe_add_column(
        "backtest_runs",
        sa.Column("snapshots_total_estimate", sa.Integer(), nullable=True),
    )

    safe_create_index(
        "idx_btr_status_created",
        "backtest_runs",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_btr_status_created", table_name="backtest_runs")
    for col in (
        "snapshots_total_estimate",
        "snapshots_processed",
        "error",
        "cancel_requested",
        "claimed_at",
        "worker_id",
        "message",
        "progress",
        "payload_json",
    ):
        op.drop_column("backtest_runs", col)
