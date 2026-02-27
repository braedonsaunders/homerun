"""Add ML training pipeline tables: snapshots, trained models, recorder config, training jobs.

Revision ID: 202602250003
Revises: 202602250002
Create Date: 2026-02-25 14:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import index_names, table_names


revision = "202602250003"
down_revision = "202602250002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = table_names()

    # ── ml_training_snapshots ──────────────────────────────────────────
    if "ml_training_snapshots" not in tables:
        op.create_table(
            "ml_training_snapshots",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("asset", sa.String(8), nullable=False),
            sa.Column("timeframe", sa.String(8), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("mid_price", sa.Float(), nullable=False),
            sa.Column("up_price", sa.Float(), nullable=True),
            sa.Column("down_price", sa.Float(), nullable=True),
            sa.Column("best_bid", sa.Float(), nullable=True),
            sa.Column("best_ask", sa.Float(), nullable=True),
            sa.Column("spread", sa.Float(), nullable=True),
            sa.Column("combined", sa.Float(), nullable=True),
            sa.Column("liquidity", sa.Float(), nullable=True),
            sa.Column("volume", sa.Float(), nullable=True),
            sa.Column("volume_24h", sa.Float(), nullable=True),
            sa.Column("oracle_price", sa.Float(), nullable=True),
            sa.Column("price_to_beat", sa.Float(), nullable=True),
            sa.Column("seconds_left", sa.Integer(), nullable=True),
            sa.Column("is_live", sa.Boolean(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    existing = index_names("ml_training_snapshots")
    for idx_name, idx_cols in [
        ("idx_mlt_asset_tf_ts", ["asset", "timeframe", "timestamp"]),
        ("idx_mlt_timestamp", ["timestamp"]),
        ("idx_mlt_asset", ["asset"]),
    ]:
        if idx_name not in existing:
            op.create_index(idx_name, "ml_training_snapshots", idx_cols)

    # ── ml_trained_models ──────────────────────────────────────────────
    if "ml_trained_models" not in tables:
        op.create_table(
            "ml_trained_models",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("model_type", sa.String(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'trained'")),
            sa.Column("weights_json", sa.JSON(), nullable=False),
            sa.Column("feature_names", sa.JSON(), nullable=False),
            sa.Column("hyperparams", sa.JSON(), nullable=True),
            sa.Column("assets", sa.JSON(), nullable=False),
            sa.Column("timeframes", sa.JSON(), nullable=False),
            sa.Column("train_accuracy", sa.Float(), nullable=True),
            sa.Column("test_accuracy", sa.Float(), nullable=True),
            sa.Column("test_auc", sa.Float(), nullable=True),
            sa.Column("feature_importance", sa.JSON(), nullable=True),
            sa.Column("train_samples", sa.Integer(), server_default=sa.text("0")),
            sa.Column("test_samples", sa.Integer(), server_default=sa.text("0")),
            sa.Column("training_date_range", sa.JSON(), nullable=True),
            sa.Column("walkforward_results", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("promoted_at", sa.DateTime(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name", "version", name="uq_ml_model_name_version"),
        )

    existing = index_names("ml_trained_models")
    for idx_name, idx_cols in [
        ("idx_mlm_status", ["status"]),
        ("idx_mlm_created", ["created_at"]),
    ]:
        if idx_name not in existing:
            op.create_index(idx_name, "ml_trained_models", idx_cols)

    # ── ml_recorder_config ─────────────────────────────────────────────
    if "ml_recorder_config" not in tables:
        op.create_table(
            "ml_recorder_config",
            sa.Column("id", sa.String(), nullable=False, server_default=sa.text("'default'")),
            sa.Column("is_recording", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("interval_seconds", sa.Integer(), nullable=False, server_default=sa.text("60")),
            sa.Column("retention_days", sa.Integer(), nullable=False, server_default=sa.text("90")),
            sa.Column("assets", sa.JSON(), nullable=False, server_default=sa.text("'[\"btc\",\"eth\",\"sol\",\"xrp\"]'::json")),
            sa.Column("timeframes", sa.JSON(), nullable=False, server_default=sa.text("'[\"5m\",\"15m\",\"1h\",\"4h\"]'::json")),
            sa.Column("schedule_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("schedule_start_utc", sa.String(), nullable=True),
            sa.Column("schedule_end_utc", sa.String(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        # Seed default row
        op.execute(
            sa.text(
                "INSERT INTO ml_recorder_config (id, is_recording, interval_seconds, retention_days) "
                "VALUES ('default', false, 60, 90) "
                "ON CONFLICT (id) DO NOTHING"
            )
        )

    # ── ml_training_jobs ───────────────────────────────────────────────
    if "ml_training_jobs" not in tables:
        op.create_table(
            "ml_training_jobs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'queued'")),
            sa.Column("model_type", sa.String(), nullable=False, server_default=sa.text("'xgboost'")),
            sa.Column("assets", sa.JSON(), nullable=False),
            sa.Column("timeframes", sa.JSON(), nullable=False),
            sa.Column("progress", sa.Float(), server_default=sa.text("0.0")),
            sa.Column("message", sa.String(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("trained_model_id", sa.String(), nullable=True),
            sa.Column("result_summary", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    existing = index_names("ml_training_jobs")
    for idx_name, idx_cols in [
        ("idx_mljob_status", ["status"]),
        ("idx_mljob_created", ["created_at"]),
    ]:
        if idx_name not in existing:
            op.create_index(idx_name, "ml_training_jobs", idx_cols)


def downgrade() -> None:
    for tbl in ("ml_training_jobs", "ml_recorder_config", "ml_trained_models", "ml_training_snapshots"):
        if tbl in table_names():
            op.drop_table(tbl)
