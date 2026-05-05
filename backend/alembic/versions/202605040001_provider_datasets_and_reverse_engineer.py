"""Provider datasets, provider import jobs, strategy reverse-engineer jobs

Revision ID: 202605040001
Revises: 202605030001
Create Date: 2026-05-04

Adds four tables that power the on-demand external data import pipeline
and the strategy reverse-engineering agent:

  * ``provider_datasets``                       — catalog of imported
    datasets from external vendors (polybacktest, etc.).  The actual
    snapshot rows continue to live in ``market_microstructure_snapshots``
    keyed by ``provider`` + synthetic ``token_id``; this table is the
    user-friendly index.
  * ``provider_import_jobs``                    — async job queue worked
    off by ``workers/provider_import_worker.py`` on the discovery plane.
  * ``strategy_reverse_engineer_jobs``          — long-running LLM agent
    job that reverse-engineers a wallet's trading strategy.
  * ``strategy_reverse_engineer_iterations``    — per-iteration audit
    of the agent loop (candidate code, score, critique, cost).

Also extends ``app_settings`` with reverse-engineer + polybacktest
configuration columns that the AI Settings UI surfaces (no hidden
hardcoded defaults — every knob is user-tunable).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "202605040001"
down_revision = "202605030001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── provider_datasets ──────────────────────────────────────────────
    op.create_table(
        "provider_datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("coin", sa.String(), nullable=True),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("external_slug", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column(
            "asset_class",
            sa.String(),
            nullable=False,
            server_default="prediction",
        ),
        sa.Column("token_ids_json", sa.JSON(), nullable=False),
        sa.Column("start_ts", sa.DateTime(), nullable=True),
        sa.Column("end_ts", sa.DateTime(), nullable=True),
        sa.Column("snapshot_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("trade_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_imported_at", sa.DateTime(), nullable=True),
        sa.Column("last_import_job_id", sa.String(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "provider", "external_id", name="uq_provider_dataset_provider_extid"
        ),
    )
    op.create_index(
        "ix_provider_datasets_provider", "provider_datasets", ["provider"]
    )
    op.create_index("ix_provider_datasets_coin", "provider_datasets", ["coin"])
    op.create_index(
        "ix_provider_datasets_start_ts", "provider_datasets", ["start_ts"]
    )
    op.create_index("ix_provider_datasets_end_ts", "provider_datasets", ["end_ts"])
    op.create_index(
        "idx_provider_dataset_provider_coin",
        "provider_datasets",
        ["provider", "coin"],
    )
    op.create_index(
        "idx_provider_dataset_updated", "provider_datasets", ["updated_at"]
    )

    # ── provider_import_jobs ───────────────────────────────────────────
    op.create_table(
        "provider_import_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("message", sa.String(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "snapshots_fetched", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "snapshots_inserted", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("trades_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("api_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "bytes_downloaded", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_provider_import_jobs_provider", "provider_import_jobs", ["provider"]
    )
    op.create_index(
        "ix_provider_import_jobs_status", "provider_import_jobs", ["status"]
    )
    op.create_index(
        "idx_provider_import_status", "provider_import_jobs", ["status"]
    )
    op.create_index(
        "idx_provider_import_created", "provider_import_jobs", ["created_at"]
    )

    # ── strategy_reverse_engineer_jobs ─────────────────────────────────
    op.create_table(
        "strategy_reverse_engineer_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("wallet_address", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column(
            "data_source_kind",
            sa.String(),
            nullable=False,
            server_default="auto",
        ),
        sa.Column("recording_session_ids_json", sa.JSON(), nullable=True),
        sa.Column("provider_dataset_ids_json", sa.JSON(), nullable=True),
        sa.Column("llm_model", sa.String(), nullable=True),
        sa.Column(
            "max_iterations", sa.Integer(), nullable=False, server_default="10"
        ),
        sa.Column(
            "target_score", sa.Float(), nullable=False, server_default="0.7"
        ),
        sa.Column("max_cost_usd", sa.Float(), nullable=True),
        sa.Column(
            "max_wallet_trades", sa.Integer(), nullable=False, server_default="2000"
        ),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "current_iteration", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("activity", sa.String(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("wallet_profile_json", sa.JSON(), nullable=True),
        sa.Column(
            "wallet_trade_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("wallet_window_start", sa.DateTime(), nullable=True),
        sa.Column("wallet_window_end", sa.DateTime(), nullable=True),
        sa.Column("best_iteration_id", sa.String(), nullable=True),
        sa.Column("best_score", sa.Float(), nullable=True),
        sa.Column("best_strategy_code", sa.Text(), nullable=True),
        sa.Column("best_strategy_class", sa.String(), nullable=True),
        sa.Column("best_backtest_run_id", sa.String(), nullable=True),
        sa.Column(
            "total_input_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_output_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_cost_usd", sa.Float(), nullable=False, server_default="0"
        ),
        sa.Column("promoted_strategy_id", sa.String(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_strategy_reverse_engineer_jobs_wallet_address",
        "strategy_reverse_engineer_jobs",
        ["wallet_address"],
    )
    op.create_index(
        "ix_strategy_reverse_engineer_jobs_status",
        "strategy_reverse_engineer_jobs",
        ["status"],
    )
    op.create_index(
        "idx_re_jobs_wallet",
        "strategy_reverse_engineer_jobs",
        ["wallet_address"],
    )
    op.create_index(
        "idx_re_jobs_status",
        "strategy_reverse_engineer_jobs",
        ["status"],
    )
    op.create_index(
        "idx_re_jobs_created",
        "strategy_reverse_engineer_jobs",
        ["created_at"],
    )

    # ── strategy_reverse_engineer_iterations ───────────────────────────
    op.create_table(
        "strategy_reverse_engineer_iterations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(),
            sa.ForeignKey(
                "strategy_reverse_engineer_jobs.id", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("strategy_code", sa.Text(), nullable=True),
        sa.Column("strategy_class", sa.String(), nullable=True),
        sa.Column("backtest_run_id", sa.String(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("score_breakdown_json", sa.JSON(), nullable=True),
        sa.Column("divergence_summary", sa.Text(), nullable=True),
        sa.Column("llm_critique", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "input_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "output_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("job_id", "iteration", name="uq_re_iter_job_iter"),
    )
    op.create_index(
        "ix_strategy_reverse_engineer_iterations_job_id",
        "strategy_reverse_engineer_iterations",
        ["job_id"],
    )
    op.create_index(
        "idx_re_iter_job_iter",
        "strategy_reverse_engineer_iterations",
        ["job_id", "iteration"],
    )

    # ── app_settings extensions ────────────────────────────────────────
    # Polybacktest API key + UI-tunable defaults for the reverse-engineer
    # agent.  All nullable so existing rows upgrade cleanly.
    op.add_column(
        "app_settings", sa.Column("polybacktest_api_key", sa.String(), nullable=True)
    )
    op.add_column(
        "app_settings",
        sa.Column("polybacktest_base_url", sa.String(), nullable=True),
    )
    # Note: the *default model* for reverse-engineer is stored in
    # ``app_settings.llm_model_assignments['strategy_reverse_engineer']``
    # (the existing per-purpose JSON column the AI → Models view manages),
    # so we do NOT add a dedicated column for it here.
    op.add_column(
        "app_settings",
        sa.Column(
            "reverse_engineer_max_iterations",
            sa.Integer(),
            nullable=True,
        ),
    )
    op.add_column(
        "app_settings",
        sa.Column(
            "reverse_engineer_target_score",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "app_settings",
        sa.Column(
            "reverse_engineer_max_cost_usd",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "app_settings",
        sa.Column(
            "reverse_engineer_max_wallet_trades",
            sa.Integer(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "reverse_engineer_max_wallet_trades")
    op.drop_column("app_settings", "reverse_engineer_max_cost_usd")
    op.drop_column("app_settings", "reverse_engineer_target_score")
    op.drop_column("app_settings", "reverse_engineer_max_iterations")
    op.drop_column("app_settings", "polybacktest_base_url")
    op.drop_column("app_settings", "polybacktest_api_key")

    op.drop_index(
        "idx_re_iter_job_iter",
        table_name="strategy_reverse_engineer_iterations",
    )
    op.drop_index(
        "ix_strategy_reverse_engineer_iterations_job_id",
        table_name="strategy_reverse_engineer_iterations",
    )
    op.drop_table("strategy_reverse_engineer_iterations")

    op.drop_index(
        "idx_re_jobs_created",
        table_name="strategy_reverse_engineer_jobs",
    )
    op.drop_index(
        "idx_re_jobs_status",
        table_name="strategy_reverse_engineer_jobs",
    )
    op.drop_index(
        "idx_re_jobs_wallet",
        table_name="strategy_reverse_engineer_jobs",
    )
    op.drop_index(
        "ix_strategy_reverse_engineer_jobs_status",
        table_name="strategy_reverse_engineer_jobs",
    )
    op.drop_index(
        "ix_strategy_reverse_engineer_jobs_wallet_address",
        table_name="strategy_reverse_engineer_jobs",
    )
    op.drop_table("strategy_reverse_engineer_jobs")

    op.drop_index(
        "idx_provider_import_created", table_name="provider_import_jobs"
    )
    op.drop_index(
        "idx_provider_import_status", table_name="provider_import_jobs"
    )
    op.drop_index(
        "ix_provider_import_jobs_status", table_name="provider_import_jobs"
    )
    op.drop_index(
        "ix_provider_import_jobs_provider", table_name="provider_import_jobs"
    )
    op.drop_table("provider_import_jobs")

    op.drop_index(
        "idx_provider_dataset_updated", table_name="provider_datasets"
    )
    op.drop_index(
        "idx_provider_dataset_provider_coin", table_name="provider_datasets"
    )
    op.drop_index("ix_provider_datasets_end_ts", table_name="provider_datasets")
    op.drop_index("ix_provider_datasets_start_ts", table_name="provider_datasets")
    op.drop_index("ix_provider_datasets_coin", table_name="provider_datasets")
    op.drop_index("ix_provider_datasets_provider", table_name="provider_datasets")
    op.drop_table("provider_datasets")
