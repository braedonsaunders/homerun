"""DB-native trader strategies + execution simulator persistence.

Revision ID: 202602160001
Revises: 202602140005
Create Date: 2026-02-16 00:00:01.000000
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from alembic import op
import sqlalchemy as sa
from alembic_helpers import table_names, index_names

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.opportunity_strategy_catalog import build_system_opportunity_strategy_rows  # noqa: E402


# revision identifiers, used by Alembic.
revision = "202602160001"
down_revision = "202602140005"
branch_labels = None
depends_on = None


def _create_trader_strategy_definitions() -> None:
    if "trader_strategy_definitions" in table_names():
        return

    op.create_table(
        "trader_strategy_definitions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("strategy_key", sa.String(), nullable=False),
        sa.Column("source_key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("class_name", sa.String(), nullable=False),
        sa.Column("source_code", sa.Text(), nullable=False),
        sa.Column("default_params_json", sa.JSON(), nullable=True),
        sa.Column("param_schema_json", sa.JSON(), nullable=True),
        sa.Column("aliases_json", sa.JSON(), nullable=True),
        sa.Column("is_system", sa.Boolean(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("strategy_key"),
    )

    op.create_index(
        "idx_trader_strategy_definitions_strategy_key",
        "trader_strategy_definitions",
        ["strategy_key"],
        unique=False,
    )
    op.create_index(
        "idx_trader_strategy_definitions_source_key",
        "trader_strategy_definitions",
        ["source_key"],
        unique=False,
    )
    op.create_index(
        "idx_trader_strategy_definitions_enabled",
        "trader_strategy_definitions",
        ["enabled"],
        unique=False,
    )
    op.create_index(
        "idx_trader_strategy_definitions_status",
        "trader_strategy_definitions",
        ["status"],
        unique=False,
    )


def _create_trade_signal_emissions() -> None:
    if "trade_signal_emissions" in table_names():
        return

    op.create_table(
        "trade_signal_emissions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_item_id", sa.String(), nullable=True),
        sa.Column("signal_type", sa.String(), nullable=False),
        sa.Column("strategy_type", sa.String(), nullable=True),
        sa.Column("market_id", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("effective_price", sa.Float(), nullable=True),
        sa.Column("edge_percent", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("liquidity", sa.Float(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("dedupe_key", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("snapshot_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["signal_id"], ["trade_signals.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "idx_trade_signal_emissions_source_created",
        "trade_signal_emissions",
        ["source", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_trade_signal_emissions_signal_created",
        "trade_signal_emissions",
        ["signal_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_trade_signal_emissions_created_at",
        "trade_signal_emissions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_trade_signal_emissions_event_type",
        "trade_signal_emissions",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        "ix_trade_signal_emissions_market_id",
        "trade_signal_emissions",
        ["market_id"],
        unique=False,
    )
    op.create_index(
        "ix_trade_signal_emissions_signal_id",
        "trade_signal_emissions",
        ["signal_id"],
        unique=False,
    )
    op.create_index(
        "ix_trade_signal_emissions_source",
        "trade_signal_emissions",
        ["source"],
        unique=False,
    )


def _create_execution_sim_runs() -> None:
    if "execution_sim_runs" in table_names():
        return

    op.create_table(
        "execution_sim_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("strategy_key", sa.String(), nullable=False),
        sa.Column("source_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("market_scope_json", sa.JSON(), nullable=True),
        sa.Column("params_json", sa.JSON(), nullable=True),
        sa.Column("requested_start_at", sa.DateTime(), nullable=True),
        sa.Column("requested_end_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["validation_jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_execution_sim_runs_status", "execution_sim_runs", ["status"], unique=False)
    op.create_index("idx_execution_sim_runs_created", "execution_sim_runs", ["created_at"], unique=False)
    op.create_index("ix_execution_sim_runs_job_id", "execution_sim_runs", ["job_id"], unique=False)
    op.create_index("ix_execution_sim_runs_source_key", "execution_sim_runs", ["source_key"], unique=False)
    op.create_index("ix_execution_sim_runs_strategy_key", "execution_sim_runs", ["strategy_key"], unique=False)


def _create_execution_sim_events() -> None:
    if "execution_sim_events" in table_names():
        return

    op.create_table(
        "execution_sim_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_at", sa.DateTime(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=True),
        sa.Column("market_id", sa.String(), nullable=True),
        sa.Column("direction", sa.String(), nullable=True),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("notional_usd", sa.Float(), nullable=True),
        sa.Column("fees_usd", sa.Float(), nullable=True),
        sa.Column("slippage_bps", sa.Float(), nullable=True),
        sa.Column("realized_pnl_usd", sa.Float(), nullable=True),
        sa.Column("unrealized_pnl_usd", sa.Float(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["execution_sim_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_execution_sim_events_run_sequence"),
    )

    op.create_index(
        "idx_execution_sim_events_run_event_at", "execution_sim_events", ["run_id", "event_at"], unique=False
    )
    op.create_index("ix_execution_sim_events_event_at", "execution_sim_events", ["event_at"], unique=False)
    op.create_index("ix_execution_sim_events_event_type", "execution_sim_events", ["event_type"], unique=False)
    op.create_index("ix_execution_sim_events_market_id", "execution_sim_events", ["market_id"], unique=False)
    op.create_index("ix_execution_sim_events_run_id", "execution_sim_events", ["run_id"], unique=False)
    op.create_index("ix_execution_sim_events_signal_id", "execution_sim_events", ["signal_id"], unique=False)


def _seed_system_trader_strategies() -> None:
    bind = op.get_bind()
    table = sa.table(
        "trader_strategy_definitions",
        sa.column("id", sa.String()),
        sa.column("strategy_key", sa.String()),
        sa.column("source_key", sa.String()),
        sa.column("label", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("class_name", sa.String()),
        sa.column("source_code", sa.Text()),
        sa.column("default_params_json", sa.JSON()),
        sa.column("param_schema_json", sa.JSON()),
        sa.column("aliases_json", sa.JSON()),
        sa.column("is_system", sa.Boolean()),
        sa.column("enabled", sa.Boolean()),
        sa.column("status", sa.String()),
        sa.column("error_message", sa.Text()),
        sa.column("version", sa.Integer()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )

    now = datetime.utcnow()
    rows = build_system_opportunity_strategy_rows()
    for row in rows:
        strategy_key = str(row.get("strategy_key") or row.get("slug") or "").strip()
        if not strategy_key:
            continue
        label = row.get("label") if row.get("label") is not None else row.get("name")
        default_params_json = (
            row.get("default_params_json") if row.get("default_params_json") is not None else row.get("config")
        )
        param_schema_json = row.get("param_schema_json") if row.get("param_schema_json") is not None else row.get(
            "config_schema"
        )
        aliases_json = row.get("aliases_json") if row.get("aliases_json") is not None else row.get("aliases")

        existing = (
            bind.execute(
                sa.select(
                    table.c.id,
                    table.c.is_system,
                    table.c.source_code,
                    table.c.version,
                ).where(table.c.strategy_key == strategy_key)
            )
            .mappings()
            .first()
        )

        if existing is None:
            bind.execute(
                table.insert().values(
                    id=row["id"],
                    strategy_key=strategy_key,
                    source_key=row["source_key"],
                    label=label,
                    description=row["description"],
                    class_name=row["class_name"],
                    source_code=row["source_code"],
                    default_params_json=default_params_json,
                    param_schema_json=param_schema_json,
                    aliases_json=aliases_json,
                    is_system=True,
                    enabled=True,
                    status="unloaded",
                    error_message=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            continue

        if bool(existing.get("is_system")):
            bind.execute(
                table.update()
                .where(table.c.strategy_key == strategy_key)
                .values(
                    source_key=row["source_key"],
                    label=label,
                    description=row["description"],
                    class_name=row["class_name"],
                    source_code=row["source_code"] or existing.get("source_code"),
                    default_params_json=default_params_json,
                    param_schema_json=param_schema_json,
                    aliases_json=aliases_json,
                    updated_at=now,
                )
            )


def upgrade() -> None:
    _create_trader_strategy_definitions()

    # Backfill missing indexes if table already exists.
    existing_indexes = index_names("trader_strategy_definitions")
    if "idx_trader_strategy_definitions_strategy_key" not in existing_indexes:
        op.create_index(
            "idx_trader_strategy_definitions_strategy_key",
            "trader_strategy_definitions",
            ["strategy_key"],
            unique=False,
        )
    if "idx_trader_strategy_definitions_source_key" not in existing_indexes:
        op.create_index(
            "idx_trader_strategy_definitions_source_key",
            "trader_strategy_definitions",
            ["source_key"],
            unique=False,
        )
    if "idx_trader_strategy_definitions_enabled" not in existing_indexes:
        op.create_index(
            "idx_trader_strategy_definitions_enabled",
            "trader_strategy_definitions",
            ["enabled"],
            unique=False,
        )
    if "idx_trader_strategy_definitions_status" not in existing_indexes:
        op.create_index(
            "idx_trader_strategy_definitions_status",
            "trader_strategy_definitions",
            ["status"],
            unique=False,
        )

    _create_trade_signal_emissions()
    _create_execution_sim_runs()
    _create_execution_sim_events()
    _seed_system_trader_strategies()


def downgrade() -> None:
    if "execution_sim_events" in table_names():
        op.drop_table("execution_sim_events")
    if "execution_sim_runs" in table_names():
        op.drop_table("execution_sim_runs")
    if "trade_signal_emissions" in table_names():
        op.drop_table("trade_signal_emissions")
    if "trader_strategy_definitions" in table_names():
        op.drop_table("trader_strategy_definitions")
