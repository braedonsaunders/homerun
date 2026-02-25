"""Add immutable strategy versions, experiment tables, and version audit columns.

Revision ID: 202602250002
Revises: 202602250001
Create Date: 2026-02-25 12:30:00.000000
"""

from __future__ import annotations

import json
import uuid

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names, index_names, table_names


revision = "202602250002"
down_revision = "202602250001"
branch_labels = None
depends_on = None


def _create_strategy_versions_table() -> None:
    tables = table_names()
    if "strategy_versions" not in tables:
        op.create_table(
            "strategy_versions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("strategy_id", sa.String(), nullable=False),
            sa.Column("strategy_slug", sa.String(), nullable=False),
            sa.Column("source_key", sa.String(), nullable=False, server_default=sa.text("'scanner'")),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("is_latest", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("source_code", sa.Text(), nullable=False),
            sa.Column("class_name", sa.String(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("config_schema", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("aliases", sa.JSON(), nullable=True, server_default=sa.text("'[]'::json")),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("parent_version", sa.Integer(), nullable=True),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["strategy_id"], ["strategies.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("strategy_id", "version", name="uq_strategy_versions_strategy_version"),
        )

    existing = index_names("strategy_versions")
    if "idx_strategy_versions_slug_version" not in existing:
        op.create_index(
            "idx_strategy_versions_slug_version",
            "strategy_versions",
            ["strategy_slug", "version"],
            unique=False,
        )
    if "idx_strategy_versions_strategy_created" not in existing:
        op.create_index(
            "idx_strategy_versions_strategy_created",
            "strategy_versions",
            ["strategy_id", "created_at"],
            unique=False,
        )
    if "idx_strategy_versions_latest" not in existing:
        op.create_index(
            "idx_strategy_versions_latest",
            "strategy_versions",
            ["strategy_id", "is_latest"],
            unique=False,
        )


def _backfill_strategy_versions_from_current_rows() -> None:
    if "strategies" not in table_names() or "strategy_versions" not in table_names():
        return

    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """
            SELECT
                id,
                slug,
                source_key,
                version,
                name,
                description,
                source_code,
                class_name,
                config,
                config_schema,
                aliases,
                enabled,
                is_system,
                sort_order,
                created_at,
                updated_at
            FROM strategies
            """
        )
    ).mappings().all()

    for row in rows:
        strategy_id = str(row.get("id") or "").strip()
        if not strategy_id:
            continue
        version = int(row.get("version") or 1)
        existing = bind.execute(
            sa.text(
                """
                SELECT 1
                FROM strategy_versions
                WHERE strategy_id = :strategy_id AND version = :version
                """
            ),
            {"strategy_id": strategy_id, "version": version},
        ).fetchone()
        if existing is None:
            bind.execute(
                sa.text(
                    """
                    INSERT INTO strategy_versions (
                        id,
                        strategy_id,
                        strategy_slug,
                        source_key,
                        version,
                        is_latest,
                        name,
                        description,
                        source_code,
                        class_name,
                        config,
                        config_schema,
                        aliases,
                        enabled,
                        is_system,
                        sort_order,
                        parent_version,
                        created_by,
                        reason,
                        created_at
                    ) VALUES (
                        :id,
                        :strategy_id,
                        :strategy_slug,
                        :source_key,
                        :version,
                        TRUE,
                        :name,
                        :description,
                        :source_code,
                        :class_name,
                        :config,
                        :config_schema,
                        :aliases,
                        :enabled,
                        :is_system,
                        :sort_order,
                        NULL,
                        NULL,
                        'seed_from_strategy_row',
                        :created_at
                    )
                    """
                ),
                {
                    "id": uuid.uuid4().hex,
                    "strategy_id": strategy_id,
                    "strategy_slug": str(row.get("slug") or "").strip(),
                    "source_key": str(row.get("source_key") or "scanner").strip().lower() or "scanner",
                    "version": version,
                    "name": str(row.get("name") or "").strip() or str(row.get("slug") or "strategy"),
                    "description": row.get("description"),
                    "source_code": str(row.get("source_code") or ""),
                    "class_name": row.get("class_name"),
                    "config": json.dumps(row.get("config") if isinstance(row.get("config"), dict) else {}),
                    "config_schema": json.dumps(row.get("config_schema") if isinstance(row.get("config_schema"), dict) else {}),
                    "aliases": json.dumps(row.get("aliases") if isinstance(row.get("aliases"), list) else []),
                    "enabled": bool(row.get("enabled")),
                    "is_system": bool(row.get("is_system")),
                    "sort_order": int(row.get("sort_order") or 0),
                    "created_at": row.get("updated_at") or row.get("created_at"),
                },
            )

        bind.execute(
            sa.text(
                """
                UPDATE strategy_versions
                SET is_latest = CASE WHEN version = :version THEN TRUE ELSE FALSE END
                WHERE strategy_id = :strategy_id
                """
            ),
            {"strategy_id": strategy_id, "version": version},
        )


def _add_version_audit_columns() -> None:
    tables = table_names()

    if "trader_decisions" in tables:
        existing_cols = column_names("trader_decisions")
        if "strategy_version" not in existing_cols:
            op.add_column("trader_decisions", sa.Column("strategy_version", sa.Integer(), nullable=True))

        existing_idx = index_names("trader_decisions")
        if "idx_trader_decisions_strategy_version" not in existing_idx:
            op.create_index(
                "idx_trader_decisions_strategy_version",
                "trader_decisions",
                ["strategy_version"],
                unique=False,
            )

    if "trader_orders" in tables:
        existing_cols = column_names("trader_orders")
        if "strategy_key" not in existing_cols:
            op.add_column("trader_orders", sa.Column("strategy_key", sa.String(), nullable=True))
        if "strategy_version" not in existing_cols:
            op.add_column("trader_orders", sa.Column("strategy_version", sa.Integer(), nullable=True))

        existing_idx = index_names("trader_orders")
        if "idx_trader_orders_strategy_key" not in existing_idx:
            op.create_index(
                "idx_trader_orders_strategy_key",
                "trader_orders",
                ["strategy_key"],
                unique=False,
            )
        if "idx_trader_orders_strategy_version" not in existing_idx:
            op.create_index(
                "idx_trader_orders_strategy_version",
                "trader_orders",
                ["strategy_version"],
                unique=False,
            )

    if "execution_sessions" in tables:
        existing_cols = column_names("execution_sessions")
        if "strategy_version" not in existing_cols:
            op.add_column("execution_sessions", sa.Column("strategy_version", sa.Integer(), nullable=True))

        existing_idx = index_names("execution_sessions")
        if "idx_execution_sessions_strategy_version" not in existing_idx:
            op.create_index(
                "idx_execution_sessions_strategy_version",
                "execution_sessions",
                ["strategy_version"],
                unique=False,
            )


def _backfill_version_audit_columns() -> None:
    tables = table_names()
    bind = op.get_bind()

    if "trader_decisions" in tables and "strategies" in tables:
        bind.execute(
            sa.text(
                """
                UPDATE trader_decisions AS d
                SET strategy_version = s.version
                FROM strategies AS s
                WHERE d.strategy_version IS NULL
                  AND LOWER(TRIM(COALESCE(d.strategy_key, ''))) = LOWER(TRIM(COALESCE(s.slug, '')))
                """
            )
        )

    if "execution_sessions" in tables and "strategies" in tables:
        bind.execute(
            sa.text(
                """
                UPDATE execution_sessions AS e
                SET strategy_version = s.version
                FROM strategies AS s
                WHERE e.strategy_version IS NULL
                  AND LOWER(TRIM(COALESCE(e.strategy_key, ''))) = LOWER(TRIM(COALESCE(s.slug, '')))
                """
            )
        )

    if "trader_orders" in tables:
        bind.execute(
            sa.text(
                """
                UPDATE trader_orders AS o
                SET strategy_key = d.strategy_key,
                    strategy_version = COALESCE(d.strategy_version, o.strategy_version)
                FROM trader_decisions AS d
                WHERE o.decision_id = d.id
                  AND (
                    o.strategy_key IS NULL
                    OR TRIM(COALESCE(o.strategy_key, '')) = ''
                    OR o.strategy_version IS NULL
                  )
                """
            )
        )
        if "execution_sessions" in tables:
            bind.execute(
                sa.text(
                    """
                    UPDATE trader_orders AS o
                    SET strategy_key = COALESCE(NULLIF(TRIM(COALESCE(o.strategy_key, '')), ''), e.strategy_key),
                        strategy_version = COALESCE(o.strategy_version, e.strategy_version)
                    FROM execution_sessions AS e
                    WHERE TRIM(COALESCE((o.payload_json -> 'execution_session' ->> 'session_id'), '')) = e.id
                      AND (
                        o.strategy_key IS NULL
                        OR TRIM(COALESCE(o.strategy_key, '')) = ''
                        OR o.strategy_version IS NULL
                      )
                    """
                )
            )
        if "strategies" in tables:
            bind.execute(
                sa.text(
                    """
                    UPDATE trader_orders AS o
                    SET strategy_version = s.version
                    FROM strategies AS s
                    WHERE o.strategy_version IS NULL
                      AND LOWER(TRIM(COALESCE(o.strategy_key, ''))) = LOWER(TRIM(COALESCE(s.slug, '')))
                    """
                )
            )


def _create_strategy_experiment_tables() -> None:
    tables = table_names()

    if "strategy_experiments" not in tables:
        op.create_table(
            "strategy_experiments",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("source_key", sa.String(), nullable=False),
            sa.Column("strategy_key", sa.String(), nullable=False),
            sa.Column("control_version", sa.Integer(), nullable=False),
            sa.Column("candidate_version", sa.Integer(), nullable=False),
            sa.Column("candidate_allocation_pct", sa.Float(), nullable=False, server_default=sa.text("50.0")),
            sa.Column("scope_json", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'active'")),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("promoted_version", sa.Integer(), nullable=True),
            sa.Column("ended_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    existing = index_names("strategy_experiments")
    if "idx_strategy_experiments_strategy_status" not in existing:
        op.create_index(
            "idx_strategy_experiments_strategy_status",
            "strategy_experiments",
            ["strategy_key", "status"],
            unique=False,
        )
    if "idx_strategy_experiments_source_status" not in existing:
        op.create_index(
            "idx_strategy_experiments_source_status",
            "strategy_experiments",
            ["source_key", "status"],
            unique=False,
        )
    if "idx_strategy_experiments_created" not in existing:
        op.create_index(
            "idx_strategy_experiments_created",
            "strategy_experiments",
            ["created_at"],
            unique=False,
        )

    if "strategy_experiment_assignments" not in tables:
        op.create_table(
            "strategy_experiment_assignments",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("experiment_id", sa.String(), nullable=False),
            sa.Column("trader_id", sa.String(), nullable=True),
            sa.Column("signal_id", sa.String(), nullable=True),
            sa.Column("decision_id", sa.String(), nullable=True),
            sa.Column("order_id", sa.String(), nullable=True),
            sa.Column("source_key", sa.String(), nullable=False),
            sa.Column("strategy_key", sa.String(), nullable=False),
            sa.Column("strategy_version", sa.Integer(), nullable=False),
            sa.Column("assignment_group", sa.String(), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["experiment_id"], ["strategy_experiments.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["trader_id"], ["traders.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["signal_id"], ["trade_signals.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["decision_id"], ["trader_decisions.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["order_id"], ["trader_orders.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "experiment_id",
                "trader_id",
                "signal_id",
                name="uq_strategy_experiment_assignment_signal",
            ),
        )

    existing_assign_idx = index_names("strategy_experiment_assignments")
    if "idx_strategy_experiment_assignments_group" not in existing_assign_idx:
        op.create_index(
            "idx_strategy_experiment_assignments_group",
            "strategy_experiment_assignments",
            ["experiment_id", "assignment_group"],
            unique=False,
        )


def upgrade() -> None:
    _create_strategy_versions_table()
    _backfill_strategy_versions_from_current_rows()
    _add_version_audit_columns()
    _backfill_version_audit_columns()
    _create_strategy_experiment_tables()


def downgrade() -> None:
    pass
