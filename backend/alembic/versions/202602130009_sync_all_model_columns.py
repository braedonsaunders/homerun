"""Dynamically add any columns present in AppSettings model but missing from DB.

This is a catch-all migration that inspects the ORM model at runtime and adds
every column that the database table is missing.  It uses the same idempotent
``_column_names()`` guard used by earlier migrations, so it is safe to re-run
even if some columns already exist.

Revision ID: 202602130009
Revises: 202602130005
Create Date: 2026-02-13 16:00:00.000000
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import op
import sqlalchemy as sa

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# revision identifiers, used by Alembic.
revision = "202602130009"
down_revision = "202602130005"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    from models.database import AppSettings
    from models.model_registry import register_all_models

    register_all_models()

    table_name = AppSettings.__tablename__
    existing = _column_names(table_name)

    if not existing:
        # Table doesn't exist yet — baseline create_all will handle it.
        return

    mapper = sa.inspect(AppSettings)
    for attr in mapper.columns:
        if attr.name in existing:
            continue

        # Build a new Column matching the ORM definition.
        col_type = attr.type.copy() if hasattr(attr.type, "copy") else type(attr.type)()
        nullable = attr.nullable if attr.nullable is not None else True

        # Derive a sensible server_default from the ORM default so that
        # existing rows get a value.
        server_default = None
        if attr.default is not None and attr.default.is_scalar:
            py_val = attr.default.arg
            if isinstance(py_val, bool):
                server_default = sa.text("1" if py_val else "0")
            elif isinstance(py_val, (int, float)):
                server_default = sa.text(str(py_val))
            elif isinstance(py_val, str):
                server_default = sa.text(f"'{py_val}'")

        new_col = sa.Column(
            attr.name,
            col_type,
            nullable=nullable,
            server_default=server_default,
        )
        op.add_column(table_name, new_col)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass
