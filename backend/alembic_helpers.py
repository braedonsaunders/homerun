"""Shared introspection helpers for Alembic migration scripts.

Placed at the backend root (not inside ``backend/alembic/``) to avoid
shadowing the installed ``alembic`` library package.  Migration scripts
can import these because ``backend/`` is on ``sys.path`` via
``alembic.ini`` ``prepend_sys_path = .``.
"""

from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa

from alembic import op


_log = logging.getLogger("alembic.runtime.migration")


def table_names() -> set[str]:
    """Return the set of existing table names in the database."""
    inspector = sa.inspect(op.get_bind())
    return set(inspector.get_table_names())


def column_names(table_name: str) -> set[str]:
    """Return column names for *table_name*, or empty set if it doesn't exist."""
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def index_names(table_name: str) -> set[str]:
    """Return index names for *table_name*, or empty set if it doesn't exist."""
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {idx["name"] for idx in inspector.get_indexes(table_name) if idx.get("name")}


# ---------------------------------------------------------------------------
# Idempotent op wrappers (Plan 0020).
#
# Why these exist
# ---------------
# The baseline migration ``202602130001_baseline_schema.py`` calls
# ``Base.metadata.create_all(bind=op.get_bind())``, which materialises
# every current ORM column / table / index at revision 1.  Production
# was originally stamped at baseline before any of those columns
# existed, so it has never re-played from base — but a fresh DB
# (developer setup, CI bootstrap, integration test) cannot bootstrap
# without colliding with later ``op.add_column(...)`` /
# ``op.create_table(...)`` / ``op.create_index(...)`` calls.
#
# Reconstructing the historical baseline schema would be archeology
# and risk; instead, every later schema-additive migration is made
# idempotent so it skips ops that the baseline already performed.
# Use these wrappers in new migrations going forward — they are the
# canonical pattern.
# ---------------------------------------------------------------------------


def safe_add_column(table_name: str, column: sa.Column) -> bool:
    """Run ``op.add_column`` only if ``column.name`` is not already on
    *table_name*.

    Returns True if the column was added, False if skipped.  Skips are
    logged at INFO so the migration log retains an audit trail.
    """
    if column.name in column_names(table_name):
        _log.info(
            "safe_add_column: skipping %s.%s — already present",
            table_name,
            column.name,
        )
        return False
    op.add_column(table_name, column)
    return True


def safe_create_table(table_name: str, *columns: Any, **kwargs: Any) -> bool:
    """Run ``op.create_table`` only if *table_name* does not exist.

    Accepts the same positional column args and keyword args (constraints,
    ``schema``, etc.) as ``op.create_table``.  Returns True if created,
    False if skipped.
    """
    if table_name in table_names():
        _log.info(
            "safe_create_table: skipping %s — table already exists",
            table_name,
        )
        return False
    op.create_table(table_name, *columns, **kwargs)
    return True


def safe_create_index(
    index_name: str,
    table_name: str,
    columns: list[Any],
    **kwargs: Any,
) -> bool:
    """Run ``op.create_index`` only if *index_name* is not already on
    *table_name*.

    Accepts the same keyword args (``unique=``, ``postgresql_where=``,
    ``postgresql_using=``, etc.) as ``op.create_index``.  Returns True
    if created, False if skipped.
    """
    if index_name in index_names(table_name):
        _log.info(
            "safe_create_index: skipping %s on %s — index already exists",
            index_name,
            table_name,
        )
        return False
    op.create_index(index_name, table_name, columns, **kwargs)
    return True
