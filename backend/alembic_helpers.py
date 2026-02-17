"""Shared introspection helpers for Alembic migration scripts.

Placed at the backend root (not inside ``backend/alembic/``) to avoid
shadowing the installed ``alembic`` library package.  Migration scripts
can import these because ``backend/`` is on ``sys.path`` via
``alembic.ini`` ``prepend_sys_path = .``.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


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
