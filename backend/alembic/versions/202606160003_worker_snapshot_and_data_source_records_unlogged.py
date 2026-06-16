"""Make worker_snapshot + data_source_records UNLOGGED.

With trade_signals converted to UNLOGGED (202606160002), the next-largest WAL
sources on the shared Postgres instance are two more derived/cache tables:

  worker_snapshot      — latest per-worker status telemetry for API/WS health
                         surfaces; one row per worker, re-written every cycle
                         (~2.3 GB WAL from heartbeat upserts). No FKs, no
                         consumer depends on crash persistence — a truncated
                         table just reads empty until the next heartbeat.

  data_source_records  — normalized raw ingestion cache; a transient,
                         retention-capped table idempotently upserted on
                         (source_slug, external_id) and rebuilt from external
                         sources every run (~2.5 GB WAL from observed_at/
                         ingested_at refreshes at 0% HOT). No processed/dedup
                         flag — signal dedup lives in the already-UNLOGGED
                         trade_signals — so empty-after-crash causes only
                         harmless re-ingestion on the next source run. Its
                         outbound FK to the permanent ``data_sources`` parent is
                         allowed (unlogged child -> permanent parent is legal).

Both fit the established UNLOGGED criterion from 202605230001 ("DERIVED /
CACHE / TELEMETRY that rebuild from their source … none hold orders/positions/
financial state"); they were simply out of scope for that pass's WAL list.

IDEMPOTENT + LOCK-SAFE (mirrors 202605230001): skip any table already at the
target persistence via a cheap pg_class read (AccessShareLock, never blocked
by writers); only convert tables that still need it, dropping lock_timeout /
statement_timeout so the ALTER waits for the lock + rewrite instead of failing
fast under the boot lock_timeout. data_source_records is ~1 GB, so the
ACCESS EXCLUSIVE rewrite should run with the app stopped. Fresh installs
create both UNLOGGED via the model ``prefixes`` + create_all.

Revision ID: 202606160003
Revises: 202606160002
Create Date: 2026-06-16
"""

import sqlalchemy as sa
from alembic import op


revision = "202606160003"
down_revision = "202606160002"
branch_labels = None
depends_on = None


_TABLES = ("worker_snapshot", "data_source_records")


def _flip_persistence(target_char: str, alter_clause: str) -> None:
    bind = op.get_bind()
    pending = []
    for table in _TABLES:
        current = bind.execute(
            sa.text(
                "SELECT relpersistence FROM pg_class "
                "WHERE relname = :t AND relkind = 'r'"
            ),
            {"t": table},
        ).scalar()
        # None = table absent (skip); already-target = skip (no lock taken).
        if current is not None and current != target_char:
            pending.append(table)
    if not pending:
        return
    op.execute("SET LOCAL lock_timeout = 0")
    op.execute("SET LOCAL statement_timeout = 0")
    for table in pending:
        op.execute(f"ALTER TABLE {table} {alter_clause}")


def upgrade() -> None:
    _flip_persistence("u", "SET UNLOGGED")


def downgrade() -> None:
    _flip_persistence("p", "SET LOGGED")
