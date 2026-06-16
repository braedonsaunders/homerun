"""Make trade_signals UNLOGGED — it is the dominant WAL source.

The 2026-06-15/16 soak showed the producer UPSERT firehose on
``trade_signals`` generating ~52% of ALL WAL on the shared Postgres
instance: a tiny ~8.8K-live-row table absorbing ~3.7M updates (92% HOT),
each rewriting two TOASTed JSON blobs (payload_json, strategy_context_json)
= ~20 GB WAL. The same WAL/HOT pressure already drove a long series of
index removals on this table (202605090005/6/7, 202605100001).

``trade_signals`` is a continuously re-emitted, TTL-bounded ephemeral
signal cache. It holds no orders / positions / financial state; nothing
FK-references it; every reader tolerates an empty table (None-guards /
[]-fallbacks); both startup hydration paths (signal_cache.bootstrap_from_db,
intent_runtime.hydrate_from_db) soft-fail on zero rows; and the durable
consumption cursor lives in the SEPARATE LOGGED ``trader_signal_cursor``.
The scanner reconstructs the full table within one cycle, so truncation on
an unclean crash is operationally identical to a fresh cleanup sweep
(cleanup_terminal_trade_signals / cleanup_trade_signals already empty it
routinely). This is the same lever already applied to
``trade_signal_emissions`` (202605220001) and the derived/cache tables
(202605230001).

IDEMPOTENT + LOCK-SAFE (mirrors 202605230001): skip if already UNLOGGED via
a cheap pg_class read (AccessShareLock, never blocked by writers); only when
an existing LOGGED DB still needs conversion do we drop lock_timeout /
statement_timeout and run ``ALTER TABLE ... SET UNLOGGED`` (which rewrites
the table and needs ACCESS EXCLUSIVE — at ~95 MB this is a few seconds).
Fresh installs create it UNLOGGED via the model ``prefixes`` + create_all,
so the ALTER branch is only hit by an existing LOGGED DB.

Revision ID: 202606160002
Revises: 202606160001
Create Date: 2026-06-16
"""

import sqlalchemy as sa
from alembic import op


revision = "202606160002"
down_revision = "202606160001"
branch_labels = None
depends_on = None

_TABLE = "trade_signals"


def _drop_inbound_foreign_keys() -> None:
    """Drop any FK that REFERENCES trade_signals.

    Postgres forbids a permanent table from FK-referencing an unlogged one, so
    the UNLOGGED flip fails if e.g. ``trader_decisions.signal_id`` still carries
    its old ``REFERENCES trade_signals(id) ON DELETE SET NULL`` constraint.  The
    live DB already lacks it (model drift, now reconciled by dropping the FK in
    the TraderDecision model), and the column stays as a plain reference — but
    drop defensively/idempotently so any DB built from an older baseline still
    converts cleanly.  Reversal does NOT recreate the FK (the column is
    intentionally unconstrained now).
    """
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT conrelid::regclass::text AS child, conname "
            "FROM pg_constraint "
            "WHERE contype = 'f' AND confrelid = 'trade_signals'::regclass"
        )
    ).fetchall()
    for child, conname in rows:
        op.execute(f'ALTER TABLE {child} DROP CONSTRAINT IF EXISTS "{conname}"')


def _flip_persistence(target_char: str, alter_clause: str) -> None:
    bind = op.get_bind()
    current = bind.execute(
        sa.text(
            "SELECT relpersistence FROM pg_class "
            "WHERE relname = :t AND relkind = 'r'"
        ),
        {"t": _TABLE},
    ).scalar()
    # None = table absent (skip); already-target = skip (no lock taken).
    if current is None or current == target_char:
        return
    # Existing LOGGED (or LOGGED-target) DB still needs the rewrite. Wait for
    # the lock + rewrite rather than failing fast under the boot lock_timeout.
    op.execute("SET LOCAL lock_timeout = 0")
    op.execute("SET LOCAL statement_timeout = 0")
    op.execute(f"ALTER TABLE {_TABLE} {alter_clause}")


def upgrade() -> None:
    # Inbound permanent->unlogged FKs must go before the table can be unlogged.
    _drop_inbound_foreign_keys()
    _flip_persistence("u", "SET UNLOGGED")


def downgrade() -> None:
    # Re-logging is always safe; the dropped FK is intentionally not recreated.
    _flip_persistence("p", "SET LOGGED")
