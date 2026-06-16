"""Convert high-volume telemetry tables to native RANGE partitioning by day.

Revision ID: 202606150002
Revises: 202606150001
Create Date: 2026-06-15

``trade_signal_emissions`` and ``trader_decision_checks`` are unbounded
append-only telemetry whose only safe long-term retention model is
time-partitioning + DROP PARTITION (O(1), no DELETE storm / dead-tuple churn /
vacuum). This converts both to native PostgreSQL RANGE partitioning on
``created_at`` (one partition per day) with a composite ``(id, created_at)`` PK.

Dual-path (the app bootstraps fresh DBs from the model via
``Base.metadata.create_all``, which now builds these tables PARTITIONED via the
partition hook in ``models.database``):

* **Fresh install / chain replay** — the table is already partitioned
  (relkind='p'); this migration is a **no-op**, so ``create_all`` and the
  migration chain stay identical (``test_alembic_roundtrip``).
* **Existing DB** — the table is plain (relkind='r'); convert in place: capture
  its secondary indexes, rename it aside, create the partitioned parent
  (composite PK + DEFAULT partition + daily partitions covering all existing
  rows + a few days ahead), backfill, verify row counts match, drop the old
  table, then recreate the secondary indexes and (decision_checks) the FK.

After this, retention is owned by ``MaintenanceService.maintain_partitions``
(create-ahead + DROP expired partitions); the DELETE-based scheduled prune for
these two tables is removed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "202606150002"
down_revision = "202606150001"
branch_labels = None
depends_on = None


# name -> (unlogged?, optional FK tuple (constraint_name, col, ref_table, ref_col))
_TABLES = (
    {"name": "trade_signal_emissions", "unlogged": True, "fk": None},
    {
        "name": "trader_decision_checks",
        "unlogged": False,
        "fk": ("trader_decision_checks_decision_id_fkey", "decision_id", "trader_decisions", "id"),
    },
)
_AHEAD_DAYS = 3


def _relkind(bind, table: str):
    # Cast to text: asyncpg returns pg's "char" type as bytes (b'r'/b'p'), which
    # would never == the str literals we compare against.
    return bind.execute(
        sa.text("SELECT relkind::text FROM pg_class WHERE relname = :t"), {"t": table}
    ).scalar()


def _secondary_index_defs(bind, table: str) -> list[str]:
    return [
        row[0]
        for row in bind.execute(
            sa.text(
                "SELECT pg_get_indexdef(i.indexrelid) "
                "FROM pg_index i "
                "WHERE i.indrelid = (SELECT c.oid FROM pg_class c WHERE c.relname = :t) "
                "AND NOT i.indisprimary"
            ),
            {"t": table},
        ).fetchall()
    ]


def _create_daily_partitions(table: str, persist: str, start_day, end_day) -> None:
    d = start_day
    while d <= end_day:
        lo = d.strftime("%Y-%m-%d")
        hi = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        pname = f"{table}_{d.strftime('%Y%m%d')}"
        op.execute(
            f"CREATE {persist}TABLE IF NOT EXISTS {pname} PARTITION OF {table} "
            f"FOR VALUES FROM ('{lo}') TO ('{hi}')"
        )
        d += timedelta(days=1)


def upgrade() -> None:
    bind = op.get_bind()
    # The bulk backfill below can exceed the server's default statement_timeout
    # (60s) on a large existing table; disable it (and lock_timeout) for this
    # migration's session. The live cutover runs with the app stopped, so there
    # is no lock contention to bound.
    bind.execute(sa.text("SET statement_timeout = 0"))
    bind.execute(sa.text("SET lock_timeout = 0"))
    today = datetime.now(timezone.utc).date()
    for cfg in _TABLES:
        table = cfg["name"]
        if _relkind(bind, table) != "r":
            # Already partitioned (fresh-install / replay path) -> no-op.
            continue
        persist = "UNLOGGED " if cfg["unlogged"] else ""
        old = f"{table}__pre_partition"

        index_defs = _secondary_index_defs(bind, table)
        rng = bind.execute(
            sa.text(f"SELECT min(created_at), max(created_at) FROM {table}")
        ).fetchone()
        min_dt = rng[0]
        start_day = min_dt.date() if min_dt is not None else today
        end_day = today + timedelta(days=_AHEAD_DAYS)

        op.execute(f"ALTER TABLE {table} RENAME TO {old}")
        op.execute(
            f"CREATE {persist}TABLE {table} "
            f"(LIKE {old} INCLUDING DEFAULTS INCLUDING STORAGE, PRIMARY KEY (id, created_at)) "
            f"PARTITION BY RANGE (created_at)"
        )
        op.execute(f"CREATE {persist}TABLE {table}_default PARTITION OF {table} DEFAULT")
        _create_daily_partitions(table, persist, start_day, end_day)

        # Backfill. The live run does this with the app stopped (no contention);
        # the replay path never reaches here (it skips above as already-'p').
        op.execute(f"INSERT INTO {table} SELECT * FROM {old}")

        old_n = bind.execute(sa.text(f"SELECT count(*) FROM {old}")).scalar()
        new_n = bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()
        if old_n != new_n:
            raise RuntimeError(
                f"partition backfill row mismatch for {table}: old={old_n} new={new_n}"
            )

        op.execute(f"DROP TABLE {old}")

        # Recreate secondary indexes (names were held by the old table until the
        # DROP above). On a partitioned parent these become partitioned indexes.
        for idef in index_defs:
            op.execute(idef)

        # FK added after backfill -> a single validation pass instead of a
        # per-row check during the bulk insert.
        if cfg["fk"]:
            name, col, ref_t, ref_c = cfg["fk"]
            op.execute(
                f"ALTER TABLE {table} ADD CONSTRAINT {name} "
                f"FOREIGN KEY ({col}) REFERENCES {ref_t}({ref_c}) ON DELETE CASCADE"
            )


def downgrade() -> None:
    bind = op.get_bind()
    for cfg in reversed(_TABLES):
        table = cfg["name"]
        if _relkind(bind, table) != "p":
            # Not partitioned -> nothing to revert.
            continue
        persist = "UNLOGGED " if cfg["unlogged"] else ""
        old = f"{table}__partitioned"

        index_defs = _secondary_index_defs(bind, table)

        op.execute(f"ALTER TABLE {table} RENAME TO {old}")
        op.execute(
            f"CREATE {persist}TABLE {table} "
            f"(LIKE {old} INCLUDING DEFAULTS INCLUDING STORAGE, PRIMARY KEY (id))"
        )
        op.execute(f"INSERT INTO {table} SELECT * FROM {old}")
        op.execute(f"DROP TABLE {old} CASCADE")

        for idef in index_defs:
            op.execute(idef)

        if cfg["fk"]:
            name, col, ref_t, ref_c = cfg["fk"]
            op.execute(
                f"ALTER TABLE {table} ADD CONSTRAINT {name} "
                f"FOREIGN KEY ({col}) REFERENCES {ref_t}({ref_c}) ON DELETE CASCADE"
            )
