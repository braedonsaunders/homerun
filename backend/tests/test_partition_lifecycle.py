"""Exercises MaintenanceService partition-lifecycle helpers against real PG.

Validates the operational DDL that ``maintain_partitions`` relies on:
* ``_ensure_daily_partition`` creates a daily partition, and DEFAULT-drains any
  rows already sitting in the DEFAULT partition for that day.
* ``_drop_expired_partitions`` drops only the daily partitions older than the
  cutoff (never the DEFAULT or current day).
* new inserts route to the correct daily partition.

The schema / migration side is covered by ``test_alembic_roundtrip``; this pins
the runtime lifecycle behaviour.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table, text

from services.maintenance import maintenance_service
from tests.postgres_test_db import build_postgres_session_factory


@pytest.mark.db
@pytest.mark.slow
@pytest.mark.asyncio
async def test_partition_lifecycle_create_drain_drop_route() -> None:
    md = MetaData()
    Table(
        "part_smoke",
        md,
        Column("id", String, primary_key=True),
        Column("created_at", DateTime, primary_key=True),
        info={"partition_by": "RANGE (created_at)"},
    )
    engine, Session = await build_postgres_session_factory(md, "part_lifecycle")
    try:
        # create_all built the partitioned parent via the model hook; add DEFAULT.
        async with Session() as s:
            await s.execute(text("CREATE TABLE part_smoke_default PARTITION OF part_smoke DEFAULT"))
            await s.commit()

        today = datetime.now(timezone.utc).date()
        today_ts = datetime.combine(today, datetime.min.time())
        pname = f"part_smoke_{today.strftime('%Y%m%d')}"

        # 1) A 'today' row lands in DEFAULT (no daily partition exists yet).
        async with Session() as s:
            await s.execute(
                text("INSERT INTO part_smoke (id, created_at) VALUES ('a', :ts)"),
                {"ts": today_ts},
            )
            await s.commit()
            assert await s.scalar(text("SELECT count(*) FROM part_smoke_default")) == 1

        # 2) Ensuring today's partition DEFAULT-drains the row into it.
        async with Session() as s:
            created = await maintenance_service._ensure_daily_partition(s, "part_smoke", today, "")
            await s.commit()
            assert created is True
        async with Session() as s:
            assert await s.scalar(text(f"SELECT count(*) FROM {pname}")) == 1
            assert await s.scalar(text("SELECT count(*) FROM part_smoke_default")) == 0

        # 3) An old partition + row is dropped by drop-expired; today's is kept.
        old = today - timedelta(days=30)
        oname = f"part_smoke_{old.strftime('%Y%m%d')}"
        lo, hi = old.strftime("%Y-%m-%d"), (old + timedelta(days=1)).strftime("%Y-%m-%d")
        async with Session() as s:
            await s.execute(
                text(f"CREATE TABLE {oname} PARTITION OF part_smoke FOR VALUES FROM ('{lo}') TO ('{hi}')")
            )
            await s.execute(text(f"INSERT INTO part_smoke (id, created_at) VALUES ('b', '{lo}')"))
            await s.commit()
        async with Session() as s:
            dropped = await maintenance_service._drop_expired_partitions(
                s, "part_smoke", cutoff_day=today - timedelta(days=7)
            )
            await s.commit()
            assert dropped == 1
        async with Session() as s:
            assert await s.scalar(text("SELECT 1 FROM pg_class WHERE relname = :p"), {"p": oname}) is None
            assert await s.scalar(text("SELECT 1 FROM pg_class WHERE relname = :p"), {"p": pname}) == 1

        # 4) New 'today' inserts route to today's partition, not DEFAULT.
        async with Session() as s:
            await s.execute(
                text("INSERT INTO part_smoke (id, created_at) VALUES ('c', :ts)"),
                {"ts": today_ts},
            )
            await s.commit()
            assert await s.scalar(text(f"SELECT count(*) FROM {pname}")) == 2
            assert await s.scalar(text("SELECT count(*) FROM part_smoke_default")) == 0
    finally:
        await engine.dispose()
