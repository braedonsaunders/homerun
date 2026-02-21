from __future__ import annotations

import pytest
from sqlalchemy import String
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from services.worker_state import _commit_with_retry
from tests.postgres_test_db import build_postgres_session_factory

Base = declarative_base()


class CommitRetryProbe(Base):
    __tablename__ = "commit_retry_probe"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)


def _make_retryable_db_lock_error() -> OperationalError:
    return OperationalError("COMMIT", {}, Exception("deadlock detected"))


@pytest.mark.asyncio
async def test_commit_with_retry_replays_dirty_updates_after_lock() -> None:
    engine, Session = await build_postgres_session_factory(Base, "worker_state_commit_retry_dirty")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as session:
            session.add(CommitRetryProbe(id="row-1", value="before"))
            await session.commit()

        async with Session() as session:
            row = await session.get(CommitRetryProbe, "row-1")
            assert row is not None
            row.value = "after"

            calls = {"count": 0}
            original_commit = session.commit

            async def flaky_commit():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise _make_retryable_db_lock_error()
                return await original_commit()

            session.commit = flaky_commit  # type: ignore[method-assign]
            await _commit_with_retry(session)

            refreshed = await session.get(CommitRetryProbe, "row-1")
            assert refreshed is not None
            assert refreshed.value == "after"
            assert calls["count"] == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_commit_with_retry_replays_pending_insert_after_lock() -> None:
    engine, Session = await build_postgres_session_factory(Base, "worker_state_commit_retry_insert")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as session:
            session.add(CommitRetryProbe(id="row-2", value="inserted"))

            calls = {"count": 0}
            original_commit = session.commit

            async def flaky_commit():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise _make_retryable_db_lock_error()
                return await original_commit()

            session.commit = flaky_commit  # type: ignore[method-assign]
            await _commit_with_retry(session)

            inserted = await session.get(CommitRetryProbe, "row-2")
            assert inserted is not None
            assert inserted.value == "inserted"
            assert calls["count"] == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_commit_with_retry_replays_pending_delete_after_lock() -> None:
    engine, Session = await build_postgres_session_factory(Base, "worker_state_commit_retry_delete")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as session:
            session.add(CommitRetryProbe(id="row-3", value="delete-me"))
            await session.commit()

        async with Session() as session:
            row = await session.get(CommitRetryProbe, "row-3")
            assert row is not None
            await session.delete(row)

            calls = {"count": 0}
            original_commit = session.commit

            async def flaky_commit():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise _make_retryable_db_lock_error()
                return await original_commit()

            session.commit = flaky_commit  # type: ignore[method-assign]
            await _commit_with_retry(session)

            deleted = await session.get(CommitRetryProbe, "row-3")
            assert deleted is None
            assert calls["count"] == 2
    finally:
        await engine.dispose()
