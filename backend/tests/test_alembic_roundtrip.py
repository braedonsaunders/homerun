"""Alembic migration tests.

Two cases:

1.  ``test_head_migration_downgrade_upgrade_roundtrip`` — stamps a
    throwaway DB at head, downgrades one revision, re-upgrades.
    Catches a new head migration whose ``downgrade()`` raises or
    isn't symmetric with its ``upgrade()``.  Cheap and runs in seconds.

2.  ``test_alembic_replay_base_to_head_on_empty_db`` (Plan 0020) —
    runs the entire migration chain from ``base`` against a true
    empty database and asserts the final revision matches head.
    Catches a new migration that breaks fresh-DB bootstrap (e.g.
    a non-idempotent ``op.add_column`` colliding with the baseline's
    lazy ``Base.metadata.create_all``).  Slower (~5–15 s) and is
    marked ``slow`` accordingly.

Both skip when no writable Postgres is reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _build_alembic_config(sync_connection) -> Config:
    """Return an Alembic ``Config`` wired to the given sync connection.

    The Alembic env.py honours ``config.attributes['connection']`` and
    skips its own engine creation when present (see
    ``backend/alembic/env.py:run_migrations_online``).  That lets us
    point migrations at the throwaway database without monkey-patching
    settings or shelling out to the alembic CLI.
    """
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    cfg.attributes["connection"] = sync_connection
    return cfg


@pytest.mark.db
@pytest.mark.slow
@pytest.mark.asyncio
async def test_head_migration_downgrade_upgrade_roundtrip() -> None:
    try:
        from models.database import Base
        from models.model_registry import register_all_models
        from tests.postgres_test_db import build_postgres_session_factory
    except Exception as exc:  # pragma: no cover — defensive
        pytest.skip(f"DB harness unavailable: {exc}")

    register_all_models()

    try:
        engine, _factory = await build_postgres_session_factory(
            Base, "alembic_head_roundtrip"
        )
    except Exception as exc:
        pytest.skip(f"Postgres unreachable for alembic round-trip: {exc}")

    try:
        # Compute head + previous revision once, before opening a
        # connection — pure script-graph traversal, no DB side effects.
        script = ScriptDirectory.from_config(
            Config(str(BACKEND_ROOT / "alembic.ini"))
        )
        head_revision = script.get_current_head()
        assert head_revision, "alembic script directory has no head revision"

        head_script = script.get_revision(head_revision)
        previous_revision = head_script.down_revision
        assert isinstance(previous_revision, str) and previous_revision, (
            "head migration has no single down_revision (merge node?); "
            "extend this test to handle the merge case"
        )

        async with engine.connect() as conn:

            def _run_alembic(sync_conn) -> None:
                from alembic.runtime.migration import MigrationContext

                cfg = _build_alembic_config(sync_conn)

                # 1. Stamp the throwaway DB at head.  ``Base.metadata``
                # was already materialised by ``build_postgres_session_factory``
                # so the schema matches the production DB; we just need
                # alembic_version to reflect that.
                command.stamp(cfg, head_revision)
                rev = MigrationContext.configure(sync_conn).get_current_revision()
                assert rev == head_revision, (
                    f"After stamp expected {head_revision!r}, got {rev!r}"
                )

                # 2. Downgrade one revision to the explicit parent of
                # head.  We compute the target ourselves because the
                # python API does not honour the CLI's ``-1`` relative
                # syntax.  The head migration's ``downgrade()`` runs.
                command.downgrade(cfg, previous_revision)
                rev = MigrationContext.configure(sync_conn).get_current_revision()
                assert rev == previous_revision, (
                    f"After downgrade expected {previous_revision!r}, "
                    f"got {rev!r} — head migration's downgrade() ran "
                    f"but moved alembic_version to an unexpected revision"
                )

                # 3. Re-upgrade to head.  The head migration's
                # ``upgrade()`` runs against a schema that was just
                # rolled back by its own ``downgrade()`` — this is the
                # load-bearing assertion: did downgrade actually undo
                # what upgrade did, or just bump the version row?
                command.upgrade(cfg, "head")
                rev = MigrationContext.configure(sync_conn).get_current_revision()
                assert rev == head_revision, (
                    f"After re-upgrade expected {head_revision!r}, "
                    f"got {rev!r}"
                )

            await conn.run_sync(_run_alembic)
    finally:
        await engine.dispose()


@pytest.mark.db
@pytest.mark.slow
@pytest.mark.asyncio
async def test_alembic_replay_base_to_head_on_empty_db() -> None:
    """Run the full migration chain from base against an empty DB.

    Asserts that ``alembic upgrade base→head`` succeeds end-to-end,
    which is the contract a new developer setup or a bare-metal CI
    bootstrap relies on.

    Plan 0020 made every later schema-additive migration idempotent
    via ``alembic_helpers.safe_*`` wrappers, so the lazy baseline
    (``Base.metadata.create_all``) plus the chain replay produces a
    correct schema without colliding ``op.add_column`` /
    ``op.create_table`` / ``op.create_index`` calls.

    Future migrations should use ``safe_add_column``,
    ``safe_create_table``, and ``safe_create_index`` from
    ``alembic_helpers`` to keep this test green.

    The replay runs in a **subprocess** with the throwaway DB URL in
    its env.  The in-process route (``command.upgrade`` against a
    shared async connection) hits an alembic transaction-state
    assertion midway through the ~130-migration chain — alembic
    expects to manage its own per-migration transaction lifecycle
    and gets confused when the connection's outer state is shared
    across many ``begin_transaction`` calls.  A subprocess gives
    alembic the standalone connection lifecycle it expects, matches
    what production's ``init_database`` actually does on cold start,
    and matches the lifespan smoke test's pattern.
    """
    import asyncio
    import os
    import sys

    try:
        from sqlalchemy import MetaData

        from tests.postgres_test_db import build_postgres_session_factory
    except Exception as exc:  # pragma: no cover — defensive
        pytest.skip(f"DB harness unavailable: {exc}")

    try:
        # Pass an empty MetaData so the factory does NOT pre-create
        # the schema — we want a true empty DB for alembic to
        # bootstrap against.
        engine, _factory = await build_postgres_session_factory(
            MetaData(), "alembic_replay"
        )
    except Exception as exc:
        pytest.skip(f"Postgres unreachable for alembic replay: {exc}")

    try:
        script = ScriptDirectory.from_config(
            Config(str(BACKEND_ROOT / "alembic.ini"))
        )
        head_revision = script.get_current_head()
        assert head_revision, "alembic script directory has no head revision"

        # Use render_as_string(hide_password=False) — str(URL) redacts
        # the password in modern SQLAlchemy and would cause the
        # subprocess to fail with InvalidPasswordError.
        test_database_url = engine.url.render_as_string(hide_password=False)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _REPLAY_DRIVER_SOURCE,
            cwd=str(BACKEND_ROOT),
            env={
                **os.environ,
                "DATABASE_URL": test_database_url,
                "LOG_LEVEL": "WARNING",
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            pytest.fail(
                "alembic replay subprocess exceeded 60 s — a migration "
                "in the chain is hanging or taking too long"
            )

        if proc.returncode != 0:
            pytest.fail(
                "alembic replay subprocess exited with "
                f"{proc.returncode}\nstdout: {stdout!r}\nstderr: {stderr!r}"
            )

        # The subprocess prints the final revision it observed so we
        # can assert it locally.
        marker = b"REPLAY_OK head="
        assert marker in stdout, (
            f"alembic replay subprocess did not emit success marker.\n"
            f"stdout: {stdout!r}\nstderr: {stderr!r}"
        )
        observed = stdout.split(marker, 1)[1].split(b"\n", 1)[0].strip().decode()
        assert observed == head_revision, (
            f"After full replay expected {head_revision!r}, "
            f"subprocess observed {observed!r}"
        )
    finally:
        await engine.dispose()


_REPLAY_DRIVER_SOURCE = """
import asyncio
import os
import subprocess
import sys

from alembic.runtime.migration import MigrationContext
from sqlalchemy.ext.asyncio import create_async_engine


async def _read_revision(engine) -> str | None:
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync_conn: MigrationContext.configure(
                sync_conn
            ).get_current_revision()
        )


async def _drive() -> int:
    engine = create_async_engine(os.environ["DATABASE_URL"])
    try:
        pre = await _read_revision(engine)
        assert pre is None, f"expected fresh DB, got revision {pre!r}"
    finally:
        await engine.dispose()

    # Shell out to the alembic CLI rather than driving ``command.upgrade``
    # in-process.  Several migrations (e.g. 202603120001_db_hot_path_indexes)
    # use ``context.autocommit_block()`` for ``CREATE INDEX CONCURRENTLY``,
    # which requires alembic to own the connection's transaction lifecycle
    # end-to-end.  Wrapping the engine externally (even via ``engine.connect()``
    # alone) breaks ``autocommit_block``'s ``assert self._transaction is not
    # None`` deep in ``MigrationContext``.  The CLI matches what
    # ``init_database`` in production does on cold start.
    proc = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "/app/backend/alembic.ini",
         "upgrade", "head"],
        cwd="/app/backend",
        env={**os.environ},
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)

    engine = create_async_engine(os.environ["DATABASE_URL"])
    try:
        post = await _read_revision(engine)
    finally:
        await engine.dispose()

    print(f"REPLAY_OK head={post}", flush=True)
    return 0


sys.exit(asyncio.run(_drive()))
"""
