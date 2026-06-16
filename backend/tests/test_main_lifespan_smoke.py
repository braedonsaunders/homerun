"""Startup smoke tests for the FastAPI app defined in ``backend/main.py``.

Why this file exists
--------------------
The backend has 195 pytest files exercising individual subsystems, but
nothing in CI verifies that the top-level ``main`` module can be
imported cleanly or that ``app``'s ``lifespan`` enters and exits
without raising.  Two real classes of regression slip past the rest of
the suite today:

1.  An import-time error in ``main.py`` (e.g. a renamed symbol in
    ``api/routes_*.py``, a circular import, a missing dependency
    pulled in by a new feature).  Every existing test happens to
    import only narrow slices of the codebase, so this never surfaces
    until the operator runs a redeploy and sees the container
    restart-loop.
2.  A misconfigured ``lifespan`` block — for example, a service whose
    ``stop()`` call hangs, or an ``init_database`` migration that
    raises against a fresh DB.

These tests cover both.  ``test_import_app`` is unconditional; the
lifespan-driving test runs only when a writable Postgres is reachable
(it allocates a throwaway database via ``build_postgres_session_factory``)
so it skips cleanly on a developer box without DB access instead of
producing a confusing failure.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ---------------------------------------------------------------------------
# Cheap, no-IO smoke — runs everywhere
# ---------------------------------------------------------------------------


def test_import_app_module() -> None:
    """``import main`` must succeed without raising.

    Catches the most common 'backend won't start' regression: a renamed
    symbol or stale import landed somewhere in the ``api`` / ``services``
    / ``workers`` tree that is only pulled in by ``main.py`` itself.
    """
    import main  # noqa: F401 — import is the assertion

    assert hasattr(main, "app"), "main.py must expose an `app` symbol"


def test_app_is_fastapi_with_routes() -> None:
    """``app`` is a FastAPI instance with a non-trivial number of routes.

    A boot regression that empties the router (e.g. the include_router
    block raising silently and the rest of main.py continuing) would be
    invisible to the import test but is caught here.
    """
    from fastapi import FastAPI

    import main

    assert isinstance(main.app, FastAPI)
    # The current app registers ~460 OpenAPI paths; assert a generous lower bound
    # so this test doesn't churn when a router is added. Use the OpenAPI schema
    # (FastAPI's canonical route enumeration) rather than walking app.routes:
    # Starlette versions differ in how included routers are nested there (newer
    # ones keep opaque _IncludedRouter wrappers with no top-level ``.path`` /
    # ``.routes``), but ``openapi()`` always reflects the real registered paths.
    path_count = len(main.app.openapi().get("paths", {}))
    assert path_count > 100, (
        f"FastAPI app exposes only {path_count} OpenAPI paths — likely a router "
        f"include block raised during import and was swallowed"
    )


def test_lifespan_is_async_context_manager() -> None:
    """The ``lifespan`` attribute on ``app`` must be a callable that
    returns an async context manager.  FastAPI's contract.
    """
    import main

    lifespan = main.app.router.lifespan_context
    assert callable(lifespan), "app.router.lifespan_context must be callable"


# ---------------------------------------------------------------------------
# Live lifespan smoke — needs DB
# ---------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.slow
@pytest.mark.asyncio
async def test_lifespan_startup_and_shutdown_complete() -> None:
    """Drive the FastAPI lifespan against a throwaway database.

    This runs in a subprocess so we can override ``DATABASE_URL`` before
    ``models.database`` is imported (the engine is created at import
    time and points at whatever ``settings.DATABASE_URL`` resolved to).
    The subprocess imports ``main``, enters and exits ``lifespan``, and
    prints ``OK`` on success.

    Skips when no writable Postgres is reachable — the
    ``build_postgres_session_factory`` admin connect step would fail
    with a confusing ``ConnectionRefusedError`` otherwise.
    """
    try:
        from models.database import Base  # noqa: F401
        from tests.postgres_test_db import build_postgres_session_factory
    except Exception as exc:  # pragma: no cover — defensive
        pytest.skip(f"DB harness unavailable: {exc}")

    try:
        engine, _session_factory = await build_postgres_session_factory(
            Base, "lifespan_smoke"
        )
    except Exception as exc:
        pytest.skip(f"Postgres unreachable for lifespan smoke: {exc}")

    # Build the asyncpg-style URL pointing at the throwaway DB.  Use
    # ``render_as_string(hide_password=False)`` rather than ``str()``
    # — SQLAlchemy redacts the password in ``str(URL)`` by default,
    # which would cause the subprocess to fail with
    # ``InvalidPasswordError`` when it tries to authenticate.
    test_database_url = engine.url.render_as_string(hide_password=False)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _LIFESPAN_DRIVER_SOURCE,
            cwd=str(BACKEND_ROOT),
            env={
                **os.environ,
                "DATABASE_URL": test_database_url,
                # Silence the noisy startup logs in the child so the
                # captured stdout is just our marker.
                "LOG_LEVEL": os.environ.get("LIFESPAN_SMOKE_LOG_LEVEL", "WARNING"),
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=45.0
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            pytest.fail(
                "lifespan smoke subprocess exceeded 45 s — startup or "
                "shutdown is hanging"
            )

        if proc.returncode != 0:
            pytest.fail(
                "lifespan subprocess exited with "
                f"{proc.returncode}\nstdout: {stdout!r}\nstderr: {stderr!r}"
            )

        assert b"LIFESPAN_OK" in stdout, (
            f"lifespan subprocess did not emit success marker.\n"
            f"stdout: {stdout!r}\nstderr: {stderr!r}"
        )
    finally:
        await engine.dispose()


_LIFESPAN_DRIVER_SOURCE = """
import asyncio
import sys


async def _drive() -> int:
    import main

    async with main.app.router.lifespan_context(main.app):
        # Startup completed without raising.  We do not exercise any
        # routes here — the smoke is "can the app boot, then quiesce."
        pass
    print("LIFESPAN_OK", flush=True)
    return 0


sys.exit(asyncio.run(_drive()))
"""


# ---------------------------------------------------------------------------
# _reset_orchestrator_boot_state branch tests (plan 0021)
# ---------------------------------------------------------------------------
#
# The boot-state hook in main.py decides whether to hard-reset the
# trader orchestrator (default safety) or auto-resume in shadow mode.
# The branch picked is a function of the prior persisted control row:
# shadow + enabled + unpaused → auto-resume, everything else (live,
# stopped, or paused) → hard reset.  These tests pin both branches
# plus the "always clear live_preflight / live_arm" invariant.


@pytest.mark.asyncio
async def test_reset_orchestrator_boot_state_auto_resumes_shadow_when_previously_running(monkeypatch):
    import main

    update_calls: list[dict] = []
    snapshot_calls: list[dict] = []
    runtime_calls: list[dict] = []

    async def _fake_read(_session):
        return {
            "mode": "shadow",
            "is_enabled": True,
            "is_paused": False,
            "run_interval_seconds": 5,
            "requested_run_at": "2026-05-10T05:34:00+00:00",
            "settings_json": {
                "selected_account_id": "acc-1",
                "shadow_account_id": "acc-1",
                "live_arm": {"armed_until": "stale"},
                "live_preflight": {"checks": []},
            },
        }

    async def _fake_update(_session, **kwargs):
        update_calls.append(kwargs)
        return {
            "mode": "shadow",
            "is_enabled": True,
            "is_paused": False,
            "run_interval_seconds": 5,
            "requested_run_at": "2026-05-10T05:34:00+00:00",
            "settings_json": {
                "selected_account_id": "acc-1",
                "shadow_account_id": "acc-1",
                "live_arm": None,
                "live_preflight": None,
            },
        }

    async def _fake_snapshot(_session, **kwargs):
        snapshot_calls.append(kwargs)
        return {}

    def _fake_runtime(**kwargs):
        runtime_calls.append(kwargs)

    monkeypatch.setattr(main, "AsyncSessionLocal", _AsyncSessionContext)
    monkeypatch.setattr(main, "read_orchestrator_control", _fake_read)
    monkeypatch.setattr(main, "update_orchestrator_control", _fake_update)
    monkeypatch.setattr(main, "write_orchestrator_snapshot", _fake_snapshot)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _fake_runtime)

    await main._reset_orchestrator_boot_state()

    assert len(update_calls) == 1
    assert set(update_calls[0].keys()) == {"settings_json"}, (
        "auto-resume branch must not pass is_enabled/is_paused/mode/requested_run_at"
    )
    assert update_calls[0]["settings_json"] == {
        "live_preflight": None,
        "live_arm": None,
    }

    assert len(snapshot_calls) == 1
    assert snapshot_calls[0]["enabled"] is True
    assert snapshot_calls[0]["current_activity"] == "Resumed in shadow on application startup"

    assert len(runtime_calls) == 1
    assert runtime_calls[0]["enabled"] is True
    assert runtime_calls[0]["control"] == {
        "is_enabled": True,
        "is_paused": False,
        "interval_seconds": 5,
        "requested_run_at": "2026-05-10T05:34:00+00:00",
    }


@pytest.mark.asyncio
async def test_reset_orchestrator_boot_state_hard_resets_when_prior_mode_was_live(monkeypatch):
    import main

    update_calls: list[dict] = []

    async def _fake_read(_session):
        return {
            "mode": "live",
            "is_enabled": True,
            "is_paused": False,
            "run_interval_seconds": 5,
            "requested_run_at": "2026-05-10T05:34:00+00:00",
            "settings_json": {
                "selected_account_id": "live:wallet-1",
                "shadow_account_id": None,
                "live_arm": {"armed_until": "..."},
                "live_preflight": {"checks": []},
            },
        }

    async def _fake_update(_session, **kwargs):
        update_calls.append(kwargs)
        return {"run_interval_seconds": 5}

    async def _fake_snapshot(_session, **kwargs):
        _fake_snapshot.last = kwargs
        return {}
    _fake_snapshot.last = {}

    def _fake_runtime(**kwargs):
        _fake_runtime.last = kwargs
    _fake_runtime.last = {}

    monkeypatch.setattr(main, "AsyncSessionLocal", _AsyncSessionContext)
    monkeypatch.setattr(main, "read_orchestrator_control", _fake_read)
    monkeypatch.setattr(main, "update_orchestrator_control", _fake_update)
    monkeypatch.setattr(main, "write_orchestrator_snapshot", _fake_snapshot)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _fake_runtime)

    await main._reset_orchestrator_boot_state()

    assert len(update_calls) == 1
    assert update_calls[0]["is_enabled"] is False
    assert update_calls[0]["is_paused"] is True
    assert update_calls[0]["mode"] == "shadow"
    assert update_calls[0]["requested_run_at"] is None
    # Account selection is preserved across the restart (only the live arm +
    # preflight tokens are cleared); trading still stays disabled + paused so
    # live never auto-resumes.
    assert update_calls[0]["settings_json"] == {
        "live_preflight": None,
        "live_arm": None,
    }
    assert _fake_snapshot.last["enabled"] is False
    assert _fake_snapshot.last["current_activity"] == "Paused on application startup"
    assert _fake_runtime.last["control"]["is_enabled"] is False
    assert _fake_runtime.last["control"]["is_paused"] is True


@pytest.mark.asyncio
async def test_reset_orchestrator_boot_state_hard_resets_when_previously_stopped(monkeypatch):
    import main

    update_calls: list[dict] = []

    async def _fake_read(_session):
        return {
            "mode": "shadow",
            "is_enabled": False,
            "is_paused": True,
            "run_interval_seconds": 5,
            "settings_json": {},
        }

    async def _fake_update(_session, **kwargs):
        update_calls.append(kwargs)
        return {"run_interval_seconds": 5}

    async def _fake_snapshot(_session, **kwargs):
        return {}

    def _fake_runtime(**kwargs):
        pass

    monkeypatch.setattr(main, "AsyncSessionLocal", _AsyncSessionContext)
    monkeypatch.setattr(main, "read_orchestrator_control", _fake_read)
    monkeypatch.setattr(main, "update_orchestrator_control", _fake_update)
    monkeypatch.setattr(main, "write_orchestrator_snapshot", _fake_snapshot)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _fake_runtime)

    await main._reset_orchestrator_boot_state()

    assert len(update_calls) == 1
    assert update_calls[0]["is_enabled"] is False
    assert update_calls[0]["is_paused"] is True
    assert update_calls[0]["mode"] == "shadow"


@pytest.mark.asyncio
async def test_reset_orchestrator_boot_state_hard_resets_when_previously_paused(monkeypatch):
    import main

    update_calls: list[dict] = []

    async def _fake_read(_session):
        return {
            "mode": "shadow",
            "is_enabled": True,
            "is_paused": True,
            "run_interval_seconds": 5,
            "settings_json": {"selected_account_id": "acc-1"},
        }

    async def _fake_update(_session, **kwargs):
        update_calls.append(kwargs)
        return {"run_interval_seconds": 5}

    async def _fake_snapshot(_session, **kwargs):
        return {}

    def _fake_runtime(**kwargs):
        pass

    monkeypatch.setattr(main, "AsyncSessionLocal", _AsyncSessionContext)
    monkeypatch.setattr(main, "read_orchestrator_control", _fake_read)
    monkeypatch.setattr(main, "update_orchestrator_control", _fake_update)
    monkeypatch.setattr(main, "write_orchestrator_snapshot", _fake_snapshot)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _fake_runtime)

    await main._reset_orchestrator_boot_state()

    assert len(update_calls) == 1
    assert update_calls[0]["is_enabled"] is False
    assert update_calls[0]["mode"] == "shadow"
    # Account preserved across restart — only live arm/preflight are cleared.
    assert "selected_account_id" not in update_calls[0]["settings_json"]


@pytest.mark.asyncio
async def test_reset_orchestrator_boot_state_always_clears_live_flags_in_auto_resume(monkeypatch):
    import main

    update_calls: list[dict] = []

    async def _fake_read(_session):
        return {
            "mode": "shadow",
            "is_enabled": True,
            "is_paused": False,
            "run_interval_seconds": 5,
            "settings_json": {
                "selected_account_id": "acc-1",
                "live_arm": {"armed_until": "2026-05-10T06:00:00+00:00"},
                "live_preflight": {"checks": [{"id": "kill_switch", "ok": True}]},
            },
        }

    async def _fake_update(_session, **kwargs):
        update_calls.append(kwargs)
        return {"run_interval_seconds": 5}

    async def _fake_snapshot(_session, **kwargs):
        return {}

    def _fake_runtime(**kwargs):
        pass

    monkeypatch.setattr(main, "AsyncSessionLocal", _AsyncSessionContext)
    monkeypatch.setattr(main, "read_orchestrator_control", _fake_read)
    monkeypatch.setattr(main, "update_orchestrator_control", _fake_update)
    monkeypatch.setattr(main, "write_orchestrator_snapshot", _fake_snapshot)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _fake_runtime)

    await main._reset_orchestrator_boot_state()

    assert update_calls[0]["settings_json"]["live_preflight"] is None
    assert update_calls[0]["settings_json"]["live_arm"] is None


@pytest.mark.asyncio
async def test_reset_orchestrator_boot_state_runtime_status_mirrors_branch(monkeypatch):
    import main

    runtime_calls_resume: list[dict] = []
    runtime_calls_reset: list[dict] = []

    async def _fake_update(_session, **_kwargs):
        return {"run_interval_seconds": 5}

    async def _fake_snapshot(_session, **_kwargs):
        return {}

    monkeypatch.setattr(main, "AsyncSessionLocal", _AsyncSessionContext)
    monkeypatch.setattr(main, "update_orchestrator_control", _fake_update)
    monkeypatch.setattr(main, "write_orchestrator_snapshot", _fake_snapshot)

    async def _read_resume(_session):
        return {"mode": "shadow", "is_enabled": True, "is_paused": False, "run_interval_seconds": 5, "settings_json": {}}

    def _runtime_resume(**kwargs):
        runtime_calls_resume.append(kwargs)

    monkeypatch.setattr(main, "read_orchestrator_control", _read_resume)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _runtime_resume)
    await main._reset_orchestrator_boot_state()

    async def _read_reset(_session):
        return {"mode": "live", "is_enabled": True, "is_paused": False, "run_interval_seconds": 5, "settings_json": {}}

    def _runtime_reset(**kwargs):
        runtime_calls_reset.append(kwargs)

    monkeypatch.setattr(main, "read_orchestrator_control", _read_reset)
    monkeypatch.setattr(main.runtime_status, "update_orchestrator", _runtime_reset)
    await main._reset_orchestrator_boot_state()

    assert runtime_calls_resume[0]["enabled"] is True
    assert runtime_calls_resume[0]["control"]["is_enabled"] is True
    assert runtime_calls_resume[0]["control"]["is_paused"] is False

    assert runtime_calls_reset[0]["enabled"] is False
    assert runtime_calls_reset[0]["control"]["is_enabled"] is False
    assert runtime_calls_reset[0]["control"]["is_paused"] is True


class _AsyncSessionContext:
    """Minimal async context manager stand-in for AsyncSessionLocal so the
    boot-state hook can `async with AsyncSessionLocal() as session:` without
    touching a real database. Returns a sentinel object that the patched
    helpers ignore."""

    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False
