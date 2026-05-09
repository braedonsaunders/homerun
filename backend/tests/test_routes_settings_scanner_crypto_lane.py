"""``GET /settings/scanner`` exposes the live ``crypto_lane_enabled``."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_settings
from models.database import AppSettings, Base
from tests.postgres_test_db import build_postgres_session_factory


@pytest.mark.asyncio
async def test_get_scanner_settings_reflects_worker_control_disabled(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(
        Base, "scanner_crypto_lane_off"
    )
    try:
        async with session_factory() as session:
            session.add(AppSettings(id="default"))
            await session.commit()

        @asynccontextmanager
        async def _session_cm():
            async with session_factory() as session:
                yield session

        def _factory():
            return _session_cm()

        monkeypatch.setattr(routes_settings, "AsyncSessionLocal", _factory)

        async def _fake_read(_session, worker_name, *, default_interval=None):
            assert worker_name == "crypto"
            return {
                "worker_name": "crypto",
                "is_enabled": False,
                "is_paused": False,
                "interval_seconds": 1,
                "requested_run_at": None,
                "updated_at": None,
            }

        monkeypatch.setattr(routes_settings, "read_worker_control", _fake_read)

        response = await routes_settings.get_scanner_settings()
        assert response.crypto_lane_enabled is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_scanner_settings_reflects_worker_control_paused(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(
        Base, "scanner_crypto_lane_paused"
    )
    try:
        async with session_factory() as session:
            session.add(AppSettings(id="default"))
            await session.commit()

        @asynccontextmanager
        async def _session_cm():
            async with session_factory() as session:
                yield session

        def _factory():
            return _session_cm()

        monkeypatch.setattr(routes_settings, "AsyncSessionLocal", _factory)

        async def _fake_read(_session, worker_name, *, default_interval=None):
            return {
                "worker_name": "crypto",
                "is_enabled": True,
                "is_paused": True,
                "interval_seconds": 1,
                "requested_run_at": None,
                "updated_at": None,
            }

        monkeypatch.setattr(routes_settings, "read_worker_control", _fake_read)

        response = await routes_settings.get_scanner_settings()
        assert response.crypto_lane_enabled is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_scanner_settings_reflects_worker_control_enabled(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(
        Base, "scanner_crypto_lane_on"
    )
    try:
        async with session_factory() as session:
            session.add(AppSettings(id="default"))
            await session.commit()

        @asynccontextmanager
        async def _session_cm():
            async with session_factory() as session:
                yield session

        def _factory():
            return _session_cm()

        monkeypatch.setattr(routes_settings, "AsyncSessionLocal", _factory)

        async def _fake_read(_session, worker_name, *, default_interval=None):
            return {
                "worker_name": "crypto",
                "is_enabled": True,
                "is_paused": False,
                "interval_seconds": 1,
                "requested_run_at": None,
                "updated_at": None,
            }

        monkeypatch.setattr(routes_settings, "read_worker_control", _fake_read)

        response = await routes_settings.get_scanner_settings()
        assert response.crypto_lane_enabled is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_read_crypto_lane_enabled_defaults_true_on_db_error(monkeypatch):
    async def _boom(_session, worker_name, *, default_interval=None):
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(routes_settings, "read_worker_control", _boom)

    @asynccontextmanager
    async def _session_cm():
        yield object()

    def _factory():
        return _session_cm()

    monkeypatch.setattr(routes_settings, "AsyncSessionLocal", _factory)

    assert await routes_settings._read_crypto_lane_enabled() is True
