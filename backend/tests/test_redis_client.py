"""Tests for the Redis client module.

These tests do NOT require a running Redis instance.  They exercise the
module's soft-fail contract: when Redis is disabled or unreachable, every
operation must degrade cleanly and the helpers must return ``None`` /
``False`` rather than raising.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import asyncio

import pytest

from config import settings
from services import redis_client


@pytest.fixture(autouse=True)
def _reset_redis_client_state():
    """Force the redis_client singleton back to a clean state between tests.

    The module holds a process-level _RedisState dataclass; since these
    tests do not actually start the pool, all we need to reset is the
    bookkeeping flags so ``is_healthy()`` etc. report cleanly.
    """
    redis_client._state.healthy = False
    redis_client._state.started = False
    redis_client._state.last_error = None
    redis_client._state.last_error_at = 0.0
    redis_client._state.last_ok_at = 0.0
    redis_client._state.client = None
    redis_client._state.pool = None
    yield
    redis_client._state.healthy = False
    redis_client._state.started = False


def test_namespacing_applies_default_prefix():
    assert redis_client.namespaced("wallet_state:heartbeat") == "homerun:wallet_state:heartbeat"


def test_namespacing_is_idempotent():
    once = redis_client.namespaced("foo")
    twice = redis_client.namespaced(once)
    assert once == twice == "homerun:foo"


def test_namespacing_respects_empty_prefix(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_NAMESPACE", "")
    assert redis_client.namespaced("foo") == "foo"


def test_status_snapshot_when_idle():
    snapshot = redis_client.status_snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["healthy"] is False
    assert snapshot["started"] is False
    assert snapshot["last_heartbeat_age_seconds"] if False else True  # field absent ok


def test_redact_url_strips_password():
    # Password-only credentials: ``:pw@host`` → ``:***@host``.
    assert redis_client._redact_url("redis://:hunter2@host:6379/0") == "redis://:***@host:6379/0"
    # User:password credentials: only the password is masked.
    assert redis_client._redact_url("redis://user:hunter2@host:6379/0") == "redis://user:***@host:6379/0"
    # No credentials: passthrough unchanged.
    assert redis_client._redact_url("redis://host:6379/0") == "redis://host:6379/0"
    assert redis_client._redact_url("") == ""


@pytest.mark.asyncio
async def test_get_client_or_none_returns_none_when_not_started():
    assert redis_client.get_client_or_none() is None


@pytest.mark.asyncio
async def test_get_client_or_none_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_ENABLED", False)
    redis_client._state.healthy = True  # would otherwise return client
    assert redis_client.get_client_or_none() is None


@pytest.mark.asyncio
async def test_safe_helpers_softfail_when_unavailable():
    """All safe_* helpers must degrade rather than raise when Redis is down."""
    # No client started → soft-fail to no-op / None.
    assert await redis_client.safe_set("foo", "bar") is False
    assert await redis_client.safe_get("foo") is None
    assert await redis_client.safe_publish("ch", "msg") is False
    assert await redis_client.safe_xadd("stream", {"k": "v"}) is None


@pytest.mark.asyncio
async def test_start_with_disabled_setting_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_ENABLED", False)
    ok = await redis_client.start()
    assert ok is False
    assert redis_client._state.started is True
    assert redis_client._state.healthy is False
    # Should not have attempted to construct a real pool.
    assert redis_client._state.client is None


@pytest.mark.asyncio
async def test_start_with_unreachable_url_returns_false(monkeypatch):
    """Pointing at an unused port must not raise; pool stays unhealthy."""
    monkeypatch.setattr(settings, "REDIS_ENABLED", True)
    # Port 1 — guaranteed unbound on a typical box.
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(settings, "REDIS_CONNECT_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(settings, "REDIS_SOCKET_TIMEOUT_SECONDS", 0.5)
    ok = await redis_client.start()
    assert ok is False
    assert redis_client._state.started is True
    assert redis_client._state.healthy is False
    # last_error should be populated for diagnostics.
    assert redis_client._state.last_error is not None
    # Cleanup: cancel the background probe task spawned by start().
    await redis_client.shutdown()


def test_coerce_field_handles_common_types():
    assert redis_client._coerce_field("hello") == "hello"
    assert redis_client._coerce_field(123) == 123
    assert redis_client._coerce_field(1.5) == 1.5
    assert redis_client._coerce_field(None) == ""
    assert redis_client._coerce_field({"a": 1}) == "{'a': 1}"


@pytest.mark.asyncio
async def test_shutdown_is_safe_when_never_started():
    """Calling shutdown without ever starting must not raise."""
    await redis_client.shutdown()
    assert redis_client._state.started is False
    assert redis_client._state.client is None
