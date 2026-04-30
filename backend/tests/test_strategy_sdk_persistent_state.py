"""Tests for ``StrategySDK.PersistentState``.

The cache mechanics (get / set / delete / dirty tracking) are
exercised in-memory and do not need a database. The async ``load`` /
``flush`` round-trip is exercised against a Postgres test database
because the upsert path uses the Postgres-specific
``ON CONFLICT DO UPDATE`` dialect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_helpers.persistent_state import PersistentState


# ---------------------------------------------------------------------------
# Construction + cache mechanics (no DB)
# ---------------------------------------------------------------------------


def test_requires_non_empty_slug():
    with pytest.raises(ValueError):
        PersistentState("")
    with pytest.raises(ValueError):
        PersistentState("   ")


def test_slug_is_normalised_to_lowercase_trimmed():
    state = PersistentState("  My_Strategy  ")
    assert state.strategy_slug == "my_strategy"


def test_get_returns_default_when_unset():
    state = PersistentState("my_strategy")
    assert state.get("missing") is None
    assert state.get("missing", default=42) == 42


def test_set_then_get_round_trip():
    state = PersistentState("my_strategy")
    state.set("count", 7)
    state.set("nested", {"a": 1, "b": [2, 3]})
    assert state.get("count") == 7
    assert state.get("nested") == {"a": 1, "b": [2, 3]}


def test_get_returns_deep_copy():
    """Mutating the value returned from get() must not silently bypass set()."""
    state = PersistentState("my_strategy")
    state.set("payload", {"items": [1, 2, 3]})
    snapshot = state.get("payload")
    snapshot["items"].append(99)
    # Cache was not mutated by the caller's edit
    assert state.get("payload") == {"items": [1, 2, 3]}


def test_set_records_deep_copy():
    """set() should snapshot the value so later mutation doesn't poison cache."""
    state = PersistentState("my_strategy")
    payload = {"items": [1, 2, 3]}
    state.set("payload", payload)
    payload["items"].append(99)
    assert state.get("payload") == {"items": [1, 2, 3]}


def test_set_marks_dirty():
    state = PersistentState("my_strategy")
    assert state.dirty is False
    state.set("foo", 1)
    assert state.dirty is True


def test_delete_marks_dirty_and_removes_from_cache():
    state = PersistentState("my_strategy")
    state.set("foo", 1)
    # Reset dirty by simulating a flush completion (private API used for the
    # test so we can isolate delete()'s behaviour).
    state._dirty.clear()
    state.delete("foo")
    assert "foo" not in state
    assert state.dirty is True


def test_delete_of_missing_key_still_marks_dirty():
    """Deleting a key that's not in cache still queues a DB delete — the
    row may exist on disk even if the cache hasn't loaded it."""
    state = PersistentState("my_strategy")
    state.delete("never_set")
    assert state.dirty is True


def test_set_rejects_empty_or_non_string_keys():
    state = PersistentState("my_strategy")
    with pytest.raises(ValueError):
        state.set("", 1)
    with pytest.raises(ValueError):
        state.set(None, 1)  # type: ignore[arg-type]


def test_keys_and_contains_and_len():
    state = PersistentState("my_strategy")
    state.set("a", 1)
    state.set("b", 2)
    assert sorted(state.keys()) == ["a", "b"]
    assert "a" in state
    assert "missing" not in state
    assert len(state) == 2


def test_loaded_property_starts_false():
    state = PersistentState("my_strategy")
    assert state.loaded is False


# ---------------------------------------------------------------------------
# Async DB round-trip (Postgres) — exercised when the test DB is reachable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_and_flush_round_trip(tmp_path):
    pytest.importorskip("asyncpg")
    from models.database import Base
    from tests.postgres_test_db import build_postgres_session_factory

    engine, session_factory = await build_postgres_session_factory(Base, "persistent_state")
    try:
        state = PersistentState("my_strategy", session_factory=session_factory)

        # Empty load — no rows, cache stays empty but loaded flag flips.
        await state.load()
        assert state.loaded is True
        assert state.dirty is False
        assert len(state) == 0

        # Set + flush persists.
        state.set("counter", 10)
        state.set("config", {"depth": 4, "labels": ["5m", "15m"]})
        assert state.dirty is True
        await state.flush()
        assert state.dirty is False

        # Fresh instance reads the same values back.
        fresh = PersistentState("my_strategy", session_factory=session_factory)
        await fresh.load()
        assert fresh.get("counter") == 10
        assert fresh.get("config") == {"depth": 4, "labels": ["5m", "15m"]}

        # Update + delete in one batch.
        fresh.set("counter", 11)
        fresh.delete("config")
        await fresh.flush()

        latest = PersistentState("my_strategy", session_factory=session_factory)
        await latest.load()
        assert latest.get("counter") == 11
        assert "config" not in latest

        # Per-strategy isolation — a different slug starts empty.
        other = PersistentState("other_strategy", session_factory=session_factory)
        await other.load()
        assert len(other) == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_flush_is_noop_when_clean(tmp_path):
    pytest.importorskip("asyncpg")
    from models.database import Base
    from tests.postgres_test_db import build_postgres_session_factory

    engine, session_factory = await build_postgres_session_factory(Base, "persistent_state_clean")
    try:
        state = PersistentState("my_strategy", session_factory=session_factory)
        await state.flush()  # no dirty entries, no DB call expected to fail
        assert state.dirty is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_load_replaces_unflushed_local_writes(tmp_path):
    """load() drops unflushed local writes to match the durable state."""
    pytest.importorskip("asyncpg")
    from models.database import Base
    from tests.postgres_test_db import build_postgres_session_factory

    engine, session_factory = await build_postgres_session_factory(Base, "persistent_state_reload")
    try:
        state = PersistentState("my_strategy", session_factory=session_factory)
        state.set("local_only", "drop_me")
        assert state.dirty is True
        await state.load()
        assert "local_only" not in state
        assert state.dirty is False
    finally:
        await engine.dispose()
