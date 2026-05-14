"""Tests for ``services.trader_cycle_context``.

These exercise the public surface (``acquire``, lifecycle, drift
detection) without depending on a real DB.  All DB-bound paths are
mocked at the wrapper-function level, the same indirection the
production module uses to delegate the actual SQL.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import trader_cycle_context as tcc  # noqa: E402
from services import trader_hot_state as hot_state  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test starts with a fresh manager singleton + clean hot_state."""
    fresh_manager = tcc._TraderCycleContextManager()
    monkeypatch.setattr(tcc, "trader_cycle_context", fresh_manager)
    monkeypatch.setattr(tcc, "is_db_pressure_active", lambda: False)
    monkeypatch.setattr(tcc, "current_backpressure_level", lambda: 0.0)
    # Hot state is a process-global; clear it for test isolation.
    hot_state._snapshots.clear()  # type: ignore[attr-defined]
    hot_state._global_gross.clear()  # type: ignore[attr-defined]
    hot_state._global_daily_pnl.clear()  # type: ignore[attr-defined]
    hot_state._global_daily_pnl_date.clear()  # type: ignore[attr-defined]
    hot_state._recently_closed_markets.clear()  # type: ignore[attr-defined]
    yield


@pytest.fixture
def mock_worker_wrappers(monkeypatch):
    """Patch the worker's hot_state-pass-through wrappers to async mocks.

    Returns a dict of wrapper-name → AsyncMock so individual tests can
    set ``return_value`` on the values they care about.  The defaults
    return zero / empty for everything.
    """
    from workers import trader_orchestrator_worker as tow

    patches = {
        "get_open_position_count_for_trader": AsyncMock(return_value=0),
        "get_open_order_count_for_trader": AsyncMock(return_value=0),
        "get_occupied_market_ids_for_trader": AsyncMock(return_value=set()),
        "get_reentry_cooldown_market_ids_for_trader": AsyncMock(return_value=set()),
        "get_daily_realized_pnl": AsyncMock(return_value=0.0),
        "get_unrealized_pnl": AsyncMock(return_value=0.0),
        "get_consecutive_loss_count": AsyncMock(return_value=0),
        "get_last_resolved_loss_at": AsyncMock(return_value=None),
        "get_pending_live_exit_summary_for_trader": AsyncMock(
            return_value={
                "count": 0,
                "order_ids": [],
                "market_ids": [],
                "signal_ids": [],
                "statuses": {},
                "terminal_statuses": [],
                "identities": [],
                "identity_keys": [],
            }
        ),
        "_live_provider_failure_snapshot": AsyncMock(
            return_value={"count": 0, "window_seconds": 180, "errors": []}
        ),
    }
    for name, mock in patches.items():
        monkeypatch.setattr(tow, name, mock)
    return patches


# ── acquire(): basic path ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_reads_per_trader_values_from_wrappers(mock_worker_wrappers):
    mock_worker_wrappers["get_open_position_count_for_trader"].return_value = 3
    mock_worker_wrappers["get_open_order_count_for_trader"].return_value = 1
    mock_worker_wrappers["get_occupied_market_ids_for_trader"].return_value = {"market-1", "market-2"}
    mock_worker_wrappers["get_reentry_cooldown_market_ids_for_trader"].return_value = {"market-9"}
    mock_worker_wrappers["get_daily_realized_pnl"].return_value = 12.5
    mock_worker_wrappers["get_unrealized_pnl"].return_value = -3.25
    mock_worker_wrappers["get_consecutive_loss_count"].return_value = 2
    sample_dt = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_worker_wrappers["get_last_resolved_loss_at"].return_value = sample_dt

    ctx = await tcc.trader_cycle_context.acquire(trader_id="trader-1", mode="live")

    assert ctx.trader_id == "trader-1"
    assert ctx.mode == "live"
    assert ctx.open_position_count == 3
    assert ctx.open_order_count == 1
    assert ctx.occupied_market_ids == frozenset({"market-1", "market-2"})
    assert ctx.reentry_cooldown_market_ids == frozenset({"market-9"})
    assert ctx.trader_daily_realized_pnl == 12.5
    assert ctx.trader_unrealized_pnl == -3.25
    assert ctx.consecutive_loss_count == 2
    assert ctx.last_resolved_loss_at == sample_dt


@pytest.mark.asyncio
async def test_acquire_short_circuits_pending_live_exit_for_non_live_mode(mock_worker_wrappers):
    """Pending-live-exit projection is a no-op for shadow mode — the
    DB delegate must not be called.
    """
    ctx = await tcc.trader_cycle_context.acquire(trader_id="trader-1", mode="shadow")
    assert ctx.mode == "shadow"
    assert ctx.pending_live_exit_summary["count"] == 0
    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].assert_not_called()


@pytest.mark.asyncio
async def test_acquire_pending_live_exit_summary_routes_through_wrapper(mock_worker_wrappers):
    """In live mode the projection delegates to the worker wrapper —
    so a mocked wrapper return shows up on the context.
    """
    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].return_value = {
        "count": 2,
        "order_ids": ["o1", "o2"],
        "market_ids": ["m1"],
        "signal_ids": ["s1"],
        "statuses": {"pending_cancel": 2},
        "terminal_statuses": [],
        "identities": [{"order_id": "o1", "market_id": "m1"}],
        "identity_keys": ["m1|long|s1"],
    }
    ctx = await tcc.trader_cycle_context.acquire(trader_id="trader-1", mode="live")

    assert ctx.pending_live_exit_summary["count"] == 2
    assert ctx.pending_live_exit_summary["order_ids"] == ("o1", "o2")
    # Frozen = immutable mapping; the test mustn't be able to mutate it.
    with pytest.raises(TypeError):
        ctx.pending_live_exit_summary["count"] = 99


@pytest.mark.asyncio
async def test_acquire_live_provider_failure_routes_through_wrapper(mock_worker_wrappers):
    mock_worker_wrappers["_live_provider_failure_snapshot"].return_value = {
        "count": 4,
        "window_seconds": 180,
        "errors": [{"order_id": "x", "error": "boom"}],
    }
    ctx = await tcc.trader_cycle_context.acquire(
        trader_id="trader-2", mode="live", provider_window_seconds=180
    )
    assert ctx.live_provider_failure_snapshot["count"] == 4
    errors = ctx.live_provider_failure_snapshot["errors"]
    assert len(errors) == 1
    assert errors[0]["error"] == "boom"


@pytest.mark.asyncio
async def test_acquire_handles_wrapper_failure_with_safe_fallback(mock_worker_wrappers):
    """A wrapper raising should never propagate — the field falls back
    to the documented empty value.
    """
    mock_worker_wrappers["get_open_position_count_for_trader"].side_effect = RuntimeError("DB down")
    ctx = await tcc.trader_cycle_context.acquire(trader_id="trader-3", mode="live")
    assert ctx.open_position_count == 0


# ── Caching + invalidation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_pending_live_exit_uses_cache_within_ttl(mock_worker_wrappers):
    """Two acquires within the TTL must hit the wrapper exactly once."""
    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].return_value = {"count": 1}
    await tcc.trader_cycle_context.acquire(trader_id="trader-4", mode="live")
    await tcc.trader_cycle_context.acquire(trader_id="trader-4", mode="live")
    assert mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].await_count == 1


@pytest.mark.asyncio
async def test_pending_live_exit_invalidated_by_event(monkeypatch, mock_worker_wrappers):
    """A trader_order event must mark the projection stale, forcing
    a refresh on the next acquire.
    """
    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].return_value = {"count": 0}

    # Prime the cache.
    await tcc.trader_cycle_context.acquire(trader_id="trader-5", mode="live")
    assert mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].await_count == 1

    # Fire an order event for that trader.
    await tcc.trader_cycle_context._on_trader_order_event(
        "trader_order", {"trader_id": "trader-5", "mode": "live"}
    )

    # Next acquire should refresh.
    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].return_value = {"count": 7}
    ctx = await tcc.trader_cycle_context.acquire(trader_id="trader-5", mode="live")
    assert ctx.pending_live_exit_summary["count"] == 7
    assert mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].await_count == 2


@pytest.mark.asyncio
async def test_event_for_other_trader_does_not_invalidate(mock_worker_wrappers):
    """An event for trader A must not invalidate trader B's cache."""
    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].return_value = {"count": 0}
    await tcc.trader_cycle_context.acquire(trader_id="trader-A", mode="live")
    await tcc.trader_cycle_context.acquire(trader_id="trader-B", mode="live")
    assert mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].await_count == 2

    await tcc.trader_cycle_context._on_trader_order_event(
        "trader_order", {"trader_id": "trader-A", "mode": "live"}
    )
    # B should still hit the cache.
    await tcc.trader_cycle_context.acquire(trader_id="trader-B", mode="live")
    assert mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].await_count == 2

    # A must refresh.
    await tcc.trader_cycle_context.acquire(trader_id="trader-A", mode="live")
    assert mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].await_count == 3


@pytest.mark.asyncio
async def test_pending_live_exit_query_timeout_serves_cached_or_empty(monkeypatch, mock_worker_wrappers):
    """When the delegate exceeds PROJECTION_QUERY_TIMEOUT_SECONDS the
    refresh times out and we serve cached or empty rather than
    propagating.
    """

    async def _slow(*args, **kwargs):
        await asyncio.sleep(10)  # > PROJECTION_QUERY_TIMEOUT_SECONDS
        return {"count": 99}

    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].side_effect = _slow
    monkeypatch.setattr(tcc, "PROJECTION_QUERY_TIMEOUT_SECONDS", 0.05)

    ctx = await tcc.trader_cycle_context.acquire(trader_id="trader-6", mode="live")
    # No prior cache → empty sentinel.
    assert ctx.pending_live_exit_summary["count"] == 0


# ── Single-flight ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_acquires_for_same_trader_single_flight_pending(mock_worker_wrappers):
    """Two concurrent acquires that miss the cache must trigger ONE
    delegate call, not two.
    """
    call_count = {"n": 0}

    async def _delegate(*args, **kwargs):
        call_count["n"] += 1
        await asyncio.sleep(0.01)  # let the second acquire enter the lock
        return {"count": call_count["n"]}

    mock_worker_wrappers["get_pending_live_exit_summary_for_trader"].side_effect = _delegate

    results = await asyncio.gather(
        tcc.trader_cycle_context.acquire(trader_id="trader-7", mode="live"),
        tcc.trader_cycle_context.acquire(trader_id="trader-7", mode="live"),
    )
    # Both contexts read the same single-flight result.
    assert results[0].pending_live_exit_summary["count"] == 1
    assert results[1].pending_live_exit_summary["count"] == 1
    assert call_count["n"] == 1


# ── Lifecycle ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_stop_idempotent(monkeypatch):
    """start() and stop() are both safe to call multiple times."""
    fresh = tcc._TraderCycleContextManager()
    # Stub out the validation_service import inside the refresh loop.
    fake_vs = SimpleNamespace(get_demoted_strategy_types=AsyncMock(return_value=set()))
    import services.validation_service as vs_module

    monkeypatch.setattr(vs_module, "validation_service", fake_vs)

    await fresh.start()
    await fresh.start()  # idempotent
    await fresh.wait_warm(timeout=2.0)
    snapshot = fresh.get_global_snapshot()
    assert snapshot.is_warm is True

    await fresh.stop()
    await fresh.stop()  # idempotent


@pytest.mark.asyncio
async def test_global_snapshot_warm_after_first_refresh(monkeypatch):
    fresh = tcc._TraderCycleContextManager()
    fake_vs = SimpleNamespace(get_demoted_strategy_types=AsyncMock(return_value={"strat-x"}))
    import services.validation_service as vs_module

    monkeypatch.setattr(vs_module, "validation_service", fake_vs)
    try:
        await fresh.start()
        warm = await fresh.wait_warm(timeout=2.0)
        assert warm is True
        snapshot = fresh.get_global_snapshot()
        assert snapshot.is_warm is True
        assert "strat-x" in snapshot.demoted_strategy_types
    finally:
        await fresh.stop()


# ── Drift comparison helpers ────────────────────────────────────────


def test_pending_live_exit_drifted_detects_count_change():
    a = {"count": 1, "order_ids": ["o1"]}
    b = {"count": 2, "order_ids": ["o1", "o2"]}
    assert tcc._pending_live_exit_drifted(a, b) is True


def test_pending_live_exit_drifted_detects_order_id_set_change_with_same_count():
    a = {"count": 1, "order_ids": ["o1"]}
    b = {"count": 1, "order_ids": ["o2"]}
    assert tcc._pending_live_exit_drifted(a, b) is True


def test_pending_live_exit_drifted_returns_false_for_identical():
    a = {"count": 2, "order_ids": ["o1", "o2"]}
    b = {"count": 2, "order_ids": ["o2", "o1"]}  # order independence
    assert tcc._pending_live_exit_drifted(a, b) is False


def test_provider_failure_drifted_detects_count():
    a = {"count": 1, "errors": [{"order_id": "o1"}]}
    b = {"count": 2, "errors": [{"order_id": "o1"}, {"order_id": "o2"}]}
    assert tcc._provider_failure_drifted(a, b) is True


# ── Frozen-mapping immutability ─────────────────────────────────────


def test_frozen_pending_live_exit_is_immutable():
    raw = {
        "count": 1,
        "order_ids": ["o1"],
        "market_ids": ["m1"],
        "signal_ids": ["s1"],
        "statuses": {"pending": 1},
        "terminal_statuses": [],
        "identities": [{"order_id": "o1"}],
        "identity_keys": ["m1|long|s1"],
    }
    frozen = tcc._freeze_pending_live_exit(raw)
    with pytest.raises(TypeError):
        frozen["count"] = 99
    with pytest.raises(TypeError):
        frozen["statuses"]["pending"] = 99
