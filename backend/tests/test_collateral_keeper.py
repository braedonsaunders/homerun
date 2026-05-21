"""Tests for the in-memory collateral position-keeper.

The keeper is the sub-second pre-trade-risk replacement for the legacy
chain-probe BUY pre-submit gate.  These tests exercise the contract the
hot path depends on:

* Bootstrap on first gate call seeds the keeper from a chain read.
* Steady-state gate is pure-in-memory: no chain call per signal.
* Atomic reserve / release semantics — two concurrent callers cannot
  both reserve when only one of them fits the budget.
* Reconciler detects chain-vs-keeper divergence and halts the keeper
  after the configured streak, halting all subsequent BUY submissions
  with a clear reason.
* Halt is operator-cleared (does not auto-resume), and a divergence
  inside tolerance resets the streak without halting.
* Stale-data window falls back to the legacy chain-probe path.
* Feature flag (``HOMERUN_COLLATERAL_KEEPER_ENABLED``) routes to the
  legacy gate when disabled.
* Reservation is released on every failure path that releases the
  stats reservation.
"""

from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
import services.live_execution_service as live_execution_module
from services.live_execution_service import (
    LiveExecutionService,
    OrderSide,
    _KEEPER_DIVERGENCE_TOLERANCE_USD,
    _KEEPER_HALT_THRESHOLD,
    _KEEPER_MAX_STALENESS_SECONDS,
    _keeper_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(*, available: float | None = 1000.0, min_balance: float = 0.0) -> LiveExecutionService:
    """Build a service whose ``get_balance`` returns the configured
    snapshot.  No network, no SDK — every test runs in pure memory.
    """
    service = LiveExecutionService()
    service._initialized = True
    # ``is_ready()`` checks ``self._client is not None`` — set a sentinel
    # so ``_validate_order`` doesn't short-circuit on "not initialized".
    service._client = object()
    # ``_validate_order`` reads settings.MIN_ORDER_SIZE_USD and
    # MAX_TRADE_SIZE_USD and friends; default values are fine for tests
    # but make MAX_TRADE_SIZE_USD generous so our $300 BUYs don't trip
    # the per-trade cap.
    settings.MAX_TRADE_SIZE_USD = 10_000.0
    settings.MIN_ORDER_SIZE_USD = 1.0
    settings.MAX_DAILY_TRADE_VOLUME = 1_000_000.0

    if available is None:
        async def _fake_balance(*, force_probe_all: bool = False) -> dict:
            return {"error": "unreachable"}
    else:
        async def _fake_balance(*, force_probe_all: bool = False) -> dict:
            return {
                "address": "0xtest",
                "balance": float(available),
                "available": float(available),
                "reserved": 0.0,
                "currency": "USDC",
                "timestamp": "1970-01-01T00:00:00+00:00",
                "positions_value": 0.0,
                "signature_type": 0,
            }

    service.get_balance = _fake_balance  # type: ignore[assignment]

    # Patch MIN_ACCOUNT_BALANCE_USD via the settings reference the
    # service holds.  Setattr on the settings object is the same hook
    # the broader test suite uses.
    settings.MIN_ACCOUNT_BALANCE_USD = float(min_balance)
    return service


async def _drive_balance(service: LiveExecutionService, available: float | None) -> None:
    """Swap the service's ``get_balance`` mid-test (used to simulate
    chain state changing between reconciler ticks).
    """
    if available is None:
        async def _fake_balance(*, force_probe_all: bool = False) -> dict:
            return {"error": "unreachable"}
    else:
        async def _fake_balance(*, force_probe_all: bool = False) -> dict:
            return {"available": float(available)}

    service.get_balance = _fake_balance  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_bootstraps_keeper_on_first_call(monkeypatch):
    """First gate call seeds the keeper from a single chain read."""
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=500.0)

    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    assert ok, err
    assert err is None
    assert service._keeper_initialized is True
    assert service._keeper_chain_available == Decimal("500")
    assert service._keeper_reserved_since_chain == Decimal("0")


@pytest.mark.asyncio
async def test_gate_falls_back_to_chain_when_bootstrap_fails(monkeypatch):
    """If the chain is unreachable on bootstrap, the gate falls back to
    legacy behavior (which itself skips on unreachable chain) so we
    don't block trading on a cold keeper.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=None)

    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    # Legacy gate's "balance unreachable" path returns True (skip gate,
    # let submit_leg do the authoritative check).
    assert ok is True
    assert service._keeper_initialized is False


# ---------------------------------------------------------------------------
# Hot-path gate (no chain call after bootstrap)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_steady_state_does_not_call_chain(monkeypatch):
    """After bootstrap, the gate is a pure-in-memory compare."""
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=500.0)

    # First call bootstraps and may call get_balance.
    await service._enforce_buy_pre_submit_gate(token_id="tok", required_notional_usd=Decimal("10"))
    # Replace get_balance with one that raises if called — second gate
    # call must NOT touch the chain.
    call_count = {"n": 0}

    async def _raise(*, force_probe_all: bool = False):
        call_count["n"] += 1
        raise AssertionError("steady-state gate must not call chain")

    service.get_balance = _raise  # type: ignore[assignment]
    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    assert ok, err
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_gate_rejects_when_effective_below_required(monkeypatch):
    """Gate rejects when keeper's effective available is below the
    requested notional + MIN_ACCOUNT_BALANCE_USD.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=50.0, min_balance=10.0)

    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    assert ok is False
    assert err is not None
    assert "insufficient effective collateral" in err


# ---------------------------------------------------------------------------
# Atomic reserve + release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_reservations_cannot_overdraw(monkeypatch):
    """Two concurrent callers requesting 600 each against $1000 cannot
    both succeed (atomic check-and-reserve under the keeper lock).
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    # Bootstrap first.
    async with service._get_keeper_lock():
        bootstrapped = await service._keeper_bootstrap_locked()
    assert bootstrapped

    async def _reserve(amount: int) -> tuple[bool, str | None]:
        return await service._keeper_try_reserve(Decimal(amount))

    results = await asyncio.gather(_reserve(600), _reserve(600))
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    assert len(successes) == 1, "exactly one reservation must succeed"
    assert len(failures) == 1
    assert "insufficient effective collateral" in (failures[0][1] or "")
    assert service._keeper_reserved_since_chain == Decimal("600")


@pytest.mark.asyncio
async def test_reserve_then_release_restores_budget(monkeypatch):
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    ok, _ = await service._keeper_try_reserve(Decimal("300"))
    assert ok
    assert service._keeper_effective_available() == Decimal("700")

    await service._keeper_release(Decimal("300"))
    assert service._keeper_effective_available() == Decimal("1000")


@pytest.mark.asyncio
async def test_release_is_idempotent_clamps_at_zero(monkeypatch):
    """Double-release must not drive reserved negative."""
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    await service._keeper_try_reserve(Decimal("100"))
    await service._keeper_release(Decimal("100"))
    await service._keeper_release(Decimal("100"))  # duplicate
    assert service._keeper_reserved_since_chain == Decimal("0")
    assert service._keeper_effective_available() == Decimal("1000")


# ---------------------------------------------------------------------------
# Reconciler / halt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciler_settles_local_reservations_into_chain(monkeypatch):
    """A successful reconcile after a reservation settles the local
    delta — the new chain reading IS the venue's view of the just-
    placed order, so the keeper resets the local reservation counter
    against it instead of double-counting.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    # Reserve $300 locally — keeper view says effective $700.
    await service._keeper_try_reserve(Decimal("300"))
    assert service._keeper_effective_available() == Decimal("700")

    # Chain has now processed the order — venue-side available is $700.
    await _drive_balance(service, 700.0)
    ok = await service._keeper_reconcile_once()
    assert ok

    # Local delta settled — effective unchanged ($700) but the chain
    # reading now is the canonical $700 and reserved_since_chain reset.
    assert service._keeper_chain_available == Decimal("700")
    assert service._keeper_reserved_since_chain == Decimal("0")
    assert service._keeper_effective_available() == Decimal("700")


@pytest.mark.asyncio
async def test_reconciler_halts_after_threshold_consecutive_down_divergences(monkeypatch):
    """``_KEEPER_HALT_THRESHOLD`` consecutive DOWN-divergent reads halt
    the keeper.  Subsequent BUY gate calls reject with the halt reason.

    DOWN-divergence (chain < expected) is the dangerous case: we may
    have spent money we didn't reserve.  This must halt to prevent
    over-spending at the venue.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    # Chain drops to $500 (we expected $1000) — DOWN by $500, dangerous.
    await _drive_balance(service, 500.0)
    for _ in range(_KEEPER_HALT_THRESHOLD):
        await service._keeper_reconcile_once()

    assert service._keeper_halted is True
    assert service._keeper_halt_reason is not None
    assert "down-divergence" in service._keeper_halt_reason.lower()

    # Gate now rejects with the halt reason surfaced.
    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("10"),
    )
    assert ok is False
    assert err is not None
    err_lower = err.lower()
    assert "divergence" in err_lower
    assert "clear_collateral_halt" in err_lower


@pytest.mark.asyncio
async def test_reconciler_auto_accepts_up_divergence_without_halt(monkeypatch):
    """UP-divergence (chain > expected) is BENIGN and must NOT halt.

    Real-world causes for chain-available going UP unexpectedly:
      * External wallet deposit
      * Venue auto-cancelled a working order on market resolution
      * Slippage refund larger than the conservative reservation
      * Position settled and venue returned the collateral

    The 5/2026/05 production halt was exactly this case — a $10.98
    UP-divergence on the trading wallet (likely a venue-side cancel
    refund) caused the keeper to halt and block 267 BUY orders.  The
    fix: auto-accept the new (higher) chain reading and converge.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    # Chain JUMPS UP by $10.98 — exactly the production scenario.
    await _drive_balance(service, 1010.98)
    for _ in range(_KEEPER_HALT_THRESHOLD + 5):
        await service._keeper_reconcile_once()

    # Must NOT halt — UP-divergence is benign.
    assert service._keeper_halted is False
    # Keeper accepted the new chain reading.
    assert service._keeper_chain_available == Decimal("1010.98")
    # Streak never accumulated past zero (each reconcile resets it).
    assert service._keeper_divergence_streak == 0

    # Gate still works normally with the new (higher) balance.
    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    assert ok, err


@pytest.mark.asyncio
async def test_reconciler_up_divergence_settles_pending_reservations(monkeypatch):
    """UP-divergence auto-accept must still settle any pending
    reservations (the chain reading IS post-settlement of the venue's
    view) so we don't double-count.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    # Reserve $200, then chain jumps to $900 (we expected $800 after
    # settlement; chain is $100 higher — UP-divergence with pending
    # reservations).
    await service._keeper_try_reserve(Decimal("200"))
    await _drive_balance(service, 900.0)
    await service._keeper_reconcile_once()

    assert service._keeper_halted is False
    assert service._keeper_chain_available == Decimal("900")
    # Reservations settled (the chain reading is post-snapshot).
    assert service._keeper_reserved_since_chain == Decimal("0")
    assert service._keeper_effective_available() == Decimal("900")


@pytest.mark.asyncio
async def test_reconciler_single_down_divergence_does_not_halt(monkeypatch):
    """A one-shot DOWN-divergence (single chain read off, then back in
    line) must not halt — that's the in-flight-order race we explicitly
    want to filter.

    Key semantics: on DOWN-divergence we do NOT silently absorb the
    new chain value into the keeper.  The streak only resets when the
    chain reading matches our model again (i.e. the in-flight order
    catches up at the venue and the next chain read shows the
    expected balance).
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    # Chain blips DOWN to $500 — DOWN-divergent vs the keeper's $1000
    # model.  Could be a real loss of bookkeeping (dangerous) or just
    # an in-flight order race — keeper holds its model.
    await _drive_balance(service, 500.0)
    await service._keeper_reconcile_once()
    assert service._keeper_divergence_streak == 1
    # Keeper model held — chain reading was NOT silently absorbed.
    assert service._keeper_chain_available == Decimal("1000")

    # Chain recovers — matches expected ($1000 with zero reserved).
    # Streak resets and the keeper accepts the (unchanged) reading.
    await _drive_balance(service, 1000.0)
    await service._keeper_reconcile_once()
    assert service._keeper_divergence_streak == 0
    assert service._keeper_halted is False
    assert service._keeper_chain_available == Decimal("1000")


@pytest.mark.asyncio
async def test_tolerance_window_absorbs_small_drift(monkeypatch):
    """Drift within ``_KEEPER_DIVERGENCE_TOLERANCE_USD`` does not bump
    the streak (covers normal slippage refunds on partial fills).
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    drift = float(_KEEPER_DIVERGENCE_TOLERANCE_USD) * 0.5
    await _drive_balance(service, 1000.0 - drift)
    for _ in range(_KEEPER_HALT_THRESHOLD + 2):
        await service._keeper_reconcile_once()

    assert service._keeper_halted is False
    assert service._keeper_divergence_streak == 0


@pytest.mark.asyncio
async def test_halt_clears_only_via_operator_action(monkeypatch):
    """Halts do NOT auto-clear on a clean reconcile — the divergence
    needs a human to acknowledge before resuming.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    await _drive_balance(service, 500.0)
    for _ in range(_KEEPER_HALT_THRESHOLD):
        await service._keeper_reconcile_once()
    assert service._keeper_halted is True

    # Reconcile cleanly — still halted.
    await _drive_balance(service, 500.0)
    await service._keeper_reconcile_once()
    assert service._keeper_halted is True

    # Operator clear — resumes.
    cleared = service.clear_collateral_halt(reason="reviewed by ops")
    assert cleared is True
    assert service._keeper_halted is False
    assert service._keeper_halt_reason is None


# ---------------------------------------------------------------------------
# Staleness / feature flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_keeper_falls_back_to_legacy_chain_probe(monkeypatch):
    """If the reconciler hasn't run in ``_KEEPER_MAX_STALENESS_SECONDS``
    we fall back to the legacy chain-probe gate.  Backstop for
    operational issues with the background loop.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0)
    # Bootstrap then artificially age the keeper.
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()
    service._keeper_last_chain_at -= _KEEPER_MAX_STALENESS_SECONDS + 1.0

    # Drive get_balance to count calls — the gate should fall back to
    # legacy, which calls get_balance.
    call_count = {"n": 0}

    async def _counted(*, force_probe_all: bool = False):
        call_count["n"] += 1
        return {
            "address": "0x",
            "balance": 1000.0,
            "available": 1000.0,
            "reserved": 0.0,
            "currency": "USDC",
            "timestamp": "",
            "positions_value": 0.0,
            "signature_type": 0,
        }

    service.get_balance = _counted  # type: ignore[assignment]
    ok, err = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    assert ok, err
    assert call_count["n"] >= 1  # legacy path called the chain


@pytest.mark.asyncio
async def test_feature_flag_disabled_uses_legacy_gate(monkeypatch):
    """``HOMERUN_COLLATERAL_KEEPER_ENABLED=0`` reverts to the legacy
    chain-probe gate — the kill-switch for rollout safety.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "0")
    assert _keeper_enabled() is False

    service = _make_service(available=1000.0)
    call_count = {"n": 0}

    async def _counted(*, force_probe_all: bool = False):
        call_count["n"] += 1
        return {
            "address": "0x",
            "balance": 1000.0,
            "available": 1000.0,
            "reserved": 0.0,
            "currency": "USDC",
            "timestamp": "",
            "positions_value": 0.0,
            "signature_type": 0,
        }

    service.get_balance = _counted  # type: ignore[assignment]
    ok, _ = await service._enforce_buy_pre_submit_gate(
        token_id="tok",
        required_notional_usd=Decimal("100"),
    )
    assert ok
    # Legacy gate calls the chain — and the keeper stayed uninitialized.
    assert call_count["n"] >= 1
    assert service._keeper_initialized is False


# ---------------------------------------------------------------------------
# Integration: reserve via _validate_and_reserve_order, release on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_and_reserve_order_claims_keeper_for_buy(monkeypatch):
    """``_validate_and_reserve_order`` must claim the keeper budget
    BEFORE the stats reservation so two concurrent BUY signals cannot
    both bypass the gate via the post-gate race window.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0, min_balance=0.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    ok, err = await service._validate_and_reserve_order(
        size_usd=Decimal("300"),
        side=OrderSide.BUY,
        token_id="tok",
    )
    assert ok, err
    # Stats AND keeper both updated.
    assert service._keeper_reserved_since_chain == Decimal("300")
    assert service._daily_volume == Decimal("300")


@pytest.mark.asyncio
async def test_release_reservation_releases_keeper_for_buy(monkeypatch):
    """``_release_reservation`` on a BUY must release the keeper claim
    so the next signal sees the freed-up budget without waiting for
    a chain reconcile.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0, min_balance=0.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    # Stub _persist_runtime_state — it touches DB.
    service._persist_runtime_state = AsyncMock(return_value=None)  # type: ignore[assignment]

    # Reserve first.
    await service._validate_and_reserve_order(
        size_usd=Decimal("300"),
        side=OrderSide.BUY,
        token_id="tok",
    )
    assert service._keeper_reserved_since_chain == Decimal("300")

    # Release.
    await service._release_reservation(
        size_usd=Decimal("300"),
        side=OrderSide.BUY,
        token_id="tok",
    )
    assert service._keeper_reserved_since_chain == Decimal("0")
    assert service._daily_volume == Decimal("0")


@pytest.mark.asyncio
async def test_sell_order_does_not_touch_keeper(monkeypatch):
    """SELL orders don't consume collateral — keeper must stay
    untouched so we don't accidentally allow more BUYs than we
    actually have collateral for.
    """
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "1")
    service = _make_service(available=1000.0, min_balance=0.0)
    async with service._get_keeper_lock():
        await service._keeper_bootstrap_locked()

    await service._validate_and_reserve_order(
        size_usd=Decimal("300"),
        side=OrderSide.SELL,
        token_id="tok",
    )
    assert service._keeper_reserved_since_chain == Decimal("0")
    assert service._keeper_chain_available == Decimal("1000")
