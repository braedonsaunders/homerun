"""Read-only mode for live_execution_service (A.3 cold reconciliation plane).

The cold reconciliation plane loads CLOB credentials so its authenticated READS
succeed (order snapshots, wallet/balance) but must make ZERO venue mutations.
`set_read_only(True)` is the single, central guarantee: every place / cancel /
allowance method short-circuits. This is what lets the cold plane carry the
heavy reconcile off the trading event loop without ever executing (the trading
plane stays the sole executor) and without the "Invalid username/password"
auth-lockout the first cut hit (creds-less authenticated reads).
"""
from __future__ import annotations

import pytest

from services.live_execution_service import LiveExecutionService


def test_set_read_only_toggles_flag():
    svc = LiveExecutionService()
    assert svc._read_only is False
    svc.set_read_only(True)
    assert svc._read_only is True
    svc.set_read_only(False)
    assert svc._read_only is False


@pytest.mark.asyncio
async def test_read_only_blocks_every_venue_mutation():
    """In read-only mode, all 5 venue-write methods short-circuit before any
    CLOB call: place_order hard-raises; cancel/allowance refreshes no-op."""
    svc = LiveExecutionService()
    svc.set_read_only(True)

    # place_order cannot return a fake Order -> hard raise.
    with pytest.raises(RuntimeError, match="read-only"):
        await svc.place_order("token-1", None, 0.5, 10.0)

    # cancel + allowance refreshes -> no-op return values (caller handles).
    assert await svc.cancel_order("order-1") is False
    assert await svc.refresh_conditional_balance_allowance("token-1") is False
    assert await svc.refresh_collateral_balance_allowance() is False
    assert await svc._approve_clob_allowance() is None


@pytest.mark.asyncio
async def test_not_read_only_passes_the_gate():
    """With read-only OFF (the trading plane), place_order passes the gate and
    reaches normal arg validation (empty token -> ValueError, NOT the read-only
    RuntimeError) — proving the gate adds nothing to the trading path."""
    svc = LiveExecutionService()
    assert svc._read_only is False
    with pytest.raises(ValueError):
        await svc.place_order("", None, 0.5, 10.0)
