"""Phase 2 regression tests for the refactored venue preflight pipeline.

Asserts that the new gate-pipeline implementation of
``pre_db_venue_preflight`` (in ``session_engine``) and the standalone
predicates in ``venue_gates`` produce **exactly the same payload shapes**
as the Phase 1 inline implementation.

These complement the existing ``test_pre_db_venue_preflight.py`` suite,
which exercises the engine end-to-end.  Here we test the predicates and
the result-dict shape directly to lock in the byte-level contract for
downstream consumers (LegSubmitResult payload, decision row writes,
Slack alerts).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator import venue_gates as venue_gates_module
from services.trader_orchestrator.gate_pipeline import (
    CostClass,
    Gate,
    GateContext,
    GatePipeline,
)
from services.trader_orchestrator.venue_gates import (
    GATE_NAME_BUY_COLLATERAL,
    GATE_NAME_MAX_SPREAD_BPS,
    REASON_BUY_PRE_SUBMIT_GATE,
    REASON_MAX_SPREAD_BPS_EXCEEDED,
    buy_collateral_gate,
    max_spread_bps_gate,
)


def _leg(
    *,
    leg_id: str = "leg-1",
    token_id: str = "11111111111111111111",
    side: str = "buy",
    limit_price: float = 0.5,
    requested_notional_usd: float = 10.0,
    requested_shares: float = 20.0,
) -> dict:
    return {
        "leg_id": leg_id,
        "market_id": "market-vg",
        "market_question": "Will the test pass?",
        "token_id": token_id,
        "side": side,
        "outcome": "yes",
        "requested_notional_usd": requested_notional_usd,
        "requested_shares": requested_shares,
        "limit_price": limit_price,
        "price_policy": "taker_limit",
        "time_in_force": "IOC",
        "post_only": False,
    }


def _signal() -> SimpleNamespace:
    return SimpleNamespace(
        id="signal-vg",
        source="scanner",
        trace_id="trace-vg",
        strategy_type="generic_strategy",
        strategy_context_json={},
        payload_json={
            "selected_token_id": "11111111111111111111",
            "live_market": {"selected_token_id": "11111111111111111111"},
        },
        market_id="market-vg",
        market_question="Will the test pass?",
        direction="buy_yes",
        entry_price=0.5,
        edge_percent=5.0,
        confidence=0.7,
    )


def _ctx(*, leg: dict, mode: str = "live", risk_limits: dict | None = None) -> GateContext:
    return GateContext(
        runtime_signal=_signal(),
        decision=None,
        leg=leg,
        live_context={},
        risk_limits=risk_limits or {},
        strategy_params={},
        mode=mode,
    )


# ---------------------------------------------------------------------------
# buy_collateral_gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buy_collateral_gate_passes_on_shadow_mode(monkeypatch):
    """Shadow mode ⇒ collateral gate is a no-op pass."""
    # Set up the live-service mock so we can assert it was NOT called.
    called = AsyncMock(return_value=(False, "should not run"))
    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        called,
    )
    result = await buy_collateral_gate(_ctx(leg=_leg(side="buy"), mode="shadow"))
    assert result.passed is True
    called.assert_not_awaited()


@pytest.mark.asyncio
async def test_buy_collateral_gate_passes_on_sell_leg(monkeypatch):
    """SELL leg ⇒ no USDC check needed; gate passes."""
    called = AsyncMock(return_value=(False, "should not run"))
    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        called,
    )
    result = await buy_collateral_gate(_ctx(leg=_leg(side="sell"), mode="live"))
    assert result.passed is True
    called.assert_not_awaited()


@pytest.mark.asyncio
async def test_buy_collateral_gate_rejects_when_live_gate_fails(monkeypatch):
    """Live-service gate returns (False, msg) → predicate emits the
    payload_extras shape Phase 1's inline code wrote."""
    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(False, "Insufficient collateral: need 10.00, have 0.00.")),
    )
    leg = _leg(leg_id="leg-buyfail", side="buy", requested_notional_usd=10.0)
    result = await buy_collateral_gate(_ctx(leg=leg, mode="live"))

    assert result.passed is False
    assert result.reason == REASON_BUY_PRE_SUBMIT_GATE
    assert result.error_message.startswith("Insufficient collateral")

    payload_extras = result.detail["payload_extras"]
    # Lock in every key Phase 1's inline code wrote, so consumers like
    # LegSubmitResult.payload don't see drift.
    assert payload_extras["mode"] == "live"
    assert payload_extras["submission"] == "skipped"
    assert payload_extras["reason"] == REASON_BUY_PRE_SUBMIT_GATE
    assert payload_extras["token_id"] == "11111111111111111111"
    assert payload_extras["preflight_rejected"] is True
    assert payload_extras["effective_notional_usd"] == 10.0
    assert payload_extras["requested_notional_usd"] == 10.0
    assert payload_extras["requested_shares"] == 20.0
    # The leg dict is copied — mutating the original must not change the
    # payload (Phase 1 inline did dict(leg_payload)).
    leg["limit_price"] = 0.99
    assert payload_extras["leg"]["limit_price"] == 0.5


@pytest.mark.asyncio
async def test_buy_collateral_gate_passes_when_live_gate_ok(monkeypatch):
    """Live-service gate returns (True, None) → predicate passes."""
    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(True, None)),
    )
    result = await buy_collateral_gate(_ctx(leg=_leg(side="buy"), mode="live"))
    assert result.passed is True


@pytest.mark.asyncio
async def test_buy_collateral_gate_falls_back_to_pass_on_exception(monkeypatch):
    """If the live-service gate raises, predicate falls back to pass —
    submit_leg's authoritative inline check is the backstop."""
    async def _boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        _boom,
    )
    result = await buy_collateral_gate(_ctx(leg=_leg(side="buy"), mode="live"))
    assert result.passed is True


# ---------------------------------------------------------------------------
# max_spread_bps_gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_spread_bps_gate_passes_when_cap_unset(monkeypatch):
    """No cap configured ⇒ gate is a no-op pass for both modes."""
    monkeypatch.setattr(
        venue_gates_module,
        "_resolve_shadow_book_and_tape",
        AsyncMock(return_value=(None, [], None, "test", None)),
    )
    result = await max_spread_bps_gate(_ctx(leg=_leg(), risk_limits={}))
    assert result.passed is True


@pytest.mark.asyncio
async def test_max_spread_bps_gate_rejects_with_phase1_payload(monkeypatch):
    """Spread cap breach ⇒ payload_extras matches Phase 1 inline shape."""
    wide_book = {
        "bids": [{"price": 0.20, "size": 100.0}],
        "asks": [{"price": 0.80, "size": 100.0}],
    }
    monkeypatch.setattr(
        venue_gates_module,
        "_resolve_shadow_book_and_tape",
        AsyncMock(return_value=(wide_book, [], 0.0, "test_wide_book", None)),
    )
    # Force deterministic rejection regardless of how the bps calc
    # interprets the book.
    monkeypatch.setattr(
        venue_gates_module,
        "_check_max_spread_bps",
        lambda *, book_payload, risk_limits: (True, 9999.0, 100.0),
    )

    leg = _leg(requested_notional_usd=15.0, requested_shares=30.0)
    result = await max_spread_bps_gate(
        _ctx(leg=leg, risk_limits={"max_spread_bps": 100.0})
    )

    assert result.passed is False
    assert result.reason == REASON_MAX_SPREAD_BPS_EXCEEDED
    assert "9999.0" in result.error_message
    assert "100.0" in result.error_message

    payload_extras = result.detail["payload_extras"]
    assert payload_extras["mode"] == "live"
    assert payload_extras["submission"] == "rejected"
    assert payload_extras["reason"] == REASON_MAX_SPREAD_BPS_EXCEEDED
    assert payload_extras["spread_bps"] == 9999.0
    assert payload_extras["max_spread_bps"] == 100.0
    # Spread rejection always sets effective_notional_usd=0.0 — the leg
    # was rejected before allocation, so nothing was effectively committed.
    assert payload_extras["effective_notional_usd"] == 0.0
    assert payload_extras["requested_notional_usd"] == 15.0
    assert payload_extras["preflight_rejected"] is True


@pytest.mark.asyncio
async def test_max_spread_bps_gate_passes_when_within_cap(monkeypatch):
    """Spread within cap ⇒ predicate passes."""
    monkeypatch.setattr(
        venue_gates_module,
        "_resolve_shadow_book_and_tape",
        AsyncMock(return_value=({"bids": [], "asks": []}, [], 0.0, "test", None)),
    )
    monkeypatch.setattr(
        venue_gates_module,
        "_check_max_spread_bps",
        lambda *, book_payload, risk_limits: (False, 50.0, 100.0),
    )
    result = await max_spread_bps_gate(
        _ctx(leg=_leg(), risk_limits={"max_spread_bps": 100.0})
    )
    assert result.passed is True


@pytest.mark.asyncio
async def test_max_spread_bps_gate_falls_back_to_pass_on_book_exception(monkeypatch):
    """Book resolution raises ⇒ predicate falls back to pass."""
    async def _boom(**kwargs):
        raise RuntimeError("book down")

    monkeypatch.setattr(
        venue_gates_module, "_resolve_shadow_book_and_tape", _boom
    )
    result = await max_spread_bps_gate(
        _ctx(leg=_leg(), risk_limits={"max_spread_bps": 100.0})
    )
    assert result.passed is True


# ---------------------------------------------------------------------------
# End-to-end via GatePipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_both_gates_pass(monkeypatch):
    """Pipeline with both venue gates passing ⇒ run.passed=True."""
    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(True, None)),
    )
    monkeypatch.setattr(
        venue_gates_module,
        "_resolve_shadow_book_and_tape",
        AsyncMock(return_value=(None, [], None, "test", None)),
    )
    pipeline = GatePipeline(
        [
            Gate(
                name=GATE_NAME_BUY_COLLATERAL,
                cost_class=CostClass.L1_CACHED,
                predicate=buy_collateral_gate,
            ),
            Gate(
                name=GATE_NAME_MAX_SPREAD_BPS,
                cost_class=CostClass.L1_CACHED,
                predicate=max_spread_bps_gate,
            ),
        ]
    )
    run = await pipeline.run(_ctx(leg=_leg()))
    assert run.passed is True
    assert len(run.per_gate) == 2


@pytest.mark.asyncio
async def test_pipeline_collateral_short_circuits_before_spread(monkeypatch):
    """Collateral fail ⇒ spread gate never runs."""
    monkeypatch.setattr(
        venue_gates_module.live_execution_service_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(False, "no money")),
    )
    spread_mock = AsyncMock(return_value=(None, [], None, "test", None))
    monkeypatch.setattr(venue_gates_module, "_resolve_shadow_book_and_tape", spread_mock)

    pipeline = GatePipeline(
        [
            Gate(
                name=GATE_NAME_BUY_COLLATERAL,
                cost_class=CostClass.L1_CACHED,
                predicate=buy_collateral_gate,
            ),
            Gate(
                name=GATE_NAME_MAX_SPREAD_BPS,
                cost_class=CostClass.L1_CACHED,
                predicate=max_spread_bps_gate,
            ),
        ]
    )
    run = await pipeline.run(_ctx(leg=_leg()))
    assert run.passed is False
    assert run.short_circuited_at == GATE_NAME_BUY_COLLATERAL
    assert len(run.per_gate) == 1
    # The book resolver was never called.
    spread_mock.assert_not_awaited()
