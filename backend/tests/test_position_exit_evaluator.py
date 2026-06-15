"""Parity tests for ``evaluate_position_exit`` (the shared exit decision).

This function was extracted line-for-line from the inline decision block in
``reconcile_live_positions`` so the slow reconcile and the fast exit-risk
loop share one source of truth. These tests pin the default SL/TP/trailing/
max-hold gates, the strategy close/reduce/hold paths, and the realizable
liquidation-mark override — deterministically (no live double-call of the
state-mutating ``should_exit``).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.optimization.vwap import OrderBook, OrderBookLevel
from services.trader_orchestrator import position_lifecycle as pl


def _row(*, entry_price=0.90, age_minutes=120.0):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id="ord-test",
        executed_at=now - timedelta(minutes=age_minutes),
        updated_at=now,
        created_at=now - timedelta(minutes=age_minutes),
        status="executed",
        direction="buy_no",
        market_id="mkt-1",
    )


async def _no_strategy(session, slug):  # noqa: ARG001
    return None


async def _eval(monkeypatch, *, ws_price, book=None, payload=None, entry_price=0.90,
                age_minutes=120.0, stop_loss_pct=None, take_profit_pct=None,
                trailing_stop_pct=None, max_hold_minutes=None, min_hold_minutes=0.0,
                market_tradable=True, strategy_instance=None, position_state=None):
    async def _fake_instance(session, slug):  # noqa: ARG001
        return strategy_instance
    monkeypatch.setattr(pl, "_strategy_exit_instance", _fake_instance)
    pay = {"strategy_type": "tail_end_carry", "strategy_exit_config": {}}
    if position_state is not None:
        pay["position_state"] = position_state
    if payload:
        pay.update(payload)
    now = datetime.now(timezone.utc)
    return await pl.evaluate_position_exit(
        session=None,
        row=_row(entry_price=entry_price, age_minutes=age_minutes),
        payload=pay,
        now=now,
        now_naive=now.replace(tzinfo=None),
        ws_side_price=ws_price,
        clob_side_price=None,
        market_side_price=None,
        wallet_mark_price=None,
        book=book,
        entry_price=entry_price,
        notional=10.0,
        filled_size=11.0,
        wallet_position_size=11.0,
        outcome_idx=0,
        market_info={"condition_id": "mkt-1"},
        market_tradable=market_tradable,
        market_seconds_left=3600.0,
        market_end_time=None,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        max_hold_minutes=max_hold_minutes,
        min_hold_minutes=min_hold_minutes,
        resolve_only=False,
        close_on_inactive_market=False,
        pending_exit=None,
        params={},
        mark_touch_interval_seconds=10.0,
    )


def _book(bids):
    return OrderBook(
        bids=[OrderBookLevel(price=p, size=s) for p, s in bids],
        asks=[OrderBookLevel(price=0.99, size=100.0)],
    )


@pytest.mark.asyncio
async def test_healthy_price_holds(monkeypatch):
    d = await _eval(monkeypatch, ws_price=0.91, entry_price=0.90, stop_loss_pct=50.0)
    assert d.action == "hold"
    assert d.current_price == 0.91


@pytest.mark.asyncio
async def test_stop_loss_fires(monkeypatch):
    # entry 0.90, price 0.40 -> pnl_pct ~ -55% <= -50%
    d = await _eval(monkeypatch, ws_price=0.40, entry_price=0.90, stop_loss_pct=50.0)
    assert d.action == "close"
    assert d.close_trigger == "stop_loss"
    assert d.close_price == 0.40


@pytest.mark.asyncio
async def test_take_profit_fires(monkeypatch):
    # buy at 0.50, price 0.60 -> +20% >= 10%
    d = await _eval(monkeypatch, ws_price=0.60, entry_price=0.50, take_profit_pct=10.0)
    assert d.action == "close"
    assert d.close_trigger == "take_profit"


@pytest.mark.asyncio
async def test_trailing_stop_fires(monkeypatch):
    # highest seeded to 0.95 via position_state; price 0.80; trail 12% -> trigger 0.836
    d = await _eval(
        monkeypatch, ws_price=0.80, entry_price=0.90, trailing_stop_pct=12.0,
        position_state={"highest_price": 0.95, "lowest_price": 0.90, "last_mark_price": 0.90},
    )
    assert d.action == "close"
    assert d.close_trigger == "trailing_stop"
    assert abs(d.trailing_trigger_price - 0.95 * 0.88) < 1e-9


@pytest.mark.asyncio
async def test_max_hold_fires(monkeypatch):
    d = await _eval(monkeypatch, ws_price=0.90, entry_price=0.90, max_hold_minutes=60.0, age_minutes=120.0)
    assert d.action == "close"
    assert d.close_trigger == "max_hold"


def test_market_seconds_left_recomputes_past_end_time_over_stale_cache():
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)
    market_info = {
        "seconds_left": 86400,
        "end_date": datetime(2026, 6, 14, tzinfo=timezone.utc).isoformat(),
    }

    assert pl._market_seconds_left(market_info, now) == 0.0


def test_live_exit_tradability_can_override_past_end_date_with_live_book():
    market_info = {
        "closed": False,
        "active": True,
        "resolved": False,
        "accepting_orders": True,
        "enable_order_book": True,
        "end_date": datetime(2026, 6, 14, tzinfo=timezone.utc).isoformat(),
    }

    assert (
        pl._market_accepts_live_exit_orders(
            market_info,
            base_market_tradable=False,
            has_live_exit_mark=True,
        )
        is True
    )


def test_live_exit_tradability_does_not_override_terminal_market():
    market_info = {
        "closed": True,
        "active": True,
        "resolved": False,
        "accepting_orders": True,
        "enable_order_book": True,
    }

    assert (
        pl._market_accepts_live_exit_orders(
            market_info,
            base_market_tradable=False,
            has_live_exit_mark=True,
        )
        is False
    )


@pytest.mark.asyncio
async def test_liquidation_mark_triggers_stop_mid_would_miss(monkeypatch):
    # WS mid says 0.90 (healthy vs 50% SL), but the realizable bid-side VWAP
    # for 11 shares is ~0.40 (thin book) -> stop SHOULD fire on the liq mark.
    book = _book([(0.40, 100.0)])
    d = await _eval(monkeypatch, ws_price=0.90, book=book, entry_price=0.90, stop_loss_pct=50.0)
    assert d.current_price == 0.40
    assert d.current_price_source == "liquidation_vwap"
    assert d.action == "close" and d.close_trigger == "stop_loss"


@pytest.mark.asyncio
async def test_strategy_reduce(monkeypatch):
    inst = SimpleNamespace(
        should_exit=lambda pos, mkt: SimpleNamespace(action="reduce", reason="scale_out", close_price=0.85, reduce_fraction=0.5),
    )
    d = await _eval(monkeypatch, ws_price=0.85, entry_price=0.90, strategy_instance=inst)
    assert d.action == "reduce"
    assert d.strategy_exit is not None and d.strategy_exit.reduce_fraction == 0.5


@pytest.mark.asyncio
async def test_strategy_close(monkeypatch):
    monkeypatch.setattr(pl, "_arm_reverse_entry_from_exit", lambda **k: None)
    inst = SimpleNamespace(
        should_exit=lambda pos, mkt: SimpleNamespace(action="close", reason="Inversion stop", close_price=0.30, reduce_fraction=None),
    )
    d = await _eval(monkeypatch, ws_price=0.30, entry_price=0.90, strategy_instance=inst)
    assert d.action == "close"
    assert d.close_trigger == "strategy:Inversion stop"
