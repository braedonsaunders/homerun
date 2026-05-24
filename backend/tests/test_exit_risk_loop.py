"""Orchestration tests for the isolated exit-risk loop.

The shared decision/submit logic is covered by test_position_exit_evaluator
and test_execute_position_exit. Here we pin the loop's per-position
orchestration: the in-flight mutex (skip positions already exiting) and the
dispatch from a close decision to execute_position_exit.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator import exit_risk_loop as erl
from services.trader_orchestrator import position_lifecycle as pl


class _FakeSession:
    async def commit(self):
        return None


def _row(payload):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id="ord-1", trader_id="trader-1", mode="live", status="executed",
        direction="buy_no", market_id="m1", effective_price=0.90, entry_price=0.90,
        notional_usd=10.0, payload_json=payload, updated_at=now,
    )


def _patch_derivations(monkeypatch, *, decision):
    monkeypatch.setattr(pl, "_extract_live_fill_metrics", lambda p: (10.0, 11.0, 0.90))
    monkeypatch.setattr(pl, "_direction_outcome_index", lambda d, market_info=None, token_id=None: 0)
    monkeypatch.setattr(pl, "_extract_market_side_price", lambda mi, idx: 0.40)
    monkeypatch.setattr(pl, "_market_seconds_left", lambda mi, now: 3600.0)
    monkeypatch.setattr(pl, "_market_end_time_iso", lambda mi: None)
    monkeypatch.setattr(pl, "_extract_wallet_position_size", lambda wp: 11.0)
    monkeypatch.setattr(pl, "_extract_wallet_mark_price", lambda wp: 0.40)
    monkeypatch.setattr(pl, "_payload_exit_param", lambda payload, prefix_key, name: None)
    monkeypatch.setattr(pl, "_safe_bool", lambda v, d=False: d)
    monkeypatch.setattr(pl.polymarket_client, "is_market_tradable", lambda mi, now=None: True)

    async def _no_instance(session, slug):  # noqa: ARG001
        return None
    monkeypatch.setattr(pl, "_strategy_exit_instance", _no_instance)

    async def _fake_eval(**kwargs):  # noqa: ARG001
        return decision
    monkeypatch.setattr(pl, "evaluate_position_exit", _fake_eval)


def _close_decision():
    return pl.PositionExitDecision(
        current_price=0.40, current_price_source="liquidation_vwap",
        highest_price=0.91, lowest_price=0.40,
        next_state={"last_mark_price": 0.40}, age_minutes=120.0, min_hold_passed=True,
        pnl_pct=-55.0, action="close", close_price=0.40, close_trigger="stop_loss",
        price_source="liquidation_vwap", strategy_exit=None,
    )


async def _process(loop, monkeypatch, payload):
    now = datetime.now(timezone.utc)
    await loop._process_position(
        session=_FakeSession(), row=_row(payload), now=now,
        now_naive=now.replace(tzinfo=None),
        ws_mid={"tok-1": 0.40}, clob_mid={}, books={},
        market_info_by_id={"m1": {"condition_id": "m1"}},
        wallet_by_token={}, params={}, submissions=[0],
    )


@pytest.mark.asyncio
async def test_close_decision_dispatches_execute(monkeypatch):
    _patch_derivations(monkeypatch, decision=_close_decision())
    calls = []

    async def _fake_exec(**kwargs):
        calls.append(kwargs)
        return {"status": "submitted", "submitted": True}
    monkeypatch.setattr(pl, "execute_position_exit", _fake_exec)

    loop = erl.ExitRiskLoop()
    await _process(loop, monkeypatch, {"token_id": "tok-1", "strategy_type": "tail_end_carry"})

    assert len(calls) == 1
    assert calls[0]["close_trigger"] == "stop_loss"
    assert calls[0]["close_price"] == 0.40
    assert calls[0]["reason"] == "exit_risk_loop"


@pytest.mark.asyncio
async def test_inflight_pending_exit_is_skipped(monkeypatch):
    _patch_derivations(monkeypatch, decision=_close_decision())
    calls = []

    async def _fake_exec(**kwargs):
        calls.append(kwargs)
        return {}
    monkeypatch.setattr(pl, "execute_position_exit", _fake_exec)

    eval_calls = []

    async def _spy_eval(**kwargs):  # noqa: ARG001
        eval_calls.append(1)
        return _close_decision()
    monkeypatch.setattr(pl, "evaluate_position_exit", _spy_eval)

    loop = erl.ExitRiskLoop()
    await _process(loop, monkeypatch, {
        "token_id": "tok-1", "strategy_type": "tail_end_carry",
        "pending_live_exit": {"status": "submitted"},
    })
    # mutex: neither evaluate nor execute should run for an in-flight exit
    assert calls == [] and eval_calls == []


@pytest.mark.asyncio
async def test_hold_decision_no_execute(monkeypatch):
    hold = _close_decision()
    hold.action = "hold"
    hold.close_price = None
    hold.close_trigger = None
    _patch_derivations(monkeypatch, decision=hold)
    calls = []

    async def _fake_exec(**kwargs):
        calls.append(kwargs)
        return {}
    monkeypatch.setattr(pl, "execute_position_exit", _fake_exec)

    loop = erl.ExitRiskLoop()
    await _process(loop, monkeypatch, {"token_id": "tok-1", "strategy_type": "tail_end_carry"})
    assert calls == []
