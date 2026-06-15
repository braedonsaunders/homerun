"""Orchestration tests for the isolated exit-risk loop.

The shared decision/submit logic is covered by test_position_exit_evaluator
and test_execute_position_exit. Here we pin the loop's per-position
orchestration: the in-flight mutex (skip positions already exiting) and the
dispatch from a close decision to execute_position_exit.
"""

from __future__ import annotations

import asyncio
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
        executed_at=now, created_at=now, source="strategy", reason=None,
        actual_profit=None,
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

    # Default: no prior exit fills (so the ghost-terminalization fast-path is
    # inert). Tests that exercise terminalization override this.
    async def _no_fills(session, *, token_id, since, wallet_address=None):  # noqa: ARG001
        return (0.0, 0.0)
    monkeypatch.setattr(pl, "summarize_live_exit_fills", _no_fills)

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
async def test_fire_cooldown_suppresses_rapid_resubmit(monkeypatch):
    # Regression: reconcile can pop/supersede pending_live_exit (wallet-flat /
    # terminalization) while a phantom is still selectable, so the mutex can't
    # always catch a re-fire. The per-order cooldown must suppress a second
    # submit for the SAME order within the window even with no pending_live_exit.
    _patch_derivations(monkeypatch, decision=_close_decision())
    calls = []

    async def _fake_exec(**kwargs):
        calls.append(kwargs)
        return {"status": "submitted", "submitted": True}
    monkeypatch.setattr(pl, "execute_position_exit", _fake_exec)

    loop = erl.ExitRiskLoop()
    payload = {"token_id": "tok-1", "strategy_type": "tail_end_carry"}
    # Two back-to-back sweeps with NO pending_live_exit set (simulating
    # reconcile having cleared it between cycles).
    await _process(loop, monkeypatch, dict(payload))
    await _process(loop, monkeypatch, dict(payload))
    assert len(calls) == 1  # second fire suppressed by cooldown

    # After the cooldown elapses, a re-fire is allowed again.
    loop._last_fire_at["ord-1"] -= (erl._FIRE_COOLDOWN_SECONDS + 1.0)
    await _process(loop, monkeypatch, dict(payload))
    assert len(calls) == 2


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
@pytest.mark.parametrize(
    "pending_status",
    ["submitted", "failed", "blocked_min_notional", "blocked_retry_exhausted"],
)
async def test_any_pending_exit_is_skipped(monkeypatch, pending_status):
    # Regression: a phantom zero-share position re-fired market_inactive every
    # 2s sweep because the loop only skipped the in-flight subset of states.
    # The loop now fires only the FIRST exit; once any pending_live_exit exists
    # (including failed/blocked retry states), the retry lifecycle is owned by
    # reconcile, so the loop must skip — neither evaluate nor execute may run.
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
        "pending_live_exit": {"status": pending_status},
    })
    assert calls == [] and eval_calls == []


@pytest.mark.asyncio
async def test_ghost_position_terminalized_when_fully_sold(monkeypatch):
    # The fast loop sold the whole position but starved reconcile never closed
    # the row (the 2026-05 ghost). When the wallet is flat AND our own exit
    # fills account for the full entry size, the loop must terminalize via the
    # shared helper — not evaluate/fire another doomed exit.
    _patch_derivations(monkeypatch, decision=_close_decision())

    # Entry was 11.06 shares @ 0.911 ($10.08). Our exit fills sold all 11.06
    # for $8.47 proceeds (~16% loss), matching the real incident.
    monkeypatch.setattr(pl, "_extract_live_fill_metrics", lambda p: (10.078, 11.06, 0.911))
    monkeypatch.setattr(pl, "_extract_wallet_position_size", lambda wp: 0.0)  # wallet flat

    async def _fills(session, *, token_id, since, wallet_address=None):  # noqa: ARG001
        return (11.06, 8.47)
    monkeypatch.setattr(pl, "summarize_live_exit_fills", _fills)

    captured = {}

    async def _fake_terminalize(**kwargs):
        captured.update(kwargs)
        kwargs["row"].status = "closed_loss"
        return "closed_loss"
    monkeypatch.setattr(pl, "terminalize_filled_exit", _fake_terminalize)

    eval_calls = []

    async def _spy_eval(**kwargs):  # noqa: ARG001
        eval_calls.append(1)
        return _close_decision()
    monkeypatch.setattr(pl, "evaluate_position_exit", _spy_eval)

    exec_calls = []

    async def _fake_exec(**kwargs):
        exec_calls.append(kwargs)
        return {}
    monkeypatch.setattr(pl, "execute_position_exit", _fake_exec)

    loop = erl.ExitRiskLoop()
    # wallet flat (wallet_by_token empty) + full sell fills -> terminalize
    await _process(loop, monkeypatch, {"token_id": "tok-1", "strategy_type": "tail_end_carry"})

    assert captured, "terminalize_filled_exit must be called"
    assert eval_calls == [] and exec_calls == []  # no eval/fire once fully sold
    assert abs(captured["realized_pnl"] - (8.47 - 10.078)) < 1e-6
    assert abs(captured["close_price"] - (8.47 / 11.06)) < 1e-6


@pytest.mark.asyncio
async def test_partial_sell_does_not_terminalize(monkeypatch):
    # Only part of the position sold -> NOT flat -> must keep managing it,
    # never terminalize on a partial exit.
    _patch_derivations(monkeypatch, decision=_close_decision())
    monkeypatch.setattr(pl, "_extract_live_fill_metrics", lambda p: (10.078, 11.06, 0.911))
    monkeypatch.setattr(pl, "_extract_wallet_position_size", lambda wp: 0.0)  # wallet flat

    async def _fills(session, *, token_id, since, wallet_address=None):  # noqa: ARG001
        return (4.58, 3.47)  # only ~41% sold
    monkeypatch.setattr(pl, "summarize_live_exit_fills", _fills)

    term_calls = []

    async def _fake_terminalize(**kwargs):
        term_calls.append(kwargs)
        return "closed_loss"
    monkeypatch.setattr(pl, "terminalize_filled_exit", _fake_terminalize)

    async def _fake_exec(**kwargs):
        return {"status": "submitted", "submitted": True}
    monkeypatch.setattr(pl, "execute_position_exit", _fake_exec)

    loop = erl.ExitRiskLoop()
    await _process(loop, monkeypatch, {"token_id": "tok-1", "strategy_type": "tail_end_carry"})
    assert term_calls == []  # partial fill must not terminalize


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


@pytest.mark.parametrize(
    "exc,expected",
    [
        (TimeoutError("x"), True),
        (asyncio.TimeoutError(), True),
        (ConnectionError("connection is closed"), True),
        (RuntimeError("another operation is in progress"), True),
        (ValueError("bad value"), False),
        (KeyError("nope"), False),
    ],
)
def test_is_transient_db_error(exc, expected):
    assert erl._is_transient_db_error(exc) is expected


@pytest.mark.asyncio
async def test_sweep_retries_once_on_transient_error(monkeypatch):
    # A connection that dies mid-sweep must trigger ONE immediate fresh-session
    # retry rather than dropping the whole 2s cycle (risk-loop resilience).
    loop = erl.ExitRiskLoop()
    monkeypatch.setattr(loop, "_register_callback", lambda: None)
    async def _no_snapshot(**kwargs):  # noqa: ARG001
        return None
    monkeypatch.setattr(loop, "_write_snapshot", _no_snapshot)
    loop._wake.set()  # make the interval wait return immediately
    calls = []

    async def _flaky():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("connection is closed")  # transient -> retry
        loop._running = False  # retry succeeds, stop the loop

    monkeypatch.setattr(loop, "_sweep", _flaky)
    await asyncio.wait_for(loop.run_forever(), timeout=5)
    assert len(calls) == 2  # original + one retry


@pytest.mark.asyncio
async def test_sweep_not_retried_on_nontransient_error(monkeypatch):
    # A non-transient error must NOT retry within the cycle (avoid masking a
    # real outage); it falls through to the next tick.
    loop = erl.ExitRiskLoop()
    monkeypatch.setattr(loop, "_register_callback", lambda: None)
    async def _no_snapshot(**kwargs):  # noqa: ARG001
        return None
    monkeypatch.setattr(loop, "_write_snapshot", _no_snapshot)
    loop._wake.set()
    calls = []

    async def _boom():
        calls.append(1)
        loop._running = False  # stop after this one call
        raise ValueError("boom")  # non-transient -> no retry

    monkeypatch.setattr(loop, "_sweep", _boom)
    await asyncio.wait_for(loop.run_forever(), timeout=5)
    assert len(calls) == 1  # no retry


@pytest.mark.asyncio
async def test_telemetry_records_decision(monkeypatch):
    # Every sweep must stamp last_exit_decision so a no-fire on a losing
    # position is visible in the DB (no log archaeology).
    hold = _close_decision()
    hold.action = "hold"
    hold.close_price = None
    hold.close_trigger = None
    hold.current_price_source = "liquidation_vwap"  # fresh
    _patch_derivations(monkeypatch, decision=hold)
    loop = erl.ExitRiskLoop()
    row = _row({"token_id": "tok-1", "strategy_type": "tail_end_carry"})
    now = datetime.now(timezone.utc)
    await loop._process_position(
        session=_FakeSession(), row=row, now=now, now_naive=now.replace(tzinfo=None),
        ws_mid={"tok-1": 0.40}, clob_mid={}, books={},
        market_info_by_id={"m1": {"condition_id": "m1"}},
        wallet_by_token={}, params={}, submissions=[0],
    )
    led = row.payload_json["position_state"]["last_exit_decision"]
    assert led["action"] == "hold"
    assert led["mark_source"] == "liquidation_vwap"
    assert led["mark_stale"] is False
    assert led["mark"] == 0.40


@pytest.mark.asyncio
async def test_past_end_date_can_still_be_live_exit_tradable(monkeypatch):
    hold = _close_decision()
    hold.action = "hold"
    hold.close_price = None
    hold.close_trigger = None
    _patch_derivations(monkeypatch, decision=hold)
    monkeypatch.setattr(pl.polymarket_client, "is_market_tradable", lambda mi, now=None: False)
    seen = {}

    async def _spy_eval(**kwargs):
        seen["market_tradable"] = kwargs["market_tradable"]
        return hold

    monkeypatch.setattr(pl, "evaluate_position_exit", _spy_eval)
    loop = erl.ExitRiskLoop()
    row = _row({"token_id": "tok-1", "strategy_type": "tail_end_carry"})
    now = datetime.now(timezone.utc)
    await loop._process_position(
        session=_FakeSession(), row=row, now=now, now_naive=now.replace(tzinfo=None),
        ws_mid={}, clob_mid={"tok-1": 0.40}, books={},
        market_info_by_id={
            "m1": {
                "condition_id": "m1",
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "enable_order_book": True,
            }
        },
        wallet_by_token={}, params={}, submissions=[0],
    )

    assert seen["market_tradable"] is True


@pytest.mark.asyncio
async def test_failsafe_forces_fresh_clob_on_stale_hold(monkeypatch):
    # A tradable HOLD on a non-fresh mark must force a direct CLOB read and
    # re-evaluate (the London 0.94->0 fix).
    hold = _close_decision()
    hold.action = "hold"
    hold.close_price = None
    hold.close_trigger = None
    hold.current_price_source = "market_mark"  # STALE/weak source
    _patch_derivations(monkeypatch, decision=hold)
    loop = erl.ExitRiskLoop()
    called = []
    async def _fresh(tok):
        called.append(tok)
        return 0.40
    monkeypatch.setattr(loop, "_force_fresh_clob_mid", _fresh)
    row = _row({"token_id": "tok-1", "strategy_type": "tail_end_carry"})
    now = datetime.now(timezone.utc)
    await loop._process_position(
        session=_FakeSession(), row=row, now=now, now_naive=now.replace(tzinfo=None),
        ws_mid={"tok-1": 0.40}, clob_mid={}, books={},
        market_info_by_id={"m1": {"condition_id": "m1"}},
        wallet_by_token={}, params={}, submissions=[0],
    )
    assert called == ["tok-1"]  # forced a fresh CLOB read on stale-hold
    assert row.payload_json["position_state"]["last_exit_decision"]["mark_stale"] is True
