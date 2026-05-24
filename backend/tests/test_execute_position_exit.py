"""Branch tests for ``execute_position_exit`` (shared exit submit path).

Extracted verbatim from reconcile's inline submit orchestration so the
slow reconcile and the fast exit-risk loop place exits identically. These
pin the simple ``execute_live_order`` path, the min-notional block, and the
missing-token failure — venue primitives mocked.
"""

from __future__ import annotations

import contextlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import services.live_execution_adapter as lea
from services.trader_orchestrator import position_lifecycle as pl


def _common_mocks(monkeypatch, *, min_order_size=5.0):
    @contextlib.asynccontextmanager
    async def _noop_release(_session):
        yield

    async def _noop_prepare(_token):  # noqa: ARG001
        return None

    monkeypatch.setattr(pl, "release_conn", _noop_release)
    monkeypatch.setattr(pl, "_prepare_sell_allowance_bounded", _noop_prepare)
    monkeypatch.setattr(pl, "_resolve_position_min_order_size_usd", lambda **k: min_order_size)
    monkeypatch.setattr(pl, "_effective_exit_min_order_size_usd", lambda base, trig: base)


def _row():
    now = datetime.now(timezone.utc)
    return SimpleNamespace(id="ord-1", status="executed", direction="buy_no", market_id="m1",
                           executed_at=now, updated_at=now, created_at=now, mode="live", source="scanner")


async def _run(monkeypatch, *, payload, close_price=0.40, filled_size=11.0, close_trigger="stop_loss"):
    return await pl.execute_position_exit(
        session=None, row=_row(), payload=payload, now=datetime.now(timezone.utc),
        close_price=close_price, close_trigger=close_trigger, price_source="ws_mid",
        pnl=-5.0, market_tradable=True, age_minutes=120.0,
        filled_size=filled_size, quantity=filled_size, wallet_position_size=filled_size,
        pending_exit=None, exit_instance=None, strategy_exit=None, params={}, reason="lifecycle",
        submissions_this_pass=[0],
    )


@pytest.mark.asyncio
async def test_simple_submit_success(monkeypatch):
    _common_mocks(monkeypatch)

    async def _fake_order(**kwargs):
        assert kwargs["side"] == "SELL"
        assert kwargs["token_id"] == "tok-1"
        return SimpleNamespace(status="submitted", order_id="exit-123", error_message=None,
                               payload={"clob_order_id": "0xabc"})

    monkeypatch.setattr(lea, "execute_live_order", _fake_order)
    payload = {"token_id": "tok-1", "strategy_type": "tail_end_carry"}
    res = await _run(monkeypatch, payload=payload, close_price=0.90)  # 11*0.90=9.9 >= 5 min
    assert res["status"] == "submitted" and res["submitted"] is True
    assert payload["pending_live_exit"]["exit_order_id"] == "exit-123"
    assert payload["pending_live_exit"]["provider_clob_order_id"] == "0xabc"


@pytest.mark.asyncio
async def test_blocked_min_notional(monkeypatch):
    _common_mocks(monkeypatch, min_order_size=5.0)
    # 11 shares * 0.001 = 0.011 << 5.0 -> blocked
    payload = {"token_id": "tok-1", "strategy_type": "tail_end_carry"}
    res = await _run(monkeypatch, payload=payload, close_price=0.001)
    assert res["status"] == "blocked_min_notional" and res["submitted"] is False
    assert payload["pending_live_exit"]["status"] == "blocked_min_notional"


@pytest.mark.asyncio
async def test_missing_token_fails(monkeypatch):
    _common_mocks(monkeypatch)
    payload = {"strategy_type": "tail_end_carry"}  # no token_id
    res = await _run(monkeypatch, payload=payload, close_price=0.40)
    assert res["status"] == "failed" and res["submitted"] is False
    assert "missing token_id" in payload["pending_live_exit"]["last_error"]


@pytest.mark.asyncio
async def test_per_pass_cap_defers(monkeypatch):
    _common_mocks(monkeypatch)
    monkeypatch.setattr(pl, "_live_exit_submission_cap", lambda: 1)
    called = {"n": 0}

    async def _fake_order(**kwargs):  # noqa: ARG001
        called["n"] += 1
        return SimpleNamespace(status="submitted", order_id="x", error_message=None, payload={})

    monkeypatch.setattr(lea, "execute_live_order", _fake_order)
    payload = {"token_id": "tok-1", "strategy_type": "tail_end_carry"}
    res = await pl.execute_position_exit(
        session=None, row=_row(), payload=payload, now=datetime.now(timezone.utc),
        close_price=0.90, close_trigger="stop_loss", price_source="ws_mid", pnl=-5.0,
        market_tradable=True, age_minutes=120.0, filled_size=11.0, quantity=11.0,
        wallet_position_size=11.0, pending_exit=None, exit_instance=None, strategy_exit=None,
        params={}, reason="lifecycle", submissions_this_pass=[1],  # already at cap
    )
    assert res["status"] == "failed" and called["n"] == 0  # deferred, no order placed
    assert payload["pending_live_exit"]["last_error"] == "deferred_per_pass_cap"
