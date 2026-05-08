import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.live_execution_adapter import LiveOrderExecution  # noqa: E402
from services.trader_orchestrator import order_manager  # noqa: E402


@pytest.mark.asyncio
async def test_submit_execution_leg_live_uses_execution_adapter(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.41,
            error_message=None,
            payload={"order_id": "ord-123"},
            order_id="ord-123",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    signal = SimpleNamespace(
        id="sig-1",
        market_id="123456789012345678",
        direction="buy_yes",
        entry_price=0.40,
        market_question="Will X happen?",
        payload_json={"selected_token_id": "123456789012345678901"},
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": signal.entry_price,
        },
        notional_usd=41.0,
    )

    assert result.status == "open"
    assert result.effective_price == 0.41
    assert result.error_message is None
    assert result.payload["mode"] == "live"
    assert result.payload["shares"] == pytest.approx(102.5, rel=1e-6)
    assert result.payload["token_id_source"] == "payload.selected_token_id"
    execution_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_execution_leg_live_prefers_live_context_price_over_stale_leg_limit(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.88,
            error_message=None,
            payload={"order_id": "ord-live-price"},
            order_id="ord-live-price",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    signal = SimpleNamespace(
        id="sig-live-price",
        market_id="123456789012345678",
        direction="buy_yes",
        entry_price=0.57,
        market_question="Will price refresh happen?",
        payload_json={"selected_token_id": "123456789012345678901"},
        live_context={
            "selected_outcome": "yes",
            "selected_token_id": "123456789012345678901",
            "yes_token_id": "123456789012345678901",
            "no_token_id": "223456789012345678901",
            "live_selected_price": 0.88,
            "live_yes_price": 0.88,
            "live_no_price": 0.12,
        },
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": 0.57,
        },
        notional_usd=57.0,
    )

    assert result.status == "open"
    execution_mock.assert_awaited_once()
    submit_kwargs = execution_mock.await_args.kwargs
    assert submit_kwargs["fallback_price"] == pytest.approx(0.88, rel=1e-9)


@pytest.mark.asyncio
async def test_submit_execution_leg_live_taker_limit_caps_execution_to_dynamic_price_bound(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.95,
            error_message=None,
            payload={"order_id": "ord-bounded-taker"},
            order_id="ord-bounded-taker",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    signal = SimpleNamespace(
        id="sig-bounded-taker",
        market_id="123456789012345678",
        direction="buy_yes",
        entry_price=0.99,
        market_question="Will capped taker execution hold?",
        payload_json={"selected_token_id": "123456789012345678901"},
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": 0.99,
            "price_policy": "taker_limit",
        },
        notional_usd=99.0,
        strategy_params={
            "min_upside_percent": 5.0,
            "max_probability": 0.999,
        },
    )

    assert result.status == "open"
    execution_mock.assert_awaited_once()
    submit_kwargs = execution_mock.await_args.kwargs
    assert submit_kwargs["quote_aggressively"] is True
    assert submit_kwargs["enforce_fallback_bound"] is False
    assert submit_kwargs["max_execution_price"] == pytest.approx(100.0 / 105.0, rel=1e-9)


@pytest.mark.asyncio
async def test_submit_execution_leg_live_prefers_leg_level_execution_flags(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.91,
            error_message=None,
            payload={"order_id": "ord-leg-flags"},
            order_id="ord-leg-flags",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    signal = SimpleNamespace(
        id="sig-leg-flags",
        market_id="123456789012345678",
        direction="buy_yes",
        entry_price=0.90,
        market_question="Will leg flags override strategy defaults?",
        payload_json={"selected_token_id": "123456789012345678901"},
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": signal.entry_price,
            "price_policy": "taker_limit",
            "allow_taker_limit_buy_above_signal": True,
            "aggressive_limit_buy_submit_as_gtc": True,
            "max_execution_price": 0.94,
        },
        notional_usd=90.0,
        strategy_params={
            "allow_taker_limit_buy_above_signal": False,
            "aggressive_limit_buy_submit_as_gtc": False,
        },
    )

    assert result.status == "open"
    execution_mock.assert_awaited_once()
    submit_kwargs = execution_mock.await_args.kwargs
    assert submit_kwargs["allow_taker_limit_buy_above_signal"] is True
    assert submit_kwargs["aggressive_limit_buy_submit_as_gtc"] is True
    assert submit_kwargs["max_execution_price"] == pytest.approx(0.94)


@pytest.mark.asyncio
async def test_submit_execution_leg_live_resolves_outcome_specific_token_and_price(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.88,
            error_message=None,
            payload={"order_id": "ord-outcome-token"},
            order_id="ord-outcome-token",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    yes_token = "333333333333333333333"
    no_token = "444444444444444444444"
    signal = SimpleNamespace(
        id="sig-outcome-token",
        market_id="123456789012345678",
        direction="buy_no",
        entry_price=0.12,
        market_question="Will outcome mapping hold?",
        payload_json={
            "selected_token_id": no_token,
            "yes_token_id": yes_token,
            "no_token_id": no_token,
        },
        live_context={
            "selected_outcome": "no",
            "selected_token_id": no_token,
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "live_selected_price": 0.12,
            "live_yes_price": 0.88,
            "live_no_price": 0.12,
        },
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_yes",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": 0.12,
        },
        notional_usd=44.0,
    )

    assert result.status == "open"
    execution_mock.assert_awaited_once()
    submit_kwargs = execution_mock.await_args.kwargs
    assert submit_kwargs["token_id"] == yes_token
    assert submit_kwargs["fallback_price"] == pytest.approx(0.88, rel=1e-9)


@pytest.mark.asyncio
async def test_submit_execution_leg_live_ignores_mismatched_signal_live_context(monkeypatch):
    execution_mock = AsyncMock(
        return_value=LiveOrderExecution(
            status="open",
            effective_price=0.50,
            error_message=None,
            payload={"order_id": "ord-mismatched-live-context"},
            order_id="ord-mismatched-live-context",
        )
    )
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    signal = SimpleNamespace(
        id="sig-mismatched-live-context",
        market_id="market-draw",
        direction="buy_yes",
        entry_price=0.29,
        market_question="Will draw happen?",
        payload_json={"selected_token_id": "123456789012345678901"},
        live_context={
            "market_id": "market-draw",
            "selected_outcome": "yes",
            "selected_token_id": "123456789012345678901",
            "yes_token_id": "123456789012345678901",
            "no_token_id": "223456789012345678901",
            "token_ids": ["123456789012345678901", "223456789012345678901"],
            "live_selected_price": 0.29,
            "live_yes_price": 0.29,
            "live_no_price": 0.71,
        },
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg-home",
            "market_id": "market-home",
            "market_question": "Will home side win?",
            "token_id": "323456789012345678901",
            "side": "buy",
            "outcome": "yes",
            "limit_price": 0.50,
        },
        notional_usd=50.0,
    )

    assert result.status == "open"
    execution_mock.assert_awaited_once()
    submit_kwargs = execution_mock.await_args.kwargs
    assert submit_kwargs["token_id"] == "323456789012345678901"
    assert submit_kwargs["fallback_price"] == pytest.approx(0.50, rel=1e-9)


@pytest.mark.asyncio
async def test_submit_execution_leg_live_fails_without_token_id():
    signal = SimpleNamespace(
        id="sig-2",
        market_id="0x" + ("a" * 64),  # condition id, not directly executable token id
        direction="buy_yes",
        entry_price=0.40,
        market_question="Will Y happen?",
        payload_json={},
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": signal.entry_price,
        },
        notional_usd=40.0,
    )

    assert result.status == "failed"
    assert "token_id" in str(result.error_message or "")
    assert result.payload["reason"] == "missing_token_id"


@pytest.mark.asyncio
async def test_submit_execution_leg_live_does_not_fallback_to_market_id_token():
    signal = SimpleNamespace(
        id="sig-2b",
        market_id="123456789012345678901",
        direction="buy_yes",
        entry_price=0.40,
        market_question="Will Y2 happen?",
        payload_json={},
    )

    result = await order_manager.submit_execution_leg(
        mode="live",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": signal.entry_price,
        },
        notional_usd=40.0,
    )

    assert result.status == "failed"
    assert "token_id" in str(result.error_message or "")
    assert result.payload["reason"] == "missing_token_id"


@pytest.mark.asyncio
async def test_submit_execution_leg_shadow_skips_without_executable_book(monkeypatch):
    async def _no_token(*_args, **_kwargs):
        return None

    monkeypatch.setattr(order_manager, "_fetch_token_id_from_market", _no_token)
    signal = SimpleNamespace(
        id="sig-3",
        market_id="m3",
        direction="buy_no",
        entry_price=0.55,
        market_question="Will Z happen?",
        payload_json={},
    )

    result = await order_manager.submit_execution_leg(
        mode="shadow",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "no",
            "limit_price": signal.entry_price,
        },
        notional_usd=25.0,
    )

    assert result.status == "skipped"
    assert result.error_message == "No order book available for shadow execution leg."
    assert result.payload["submission"] == "skipped"
    assert result.payload["mode"] == "shadow"
    assert result.payload["reason"] == "missing_order_book"
    assert result.notional_usd == 0.0


@pytest.mark.asyncio
async def test_submit_execution_leg_shadow_uses_microstructure_context(monkeypatch):
    execution_mock = AsyncMock(side_effect=AssertionError("shadow mode must not place live orders"))
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    token_id = "123456789012345678901"
    signal = SimpleNamespace(
        id="sig-shadow-1",
        market_id="m-shadow-1",
        direction="buy_yes",
        entry_price=0.60,
        market_question="Will shadow quote execute?",
        payload_json={
            "selected_token_id": token_id,
            "live_market": {
                "execution_order_book": {
                    "bids": [{"price": 0.58, "size": 100.0}],
                    "asks": [{"price": 0.60, "size": 150.0}],
                },
                "execution_recent_trades": [
                    {"price": 0.60, "size": 50.0, "side": "BUY", "timestamp": 100.0},
                ],
                "execution_order_book_age_ms": 100.0,
            },
        },
    )

    result = await order_manager.submit_execution_leg(
        mode="shadow",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": signal.entry_price,
        },
        notional_usd=60.0,
    )

    assert result.status == "executed"
    assert result.effective_price == pytest.approx(0.60, rel=1e-9)
    assert result.error_message is None
    assert result.payload["mode"] == "shadow"
    assert result.payload["submission"] == "shadow_microstructure_simulated"
    assert result.payload["quote_source"] == "signal_microstructure_context"
    assert result.payload["token_id_source"] == "payload.selected_token_id"
    assert result.payload["shadow_simulation"]["execution_estimate"]["levels_consumed"] == 1
    execution_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_execution_leg_rejects_tiny_price_without_floor_inflation():
    signal = SimpleNamespace(
        id="sig-4",
        market_id="m4",
        direction="buy_yes",
        entry_price=0.00005,
        market_question="Will tiny price be rejected?",
        payload_json={},
    )

    result = await order_manager.submit_execution_leg(
        mode="shadow",
        signal=signal,
        leg={
            "leg_id": "leg_1",
            "market_id": signal.market_id,
            "market_question": signal.market_question,
            "side": "buy",
            "outcome": "yes",
            "limit_price": signal.entry_price,
        },
        notional_usd=25.0,
    )

    assert result.status == "failed"
    assert result.payload["reason"] == "invalid_price_too_small"
    assert result.shares is None


@pytest.mark.asyncio
async def test_submit_execution_wave_fails_fast_when_leg_submission_hangs(monkeypatch):
    async def _hung_leg(*args, **kwargs):
        await asyncio.sleep(0.05)
        raise AssertionError("submit_execution_leg should have timed out")

    monkeypatch.setattr(order_manager, "submit_execution_leg", _hung_leg)
    monkeypatch.setattr(order_manager, "_LEG_SUBMIT_TIMEOUT_SECONDS", 0.01)

    signal = SimpleNamespace(
        id="sig-timeout",
        market_id="m-timeout",
        direction="buy_yes",
        entry_price=0.45,
        market_question="Will timeout handling fail fast?",
        payload_json={},
    )

    results = await order_manager.submit_execution_wave(
        mode="live",
        signal=signal,
        legs_with_notionals=[
            (
                {
                    "leg_id": "leg_1",
                    "market_id": signal.market_id,
                    "market_question": signal.market_question,
                    "side": "buy",
                    "outcome": "yes",
                    "limit_price": signal.entry_price,
                },
                10.0,
            )
        ],
        strategy_params={},
    )

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_message == "Order submission timed out."


def test_allow_taker_limit_buy_above_signal_strategy_params_take_priority_over_risk_limits():
    """Strategy_params explicit setting wins over risk_limits."""
    assert order_manager._allow_taker_limit_buy_above_signal(
        {"allow_taker_limit_buy_above_signal": False},
        {"allow_taker_limit_buy_above_signal": True},
    ) is False
    assert order_manager._allow_taker_limit_buy_above_signal(
        {"allow_taker_limit_buy_above_signal": True},
        {"allow_taker_limit_buy_above_signal": False},
    ) is True


def test_allow_taker_limit_buy_above_signal_falls_back_to_risk_limits_when_unset():
    """When strategy_params has no policy key, risk_limits.allow_taker_limit_buy_above_signal wins."""
    assert order_manager._allow_taker_limit_buy_above_signal(
        {},
        {"allow_taker_limit_buy_above_signal": True},
    ) is True
    assert order_manager._allow_taker_limit_buy_above_signal(
        None,
        {"allow_taker_limit_buy_above_signal": True},
    ) is True


def test_allow_taker_limit_buy_above_signal_defaults_to_false_when_neither_set():
    assert order_manager._allow_taker_limit_buy_above_signal({}, {}) is False
    assert order_manager._allow_taker_limit_buy_above_signal(None, None) is False


def test_aggressive_limit_buy_submit_as_gtc_falls_back_to_risk_limits_when_unset():
    assert order_manager._aggressive_limit_buy_submit_as_gtc(
        {},
        {"aggressive_limit_buy_submit_as_gtc": True},
    ) is True
    assert order_manager._aggressive_limit_buy_submit_as_gtc({}, {}) is False


@pytest.mark.asyncio
async def test_shadow_buy_with_chase_up_lifts_simulator_limit_so_asks_above_mid_fill(monkeypatch):
    """Without chase-up, mid=0.55 vs ask=0.60 → simulator break → no fill.

    With risk_limits.allow_taker_limit_buy_above_signal=True the simulator
    receives an effective ceiling at max_execution_price (or 1.0 when no
    cap), so it walks asks above mid and produces a real fill — which is
    the whole point of the toggle in shadow mode.
    """
    execution_mock = AsyncMock(side_effect=AssertionError("shadow must not call live broker"))
    monkeypatch.setattr(order_manager, "execute_live_order", execution_mock)

    token_id = "999988887777666655554"
    signal = SimpleNamespace(
        id="sig-chase-shadow",
        market_id="m-chase-shadow",
        direction="buy_yes",
        entry_price=0.55,
        market_question="Will chase-up shadow fill?",
        payload_json={
            "selected_token_id": token_id,
            "live_market": {
                "live_selected_price": 0.55,
                "execution_order_book": {
                    "bids": [{"price": 0.54, "size": 100.0}],
                    "asks": [{"price": 0.60, "size": 200.0}],
                },
                "execution_recent_trades": [
                    {"price": 0.60, "size": 50.0, "side": "BUY", "timestamp": 100.0},
                ],
                "execution_order_book_age_ms": 100.0,
            },
        },
    )

    leg = {
        "leg_id": "leg_chase",
        "market_id": signal.market_id,
        "market_question": signal.market_question,
        "side": "buy",
        "outcome": "yes",
        "limit_price": 0.55,
        "price_policy": "taker_limit",
    }

    no_chase_result = await order_manager.submit_execution_leg(
        mode="shadow",
        signal=signal,
        leg=leg,
        notional_usd=10.0,
        strategy_params={},
        risk_limits={"allow_taker_limit_buy_above_signal": False},
    )
    assert no_chase_result.payload["reason"] == "limit_price_not_executable", (
        "Without chase-up the simulator must reject mid<ask as before; otherwise the "
        "test fixture has drifted and no longer exercises the bug we are fixing."
    )

    chase_result = await order_manager.submit_execution_leg(
        mode="shadow",
        signal=signal,
        leg=dict(leg),
        notional_usd=10.0,
        strategy_params={},
        risk_limits={"allow_taker_limit_buy_above_signal": True},
    )
    assert chase_result.status == "executed", (
        f"chase-up should produce a fill; got status={chase_result.status} "
        f"reason={chase_result.payload.get('reason')!r}"
    )
    assert chase_result.effective_price == pytest.approx(0.60, rel=1e-9)
    assert chase_result.payload["submission"] == "shadow_microstructure_simulated"
    assert chase_result.payload["shadow_simulation"]["execution_estimate"]["levels_consumed"] >= 1
