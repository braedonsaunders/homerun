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
async def test_submit_execution_leg_paper_still_simulates_execution():
    signal = SimpleNamespace(
        id="sig-3",
        market_id="m3",
        direction="buy_no",
        entry_price=0.55,
        market_question="Will Z happen?",
        payload_json={},
    )

    result = await order_manager.submit_execution_leg(
        mode="paper",
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

    assert result.status == "executed"
    assert result.effective_price == 0.55
    assert result.error_message is None
    assert result.payload["submission"] == "simulated"


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
        mode="paper",
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
