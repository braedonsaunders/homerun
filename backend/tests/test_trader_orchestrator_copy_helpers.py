import sys
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator_state import (
    _extract_copy_side_from_payload,
    _extract_copy_source_wallet_from_payload,
    _serialize_order,
)
from utils.utcnow import utcnow


def _build_order(payload: dict) -> SimpleNamespace:
    now = utcnow().astimezone(timezone.utc)
    return SimpleNamespace(
        id="order-1",
        trader_id="trader-1",
        signal_id="signal-1",
        decision_id="decision-1",
        source="traders",
        strategy_key="traders_copy_trade",
        strategy_version=1,
        market_id="0x" + "1" * 64,
        market_question="Will this resolve?",
        direction="buy_yes",
        mode="live",
        status="open",
        notional_usd=100.0,
        entry_price=0.5,
        effective_price=0.52,
        edge_percent=1.2,
        confidence=0.7,
        actual_profit=None,
        reason=None,
        payload_json=payload,
        error_message=None,
        event_id=None,
        trace_id=None,
        created_at=now,
        executed_at=now,
        updated_at=now,
    )


def test_extract_copy_source_wallet_from_payload_uses_copy_event_first():
    payload = {
        "source_trade": {"wallet_address": "0xbbb"},
        "strategy_context": {"copy_event": {"wallet_address": "0xaaa"}},
    }
    assert _extract_copy_source_wallet_from_payload(payload) == "0xaaa"


def test_extract_copy_side_from_payload_falls_back_to_source_trade():
    payload = {
        "source_trade": {"side": "sell"},
    }
    assert _extract_copy_side_from_payload(payload) == "SELL"


def test_serialize_order_populates_copy_attribution_slippage():
    detected_at = utcnow().isoformat()
    row = _build_order(
        {
            "source_trade": {
                "wallet_address": "0xabc",
                "side": "BUY",
                "price": 0.5,
                "detected_at": detected_at,
            },
            "strategy_context": {
                "copy_event": {
                    "wallet_address": "0xabc",
                    "side": "BUY",
                    "price": 0.5,
                    "detected_at": detected_at,
                }
            },
        }
    )

    serialized = _serialize_order(row)
    copy_attribution = serialized["copy_attribution"]

    assert copy_attribution["source_wallet"] == "0xabc"
    assert abs(float(copy_attribution["source_price"]) - 0.5) < 1e-9
    assert abs(float(copy_attribution["follower_effective_price"]) - 0.52) < 1e-9
    assert abs(float(copy_attribution["slippage_bps"]) - 400.0) < 1e-6
    assert abs(float(copy_attribution["adverse_slippage_bps"]) - 400.0) < 1e-6
