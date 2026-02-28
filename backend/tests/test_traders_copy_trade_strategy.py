import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.traders_copy_trade import TradersCopyTradeStrategy
from utils.utcnow import utcnow


def _build_signal(*, detected_at: str | None):
    copy_event = {
        "wallet_address": "0xabc",
        "token_id": "token-1",
        "side": "BUY",
        "size": 20.0,
        "price": 0.55,
        "tx_hash": "0xhash",
    }
    source_trade = {
        "wallet_address": "0xabc",
        "side": "BUY",
        "source_notional_usd": 11.0,
        "size": 20.0,
        "price": 0.55,
        "tx_hash": "0xhash",
    }
    if detected_at is not None:
        copy_event["detected_at"] = detected_at
        source_trade["detected_at"] = detected_at
    payload = {
        "selected_token_id": "token-1",
        "token_id": "token-1",
        "source_trade": source_trade,
        "strategy_context": {
            "copy_event": copy_event,
        },
    }
    return SimpleNamespace(
        source="traders",
        strategy_type="traders_copy_trade",
        entry_price=0.55,
        confidence=0.8,
        payload_json=payload,
    )


def _default_context(*, max_signal_age_seconds: int):
    return {
        "params": {
            "max_signal_age_seconds": max_signal_age_seconds,
            "copy_buys": True,
            "copy_sells": True,
            "min_source_notional_usd": 10.0,
            "min_live_liquidity_usd": 100.0,
            "max_adverse_entry_drift_pct": 5.0,
            "max_entry_price": 0.99,
            "min_confidence": 0.1,
        },
        "live_market": {
            "live_selected_price": 0.55,
            "liquidity_usd": 1000.0,
            "entry_price_delta_pct": 0.0,
        },
    }


def test_copy_trade_signal_is_selected_when_fresh():
    strategy = TradersCopyTradeStrategy()
    detected_at = utcnow().isoformat()
    signal = _build_signal(detected_at=detected_at)

    decision = strategy.evaluate(signal, _default_context(max_signal_age_seconds=5))

    assert decision.decision == "selected"


def test_copy_trade_signal_is_rejected_when_stale_even_if_config_allows_more():
    strategy = TradersCopyTradeStrategy()
    detected_at = (utcnow() - timedelta(seconds=20)).isoformat()
    signal = _build_signal(detected_at=detected_at)

    decision = strategy.evaluate(signal, _default_context(max_signal_age_seconds=300))
    checks = {check.key: check for check in decision.checks}

    assert decision.decision == "skipped"
    assert checks["max_age"].passed is False


def test_copy_trade_signal_requires_timestamp():
    strategy = TradersCopyTradeStrategy()
    signal = _build_signal(detected_at=None)

    decision = strategy.evaluate(signal, _default_context(max_signal_age_seconds=5))
    checks = {check.key: check for check in decision.checks}

    assert decision.decision == "skipped"
    assert checks["signal_timestamp"].passed is False
