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


def _default_context(*, max_signal_age_seconds: int, mode: str = "shadow"):
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
        "mode": mode,
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


def test_copy_trade_detect_builds_opportunity_from_copy_event():
    strategy = TradersCopyTradeStrategy()
    detected_at = utcnow().isoformat()
    opportunities = strategy.detect(
        [
            {
                "copy_event": {
                    "wallet_address": "0xabc",
                    "token_id": "token-1",
                    "side": "BUY",
                    "size": 20.0,
                    "price": 0.55,
                    "tx_hash": "0xhash",
                    "order_hash": "0xorder",
                    "log_index": 1,
                    "detected_at": detected_at,
                },
                "source_trade": {
                    "wallet_address": "0xabc",
                    "side": "BUY",
                    "source_notional_usd": 11.0,
                    "size": 20.0,
                    "price": 0.55,
                    "tx_hash": "0xhash",
                    "detected_at": detected_at,
                },
                "market": {
                    "market_id": "market-1",
                    "market_question": "Will this happen?",
                    "market_slug": "will-this-happen",
                    "outcome": "YES",
                    "liquidity": 5000.0,
                    "token_id": "token-1",
                },
                "source_item_id": "source-item-1",
                "dedupe_key": "dedupe-1",
            }
        ],
        [],
        {},
    )

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.strategy == "traders_copy_trade"
    assert opportunity.strategy_context["copy_event"]["wallet_address"] == "0xabc"
    assert opportunity.positions_to_take[0]["action"] == "BUY"
    assert opportunity.positions_to_take[0]["outcome"] == "YES"


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


def test_copy_trade_signal_uses_strategy_context_source_trade_when_payload_source_trade_missing():
    strategy = TradersCopyTradeStrategy()
    detected_at = utcnow().isoformat()
    signal = _build_signal(detected_at=detected_at)
    source_trade = dict(signal.payload_json["source_trade"])
    signal.payload_json.pop("source_trade", None)
    signal.payload_json["strategy_context"]["source_trade"] = source_trade

    decision = strategy.evaluate(signal, _default_context(max_signal_age_seconds=5))

    assert decision.decision == "selected"


def test_copy_trade_signal_rejects_when_source_exposure_cap_exceeded():
    strategy = TradersCopyTradeStrategy()
    signal = _build_signal(detected_at=utcnow().isoformat())
    context = _default_context(max_signal_age_seconds=5)
    context["params"]["max_copy_source_exposure_usd"] = 200.0
    context["copy_allocation_context"] = {
        "current_source_exposure_usd": 250.0,
        "current_leader_exposure_usd": 50.0,
    }

    decision = strategy.evaluate(signal, context)
    checks = {check.key: check for check in decision.checks}

    assert decision.decision == "skipped"
    assert checks["copy_source_exposure"].passed is False


def test_copy_trade_signal_applies_leader_weight_and_allocation_cap_to_size():
    strategy = TradersCopyTradeStrategy()
    signal = _build_signal(detected_at=utcnow().isoformat())
    context = _default_context(max_signal_age_seconds=5)
    context["params"].update(
        {
            "leader_weights": {"0xabc": 0.5},
            "default_leader_weight": 1.0,
            "leader_allocation_cap_pct": 60.0,
            "proportional_multiplier": 2.0,
            "max_position_size": 1000.0,
            "max_size_usd": 1000.0,
        }
    )
    context["copy_allocation_context"] = {
        "source_wallet": "0xabc",
        "current_source_exposure_usd": 0.0,
        "current_leader_exposure_usd": 0.0,
    }

    decision = strategy.evaluate(signal, context)
    assert decision.decision == "selected"
    assert decision.size_usd is not None
    # source_notional=11, proportional_multiplier=2 => 22
    # leader_weight=0.5 => 11
    # leader_allocation_cap_pct=60 => cap=6.6
    assert abs(float(decision.size_usd) - 6.6) < 1e-6


def test_copy_trade_live_sell_allows_partial_inventory_when_enabled():
    strategy = TradersCopyTradeStrategy()
    detected_at = utcnow().isoformat()
    signal = _build_signal(detected_at=detected_at)
    copy_event = signal.payload_json["strategy_context"]["copy_event"]
    source_trade = signal.payload_json["source_trade"]
    copy_event["side"] = "SELL"
    source_trade["side"] = "SELL"
    copy_event["size"] = 20.0
    source_trade["size"] = 20.0
    source_trade["source_notional_usd"] = 11.0
    signal.entry_price = 0.55

    context = _default_context(max_signal_age_seconds=5, mode="live")
    context["params"].update(
        {
            "allow_partial_inventory_sells": True,
            "require_inventory_for_sells": True,
            "min_inventory_fraction": 0.2,
            "proportional_multiplier": 1.0,
            "max_position_size": 1000.0,
            "max_size_usd": 1000.0,
        }
    )
    context["copy_inventory_context"] = {
        "available": True,
        "wallet_address": "0xwallet",
        "token_inventory": {
            "token-1": {
                "size": 5.0,
            }
        },
    }
    context["copy_allocation_context"] = {
        "source_wallet": "0xabc",
        "current_source_exposure_usd": 0.0,
        "current_leader_exposure_usd": 0.0,
    }

    decision = strategy.evaluate(signal, context)
    checks = {check.key: check for check in decision.checks}

    assert decision.decision == "selected"
    assert checks["sell_inventory"].passed is True
    # available_shares=5 at price=0.55 => capped size=2.75
    assert abs(float(decision.size_usd or 0.0) - 2.75) < 1e-6
