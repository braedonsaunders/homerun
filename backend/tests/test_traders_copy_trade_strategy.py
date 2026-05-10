"""Regression tests for ``TradersCopyTradeStrategy._build_copy_opportunity``.

Pin the post-fix contract from plan 0018: the strategy emits the
canonical ``buy_yes``/``buy_no`` for binary YES/NO outcomes and an
empty string otherwise so the session_engine fallback resolver
(``_resolve_leg_direction``) reconstructs the direction from
``(side, outcome)``. Emitting a synthetic bare ``"buy"`` here was
the upstream cause of stuck shadow positions on the Sandbox
``traders_copy_trade`` bot.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.traders_copy_trade import TradersCopyTradeStrategy


def _payload(*, outcome: str, token_id: str = "token-leader") -> dict:
    return {
        "copy_event": {
            "wallet_address": "0xabc",
            "token_id": token_id,
            "side": "BUY",
            "size": 25.0,
            "price": 0.62,
            "tx_hash": "0xhash",
            "order_hash": "0xorder",
            "log_index": 1,
            "block_number": 123,
            "timestamp": "2026-05-09T10:00:00+00:00",
            "detected_at": "2026-05-09T10:00:01+00:00",
            "latency_ms": 12.5,
            "confidence": 0.7,
            "outcome": outcome,
            "market_id": "market-1",
            "market_question": "Test market",
            "market_slug": "test-market",
            "signal_type": "single_wallet_buy",
        },
        "market": {
            "market_id": "market-1",
            "id": "market-1",
            "outcome": outcome,
            "token_id": token_id,
            "question": "Test market",
            "slug": "test-market",
            "liquidity": 10_000.0,
        },
        "source_trade": {
            "wallet_address": "0xabc",
            "side": "BUY",
            "source_notional_usd": 1_000.0,
            "size": 25.0,
            "price": 0.62,
            "tx_hash": "0xhash",
            "order_hash": "0xorder",
            "log_index": 1,
            "detected_at": "2026-05-09T10:00:01+00:00",
        },
        "source_item_id": "src-1",
        "dedupe_key": "dedupe-1",
    }


def test_build_copy_opportunity_emits_buy_yes_for_yes_outcome():
    strategy = TradersCopyTradeStrategy()
    opportunity = strategy._build_copy_opportunity(_payload(outcome="Yes"))
    assert opportunity is not None
    assert len(opportunity.positions_to_take) == 1
    assert opportunity.positions_to_take[0]["direction"] == "buy_yes"


def test_build_copy_opportunity_emits_buy_no_for_no_outcome():
    strategy = TradersCopyTradeStrategy()
    opportunity = strategy._build_copy_opportunity(_payload(outcome="No"))
    assert opportunity is not None
    assert len(opportunity.positions_to_take) == 1
    assert opportunity.positions_to_take[0]["direction"] == "buy_no"


def test_build_copy_opportunity_emits_empty_direction_for_non_binary_outcome():
    strategy = TradersCopyTradeStrategy()
    opportunity = strategy._build_copy_opportunity(_payload(outcome="Fighter A"))
    assert opportunity is not None
    assert len(opportunity.positions_to_take) == 1
    assert opportunity.positions_to_take[0]["direction"] == ""
    assert opportunity.positions_to_take[0]["token_id"] == "token-leader"
