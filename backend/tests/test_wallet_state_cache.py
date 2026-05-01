"""Tests for WalletStateCache: position derivation, freshness gate,
REST seeding, WS event application.

These tests are pure-Python (no DB, no network) so they run fast and
cleanly in any environment.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import time

import pytest

from services.wallet_state_cache import (
    WalletStateCache,
    reset_wallet_state_cache,
)


@pytest.fixture
def cache():
    reset_wallet_state_cache()
    c = WalletStateCache()
    c.configure_wallet("0xabcdef0123456789abcdef0123456789abcdef01")
    return c


def _trade_event(
    *,
    trade_id="t1",
    asset_id="token-yes",
    market="0xcondition",
    side="BUY",
    price="0.42",
    size="100",
    status="MATCHED",
    owner="0xabcdef0123456789abcdef0123456789abcdef01",
    taker_order_id="order-1",
    timestamp="1700000000",
):
    return {
        "event_type": "trade",
        "id": trade_id,
        "asset_id": asset_id,
        "market": market,
        "owner": owner,
        "trade_owner": owner,
        "side": side,
        "price": price,
        "size": size,
        "status": status,
        "taker_order_id": taker_order_id,
        "timestamp": timestamp,
        "matchtime": timestamp,
    }


def _order_event(
    *,
    order_id="order-1",
    asset_id="token-yes",
    market="0xcondition",
    side="BUY",
    price="0.42",
    original_size="100",
    size_matched="0",
    type_="PLACEMENT",
    owner="0xabcdef0123456789abcdef0123456789abcdef01",
    timestamp="1700000000",
):
    return {
        "event_type": "order",
        "id": order_id,
        "asset_id": asset_id,
        "market": market,
        "owner": owner,
        "order_owner": owner,
        "side": side,
        "price": price,
        "original_size": original_size,
        "size_matched": size_matched,
        "type": type_,
        "timestamp": timestamp,
    }


class TestApplyTrade:
    def test_buy_adds_position(self, cache):
        cache.apply_trade(_trade_event(side="BUY", price="0.42", size="100"))
        pos = cache.get_position("token-yes")
        assert pos is not None
        assert pos.size == 100.0
        assert pos.cost_basis_usd == pytest.approx(42.0)
        assert pos.avg_entry_price == pytest.approx(0.42)
        assert pos.condition_id == "0xcondition"

    def test_sell_reduces_position(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="200"))
        cache.apply_trade(_trade_event(trade_id="t2", side="SELL", price="0.50", size="100"))
        pos = cache.get_position("token-yes")
        assert pos is not None
        assert pos.size == 100.0
        # Cost basis reduced proportionally: original 80, sold half => 40 remaining
        assert pos.cost_basis_usd == pytest.approx(40.0)

    def test_full_exit_removes_position(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100"))
        cache.apply_trade(_trade_event(trade_id="t2", side="SELL", price="0.50", size="100"))
        assert cache.get_position("token-yes") is None

    def test_duplicate_trade_id_skipped(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100"))
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100"))
        pos = cache.get_position("token-yes")
        assert pos is not None
        assert pos.size == 100.0  # not 200

    def test_status_update_only_changes_fill_status(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100", status="MATCHED"))
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100", status="CONFIRMED"))
        pos = cache.get_position("token-yes")
        assert pos.size == 100.0
        # Fill record should reflect CONFIRMED status
        fills = cache.get_recent_fills_for_order("order-1")
        assert len(fills) == 1
        assert fills[0].status == "CONFIRMED"

    def test_failed_trade_reverses_position(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100", status="MATCHED"))
        assert cache.get_position("token-yes").size == 100.0
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100", status="FAILED"))
        # Position fully reversed → removed.
        assert cache.get_position("token-yes") is None

    def test_other_wallet_trade_dropped(self, cache):
        cache.apply_trade(
            _trade_event(
                owner="0xdeadbeef00000000000000000000000000000000",
            )
        )
        assert cache.get_position("token-yes") is None

    def test_invalid_event_dropped(self, cache):
        cache.apply_trade(_trade_event(side="INVALID"))
        cache.apply_trade(_trade_event(price="0"))
        cache.apply_trade(_trade_event(size="0"))
        cache.apply_trade(_trade_event(asset_id=""))
        assert cache.get_position("token-yes") is None


class TestApplyOrder:
    def test_placement_creates_order(self, cache):
        cache.apply_order(_order_event(order_id="o1", type_="PLACEMENT"))
        order = cache.get_order("o1")
        assert order is not None
        assert order.status == "PLACEMENT"
        assert not order.is_terminal

    def test_cancellation_marks_terminal(self, cache):
        cache.apply_order(_order_event(order_id="o1", type_="PLACEMENT"))
        cache.apply_order(_order_event(order_id="o1", type_="CANCELLATION"))
        order = cache.get_order("o1")
        assert order.is_terminal

    def test_size_matched_increases(self, cache):
        cache.apply_order(_order_event(order_id="o1", original_size="100", size_matched="0"))
        cache.apply_order(_order_event(order_id="o1", original_size="100", size_matched="60", type_="UPDATE"))
        order = cache.get_order("o1")
        assert order.size_matched == 60.0
        assert order.remaining_size == 40.0

    def test_open_orders_filter_terminal(self, cache):
        cache.apply_order(_order_event(order_id="o1", type_="PLACEMENT"))
        cache.apply_order(_order_event(order_id="o2", type_="PLACEMENT"))
        cache.apply_order(_order_event(order_id="o2", type_="CANCELLATION"))
        open_orders = cache.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == "o1"


class TestSeedFromRest:
    def test_open_positions_seeded(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[
                {
                    "asset": "token-yes",
                    "conditionId": "0xcond",
                    "outcomeIndex": 0,
                    "size": "150",
                    "avgPrice": "0.40",
                    "curPrice": "0.45",
                }
            ],
            closed_positions=[],
        )
        pos = cache.get_position("token-yes")
        assert pos is not None
        assert pos.size == 150.0
        assert pos.cost_basis_usd == pytest.approx(60.0)
        assert pos.last_rest_mark_price == pytest.approx(0.45)

    def test_seed_overwrites_ws_derived(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100"))
        # REST snapshot says size is actually 110 — overwrite WS
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[
                {
                    "asset": "token-yes",
                    "conditionId": "0xcondition",
                    "outcomeIndex": 0,
                    "size": "110",
                    "avgPrice": "0.40",
                }
            ],
            closed_positions=[],
        )
        pos = cache.get_position("token-yes")
        assert pos.size == 110.0

    def test_closed_positions_marked_resolved(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[],
            closed_positions=[
                {
                    "asset": "token-yes",
                    "conditionId": "0xcond",
                    "redeemable": True,
                    "curPrice": "1.0",
                }
            ],
        )
        pos = cache.get_position("token-yes")
        assert pos is not None
        assert pos.is_resolved
        assert pos.settlement_price == pytest.approx(1.0)
        assert pos.redeemable

    def test_stale_position_removed(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[
                {"asset": "token-A", "conditionId": "0x1", "outcomeIndex": 0, "size": "10", "avgPrice": "0.5"}
            ],
            closed_positions=[],
        )
        # Subsequent seed without token-A — should be removed.
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[],
            closed_positions=[],
        )
        assert cache.get_position("token-A") is None

    def test_failed_seed_marks_unhealthy(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[],
            closed_positions=[],
            succeeded=False,
        )
        fresh, reason = cache.is_fresh()
        assert not fresh
        assert "failed" in reason


class TestFreshnessGate:
    def test_no_seed_yet(self, cache):
        fresh, reason = cache.is_fresh()
        assert not fresh
        assert reason == "no_rest_seed_yet"

    def test_idle_wallet_fresh_after_seed(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[],
            closed_positions=[],
        )
        fresh, reason = cache.is_fresh()
        # Empty positions → idle wallet → trust the seed alone
        assert fresh
        assert reason == "ok_idle"

    def test_active_wallet_requires_ws(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[
                {"asset": "token-yes", "conditionId": "0x1", "outcomeIndex": 0, "size": "10", "avgPrice": "0.5"}
            ],
            closed_positions=[],
        )
        # WS not connected
        fresh, reason = cache.is_fresh()
        assert not fresh
        assert reason == "ws_disconnected"

    def test_active_wallet_ws_warming(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[
                {"asset": "token-yes", "conditionId": "0x1", "outcomeIndex": 0, "size": "10", "avgPrice": "0.5"}
            ],
            closed_positions=[],
        )
        cache.mark_ws_state(True)
        fresh, reason = cache.is_fresh()
        assert fresh
        assert reason == "ok_ws_warming"

    def test_seed_stale_blocks(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[],
            closed_positions=[],
        )
        # Manually age the seed monotonic
        cache._last_rest_seed_mono = time.monotonic() - 600.0
        fresh, reason = cache.is_fresh(max_seed_age_seconds=120.0)
        assert not fresh
        assert "rest_seed_stale" in reason


class TestSubscriptionHelpers:
    def test_iter_tracked_condition_ids(self, cache):
        cache.seed_from_rest(
            wallet_address="0xabc",
            positions=[
                {"asset": "tA", "conditionId": "0xc1", "outcomeIndex": 0, "size": "10", "avgPrice": "0.5"},
                {"asset": "tB", "conditionId": "0xc2", "outcomeIndex": 0, "size": "20", "avgPrice": "0.5"},
            ],
            closed_positions=[],
        )
        cids = cache.iter_tracked_condition_ids()
        assert set(cids) == {"0xc1", "0xc2"}


class TestLegacyDictShape:
    def test_to_legacy_dict_has_all_keys(self, cache):
        cache.apply_trade(_trade_event(trade_id="t1", side="BUY", price="0.40", size="100"))
        snapshot = cache.positions_by_token()
        assert "token-yes" in snapshot
        legacy = snapshot["token-yes"]
        # Confirm the orchestrator's existing _extract_wallet_position_size
        # / _extract_wallet_entry_price code paths see the fields they
        # expect.
        assert legacy["size"] == 100.0
        assert legacy["positionSize"] == 100.0
        assert legacy["shares"] == 100.0
        assert legacy["avgPrice"] == pytest.approx(0.40)
        assert legacy["asset"] == "token-yes"
        assert legacy["conditionId"] == "0xcondition"
        assert legacy["counts_as_open"] is True
