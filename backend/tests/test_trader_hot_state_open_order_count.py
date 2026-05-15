from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import trader_hot_state


def _reset_hot_state() -> None:
    trader_hot_state._snapshots.clear()
    trader_hot_state._global_gross.clear()
    trader_hot_state._global_daily_pnl.clear()
    trader_hot_state._global_daily_pnl_date.clear()
    trader_hot_state._recently_closed_markets.clear()
    trader_hot_state._seeded = False


def test_executed_orders_do_not_increment_open_order_count():
    _reset_hot_state()
    try:
        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="executed",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=50.0,
            payload={"token_id": "token-1"},
        )

        assert trader_hot_state.get_open_order_count("trader-1", "live") == 0
        assert trader_hot_state.get_open_position_count("trader-1", "live") == 1
    finally:
        _reset_hot_state()


def test_unfilled_orders_increment_and_cancelled_orders_clear_open_order_count():
    _reset_hot_state()
    try:
        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="open",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=0.0,
            payload={"token_id": "token-1"},
        )

        assert trader_hot_state.get_open_order_count("trader-1", "live") == 1

        trader_hot_state.record_order_cancelled(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            source="scanner",
        )

        assert trader_hot_state.get_open_order_count("trader-1", "live") == 0
    finally:
        _reset_hot_state()


def test_cancelled_orders_clear_open_position_count():
    _reset_hot_state()
    try:
        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="open",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=0.0,
            payload={"token_id": "token-1"},
        )

        assert trader_hot_state.get_open_position_count("trader-1", "live") == 1

        trader_hot_state.record_order_cancelled(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            source="scanner",
        )

        assert trader_hot_state.get_open_position_count("trader-1", "live") == 0
    finally:
        _reset_hot_state()


def test_unfilled_cancelled_orders_do_not_enter_reentry_cooldown():
    _reset_hot_state()
    try:
        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="open",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=0.0,
            payload={"token_id": "token-1"},
        )

        trader_hot_state.record_order_cancelled(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            source="scanner",
        )

        assert trader_hot_state.get_reentry_cooldown_market_ids("trader-1", "live") == set()
    finally:
        _reset_hot_state()


def test_active_order_status_transition_to_executed_clears_open_order_count():
    _reset_hot_state()
    try:
        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="open",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=0.0,
            payload={"token_id": "token-1"},
        )

        assert trader_hot_state.get_open_order_count("trader-1", "live") == 1

        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="executed",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=50.0,
            payload={"token_id": "token-1", "filled_size": 50.0},
        )

        assert trader_hot_state.get_open_order_count("trader-1", "live") == 0
        assert trader_hot_state.get_open_position_count("trader-1", "live") == 1
        assert trader_hot_state.get_occupied_market_ids("trader-1", "live") == {"market-1"}
    finally:
        _reset_hot_state()


def test_completed_orders_keep_market_occupied_without_counting_as_open_orders():
    _reset_hot_state()
    try:
        trader_hot_state.upsert_active_order(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            status="completed",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=25.0,
            entry_price=0.5,
            token_id="token-1",
            filled_shares=50.0,
            payload={"token_id": "token-1", "filled_size": 50.0},
        )

        assert trader_hot_state.get_open_order_count("trader-1", "live") == 0
        assert trader_hot_state.get_open_position_count("trader-1", "live") == 1
        assert trader_hot_state.get_occupied_market_ids("trader-1", "live") == {"market-1"}
    finally:
        _reset_hot_state()


def test_resolved_order_updates_daily_pnl_by_order_delta_only():
    _reset_hot_state()
    try:
        trader_hot_state.record_order_resolved(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            status="closed_loss",
            actual_profit=-1.25,
            payload={},
        )

        assert trader_hot_state.get_daily_realized_pnl("trader-1", "live") == -1.25
        assert trader_hot_state.get_daily_realized_pnl(None, "live") == -1.25
        assert trader_hot_state.get_consecutive_loss_count("trader-1", "live") == 0

        trader_hot_state.record_order_resolved(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            status="closed_loss",
            actual_profit=-1.25,
            payload={},
        )

        assert trader_hot_state.get_daily_realized_pnl("trader-1", "live") == -1.25
        assert trader_hot_state.get_daily_realized_pnl(None, "live") == -1.25
        assert trader_hot_state.get_consecutive_loss_count("trader-1", "live") == 0

        trader_hot_state.record_order_resolved(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            status="closed_loss",
            actual_profit=-0.75,
            payload={},
        )

        assert trader_hot_state.get_daily_realized_pnl("trader-1", "live") == -0.75
        assert trader_hot_state.get_daily_realized_pnl(None, "live") == -0.75
        assert trader_hot_state.get_consecutive_loss_count("trader-1", "live") == 0

        trader_hot_state.record_order_resolved(
            trader_id="trader-1",
            mode="live",
            order_id="order-1",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            status="closed_win",
            actual_profit=0.5,
            payload={},
        )

        assert trader_hot_state.get_daily_realized_pnl("trader-1", "live") == 0.5
        assert trader_hot_state.get_daily_realized_pnl(None, "live") == 0.5
        assert trader_hot_state.get_consecutive_loss_count("trader-1", "live") == 0
    finally:
        _reset_hot_state()
