import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, Trader, TraderOrder, TraderOrderVerificationEvent
from services.trader_orchestrator import position_lifecycle
from services.strategies.base import ExitDecision
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_position_lifecycle_resolution")


async def _seed_order(
    session: AsyncSession,
    *,
    direction: str = "buy_yes",
    order_id: str = "order-1",
    signal_id: str = "signal-1",
    mode: str = "shadow",
    status: str = "executed",
    payload_json: dict | None = None,
) -> None:
    now = datetime.utcnow()
    session.add(
        Trader(
            id="trader-1",
            name="Crypto Trader",
            source_configs_json=[{"source_key": "crypto", "strategy_key": "btc_eth_maker_quote", "strategy_params": {}}],
            risk_limits_json={},
            metadata_json={},
            is_enabled=True,
            is_paused=False,
            interval_seconds=60,
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        TradeSignal(
            id=signal_id,
            source="crypto",
            signal_type="entry",
            strategy_type="crypto_15m",
            market_id="market-1",
            direction=direction,
            entry_price=0.4,
            dedupe_key=f"dedupe-{signal_id}",
            payload_json={"yes_price": 0.4, "no_price": 0.6},
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        TraderOrder(
            id=order_id,
            trader_id="trader-1",
            signal_id=signal_id,
            source="crypto",
            market_id="market-1",
            direction=direction,
            mode=mode,
            status=status,
            notional_usd=40.0,
            entry_price=0.4,
            effective_price=0.4,
            payload_json=payload_json or {},
            created_at=now,
            executed_at=now,
            updated_at=now,
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_load_market_info_for_orders_uses_payload_aliases_when_market_id_is_numeric(monkeypatch):
    order = TraderOrder(
        id="order-alias-load",
        trader_id="trader-1",
        market_id="1456183",
        payload_json={
            "token_id": "114272436076579993156006679750078555145975747668079881983093157238223214149562",
            "live_market": {
                "condition_id": "0x2baff1bf5e4771cc3728df56a825374733f3c9b3faa8da72e0572419f17efaa8",
                "selected_token_id": "114272436076579993156006679750078555145975747668079881983093157238223214149562",
            },
        },
    )
    expected = {"id": "1456183", "closed": True, "accepting_orders": False, "outcome_prices": [0.0, 1.0]}
    get_by_condition = AsyncMock(return_value=expected)
    get_by_token = AsyncMock(return_value=None)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_condition_id", get_by_condition)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_token_id", get_by_token)

    result = await position_lifecycle.load_market_info_for_orders([order])

    assert result["1456183"]["id"] == expected["id"]
    assert result["1456183"]["closed"] is True
    assert result["1456183"]["accepting_orders"] is False
    assert result["1456183"]["outcome_prices"] == [0.0, 1.0]
    get_by_condition.assert_awaited()


def test_fallback_market_info_for_orders_uses_payload_live_market():
    order = SimpleNamespace(
        market_id="market-1",
        payload_json={
            "token_id": "token-1",
            "live_market": {
                "market_id": "market-1",
                "selected_token_id": "token-1",
                "closed": False,
                "accepting_orders": True,
                "outcome_prices": [0.4, 0.6],
            },
        },
    )

    result = position_lifecycle._fallback_market_info_for_orders([order])

    assert result == {
        "market-1": {
            "id": "market-1",
            "market_id": "market-1",
            "selected_token_id": "token-1",
            "closed": False,
            "accepting_orders": True,
            "outcome_prices": [0.4, 0.6],
            "token_ids": ["token-1"],
        }
    }


def test_fallback_market_info_for_orders_uses_payload_markets_when_live_market_missing():
    order = SimpleNamespace(
        market_id="market-1",
        direction="buy_yes",
        payload_json={
            "token_id": "token-1",
            "resolution_date": "2026-04-18T12:00:00Z",
            "market_question": "Will market 1 resolve yes?",
            "markets": [
                {
                    "id": "market-1",
                    "condition_id": "condition-1",
                    "question": "Will market 1 resolve yes?",
                    "yes_price": 0.145,
                    "no_price": 0.855,
                    "clob_token_ids": ["token-1", "token-2"],
                }
            ],
            "positions_to_take": [
                {
                    "market_id": "market-1",
                    "outcome": "YES",
                    "token_id": "token-1",
                    "price": 0.145,
                }
            ],
        },
    )

    result = position_lifecycle._fallback_market_info_for_orders([order])

    assert result["market-1"]["market_id"] == "market-1"
    assert result["market-1"]["condition_id"] == "condition-1"
    assert result["market-1"]["yes_price"] == pytest.approx(0.145, abs=1e-9)
    assert result["market-1"]["end_date"] == "2026-04-18T12:00:00Z"
    assert result["market-1"]["selected_outcome"] == "yes"
    assert result["market-1"]["selected_token_id"] == "token-1"
    assert result["market-1"]["token_ids"] == ["token-1", "token-2"]


@pytest.mark.asyncio
async def test_load_market_info_merges_partial_lookup_with_payload_fallback(monkeypatch):
    order = SimpleNamespace(
        market_id="market-1",
        payload_json={
            "token_id": "token-1",
            "live_market": {
                "market_id": "market-1",
                "selected_token_id": "token-1",
                "selected_outcome": "yes",
                "live_selected_price": 0.0005,
                "market_end_time": "2026-04-18T12:00:00Z",
            },
        },
    )
    token_lookup = AsyncMock(
        return_value={
            "condition_id": "0xcondition-1",
            "question": "Old market stub",
            "token_ids": ["token-1", "token-2"],
        }
    )
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_condition_id", AsyncMock(return_value=None))
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_token_id", token_lookup)

    result = await position_lifecycle.load_market_info_for_orders([order])

    assert result["market-1"]["market_id"] == "market-1"
    assert result["market-1"]["id"] == "market-1"
    assert result["market-1"]["yes_price"] == pytest.approx(0.0005, abs=1e-9)
    assert result["market-1"]["end_date"] == "2026-04-18T12:00:00Z"
    assert result["market-1"]["token_ids"] == ["token-1", "token-2"]


def test_direction_outcome_index_canonical_buy_yes_unchanged_by_widening():
    assert position_lifecycle._direction_outcome_index("buy_yes") == 0
    assert position_lifecycle._direction_outcome_index("buy_no") == 1
    assert (
        position_lifecycle._direction_outcome_index(
            "buy_yes",
            market_info={"token_ids": ["only-one"]},
            token_id="ignored",
        )
        == 0
    )


def test_direction_outcome_index_resolves_buy_via_token_id_with_market_info():
    market_info = {"clob_token_ids": ["token-yes", "token-no"]}
    assert (
        position_lifecycle._direction_outcome_index(
            "buy",
            market_info=market_info,
            token_id="token-no",
        )
        == 1
    )
    assert (
        position_lifecycle._direction_outcome_index(
            "buy",
            market_info=market_info,
            token_id="token-yes",
        )
        == 0
    )


def test_direction_outcome_index_returns_none_when_token_id_not_in_market():
    assert (
        position_lifecycle._direction_outcome_index(
            "buy",
            market_info={"token_ids": ["token-yes", "token-no"]},
            token_id="ghost-token",
        )
        is None
    )


def test_direction_outcome_index_returns_none_for_truly_multi_outcome():
    assert (
        position_lifecycle._direction_outcome_index(
            "buy",
            market_info={"token_ids": ["fighter-a", "fighter-b", "fighter-c"]},
            token_id="fighter-b",
        )
        is None
    )


def test_direction_outcome_index_returns_none_when_widening_inputs_missing():
    assert position_lifecycle._direction_outcome_index("buy") is None
    assert (
        position_lifecycle._direction_outcome_index("buy", market_info=None, token_id="t")
        is None
    )


def test_direction_outcome_index_canonical_sell_yes_resolves_to_yes_index():
    assert position_lifecycle._direction_outcome_index("sell_yes") == 0
    assert position_lifecycle._direction_outcome_index("sell_no") == 1
    assert (
        position_lifecycle._direction_outcome_index(
            "sell_yes",
            market_info={"token_ids": ["only-one"]},
            token_id="ignored",
        )
        == 0
    )


def test_extract_leg_token_id_prefers_top_level_then_leg_then_yes_no_aliases():
    assert position_lifecycle._extract_leg_token_id({"token_id": "t1"}) == "t1"
    assert position_lifecycle._extract_leg_token_id({"selected_token_id": "t2"}) == "t2"
    assert position_lifecycle._extract_leg_token_id({"leg": {"token_id": "t3"}}) == "t3"
    assert (
        position_lifecycle._extract_leg_token_id(
            {"direction": "buy_yes", "yes_token_id": "t4", "no_token_id": "t5"}
        )
        == "t4"
    )
    assert position_lifecycle._extract_leg_token_id({}) == ""


def test_extract_wallet_settlement_price_treats_non_open_redeemable_position_as_winner():
    settlement_price = position_lifecycle._extract_wallet_settlement_price(
        {
            "redeemable": True,
            "counts_as_open": False,
            "size": 5.0,
            "currentPrice": None,
        }
    )

    assert settlement_price == 1.0


def test_provider_snapshot_status_prefers_raw_terminal_status_over_stale_pending_mapping():
    status = position_lifecycle._provider_snapshot_status(
        {
            "provider_reconciliation": {
                "snapshot_status": "placing",
                "mapped_status": "placing",
                "snapshot": {
                    "normalized_status": "placing",
                    "status": "cancelled",
                    "filled_size": 0.0,
                },
            }
        }
    )

    assert status == "cancelled"


def test_pending_exit_fill_evidence_prefers_raw_terminal_status_over_stale_pending_mapping():
    provider_status, filled_size, filled_notional, average_fill_price = position_lifecycle._pending_exit_fill_evidence(
        {
            "provider_status": "placing",
            "snapshot": {
                "normalized_status": "placing",
                "status": "cancelled",
                "filled_size": 0.0,
                "average_fill_price": 0.0,
                "filled_notional_usd": 0.0,
            },
        }
    )

    assert provider_status == "cancelled"
    assert filled_size == 0.0
    assert filled_notional == 0.0
    assert average_fill_price == 0.0


def test_cached_order_snapshots_by_clob_ids_returns_cached_matches(monkeypatch):
    cached_order = SimpleNamespace(
        clob_order_id="clob-1",
        status="open",
        filled_size=3.0,
        average_fill_price=0.41,
        size=5.0,
        price=0.42,
    )
    monkeypatch.setattr(position_lifecycle.live_execution_service, "_orders", {"order-1": cached_order})
    monkeypatch.setattr(
        position_lifecycle.live_execution_service,
        "_snapshot_from_cached_order",
        lambda order: {"clob_order_id": order.clob_order_id, "filled_size": order.filled_size},
    )

    result = position_lifecycle._cached_order_snapshots_by_clob_ids(["clob-1", "missing"])

    assert result == {"clob-1": {"clob_order_id": "clob-1", "filled_size": 3.0}}


@pytest.mark.asyncio
async def test_reconcile_infers_resolution_from_settled_prices_when_terminal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(session, direction="buy_yes")
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_shadow_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result["by_status"]["resolved_win"] == 1
            assert order is not None
            assert order.status == "resolved_win"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "resolution_inferred"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_prefers_explicit_winner_over_price_inference(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(session, direction="buy_yes")
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 1,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_shadow_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result["by_status"]["resolved_loss"] == 1
            assert order is not None
            assert order.status == "resolved_loss"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "resolution"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_skips_wallet_history_when_position_is_still_open(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "token_id": "token-1",
                    "live_market": {"selected_token_id": "token-1"},
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.4, 0.6],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"size": 100.0, "curPrice": 0.4}}),
            )
            closed_loader = AsyncMock(return_value={})
            sell_loader = AsyncMock(return_value={})
            activity_loader = AsyncMock(return_value={})
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_closed_positions_by_token", closed_loader)
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_sell_trades_by_token", sell_loader)
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_close_activity_by_token", activity_loader)
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle, "_wallet_closed_positions_last_refresh_succeeded", False)
            monkeypatch.setattr(position_lifecycle, "_wallet_activity_last_refresh_succeeded", False)
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            assert result["held"] == 1
            closed_loader.assert_not_awaited()
            sell_loader.assert_not_awaited()
            activity_loader.assert_not_awaited()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_uses_cached_wallet_positions_when_refresh_times_out(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "token_id": "token-1",
                    "live_market": {"selected_token_id": "token-1"},
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.4, 0.6],
                        }
                    }
                ),
            )

            async def slow_wallet_positions():
                await asyncio.sleep(0.05)
                return {}

            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_positions_by_token", slow_wallet_positions)
            monkeypatch.setattr(position_lifecycle, "_WALLET_POSITIONS_LOAD_TIMEOUT_SECONDS", 0.01)
            monkeypatch.setattr(
                position_lifecycle,
                "_wallet_positions_cache",
                (
                    position_lifecycle._time_mod.monotonic(),
                    {"token-1": {"size": 100.0, "curPrice": 0.4}},
                ),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            assert result["held"] == 1
            assert result["closed"] == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_uses_payload_market_snapshot_when_market_refresh_times_out(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "token_id": "token-1",
                    "live_market": {
                        "market_id": "market-1",
                        "selected_token_id": "token-1",
                        "closed": False,
                        "accepting_orders": True,
                        "winner": None,
                        "winning_outcome": None,
                        "outcome_prices": [0.4, 0.6],
                    },
                },
            )

            async def slow_market_info(_orders):
                await asyncio.sleep(0.05)
                return {}

            monkeypatch.setattr(position_lifecycle, "load_market_info_for_orders", slow_market_info)
            monkeypatch.setattr(position_lifecycle, "_MARKET_INFO_LOAD_TIMEOUT_SECONDS", 0.01)
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"size": 100.0, "curPrice": 0.4}}),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            assert result["held"] == 1
            assert result["closed"] == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_skips_wallet_history_for_fresh_wallet_absent_order(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "token_id": "token-1",
                    "live_market": {"selected_token_id": "token-1"},
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.4, 0.6],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            closed_loader = AsyncMock(return_value={})
            sell_loader = AsyncMock(return_value={})
            activity_loader = AsyncMock(return_value={})
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_closed_positions_by_token", closed_loader)
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_sell_trades_by_token", sell_loader)
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_close_activity_by_token", activity_loader)
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle, "_wallet_closed_positions_last_refresh_succeeded", False)
            monkeypatch.setattr(position_lifecycle, "_wallet_activity_last_refresh_succeeded", False)
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            assert result["held"] == 1
            closed_loader.assert_not_awaited()
            sell_loader.assert_not_awaited()
            activity_loader.assert_not_awaited()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_does_not_reopen_unfilled_failed_row_from_same_token_wallet_position(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="failed",
                payload_json={
                    "token_id": "token-1",
                    "live_market": {"selected_token_id": "token-1"},
                    "provider_reconciliation": {
                        "snapshot_status": "failed",
                        "mapped_status": "cancelled",
                        "snapshot": {
                            "normalized_status": "failed",
                            "filled_size": 0.0,
                            "average_fill_price": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                    },
                },
            )
            row = await session.get(TraderOrder, "order-1")
            assert row is not None
            row.verification_status = "local"
            row.error_message = "PolyApiException[status_code=400, error_message={'error': 'no orders found to match with FAK order. FAK orders are partially filled or killed if no match is found.'}]"
            await session.commit()

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"size": 5.95, "curPrice": 0.749}}),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle, "_wallet_closed_positions_last_refresh_succeeded", False)
            monkeypatch.setattr(position_lifecycle, "_wallet_activity_last_refresh_succeeded", False)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            refreshed = await session.get(TraderOrder, "order-1")
            assert refreshed is not None
            assert refreshed.status == "failed"
            assert result["state_updates"] == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_reopens_cancelled_order_when_wallet_position_exists(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="cancelled",
                payload_json={
                    "token_id": "token-1",
                    "live_wallet_authority": {
                        "source": "live_trading_orders",
                        "token_id": "token-1",
                        "live_trading_order_id": "live-order-1",
                    },
                    "provider_reconciliation": {
                        "snapshot_status": "cancelled",
                        "snapshot": {
                            "normalized_status": "cancelled",
                            "status": "cancelled",
                            "filled_size": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                    },
                },
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            order.notional_usd = 0.0
            order.entry_price = 0.0
            order.effective_price = 0.0
            order.verification_status = "venue_order"
            order.verification_source = "live_order_ack"
            order.updated_at = datetime.utcnow() - timedelta(minutes=1)
            await session.commit()

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "asset_id": "token-1",
                            "token_id": "token-1",
                            "size": 37.5,
                            "avgPrice": 0.24,
                            "currentPrice": 0.24,
                            "conditionId": "condition-1",
                            "outcomeIndex": 0,
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1"],
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] >= 1
            assert order is not None
            assert order.status == "open"
            assert order.notional_usd == pytest.approx(9.0)
            assert order.entry_price == pytest.approx(0.24)
            assert order.effective_price == pytest.approx(0.24)
            assert order.verification_status == "wallet_position"
            assert order.verification_source == "wallet_positions_api"
            payload = order.payload_json or {}
            assert payload.get("wallet_position_reopen", {}).get("reopen_reason") == "wallet_position_still_held"
            assert payload.get("provider_reconciliation", {}).get("snapshot_status") == "filled"
            assert payload.get("provider_reconciliation", {}).get("filled_size") == pytest.approx(37.5)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_allocates_wallet_position_once_across_duplicate_terminal_rows(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="cancelled",
                payload_json={
                    "token_id": "token-1",
                    "live_wallet_authority": {
                        "source": "live_trading_orders",
                        "token_id": "token-1",
                        "live_trading_order_id": "live-order-1",
                    },
                    "provider_reconciliation": {
                        "snapshot_status": "cancelled",
                        "snapshot": {
                            "normalized_status": "cancelled",
                            "status": "cancelled",
                            "filled_size": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                    },
                },
            )
            now = datetime.utcnow()
            session.add(
                TradeSignal(
                    id="signal-2",
                    source="scanner",
                    signal_type="entry",
                    strategy_type="generic_strategy",
                    market_id="market-1",
                    direction="buy_yes",
                    entry_price=0.24,
                    dedupe_key="dedupe-signal-2",
                    payload_json={"token_id": "token-1"},
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                TraderOrder(
                    id="order-2",
                    trader_id="trader-1",
                    signal_id="signal-2",
                    source="scanner",
                    market_id="market-1",
                    direction="buy_yes",
                    mode="live",
                    status="cancelled",
                    notional_usd=0.0,
                    entry_price=0.0,
                    effective_price=0.0,
                    payload_json={
                        "token_id": "token-1",
                        "live_wallet_authority": {
                            "source": "live_trading_orders",
                            "token_id": "token-1",
                            "live_trading_order_id": "live-order-2",
                        },
                        "provider_reconciliation": {
                            "snapshot_status": "cancelled",
                            "snapshot": {
                                "normalized_status": "cancelled",
                                "status": "cancelled",
                                "filled_size": 0.0,
                                "filled_notional_usd": 0.0,
                            },
                        },
                    },
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                    verification_status="venue_order",
                    verification_source="live_order_ack",
                )
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            order.notional_usd = 0.0
            order.entry_price = 0.0
            order.effective_price = 0.0
            order.verification_status = "venue_order"
            order.verification_source = "live_order_ack"
            await session.commit()

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={"market-1": {"closed": False, "accepting_orders": True}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "asset_id": "token-1",
                            "token_id": "token-1",
                            "size": 37.5,
                            "avgPrice": 0.24,
                            "currentPrice": 0.24,
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1", "order-2"],
            )
            first = await session.get(TraderOrder, "order-1")
            second = await session.get(TraderOrder, "order-2")

            assert result["state_updates"] >= 1
            assert first is not None
            assert second is not None
            reopened_orders = [order for order in (first, second) if order.status == "open"]
            terminal_orders = [order for order in (first, second) if order.status == "cancelled"]
            assert len(reopened_orders) == 1
            assert len(terminal_orders) == 1
            assert reopened_orders[0].notional_usd == pytest.approx(9.0)
            assert terminal_orders[0].notional_usd == pytest.approx(0.0)
            assert (terminal_orders[0].payload_json or {}).get("wallet_position_reopen") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_does_not_reopen_verified_wallet_activity_close_for_partial_exit(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_loss",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 10.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 4.0,
                    },
                    "pending_live_exit": {
                        "status": "filled",
                        "exit_size": 10.0,
                        "filled_size": 2.0,
                        "fill_ratio": 0.2,
                        "close_trigger": "manual_mark_to_market",
                        "provider_clob_order_id": "0xexit",
                    },
                    "position_close": {
                        "close_price": 0.35,
                        "price_source": "wallet_activity",
                        "close_trigger": "wallet_activity",
                        "realized_pnl": -0.5,
                        "closed_at": "2026-03-02T13:00:00Z",
                        "wallet_activity_timestamp": "2026-03-02T12:58:00Z",
                        "wallet_activity_transaction_hash": "0xwalletclose",
                    },
                },
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            # Simulates a row that polymarket_trade_verifier already
            # matched to an on-chain SELL trade — actual_profit + the
            # full verification fields are written together so the
            # downstream reconcile path treats the close as already
            # verified rather than re-computing a baseline value.
            order.actual_profit = -0.5
            order.verification_status = "wallet_activity"
            order.verification_source = "polymarket_wallet_trades"
            order.verification_tx_hash = "0xwalletclose"
            order.verified_at = datetime.utcnow()
            await session.commit()

            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_buy_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            refreshed = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] == 0
            assert refreshed is not None
            assert refreshed.status == "closed_loss"
            assert refreshed.actual_profit == pytest.approx(-0.5)
            assert refreshed.verification_status == "wallet_activity"
            payload = dict(refreshed.payload_json or {})
            assert payload["position_close"]["wallet_activity_transaction_hash"] == "0xwalletclose"
            assert "reopened_at" not in payload["pending_live_exit"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_does_not_reopen_unfilled_failed_row_when_provider_cancelled_zero_fill(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="failed",
                payload_json={
                    "token_id": "token-1",
                    "live_market": {"selected_token_id": "token-1"},
                    "live_wallet_authority": {
                        "source": "live_trading_orders",
                        "live_trading_order_id": "live-order-1",
                        "wallet_address": "0xwallet",
                        "token_id": "token-1",
                    },
                    "provider_reconciliation": {
                        "snapshot_status": "cancelled",
                        "mapped_status": "cancelled",
                        "snapshot": {
                            "normalized_status": "cancelled",
                            "filled_size": 0.0,
                            "average_fill_price": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                    },
                },
            )
            row = await session.get(TraderOrder, "order-1")
            assert row is not None
            row.verification_status = "venue_order"
            row.error_message = "order recovery reported cancelled"
            await session.commit()

            monkeypatch.setattr(position_lifecycle, "load_market_info_for_orders", AsyncMock(return_value={}))
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"size": 5.95, "curPrice": 0.749}}),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(position_lifecycle, "_wallet_closed_positions_last_refresh_succeeded", False)
            monkeypatch.setattr(position_lifecycle, "_wallet_activity_last_refresh_succeeded", False)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            refreshed = await session.get(TraderOrder, "order-1")
            assert refreshed is not None
            assert refreshed.status == "failed"
            assert result["state_updates"] == 0
            assert any(detail.get("note") == "kept_terminal_row_specific_authority_missing" for detail in result["details"])
    finally:
        await engine.dispose()


def test_terminal_row_requires_reopen_audit_accepts_aware_datetimes():
    row = SimpleNamespace(
        updated_at=datetime.now(timezone.utc),
        executed_at=None,
        created_at=None,
        payload_json={
            "position_close": {"close_trigger": "wallet_absent_close"},
        },
    )

    assert position_lifecycle._terminal_row_requires_reopen_audit(
        row,
        datetime.now(timezone.utc).replace(tzinfo=None),
    )


def test_terminal_row_requires_reopen_audit_rejects_local_failed_submission_without_order_evidence():
    row = SimpleNamespace(
        status="failed",
        verification_status="local",
        provider_clob_order_id=None,
        provider_order_id=None,
        updated_at=datetime.now(timezone.utc),
        executed_at=None,
        created_at=None,
        payload_json={
            "provider_reconciliation": {
                "snapshot_status": "failed",
                "mapped_status": "failed",
                "snapshot": {"normalized_status": "failed"},
            },
        },
    )

    assert not position_lifecycle._terminal_row_requires_reopen_audit(
        row,
        datetime.now(timezone.utc).replace(tzinfo=None),
    )


def test_terminal_row_requires_reopen_audit_accepts_provider_order_evidence():
    row = SimpleNamespace(
        status="failed",
        verification_status="venue_order",
        provider_clob_order_id="0xabc",
        provider_order_id=None,
        updated_at=datetime.now(timezone.utc),
        executed_at=None,
        created_at=None,
        payload_json={
            "provider_reconciliation": {
                "snapshot_status": "failed",
                "mapped_status": "failed",
                "snapshot": {"normalized_status": "failed", "clob_order_id": "0xabc"},
            },
        },
    )

    assert position_lifecycle._terminal_row_requires_reopen_audit(
        row,
        datetime.now(timezone.utc).replace(tzinfo=None),
    )


@pytest.mark.asyncio
async def test_reconcile_does_not_infer_resolution_on_tradable_market(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(session, direction="buy_yes")
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_shadow_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
    finally:
        await engine.dispose()


def test_price_inference_requires_terminal_market_signal():
    market_info = {
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "enable_order_book": True,
        "resolved": None,
        "winner": None,
        "winning_outcome": None,
        "end_date": (datetime.utcnow() - timedelta(minutes=30)).isoformat() + "Z",
        "outcome_prices": [0.0035, 0.9965],
    }

    inferred = position_lifecycle._extract_winning_outcome_index_from_prices(
        market_info,
        market_tradable=False,
        settle_floor=0.98,
    )

    assert inferred is None


@pytest.mark.asyncio
async def test_load_market_info_prefers_force_refreshed_condition_lookup(monkeypatch):
    market_id = "0x67ac92b23bf20382bff35cf4519403d3aab4b2015dd2e3943b6a511e32f1687e"
    stale_info = {
        "condition_id": market_id,
        "closed": False,
        "active": True,
        "outcomes": ["Up", "Down"],
        "outcome_prices": [0.52, 0.48],
    }
    fresh_info = {
        "condition_id": market_id,
        "closed": True,
        "active": True,
        "accepting_orders": False,
        "outcomes": ["Up", "Down"],
        "outcome_prices": [1.0, 0.0],
    }

    async def _condition_lookup(_market_id: str, **kwargs):
        if kwargs.get("force_refresh"):
            return fresh_info
        return stale_info

    condition_mock = AsyncMock(side_effect=_condition_lookup)
    token_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_condition_id", condition_mock)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_token_id", token_mock)

    result = await position_lifecycle.load_market_info_for_orders([SimpleNamespace(market_id=market_id)])

    assert result[market_id] == {**fresh_info, "id": market_id, "market_id": market_id}
    condition_mock.assert_awaited_once_with(market_id, force_refresh=True)
    assert token_mock.await_count == 0


@pytest.mark.asyncio
async def test_load_market_info_falls_back_to_cached_condition_lookup_when_refresh_misses(monkeypatch):
    market_id = "0x23e435a43d5304be68ddeffc53a1ad57163407672e37456c3a29df4c6c8dbe5b"
    cached_info = {
        "condition_id": market_id,
        "closed": False,
        "active": True,
        "outcomes": ["Up", "Down"],
        "outcome_prices": [0.61, 0.39],
    }

    async def _condition_lookup(_market_id: str, **kwargs):
        if kwargs.get("force_refresh"):
            return None
        return cached_info

    condition_mock = AsyncMock(side_effect=_condition_lookup)
    token_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_condition_id", condition_mock)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_token_id", token_mock)

    result = await position_lifecycle.load_market_info_for_orders([SimpleNamespace(market_id=market_id)])

    assert result[market_id] == {**cached_info, "id": market_id, "market_id": market_id}
    assert condition_mock.await_count == 2
    first_call, second_call = condition_mock.await_args_list
    assert first_call.args == (market_id,)
    assert first_call.kwargs == {"force_refresh": True}
    assert second_call.args == (market_id,)
    assert second_call.kwargs == {}
    assert token_mock.await_count == 0


@pytest.mark.asyncio
async def test_live_mark_to_market_keeps_position_open_and_records_pending_exit(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 100.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 40.0,
                    }
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.55, 0.45],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                force_mark_to_market=True,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["would_close"] == 1
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
            assert order.actual_profit is None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("close_trigger") == "manual_mark_to_market"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reconcile_does_not_infer_resolution_for_ended_but_active_market(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                direction="buy_no",
                mode="live",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.875,
                        "filled_notional_usd": 4.375,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "active": True,
                            "closed": False,
                            "accepting_orders": True,
                            "enable_order_book": True,
                            "resolved": None,
                            "winner": None,
                            "winning_outcome": None,
                            "end_date": (datetime.utcnow() - timedelta(minutes=30)).isoformat() + "Z",
                            "outcome_prices": [0.0035, 0.9965],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.9965}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
            assert (order.payload_json or {}).get("position_close") is None
            assert (order.payload_json or {}).get("pending_live_exit") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reconcile_closes_post_end_extreme_mark_without_fresh_market_snapshot(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        stale_marked_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
        ended_at = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat().replace("+00:00", "Z")
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "position_state": {
                        "last_mark_price": 0.0005,
                        "last_mark_source": "position_state_mark",
                        "last_marked_at": stale_marked_at,
                    },
                    "provider_reconciliation": {
                        "filled_size": 100.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 40.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={"market-1": {"market_id": "market-1", "end_date": ended_at}}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "is_market_tradable", lambda *_args, **_kwargs: False)
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_positions_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_closed_positions_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_sell_trades_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_close_activity_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "resolved_loss"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "resolution_extreme_mark"
            assert (order.payload_json or {}).get("position_close", {}).get("close_price") == 0.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reconcile_uses_redeemable_wallet_position_without_mark_price(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        ended_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "token_id": "winner-token",
                    "provider_reconciliation": {
                        "filled_size": 100.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 40.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={"market-1": {"market_id": "market-1", "end_date": ended_at}}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "is_market_tradable", lambda *_args, **_kwargs: False)
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "winner-token": {
                            "token_id": "winner-token",
                            "redeemable": True,
                            "counts_as_open": False,
                            "size": 100.0,
                            "avgPrice": 0.4,
                            "currentPrice": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_closed_positions_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_sell_trades_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_close_activity_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "resolved_win"
            assert (order.payload_json or {}).get("position_close", {}).get("price_source") == "wallet_redeemable_mark"
            assert (order.payload_json or {}).get("position_close", {}).get("close_price") == 1.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_does_not_resolve_until_wallet_position_is_terminal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 0,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "size": 5.0,
                            "curPrice": 0.9995,
                            "redeemable": False,
                            "counts_as_open": True,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_does_not_resolve_on_wallet_absence_without_terminal_wallet_state(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 0,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reopens_resolved_settlement_when_wallet_position_is_not_terminal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="resolved_win",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                    "position_close": {
                        "close_trigger": "resolution_inferred",
                        "price_source": "resolved_settlement",
                        "realized_pnl": 3.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 0,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "size": 5.0,
                            "curPrice": 0.9995,
                            "redeemable": False,
                            "counts_as_open": True,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] >= 1
            assert order is not None
            assert order.status == "open"
            assert (order.payload_json or {}).get("resolution_reopen", {}).get("reopen_reason") == (
                "resolved_settlement_not_wallet_terminal"
            )
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_external_wallet_flatten_closes_position_from_wallet_trade(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"take_profit_pct": 1.0},
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 0.0}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "trade_id": "trade-1",
                            "token_id": "token-1",
                            "size": 5.0,
                            "price": 0.21,
                            "timestamp": datetime.utcnow(),
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "closed_loss"
            position_close = (order.payload_json or {}).get("position_close", {})
            assert position_close.get("close_trigger") == "external_wallet_flatten"
            assert position_close.get("price_source") == "wallet_trade"
            assert position_close.get("wallet_trade_id") == "trade-1"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_external_wallet_flatten_uses_wallet_buy_history_per_order(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_clob_order_id": "clob-order-1",
                    "strategy_exit_config": {"take_profit_pct": 1.0},
                },
            )
            base_now = datetime.utcnow() - timedelta(minutes=5)
            order_1 = await session.get(TraderOrder, "order-1")
            signal_1 = await session.get(TradeSignal, "signal-1")
            assert order_1 is not None
            assert signal_1 is not None
            order_1.created_at = base_now
            order_1.executed_at = base_now
            order_1.updated_at = base_now
            signal_1.created_at = base_now
            signal_1.updated_at = base_now

            second_created_at = base_now + timedelta(minutes=1)
            session.add(
                TradeSignal(
                    id="signal-2",
                    source="crypto",
                    signal_type="entry",
                    strategy_type="crypto_15m",
                    market_id="market-1",
                    direction="buy_yes",
                    entry_price=0.5,
                    dedupe_key="dedupe-signal-2",
                    payload_json={"yes_price": 0.5, "no_price": 0.5},
                    created_at=second_created_at,
                    updated_at=second_created_at,
                )
            )
            session.add(
                TraderOrder(
                    id="order-2",
                    trader_id="trader-1",
                    signal_id="signal-2",
                    source="crypto",
                    market_id="market-1",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=40.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={
                        "token_id": "token-1",
                        "provider_clob_order_id": "clob-order-2",
                        "strategy_exit_config": {"take_profit_pct": 1.0},
                    },
                    created_at=second_created_at,
                    executed_at=second_created_at,
                    updated_at=second_created_at,
                )
            )
            await session.commit()

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 0.0}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_buy_trades_by_token",
                AsyncMock(
                    return_value={
                        "token-1": [
                            {
                                "token_id": "token-1",
                                "size": 2.0,
                                "price": 0.4,
                                "timestamp": base_now + timedelta(seconds=5),
                            },
                            {
                                "token_id": "token-1",
                                "size": 3.0,
                                "price": 0.5,
                                "timestamp": second_created_at + timedelta(seconds=5),
                            },
                        ]
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "trade_id": "trade-1",
                            "token_id": "token-1",
                            "size": 5.0,
                            "price": 0.6,
                            "timestamp": second_created_at + timedelta(minutes=1),
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )

            refreshed_order_1 = await session.get(TraderOrder, "order-1")
            refreshed_order_2 = await session.get(TraderOrder, "order-2")

            assert result["closed"] == 2
            assert refreshed_order_1 is not None
            assert refreshed_order_2 is not None
            assert refreshed_order_1.status == "closed_win"
            assert refreshed_order_2.status == "closed_win"
            assert refreshed_order_1.actual_profit == pytest.approx(0.4)
            assert refreshed_order_2.actual_profit == pytest.approx(0.3)
            assert refreshed_order_1.notional_usd == pytest.approx(0.8)
            assert refreshed_order_2.notional_usd == pytest.approx(1.5)
            assert (refreshed_order_1.payload_json or {}).get("position_close", {}).get("wallet_trade_size") == pytest.approx(2.0)
            assert (refreshed_order_2.payload_json or {}).get("position_close", {}).get("wallet_trade_size") == pytest.approx(3.0)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_terminal_wallet_trade_close_reopens_for_entry_fill_repair(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_loss",
                payload_json={
                    "token_id": "token-1",
                    "provider_clob_order_id": "clob-order-1",
                    "live_wallet_authority": {
                        "source": "live_trading_orders",
                        "token_id": "token-1",
                        "live_trading_order_id": "live-order-1",
                    },
                    "provider_reconciliation": {
                        "snapshot_status": "placing",
                        "snapshot": {
                            "normalized_status": "placing",
                            "status": "cancelled",
                            "filled_size": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                    },
                    "position_close": {
                        "close_price": 0.6,
                        "price_source": "wallet_trade",
                        "close_trigger": "external_wallet_flatten",
                        "realized_pnl": -39.2,
                        "filled_size": 80.0,
                        "wallet_trade_id": "trade-close-1",
                    },
                },
            )
            repaired_order = await session.get(TraderOrder, "order-1")
            assert repaired_order is not None
            repaired_order.notional_usd = 40.0
            repaired_order.actual_profit = -39.2
            repaired_order.updated_at = datetime.utcnow() - timedelta(minutes=1)
            await session.commit()

            created_at = repaired_order.created_at
            assert created_at is not None
            created_at_naive = created_at.astimezone(timezone.utc).replace(tzinfo=None) if created_at.tzinfo else created_at

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.55, 0.45],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_buy_trades_by_token",
                AsyncMock(
                    return_value={
                        "token-1": [
                            {
                                "token_id": "token-1",
                                "size": 2.0,
                                "price": 0.4,
                                "timestamp": created_at_naive + timedelta(seconds=5),
                                "transactionHash": "0xbuy-fill-1",
                            }
                        ]
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "trade_id": "trade-close-1",
                            "token_id": "token-1",
                            "size": 2.0,
                            "price": 0.6,
                            "timestamp": created_at_naive + timedelta(minutes=5),
                            "transactionHash": "0xsell-fill-1",
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            repaired_order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result["state_updates"] >= 1
            assert repaired_order is not None
            assert repaired_order.status == "closed_win"
            assert repaired_order.notional_usd == pytest.approx(0.8)
            assert repaired_order.actual_profit == pytest.approx(0.4)
            assert (repaired_order.payload_json or {}).get("entry_fill_recovery", {}).get("source") == "wallet_trade_history"
            assert (repaired_order.payload_json or {}).get("wallet_entry_fill_repair", {}).get("reopen_reason") == (
                "wallet_trade_entry_fill_repair"
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_exit_submission_uses_gtc_for_lifecycle_close(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"take_profit_pct": 1.0},
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.5, 0.5],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.5}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="mock_exit_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "failed"
            assert execute_mock.await_count == 1
            assert execute_mock.await_args.kwargs.get("time_in_force") == "GTC"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_exit_submission_uses_ioc_for_rapid_strategy_close(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "strategy_type": "btc_eth_maker_quote",
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.5, 0.5],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.5}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )

            class _RapidStrategy:
                def should_exit(self, _position, _market_state):
                    return ExitDecision(action="close", reason="Rapid hard-cap take profit (23.8% >= 18.0%)")

            monkeypatch.setattr(
                "services.strategy_loader.strategy_loader.get_strategy",
                lambda _slug: SimpleNamespace(instance=_RapidStrategy()),
            )

            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="mock_exit_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "failed"
            assert execute_mock.await_count == 1
            assert execute_mock.await_args.kwargs.get("time_in_force") == "IOC"
            assert execute_mock.await_args.kwargs.get("resolve_live_price") is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_failed_rapid_exit_retry_uses_ioc_and_does_not_block_min_notional(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            last_attempt_at = (datetime.utcnow() - timedelta(seconds=30)).isoformat() + "Z"
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 3.33,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 1.332,
                    },
                    "pending_live_exit": {
                        "status": "failed",
                        "close_trigger": "strategy:Rapid hard-cap take profit (23.8% >= 18.0%)",
                        "close_price": 0.7,
                        "exit_size": 3.33,
                        "filled_size": 3.1,
                        "retry_count": 0,
                        "last_attempt_at": last_attempt_at,
                        "last_error": "mock_prior_failure",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.7, 0.3],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="retry_mock_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "failed"
            assert pending_exit.get("last_error") == "retry_mock_failure"
            assert execute_mock.await_count == 1
            assert execute_mock.await_args.kwargs.get("time_in_force") == "IOC"
            assert execute_mock.await_args.kwargs.get("resolve_live_price") is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_failed_exit_does_not_resolve_on_wallet_absence(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            last_attempt_at = (datetime.utcnow() - timedelta(seconds=30)).isoformat() + "Z"
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 3.33,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 1.332,
                    },
                    "pending_live_exit": {
                        "status": "failed",
                        "close_trigger": "strategy:Stop loss hit",
                        "close_price": 0.2,
                        "exit_size": 3.33,
                        "retry_count": 0,
                        "last_attempt_at": last_attempt_at,
                        "last_error": "mock_prior_failure",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 0,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "prepare_sell_balance_allowance",
                AsyncMock(return_value=None),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="retry_mock_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert execute_mock.await_count == 1
            assert order is not None
            assert order.status == "open"
            assert (order.payload_json or {}).get("position_close") is None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "failed"
            assert pending_exit.get("last_error") == "retry_mock_failure"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_partial_fill_does_not_close_position(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "submitted"
            assert pending_exit.get("fill_ratio") == pytest.approx(6.22 / 7.4, rel=1e-9)
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_terminal_partial_fill_marks_failed_for_retry_when_wallet_still_open(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 1.18, "curPrice": 0.18}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )
            execute_mock = AsyncMock()
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "failed"
            assert str(pending_exit.get("last_error") or "").startswith("provider_partial_fill_terminal")
            assert pending_exit.get("next_retry_at")
            assert execute_mock.await_count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_failed_exit_retry_exhausted_soft_bypass_retries_when_wallet_still_open(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            stale_attempt_at = (datetime.utcnow() - timedelta(minutes=2)).isoformat() + "Z"
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 10.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 4.0,
                    },
                    "pending_live_exit": {
                        "status": "failed",
                        "close_trigger": "strategy:Stop loss hit",
                        "close_price": 0.2,
                        "exit_size": 10.0,
                        "retry_count": 5,
                        "last_attempt_at": stale_attempt_at,
                        "last_error": "mock_retry_exhausted",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 2.0, "curPrice": 0.2}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "prepare_sell_balance_allowance",
                AsyncMock(return_value=None),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="submitted",
                    order_id="exit-retry-1",
                    payload={"clob_order_id": "0xretry-order", "trading_status": "open"},
                    error_message=None,
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert execute_mock.await_count == 1
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "submitted"
            assert pending_exit.get("provider_clob_order_id") == "0xretry-order"
            assert pending_exit.get("retry_count") == 5
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_stale_mark_does_not_trigger_max_hold_exit(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            stale_marked_at = (datetime.utcnow() - timedelta(minutes=4)).isoformat() + "Z"
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"max_hold_minutes": 0.0},
                    "position_state": {
                        "last_mark_price": 0.45,
                        "last_mark_source": "clob_midpoint",
                        "last_marked_at": stale_marked_at,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [],
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="mock_exit_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "executed"
            assert (order.payload_json or {}).get("position_close") is None
            assert execute_mock.await_count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_fresh_mark_allows_max_hold_exit_attempt(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            fresh_marked_at = datetime.utcnow().isoformat() + "Z"
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"max_hold_minutes": 0.0},
                    "position_state": {
                        "last_mark_price": 0.45,
                        "last_mark_source": "clob_midpoint",
                        "last_marked_at": fresh_marked_at,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="mock_exit_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "failed"
            assert pending_exit.get("close_trigger") == "max_hold"
            assert execute_mock.await_count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_stop_loss_does_not_exit_when_market_not_tradable(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            fresh_marked_at = datetime.utcnow().isoformat() + "Z"
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"stop_loss_pct": 5.0},
                    "position_state": {
                        "last_mark_price": 0.2,
                        "last_mark_source": "clob_midpoint",
                        "last_marked_at": fresh_marked_at,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": False,
                            "enable_order_book": False,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.2, 0.8],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 100.0, "curPrice": 0.2}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="should_not_submit",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "executed"
            assert (order.payload_json or {}).get("pending_live_exit") is None
            assert execute_mock.await_count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_does_not_close_when_fill_threshold_met_but_provider_open(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "open",
                            "filled_size": 7.4,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "submitted"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_blocked_retry_exhausted_resolves_when_market_is_terminal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                    "pending_live_exit": {
                        "status": "blocked_retry_exhausted",
                        "provider_status": "open",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "size": 5.0,
                            "curPrice": 1.0,
                            "redeemable": True,
                            "counts_as_open": False,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                force_mark_to_market=True,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "resolved_win"
            assert (order.payload_json or {}).get("superseded_pending_exit", {}).get("status") == "superseded_manual_sell"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_blocked_retry_exhausted_closes_from_post_end_extreme_mark(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        ended_at = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat().replace("+00:00", "Z")
        stale_marked_at = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat().replace("+00:00", "Z")
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                    "position_state": {
                        "last_mark_price": 0.0005,
                        "last_mark_source": "market_mark",
                        "last_marked_at": stale_marked_at,
                    },
                    "pending_live_exit": {
                        "status": "blocked_retry_exhausted",
                        "provider_status": "open",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={"market-1": {"market_id": "market-1", "end_date": ended_at}}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "is_market_tradable", lambda *_args, **_kwargs: False)
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_positions_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "resolved_loss"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "resolution_extreme_mark"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_unfilled_terminal_order_is_cancelled(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 0.0,
                        "average_fill_price": 0.0,
                        "filled_notional_usd": 0.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.0, 1.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                force_mark_to_market=True,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "cancelled"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "terminal_unfilled_cancel"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_placing_unfilled_terminal_order_is_cancelled(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="placing",
                payload_json={
                    "token_id": "token-1",
                    "live_wallet_authority": {
                        "source": "live_trading_orders",
                        "token_id": "token-1",
                        "live_trading_order_id": "live-order-1",
                    },
                    "provider_reconciliation": {
                        "snapshot_status": "placing",
                        "mapped_status": "placing",
                        "snapshot": {
                            "normalized_status": "placing",
                            "status": "cancelled",
                            "filled_size": 0.0,
                            "average_fill_price": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                        "filled_size": 0.0,
                        "average_fill_price": 0.0,
                        "filled_notional_usd": 0.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.0, 1.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                force_mark_to_market=True,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["skipped_reasons"]["unfilled_provider_failure"] == 1
            assert order is not None
            assert order.status == "cancelled"
            assert (order.payload_json or {}).get("provider_failure_finalized", {}).get("provider_status") == "cancelled"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_exit_blocks_no_inventory_after_pre_submit_gate_failure(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"max_hold_minutes": 0.0},
                    "provider_reconciliation": {
                        "filled_size": 11.32,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 4.528,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message=(
                        "SELL pre-submit gate failed: not enough conditional token balance/allowance. "
                        "token_id=token-1 required_shares=11.32 available_shares=0 shortfall_shares=11.32 "
                        "balance_shares=0 allowance_shares=1 signature_type=1 funder_wallet=0xabc"
                    ),
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "blocked_no_inventory"
            assert pending_exit.get("next_retry_at") is None
            assert pending_exit.get("retry_count") == 1
            assert pending_exit.get("last_error", "")
            assert execute_mock.await_count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_blocked_no_inventory_does_not_resubmit_exit(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "strategy_exit_config": {"max_hold_minutes": 0.0},
                    "provider_reconciliation": {
                        "filled_size": 11.32,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 4.528,
                    },
                    "pending_live_exit": {
                        "status": "blocked_no_inventory",
                        "close_trigger": "strategy:Smart take profit near max",
                        "close_price": 0.45,
                        "exit_size": 11.32,
                        "retry_count": 1,
                        "last_error": "available_shares=0 balance_shares=0",
                        "last_attempt_at": "2099-01-01T00:00:00Z",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock()
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "blocked_no_inventory"
            assert execute_mock.await_count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reconcile_forces_flatten_when_full_bundle_position_is_incomplete(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-yes",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            signal = await session.get(TradeSignal, "signal-1")
            assert signal is not None
            signal.payload_json = {
                "is_guaranteed": True,
                "positions_to_take": [
                    {"action": "BUY", "outcome": "YES", "token_id": "token-yes", "market": "Will Team A win?"},
                    {"action": "BUY", "outcome": "NO", "token_id": "token-no", "market": "Will Team A win?"},
                ],
                "execution_plan": {
                    "plan_id": "plan-1",
                    "metadata": {
                        "full_bundle_execution_required": True,
                        "full_bundle_execution_mode": "live_ioc_complete_or_flatten",
                    },
                    "legs": [
                        {"leg_id": "leg-yes", "market_id": "market-1", "token_id": "token-yes", "side": "buy", "outcome": "yes"},
                        {"leg_id": "leg-no", "market_id": "market-1", "token_id": "token-no", "side": "buy", "outcome": "no"},
                    ],
                },
            }
            await session.commit()
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.46, 0.54],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-yes": {"asset": "token-yes", "size": 5.0, "curPrice": 0.46}}),
            )
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_sell_trades_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_close_activity_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle.live_execution_service, "prepare_sell_balance_allowance", AsyncMock(return_value=None))
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="submitted",
                    error_message=None,
                    order_id="exit-order-1",
                    payload={"clob_order_id": "exit-clob-1"},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["would_close"] >= 1
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            bundle_state = (order.payload_json or {}).get("bundle_execution_state", {})
            assert pending_exit.get("close_trigger") == "force_flatten_bundle_residual"
            assert pending_exit.get("status") == "submitted"
            assert bundle_state.get("missing_token_ids") == ["token-no"]
            assert execute_mock.await_count == 1
            assert execute_mock.await_args.kwargs.get("time_in_force") == "IOC"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reconcile_reopens_cancelled_wallet_position_and_flattens_bundle_residual(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="cancelled",
                payload_json={
                    "token_id": "token-yes",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            signal = await session.get(TradeSignal, "signal-1")
            assert signal is not None
            signal.payload_json = {
                "is_guaranteed": True,
                "positions_to_take": [
                    {"action": "BUY", "outcome": "YES", "token_id": "token-yes", "market": "Will Team A win?"},
                    {"action": "BUY", "outcome": "NO", "token_id": "token-no", "market": "Will Team A win?"},
                ],
                "execution_plan": {
                    "plan_id": "plan-1",
                    "metadata": {
                        "full_bundle_execution_required": True,
                        "full_bundle_execution_mode": "live_ioc_complete_or_flatten",
                    },
                    "legs": [
                        {"leg_id": "leg-yes", "market_id": "market-1", "token_id": "token-yes", "side": "buy", "outcome": "yes"},
                        {"leg_id": "leg-no", "market_id": "market-1", "token_id": "token-no", "side": "buy", "outcome": "no"},
                    ],
                },
            }
            await session.commit()
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.46, 0.54],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-yes": {"asset": "token-yes", "size": 5.0, "curPrice": 0.46}}),
            )
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_sell_trades_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle, "_load_execution_wallet_recent_close_activity_by_token", AsyncMock(return_value={}))
            monkeypatch.setattr(position_lifecycle.live_execution_service, "prepare_sell_balance_allowance", AsyncMock(return_value=None))
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="submitted",
                    error_message=None,
                    order_id="exit-order-2",
                    payload={"clob_order_id": "exit-clob-2"},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] >= 1
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("close_trigger") == "force_flatten_bundle_residual"
            assert pending_exit.get("status") == "submitted"
            assert execute_mock.await_count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_closes_when_provider_filled_and_wallet_flat_with_lower_fill_ratio(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "other-token": {"asset": "other-token", "size": 1.0, "curPrice": 0.5},
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "closed_loss"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "filled"
            assert pending_exit.get("allow_partial_fill_terminal") is True
            assert (order.payload_json or {}).get("position_close", {}).get("price_source") == "provider_exit_fill"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_does_not_close_on_provider_exit_fill_until_wallet_is_flat(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Max hold exceeded",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {"asset": "token-1", "size": 1.18, "curPrice": 0.18},
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 7.4,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )
            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            assert order.status == "open"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_does_not_resolve_on_wallet_absence(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 5.0,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 0,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            assert (order.payload_json or {}).get("position_close") is None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "submitted"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reopens_terminal_provider_exit_fill_when_wallet_still_holds_position(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_loss",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "filled",
                        "close_trigger": "strategy:Max hold exceeded",
                        "exit_size": 7.4,
                        "filled_size": 7.4,
                        "fill_ratio": 1.0,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                    "position_close": {
                        "close_trigger": "strategy:Max hold exceeded",
                        "price_source": "provider_exit_fill",
                        "filled_size": 7.4,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {"asset": "token-1", "size": 1.18, "curPrice": 0.18},
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 7.4,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "submitted"
            assert pending_exit.get("reopen_reason") == "provider_exit_fill_not_wallet_flat"
            assert (order.payload_json or {}).get("provider_exit_reopen", {}).get("reopen_reason") == (
                "provider_exit_fill_not_wallet_flat"
            )
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reopens_terminal_order_when_exit_fill_is_partial(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_loss",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "filled",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "filled_size": 6.22,
                        "fill_ratio": 6.22 / 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                    "position_close": {
                        "close_trigger": "strategy:Stop loss hit",
                        "filled_size": 6.22,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "size": 5.0,
                            "curPrice": 1.0,
                            "redeemable": True,
                            "counts_as_open": False,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("reopen_reason") == "partial_exit_fill_below_threshold"
            assert pending_exit.get("status") == "submitted"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_does_not_close_when_provider_exit_status_is_open_even_if_fill_ratio_high(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 47.61,
                        "average_fill_price": 0.1,
                        "filled_notional_usd": 4.761,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Trailing stop",
                        "exit_size": 47.61,
                        "filled_size": 47.0,
                        "fill_ratio": 47.0 / 47.61,
                        "provider_status": "open",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 0.61, "curPrice": 0.06}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "open",
                            "filled_size": 47.0,
                            "average_fill_price": 0.06,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "submitted"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_price_uses_wallet_mark_when_market_price_unavailable(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.47}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            signal = await session.get(TradeSignal, "signal-1")
            assert signal is not None
            signal.payload_json = {}
            await session.commit()

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            position_state = (order.payload_json or {}).get("position_state", {})
            assert position_state.get("last_mark_source") == "wallet_mark"
            assert position_state.get("last_mark_price") == pytest.approx(0.47, abs=1e-9)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_shadow_strategy_reverse_intent_emits_reverse_signal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                payload_json={
                    "strategy_type": "reverse_test",
                    "strategy_context": {},
                },
            )

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.28, 0.72],
                            "token_ids": ["yes-token", "no-token"],
                            "outcomes": ["Yes", "No"],
                        }
                    }
                ),
            )

            class _ReverseExitStrategy:
                def should_exit(self, position, market_state):
                    return ExitDecision(
                        "close",
                        "Velocity guard",
                        close_price=0.28,
                        payload={
                            "reverse_intent": {
                                "enabled": True,
                                "direction": "buy_no",
                                "entry_price": 0.72,
                                "signal_type": "crypto_worker_reverse",
                                "confidence": 0.68,
                                "edge_percent": 2.8,
                                "size_multiplier": 1.2,
                                "strategy_type": "reverse_test",
                                "expires_in_seconds": 120,
                            }
                        },
                    )

            monkeypatch.setattr(
                "services.strategy_loader.strategy_loader.get_strategy",
                lambda strategy_key: (
                    SimpleNamespace(instance=_ReverseExitStrategy())
                    if str(strategy_key or "").strip().lower() == "reverse_test"
                    else None
                ),
            )

            upsert_mock = AsyncMock(return_value=SimpleNamespace(id="reverse-signal-paper-1"))
            publish_mock = AsyncMock(return_value=None)
            monkeypatch.setattr(position_lifecycle, "upsert_trade_signal", upsert_mock)
            monkeypatch.setattr("services.event_bus.event_bus.publish", publish_mock)

            result = await position_lifecycle.reconcile_shadow_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result.get("reverse_signals_emitted") == 1
            assert upsert_mock.await_count == 1
            kwargs = upsert_mock.await_args.kwargs
            assert kwargs["source"] == "crypto"
            assert kwargs["direction"] == "buy_no"
            assert kwargs["strategy_type"] == "reverse_test"

            assert order is not None
            pending_reverse = (order.payload_json or {}).get("pending_reverse_entry", {})
            assert pending_reverse.get("status") == "emitted"
            assert pending_reverse.get("signal_id") == "reverse-signal-paper-1"
            assert any(call.args and call.args[0] == "trade_signal_batch" for call in publish_mock.await_args_list)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_fill_emits_reverse_signal_when_armed(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "strategy_type": "reverse_test",
                    "strategy_context": {},
                    "token_id": "yes-token",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Velocity guard",
                        "close_price": 0.20,
                        "exit_size": 5.0,
                        "provider_clob_order_id": "0xexit-order-1",
                    },
                    "pending_reverse_entry": {
                        "status": "armed",
                        "direction": "buy_no",
                        "entry_price": 0.72,
                        "signal_type": "crypto_worker_reverse",
                        "confidence": 0.66,
                        "edge_percent": 2.4,
                        "size_multiplier": 1.1,
                        "strategy_type": "reverse_test",
                        "armed_at": "2099-01-01T00:00:00Z",
                        "attempt_index": 1,
                        "expires_at": "2099-01-01T00:05:00Z",
                    },
                },
            )

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.20, 0.80],
                            "token_ids": ["yes-token", "no-token"],
                            "outcomes": ["Yes", "No"],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"yes-token": {"asset": "yes-token", "size": 0.0}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order-1": {
                            "normalized_status": "filled",
                            "filled_size": 5.0,
                            "average_fill_price": 0.20,
                        }
                    }
                ),
            )

            upsert_mock = AsyncMock(return_value=SimpleNamespace(id="reverse-signal-live-1"))
            publish_mock = AsyncMock(return_value=None)
            monkeypatch.setattr(position_lifecycle, "upsert_trade_signal", upsert_mock)
            monkeypatch.setattr("services.event_bus.event_bus.publish", publish_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result.get("reverse_signals_emitted") == 1
            assert upsert_mock.await_count == 1
            kwargs = upsert_mock.await_args.kwargs
            assert kwargs["source"] == "crypto"
            assert kwargs["direction"] == "buy_no"
            assert kwargs["strategy_type"] == "reverse_test"

            assert order is not None
            assert order.status == "closed_loss"
            pending_reverse = (order.payload_json or {}).get("pending_reverse_entry", {})
            assert pending_reverse.get("status") == "emitted"
            assert pending_reverse.get("signal_id") == "reverse-signal-live-1"
            verification_events = (
                await session.execute(
                    select(TraderOrderVerificationEvent).where(
                        TraderOrderVerificationEvent.trader_order_id == "order-1"
                    )
                )
            ).scalars().all()
            assert verification_events
            assert any(call.args and call.args[0] == "trade_signal_batch" for call in publish_mock.await_args_list)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_skips_wallet_absent_close_when_provider_entry_still_open(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 12.36,
                        "average_fill_price": 0.29,
                        "filled_notional_usd": 3.5844,
                        "snapshot": {
                            "normalized_status": "open",
                            "size": 12.64,
                            "filled_size": 12.36,
                            "average_fill_price": 0.29,
                            "filled_notional_usd": 3.5844,
                        },
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1"],
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "open"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_finalizes_provider_snapshot_without_fill_or_wallet_position(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                direction="hold",
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 0.0,
                        "average_fill_price": 0.0,
                        "filled_notional_usd": 0.0,
                        "snapshot": {
                            "normalized_status": "filled",
                            "status": "filled",
                            "filled_size": 0.0,
                            "average_fill_price": 0.0,
                            "filled_notional_usd": 0.0,
                        },
                    },
                },
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            order.notional_usd = None
            order.entry_price = 0.91
            order.effective_price = 0.91
            await session.commit()
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_closed_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_close_activity_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1"],
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result["would_close"] == 1
            assert order is not None
            assert order.status == "cancelled"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == (
                "terminal_unfilled_provider_snapshot"
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_holds_wallet_absent_position_without_verified_close_evidence(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="executed",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 12.36,
                        "average_fill_price": 0.29,
                        "filled_notional_usd": 3.5844,
                        "snapshot": {
                            "normalized_status": "filled",
                            "size": 12.36,
                            "filled_size": 12.36,
                            "average_fill_price": 0.29,
                            "filled_notional_usd": 3.5844,
                        },
                    },
                },
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            aged = datetime.now(timezone.utc) - timedelta(minutes=5)
            order.notional_usd = 3.5844
            order.entry_price = 0.29
            order.effective_price = 0.29
            order.created_at = aged
            order.executed_at = aged
            order.updated_at = aged
            await session.commit()
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1"],
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["would_close"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
            assert order.actual_profit is None
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_reopens_wallet_absent_terminal_when_provider_entry_is_working(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_win",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 12.36,
                        "average_fill_price": 0.29,
                        "filled_notional_usd": 3.5844,
                        "snapshot": {
                            "normalized_status": "open",
                            "size": 12.64,
                            "filled_size": 12.36,
                            "average_fill_price": 0.29,
                            "filled_notional_usd": 3.5844,
                        },
                    },
                    "position_close": {
                        "close_price": 0.295,
                        "price_source": "wallet_absent_best_available",
                        "close_trigger": "wallet_absent_close",
                        "realized_pnl": 0.0618,
                        "filled_size": 12.36,
                        "closed_at": "2099-01-01T00:00:00Z",
                        "reason": "reconciliation_worker",
                    },
                },
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            order.actual_profit = 0.0618
            await session.commit()

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1"],
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            assert order.actual_profit is None
            assert (order.payload_json or {}).get("position_close") is None
            assert (order.payload_json or {}).get("wallet_absent_reopen", {}).get("reopen_reason") == (
                "provider_entry_order_still_open"
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_positions_reopens_wallet_absent_terminal_when_wallet_position_exists(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_loss",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 12.36,
                        "average_fill_price": 0.29,
                        "filled_notional_usd": 3.5844,
                        "snapshot": {
                            "normalized_status": "filled",
                            "size": 12.36,
                            "filled_size": 12.36,
                            "average_fill_price": 0.29,
                            "filled_notional_usd": 3.5844,
                        },
                    },
                    "position_close": {
                        "close_price": 0.295,
                        "price_source": "wallet_absent_best_available",
                        "close_trigger": "wallet_absent_close",
                        "realized_pnl": -0.0618,
                        "filled_size": 12.36,
                        "closed_at": "2099-01-01T00:00:00Z",
                        "reason": "reconciliation_worker",
                    },
                },
            )
            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            order.actual_profit = -0.0618
            await session.commit()

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "asset": "token-1",
                            "asset_id": "token-1",
                            "size": 12.36,
                            "avgPrice": 0.29,
                            "currentPrice": 0.295,
                            "conditionId": "condition-1",
                            "outcomeIndex": 0,
                        }
                    }
                ),
            )
            monkeypatch.setattr(position_lifecycle, "_wallet_positions_last_refresh_succeeded", True)
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(position_lifecycle.polymarket_client, "get_midpoint", AsyncMock(return_value=None))

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                order_ids=["order-1"],
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            assert order.actual_profit is None
            assert (order.payload_json or {}).get("position_close") is None
            assert (order.payload_json or {}).get("wallet_absent_reopen", {}).get("reopen_reason") == (
                "fallback_close_not_verified"
            )
    finally:
        await engine.dispose()
