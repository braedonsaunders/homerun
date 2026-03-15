import sys
from pathlib import Path
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, LiveTradingOrder, Trader, TraderOrder
from services import trader_orchestrator_state
from workers import trader_reconciliation_worker
from services.trader_orchestrator_state import (
    cleanup_trader_open_orders,
    get_open_order_count_for_trader,
    get_open_position_count_for_trader,
    recover_missing_live_trader_orders,
    reconcile_live_provider_orders,
    sync_trader_position_inventory,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_live_provider_reconciliation")


async def _seed_trader(session, trader_id: str) -> None:
    now = utcnow()
    session.add(
        Trader(
            id=trader_id,
            name="Live Trader",
            source_configs_json=[{"source_key": "crypto", "strategy_key": "btc_eth_highfreq", "strategy_params": {}}],
            risk_limits_json={},
            metadata_json={},
            mode="live",
            is_enabled=True,
            is_paused=False,
            interval_seconds=60,
            created_at=now,
            updated_at=now,
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_sync_inventory_ignores_unfilled_live_open_orders(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-unfilled"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-unfilled",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-1",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=40.0,
                    entry_price=0.4,
                    effective_price=0.4,
                    payload_json={"provider_clob_order_id": "clob-unfilled"},
                    created_at=now,
                    executed_at=None,
                    updated_at=now,
                )
            )
            await session.commit()

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")

            open_orders = await get_open_order_count_for_trader(session, trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_orders == 1
            assert open_positions == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sync_inventory_noop_does_not_leave_transaction_open(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-noop-inventory"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)

            result = await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")

            assert result["open_positions"] == 0
            assert not session.in_transaction()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_provider_orders_noop_does_not_leave_transaction_open(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-noop-reconcile"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)

            result = await reconcile_live_provider_orders(session, trader_id=trader_id, commit=True)

            assert result["active_seen"] == 0
            assert not session.in_transaction()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_provider_orders_unchanged_snapshot_does_not_leave_transaction_open(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-unchanged-reconcile"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-unchanged",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-unchanged",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=50.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={"provider_clob_order_id": "clob-live-unchanged"},
                    created_at=now,
                    executed_at=None,
                    updated_at=now,
                )
            )
            await session.commit()

            monkeypatch.setattr(
                trader_orchestrator_state.live_execution_service,
                "ensure_initialized",
                AsyncMock(return_value=True),
            )
            monkeypatch.setattr(
                trader_orchestrator_state.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(return_value={}),
            )

            result = await reconcile_live_provider_orders(session, trader_id=trader_id, commit=True)

            assert result["active_seen"] == 1
            assert result["updated_orders"] == 0
            assert not session.in_transaction()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_provider_orders_ignores_executed_orders(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-ignore-executed"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-executed",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-executed",
                    direction="buy_yes",
                    mode="live",
                    status="executed",
                    notional_usd=12.0,
                    entry_price=0.4,
                    effective_price=0.4,
                    payload_json={"provider_clob_order_id": "clob-live-executed"},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            ensure_mock = AsyncMock(return_value=True)
            snapshots_mock = AsyncMock(return_value={})
            monkeypatch.setattr(
                trader_orchestrator_state.live_execution_service,
                "ensure_initialized",
                ensure_mock,
            )
            monkeypatch.setattr(
                trader_orchestrator_state.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                snapshots_mock,
            )

            result = await reconcile_live_provider_orders(session, trader_id=trader_id, commit=True)

            assert result["active_seen"] == 0
            ensure_mock.assert_not_awaited()
            snapshots_mock.assert_not_awaited()
            assert not session.in_transaction()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_provider_orders_updates_fill_state(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-reconcile"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-1",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-2",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=50.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={"provider_clob_order_id": "clob-live-2"},
                    created_at=now,
                    executed_at=None,
                    updated_at=now,
                )
            )
            await session.commit()

            monkeypatch.setattr(
                trader_orchestrator_state.live_execution_service,
                "ensure_initialized",
                AsyncMock(return_value=True),
            )
            monkeypatch.setattr(
                trader_orchestrator_state.live_execution_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "clob-live-2": {
                            "clob_order_id": "clob-live-2",
                            "normalized_status": "filled",
                            "filled_size": 30.0,
                            "average_fill_price": 0.41,
                            "filled_notional_usd": 12.3,
                            "limit_price": 0.42,
                        }
                    }
                ),
            )

            result = await reconcile_live_provider_orders(session, trader_id=trader_id, commit=True)
            assert result["updated_orders"] == 1
            assert result["status_changes"] == 1

            refreshed = await session.get(TraderOrder, "live-order-1")
            assert refreshed is not None
            assert refreshed.status == "executed"
            assert refreshed.notional_usd == pytest.approx(12.3)
            assert refreshed.effective_price == pytest.approx(0.41)
            assert refreshed.executed_at is not None
            assert isinstance((refreshed.payload_json or {}).get("provider_reconciliation"), dict)

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_positions == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_adopts_existing_provider_authority(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-adopt-authority"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-existing",
                    trader_id=trader_id,
                    source="crypto",
                    strategy_key="tail_end_carry",
                    market_id="market-mavs-cavs",
                    market_question="Mavericks vs. Cavaliers",
                    direction="buy_no",
                    mode="live",
                    status="resolved_win",
                    notional_usd=4.47,
                    entry_price=0.895,
                    effective_price=0.895,
                    reason="Tail carry signal selected",
                    payload_json={
                        "provider_clob_order_id": "clob-mavs-cavs",
                        "position_close": {
                            "close_trigger": "resolution_inferred",
                            "closed_at": now.isoformat(),
                            "close_price": 1.0,
                        },
                    },
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                    actual_profit=0.53,
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-mavs-cavs",
                    wallet_address="0xwallet",
                    clob_order_id="clob-mavs-cavs",
                    token_id="token-mavs-cavs",
                    side="BUY",
                    price=0.895,
                    size=5.0,
                    order_type="IOC",
                    status="filled",
                    filled_size=5.0,
                    average_fill_price=0.895,
                    market_question="Mavericks vs. Cavaliers",
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            result = await recover_missing_live_trader_orders(
                session,
                trader_ids=[trader_id],
                commit=True,
                broadcast=False,
            )

            assert result["recovered_orders"] == 0
            assert result["adopted_existing_orders"] == 1

            rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.created_at.asc(), TraderOrder.id.asc())
                )
            ).scalars().all()
            assert len(rows) == 1
            payload = dict(rows[0].payload_json or {})
            assert payload["provider_clob_order_id"] == "clob-mavs-cavs"
            assert payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-mavs-cavs"
            assert rows[0].reason == "Tail carry signal selected"
            assert rows[0].status == "resolved_win"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_collapses_duplicate_authority_rows(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-collapse-authority"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add_all(
                [
                    TraderOrder(
                        id="live-order-canonical",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="market-mavs-cavs",
                        market_question="Mavericks vs. Cavaliers",
                        direction="buy_no",
                        mode="live",
                        status="resolved_win",
                        notional_usd=4.47,
                        entry_price=0.895,
                        effective_price=0.895,
                        reason="Tail carry signal selected",
                        payload_json={
                            "provider_clob_order_id": "clob-mavs-cavs",
                            "position_close": {
                                "close_trigger": "resolution_inferred",
                                "closed_at": now.isoformat(),
                                "close_price": 1.0,
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        actual_profit=0.53,
                    ),
                    TraderOrder(
                        id="live-order-duplicate-a",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="market-mavs-cavs",
                        market_question="Mavericks vs. Cavaliers",
                        direction="buy_no",
                        mode="live",
                        status="resolved_win",
                        notional_usd=4.47,
                        entry_price=0.895,
                        effective_price=0.895,
                        reason="Recovered from live venue authority",
                        payload_json={
                            "provider_clob_order_id": "clob-mavs-cavs",
                            "live_wallet_authority": {
                                "live_trading_order_id": "venue-order-mavs-cavs",
                            },
                            "position_close": {
                                "close_trigger": "resolution_inferred",
                                "closed_at": now.isoformat(),
                                "close_price": 1.0,
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        actual_profit=0.53,
                    ),
                    TraderOrder(
                        id="live-order-duplicate-b",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="market-mavs-cavs",
                        market_question="Mavericks vs. Cavaliers",
                        direction="buy_no",
                        mode="live",
                        status="resolved_win",
                        notional_usd=4.47,
                        entry_price=0.895,
                        effective_price=0.895,
                        reason="Recovered from live venue authority",
                        payload_json={
                            "provider_clob_order_id": "clob-mavs-cavs",
                            "live_wallet_authority": {
                                "live_trading_order_id": "venue-order-mavs-cavs",
                            },
                            "position_close": {
                                "close_trigger": "resolution_inferred",
                                "closed_at": now.isoformat(),
                                "close_price": 1.0,
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        actual_profit=0.53,
                    ),
                    LiveTradingOrder(
                        id="venue-order-mavs-cavs",
                        wallet_address="0xwallet",
                        clob_order_id="clob-mavs-cavs",
                        token_id="token-mavs-cavs",
                        side="BUY",
                        price=0.895,
                        size=5.0,
                        order_type="IOC",
                        status="filled",
                        filled_size=5.0,
                        average_fill_price=0.895,
                        market_question="Mavericks vs. Cavaliers",
                        created_at=now,
                        updated_at=now,
                    ),
                ]
            )
            await session.commit()

            result = await recover_missing_live_trader_orders(
                session,
                trader_ids=[trader_id],
                commit=True,
                broadcast=False,
            )

            assert result["recovered_orders"] == 0
            assert result["collapsed_duplicates"] == 2

            rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.id.asc())
                )
            ).scalars().all()
            assert [row.id for row in rows] == ["live-order-canonical"]
            canonical = rows[0]
            assert canonical.status == "resolved_win"
            canonical_payload = dict(canonical.payload_json or {})
            assert canonical_payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-mavs-cavs"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_worker_runs_lifecycle_for_live_orders_without_provider_activity(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-lifecycle-no-provider-activity"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-manual-open",
                    trader_id=trader_id,
                    source="scanner",
                    market_id="market-live-manual",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=25.0,
                    entry_price=0.8,
                    effective_price=0.8,
                    payload_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        monkeypatch.setattr(trader_reconciliation_worker, "AsyncSessionLocal", session_factory)
        monkeypatch.setattr(
            trader_reconciliation_worker,
            "reconcile_live_provider_orders",
            AsyncMock(
                return_value={
                    "provider_ready": True,
                    "active_seen": 0,
                    "updated_orders": 0,
                    "status_changes": 0,
                    "notional_updates": 0,
                    "price_updates": 0,
                }
            ),
        )
        lifecycle_mock = AsyncMock(return_value={"would_close": 1, "closed": 1})
        monkeypatch.setattr(trader_reconciliation_worker, "reconcile_live_positions", lifecycle_mock)
        monkeypatch.setattr(
            trader_reconciliation_worker,
            "sync_trader_position_inventory",
            AsyncMock(return_value={"open_positions": 0, "updates": 0, "inserts": 0, "closures": 1}),
        )

        result = await trader_reconciliation_worker._reconcile_live_state_for_trader(
            {
                "id": trader_id,
                "source_configs": [{"source_key": "scanner", "strategy_key": "tail_end_carry", "strategy_params": {}}],
            },
            provider_pass=True,
        )

        lifecycle_mock.assert_awaited_once()
        assert result["lifecycle"]["closed"] == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_cleanup_requires_provider_cancel(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-cleanup"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-no-provider",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-3",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=25.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            with pytest.raises(ValueError, match="missing provider order identifiers"):
                await cleanup_trader_open_orders(
                    session,
                    trader_id=trader_id,
                    scope="live",
                    dry_run=False,
                    target_status="cancelled",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cleanup_filters_by_source_age_seconds_and_unfilled_only(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "shadow-cleanup-filters"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add_all(
                [
                    TraderOrder(
                        id="order-crypto-old-unfilled",
                        trader_id=trader_id,
                        source="crypto",
                        market_id="market-1",
                        direction="buy_yes",
                        mode="shadow",
                        status="open",
                        notional_usd=0.0,
                        entry_price=0.5,
                        effective_price=0.5,
                        payload_json={},
                        created_at=now - timedelta(seconds=45),
                        updated_at=now - timedelta(seconds=45),
                    ),
                    TraderOrder(
                        id="order-crypto-old-filled",
                        trader_id=trader_id,
                        source="crypto",
                        market_id="market-2",
                        direction="buy_yes",
                        mode="shadow",
                        status="open",
                        notional_usd=30.0,
                        entry_price=0.5,
                        effective_price=0.5,
                        payload_json={"filled_notional_usd": 30.0},
                        created_at=now - timedelta(seconds=45),
                        updated_at=now - timedelta(seconds=45),
                    ),
                    TraderOrder(
                        id="order-weather-old-unfilled",
                        trader_id=trader_id,
                        source="weather",
                        market_id="market-3",
                        direction="buy_yes",
                        mode="shadow",
                        status="open",
                        notional_usd=0.0,
                        entry_price=0.5,
                        effective_price=0.5,
                        payload_json={},
                        created_at=now - timedelta(seconds=45),
                        updated_at=now - timedelta(seconds=45),
                    ),
                    TraderOrder(
                        id="order-crypto-fresh-unfilled",
                        trader_id=trader_id,
                        source="crypto",
                        market_id="market-4",
                        direction="buy_yes",
                        mode="shadow",
                        status="open",
                        notional_usd=0.0,
                        entry_price=0.5,
                        effective_price=0.5,
                        payload_json={},
                        created_at=now - timedelta(seconds=5),
                        updated_at=now - timedelta(seconds=5),
                    ),
                ]
            )
            await session.commit()

            result = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope="shadow",
                max_age_seconds=20.0,
                source="crypto",
                require_unfilled=True,
                dry_run=False,
                target_status="cancelled",
                reason="test_cleanup",
            )
            assert result["updated"] == 1

            crypto_old_unfilled = await session.get(TraderOrder, "order-crypto-old-unfilled")
            crypto_old_filled = await session.get(TraderOrder, "order-crypto-old-filled")
            weather_old_unfilled = await session.get(TraderOrder, "order-weather-old-unfilled")
            crypto_fresh_unfilled = await session.get(TraderOrder, "order-crypto-fresh-unfilled")

            assert crypto_old_unfilled is not None and crypto_old_unfilled.status == "cancelled"
            assert crypto_old_filled is not None and crypto_old_filled.status == "open"
            assert weather_old_unfilled is not None and weather_old_unfilled.status == "open"
            assert crypto_fresh_unfilled is not None and crypto_fresh_unfilled.status == "open"
    finally:
        await engine.dispose()
