import sys
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import (
    Base,
    ExecutionSession,
    ExecutionSessionLeg,
    ExecutionSessionOrder,
    LiveTradingOrder,
    LiveTradingPosition,
    TradeSignal,
    Trader,
    TraderDecision,
    TraderOrder,
    TraderPosition,
)
from services import trader_hot_state, trader_orchestrator_state
from workers import trader_reconciliation_worker
from services.trader_orchestrator_state import (
    cleanup_trader_open_orders,
    get_open_order_count_for_trader,
    get_open_position_count_for_trader,
    get_trader_orders_summary,
    recover_missing_live_trader_orders,
    reconcile_live_provider_orders,
    sync_trader_position_inventory,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow


def test_sync_recovered_order_hot_state_routes_terminal_unfilled_order_to_cancel(monkeypatch):
    resolved: list[dict] = []
    cancelled: list[dict] = []

    monkeypatch.setattr(trader_hot_state, "record_order_resolved", lambda **kwargs: resolved.append(kwargs))
    monkeypatch.setattr(trader_hot_state, "record_order_cancelled", lambda **kwargs: cancelled.append(kwargs))

    trader_orchestrator_state._sync_recovered_order_hot_state(
        SimpleNamespace(
            id="recovered-order-1",
            trader_id="trader-1",
            mode="live",
            status="rejected",
            market_id="market-1",
            direction="buy_yes",
            source="scanner",
            notional_usd=0.0,
            entry_price=0.55,
            effective_price=0.55,
            actual_profit=None,
            payload_json={
                "token_id": "token-1",
                "provider_reconciliation": {
                    "filled_size": 0.0,
                    "filled_notional_usd": 0.0,
                },
            },
        )
    )

    assert resolved == []
    assert len(cancelled) == 1
    assert cancelled[0]["order_id"] == "recovered-order-1"


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_live_provider_reconciliation")


async def _seed_trader(
    session,
    trader_id: str,
    *,
    source_configs: list[dict] | None = None,
    name: str = "Live Trader",
) -> None:
    now = utcnow()
    session.add(
        Trader(
            id=trader_id,
            name=name,
            source_configs_json=source_configs
            or [{"source_key": "crypto", "strategy_key": "btc_eth_maker_quote", "strategy_params": {}}],
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


async def _seed_decision(
    session,
    *,
    trader_id: str,
    decision_id: str,
    signal_id: str,
    market_id: str,
    market_question: str,
    direction: str,
    source: str = "scanner",
    strategy_key: str = "generic_strategy",
    strategy_version: int = 1,
) -> None:
    created_at = utcnow()
    session.add(
        TradeSignal(
            id=signal_id,
            source=source,
            signal_type="entry",
            strategy_type=strategy_key,
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            entry_price=0.5,
            dedupe_key=f"dedupe-{signal_id}",
            payload_json={},
            created_at=created_at,
            updated_at=created_at,
        )
    )
    session.add(
        TraderDecision(
            id=decision_id,
            trader_id=trader_id,
            signal_id=signal_id,
            source=source,
            strategy_key=strategy_key,
            strategy_version=strategy_version,
            decision="selected",
            created_at=created_at,
        )
    )
    await session.flush()


def test_inter_trader_sleep_skips_startup_delay():
    assert trader_reconciliation_worker._inter_trader_sleep_seconds(reason="startup") == 0.0
    assert trader_reconciliation_worker._inter_trader_sleep_seconds(reason="scheduled") == 0.1


def test_post_cycle_cooldown_debounces_event_cycles():
    assert (
        trader_reconciliation_worker._post_cycle_cooldown_seconds(
            reason="scheduled",
            provider_pass=True,
        )
        == 1.0
    )
    assert (
        trader_reconciliation_worker._post_cycle_cooldown_seconds(
            reason="event:trader_order",
            provider_pass=True,
        )
        == 30.0
    )
    assert (
        trader_reconciliation_worker._post_cycle_cooldown_seconds(
            reason="position_tick",
            provider_pass=False,
        )
        == 0.25
    )


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
async def test_sync_inventory_treats_completed_live_orders_as_open_positions(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-completed"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-completed",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-1",
                    direction="buy_yes",
                    mode="live",
                    status="completed",
                    notional_usd=40.0,
                    entry_price=0.4,
                    effective_price=0.4,
                    payload_json={
                        "token_id": "token-live-1",
                        "provider_clob_order_id": "clob-completed",
                        "provider_reconciliation": {
                            "filled_size": 100.0,
                            "filled_notional_usd": 40.0,
                            "average_fill_price": 0.4,
                        },
                    },
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")

            open_orders = await get_open_order_count_for_trader(session, trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_orders == 0
            assert open_positions == 1
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
async def test_trader_orders_summary_uses_position_inventory_for_open_trade_counts(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-summary-open-positions"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add_all(
                [
                    TraderOrder(
                        id="summary-open-order-a",
                        trader_id=trader_id,
                        signal_id=None,
                        source="crypto",
                        market_id="market-summary",
                        market_question="Will Example settle above 50?",
                        direction="buy_yes",
                        mode="live",
                        status="executed",
                        notional_usd=10.0,
                        entry_price=0.5,
                        effective_price=0.5,
                        payload_json={
                            "token_id": "token-summary-a",
                            "provider_reconciliation": {
                                "filled_size": 20.0,
                                "average_fill_price": 0.5,
                                "filled_notional_usd": 10.0,
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                    ),
                    TraderOrder(
                        id="summary-open-order-b",
                        trader_id=trader_id,
                        signal_id=None,
                        source="crypto",
                        market_id="market-summary",
                        market_question="Will Example settle above 50?",
                        direction="buy_yes",
                        mode="live",
                        status="executed",
                        notional_usd=8.0,
                        entry_price=0.4,
                        effective_price=0.4,
                        payload_json={
                            "token_id": "token-summary-b",
                            "provider_reconciliation": {
                                "filled_size": 20.0,
                                "average_fill_price": 0.4,
                                "filled_notional_usd": 8.0,
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                    ),
                ]
            )
            await session.commit()

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
            summary = await get_trader_orders_summary(session, mode="live")
            trader_summary = next(row for row in summary["by_trader"] if row["trader_id"] == trader_id)

            assert summary["open"] == 2
            assert summary["open_trades"] == 1
            assert trader_summary["open"] == 2
            assert trader_summary["open_trades"] == 1
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
            await _seed_trader(
                session,
                trader_id,
                source_configs=[
                    {
                        "source_key": "scanner",
                        "strategy_key": "generic_strategy",
                        "strategy_params": {
                            "take_profit_pct": 8,
                            "stop_loss_pct": 12,
                        },
                    }
                ],
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-existing",
                    trader_id=trader_id,
                    source="scanner",
                    strategy_key="generic_strategy",
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
            assert payload["strategy_type"] == "generic_strategy"
            assert payload["strategy_exit_config"]["take_profit_pct"] == 8
            assert payload["strategy_exit_config"]["stop_loss_pct"] == 12
            assert rows[0].reason == "Tail carry signal selected"
            assert rows[0].status == "resolved_win"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_does_not_duplicate_global_provider_authority_for_scoped_trader(
    tmp_path,
):
    engine, session_factory = await _build_session_factory(tmp_path)
    canonical_trader_id = "live-trader-canonical-authority"
    scoped_trader_id = "live-trader-scoped-authority"
    try:
        async with session_factory() as session:
            await _seed_trader(session, canonical_trader_id, name="Canonical Live Trader")
            await _seed_trader(session, scoped_trader_id, name="Scoped Live Trader")
            await _seed_decision(
                session,
                trader_id=scoped_trader_id,
                decision_id="decision-scoped-authority",
                signal_id="signal-scoped-authority",
                market_id="market-shared-authority",
                market_question="Will shared venue authority stay unique?",
                direction="buy_yes",
                strategy_key="generic_strategy",
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="canonical-authority-order",
                    trader_id=canonical_trader_id,
                    source="scanner",
                    strategy_key="generic_strategy",
                    market_id="market-shared-authority",
                    market_question="Will shared venue authority stay unique?",
                    direction="buy_yes",
                    mode="live",
                    status="closed_loss",
                    notional_usd=9.0,
                    entry_price=0.9,
                    effective_price=0.9,
                    provider_clob_order_id="clob-shared-authority",
                    reason="Signal selected",
                    payload_json={
                        "provider_clob_order_id": "clob-shared-authority",
                        "token_id": "token-shared-authority",
                        "position_close": {
                            "close_trigger": "wallet_activity",
                            "price_source": "wallet_activity",
                            "realized_pnl": -0.09,
                            "wallet_activity_transaction_hash": "0xsharedauthorityclose",
                            "wallet_activity_timestamp": now.isoformat(),
                        },
                    },
                    created_at=now - timedelta(minutes=5),
                    executed_at=now - timedelta(minutes=5),
                    updated_at=now,
                    actual_profit=-0.09,
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-shared-authority",
                    wallet_address="0xwallet",
                    clob_order_id="clob-shared-authority",
                    token_id="token-shared-authority",
                    side="BUY",
                    price=0.9,
                    size=10.0,
                    order_type="GTC",
                    status="filled",
                    filled_size=10.0,
                    average_fill_price=0.9,
                    market_question="Will shared venue authority stay unique?",
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            result = await recover_missing_live_trader_orders(
                session,
                trader_ids=[scoped_trader_id],
                commit=True,
                broadcast=False,
            )

            assert result["recovered_orders"] == 0
            assert result["adopted_existing_orders"] == 1

            scoped_rows = (
                await session.execute(select(TraderOrder).where(TraderOrder.trader_id == scoped_trader_id))
            ).scalars().all()
            canonical_rows = (
                await session.execute(select(TraderOrder).where(TraderOrder.trader_id == canonical_trader_id))
            ).scalars().all()
            assert scoped_rows == []
            assert len(canonical_rows) == 1
            assert canonical_rows[0].provider_clob_order_id == "clob-shared-authority"
            assert canonical_rows[0].status == "closed_loss"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_preserves_existing_wallet_close_authority(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-preserve-wallet-close-authority"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="existing-wallet-close-authority",
                    trader_id=trader_id,
                    source="scanner",
                    strategy_key="generic_strategy",
                    market_id="market-wallet-close-authority",
                    market_question="Will wallet close authority remain terminal?",
                    direction="buy_yes",
                    mode="live",
                    status="executed",
                    notional_usd=9.0,
                    entry_price=0.9,
                    effective_price=0.9,
                    provider_clob_order_id="clob-wallet-close-authority",
                    reason="Signal selected",
                    payload_json={
                        "provider_clob_order_id": "clob-wallet-close-authority",
                        "token_id": "token-wallet-close-authority",
                        "position_close": {
                            "close_trigger": "wallet_activity",
                            "price_source": "wallet_activity",
                            "realized_pnl": -0.12,
                            "wallet_activity_transaction_hash": "0xwalletcloseauthority",
                            "wallet_activity_timestamp": now.isoformat(),
                        },
                    },
                    created_at=now - timedelta(minutes=5),
                    executed_at=now - timedelta(minutes=5),
                    updated_at=now - timedelta(minutes=1),
                    actual_profit=None,
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-wallet-close-authority",
                    wallet_address="0xwallet",
                    clob_order_id="clob-wallet-close-authority",
                    token_id="token-wallet-close-authority",
                    side="BUY",
                    price=0.9,
                    size=10.0,
                    order_type="GTC",
                    status="filled",
                    filled_size=10.0,
                    average_fill_price=0.9,
                    market_question="Will wallet close authority remain terminal?",
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
            row = await session.get(TraderOrder, "existing-wallet-close-authority")
            assert row is not None
            assert row.status == "closed_loss"
            assert row.actual_profit == pytest.approx(-0.12)
            payload = dict(row.payload_json or {})
            assert payload["position_close_status_repair"]["reason"] == (
                "live_authority_recovery_preserved_terminal_position_close"
            )
            assert payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-wallet-close-authority"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_adopts_pre_submit_placeholder_without_creating_recovered_row(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-adopt-pre-submit-placeholder"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-pre-submit-placeholder",
                signal_id="signal-pre-submit-placeholder",
                market_id="market-pre-submit-placeholder",
                market_question="Will Placeholder Market settle YES?",
                direction="buy_yes",
                strategy_key="generic_strategy",
            )
            now = utcnow()
            recovered_at = now + timedelta(seconds=4)
            submission_intent = {
                "state": "pre_submit_persisted",
                "persisted_at": now.isoformat(),
                "requested_notional_usd": 5.0,
                "requested_shares": 10.0,
                "time_in_force": "IOC",
                "post_only": False,
                "price_policy": "taker_limit",
            }
            session.add_all(
                [
                    ExecutionSession(
                        id="execution-session-placeholder",
                        trader_id=trader_id,
                        signal_id="signal-pre-submit-placeholder",
                        decision_id="decision-pre-submit-placeholder",
                        source="scanner",
                        strategy_key="generic_strategy",
                        strategy_version=1,
                        mode="live",
                        status="placing",
                        policy="SINGLE_LEG",
                        plan_id="plan-pre-submit-placeholder",
                        market_ids_json=["market-pre-submit-placeholder"],
                        legs_total=1,
                        legs_completed=0,
                        legs_failed=0,
                        legs_open=1,
                        requested_notional_usd=5.0,
                        executed_notional_usd=0.0,
                        max_unhedged_notional_usd=0.0,
                        unhedged_notional_usd=0.0,
                        trace_id="trace-pre-submit-placeholder",
                        started_at=now,
                        payload_json={},
                        created_at=now,
                        updated_at=now,
                    ),
                    ExecutionSessionLeg(
                        id="execution-leg-placeholder",
                        session_id="execution-session-placeholder",
                        leg_index=0,
                        leg_id="leg-pre-submit-placeholder",
                        market_id="market-pre-submit-placeholder",
                        market_question="Will Placeholder Market settle YES?",
                        token_id="token-pre-submit-placeholder",
                        side="buy",
                        outcome="yes",
                        price_policy="taker_limit",
                        time_in_force="IOC",
                        post_only=False,
                        target_price=0.5,
                        requested_notional_usd=5.0,
                        requested_shares=10.0,
                        filled_notional_usd=0.0,
                        filled_shares=0.0,
                        status="placing",
                        metadata_json={},
                        created_at=now,
                        updated_at=now,
                    ),
                    TraderOrder(
                        id="pre-submit-placeholder-order",
                        trader_id=trader_id,
                        signal_id="signal-pre-submit-placeholder",
                        decision_id="decision-pre-submit-placeholder",
                        source="scanner",
                        strategy_key="generic_strategy",
                        market_id="market-pre-submit-placeholder",
                        market_question="Will Placeholder Market settle YES?",
                        direction="buy_yes",
                        mode="live",
                        status="placing",
                        notional_usd=None,
                        entry_price=0.5,
                        effective_price=None,
                        reason="Signal selected",
                        payload_json={
                            "token_id": "token-pre-submit-placeholder",
                            "market_id": "market-pre-submit-placeholder",
                            "direction": "buy_yes",
                            "submission_intent": dict(submission_intent),
                            "execution_session": {
                                "session_id": "execution-session-placeholder",
                                "leg_id": "execution-leg-placeholder",
                                "leg_ref": "leg-pre-submit-placeholder",
                                "policy": "SINGLE_LEG",
                            },
                        },
                        created_at=now,
                        updated_at=now,
                        verification_status="local",
                    ),
                    LiveTradingOrder(
                        id="venue-order-pre-submit-placeholder",
                        wallet_address="0xwallet",
                        market_id="market-pre-submit-placeholder",
                        clob_order_id="clob-pre-submit-placeholder",
                        token_id="token-pre-submit-placeholder",
                        side="BUY",
                        price=0.5,
                        size=10.0,
                        order_type="IOC",
                        status="filled",
                        filled_size=10.0,
                        average_fill_price=0.5,
                        market_question="Will Placeholder Market settle YES?",
                        created_at=recovered_at,
                        updated_at=recovered_at,
                    ),
                    LiveTradingPosition(
                        id="0xwallet:token-pre-submit-placeholder",
                        wallet_address="0xwallet",
                        token_id="token-pre-submit-placeholder",
                        market_id="market-pre-submit-placeholder",
                        market_question="Will Placeholder Market settle YES?",
                        outcome="Yes",
                        size=10.0,
                        average_cost=0.5,
                        current_price=0.5,
                        unrealized_pnl=0.0,
                        redeemable=False,
                        counts_as_open=True,
                        end_date=None,
                        created_at=recovered_at,
                        updated_at=recovered_at,
                    ),
                ]
            )
            await session.flush()
            session.add(
                ExecutionSessionOrder(
                    id="execution-order-placeholder",
                    session_id="execution-session-placeholder",
                    leg_id="execution-leg-placeholder",
                    trader_order_id="pre-submit-placeholder-order",
                    provider_order_id=None,
                    provider_clob_order_id=None,
                    action="submit",
                    side="buy",
                    price=0.5,
                    size=10.0,
                    notional_usd=5.0,
                    status="placing",
                    reason="Signal selected",
                    payload_json={"submission_intent": dict(submission_intent)},
                    error_message=None,
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

            trader_rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.created_at.asc(), TraderOrder.id.asc())
                )
            ).scalars().all()
            assert [row.id for row in trader_rows] == ["pre-submit-placeholder-order"]
            placeholder_order = trader_rows[0]
            assert placeholder_order.status == "executed"
            assert placeholder_order.reason == "Signal selected"
            assert placeholder_order.provider_clob_order_id == "clob-pre-submit-placeholder"
            assert placeholder_order.notional_usd == pytest.approx(5.0)
            placeholder_payload = dict(placeholder_order.payload_json or {})
            assert placeholder_payload["submission_intent"]["state"] == "recovered_from_live_authority"
            assert placeholder_payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-pre-submit-placeholder"

            execution_order = await session.get(ExecutionSessionOrder, "execution-order-placeholder")
            assert execution_order is not None
            assert execution_order.status == "executed"
            assert execution_order.provider_clob_order_id == "clob-pre-submit-placeholder"
            assert dict(execution_order.payload_json or {})["submission_intent"]["state"] == "recovered_from_live_authority"

            execution_leg = await session.get(ExecutionSessionLeg, "execution-leg-placeholder")
            assert execution_leg is not None
            assert execution_leg.status == "completed"
            assert execution_leg.filled_notional_usd == pytest.approx(5.0)
            assert execution_leg.filled_shares == pytest.approx(10.0)
            assert execution_leg.avg_fill_price == pytest.approx(0.5)

            execution_session = await session.get(ExecutionSession, "execution-session-placeholder")
            assert execution_session is not None
            assert execution_session.status == "completed"
            assert execution_session.legs_completed == 1
            assert execution_session.legs_failed == 0
            assert execution_session.legs_open == 0
            assert execution_session.executed_notional_usd == pytest.approx(5.0)
            assert execution_session.completed_at is not None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_adopts_pre_submit_placeholder_when_live_position_market_id_uses_condition_id(
    tmp_path,
):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-adopt-placeholder-condition-id-market"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-placeholder-condition-id-market",
                signal_id="signal-placeholder-condition-id-market",
                market_id="2034715",
                market_question="Will Placeholder Condition-ID Market settle YES?",
                direction="buy_no",
                strategy_key="generic_strategy",
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="placeholder-condition-id-market-order",
                    trader_id=trader_id,
                    signal_id="signal-placeholder-condition-id-market",
                    decision_id="decision-placeholder-condition-id-market",
                    source="scanner",
                    strategy_key="generic_strategy",
                    market_id="2034715",
                    market_question="Will Placeholder Condition-ID Market settle YES?",
                    direction="buy_no",
                    mode="live",
                    status="placing",
                    notional_usd=None,
                    entry_price=0.87,
                    effective_price=None,
                    reason="Signal selected",
                    payload_json={
                        "token_id": "token-placeholder-condition-id-market",
                        "market_id": "2034715",
                        "direction": "buy_no",
                        "submission_intent": {
                            "state": "pre_submit_persisted",
                            "persisted_at": now.isoformat(),
                            "requested_notional_usd": 9.57,
                            "requested_shares": 11.0,
                            "time_in_force": "IOC",
                            "post_only": False,
                            "price_policy": "taker_limit",
                        },
                    },
                    created_at=now,
                    updated_at=now,
                    verification_status="local",
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-placeholder-condition-id-market",
                    wallet_address="0xwallet",
                    market_id=None,
                    clob_order_id="clob-placeholder-condition-id-market",
                    token_id="token-placeholder-condition-id-market",
                    side="BUY",
                    price=0.87,
                    size=11.0,
                    order_type="IOC",
                    status="filled",
                    filled_size=11.0,
                    average_fill_price=0.87,
                    market_question="Will Placeholder Condition-ID Market settle YES?",
                    created_at=now + timedelta(seconds=2),
                    updated_at=now + timedelta(seconds=2),
                )
            )
            session.add(
                LiveTradingPosition(
                    id="0xwallet:token-placeholder-condition-id-market",
                    wallet_address="0xwallet",
                    token_id="token-placeholder-condition-id-market",
                    market_id="0x747a380bab6e04278667e1e881cc62bd2a6a6090d295678f426249d06ca79712",
                    market_question="Will Placeholder Condition-ID Market settle YES?",
                    outcome="No",
                    size=11.0,
                    average_cost=0.87,
                    current_price=0.86,
                    unrealized_pnl=-0.11,
                    redeemable=False,
                    counts_as_open=True,
                    end_date=None,
                    created_at=now + timedelta(seconds=2),
                    updated_at=now + timedelta(seconds=2),
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

            trader_rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.created_at.asc(), TraderOrder.id.asc())
                )
            ).scalars().all()
            assert [row.id for row in trader_rows] == ["placeholder-condition-id-market-order"]
            placeholder_order = trader_rows[0]
            assert placeholder_order.status == "executed"
            assert placeholder_order.provider_clob_order_id == "clob-placeholder-condition-id-market"
            assert placeholder_order.notional_usd == pytest.approx(9.57)
            placeholder_payload = dict(placeholder_order.payload_json or {})
            assert placeholder_payload["submission_intent"]["state"] == "recovered_from_live_authority"
            assert placeholder_payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-placeholder-condition-id-market"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_adopts_timed_out_pre_submit_placeholder_without_creating_recovered_row(
    tmp_path,
):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-adopt-timed-out-placeholder"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-timed-out-placeholder",
                signal_id="signal-timed-out-placeholder",
                market_id="market-timed-out-placeholder",
                market_question="Will Timed-Out Placeholder Market settle YES?",
                direction="buy_yes",
                strategy_key="generic_strategy",
            )
            now = utcnow()
            timeout_at = now + timedelta(seconds=45)
            recovered_at = timeout_at + timedelta(seconds=4)
            timeout_reason = "Live order submission did not persist provider acknowledgement within 45s."
            submission_intent = {
                "state": "submit_timeout",
                "persisted_at": now.isoformat(),
                "requested_notional_usd": 5.0,
                "requested_shares": 10.0,
                "time_in_force": "IOC",
                "post_only": False,
                "price_policy": "taker_limit",
                "submit_status": "timed_out",
                "error_message": timeout_reason,
                "last_submit_result_at": timeout_at.isoformat(),
            }
            session.add_all(
                [
                    ExecutionSession(
                        id="execution-session-timed-out-placeholder",
                        trader_id=trader_id,
                        signal_id="signal-timed-out-placeholder",
                        decision_id="decision-timed-out-placeholder",
                        source="scanner",
                        strategy_key="generic_strategy",
                        strategy_version=1,
                        mode="live",
                        status="failed",
                        policy="SINGLE_LEG",
                        plan_id="plan-timed-out-placeholder",
                        market_ids_json=["market-timed-out-placeholder"],
                        legs_total=1,
                        legs_completed=0,
                        legs_failed=1,
                        legs_open=0,
                        requested_notional_usd=5.0,
                        executed_notional_usd=0.0,
                        max_unhedged_notional_usd=0.0,
                        unhedged_notional_usd=0.0,
                        trace_id="trace-timed-out-placeholder",
                        started_at=now,
                        completed_at=timeout_at,
                        error_message=timeout_reason,
                        payload_json={},
                        created_at=now,
                        updated_at=timeout_at,
                    ),
                    ExecutionSessionLeg(
                        id="execution-leg-timed-out-placeholder",
                        session_id="execution-session-timed-out-placeholder",
                        leg_index=0,
                        leg_id="leg-timed-out-placeholder",
                        market_id="market-timed-out-placeholder",
                        market_question="Will Timed-Out Placeholder Market settle YES?",
                        token_id="token-timed-out-placeholder",
                        side="buy",
                        outcome="yes",
                        price_policy="taker_limit",
                        time_in_force="IOC",
                        post_only=False,
                        target_price=0.5,
                        requested_notional_usd=5.0,
                        requested_shares=10.0,
                        filled_notional_usd=0.0,
                        filled_shares=0.0,
                        status="failed",
                        last_error=timeout_reason,
                        metadata_json={},
                        created_at=now,
                        updated_at=timeout_at,
                    ),
                    TraderOrder(
                        id="timed-out-placeholder-order",
                        trader_id=trader_id,
                        signal_id="signal-timed-out-placeholder",
                        decision_id="decision-timed-out-placeholder",
                        source="scanner",
                        strategy_key="generic_strategy",
                        market_id="market-timed-out-placeholder",
                        market_question="Will Timed-Out Placeholder Market settle YES?",
                        direction="buy_yes",
                        mode="live",
                        status="failed",
                        notional_usd=0.0,
                        entry_price=0.5,
                        effective_price=None,
                        reason="Signal selected",
                        payload_json={
                            "token_id": "token-timed-out-placeholder",
                            "market_id": "market-timed-out-placeholder",
                            "direction": "buy_yes",
                            "submission_intent": dict(submission_intent),
                            "execution_session": {
                                "session_id": "execution-session-timed-out-placeholder",
                                "leg_id": "execution-leg-timed-out-placeholder",
                                "leg_ref": "leg-timed-out-placeholder",
                                "policy": "SINGLE_LEG",
                            },
                            "session_cancel": {
                                "session_id": "execution-session-timed-out-placeholder",
                                "terminal_status": "failed",
                                "reason": timeout_reason,
                                "cancelled_at": timeout_at.isoformat(),
                            },
                        },
                        created_at=now,
                        updated_at=timeout_at,
                        verification_status="local",
                        error_message=timeout_reason,
                    ),
                    LiveTradingOrder(
                        id="venue-order-timed-out-placeholder",
                        wallet_address="0xwallet",
                        market_id="market-timed-out-placeholder",
                        clob_order_id="clob-timed-out-placeholder",
                        token_id="token-timed-out-placeholder",
                        side="BUY",
                        price=0.5,
                        size=10.0,
                        order_type="IOC",
                        status="filled",
                        filled_size=10.0,
                        average_fill_price=0.5,
                        market_question="Will Timed-Out Placeholder Market settle YES?",
                        created_at=recovered_at,
                        updated_at=recovered_at,
                    ),
                    LiveTradingPosition(
                        id="0xwallet:token-timed-out-placeholder",
                        wallet_address="0xwallet",
                        token_id="token-timed-out-placeholder",
                        market_id="market-timed-out-placeholder",
                        market_question="Will Timed-Out Placeholder Market settle YES?",
                        outcome="Yes",
                        size=10.0,
                        average_cost=0.5,
                        current_price=0.5,
                        unrealized_pnl=0.0,
                        redeemable=False,
                        counts_as_open=True,
                        end_date=None,
                        created_at=recovered_at,
                        updated_at=recovered_at,
                    ),
                ]
            )
            await session.flush()
            session.add(
                ExecutionSessionOrder(
                    id="execution-order-timed-out-placeholder",
                    session_id="execution-session-timed-out-placeholder",
                    leg_id="execution-leg-timed-out-placeholder",
                    trader_order_id="timed-out-placeholder-order",
                    provider_order_id=None,
                    provider_clob_order_id=None,
                    action="submit",
                    side="buy",
                    price=0.5,
                    size=10.0,
                    notional_usd=5.0,
                    status="placing",
                    reason="Signal selected",
                    payload_json={"submission_intent": dict(submission_intent)},
                    error_message=None,
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

            trader_rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.created_at.asc(), TraderOrder.id.asc())
                )
            ).scalars().all()
            assert [row.id for row in trader_rows] == ["timed-out-placeholder-order"]
            placeholder_order = trader_rows[0]
            assert placeholder_order.status == "executed"
            assert placeholder_order.error_message is None
            assert placeholder_order.provider_clob_order_id == "clob-timed-out-placeholder"
            assert placeholder_order.notional_usd == pytest.approx(5.0)
            placeholder_payload = dict(placeholder_order.payload_json or {})
            assert placeholder_payload["submission_intent"]["state"] == "recovered_from_live_authority"
            assert placeholder_payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-timed-out-placeholder"

            execution_order = await session.get(ExecutionSessionOrder, "execution-order-timed-out-placeholder")
            assert execution_order is not None
            assert execution_order.status == "executed"
            assert execution_order.provider_clob_order_id == "clob-timed-out-placeholder"
            assert dict(execution_order.payload_json or {})["submission_intent"]["state"] == "recovered_from_live_authority"

            execution_leg = await session.get(ExecutionSessionLeg, "execution-leg-timed-out-placeholder")
            assert execution_leg is not None
            assert execution_leg.status == "completed"
            assert execution_leg.filled_notional_usd == pytest.approx(5.0)
            assert execution_leg.filled_shares == pytest.approx(10.0)
            assert execution_leg.avg_fill_price == pytest.approx(0.5)
            assert execution_leg.last_error is None

            execution_session = await session.get(ExecutionSession, "execution-session-timed-out-placeholder")
            assert execution_session is not None
            assert execution_session.status == "completed"
            assert execution_session.error_message is None
            assert execution_session.legs_completed == 1
            assert execution_session.legs_failed == 0
            assert execution_session.legs_open == 0
            assert execution_session.executed_notional_usd == pytest.approx(5.0)
            assert execution_session.completed_at is not None
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
async def test_recover_missing_live_trader_orders_collapses_duplicate_authority_rows_when_position_market_ids_differ(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-collapse-mismatched-market-ids"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add_all(
                [
                    TraderOrder(
                        id="live-order-canonical-mismatch",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="1871044",
                        market_question="Jazz vs. Thunder: 1H Moneyline",
                        direction="buy_no",
                        mode="live",
                        status="executed",
                        notional_usd=6.85,
                        entry_price=0.92,
                        effective_price=0.92,
                        reason="Tail carry signal selected",
                        provider_clob_order_id="clob-jazz-thunder",
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        payload_json={},
                    ),
                    TraderOrder(
                        id="live-order-duplicate-mismatch",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="1871044",
                        market_question="Jazz vs. Thunder: 1H Moneyline",
                        direction="buy_no",
                        mode="live",
                        status="executed",
                        notional_usd=6.85,
                        entry_price=0.92,
                        effective_price=0.92,
                        reason="Recovered from live venue authority",
                        provider_clob_order_id="clob-jazz-thunder",
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        payload_json={},
                    ),
                    LiveTradingOrder(
                        id="venue-order-jazz-thunder",
                        wallet_address="0xwallet",
                        market_id="1871044",
                        clob_order_id="clob-jazz-thunder",
                        token_id="token-jazz-thunder",
                        side="BUY",
                        price=0.92,
                        size=7.4,
                        order_type="IOC",
                        status="filled",
                        filled_size=7.4,
                        average_fill_price=0.92,
                        market_question="Jazz vs. Thunder: 1H Moneyline",
                        created_at=now,
                        updated_at=now,
                    ),
                    LiveTradingPosition(
                        id="0xwallet:token-jazz-thunder",
                        wallet_address="0xwallet",
                        token_id="token-jazz-thunder",
                        market_id="0x31f5380ecd1201be8724b01223acd97f49a3d19e68e1db78ed37042702a8f078",
                        market_question="Jazz vs. Thunder: 1H Moneyline",
                        outcome="No",
                        size=7.4,
                        average_cost=0.92,
                        current_price=0.915,
                        unrealized_pnl=-0.04,
                        created_at=now,
                        updated_at=now,
                    ),
                ]
            )
            await session.commit()

            result = await recover_missing_live_trader_orders(
                session,
                trader_ids=None,
                commit=True,
                broadcast=False,
            )

            assert result["collapsed_duplicates"] == 1

            rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.id.asc())
                )
            ).scalars().all()
            assert [row.id for row in rows] == ["live-order-canonical-mismatch"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_adopts_sell_exit_into_pending_exit_and_deletes_auxiliary_row(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-sell-exit-adoption"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add_all(
                [
                    TraderOrder(
                        id="parent-entry-order",
                        trader_id=trader_id,
                        source="scanner",
                        strategy_key="tail_end_carry",
                        market_id="market-hormuz",
                        market_question="Will 20 ships transit the Strait of Hormuz on any day in March?",
                        direction="buy_no",
                        mode="live",
                        status="executed",
                        notional_usd=8.54,
                        entry_price=0.922,
                        effective_price=0.922,
                        reason="Tail carry signal selected",
                        payload_json={
                            "token_id": "token-hormuz-no",
                            "pending_live_exit": {
                                "status": "submitted",
                                "provider_clob_order_id": "clob-hormuz-exit",
                                "exit_size": 9.26,
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                    ),
                    TraderOrder(
                        id="auxiliary-sell-authority-row",
                        trader_id=trader_id,
                        source="scanner",
                        strategy_key="tail_end_carry",
                        market_id="market-hormuz",
                        market_question="Will 20 ships transit the Strait of Hormuz on any day in March?",
                        direction="buy_no",
                        mode="live",
                        status="open",
                        notional_usd=0.0,
                        entry_price=0.97,
                        effective_price=0.97,
                        reason="Recovered from live venue authority",
                        provider_clob_order_id="clob-hormuz-exit",
                        payload_json={
                            "provider_clob_order_id": "clob-hormuz-exit",
                            "provider_reconciliation": {
                                "snapshot": {
                                    "side": "SELL",
                                    "normalized_status": "open",
                                }
                            },
                            "live_wallet_authority": {
                                "live_trading_order_id": "venue-exit-order-hormuz",
                            },
                        },
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                    ),
                    LiveTradingOrder(
                        id="venue-exit-order-hormuz",
                        wallet_address="0xwallet",
                        market_id="market-hormuz",
                        clob_order_id="clob-hormuz-exit",
                        token_id="token-hormuz-no",
                        side="SELL",
                        price=0.97,
                        size=9.26,
                        order_type="GTC",
                        status="open",
                        filled_size=0.0,
                        average_fill_price=0.0,
                        market_question="Will 20 ships transit the Strait of Hormuz on any day in March?",
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
            assert result["collapsed_duplicates"] == 1
            assert result["adopted_existing_orders"] == 1

            rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.id.asc())
                )
            ).scalars().all()
            assert [row.id for row in rows] == ["parent-entry-order"]
            pending_exit = dict((rows[0].payload_json or {}).get("pending_live_exit") or {})
            assert pending_exit["status"] == "submitted"
            assert pending_exit["provider_clob_order_id"] == "clob-hormuz-exit"
            assert pending_exit["exit_order_id"] == "venue-exit-order-hormuz"
            assert pending_exit["provider_status"] == "open"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_open_order_counts_dedupe_provider_clob_rows_without_payload_authority(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-dedupe-open-counts"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add_all(
                [
                    TraderOrder(
                        id="live-order-open-a",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="market-mavs-cavs",
                        market_question="Mavericks vs. Cavaliers",
                        direction="buy_no",
                        mode="live",
                        status="open",
                        notional_usd=4.47,
                        entry_price=0.895,
                        effective_price=0.895,
                        provider_clob_order_id="clob-mavs-cavs-open",
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        payload_json={},
                    ),
                    TraderOrder(
                        id="live-order-open-b",
                        trader_id=trader_id,
                        source="crypto",
                        strategy_key="tail_end_carry",
                        market_id="market-mavs-cavs",
                        market_question="Mavericks vs. Cavaliers",
                        direction="buy_no",
                        mode="live",
                        status="open",
                        notional_usd=4.47,
                        entry_price=0.895,
                        effective_price=0.895,
                        provider_clob_order_id="clob-mavs-cavs-open",
                        created_at=now,
                        executed_at=now,
                        updated_at=now,
                        payload_json={},
                    ),
                ]
            )
            await session.commit()

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
            open_orders = await get_open_order_count_for_trader(session, trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")

            assert open_orders == 1
            assert open_positions == 1

            position_rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.id.asc())
                )
            ).scalars().all()
            assert len(position_rows) == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_promotes_failed_existing_entry_with_live_authority(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-promote-existing"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-existing-order",
                signal_id="signal-existing-order",
                market_id="market-manchester-ou",
                market_question="Manchester City FC vs. Liverpool FC: O/U 4.5",
                direction="buy_yes",
                strategy_key="basic",
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="failed-existing-order",
                    trader_id=trader_id,
                    signal_id="signal-existing-order",
                    decision_id="decision-existing-order",
                    source="scanner",
                    strategy_key="basic",
                    market_id="market-manchester-ou",
                    market_question="Manchester City FC vs. Liverpool FC: O/U 4.5",
                    direction="buy_yes",
                    mode="live",
                    status="failed",
                    notional_usd=5.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    reason="Basic Arbitrage: signal selected",
                    payload_json={
                        "token_id": "token-over",
                        "provider_clob_order_id": "clob-manchester-over",
                    },
                    provider_clob_order_id="clob-manchester-over",
                    created_at=now,
                    updated_at=now,
                    error_message="timeout waiting for provider",
                    verification_status="local",
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-manchester-over",
                    wallet_address="0xwallet",
                    clob_order_id="clob-manchester-over",
                    token_id="token-over",
                    side="BUY",
                    price=0.5,
                    size=10.0,
                    order_type="IOC",
                    status="filled",
                    filled_size=10.0,
                    average_fill_price=0.5,
                    market_question="Manchester City FC vs. Liverpool FC: O/U 4.5",
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

            refreshed = await session.get(TraderOrder, "failed-existing-order")
            assert refreshed is not None
            assert refreshed.status == "executed"
            assert refreshed.error_message is None
            assert refreshed.provider_clob_order_id == "clob-manchester-over"
            assert refreshed.verification_status == "venue_fill"
            payload = dict(refreshed.payload_json or {})
            assert payload["live_wallet_authority"]["live_trading_order_id"] == "venue-order-manchester-over"
            assert payload["provider_reconciliation"]["snapshot_status"] == "filled"
            open_orders = await get_open_order_count_for_trader(session, trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_orders == 0
            assert open_positions == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_refreshes_existing_authority_fill_metrics_from_order_size(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-refresh-authority-fill-metrics"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-refresh-authority-fill-metrics",
                signal_id="signal-refresh-authority-fill-metrics",
                market_id="market-refresh-authority-fill-metrics",
                market_question="Will Example City reach 70F?",
                direction="buy_yes",
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="existing-recovered-order",
                    trader_id=trader_id,
                    signal_id="signal-refresh-authority-fill-metrics",
                    decision_id="decision-refresh-authority-fill-metrics",
                    source="scanner",
                    strategy_key="generic_strategy",
                    market_id="market-refresh-authority-fill-metrics",
                    market_question="Will Example City reach 70F?",
                    direction="buy_yes",
                    mode="live",
                    status="executed",
                    notional_usd=28.71,
                    entry_price=0.86,
                    effective_price=0.87,
                    reason="Recovered from live venue authority",
                    payload_json={
                        "token_id": "token-refresh-authority-fill-metrics",
                        "provider_clob_order_id": "clob-refresh-authority-fill-metrics",
                        "provider_reconciliation": {
                            "source": "live_trading_orders_recovery",
                            "snapshot_status": "filled",
                            "mapped_status": "executed",
                            "reconciled_at": now.isoformat(),
                            "filled_size": 33.0,
                            "average_fill_price": 0.87,
                            "filled_notional_usd": 28.71,
                            "snapshot": {
                                "normalized_status": "filled",
                                "status": "filled",
                                "side": "BUY",
                                "clob_order_id": "clob-refresh-authority-fill-metrics",
                                "asset_id": "token-refresh-authority-fill-metrics",
                                "filled_size": 33.0,
                                "average_fill_price": 0.87,
                                "filled_notional_usd": 28.71,
                                "limit_price": 0.86,
                                "size": 11.0,
                                "updated_at": now.isoformat(),
                            },
                        },
                        "live_wallet_authority": {
                            "source": "live_trading_orders",
                            "live_trading_order_id": "venue-order-refresh-authority-fill-metrics",
                            "wallet_address": "0xwallet",
                            "token_id": "token-refresh-authority-fill-metrics",
                            "recovered_at": now.isoformat(),
                        },
                    },
                    provider_clob_order_id="clob-refresh-authority-fill-metrics",
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                    verification_status="venue_fill",
                    verification_source="live_wallet_authority",
                    verification_reason="Recovered from live venue authority",
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-refresh-authority-fill-metrics",
                    wallet_address="0xwallet",
                    market_id="market-refresh-authority-fill-metrics",
                    clob_order_id="clob-refresh-authority-fill-metrics",
                    token_id="token-refresh-authority-fill-metrics",
                    side="BUY",
                    price=0.86,
                    size=11.0,
                    order_type="IOC",
                    status="filled",
                    filled_size=0.0,
                    average_fill_price=0.87,
                    market_question="Will Example City reach 70F?",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                LiveTradingPosition(
                    id="0xwallet:token-refresh-authority-fill-metrics",
                    wallet_address="0xwallet",
                    token_id="token-refresh-authority-fill-metrics",
                    market_id="market-refresh-authority-fill-metrics",
                    market_question="Will Example City reach 70F?",
                    outcome="Yes",
                    size=33.0,
                    average_cost=0.87,
                    current_price=0.84,
                    unrealized_pnl=-0.99,
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

            refreshed = await session.get(TraderOrder, "existing-recovered-order")
            assert refreshed is not None
            assert refreshed.status == "executed"
            assert refreshed.notional_usd == pytest.approx(9.57)
            payload = dict(refreshed.payload_json or {})
            provider_reconciliation = payload.get("provider_reconciliation")
            provider_reconciliation = provider_reconciliation if isinstance(provider_reconciliation, dict) else {}
            snapshot = provider_reconciliation.get("snapshot")
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            assert provider_reconciliation.get("filled_size") == pytest.approx(11.0)
            assert provider_reconciliation.get("filled_notional_usd") == pytest.approx(9.57)
            assert snapshot.get("filled_size") == pytest.approx(11.0)
            assert snapshot.get("filled_notional_usd") == pytest.approx(9.57)
            assert dict(payload.get("live_wallet_authority") or {}).get("live_trading_order_id") == "venue-order-refresh-authority-fill-metrics"

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
            position = (
                await session.execute(
                    select(TraderPosition).where(
                        TraderPosition.trader_id == trader_id,
                        TraderPosition.market_id == "market-refresh-authority-fill-metrics",
                        TraderPosition.status == "open",
                    )
                )
            ).scalar_one()
            assert position.total_notional_usd == pytest.approx(9.57)
            assert position.open_order_count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_finalizes_unfilled_existing_entry_with_failed_live_authority(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-finalize-unfilled-existing"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-open-existing-order",
                signal_id="signal-open-existing-order",
                market_id="market-golf-winner",
                market_question="Will Golfer X win the tournament?",
                direction="buy_no",
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="open-existing-order",
                    trader_id=trader_id,
                    signal_id="signal-open-existing-order",
                    decision_id="decision-open-existing-order",
                    source="scanner",
                    strategy_key="generic_strategy",
                    market_id="market-golf-winner",
                    market_question="Will Golfer X win the tournament?",
                    direction="buy_no",
                    mode="live",
                    status="open",
                    notional_usd=9.3,
                    entry_price=0.86,
                    effective_price=0.86,
                    reason="Signal selected",
                    payload_json={},
                    created_at=now,
                    updated_at=now,
                    verification_status="local",
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-golf-failed",
                    wallet_address="0xwallet",
                    clob_order_id=None,
                    token_id="token-golf-no",
                    side="BUY",
                    price=0.86,
                    size=10.8,
                    order_type="IOC",
                    status="failed",
                    filled_size=0.0,
                    average_fill_price=0.0,
                    market_question="Will Golfer X win the tournament?",
                    error_message="PolyApiException[status_code=400, error_message={'error': 'no orders found to match with FAK order. FAK orders are partially filled or killed if no match is found.'}]",
                    created_at=now,
                    updated_at=now,
                )
            )
            existing = await session.get(TraderOrder, "open-existing-order")
            existing.provider_order_id = "venue-order-golf-failed"
            await session.commit()

            result = await recover_missing_live_trader_orders(
                session,
                trader_ids=[trader_id],
                commit=True,
                broadcast=False,
            )

            assert result["recovered_orders"] == 0
            assert result["adopted_existing_orders"] == 1

            refreshed = await session.get(TraderOrder, "open-existing-order")
            assert refreshed is not None
            assert refreshed.status == "rejected"
            assert refreshed.notional_usd == pytest.approx(0.0)
            assert refreshed.executed_at is None
            assert "no orders found to match with FAK order" in str(refreshed.error_message or "")
            payload = dict(refreshed.payload_json or {})
            provider_reconciliation = payload.get("provider_reconciliation")
            provider_reconciliation = provider_reconciliation if isinstance(provider_reconciliation, dict) else {}
            assert provider_reconciliation.get("snapshot_status") == "failed"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_ignores_old_terminal_live_orders(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-ignore-old-terminal-authority"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            await _seed_decision(
                session,
                trader_id=trader_id,
                decision_id="decision-ignore-old-terminal-authority",
                signal_id="signal-ignore-old-terminal-authority",
                market_id="market-old-terminal-authority",
                market_question="Will Example Corp close above $50?",
                direction="buy_yes",
            )
            stale_terminal_at = utcnow() - timedelta(hours=2)
            session.add(
                LiveTradingOrder(
                    id="venue-order-old-terminal-authority",
                    wallet_address="0xwallet",
                    clob_order_id=None,
                    token_id="token-old-terminal-authority",
                    side="BUY",
                    price=0.86,
                    size=10.0,
                    order_type="IOC",
                    status="failed",
                    filled_size=0.0,
                    average_fill_price=0.0,
                    market_question="Will Example Corp close above $50?",
                    error_message="stale terminal venue row",
                    created_at=stale_terminal_at,
                    updated_at=stale_terminal_at,
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
            recovered_rows = (
                await session.execute(select(TraderOrder).where(TraderOrder.trader_id == trader_id))
            ).scalars().all()
            assert recovered_rows == []
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_carries_full_bundle_signal_metadata(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-bundle-authority"
    try:
        async with session_factory() as session:
            await _seed_trader(
                session,
                trader_id,
                source_configs=[
                    {
                        "source_key": "scanner",
                        "strategy_key": "generic_strategy",
                        "strategy_params": {
                            "take_profit_pct": 9,
                            "stop_loss_pct": 11,
                        },
                    }
                ],
            )
            now = utcnow()
            session.add(
                TradeSignal(
                    id="signal-bundle-authority",
                    source="scanner",
                    signal_type="opportunity",
                    strategy_type="generic_strategy",
                    market_id="market-bundle-1",
                    market_question="Will Team A win?",
                    direction="buy_yes",
                    dedupe_key="bundle-authority",
                    payload_json={
                        "is_guaranteed": True,
                        "positions_to_take": [
                            {"action": "BUY", "outcome": "YES", "token_id": "token-yes", "market": "Will Team A win?"},
                            {"action": "BUY", "outcome": "NO", "token_id": "token-no", "market": "Will Team A win?"},
                        ],
                        "execution_plan": {
                            "plan_id": "plan-bundle-authority",
                            "metadata": {
                                "full_bundle_execution_required": True,
                                "full_bundle_execution_mode": "live_ioc_complete_or_flatten",
                            },
                            "legs": [
                                {
                                    "leg_id": "leg-yes",
                                    "market_id": "market-bundle-1",
                                    "market_question": "Will Team A win?",
                                    "token_id": "token-yes",
                                    "side": "buy",
                                    "outcome": "yes",
                                    "limit_price": 0.44,
                                },
                                {
                                    "leg_id": "leg-no",
                                    "market_id": "market-bundle-1",
                                    "market_question": "Will Team A win?",
                                    "token_id": "token-no",
                                    "side": "buy",
                                    "outcome": "no",
                                    "limit_price": 0.45,
                                },
                            ],
                        },
                    },
                    created_at=now - timedelta(minutes=1),
                    updated_at=now - timedelta(minutes=1),
                )
            )
            session.add(
                TraderDecision(
                    id="decision-bundle-authority",
                    trader_id=trader_id,
                    signal_id="signal-bundle-authority",
                    source="scanner",
                    strategy_key="generic_strategy",
                    strategy_version=1,
                    decision="selected",
                    created_at=now - timedelta(minutes=1),
                )
            )
            session.add(
                LiveTradingOrder(
                    id="venue-order-bundle-authority",
                    wallet_address="0xwallet",
                    clob_order_id="clob-bundle-authority",
                    token_id="token-yes",
                    side="BUY",
                    price=0.44,
                    size=5.0,
                    order_type="IOC",
                    status="filled",
                    filled_size=5.0,
                    average_fill_price=0.44,
                    market_question="Will Team A win?",
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

            assert result["recovered_orders"] == 1

            recovered_rows = (
                await session.execute(select(TraderOrder).where(TraderOrder.trader_id == trader_id).order_by(TraderOrder.created_at.asc()))
            ).scalars().all()
            assert len(recovered_rows) == 1
            assert recovered_rows[0].status == "executed"
            payload = dict(recovered_rows[0].payload_json or {})
            execution_plan = payload.get("execution_plan")
            execution_plan = execution_plan if isinstance(execution_plan, dict) else {}
            metadata = execution_plan.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            assert payload.get("is_guaranteed") is True
            assert payload.get("strategy_type") == "generic_strategy"
            assert payload.get("strategy_exit_config", {}).get("take_profit_pct") == 9
            assert payload.get("strategy_exit_config", {}).get("stop_loss_pct") == 11
            assert len(payload.get("positions_to_take") or []) == 2
            assert metadata.get("full_bundle_execution_required") is True
            assert metadata.get("full_bundle_execution_mode") == "live_ioc_complete_or_flatten"
            open_orders = await get_open_order_count_for_trader(session, trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_orders == 0
            assert open_positions == 1
            position_rows = (
                await session.execute(
                    select(TraderPosition)
                    .where(TraderPosition.trader_id == trader_id)
                    .order_by(TraderPosition.created_at.asc())
                )
            ).scalars().all()
            assert len(position_rows) == 1
            position_payload = dict(position_rows[0].payload_json or {})
            assert position_payload.get("strategy_exit_config", {}).get("take_profit_pct") == 9
            assert position_payload.get("strategy_exit_config", {}).get("stop_loss_pct") == 11
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_recover_missing_live_trader_orders_imports_multiple_distinct_orders_on_same_market(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-multiple-authority-same-market"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow() - timedelta(minutes=3)
            session.add(
                TradeSignal(
                    id="signal-multiple-authority",
                    source="scanner",
                    signal_type="entry",
                    strategy_type="generic_strategy",
                    market_id="market-multiple-authority",
                    market_question="Will Player A win the match?",
                    direction="buy_yes",
                    entry_price=0.36,
                    dedupe_key="dedupe-multiple-authority",
                    payload_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                TraderDecision(
                    id="decision-multiple-authority",
                    trader_id=trader_id,
                    signal_id="signal-multiple-authority",
                    source="scanner",
                    strategy_key="generic_strategy",
                    strategy_version=1,
                    decision="selected",
                    reason="Generic signal selected",
                    created_at=now,
                )
            )
            session.add_all(
                [
                    LiveTradingOrder(
                        id="venue-order-multiple-authority-a",
                        wallet_address="0xwallet",
                        market_id="market-multiple-authority",
                        clob_order_id="clob-multiple-authority-a",
                        token_id="token-multiple-authority",
                        side="BUY",
                        price=0.36,
                        size=7.8,
                        order_type="GTC",
                        status="filled",
                        filled_size=7.8,
                        average_fill_price=0.36,
                        market_question="Will Player A win the match?",
                        created_at=now + timedelta(seconds=4),
                        updated_at=now + timedelta(seconds=20),
                    ),
                    LiveTradingOrder(
                        id="venue-order-multiple-authority-b",
                        wallet_address="0xwallet",
                        market_id="market-multiple-authority",
                        clob_order_id="clob-multiple-authority-b",
                        token_id="token-multiple-authority",
                        side="BUY",
                        price=0.36,
                        size=7.8,
                        order_type="GTC",
                        status="filled",
                        filled_size=7.8,
                        average_fill_price=0.36,
                        market_question="Will Player A win the match?",
                        created_at=now + timedelta(seconds=8),
                        updated_at=now + timedelta(seconds=24),
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

            assert result["recovered_orders"] == 2
            recovered_rows = (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(TraderOrder.provider_clob_order_id.asc())
                )
            ).scalars().all()
            assert [row.provider_clob_order_id for row in recovered_rows] == [
                "clob-multiple-authority-a",
                "clob-multiple-authority-b",
            ]
            assert sum(float(row.notional_usd or 0.0) for row in recovered_rows) == pytest.approx(5.616)

            position_rows = (
                await session.execute(
                    select(TraderPosition).where(TraderPosition.trader_id == trader_id)
                )
            ).scalars().all()
            assert len(position_rows) == 1
            assert position_rows[0].total_notional_usd == pytest.approx(5.616)
            assert position_rows[0].open_order_count == 2
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


@pytest.mark.asyncio
async def test_cleanup_caps_candidates_per_call_oldest_first(tmp_path):
    """Regression: a trader with 50+ stuck unfilled live orders used to
    blow the 20s maintenance step budget every cycle.  The cleanup now
    processes at most _OPEN_ORDER_CLEANUP_MAX_CANDIDATES_PER_CALL per
    invocation, oldest-first, with the rest reported as
    candidates_deferred so the orchestrator can log the backlog."""
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "shadow-cleanup-cap"
    cap = trader_orchestrator_state._OPEN_ORDER_CLEANUP_MAX_CANDIDATES_PER_CALL
    over_cap = cap + 5
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            base = utcnow()
            for i in range(over_cap):
                # Older orders get smaller indices => seeded earlier, so
                # they should be the ones picked first by the cap.
                age_seconds = 1000 - i
                session.add(
                    TraderOrder(
                        id=f"order-cap-{i:03d}",
                        trader_id=trader_id,
                        source="crypto",
                        market_id=f"market-{i}",
                        direction="buy_yes",
                        mode="shadow",
                        status="open",
                        notional_usd=0.0,
                        entry_price=0.5,
                        effective_price=0.5,
                        payload_json={},
                        created_at=base - timedelta(seconds=age_seconds),
                        updated_at=base - timedelta(seconds=age_seconds),
                    )
                )
            await session.commit()

            result = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope="shadow",
                max_age_seconds=10.0,
                source="crypto",
                require_unfilled=True,
                dry_run=False,
                target_status="cancelled",
                reason="test_cap",
            )

            assert result["candidates_total"] == over_cap, (
                f"all {over_cap} orders should match the filter pre-cap"
            )
            assert result["candidates_deferred"] == over_cap - cap, (
                "deferred count should equal total minus per-call cap"
            )
            assert result["matched"] == cap, "matched is the post-cap candidate count"
            assert result["updated"] == cap, "exactly cap orders should be cancelled"

            # The OLDEST `cap` orders should have been processed.  In our
            # seed, that's order-cap-000 .. order-cap-{cap-1}.
            for i in range(cap):
                row = await session.get(TraderOrder, f"order-cap-{i:03d}")
                assert row is not None
                assert row.status == "cancelled", (
                    f"order-cap-{i:03d} (oldest set) should be cancelled this cycle"
                )
            # The FRESHER orders (indices cap..over_cap) are deferred.
            for i in range(cap, over_cap):
                row = await session.get(TraderOrder, f"order-cap-{i:03d}")
                assert row is not None
                assert row.status == "open", (
                    f"order-cap-{i:03d} (fresher) should be deferred to a future cycle"
                )
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Plan 0024: sync_trader_position_inventory UPSERT regression tests
# ---------------------------------------------------------------------------
#
# The function used to INSERT new TraderPosition rows when the in-memory
# `existing_by_identity` snapshot missed an identity tuple. Under
# concurrent callers (or after circuit_breaker_safe_exit left stale
# closed rows in the same identity tuple), the INSERT collided with
# `uq_trader_position_identity` and raised IntegrityError. Plan 0024
# replaced the INSERT with `pg_insert(...).on_conflict_do_update(...)`
# so the DB owns the conflict resolution.


@pytest.mark.asyncio
async def test_sync_trader_position_inventory_upsert_reopens_existing_closed_row(tmp_path):
    """A closed position in the same (trader, mode, market_id, direction)
    identity must be re-opened on the next sync, not crash with
    IntegrityError. This is the exact production scenario that took
    down the Sandbox bot via circuit_breaker after Phase 0 caps."""
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "upsert-reopen-trader"
    market_id = "0xabc-reopen"
    direction = "buy_no"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            # Pre-seed a CLOSED position on the same identity that the
            # next sync will hit. This mirrors the post-circuit-breaker
            # state where safe_exit force-closed positions but their
            # rows remain in the table.
            session.add(
                TraderPosition(
                    id="pre-closed-position",
                    trader_id=trader_id,
                    mode="shadow",
                    market_id=market_id,
                    market_question="Pre-closed market",
                    direction=direction,
                    status="closed",
                    open_order_count=0,
                    total_notional_usd=0.0,
                    avg_entry_price=0.55,
                    first_order_at=now - timedelta(hours=2),
                    last_order_at=now - timedelta(hours=1),
                    closed_at=now - timedelta(hours=1),
                    payload_json={"sync_source": "manual_close", "preserved_key": "keep-me"},
                    created_at=now - timedelta(hours=2),
                    updated_at=now - timedelta(hours=1),
                )
            )
            # Fresh active order on the same identity — sync should
            # re-open the closed position, not INSERT a duplicate.
            session.add(
                TraderOrder(
                    id="reopen-order",
                    trader_id=trader_id,
                    source="traders",
                    market_id=market_id,
                    market_question="Pre-closed market",
                    direction=direction,
                    mode="shadow",
                    status="open",
                    notional_usd=10.0,
                    entry_price=0.6,
                    effective_price=0.6,
                    payload_json={"token_id": "tok-1"},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            result = await sync_trader_position_inventory(
                session, trader_id=trader_id, mode="shadow"
            )
            assert result["closures"] == 0
            # The pre-existing row was found in the snapshot, so the
            # call is counted as an update (not insert). The actual
            # write goes through pg_insert ON CONFLICT.
            assert result["updates"] == 1
            assert result["inserts"] == 0

            # Verify exactly ONE position row exists for the identity
            # — i.e. UPSERT did not create a duplicate.
            position_rows = (
                await session.execute(
                    select(TraderPosition).where(
                        TraderPosition.trader_id == trader_id,
                        TraderPosition.mode == "shadow",
                        TraderPosition.market_id == market_id,
                        TraderPosition.direction == direction,
                    )
                )
            ).scalars().all()
            assert len(position_rows) == 1
            row = position_rows[0]
            assert row.id == "pre-closed-position", (
                "UPSERT must update the existing row, not replace its id"
            )
            assert row.status == "open"
            assert row.closed_at is None
            assert row.open_order_count == 1
            # payload_json merge: existing keys preserved unless
            # overwritten by new bucket; new keys added.
            assert row.payload_json["preserved_key"] == "keep-me"
            assert row.payload_json["sync_source"] == "order_inventory"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sync_trader_position_inventory_upsert_no_integrity_error_on_concurrent_seed(tmp_path):
    """Simulate the race that the IntegrityError was firing on:
    `existing_by_identity` snapshot misses a row that DOES exist in
    the DB at write time. Pre-Plan-0024 this raised IntegrityError;
    Plan 0024 makes the UPSERT detect conflict and update instead.

    Simulates the race by directly inserting a row via raw SQL AFTER
    monkey-patching the SELECT to return empty (so the in-memory
    snapshot is empty), then calling sync to drive the UPSERT path."""
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "upsert-race-trader"
    market_id = "0xabc-race"
    direction = "buy_yes"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            # Pre-seed an open position directly via ORM. This row
            # WILL be found by sync's SELECT.
            session.add(
                TraderPosition(
                    id="race-existing",
                    trader_id=trader_id,
                    mode="shadow",
                    market_id=market_id,
                    market_question="Race market",
                    direction=direction,
                    status="open",
                    open_order_count=1,
                    total_notional_usd=5.0,
                    avg_entry_price=0.5,
                    first_order_at=now - timedelta(minutes=5),
                    last_order_at=now - timedelta(minutes=5),
                    closed_at=None,
                    payload_json={"sync_source": "order_inventory"},
                    created_at=now - timedelta(minutes=5),
                    updated_at=now - timedelta(minutes=5),
                )
            )
            session.add(
                TraderOrder(
                    id="race-order",
                    trader_id=trader_id,
                    source="traders",
                    market_id=market_id,
                    market_question="Race market",
                    direction=direction,
                    mode="shadow",
                    status="open",
                    notional_usd=10.0,
                    entry_price=0.55,
                    effective_price=0.55,
                    payload_json={"token_id": "tok-race"},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            # Should not raise IntegrityError. Pre-Plan-0024 the
            # function would have UPDATED the existing row in the
            # ORM; with UPSERT it goes through ON CONFLICT and ends
            # at the same final state.
            result = await sync_trader_position_inventory(
                session, trader_id=trader_id, mode="shadow"
            )
            assert result["updates"] == 1

            # Final invariant: exactly one position row, with refreshed
            # state from the new order.
            position_rows = (
                await session.execute(
                    select(TraderPosition).where(
                        TraderPosition.trader_id == trader_id,
                        TraderPosition.mode == "shadow",
                        TraderPosition.market_id == market_id,
                        TraderPosition.direction == direction,
                    )
                )
            ).scalars().all()
            assert len(position_rows) == 1
            assert position_rows[0].open_order_count == 1
            assert position_rows[0].total_notional_usd == 10.0
    finally:
        await engine.dispose()
