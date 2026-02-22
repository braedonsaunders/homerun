import sys
import uuid
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import (  # noqa: E402
    Base,
    SimulationAccount,
    SimulationPosition,
    SimulationTrade,
    TradeStatus,
)
from services.simulation import SimulationService  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "simulation_orchestrator_ledger")


@pytest.mark.asyncio
async def test_record_and_close_orchestrator_paper_fill_updates_simulation_account(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    service = SimulationService()
    account_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            account = SimulationAccount(
                id=account_id,
                name="Paper Account",
                initial_capital=1000.0,
                current_capital=1000.0,
            )
            session.add(account)
            await session.commit()

            opened = await service.record_orchestrator_paper_fill(
                account_id=account_id,
                trader_id="trader-1",
                signal_id="signal-1",
                market_id="market-1",
                market_question="Will test pass?",
                direction="buy_yes",
                notional_usd=200.0,
                entry_price=0.5,
                strategy_type="crypto_15m",
                token_id="token-1",
                payload={"edge_percent": 8.0, "confidence": 0.9},
                session=session,
                commit=False,
            )
            await session.commit()

            refreshed_account = await session.get(SimulationAccount, account_id)
            trade = await session.get(SimulationTrade, opened["trade_id"])
            position = await session.get(SimulationPosition, opened["position_id"])

            assert refreshed_account is not None
            assert trade is not None
            assert position is not None
            assert refreshed_account.current_capital == pytest.approx(800.0, rel=1e-9)
            assert refreshed_account.total_trades == 1
            assert trade.status == TradeStatus.OPEN
            assert position.status == TradeStatus.OPEN
            assert position.quantity == pytest.approx(400.0, rel=1e-9)

            closed = await service.close_orchestrator_paper_fill(
                account_id=account_id,
                trade_id=opened["trade_id"],
                position_id=opened["position_id"],
                close_price=0.8,
                close_trigger="manual_mark_to_market",
                price_source="test",
                reason="unit_test",
                session=session,
                commit=False,
            )
            await session.commit()

            refreshed_account = await session.get(SimulationAccount, account_id)
            trade = await session.get(SimulationTrade, opened["trade_id"])
            position = await session.get(SimulationPosition, opened["position_id"])

            assert refreshed_account is not None
            assert trade is not None
            assert position is not None
            assert closed["closed"] is True
            assert closed["trade_status"] == TradeStatus.CLOSED_WIN.value
            assert closed["actual_pnl"] == pytest.approx(117.6, rel=1e-9)
            assert closed["winner_fee_usd"] == pytest.approx(2.4, rel=1e-9)
            assert closed["fees_paid"] == pytest.approx(2.4, rel=1e-9)
            assert refreshed_account.current_capital == pytest.approx(1117.6, rel=1e-9)
            assert refreshed_account.total_pnl == pytest.approx(117.6, rel=1e-9)
            assert refreshed_account.winning_trades == 1
            assert trade.status == TradeStatus.CLOSED_WIN
            assert trade.fees_paid == pytest.approx(2.4, rel=1e-9)
            assert position.status == TradeStatus.CLOSED_WIN
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_record_orchestrator_fill_rejects_when_insufficient_capital(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    service = SimulationService()
    account_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            account = SimulationAccount(
                id=account_id,
                name="Small Account",
                initial_capital=100.0,
                current_capital=100.0,
            )
            session.add(account)
            await session.commit()

            with pytest.raises(ValueError, match="Insufficient paper capital"):
                await service.record_orchestrator_paper_fill(
                    account_id=account_id,
                    trader_id="trader-1",
                    signal_id="signal-2",
                    market_id="market-2",
                    market_question="Will it fail?",
                    direction="buy_no",
                    notional_usd=250.0,
                    entry_price=0.4,
                    strategy_type="crypto_15m",
                    session=session,
                    commit=False,
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_record_orchestrator_fill_tracks_entry_fee_and_slippage(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    service = SimulationService()
    account_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            account = SimulationAccount(
                id=account_id,
                name="Costed Account",
                initial_capital=1000.0,
                current_capital=1000.0,
            )
            session.add(account)
            await session.commit()

            opened = await service.record_orchestrator_paper_fill(
                account_id=account_id,
                trader_id="trader-2",
                signal_id="signal-3",
                market_id="market-3",
                market_question="Will fees/slippage be tracked?",
                direction="buy_yes",
                notional_usd=200.0,
                entry_price=0.5,
                execution_fee_usd=3.0,
                execution_slippage_usd=1.25,
                session=session,
                commit=False,
            )
            await session.commit()

            refreshed_account = await session.get(SimulationAccount, account_id)
            trade = await session.get(SimulationTrade, opened["trade_id"])
            position = await session.get(SimulationPosition, opened["position_id"])

            assert refreshed_account is not None
            assert trade is not None
            assert position is not None
            assert opened["entry_notional_usd"] == pytest.approx(200.0, rel=1e-9)
            assert opened["entry_fee_usd"] == pytest.approx(3.0, rel=1e-9)
            assert opened["entry_slippage_usd"] == pytest.approx(1.25, rel=1e-9)
            assert opened["entry_cost"] == pytest.approx(203.0, rel=1e-9)
            assert refreshed_account.current_capital == pytest.approx(797.0, rel=1e-9)
            assert trade.total_cost == pytest.approx(203.0, rel=1e-9)
            assert trade.fees_paid == pytest.approx(3.0, rel=1e-9)
            assert trade.slippage == pytest.approx(1.25, rel=1e-9)
            assert position.entry_cost == pytest.approx(203.0, rel=1e-9)
    finally:
        await engine.dispose()
