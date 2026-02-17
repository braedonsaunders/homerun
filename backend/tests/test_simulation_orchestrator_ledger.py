import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

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


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "simulation_orchestrator_ledger.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


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
            assert closed["trade_status"] == TradeStatus.RESOLVED_WIN.value
            assert closed["actual_pnl"] == pytest.approx(120.0, rel=1e-9)
            assert refreshed_account.current_capital == pytest.approx(1120.0, rel=1e-9)
            assert refreshed_account.total_pnl == pytest.approx(120.0, rel=1e-9)
            assert refreshed_account.winning_trades == 1
            assert trade.status == TradeStatus.RESOLVED_WIN
            assert position.status == TradeStatus.RESOLVED_WIN
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
