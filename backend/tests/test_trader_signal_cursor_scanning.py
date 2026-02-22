import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from utils.utcnow import utcnow

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, Trader  # noqa: E402
from services.trader_orchestrator_state import (  # noqa: E402
    get_trader_signal_cursor,
    list_unconsumed_trade_signals,
    record_signal_consumption,
    upsert_trader_signal_cursor,
)
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_signal_cursor")


@pytest.mark.asyncio
async def test_unconsumed_signals_default_to_pending_and_use_cursor(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            s1 = TradeSignal(
                id=uuid.uuid4().hex,
                source="crypto",
                source_item_id="a",
                signal_type="crypto_worker_15m",
                strategy_type="crypto_15m",
                market_id="m1",
                direction="buy_yes",
                dedupe_key="d1",
                status="pending",
                created_at=now,
                updated_at=now,
            )
            s2 = TradeSignal(
                id=uuid.uuid4().hex,
                source="crypto",
                source_item_id="b",
                signal_type="crypto_worker_15m",
                strategy_type="crypto_15m",
                market_id="m2",
                direction="buy_yes",
                dedupe_key="d2",
                status="submitted",
                created_at=now + timedelta(seconds=1),
                updated_at=now + timedelta(seconds=1),
            )
            s3 = TradeSignal(
                id=uuid.uuid4().hex,
                source="crypto",
                source_item_id="c",
                signal_type="crypto_worker_15m",
                strategy_type="crypto_15m",
                market_id="m3",
                direction="buy_no",
                dedupe_key="d3",
                status="pending",
                created_at=now + timedelta(seconds=2),
                updated_at=now + timedelta(seconds=2),
            )
            trader = Trader(
                id=trader_id,
                name="Cursor Trader",
                strategy_key="crypto_15m",
            )
            session.add_all([trader, s1, s2, s3])
            await session.commit()

            initial = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=["crypto"],
                limit=10,
            )
            assert [row.id for row in initial] == [s1.id, s3.id]

            await upsert_trader_signal_cursor(
                session,
                trader_id=trader_id,
                last_signal_created_at=s1.created_at,
                last_signal_id=s1.id,
            )
            cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(session, trader_id=trader_id)
            assert cursor_created_at == s1.created_at
            assert cursor_signal_id == s1.id

            remaining = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=["crypto"],
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                limit=10,
            )
            assert [row.id for row in remaining] == [s3.id]

            await record_signal_consumption(
                session,
                trader_id=trader_id,
                signal_id=s3.id,
                outcome="skipped",
            )
            after_consumption = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=["crypto"],
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                limit=10,
            )
            assert after_consumption == []

            s3.updated_at = s3.updated_at + timedelta(seconds=30)
            await session.commit()

            refreshed = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=["crypto"],
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                limit=10,
            )
            assert [row.id for row in refreshed] == [s3.id]

            await record_signal_consumption(
                session,
                trader_id=trader_id,
                signal_id=s3.id,
                outcome="selected",
            )
            refreshed_consumed = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=["crypto"],
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                limit=10,
            )
            assert refreshed_consumed == []

            with_submitted = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=["crypto"],
                statuses=["pending", "submitted"],
                limit=10,
            )
            assert [row.id for row in with_submitted] == [s1.id, s2.id]
    finally:
        await engine.dispose()
