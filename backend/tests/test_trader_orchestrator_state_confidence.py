import sys
from datetime import datetime
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, LiveTradingPosition
from services.trader_orchestrator_state import (
    _normalize_trader_payload,
    get_gross_exposure,
    read_orchestrator_snapshot,
    write_orchestrator_snapshot,
)
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_state_confidence")


@pytest.mark.asyncio
async def test_normalize_trader_payload_converts_percent_min_confidence(tmp_path):
    payload = {
        "name": "Crypto HF Trader",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "btc_eth_maker_quote",
                "strategy_params": {"min_confidence": 45, "min_edge_percent": 3.0},
            }
        ],
    }
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        normalized = await _normalize_trader_payload(session, payload)
    await engine.dispose()
    assert normalized["source_configs"][0]["strategy_params"]["min_confidence"] == 0.45


@pytest.mark.asyncio
async def test_normalize_trader_payload_preserves_fraction_min_confidence(tmp_path):
    payload = {
        "name": "Crypto HF Trader",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "btc_eth_maker_quote",
                "strategy_params": {"min_confidence": 0.45},
            }
        ],
    }
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        normalized = await _normalize_trader_payload(session, payload)
    await engine.dispose()
    assert normalized["source_configs"][0]["strategy_params"]["min_confidence"] == 0.45


@pytest.mark.asyncio
async def test_get_gross_exposure_floors_live_exposure_from_wallet_positions(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            now = datetime.utcnow()
            session.add(
                LiveTradingPosition(
                    id="0xwallet:token-1",
                    wallet_address="0xwallet",
                    token_id="token-1",
                    market_id="market-1",
                    market_question="Will market 1 settle YES?",
                    outcome="Yes",
                    size=11.4,
                    average_cost=0.9,
                    current_price=1.0,
                    unrealized_pnl=1.14,
                    redeemable=False,
                    counts_as_open=True,
                    end_date=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            assert await get_gross_exposure(session, mode="live") == pytest.approx(11.4)
            assert await get_gross_exposure(session) == pytest.approx(11.4)
            assert await get_gross_exposure(session, mode="shadow") == 0.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_orchestrator_snapshot_keeps_wallet_exposure_when_disabled(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            now = datetime.utcnow()
            session.add(
                LiveTradingPosition(
                    id="0xwallet:token-1",
                    wallet_address="0xwallet",
                    token_id="token-1",
                    market_id="market-1",
                    market_question="Will market 1 settle YES?",
                    outcome="Yes",
                    size=11.4,
                    average_cost=0.9,
                    current_price=1.0,
                    unrealized_pnl=1.14,
                    redeemable=False,
                    counts_as_open=True,
                    end_date=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await write_orchestrator_snapshot(
                session,
                running=False,
                enabled=False,
                current_activity="Disabled",
                interval_seconds=5,
                stats={"gross_exposure_usd": 0.0, "open_orders": 0},
            )
            snapshot = await read_orchestrator_snapshot(session)

            assert snapshot["gross_exposure_usd"] == pytest.approx(11.4)
            assert snapshot["stats"]["gross_exposure_usd"] == pytest.approx(11.4)
            assert snapshot["stats"]["wallet_gross_exposure_floor_usd"] == pytest.approx(11.4)
    finally:
        await engine.dispose()
