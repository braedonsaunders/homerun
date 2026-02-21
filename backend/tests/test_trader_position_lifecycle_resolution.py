import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, Trader, TraderOrder
from services.trader_orchestrator import position_lifecycle
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_position_lifecycle_resolution")


async def _seed_order(
    session: AsyncSession,
    *,
    direction: str = "buy_yes",
    order_id: str = "order-1",
    signal_id: str = "signal-1",
) -> None:
    now = datetime.utcnow()
    session.add(
        Trader(
            id="trader-1",
            name="Crypto Trader",
            strategy_key="crypto_15m",
            strategy_version="v1",
            sources_json=["crypto"],
            params_json={},
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
            mode="paper",
            status="executed",
            notional_usd=40.0,
            entry_price=0.4,
            effective_price=0.4,
            created_at=now,
            executed_at=now,
            updated_at=now,
        )
    )
    await session.commit()


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

            result = await position_lifecycle.reconcile_paper_positions(
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

            result = await position_lifecycle.reconcile_paper_positions(
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

            result = await position_lifecycle.reconcile_paper_positions(
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
