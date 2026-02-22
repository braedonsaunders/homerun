import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader, TraderOrder
from tests.postgres_test_db import build_postgres_session_factory
from workers import trader_orchestrator_worker


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_orchestrator_paper_backfill")


@pytest.mark.asyncio
async def test_backfill_passes_paper_simulation_fee_and_slippage_to_ledger(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            now = datetime.utcnow()
            session.add(
                Trader(
                    id="trader-1",
                    name="Backfill Trader",
                    strategy_key="btc_eth_highfreq",
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
                TraderOrder(
                    id="order-1",
                    trader_id="trader-1",
                    signal_id=None,
                    source="crypto",
                    market_id="market-1",
                    market_question="Will this backfill?",
                    direction="buy_yes",
                    mode="paper",
                    status="executed",
                    notional_usd=50.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={
                        "paper_simulation": {
                            "estimated_fee_usd": 1.2,
                            "slippage_usd": 0.75,
                            "fill_ratio": 0.9,
                        }
                    },
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            record_mock = AsyncMock(
                return_value={
                    "account_id": "paper-1",
                    "trade_id": "trade-1",
                    "position_id": "position-1",
                }
            )
            monkeypatch.setattr(
                trader_orchestrator_worker.simulation_service,
                "record_orchestrator_paper_fill",
                record_mock,
            )

            result = await trader_orchestrator_worker._backfill_simulation_ledger_for_active_paper_orders(
                session,
                trader_id="trader-1",
                paper_account_id="paper-1",
            )
            await session.flush()

            order = await session.get(TraderOrder, "order-1")
            assert order is not None
            assert result["attempted"] == 1
            assert result["backfilled"] == 1
            assert result["errors"] == []
            assert isinstance((order.payload_json or {}).get("simulation_ledger"), dict)

            record_mock.assert_awaited_once()
            kwargs = record_mock.await_args.kwargs
            assert kwargs["execution_fee_usd"] == pytest.approx(1.2, rel=1e-9)
            assert kwargs["execution_slippage_usd"] == pytest.approx(0.75, rel=1e-9)
            assert kwargs["payload"]["paper_simulation"]["fill_ratio"] == pytest.approx(0.9, rel=1e-9)
    finally:
        await engine.dispose()
