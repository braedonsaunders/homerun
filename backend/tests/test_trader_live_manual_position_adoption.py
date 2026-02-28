import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Strategy, Trader, TraderOrder
from services import trader_orchestrator_state
from services.trader_orchestrator_state import (
    adopt_live_wallet_position,
    list_live_wallet_positions_for_trader,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow

CONDITION_ID = "0x" + ("1" * 64)
TOKEN_YES = "token-yes-123"
TOKEN_NO = "token-no-456"
WALLET_ADDRESS = "0xabc1230000000000000000000000000000000000"
TRADER_ID = "trader-live-1"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _seed_live_trader(session, *, strategy_key: str = "manual_wallet_position") -> None:
    now = utcnow()
    session.add(
        Trader(
            id=TRADER_ID,
            name="Manual Live Adopt",
            description="Live trader for manual position adoption tests",
            source_configs_json=[{"source_key": "manual", "strategy_key": strategy_key, "strategy_params": {}}],
            risk_limits_json={},
            metadata_json={},
            mode="live",
            is_enabled=True,
            is_paused=False,
            interval_seconds=30,
            created_at=now,
            updated_at=now,
        )
    )
    await session.commit()


async def _seed_exit_only_strategy(session, slug: str) -> None:
    now = utcnow()
    source_code = "\n".join(
        [
            "from services.strategies.base import BaseStrategy, ExitDecision",
            "",
            "class ManualExitManagerStrategy(BaseStrategy):",
            "    name = 'Manual Exit Manager'",
            "    description = 'Exit-only strategy for adopted positions'",
            "    def should_exit(self, position, market_state):",
            "        return ExitDecision(action='hold', reason='wait')",
        ]
    )
    session.add(
        Strategy(
            id=f"strategy-{slug}",
            slug=slug,
            source_key="manual",
            name="Manual Exit Manager",
            description="Exit-only strategy for adopted positions",
            class_name="ManualExitManagerStrategy",
            source_code=source_code,
            config={},
            config_schema={},
            aliases=[],
            enabled=True,
            is_system=False,
            status="loaded",
            version=1,
            sort_order=0,
            created_at=now,
            updated_at=now,
        )
    )
    await session.commit()


def _wallet_positions_payload() -> list[dict]:
    return [
        {
            "asset": TOKEN_YES,
            "conditionId": CONDITION_ID,
            "title": "Will BTC close above $100k on March 1?",
            "outcome": "Yes",
            "size": 20.0,
            "avgPrice": 0.41,
            "curPrice": 0.58,
            "initialValue": 8.2,
            "currentValue": 11.6,
            "cashPnl": 3.4,
        }
    ]


def _market_info_payload() -> dict:
    return {
        "condition_id": CONDITION_ID,
        "question": "Will BTC close above $100k on March 1?",
        "slug": "btc-above-100k-march-1",
        "event_slug": "btc-march-1-2026",
        "token_ids": [TOKEN_YES, TOKEN_NO],
        "outcomes": ["Yes", "No"],
        "yes_price": 0.58,
        "no_price": 0.42,
    }


@pytest_asyncio.fixture(scope="function")
async def postgres_session_factory():
    engine, session_factory = await build_postgres_session_factory(Base, "trader_live_manual_position_adoption")
    try:
        yield session_factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _reset_database_state(postgres_session_factory):
    table_names = [_quote_ident(table.name) for table in Base.metadata.sorted_tables]
    if not table_names:
        return
    async with postgres_session_factory() as session:
        await session.execute(text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE"))
        await session.commit()


@pytest.mark.asyncio
async def test_adopt_manual_live_wallet_position_creates_open_live_order(postgres_session_factory, monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_state,
        "_resolve_execution_wallet_address",
        AsyncMock(return_value=WALLET_ADDRESS),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_wallet_positions_with_prices",
        AsyncMock(return_value=_wallet_positions_payload()),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_market_by_condition_id",
        AsyncMock(return_value=_market_info_payload()),
    )

    async with postgres_session_factory() as session:
        await _seed_live_trader(session)

        before = await list_live_wallet_positions_for_trader(session, trader_id=TRADER_ID)
        assert before["wallet_address"] == WALLET_ADDRESS
        assert before["summary"]["unmanaged_positions"] == 1
        assert before["positions"][0]["is_managed"] is False

        adopted = await adopt_live_wallet_position(
            session,
            trader_id=TRADER_ID,
            token_id=TOKEN_YES,
            reason="manual_import_test",
        )
        assert adopted["status"] == "adopted"
        assert adopted["token_id"] == TOKEN_YES
        order_id = str(adopted["order"]["id"])

        order = await session.get(TraderOrder, order_id)
        assert order is not None
        assert order.mode == "live"
        assert order.status == "open"
        assert order.source == "manual_wallet_position"
        assert order.strategy_key == "manual_wallet_position"

        payload = dict(order.payload_json or {})
        assert payload["token_id"] == TOKEN_YES
        assert payload["selected_token_id"] == TOKEN_YES
        assert payload["condition_id"] == CONDITION_ID
        assert payload["position_state"]["last_mark_source"] == "wallet_mark"
        assert payload["provider_reconciliation"]["snapshot"]["filled_size"] == pytest.approx(20.0)

        after = await list_live_wallet_positions_for_trader(session, trader_id=TRADER_ID)
        assert after["summary"]["managed_positions"] == 1
        assert after["positions"][0]["is_managed"] is True
        assert after["positions"][0]["managed_order_id"] == order_id


@pytest.mark.asyncio
async def test_adopt_manual_live_wallet_position_rejects_duplicate_token(postgres_session_factory, monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_state,
        "_resolve_execution_wallet_address",
        AsyncMock(return_value=WALLET_ADDRESS),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_wallet_positions_with_prices",
        AsyncMock(return_value=_wallet_positions_payload()),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_market_by_condition_id",
        AsyncMock(return_value=_market_info_payload()),
    )

    async with postgres_session_factory() as session:
        await _seed_live_trader(session)
        await adopt_live_wallet_position(session, trader_id=TRADER_ID, token_id=TOKEN_YES)

        with pytest.raises(ValueError, match="already managed"):
            await adopt_live_wallet_position(session, trader_id=TRADER_ID, token_id=TOKEN_YES)


@pytest.mark.asyncio
async def test_adopt_manual_live_wallet_position_uses_trader_strategy_key(postgres_session_factory, monkeypatch):
    strategy_slug = "manual_position_manager"
    monkeypatch.setattr(
        trader_orchestrator_state,
        "_resolve_execution_wallet_address",
        AsyncMock(return_value=WALLET_ADDRESS),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_wallet_positions_with_prices",
        AsyncMock(return_value=_wallet_positions_payload()),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_market_by_condition_id",
        AsyncMock(return_value=_market_info_payload()),
    )

    async with postgres_session_factory() as session:
        await _seed_live_trader(session, strategy_key=strategy_slug)
        await _seed_exit_only_strategy(session, strategy_slug)

        adopted = await adopt_live_wallet_position(
            session,
            trader_id=TRADER_ID,
            token_id=TOKEN_YES,
        )
        assert adopted["status"] == "adopted"
        assert adopted["strategy_key"] == strategy_slug

        order = await session.get(TraderOrder, str(adopted["order"]["id"]))
        assert order is not None
        assert order.strategy_key == strategy_slug
        payload = dict(order.payload_json or {})
        assert payload["strategy_type"] == strategy_slug


@pytest.mark.asyncio
async def test_adopt_manual_live_wallet_position_requires_configured_strategy(postgres_session_factory, monkeypatch):
    monkeypatch.setattr(
        trader_orchestrator_state,
        "_resolve_execution_wallet_address",
        AsyncMock(return_value=WALLET_ADDRESS),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_wallet_positions_with_prices",
        AsyncMock(return_value=_wallet_positions_payload()),
    )
    monkeypatch.setattr(
        trader_orchestrator_state.polymarket_client,
        "get_market_by_condition_id",
        AsyncMock(return_value=_market_info_payload()),
    )

    async with postgres_session_factory() as session:
        await _seed_live_trader(session, strategy_key="")

        with pytest.raises(ValueError, match="configured strategy"):
            await adopt_live_wallet_position(
                session,
                trader_id=TRADER_ID,
                token_id=TOKEN_YES,
            )
