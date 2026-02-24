import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TraderOrder
from services.market_cache import CachedMarket
from services.trader_orchestrator_state import list_serialized_trader_orders
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_order_market_links")


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_enriches_payload_from_cached_market(tmp_path):
    condition_id = "0xf0eac4f4ef5b0f09952b9635beba71e2130fc1ded8da4ba1c5b29a2bc79b34e8"
    market_slug = "jazz-vs-rockets-over-under-228-5"
    event_slug = "nba-jazz-rockets-2026-02-23"
    expected_url = f"https://polymarket.com/event/{event_slug}/{market_slug}"

    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            CachedMarket(
                condition_id=condition_id,
                question="Jazz vs Rockets",
                slug=market_slug,
                extra_data={
                    "id": "123456",
                    "condition_id": condition_id,
                    "event_slug": event_slug,
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_1",
                trader_id="trader_1",
                source="traders",
                market_id=condition_id,
                market_question="Jazz vs Rockets",
                mode="paper",
                status="open",
                payload_json={},
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_1", limit=20)

    await engine.dispose()
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["market_slug"] == market_slug
    assert payload["event_slug"] == event_slug
    assert payload["condition_id"] == condition_id
    assert payload["market_url"] == expected_url
    assert payload["polymarket_url"] == expected_url


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_keeps_existing_payload_metadata(tmp_path):
    condition_id = "0x6f4a82c35b64e93d8f6299654f65e82c53d53f42d8df95de57d802c62af70a26"
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            CachedMarket(
                condition_id=condition_id,
                question="Cache metadata should not override payload metadata",
                slug="cache-market-slug",
                extra_data={
                    "id": "777777",
                    "condition_id": condition_id,
                    "event_slug": "cache-event-slug",
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_2",
                trader_id="trader_2",
                source="traders",
                market_id=condition_id,
                market_question="Manual payload metadata",
                mode="paper",
                status="open",
                payload_json={
                    "market_slug": "payload-market-slug",
                    "event_slug": "payload-event-slug",
                    "market_url": "https://polymarket.com/market/payload-market-slug",
                },
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_2", limit=20)

    await engine.dispose()
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["market_slug"] == "payload-market-slug"
    assert payload["event_slug"] == "payload-event-slug"
    assert payload["market_url"] == "https://polymarket.com/market/payload-market-slug"
    assert payload["polymarket_url"] == "https://polymarket.com/market/payload-market-slug"
