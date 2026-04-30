import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, TraderOrder
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
                mode="shadow",
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
                mode="shadow",
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


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_uses_cached_market_outcome_labels_for_direction(tmp_path):
    condition_id = "0x5e20ce52f9449f89d6a14341a26167e6eea76a3ec78357d5a17ad5df85c74bc2"
    yes_label = "BOS"
    no_label = "DEN"

    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            CachedMarket(
                condition_id=condition_id,
                question="BOS vs DEN",
                slug="bos-vs-den",
                extra_data={
                    "condition_id": condition_id,
                    "outcome_labels": [yes_label, no_label],
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_3",
                trader_id="trader_3",
                source="traders",
                market_id=condition_id,
                market_question="BOS vs DEN",
                direction="buy_yes",
                mode="shadow",
                status="open",
                payload_json={},
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_3", limit=20)

    await engine.dispose()
    assert len(rows) == 1
    row = rows[0]
    assert row["direction_side"] == "YES"
    assert row["direction_label"] == yes_label
    assert row["yes_label"] == yes_label
    assert row["no_label"] == no_label


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_includes_trade_bundle_metadata_for_multileg_signal(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            TradeSignal(
                id="signal_bundle_1",
                source="scanner",
                signal_type="opportunity",
                strategy_type="settlement_lag",
                market_id="market-atletico",
                market_question="Will Atlético Nacional win on 2026-04-01?",
                direction="buy_yes",
                dedupe_key="bundle-signal-1",
                payload_json={
                    "total_cost": 0.865,
                    "expected_payout": 1.0,
                    "gross_profit": 0.135,
                    "net_profit": 0.112405,
                    "roi_percent": 12.9948,
                    "is_guaranteed": True,
                    "roi_type": "guaranteed_spread",
                    "mispricing_type": "settlement_lag",
                    "market_roster": {
                        "market_count": 2,
                        "markets": [
                            {
                                "id": "market-atletico",
                                "condition_id": "market-atletico",
                                "question": "Will AtlÃ©tico Nacional win on 2026-04-01?",
                            },
                            {
                                "id": "market-cucuta",
                                "condition_id": "market-cucuta",
                                "question": "Will CÃºcuta Deportivo FC win on 2026-04-01?",
                            },
                        ],
                    },
                    "positions_to_take": [
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": "Will Atlético Nacional win on 2026-04-01?",
                            "price": 0.79,
                            "token_id": "token-atletico",
                        },
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": "Will Cúcuta Deportivo FC win on 2026-04-01?",
                            "price": 0.075,
                            "token_id": "token-cucuta",
                        },
                    ],
                    "execution_plan": {
                        "plan_id": "plan-bundle-1",
                        "legs": [
                            {
                                "leg_id": "leg_1",
                                "market_id": "market-atletico",
                                "market_question": "Will Atlético Nacional win on 2026-04-01?",
                                "token_id": "token-atletico",
                                "side": "buy",
                                "outcome": "yes",
                                "limit_price": 0.79,
                                "notional_weight": 0.79,
                            },
                            {
                                "leg_id": "leg_2",
                                "market_id": "market-cucuta",
                                "market_question": "Will Cúcuta Deportivo FC win on 2026-04-01?",
                                "token_id": "token-cucuta",
                                "side": "buy",
                                "outcome": "yes",
                                "limit_price": 0.075,
                                "notional_weight": 0.075,
                            },
                        ],
                    },
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_bundle_1",
                trader_id="trader_bundle_1",
                signal_id="signal_bundle_1",
                source="scanner",
                market_id="market-atletico",
                market_question="Will Atlético Nacional win on 2026-04-01?",
                direction="buy_yes",
                mode="shadow",
                status="open",
                payload_json={
                    "token_id": "token-atletico",
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_bundle_2",
                trader_id="trader_bundle_1",
                signal_id="signal_bundle_1",
                source="scanner",
                market_id="market-cucuta",
                market_question="Will Cúcuta Deportivo FC win on 2026-04-01?",
                direction="buy_yes",
                mode="shadow",
                status="open",
                payload_json={
                    "token_id": "token-cucuta",
                },
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_bundle_1", limit=20)

    await engine.dispose()
    assert len(rows) == 2
    row_by_id = {row["id"]: row for row in rows}
    bundle = row_by_id["order_bundle_1"]["trade_bundle"]
    assert bundle is not None
    assert bundle["bundle_id"] == "signal_bundle_1"
    assert bundle["plan_id"] == "plan-bundle-1"
    assert bundle["kind"] == "multi_outcome_yes"
    assert bundle["leg_count"] == 2
    assert bundle["is_guaranteed"] is True
    assert bundle["signal_is_guaranteed"] is True
    assert bundle["guarantee_reason"] == "full_market_roster"
    assert bundle["total_cost"] == pytest.approx(0.865)
    assert bundle["expected_payout"] == pytest.approx(1.0)
    assert bundle["net_profit"] == pytest.approx(0.112405)
    assert bundle["current_leg_id"] == "leg_1"
    assert bundle["current_leg_index"] == 0
    assert [leg["token_id"] for leg in bundle["legs"]] == ["token-atletico", "token-cucuta"]
    assert [leg["notional_weight"] for leg in bundle["legs"]] == pytest.approx([0.79, 0.075])


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_hides_failed_nonmaterialized_bundle_attempts(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            TradeSignal(
                id="signal_bundle_materialized",
                source="scanner",
                signal_type="opportunity",
                strategy_type="generic_strategy",
                market_id="market-1",
                market_question="Will Team A win?",
                direction="buy_yes",
                dedupe_key="bundle-materialized",
                payload_json={
                    "is_guaranteed": True,
                    "roi_type": "guaranteed_spread",
                    "positions_to_take": [
                        {"action": "BUY", "outcome": "YES", "market": "Will Team A win?", "price": 0.44, "token_id": "token-yes"},
                        {"action": "BUY", "outcome": "NO", "market": "Will Team A win?", "price": 0.45, "token_id": "token-no"},
                    ],
                    "execution_plan": {
                        "plan_id": "plan-materialized",
                        "legs": [
                            {"leg_id": "leg-yes", "market_id": "market-1", "market_question": "Will Team A win?", "token_id": "token-yes", "side": "buy", "outcome": "yes", "limit_price": 0.44},
                            {"leg_id": "leg-no", "market_id": "market-1", "market_question": "Will Team A win?", "token_id": "token-no", "side": "buy", "outcome": "no", "limit_price": 0.45},
                        ],
                    },
                    "markets": [
                        {
                            "id": "market-1",
                            "condition_id": "market-1",
                            "question": "Will Team A win?",
                            "clob_token_ids": ["token-yes", "token-no"],
                        }
                    ],
                    "market_roster": {
                        "market_count": 1,
                        "markets": [
                            {
                                "id": "market-1",
                                "condition_id": "market-1",
                                "question": "Will Team A win?",
                            }
                        ],
                    },
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_bundle_materialized_yes",
                trader_id="trader_bundle_materialized",
                signal_id="signal_bundle_materialized",
                source="scanner",
                market_id="market-1",
                market_question="Will Team A win?",
                direction="buy_yes",
                mode="live",
                status="open",
                verification_status="venue_order",
                verification_source="live_order_ack",
                provider_order_id="provider-yes",
                payload_json={"token_id": "token-yes"},
            )
        )
        session.add(
            TraderOrder(
                id="order_bundle_materialized_no",
                trader_id="trader_bundle_materialized",
                signal_id="signal_bundle_materialized",
                source="scanner",
                market_id="market-1",
                market_question="Will Team A win?",
                direction="buy_no",
                mode="live",
                status="failed",
                verification_status="local",
                verification_source="local_runtime",
                payload_json={"token_id": "token-no"},
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_bundle_materialized", limit=20)

    await engine.dispose()
    assert len(rows) == 2
    by_id = {row["id"]: row for row in rows}
    assert by_id["order_bundle_materialized_yes"]["trade_bundle"] is None
    assert by_id["order_bundle_materialized_no"]["trade_bundle"] is None


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_demotes_unproven_guaranteed_bundle_without_market_roster(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            TradeSignal(
                id="signal_bundle_legacy",
                source="scanner",
                signal_type="opportunity",
                strategy_type="settlement_lag",
                market_id="market-draw",
                market_question="Will draw happen?",
                direction="buy_yes",
                dedupe_key="bundle-legacy-1",
                payload_json={
                    "is_guaranteed": True,
                    "positions_to_take": [
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": "Will draw happen?",
                            "price": 0.29,
                            "token_id": "token-draw",
                        },
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": "Will home side win?",
                            "price": 0.50,
                            "token_id": "token-home",
                        },
                    ],
                    "execution_plan": {
                        "plan_id": "plan-legacy-bundle",
                        "legs": [
                            {
                                "leg_id": "leg_1",
                                "market_id": "market-draw",
                                "market_question": "Will draw happen?",
                                "token_id": "token-draw",
                                "side": "buy",
                                "outcome": "yes",
                                "limit_price": 0.29,
                                "notional_weight": 0.29,
                            },
                            {
                                "leg_id": "leg_2",
                                "market_id": "market-home",
                                "market_question": "Will home side win?",
                                "token_id": "token-home",
                                "side": "buy",
                                "outcome": "yes",
                                "limit_price": 0.50,
                                "notional_weight": 0.50,
                            },
                        ],
                    },
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_bundle_legacy",
                trader_id="trader_bundle_legacy",
                signal_id="signal_bundle_legacy",
                source="scanner",
                market_id="market-draw",
                market_question="Will draw happen?",
                direction="buy_yes",
                mode="shadow",
                status="open",
                payload_json={
                    "token_id": "token-draw",
                },
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_bundle_legacy", limit=20)

    await engine.dispose()
    assert len(rows) == 1
    bundle = rows[0]["trade_bundle"]
    assert bundle is not None
    assert bundle["signal_is_guaranteed"] is True
    assert bundle["is_guaranteed"] is False
    assert bundle["guarantee_reason"] == "missing_market_roster"


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_prefers_order_execution_shape_over_mutated_signal_bundle(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            TradeSignal(
                id="signal_mutated_bundle",
                source="scanner",
                signal_type="opportunity",
                strategy_type="generic_bundle",
                market_id="market-seoul",
                market_question="Will the highest temperature in Seoul be 15°C on April 4?",
                direction="buy_yes",
                dedupe_key="mutated-bundle-1",
                payload_json={
                    "is_guaranteed": True,
                    "positions_to_take": [
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": "Will the highest temperature in Seoul be 15°C on April 4?",
                            "token_id": "token-yes",
                            "price": 0.085,
                        },
                        {
                            "action": "BUY",
                            "outcome": "NO",
                            "market": "Will the highest temperature in Seoul be 15°C on April 4?",
                            "token_id": "token-no",
                            "price": 0.845,
                        },
                    ],
                    "execution_plan": {
                        "plan_id": "plan-mutated-bundle",
                        "policy": "PAIR_LOCK",
                        "legs": [
                            {
                                "leg_id": "leg_1",
                                "market_id": "market-seoul",
                                "market_question": "Will the highest temperature in Seoul be 15°C on April 4?",
                                "token_id": "token-yes",
                                "side": "buy",
                                "outcome": "yes",
                                "limit_price": 0.085,
                            },
                            {
                                "leg_id": "leg_2",
                                "market_id": "market-seoul",
                                "market_question": "Will the highest temperature in Seoul be 15°C on April 4?",
                                "token_id": "token-no",
                                "side": "buy",
                                "outcome": "no",
                                "limit_price": 0.845,
                            },
                        ],
                    },
                },
            )
        )
        session.add(
            TraderOrder(
                id="order_mutated_bundle",
                trader_id="trader_mutated_bundle",
                signal_id="signal_mutated_bundle",
                source="scanner",
                market_id="market-seoul",
                market_question="Will the highest temperature in Seoul be 15°C on April 4?",
                direction="buy_yes",
                mode="shadow",
                status="closed_win",
                actual_profit=0.17,
                payload_json={
                    "token_id": "token-yes",
                    "leg": {
                        "leg_id": "leg_1",
                        "market_id": "market-seoul",
                        "market_question": "Will the highest temperature in Seoul be 15°C on April 4?",
                        "token_id": "token-yes",
                        "side": "buy",
                        "outcome": "yes",
                        "limit_price": 0.07,
                    },
                    "execution_session": {
                        "session_id": "session_mutated_bundle",
                        "leg_id": "leg_1_runtime",
                        "leg_ref": "leg_1",
                        "policy": "SINGLE_LEG",
                    },
                    "strategy_context": {
                        "execution_plan": {
                            "plan_id": "plan-runtime-single",
                            "policy": "SINGLE_LEG",
                            "legs": [
                                {
                                    "leg_id": "leg_1",
                                    "market_id": "market-seoul",
                                    "market_question": "Will the highest temperature in Seoul be 15°C on April 4?",
                                    "token_id": "token-yes",
                                    "side": "buy",
                                    "outcome": "yes",
                                    "limit_price": 0.07,
                                }
                            ],
                        }
                    },
                },
            )
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_mutated_bundle", limit=20)

    await engine.dispose()
    assert len(rows) == 1
    assert rows[0]["trade_bundle"] is None


@pytest.mark.asyncio
async def test_list_serialized_trader_orders_hides_unmaterialized_pair_signal_when_only_one_live_leg_is_observed(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        session.add(
            TradeSignal(
                id="signal_partial_pair",
                source="scanner",
                signal_type="opportunity",
                strategy_type="basic",
                market_id="market-london",
                market_question="Will the highest temperature in London be 15°C on April 4?",
                direction="buy_yes",
                dedupe_key="partial-pair-1",
                payload_json={
                    "is_guaranteed": True,
                    "positions_to_take": [
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": "Will the highest temperature in London be 15°C on April 4?",
                            "token_id": "token-yes",
                            "price": 0.24,
                        },
                        {
                            "action": "BUY",
                            "outcome": "NO",
                            "market": "Will the highest temperature in London be 15°C on April 4?",
                            "token_id": "token-no",
                            "price": 0.76,
                        },
                    ],
                    "execution_plan": {
                        "plan_id": "plan-partial-pair",
                        "policy": "PAIR_LOCK",
                        "legs": [
                            {
                                "leg_id": "leg_yes",
                                "market_id": "market-london",
                                "market_question": "Will the highest temperature in London be 15°C on April 4?",
                                "token_id": "token-yes",
                                "side": "buy",
                                "outcome": "yes",
                                "limit_price": 0.24,
                            },
                            {
                                "leg_id": "leg_no",
                                "market_id": "market-london",
                                "market_question": "Will the highest temperature in London be 15°C on April 4?",
                                "token_id": "token-no",
                                "side": "buy",
                                "outcome": "no",
                                "limit_price": 0.76,
                            },
                        ],
                    },
                },
            )
        )
        session.add_all(
            [
                TraderOrder(
                    id="order_partial_pair_yes",
                    trader_id="trader_partial_pair",
                    signal_id="signal_partial_pair",
                    source="scanner",
                    market_id="market-london",
                    market_question="Will the highest temperature in London be 15°C on April 4?",
                    direction="buy_yes",
                    mode="live",
                    status="cancelled",
                    payload_json={
                        "token_id": "token-yes",
                    },
                    verification_status="local",
                ),
                TraderOrder(
                    id="order_partial_pair_no",
                    trader_id="trader_partial_pair",
                    signal_id="signal_partial_pair",
                    source="scanner",
                    market_id="market-london",
                    market_question="Will the highest temperature in London be 15°C on April 4?",
                    direction="buy_no",
                    mode="live",
                    status="open",
                    payload_json={
                        "token_id": "token-no",
                        "provider_clob_order_id": "clob-token-no",
                    },
                    provider_clob_order_id="clob-token-no",
                    verification_status="wallet_position",
                ),
            ]
        )
        await session.commit()

        rows = await list_serialized_trader_orders(session, trader_id="trader_partial_pair", limit=20)

    await engine.dispose()
    assert len(rows) == 2
    assert all(row["trade_bundle"] is None for row in rows)
