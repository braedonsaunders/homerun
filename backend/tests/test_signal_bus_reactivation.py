import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from utils.utcnow import utcnow

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, TradeSignalEmission  # noqa: E402
from models.opportunity import Opportunity  # noqa: E402
from services.signal_bus import expire_stale_signals, list_trade_signals, upsert_trade_signal  # noqa: E402
from services.strategy_signal_bridge import bridge_opportunities_to_signals  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "signal_bus_reactivation")


@pytest.mark.asyncio
async def test_upsert_reactivates_expired_signal(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="custom_old",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy_old",
                    market_id="market_1",
                    market_question="Old market question",
                    direction="buy_yes",
                    entry_price=0.11,
                    edge_percent=1.0,
                    confidence=0.2,
                    liquidity=10.0,
                    expires_at=now,
                    status="expired",
                    dedupe_key="dedupe_weather_1",
                    payload_json={"id": "old"},
                    strategy_context_json={"old": True},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="custom_new",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy_new",
                market_id="market_1",
                market_question="New market question",
                direction="buy_no",
                entry_price=0.42,
                edge_percent=12.3,
                confidence=0.91,
                liquidity=321.0,
                expires_at=now,
                payload_json={"id": "new"},
                strategy_context_json={"new": True},
                dedupe_key="dedupe_weather_1",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
            assert refreshed.source_item_id == "custom_new"
            assert refreshed.strategy_type == "custom_strategy_new"
            assert refreshed.direction == "buy_no"
            assert refreshed.entry_price == pytest.approx(0.42)
            assert refreshed.edge_percent == pytest.approx(12.3)
            assert refreshed.confidence == pytest.approx(0.91)
            assert refreshed.effective_price is None

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert emissions
            assert emissions[-1].event_type == "upsert_reactivated"
            assert emissions[-1].reason == "reactivated_from:expired"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_reactivates_executed_signal(tmp_path):
    """Executed signals are in SIGNAL_REACTIVATABLE_STATUSES and get reactivated on upsert."""
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="custom_old",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_2",
                    market_question="Executed signal",
                    direction="buy_yes",
                    entry_price=0.55,
                    edge_percent=4.0,
                    confidence=0.7,
                    liquidity=99.0,
                    expires_at=now,
                    status="executed",
                    dedupe_key="dedupe_traders_2",
                    payload_json={"id": "executed"},
                    strategy_context_json={"executed": True},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="custom_new",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_2",
                market_question="Attempted overwrite",
                direction="buy_no",
                entry_price=0.22,
                edge_percent=15.0,
                confidence=0.95,
                liquidity=123.0,
                expires_at=now,
                payload_json={"id": "new"},
                strategy_context_json={"new": True},
                dedupe_key="dedupe_traders_2",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
            assert refreshed.source_item_id == "custom_new"
            assert refreshed.direction == "buy_no"
            assert refreshed.entry_price == pytest.approx(0.22)
            assert refreshed.effective_price is None

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert emissions
            assert emissions[-1].event_type == "upsert_reactivated"
            assert emissions[-1].reason == "reactivated_from:executed"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_skipped_signal_stays_skipped_when_only_bridge_metadata_changes(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="custom_old",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_3",
                    market_question="Skipped signal",
                    direction="buy_yes",
                    entry_price=0.44,
                    edge_percent=6.0,
                    confidence=0.72,
                    liquidity=88.0,
                    expires_at=now + timedelta(hours=1),
                    status="skipped",
                    dedupe_key="dedupe_traders_3",
                    payload_json={"id": "stable", "ingested_at": "2026-02-24T00:00:00Z"},
                    strategy_context_json={"mode": "directional", "bridge_run_at": "2026-02-24T00:00:00Z"},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="custom_old",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_3",
                market_question="Skipped signal",
                direction="buy_yes",
                entry_price=0.44,
                edge_percent=6.0,
                confidence=0.72,
                liquidity=88.0,
                expires_at=now + timedelta(hours=1),
                payload_json={"id": "stable", "ingested_at": "2026-02-24T00:00:05Z"},
                strategy_context_json={"mode": "directional", "bridge_run_at": "2026-02-24T00:00:05Z"},
                dedupe_key="dedupe_traders_3",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "skipped"
            assert refreshed.updated_at == now

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert not emissions
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_reactivates_skipped_signal_on_material_change(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="custom_old",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_4",
                    market_question="Skipped signal",
                    direction="buy_yes",
                    entry_price=0.44,
                    edge_percent=6.0,
                    confidence=0.72,
                    liquidity=88.0,
                    expires_at=now + timedelta(hours=1),
                    status="skipped",
                    dedupe_key="dedupe_traders_4",
                    payload_json={"id": "stable", "ingested_at": "2026-02-24T00:00:00Z"},
                    strategy_context_json={"mode": "directional"},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="custom_old",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_4",
                market_question="Skipped signal",
                direction="buy_yes",
                entry_price=0.56,
                edge_percent=6.0,
                confidence=0.72,
                liquidity=88.0,
                expires_at=now + timedelta(hours=1),
                payload_json={"id": "stable", "ingested_at": "2026-02-24T00:00:05Z"},
                strategy_context_json={"mode": "directional"},
                dedupe_key="dedupe_traders_4",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
            assert refreshed.entry_price == pytest.approx(0.56)

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert emissions
            assert emissions[-1].event_type == "upsert_reactivated"
            assert emissions[-1].reason == "reactivated_from:skipped"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_skipped_signal_ignores_micro_jitter(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="custom_old",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_4b",
                    market_question="Skipped jitter signal",
                    direction="buy_yes",
                    entry_price=0.44,
                    edge_percent=6.0,
                    confidence=0.72,
                    liquidity=5000.0,
                    expires_at=now + timedelta(hours=1),
                    status="skipped",
                    dedupe_key="dedupe_traders_4b",
                    payload_json={"oracle_status": {"availability_state": "available"}},
                    strategy_context_json={"timeframe": "1h", "bridge_run_at": "2026-02-24T00:00:00Z"},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="custom_old",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_4b",
                market_question="Skipped jitter signal",
                direction="buy_yes",
                entry_price=0.443,
                edge_percent=6.1,
                confidence=0.725,
                liquidity=5200.0,
                expires_at=now + timedelta(hours=1, seconds=10),
                payload_json={"oracle_status": {"availability_state": "available"}},
                strategy_context_json={"timeframe": "1h", "bridge_run_at": "2026-02-24T00:00:05Z"},
                dedupe_key="dedupe_traders_4b",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "skipped"
            assert refreshed.updated_at == now

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert not emissions
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_reactivates_skipped_signal_on_liquidity_band_change(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="custom_old",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_4c",
                    market_question="Skipped liquidity signal",
                    direction="buy_yes",
                    entry_price=0.44,
                    edge_percent=6.0,
                    confidence=0.72,
                    liquidity=3900.0,
                    expires_at=now + timedelta(hours=1),
                    status="skipped",
                    dedupe_key="dedupe_traders_4c",
                    payload_json={"id": "stable"},
                    strategy_context_json={"timeframe": "1h"},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="custom_old",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_4c",
                market_question="Skipped liquidity signal",
                direction="buy_yes",
                entry_price=0.44,
                edge_percent=6.0,
                confidence=0.72,
                liquidity=4100.0,
                expires_at=now + timedelta(hours=1),
                payload_json={"id": "stable"},
                strategy_context_json={"timeframe": "1h"},
                dedupe_key="dedupe_traders_4c",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
            assert refreshed.liquidity == pytest.approx(4100.0)

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert emissions
            assert emissions[-1].event_type == "upsert_reactivated"
            assert emissions[-1].reason == "reactivated_from:skipped"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_pending_signal_unchanged_suppresses_update_emission(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source",
                    source_item_id="pending_1",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_pending_1",
                    market_question="Pending signal",
                    direction="buy_yes",
                    entry_price=0.51,
                    edge_percent=5.2,
                    confidence=0.73,
                    liquidity=120.0,
                    expires_at=now + timedelta(hours=1),
                    status="pending",
                    dedupe_key="dedupe_pending_unchanged",
                    payload_json={"id": "pending", "mode": "steady"},
                    strategy_context_json={"strategy_mode": "steady"},
                    quality_passed=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="pending_1",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_pending_1",
                market_question="Pending signal",
                direction="buy_yes",
                entry_price=0.51,
                edge_percent=5.2,
                confidence=0.73,
                liquidity=120.0,
                expires_at=now + timedelta(hours=1),
                payload_json={"id": "pending", "mode": "steady"},
                strategy_context_json={"strategy_mode": "steady"},
                quality_passed=True,
                quality_rejection_reasons=None,
                dedupe_key="dedupe_pending_unchanged",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
            assert refreshed.updated_at == now

            emissions = (
                (
                    await session.execute(
                        select(TradeSignalEmission)
                        .where(TradeSignalEmission.signal_id == signal_id)
                        .order_by(TradeSignalEmission.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert not emissions
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_expire_stale_signals_skips_price_staleness_for_non_scanner_source(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source_non_scanner",
                    source_item_id="custom_1",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="custom_market_1",
                    direction="buy_yes",
                    dedupe_key="dedupe_custom_stale",
                    status="pending",
                    payload_json={"markets": [{"clob_token_ids": ["yes", "no"]}]},
                    expires_at=now + timedelta(hours=2),
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await expire_stale_signals(session, commit=True)

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_expire_stale_signals_expires_scanner_signal_without_price_timestamp(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            stale_created = now - timedelta(minutes=15)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="scanner",
                    source_item_id="scanner_1",
                    signal_type="scanner_opportunity",
                    strategy_type="custom_scanner_strategy",
                    market_id="scanner_market_1",
                    direction="buy_yes",
                    dedupe_key="dedupe_scanner_stale",
                    status="pending",
                    payload_json={"markets": [{"clob_token_ids": ["yes", "no"]}]},
                    expires_at=now + timedelta(hours=2),
                    created_at=stale_created,
                    updated_at=stale_created,
                )
            )
            await session.commit()

            await expire_stale_signals(session, commit=True)

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "expired"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_list_trade_signals_skips_price_staleness_for_non_scanner_source(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            stale_created = now - timedelta(minutes=15)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="custom_source_non_scanner",
                    source_item_id="custom_list_1",
                    signal_type="custom_opportunity",
                    strategy_type="custom_strategy",
                    market_id="custom_market_list_1",
                    direction="buy_yes",
                    dedupe_key="dedupe_custom_list_stale",
                    status="pending",
                    payload_json={"markets": [{"clob_token_ids": ["yes", "no"]}]},
                    expires_at=now + timedelta(hours=2),
                    created_at=stale_created,
                    updated_at=stale_created,
                )
            )
            await session.commit()

            rows = await list_trade_signals(
                session,
                source="custom_source_non_scanner",
                status=None,
                limit=10,
                offset=0,
            )
            assert any(row.id == signal_id for row in rows)

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_list_trade_signals_expires_scanner_signal_without_price_timestamp(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            stale_created = now - timedelta(minutes=15)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="scanner",
                    source_item_id="scanner_list_1",
                    signal_type="scanner_opportunity",
                    strategy_type="custom_scanner_strategy",
                    market_id="scanner_market_list_1",
                    direction="buy_yes",
                    dedupe_key="dedupe_scanner_list_stale",
                    status="pending",
                    payload_json={"markets": [{"clob_token_ids": ["yes", "no"]}]},
                    expires_at=now + timedelta(hours=2),
                    created_at=stale_created,
                    updated_at=stale_created,
                )
            )
            await session.commit()

            rows = await list_trade_signals(session, source="scanner", status=None, limit=10, offset=0)
            assert all(row.id != signal_id for row in rows)

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "expired"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_bridge_refreshes_prices_before_upsert(monkeypatch, tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            opportunity = Opportunity(
                strategy="weather_distribution",
                title="Weather edge",
                description="Weather edge test",
                total_cost=0.4,
                expected_payout=0.6,
                gross_profit=0.2,
                fee=0.0,
                net_profit=0.2,
                roi_percent=50.0,
                risk_score=0.2,
                confidence=0.8,
                markets=[
                    {
                        "id": "market_live_1",
                        "question": "Will it rain?",
                        "clob_token_ids": ["yes_token", "no_token"],
                        "yes_price": 0.4,
                        "no_price": 0.6,
                    }
                ],
                positions_to_take=[
                    {
                        "market_id": "market_live_1",
                        "action": "BUY",
                        "outcome": "YES",
                        "price": 0.4,
                    }
                ],
                min_liquidity=100.0,
                max_position_size=50.0,
            )

            async def _refresh_stub(opportunities, **kwargs):
                market = opportunities[0].markets[0]
                market["yes_price"] = 0.61
                market["no_price"] = 0.39
                market["price_updated_at"] = "2026-02-21T00:00:00Z"
                market["price_age_seconds"] = 0.0
                market["is_price_fresh"] = True
                opportunities[0].positions_to_take[0]["price"] = 0.61
                return opportunities

            from services.scanner import scanner as market_scanner

            monkeypatch.setattr(market_scanner, "refresh_opportunity_prices", _refresh_stub)

            emitted = await bridge_opportunities_to_signals(
                session,
                [opportunity],
                source="weather",
                sweep_missing=False,
            )
            assert emitted == 1

            rows = (await session.execute(select(TradeSignal).where(TradeSignal.source == "weather"))).scalars().all()
            assert len(rows) == 1
            row = rows[0]
            assert row.entry_price == pytest.approx(0.61)
            assert isinstance(row.payload_json, dict)
            markets = row.payload_json.get("markets")
            assert isinstance(markets, list) and markets
            assert markets[0].get("yes_price") == pytest.approx(0.61)
            assert markets[0].get("is_price_fresh") is True
            assert markets[0].get("price_updated_at") == "2026-02-21T00:00:00Z"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_bridge_serializes_datetime_strategy_context(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            observed_at = datetime.now(timezone.utc).replace(microsecond=0)
            opportunity = Opportunity(
                strategy="traders_confluence",
                title="Datetime strategy context",
                description="Bridge should serialize non-JSON-native context values.",
                total_cost=0.42,
                expected_payout=1.0,
                gross_profit=0.58,
                fee=0.0,
                net_profit=0.58,
                roi_percent=138.0,
                risk_score=0.2,
                confidence=0.77,
                markets=[
                    {
                        "id": "market_datetime_ctx",
                        "question": "Will this serialize?",
                        "clob_token_ids": ["yes_token", "no_token"],
                        "yes_price": 0.42,
                        "no_price": 0.58,
                    }
                ],
                positions_to_take=[
                    {
                        "market_id": "market_datetime_ctx",
                        "action": "BUY",
                        "outcome": "YES",
                        "price": 0.42,
                    }
                ],
                strategy_context={
                    "observed_at": observed_at,
                    "nested": {"expires_at": observed_at + timedelta(minutes=5)},
                },
            )

            emitted = await bridge_opportunities_to_signals(
                session,
                [opportunity],
                source="traders",
                sweep_missing=True,
                refresh_prices=False,
            )
            assert emitted == 1

            rows = (await session.execute(select(TradeSignal).where(TradeSignal.source == "traders"))).scalars().all()
            assert len(rows) == 1
            context = dict(rows[0].strategy_context_json or {})
            assert isinstance(context.get("observed_at"), str)
            assert isinstance((context.get("nested") or {}).get("expires_at"), str)
    finally:
        await engine.dispose()
