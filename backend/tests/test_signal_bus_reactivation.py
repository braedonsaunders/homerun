import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from utils.utcnow import utcnow

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, TradeSignalEmission  # noqa: E402
from models.opportunity import Opportunity  # noqa: E402
from services import strategy_signal_bridge  # noqa: E402
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
async def test_upsert_keeps_unchanged_skipped_scanner_signal_skipped(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            prior = now - timedelta(minutes=4)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="scanner",
                    source_item_id="scanner_old",
                    signal_type="scanner_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_scanner_skipped",
                    market_question="Scanner skipped signal",
                    direction="buy_yes",
                    entry_price=0.44,
                    edge_percent=6.0,
                    confidence=0.72,
                    liquidity=88.0,
                    expires_at=now + timedelta(hours=1),
                    status="skipped",
                    dedupe_key="dedupe_scanner_skipped",
                    payload_json={"id": "stable", "ingested_at": "2026-02-24T00:00:00Z"},
                    strategy_context_json={"mode": "directional", "bridge_run_at": "2026-02-24T00:00:00Z"},
                    created_at=prior,
                    updated_at=prior,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="scanner",
                source_item_id="scanner_old",
                signal_type="scanner_opportunity",
                strategy_type="custom_strategy",
                market_id="market_scanner_skipped",
                market_question="Scanner skipped signal",
                direction="buy_yes",
                entry_price=0.44,
                edge_percent=6.0,
                confidence=0.72,
                liquidity=88.0,
                expires_at=now + timedelta(hours=1),
                payload_json={"id": "stable", "ingested_at": "2026-02-24T00:00:05Z"},
                strategy_context_json={"mode": "directional", "bridge_run_at": "2026-02-24T00:00:05Z"},
                dedupe_key="dedupe_scanner_skipped",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "skipped"
            assert refreshed.updated_at == prior

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
async def test_upsert_skips_when_incoming_runtime_sequence_is_stale(tmp_path):
    """Cross-process row-lock contention guard: when the existing row's
    ``runtime_sequence`` is already >= the incoming sequence, the upsert
    must short-circuit without modifying the row, taking row locks, or
    emitting an audit event.

    The intent_runtime projection loop runs in every worker process and
    does an in-memory sequence gate at projection-pop time, but that
    snapshot was captured before the transaction started.  By the time
    we reach the DB, another worker may have already committed a fresher
    state.  This test pins the DB-level fallback gate so stale upserts
    don't compete for the row lock — that contention was the dominant
    bottleneck after the orchestrator audit-buffer fixes (e6c78e9 /
    4493734) shifted the lock pattern from orchestrator status writes
    onto the projection upsert path.
    """
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            existing = TradeSignal(
                id=signal_id,
                source="custom_source",
                source_item_id="seq_existing",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_seq_1",
                market_question="Sequence-gated signal",
                direction="buy_yes",
                entry_price=0.55,
                edge_percent=8.0,
                confidence=0.80,
                liquidity=500.0,
                expires_at=now + timedelta(hours=1),
                status="pending",
                dedupe_key="dedupe_seq_gate",
                payload_json={"id": "fresh", "version": 5},
                strategy_context_json={"version": 5},
                quality_passed=True,
                runtime_sequence=110,
                created_at=now,
                updated_at=now,
            )
            session.add(existing)
            await session.commit()

            # Stale upsert: incoming sequence (100) < existing (110).
            # Should be a no-op — return the existing row, no UPDATE
            # emitted, no audit event written.
            returned = await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="seq_stale",  # different — must NOT be applied
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_seq_1",
                market_question="Stale rewrite (should not stick)",
                direction="buy_no",  # inverted — must NOT be applied
                entry_price=0.99,  # garbage — must NOT be applied
                edge_percent=99.0,
                confidence=0.10,
                liquidity=1.0,
                expires_at=now + timedelta(hours=99),
                payload_json={"id": "STALE", "version": 1},
                strategy_context_json={"version": 1},
                quality_passed=False,
                quality_rejection_reasons=["stale_sequence"],
                dedupe_key="dedupe_seq_gate",
                runtime_sequence=100,
                commit=True,
            )

            # Returned row must be the existing one, with all original
            # values intact.
            assert returned is not None
            assert returned.id == signal_id
            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.source_item_id == "seq_existing"
            assert refreshed.direction == "buy_yes"
            assert refreshed.entry_price == pytest.approx(0.55)
            assert refreshed.edge_percent == pytest.approx(8.0)
            assert refreshed.runtime_sequence == 110
            assert refreshed.payload_json == {"id": "fresh", "version": 5}
            assert refreshed.updated_at == now

            # No audit row should have been written for the stale call.
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

            # Equal sequence is also stale — same short-circuit.
            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="seq_stale_equal",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_seq_1",
                market_question="Equal-sequence rewrite",
                direction="buy_no",
                entry_price=0.99,
                edge_percent=99.0,
                confidence=0.10,
                liquidity=1.0,
                expires_at=now + timedelta(hours=99),
                payload_json={"id": "STALE_EQUAL"},
                strategy_context_json={},
                dedupe_key="dedupe_seq_gate",
                runtime_sequence=110,  # equal to existing
                commit=True,
            )
            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed.source_item_id == "seq_existing"
            assert refreshed.runtime_sequence == 110

            # Fresh upsert (sequence > existing) must go through and
            # apply the new field values.
            await upsert_trade_signal(
                session,
                source="custom_source",
                source_item_id="seq_fresh",
                signal_type="custom_opportunity",
                strategy_type="custom_strategy",
                market_id="market_seq_1",
                market_question="Fresh signal",
                direction="buy_yes",
                entry_price=0.60,
                edge_percent=9.5,
                confidence=0.85,
                liquidity=600.0,
                expires_at=now + timedelta(hours=2),
                payload_json={"id": "fresher", "version": 6},
                strategy_context_json={"version": 6},
                quality_passed=True,
                dedupe_key="dedupe_seq_gate",
                runtime_sequence=120,
                commit=True,
            )
            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed.source_item_id == "seq_fresh"
            assert refreshed.entry_price == pytest.approx(0.60)
            assert refreshed.edge_percent == pytest.approx(9.5)
            assert refreshed.runtime_sequence == 120
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_pending_scanner_signal_refreshes_volatile_payload_without_emission(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    signal_id = uuid.uuid4().hex
    try:
        async with session_factory() as session:
            now = utcnow().replace(microsecond=0)
            session.add(
                TradeSignal(
                    id=signal_id,
                    source="scanner",
                    source_item_id="pending_scanner_1",
                    signal_type="scanner_opportunity",
                    strategy_type="custom_strategy",
                    market_id="market_pending_scanner_1",
                    market_question="Pending scanner signal",
                    direction="buy_yes",
                    entry_price=0.51,
                    edge_percent=5.2,
                    confidence=0.73,
                    liquidity=1200.0,
                    expires_at=now + timedelta(seconds=10),
                    status="pending",
                    dedupe_key="dedupe_pending_scanner_refresh",
                    payload_json={
                        "id": "pending",
                        "markets": [
                            {
                                "id": "market_pending_scanner_1",
                                "price_age_seconds": 1.0,
                                "price_updated_at": "2026-04-25T16:00:00Z",
                                "outcome_prices": [0.51, 0.49],
                            }
                        ],
                        "signal_emitted_at": "2026-04-25T16:00:00Z",
                    },
                    strategy_context_json={"strategy_mode": "steady"},
                    quality_passed=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await upsert_trade_signal(
                session,
                source="scanner",
                source_item_id="pending_scanner_1",
                signal_type="scanner_opportunity",
                strategy_type="custom_strategy",
                market_id="market_pending_scanner_1",
                market_question="Pending scanner signal",
                direction="buy_yes",
                entry_price=0.511,
                edge_percent=5.24,
                confidence=0.732,
                liquidity=1225.0,
                expires_at=now + timedelta(seconds=40),
                payload_json={
                    "id": "pending",
                    "markets": [
                        {
                            "id": "market_pending_scanner_1",
                            "price_age_seconds": 4.0,
                            "price_updated_at": "2026-04-25T16:00:30Z",
                            "outcome_prices": [0.511, 0.489],
                        }
                    ],
                    "signal_emitted_at": "2026-04-25T16:00:30Z",
                },
                strategy_context_json={"strategy_mode": "steady"},
                quality_passed=True,
                quality_rejection_reasons=None,
                dedupe_key="dedupe_pending_scanner_refresh",
                commit=True,
            )

            refreshed = await session.get(TradeSignal, signal_id)
            assert refreshed is not None
            assert refreshed.status == "pending"
            assert refreshed.expires_at == now + timedelta(seconds=40)
            assert refreshed.payload_json["markets"][0]["price_age_seconds"] == pytest.approx(4.0)

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
async def test_expire_stale_signals_keeps_runtime_revalidated_scanner_signal_pending(tmp_path):
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
                    source_item_id="scanner_runtime_revalidated_1",
                    signal_type="scanner_opportunity",
                    strategy_type="custom_scanner_strategy",
                    market_id="scanner_market_runtime_revalidated_1",
                    direction="buy_yes",
                    dedupe_key="dedupe_scanner_runtime_revalidated_stale",
                    status="pending",
                    payload_json={
                        "strategy_runtime": {"execution_activation": "ws_current"},
                        "markets": [{"clob_token_ids": ["yes", "no"]}],
                    },
                    strategy_context_json={"execution_activation": "ws_current"},
                    expires_at=now + timedelta(hours=2),
                    created_at=stale_created,
                    updated_at=stale_created,
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
async def test_list_trade_signals_keeps_runtime_revalidated_scanner_signal_pending(tmp_path):
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
                    source_item_id="scanner_runtime_revalidated_list_1",
                    signal_type="scanner_opportunity",
                    strategy_type="custom_scanner_strategy",
                    market_id="scanner_market_runtime_revalidated_list_1",
                    direction="buy_yes",
                    dedupe_key="dedupe_scanner_runtime_revalidated_list_stale",
                    status="pending",
                    payload_json={
                        "strategy_runtime": {"execution_activation": "ws_current"},
                        "markets": [{"clob_token_ids": ["yes", "no"]}],
                    },
                    strategy_context_json={"execution_activation": "ws_current"},
                    expires_at=now + timedelta(hours=2),
                    created_at=stale_created,
                    updated_at=stale_created,
                )
            )
            await session.commit()

            rows = await list_trade_signals(session, source="scanner", status=None, limit=10, offset=0)
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
async def test_bridge_refreshes_prices_before_upsert(monkeypatch):
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

    fake_runtime = SimpleNamespace(
        started=True,
        prewarm_source_tokens=AsyncMock(return_value=None),
        publish_opportunities=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(strategy_signal_bridge, "get_intent_runtime", lambda: fake_runtime)

    emitted = await bridge_opportunities_to_signals(
        [opportunity],
        source="weather",
        sweep_missing=False,
        refresh_prices=True,
    )
    assert emitted == 1
    fake_runtime.prewarm_source_tokens.assert_called_once_with([opportunity], source="weather")
    call_kwargs = fake_runtime.publish_opportunities.call_args
    assert call_kwargs.kwargs.get("refresh_prices") is True
    assert call_kwargs.kwargs.get("sweep_missing") is False
    assert call_kwargs.args[0] == [opportunity]


@pytest.mark.asyncio
async def test_bridge_serializes_datetime_strategy_context(monkeypatch):
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

    fake_runtime = SimpleNamespace(
        started=True,
        prewarm_source_tokens=AsyncMock(return_value=None),
        publish_opportunities=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(strategy_signal_bridge, "get_intent_runtime", lambda: fake_runtime)

    emitted = await bridge_opportunities_to_signals(
        [opportunity],
        source="traders",
        sweep_missing=True,
        refresh_prices=False,
    )
    assert emitted == 1
    fake_runtime.publish_opportunities.assert_called_once()
    call_args = fake_runtime.publish_opportunities.call_args
    passed_opps = call_args.args[0]
    assert len(passed_opps) == 1
    assert passed_opps[0] is opportunity
    call_kwargs = call_args.kwargs
    assert call_kwargs.get("source") == "traders"
    assert call_kwargs.get("sweep_missing") is True
