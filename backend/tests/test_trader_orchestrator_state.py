"""Tests for ``services.trader_orchestrator_state`` helpers added by Plan 0032.

The fast-tier worker's cold-start consumed-set hydration goes through
``fetch_recent_consumed_signal_ids``: the function MUST return the
in-window ids in newest-first order and respect the limit cap so a
high-throughput trader can't blow up worker memory at start-up.
"""

import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import (  # noqa: E402
    Base,
    Trader,
    TraderSignalConsumption,
)
from services.trader_orchestrator_state import (  # noqa: E402
    fetch_recent_consumed_signal_ids,
)
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


def _seed_trader(session, trader_id: str = "fast-trader") -> None:
    now = utcnow().replace(tzinfo=None)
    session.add(
        Trader(
            id=trader_id,
            name="Fast trader",
            source_configs_json=[],
            risk_limits_json={},
            metadata_json={},
            mode="live",
            latency_class="fast",
            is_enabled=True,
            is_paused=False,
            interval_seconds=1,
            created_at=now,
            updated_at=now,
        )
    )


def _make_consumption(
    *,
    trader_id: str,
    signal_id: str,
    consumed_at,
    outcome: str = "skipped",
) -> TraderSignalConsumption:
    return TraderSignalConsumption(
        id=uuid.uuid4().hex,
        trader_id=trader_id,
        signal_id=signal_id,
        decision_id=None,
        outcome=outcome,
        reason="test",
        payload_json={},
        consumed_at=consumed_at,
    )


@pytest.mark.asyncio
async def test_fetch_recent_consumed_signal_ids_filters_window_and_orders_desc():
    engine, session_factory = await build_postgres_session_factory(
        Base, "fetch_recent_consumed_window"
    )
    try:
        now = utcnow().replace(tzinfo=None)
        async with session_factory() as session:
            _seed_trader(session)
            session.add(
                _make_consumption(
                    trader_id="fast-trader",
                    signal_id="sig-fresh",
                    consumed_at=now - timedelta(hours=1),
                )
            )
            session.add(
                _make_consumption(
                    trader_id="fast-trader",
                    signal_id="sig-mid",
                    consumed_at=now - timedelta(hours=12),
                )
            )
            session.add(
                _make_consumption(
                    trader_id="fast-trader",
                    signal_id="sig-stale",
                    consumed_at=now - timedelta(hours=72),
                )
            )
            session.add(
                _make_consumption(
                    trader_id="other-trader",
                    signal_id="sig-other",
                    consumed_at=now - timedelta(hours=1),
                )
            )
            await session.commit()

        async with session_factory() as session:
            ids = await fetch_recent_consumed_signal_ids(
                session,
                trader_id="fast-trader",
                hours=48,
            )

        assert ids == ["sig-fresh", "sig-mid"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fetch_recent_consumed_signal_ids_respects_limit_cap():
    engine, session_factory = await build_postgres_session_factory(
        Base, "fetch_recent_consumed_limit"
    )
    try:
        now = utcnow().replace(tzinfo=None)
        async with session_factory() as session:
            _seed_trader(session)
            for offset_minutes in range(10):
                session.add(
                    _make_consumption(
                        trader_id="fast-trader",
                        signal_id=f"sig-{offset_minutes:02d}",
                        consumed_at=now - timedelta(minutes=offset_minutes),
                    )
                )
            await session.commit()

        async with session_factory() as session:
            ids = await fetch_recent_consumed_signal_ids(
                session,
                trader_id="fast-trader",
                hours=48,
                limit=3,
            )

        assert ids == ["sig-00", "sig-01", "sig-02"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fetch_recent_consumed_signal_ids_empty_for_unknown_trader():
    engine, session_factory = await build_postgres_session_factory(
        Base, "fetch_recent_consumed_unknown"
    )
    try:
        async with session_factory() as session:
            ids = await fetch_recent_consumed_signal_ids(
                session,
                trader_id="never-existed",
                hours=48,
            )
        assert ids == []
        async with session_factory() as session:
            ids = await fetch_recent_consumed_signal_ids(
                session,
                trader_id="",
                hours=48,
            )
        assert ids == []
    finally:
        await engine.dispose()
