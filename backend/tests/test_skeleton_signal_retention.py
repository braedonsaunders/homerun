"""Retention-sweep tests for plan 0011 stuck-skeleton DELETEs.

Plan 0011 (`docs/plans/0011-skeleton-trade-signal-ttl-and-retention.md`)
introduces ``services.skeleton_signal_retention.prune_stuck_skeletons``,
a periodic DELETE that bounds the lifetime of orphaned skeleton rows
created by the publish-side skeleton-INSERT pass when
``publish_opportunities`` dies between the skeleton commit and the
projection-loop UPSERT.

Three invariants pinned here:

1. **Stuck skeletons are DELETEd.**  Old skeleton rows
   (``payload_json IS NULL`` AND ``runtime_sequence IS NULL`` AND
   ``status = 'pending'`` AND ``created_at < now() - max_age``) are
   removed outright.  No status flip — they never carried any
   consumer-visible state.
2. **In-flight skeletons are preserved.**  A fresh skeleton (younger
   than ``max_age_seconds``) MUST survive the sweep; the projection
   loop will commit at any moment.
3. **Fully-projected rows are NEVER touched.**  Any row with
   ``payload_json IS NOT NULL`` (or ``runtime_sequence IS NOT NULL``)
   has been enriched by the projection loop and is now part of the
   live signal pipeline; the retention sweep must not delete it
   regardless of age.

The test also pins idempotency: a second call after no new orphans
must DELETE 0 rows.

**Implementation note.**  The skeleton fixtures use ``pg_insert``
directly (not the ORM) to mirror the production skeleton-INSERT
shape exactly.  The ORM treats Python ``None`` on a ``JSON`` column
as JSON ``null`` (a serialized literal, not SQL NULL) unless
``none_as_null=True`` is set on the column — and
``trade_signals.payload_json`` does NOT set it.  Production code
sidesteps this by omitting ``payload_json`` from the skeleton
``pg_insert`` values dict, so the column defaults to SQL NULL.
We do the same here so the sweep's
``payload_json IS NULL`` filter matches both fixtures and prod.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal  # noqa: E402
from services.skeleton_signal_retention import prune_stuck_skeletons  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


async def _existing_signal_ids(session_factory) -> set[str]:
    async with session_factory() as session:
        rows = (await session.execute(select(TradeSignal.id))).all()
    return {str(row[0]) for row in rows}


async def _insert_skeleton_via_pg_insert(
    session_factory,
    *,
    signal_id: str,
    source: str,
    market_id: str,
    dedupe_key: str,
    created_at,
    expires_at,
) -> None:
    """Insert a skeleton-shaped row exactly the way
    ``intent_runtime.publish_opportunities`` does — via ``pg_insert``
    with ``payload_json``/``runtime_sequence`` absent so they default
    to SQL NULL.
    """
    async with session_factory() as session:
        await session.execute(
            pg_insert(TradeSignal).values(
                id=signal_id,
                source=source,
                signal_type="copy_trade",
                market_id=market_id,
                dedupe_key=dedupe_key,
                status="pending",
                expires_at=expires_at,
                created_at=created_at,
                updated_at=created_at,
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_prune_stuck_skeletons_deletes_only_aged_orphan_rows():
    engine, session_factory = await build_postgres_session_factory(
        Base, "skeleton_retention_basic"
    )
    try:
        now_naive = utcnow().replace(tzinfo=None, microsecond=0)
        fresh_created = now_naive - timedelta(seconds=10)
        stuck_created = now_naive - timedelta(hours=2)
        projected_created = now_naive - timedelta(hours=2)

        # (a) Fresh skeleton — too young to delete.
        await _insert_skeleton_via_pg_insert(
            session_factory,
            signal_id="fresh-skeleton-id",
            source="traders",
            market_id="market-fresh",
            dedupe_key="dedupe-fresh",
            created_at=fresh_created,
            expires_at=fresh_created + timedelta(minutes=5),
        )
        # (b) Stuck skeleton — orphaned by a publish that died before
        # projection.  Must be deleted.  Defensive TTL set by plan 0011
        # publish path is in the past too, so the terminal-row pruner
        # could ALSO reach this row, but only if it ran on the trading
        # plane; the retention sweep is the authoritative path for
        # orphans regardless of which plane is alive.
        await _insert_skeleton_via_pg_insert(
            session_factory,
            signal_id="stuck-skeleton-id",
            source="traders",
            market_id="market-stuck",
            dedupe_key="dedupe-stuck",
            created_at=stuck_created,
            expires_at=stuck_created + timedelta(minutes=5),
        )

        # (c) Fully-projected row — must NEVER be touched.
        # `payload_json` is non-NULL because the projection loop ran
        # and enriched the row.  ORM is fine here because we WANT a
        # non-NULL payload value.
        async with session_factory() as session:
            session.add(
                TradeSignal(
                    id="projected-row-id",
                    source="scanner",
                    source_item_id="scanner-fixture-1",
                    signal_type="basic_arb",
                    strategy_type="basic",
                    market_id="market-projected",
                    market_question="Will it happen?",
                    direction="buy_yes",
                    entry_price=0.42,
                    edge_percent=11.0,
                    confidence=0.8,
                    liquidity=2_500.0,
                    dedupe_key="dedupe-projected",
                    status="pending",
                    payload_json={"signal_emitted_at": projected_created.isoformat()},
                    strategy_context_json={"source_key": "scanner"},
                    runtime_sequence=42,
                    expires_at=now_naive + timedelta(hours=6),
                    created_at=projected_created,
                    updated_at=projected_created,
                )
            )
            await session.commit()

        # Sanity: all three fixtures present.
        assert await _existing_signal_ids(session_factory) == {
            "fresh-skeleton-id",
            "stuck-skeleton-id",
            "projected-row-id",
        }

        async with session_factory() as session:
            deleted = await prune_stuck_skeletons(
                session, max_age_seconds=60
            )
        assert deleted == 1, (
            "plan 0011 invariant: prune_stuck_skeletons must delete "
            "exactly the aged-orphan row; got "
            f"{deleted} deletions instead of 1."
        )

        survivors = await _existing_signal_ids(session_factory)
        assert survivors == {"fresh-skeleton-id", "projected-row-id"}, (
            f"plan 0011 invariant: only the stuck skeleton may be "
            f"deleted.  Survivors after sweep: {sorted(survivors)!r}; "
            f"expected fresh-skeleton-id + projected-row-id."
        )

        # Idempotency: a second call removes nothing.
        async with session_factory() as session:
            deleted_again = await prune_stuck_skeletons(
                session, max_age_seconds=60
            )
        assert deleted_again == 0, (
            f"plan 0011 invariant: re-running the sweep with no new "
            f"orphans must delete 0 rows; got {deleted_again}."
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_prune_stuck_skeletons_bounds_max_age_to_60_seconds():
    """A caller-supplied ``max_age_seconds < 60`` must be silently
    raised to 60 to avoid racing the projection loop in dev / under
    heavy load.
    """
    engine, session_factory = await build_postgres_session_factory(
        Base, "skeleton_retention_bound"
    )
    try:
        now_naive = utcnow().replace(tzinfo=None, microsecond=0)
        thirty_seconds_old = now_naive - timedelta(seconds=30)

        await _insert_skeleton_via_pg_insert(
            session_factory,
            signal_id="thirty-second-skeleton-id",
            source="traders",
            market_id="market-thirty",
            dedupe_key="dedupe-thirty",
            created_at=thirty_seconds_old,
            expires_at=thirty_seconds_old + timedelta(minutes=5),
        )

        # Caller asks for `max_age_seconds=10`, which would otherwise
        # match this 30-second-old row.  The bound must clamp to 60 s,
        # so the row survives.
        async with session_factory() as session:
            deleted = await prune_stuck_skeletons(
                session, max_age_seconds=10
            )
        assert deleted == 0, (
            "plan 0011 invariant: prune_stuck_skeletons must clamp "
            "max_age_seconds to >= 60 s so the projection loop is "
            "never raced.  Caller passed 10 s and a 30-s-old row was "
            "deleted, which means the bound is not enforced."
        )

        survivors = await _existing_signal_ids(session_factory)
        assert survivors == {"thirty-second-skeleton-id"}
    finally:
        await engine.dispose()
