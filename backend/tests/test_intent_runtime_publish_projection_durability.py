"""Durability tests for the publish path's coordination with `trade_signals`.

Plan 0010 (`docs/plans/0010-fix-traders-publish-fk-race.md`) closes
two distinct modes of the publish/projection FK race that surface
downstream as `trader_decisions_signal_id_fkey` violations:

1. **Post-restart staleness.** A `trade_signals` row outlives a
   worker-trading restart while the in-memory cache is wiped clean.
   Naïve republish would mint a fresh uuid, the projection's
   `upsert_trade_signal` would UPDATE the existing row in place
   (keeping the OLD id), and every downstream `trader_decisions`
   write would FK-fail against the in-memory-only id.
2. **In-process publish→consume gap.** A genuinely new dedupe_key
   (no DB row, no cache entry) is consumed by an in-process
   subscriber (`trader_orchestrator_worker` for `traders` source)
   microseconds after publish — well before the asynchronous
   projection loop has committed the corresponding row.  The
   orchestrator's `_ensure_runtime_signal_persisted` is not enough
   to make the FK pass at flush time on the live worker-trading
   plane.

The fix forces both modes into a single invariant: by the time
`publish_opportunities` returns, every published `(source,
dedupe_key)` MUST have a committed row in `trade_signals` whose id
matches the in-memory cache entry.  These tests pin that invariant
end-to-end.

On `main` (pre-fix) test 1 fails because line 2174
(`signal_id = uuid.uuid4().hex`) ignores any pre-existing canonical
id.  Test 2 fails because the publish path returns BEFORE the
projection loop has committed the row, leaving downstream
consumers staring at an in-memory-only id.

Plan 0011 (`docs/plans/0011-skeleton-trade-signal-ttl-and-retention.md`)
adds a fourth invariant:

3. **Defensive TTL on the skeleton row.**  The skeleton-INSERT pass
   commits ``expires_at = now + INTENT_RUNTIME_SKELETON_TTL_SECONDS``
   so a skeleton orphaned by a mid-call publish failure becomes
   visible to the existing terminal-row pruner.  The projection
   loop's later UPSERT overwrites this defensive TTL with the
   strategy-intended ``expires_at``; the TTL only matters when
   projection never ran.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings  # noqa: E402
from models.database import Base, TradeSignal  # noqa: E402
from models.opportunity import Opportunity  # noqa: E402
from services import intent_runtime as intent_runtime_module  # noqa: E402
from services.intent_runtime import IntentRuntime  # noqa: E402
from services.signal_bus import make_dedupe_key, upsert_trade_signal  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


CANONICAL_OLD_ID = "0123456789abcdef0123456789abcdef"
COPY_TRADE_STRATEGY = "custom_copy_trade"
COPY_TRADE_MARKET_ID = "market-copy-trade-1"


def _patch_traders_strategy_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_loaded = SimpleNamespace(
        instance=SimpleNamespace(
            source_key="traders",
            subscriptions=["trader_activity"],
        )
    )
    monkeypatch.setattr(
        "services.strategy_loader.strategy_loader.get_strategy",
        lambda _slug: fake_loaded,
    )


def _make_traders_opportunity() -> Opportunity:
    return Opportunity(
        strategy=COPY_TRADE_STRATEGY,
        title="Leader bought YES on market-1",
        description="Copy-trade follow-on for fixture market-1.",
        total_cost=0.41,
        expected_payout=1.0,
        gross_profit=0.09,
        fee=0.0,
        net_profit=0.09,
        roi_percent=9.0,
        markets=[{"id": COPY_TRADE_MARKET_ID, "question": "Will it happen?"}],
        positions_to_take=[
            {
                "market_id": COPY_TRADE_MARKET_ID,
                "token_id": "trader-token-1",
                "outcome": "YES",
                "side": "buy",
                "price": 0.41,
            }
        ],
    )


def _expected_dedupe_key(opportunity: Opportunity) -> str:
    """Mirrors the computation in `intent_runtime.publish_opportunities`
    (line 2016) so the test can pre-insert a row that the publish path
    will collide with."""
    return make_dedupe_key(
        getattr(opportunity, "stable_id", None),
        getattr(opportunity, "strategy", None),
        COPY_TRADE_MARKET_ID,
    )


async def _insert_pre_restart_trade_signal(
    session_factory,
    *,
    signal_id: str,
    source: str,
    strategy_type: str,
    market_id: str,
    dedupe_key: str,
    status: str = "pending",
) -> None:
    """Insert a row that mimics the pre-restart `trade_signals` state.

    In production this row is left over from a previous worker-trading
    process that published the same wallet-trade dedupe_key before the
    deploy/restart. After restart, `intent_runtime` has an empty cache
    and would mint a fresh id for the same dedupe_key — that mismatch
    is what plan 0010 fixes.
    """
    now = utcnow().replace(microsecond=0, tzinfo=None)
    async with session_factory() as session:
        session.add(
            TradeSignal(
                id=signal_id,
                source=source,
                source_item_id="pre-restart-source-item-id",
                signal_type="copy_trade",
                strategy_type=strategy_type,
                market_id=market_id,
                market_question="Will it happen?",
                direction="buy_yes",
                entry_price=0.41,
                edge_percent=9.0,
                confidence=0.7,
                liquidity=120.0,
                expires_at=now,
                status=status,
                payload_json={"signal_emitted_at": now.isoformat()},
                strategy_context_json={"source_key": source},
                dedupe_key=dedupe_key,
                runtime_sequence=42,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def _trade_signals_by_dedupe(
    session_factory, *, source: str, dedupe_key: str
) -> list[tuple[str, str]]:
    """Return [(id, status), ...] for `(source, dedupe_key)` rows."""
    from sqlalchemy import select

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(TradeSignal.id, TradeSignal.status).where(
                        TradeSignal.source == source,
                        TradeSignal.dedupe_key == dedupe_key,
                    )
                )
            )
            .all()
        )
        return [(str(row[0]), str(row[1])) for row in rows]


@pytest.mark.asyncio
async def test_publish_adopts_existing_trade_signals_id_for_known_dedupe_key(
    monkeypatch, tmp_path
):
    """Plan 0010 fix invariant. With a pre-existing `trade_signals` row for
    a given `(source, dedupe_key)`, `publish_opportunities` MUST adopt
    that row's id rather than mint a fresh uuid.

    On `main` this fails: the in-memory id is a brand-new uuid, the
    `(source, dedupe_key)` unique constraint blocks the projection's
    insert silently (existing-by-dedupe path keeps the OLD id), and
    `_ensure_runtime_signal_persisted` later hits the same constraint
    via ON CONFLICT DO NOTHING. Net effect: the orchestrator's
    `trader_decisions.signal_id` references an id that was never
    persisted, and FK fires.
    """
    _patch_traders_strategy_loader(monkeypatch)
    engine, session_factory = await build_postgres_session_factory(
        Base, "publish_projection_durability_known_dedupe"
    )
    try:
        opportunity = _make_traders_opportunity()
        dedupe_key = _expected_dedupe_key(opportunity)

        # Pre-restart state: a `trade_signals` row for this dedupe_key
        # already exists with `CANONICAL_OLD_ID`. The in-memory cache
        # is fresh (post-restart).
        await _insert_pre_restart_trade_signal(
            session_factory,
            signal_id=CANONICAL_OLD_ID,
            source="traders",
            strategy_type=COPY_TRADE_STRATEGY,
            market_id=COPY_TRADE_MARKET_ID,
            dedupe_key=dedupe_key,
            status="pending",
        )

        # Patch the runtime's session factory so it sees the test DB.
        monkeypatch.setattr(
            intent_runtime_module, "AsyncSessionLocal", session_factory
        )

        runtime = IntentRuntime()
        runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
        # Don't actually publish to the runtime queue; we only care
        # about the in-memory cache vs. DB id alignment here.
        monkeypatch.setattr(
            intent_runtime_module,
            "publish_signal_batch",
            AsyncMock(return_value="batch-test"),
        )

        published = await runtime.publish_opportunities(
            [opportunity], source="traders", signal_type_override="copy_trade"
        )
        assert published == 1
        assert len(runtime._signals_by_id) == 1
        in_memory_id = next(iter(runtime._signals_by_id.keys()))

        # The fix: in-memory id MUST equal the canonical (pre-existing) id.
        assert in_memory_id == CANONICAL_OLD_ID, (
            f"plan 0010 invariant: post-fix `publish_opportunities` must adopt "
            f"the existing `trade_signals.id` for known `(source, dedupe_key)` "
            f"keys. Got fresh id={in_memory_id}; expected canonical "
            f"id={CANONICAL_OLD_ID}. On `main` this fails because line 2174 "
            f"(`signal_id = uuid.uuid4().hex`) ignores the DB."
        )

        snapshot = runtime._signals_by_id[in_memory_id]
        assert snapshot["dedupe_key"] == dedupe_key
        assert snapshot["source"] == "traders"

        # And `trade_signals` still has exactly one row for this
        # `(source, dedupe_key)`, with the canonical id.
        rows = await _trade_signals_by_dedupe(
            session_factory, source="traders", dedupe_key=dedupe_key
        )
        assert rows == [(CANONICAL_OLD_ID, "pending")], (
            f"trade_signals must end up with exactly one row for "
            f"(source=traders, dedupe_key={dedupe_key!r}); got {rows!r}"
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_publish_commits_skeleton_for_unknown_dedupe_key(
    monkeypatch, tmp_path
):
    """Plan 0010 fix invariant for the in-process race.  When the
    dedupe_key is genuinely new (no pre-existing row, no cache entry),
    `publish_opportunities` MUST synchronously commit a skeleton row
    into `trade_signals` keyed by the same id it stores in the
    in-memory cache.  Without this, an in-process consumer
    (`trader_orchestrator_worker` for the `traders` source) racing
    the asynchronous projection loop would FK-fail when writing
    `trader_decisions(signal_id=X)`.

    On `main` (pre-fix) this test fails because the publish path
    returns BEFORE the projection has committed; the published id is
    only in memory, not in DB.
    """
    _patch_traders_strategy_loader(monkeypatch)
    engine, session_factory = await build_postgres_session_factory(
        Base, "publish_projection_durability_unknown_dedupe"
    )
    try:
        opportunity = _make_traders_opportunity()
        dedupe_key = _expected_dedupe_key(opportunity)

        # No pre-existing row: the dedupe_key is unknown to the DB.
        rows = await _trade_signals_by_dedupe(
            session_factory, source="traders", dedupe_key=dedupe_key
        )
        assert rows == [], "fixture sanity: dedupe_key must not pre-exist"

        monkeypatch.setattr(
            intent_runtime_module, "AsyncSessionLocal", session_factory
        )

        runtime = IntentRuntime()
        runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
        monkeypatch.setattr(
            intent_runtime_module,
            "publish_signal_batch",
            AsyncMock(return_value="batch-test"),
        )

        published = await runtime.publish_opportunities(
            [opportunity], source="traders", signal_type_override="copy_trade"
        )
        assert published == 1
        assert len(runtime._signals_by_id) == 1
        in_memory_id = next(iter(runtime._signals_by_id.keys()))

        # The id is a fresh uuid (32 hex chars), distinct from the
        # canonical-old constant used in other tests.
        assert len(in_memory_id) == 32
        assert all(c in "0123456789abcdef" for c in in_memory_id)
        assert in_memory_id != CANONICAL_OLD_ID

        snapshot = runtime._signals_by_id[in_memory_id]
        assert snapshot["dedupe_key"] == dedupe_key
        assert snapshot["source"] == "traders"

        # The published id MUST already be present in `trade_signals`
        # — that is the durability invariant plan 0010 is fixing.
        # Pre-fix this is `[]` because the projection loop has not yet
        # run (and may not run at all in this test fixture).  Post-fix
        # the publish path synchronously writes the skeleton row before
        # returning, regardless of projection-loop timing.
        rows_after = await _trade_signals_by_dedupe(
            session_factory, source="traders", dedupe_key=dedupe_key
        )
        assert rows_after == [(in_memory_id, "pending")], (
            f"plan 0010 invariant: post-fix `publish_opportunities` "
            f"must commit a `trade_signals` row keyed by the same id "
            f"it stores in the in-memory cache, BEFORE returning.  "
            f"Got {rows_after!r}; expected [({in_memory_id!r}, 'pending')]."
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_publish_adopts_existing_id_when_db_row_is_terminal(
    monkeypatch, tmp_path
):
    """Edge case: even when the existing `trade_signals` row is in a
    terminal status (`expired`, `failed`), the in-memory id must adopt
    it. Otherwise, after restart, the new "republish" of the same
    wallet trade gets a fresh uuid that the projection cannot
    persist (due to `uq_trade_signals_source_dedupe`), and the FK
    race fires when the orchestrator picks it up.

    The terminal row should also be reactivated by the projection
    loop's existing logic (covered by `test_signal_bus_reactivation`),
    but the publish-side id adoption is what plan 0010 specifically
    fixes.
    """
    _patch_traders_strategy_loader(monkeypatch)
    engine, session_factory = await build_postgres_session_factory(
        Base, "publish_projection_durability_terminal_dedupe"
    )
    try:
        opportunity = _make_traders_opportunity()
        dedupe_key = _expected_dedupe_key(opportunity)

        await _insert_pre_restart_trade_signal(
            session_factory,
            signal_id=CANONICAL_OLD_ID,
            source="traders",
            strategy_type=COPY_TRADE_STRATEGY,
            market_id=COPY_TRADE_MARKET_ID,
            dedupe_key=dedupe_key,
            status="expired",
        )

        monkeypatch.setattr(
            intent_runtime_module, "AsyncSessionLocal", session_factory
        )

        runtime = IntentRuntime()
        runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
        monkeypatch.setattr(
            intent_runtime_module,
            "publish_signal_batch",
            AsyncMock(return_value="batch-test"),
        )

        await runtime.publish_opportunities(
            [opportunity], source="traders", signal_type_override="copy_trade"
        )
        assert len(runtime._signals_by_id) == 1
        in_memory_id = next(iter(runtime._signals_by_id.keys()))

        assert in_memory_id == CANONICAL_OLD_ID, (
            "plan 0010 invariant: even terminal-status pre-existing rows "
            "must have their id adopted on republish, otherwise the "
            "fresh-uuid path collides with `uq_trade_signals_source_dedupe`."
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_skeleton_row_carries_defensive_expires_at_overwritten_by_projection(
    monkeypatch, tmp_path
):
    """Plan 0011 invariant.  The skeleton-INSERT pass MUST stamp the
    placeholder row with ``expires_at = now + skeleton_ttl_seconds``
    so that an orphaned skeleton (publish dies before projection
    commits) becomes visible to the existing
    ``_run_trade_signal_pruner_loop`` (which keys on
    ``expires_at < now()``).  The projection loop's later UPSERT
    MUST then overwrite that defensive TTL with the strategy's
    intended ``expires_at`` — the skeleton TTL is purely a safety
    net for the orphan path, never the operational expiry.
    """
    _patch_traders_strategy_loader(monkeypatch)
    engine, session_factory = await build_postgres_session_factory(
        Base, "publish_projection_skeleton_expires_at"
    )
    try:
        opportunity = _make_traders_opportunity()
        dedupe_key = _expected_dedupe_key(opportunity)

        rows = await _trade_signals_by_dedupe(
            session_factory, source="traders", dedupe_key=dedupe_key
        )
        assert rows == [], "fixture sanity: dedupe_key must not pre-exist"

        monkeypatch.setattr(
            intent_runtime_module, "AsyncSessionLocal", session_factory
        )

        runtime = IntentRuntime()
        runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
        monkeypatch.setattr(
            intent_runtime_module,
            "publish_signal_batch",
            AsyncMock(return_value="batch-test"),
        )

        skeleton_ttl_seconds = max(
            60,
            int(getattr(settings, "INTENT_RUNTIME_SKELETON_TTL_SECONDS", 300) or 300),
        )

        publish_started_at = utcnow().replace(tzinfo=None)
        published = await runtime.publish_opportunities(
            [opportunity], source="traders", signal_type_override="copy_trade"
        )
        assert published == 1
        in_memory_id = next(iter(runtime._signals_by_id.keys()))

        # Read the skeleton row directly from the DB (not the cache):
        # we need to verify the column the publish path committed.
        async with session_factory() as session:
            skeleton_row = await session.get(TradeSignal, in_memory_id)
        assert skeleton_row is not None, (
            "plan 0011 invariant: skeleton row must exist in DB after "
            "publish (this is also the plan 0010 invariant)."
        )
        assert skeleton_row.payload_json is None, (
            "fixture sanity: this test simulates the orphan path "
            "(publish dies before projection commits), so payload_json "
            "must still be NULL when we observe the row."
        )
        skeleton_expires_at = skeleton_row.expires_at
        assert skeleton_expires_at is not None, (
            "plan 0011 invariant: skeleton row MUST carry a non-NULL "
            "expires_at so the terminal-row pruner can reach it.  "
            "Pre-fix the column was NULL, leaving orphans invisible."
        )
        # Normalize to naive UTC for tz-agnostic comparison.  The
        # ``expires_at`` column is ``TIMESTAMP WITHOUT TIME ZONE`` per
        # the model definition, but asyncpg's tz handling depends on
        # the server's timezone setting, so we normalize defensively.
        skeleton_expires_naive = (
            skeleton_expires_at.replace(tzinfo=None)
            if skeleton_expires_at.tzinfo is not None
            else skeleton_expires_at
        )
        # Allow up to +5 s of skew: we measured `publish_started_at`
        # before the publish call, the publish path itself sets
        # `now = utcnow()` slightly later, and DB round-trip adds a
        # few more ms.  The lower bound is `publish_started_at + ttl`
        # minus a small slack; the upper bound is whatever-now-was-
        # inside-publish + ttl + small slack.
        ttl_lower_bound = publish_started_at + timedelta(seconds=skeleton_ttl_seconds - 5)
        ttl_upper_bound = utcnow().replace(tzinfo=None) + timedelta(
            seconds=skeleton_ttl_seconds + 5
        )
        assert ttl_lower_bound <= skeleton_expires_naive <= ttl_upper_bound, (
            f"skeleton expires_at must be within "
            f"[{ttl_lower_bound!r}, {ttl_upper_bound!r}] for "
            f"ttl_seconds={skeleton_ttl_seconds}; got "
            f"{skeleton_expires_naive!r} (raw={skeleton_expires_at!r})."
        )

        # Now simulate the projection loop committing the strategy-
        # intended row.  Use a clearly-different `expires_at` so we
        # can prove the skeleton TTL was overwritten.
        strategy_expires_at = utcnow().replace(tzinfo=None) + timedelta(hours=6)
        async with session_factory() as session:
            await upsert_trade_signal(
                session,
                source="traders",
                source_item_id="projection-source-item-id",
                signal_type="copy_trade",
                strategy_type=COPY_TRADE_STRATEGY,
                market_id=COPY_TRADE_MARKET_ID,
                market_question="Will it happen?",
                direction="buy_yes",
                entry_price=0.41,
                edge_percent=9.0,
                confidence=0.7,
                liquidity=120.0,
                expires_at=strategy_expires_at,
                payload_json={"signal_emitted_at": utcnow().isoformat()},
                strategy_context_json={"source_key": "traders"},
                dedupe_key=dedupe_key,
                commit=True,
            )

        async with session_factory() as session:
            after_projection = await session.get(TradeSignal, in_memory_id)
        assert after_projection is not None
        assert after_projection.payload_json is not None, (
            "fixture sanity: projection upsert must have written "
            "payload_json — without it, the skeleton-orphan filter "
            "(payload_json IS NULL) would still match this row."
        )
        # The strategy's intended expires_at MUST win — the defensive
        # TTL is purely a safety net, never the operational value.
        # Normalize both sides to naive UTC for tz-agnostic equality.
        after_projection_expires_naive = (
            after_projection.expires_at.replace(tzinfo=None)
            if after_projection.expires_at is not None
            and after_projection.expires_at.tzinfo is not None
            else after_projection.expires_at
        )
        assert after_projection_expires_naive == strategy_expires_at, (
            f"plan 0011 invariant: projection-loop UPSERT must overwrite "
            f"the skeleton's defensive TTL with the strategy's intended "
            f"expires_at.  Expected {strategy_expires_at!r}; got "
            f"{after_projection_expires_naive!r}.  If they don't match, "
            f"the skeleton TTL is leaking into operational expiry "
            f"semantics — escalate as a missed UPSERT column."
        )
        assert after_projection_expires_naive != skeleton_expires_naive, (
            "plan 0011 sanity: this test is only meaningful when the "
            "strategy expires_at is observably different from the "
            "skeleton TTL — pick a wider gap if these values can ever "
            "collide in production."
        )
    finally:
        await engine.dispose()
