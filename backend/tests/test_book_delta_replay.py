"""Tests for ``BookDeltaReplay`` — the live-parity backtest source.

Reconstructs book state by walking ``book_delta_events`` from a most-
recent ``MarketMicrostructureSnapshot`` anchor.  These tests drive
the replay against an in-memory fake AsyncSession that returns
hand-crafted rows, so we can verify the state-reconstruction logic
without a live Postgres.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest

from services.backtest.book_replay import BookDeltaReplay


# ── Fake DB rows + session ─────────────────────────────────────────────
#
# We don't want to spin up Postgres just for these tests.  The
# BookDeltaReplay only uses the session for ``execute(stmt)`` and
# expects a result with ``.scalars().all()``.  We give it a mock that
# returns pre-canned rows based on which table the SELECT targets.


@dataclass
class _FakeMmsRow:
    id: str = ""
    token_id: str = ""
    observed_at: datetime = datetime.now(timezone.utc)
    snapshot_type: str = "book"
    bids_json: list[dict[str, Any]] = field(default_factory=list)
    asks_json: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _FakeBdeRow:
    id: str = ""
    token_id: str = ""
    observed_at: datetime = datetime.now(timezone.utc)
    side: str = "bid"
    price: float = 0.0
    queue_depth_before: float | None = None
    queue_depth_after: float | None = None
    spread_bps_at_event: float | None = None


class _FakeSession:
    """Async session double for BookDeltaReplay tests.

    Routes ``execute(stmt)`` based on the table targeted in the SELECT.
    Rows are pre-loaded by the test.  Sequential queries with keyset
    pagination are handled by re-checking every call (small fixtures
    so this is cheap).
    """

    def __init__(self, *, anchors: list[_FakeMmsRow], deltas: list[_FakeBdeRow]):
        self._anchors = list(anchors)
        self._deltas = sorted(deltas, key=lambda r: (r.observed_at, r.id))
        self._delta_cursor = 0

    async def execute(self, stmt):
        # Inspect the FROM clause to pick a result type.  SQLAlchemy
        # Select objects have a ``froms`` attribute or the table can
        # be derived from the column descriptions; we cheat by
        # stringifying the compiled SQL.  Cheap and works for our
        # narrow surface.
        sql = str(stmt).lower()
        if "market_microstructure_snapshots" in sql:
            return _FakeResult(self._anchors)
        if "book_delta_events" in sql:
            # Honor LIMIT in the stmt by extracting it.  Simpler: just
            # return all remaining rows; BookDeltaReplay's chunk_size
            # filter happens client-side anyway.
            remaining = self._deltas[self._delta_cursor :]
            self._delta_cursor = len(self._deltas)
            return _FakeResult(remaining)
        return _FakeResult([])

    async def rollback(self):
        return None


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


# ── Tests ──────────────────────────────────────────────────────────────


def _utc(year, month, day, hour=0, minute=0, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_anchor_only_no_deltas_yields_nothing():
    """An anchor without subsequent deltas means no snapshots flow.
    The anchor is the seed state — snapshots are emitted on each
    delta, not from the anchor itself.
    """
    anchor = _FakeMmsRow(
        id="a1", token_id="tok",
        observed_at=_utc(2026, 5, 1),
        bids_json=[{"price": 0.50, "size": 100}],
        asks_json=[{"price": 0.51, "size": 80}],
    )
    sess = _FakeSession(anchors=[anchor], deltas=[])
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = []
    async for s in replay.iter_snapshots():
        snaps.append(s)
    assert snaps == []
    assert replay.snapshots_yielded == 0


@pytest.mark.asyncio
async def test_anchor_seeds_state_then_delta_emits_snapshot():
    anchor = _FakeMmsRow(
        id="a1", token_id="tok",
        observed_at=_utc(2026, 5, 2),
        bids_json=[{"price": 0.50, "size": 100}],
        asks_json=[{"price": 0.51, "size": 80}],
    )
    delta = _FakeBdeRow(
        id="d1", token_id="tok",
        observed_at=_utc(2026, 5, 2, 0, 1),
        side="bid", price=0.50,
        queue_depth_before=100.0, queue_depth_after=60.0,
        spread_bps_at_event=200.0,
    )
    sess = _FakeSession(anchors=[anchor], deltas=[delta])
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = []
    async for s in replay.iter_snapshots():
        snaps.append(s)
    assert len(snaps) == 1
    snap = snaps[0]
    # Bid level decreased to 60 (queue_depth_after).
    bid_sizes = {lvl.price: lvl.size for lvl in snap.bids}
    assert bid_sizes[0.50] == pytest.approx(60.0)
    # Ask side untouched — still 80 from the anchor.
    ask_sizes = {lvl.price: lvl.size for lvl in snap.asks}
    assert ask_sizes[0.51] == pytest.approx(80.0)


@pytest.mark.asyncio
async def test_level_emptied_removed_from_state():
    """When queue_depth_after=0, the level is removed entirely so it
    doesn't show up as a zero-size phantom level in the snapshot.
    """
    anchor = _FakeMmsRow(
        id="a1", token_id="tok",
        observed_at=_utc(2026, 5, 2),
        bids_json=[{"price": 0.50, "size": 100}, {"price": 0.49, "size": 50}],
        asks_json=[{"price": 0.51, "size": 80}],
    )
    delta = _FakeBdeRow(
        id="d1", token_id="tok",
        observed_at=_utc(2026, 5, 2, 0, 1),
        side="bid", price=0.50,
        queue_depth_before=100.0, queue_depth_after=0.0,
    )
    sess = _FakeSession(anchors=[anchor], deltas=[delta])
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = []
    async for s in replay.iter_snapshots():
        snaps.append(s)
    snap = snaps[0]
    # 0.50 should be gone; 0.49 still there.
    bid_prices = {lvl.price for lvl in snap.bids}
    assert 0.50 not in bid_prices
    assert 0.49 in bid_prices


@pytest.mark.asyncio
async def test_bootstrap_mode_no_anchor_uses_queue_depth_before():
    """When a token has no anchor snapshot, the FIRST delta's
    queue_depth_before is used to seed the level.  This is the path
    that's hit when the unified ingestor only just started running and
    the snapshot table is empty for the window's start.
    """
    delta1 = _FakeBdeRow(
        id="d1", token_id="tok",
        observed_at=_utc(2026, 5, 2, 0, 1),
        side="bid", price=0.50,
        queue_depth_before=200.0, queue_depth_after=150.0,
    )
    sess = _FakeSession(anchors=[], deltas=[delta1])  # no anchor
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = []
    async for s in replay.iter_snapshots():
        snaps.append(s)
    assert len(snaps) == 1
    bid_sizes = {lvl.price: lvl.size for lvl in snaps[0].bids}
    # Bootstrapped to queue_depth_before then immediately decremented to 150.
    assert bid_sizes[0.50] == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_multiple_deltas_emit_snapshots_in_order():
    anchor = _FakeMmsRow(
        id="a1", token_id="tok",
        observed_at=_utc(2026, 5, 2),
        bids_json=[{"price": 0.50, "size": 100}],
        asks_json=[{"price": 0.51, "size": 80}],
    )
    deltas = [
        _FakeBdeRow(id="d1", token_id="tok",
                    observed_at=_utc(2026, 5, 2, 0, 1),
                    side="bid", price=0.50,
                    queue_depth_before=100.0, queue_depth_after=80.0),
        _FakeBdeRow(id="d2", token_id="tok",
                    observed_at=_utc(2026, 5, 2, 0, 2),
                    side="ask", price=0.51,
                    queue_depth_before=80.0, queue_depth_after=60.0),
        _FakeBdeRow(id="d3", token_id="tok",
                    observed_at=_utc(2026, 5, 2, 0, 3),
                    side="bid", price=0.50,
                    queue_depth_before=80.0, queue_depth_after=70.0),
    ]
    sess = _FakeSession(anchors=[anchor], deltas=deltas)
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = []
    async for s in replay.iter_snapshots():
        snaps.append(s)
    assert len(snaps) == 3
    # Snapshots are monotonic in observed_at.
    assert all(snaps[i].observed_at <= snaps[i + 1].observed_at for i in range(len(snaps) - 1))
    # State evolution: bid 100→80→80→70, ask 80→80→60→60
    final = snaps[-1]
    bid = {lvl.price: lvl.size for lvl in final.bids}
    ask = {lvl.price: lvl.size for lvl in final.asks}
    assert bid[0.50] == pytest.approx(70.0)
    assert ask[0.51] == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_snapshot_at_combines_anchor_plus_deltas_up_to_ts():
    anchor = _FakeMmsRow(
        id="a1", token_id="tok",
        observed_at=_utc(2026, 5, 2),
        bids_json=[{"price": 0.50, "size": 100}],
        asks_json=[{"price": 0.51, "size": 80}],
    )
    deltas = [
        _FakeBdeRow(id="d1", token_id="tok",
                    observed_at=_utc(2026, 5, 2, 0, 1),
                    side="bid", price=0.50,
                    queue_depth_before=100.0, queue_depth_after=70.0),
        _FakeBdeRow(id="d2", token_id="tok",
                    observed_at=_utc(2026, 5, 2, 0, 2),
                    side="bid", price=0.50,
                    queue_depth_before=70.0, queue_depth_after=40.0),
    ]
    sess = _FakeSession(anchors=[anchor], deltas=deltas)
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snap = await replay.snapshot_at(token_id="tok", ts=_utc(2026, 5, 2, 0, 5))
    assert snap is not None
    bid = {lvl.price: lvl.size for lvl in snap.bids}
    # Both deltas applied → bid level at 40.
    assert bid[0.50] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_invalid_side_skipped_gracefully():
    delta = _FakeBdeRow(
        id="d1", token_id="tok",
        observed_at=_utc(2026, 5, 2, 0, 1),
        side="garbage", price=0.50,
        queue_depth_before=100.0, queue_depth_after=50.0,
    )
    sess = _FakeSession(anchors=[], deltas=[delta])
    replay = BookDeltaReplay(
        session=sess, token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = [s async for s in replay.iter_snapshots()]
    assert snaps == []  # invalid side → row skipped, no snapshot emitted


@pytest.mark.asyncio
async def test_truncation_flag_set_on_chunk_failure():
    """If the delta query raises, the replay should set ``truncated``
    rather than propagating the exception (we'd rather return partial
    data than crash the whole backtest)."""

    class _BoomSession:
        async def execute(self, stmt):
            sql = str(stmt).lower()
            if "market_microstructure_snapshots" in sql:
                return _FakeResult([])
            if "book_delta_events" in sql:
                raise RuntimeError("simulated chunk failure")
            return _FakeResult([])

        async def rollback(self):
            return None

    replay = BookDeltaReplay(
        session=_BoomSession(), token_ids=["tok"],
        start=_utc(2026, 5, 2), end=_utc(2026, 5, 3),
    )
    snaps = [s async for s in replay.iter_snapshots()]
    assert snaps == []
    assert replay.truncated is True
    assert "simulated chunk failure" in (replay.truncation_reason or "")
