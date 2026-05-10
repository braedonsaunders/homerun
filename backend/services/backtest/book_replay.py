"""L2 order-book replay backed by ``MarketMicrostructureSnapshot``.

The microstructure recorder samples Polymarket CLOB books at ~0.5s intervals
and persists up to 25 levels per side as JSON. This module exposes that data
to the backtester as a stream of immutable ``BookSnapshot`` instances and
provides ``snapshot_at(token_id, ts)`` for point-in-time queries.

Two access modes:

1. **Streaming replay** (``iter_snapshots``) — yield snapshots in
   chronological order for a token. Used by the matching engine to advance
   simulated time and re-evaluate resting orders against book updates.

2. **Point-in-time** (``snapshot_at``) — most-recent snapshot at-or-before a
   timestamp. Used to evaluate fills for orders submitted at a specific
   wall-clock time.

The replay is **read-only** and pure: it never writes to the snapshot table.
For tests or sparse-data tokens, ``InMemoryBookReplay`` lets callers seed
synthetic snapshots directly.
"""

from __future__ import annotations

import bisect
import logging
import os as _os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable, Optional, Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import MarketMicrostructureSnapshot
from utils.converters import safe_float

logger = logging.getLogger(__name__)


# Backtests are long-running batch work and shouldn't be killed by the
# default API ``statement_timeout`` (typically 30s). The run-session
# raises its own per-statement budget to 5 min before streaming the
# replay; under heavy DB load (live trading hammering the same
# Postgres) a 7-day × 500-token chunk can legitimately take >30s. If
# the env var is set, we use it; otherwise default to 5 min.
_BACKTEST_STATEMENT_TIMEOUT_MS = int(
    _os.getenv("HOMERUN_BACKTEST_STATEMENT_TIMEOUT_MS", "300000")
)


@dataclass(frozen=True)
class PriceLevel:
    """One side of a single book level."""

    price: float
    size: float


@dataclass(frozen=True)
class BookSnapshot:
    """Immutable L2 snapshot at a point in time.

    ``bids`` are descending by price (best bid first); ``asks`` are
    ascending (best ask first). ``mid`` is None when either side is empty.
    """

    token_id: str
    observed_at: datetime
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    sequence: Optional[int] = None
    spread_bps: Optional[float] = None
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    trade_side: Optional[str] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    def total_size_at_or_better(self, side: str, price: float) -> float:
        """Sum of size on ``side`` at or better than ``price``.

        For a SELL order at ``price`` we want to walk the bid side:
        bids with bid_price >= price are accessible. For a BUY at ``price``,
        walk asks with ask_price <= price.
        """
        side_norm = (side or "").upper()
        if side_norm == "SELL":
            return sum(lvl.size for lvl in self.bids if lvl.price >= price - 1e-12)
        if side_norm == "BUY":
            return sum(lvl.size for lvl in self.asks if lvl.price <= price + 1e-12)
        return 0.0

    def walk_for_taker(
        self,
        side: str,
        size: float,
        limit_price: Optional[float] = None,
    ) -> list[tuple[float, float]]:
        """Simulate a taker walking the visible book.

        Returns a list of (fill_price, fill_size) pairs that consume
        liquidity from the side opposite ``side`` (a SELL hits bids; a BUY
        hits asks). Stops when ``size`` is exhausted or the next level
        would cross ``limit_price``. The sum of returned sizes is <= the
        requested ``size``.
        """
        remaining = max(0.0, float(size))
        if remaining <= 0:
            return []
        side_norm = (side or "").upper()
        if side_norm == "SELL":
            levels = self.bids

            def crosses(lp: float) -> bool:
                return limit_price is not None and lp + 1e-12 < float(limit_price)

        elif side_norm == "BUY":
            levels = self.asks

            def crosses(lp: float) -> bool:
                return limit_price is not None and lp - 1e-12 > float(limit_price)
        else:
            return []
        fills: list[tuple[float, float]] = []
        for lvl in levels:
            if remaining <= 0:
                break
            if crosses(lvl.price):
                break
            take = min(lvl.size, remaining)
            if take <= 0:
                continue
            fills.append((lvl.price, take))
            remaining -= take
        return fills


def _parse_levels(raw: Any, *, descending: bool) -> tuple[PriceLevel, ...]:
    """Parse a JSON book side into a sorted tuple of PriceLevel.

    Accepts either ``[[price, size], ...]`` or ``[{"price":..., "size":...}, ...]``.
    Filters out non-positive sizes and clamps to [0.01, 0.99] tick-legal range.
    """
    if not raw:
        return ()
    out: list[PriceLevel] = []
    for entry in raw:
        if isinstance(entry, dict):
            price = safe_float(entry.get("price"))
            size = safe_float(entry.get("size"))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            price = safe_float(entry[0])
            size = safe_float(entry[1])
        else:
            continue
        if price is None or size is None:
            continue
        if price <= 0 or price >= 1.0:
            continue
        if size <= 0:
            continue
        out.append(PriceLevel(price=float(price), size=float(size)))
    out.sort(key=lambda lvl: lvl.price, reverse=descending)
    return tuple(out)


def _row_to_snapshot(row: MarketMicrostructureSnapshot) -> BookSnapshot:
    observed = row.observed_at
    if observed is not None and observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return BookSnapshot(
        token_id=str(row.token_id or ""),
        observed_at=observed or datetime.now(timezone.utc),
        bids=_parse_levels(row.bids_json, descending=True),
        asks=_parse_levels(row.asks_json, descending=False),
        sequence=int(row.sequence) if row.sequence is not None else None,
        spread_bps=(float(row.spread_bps) if row.spread_bps is not None else None),
        trade_price=(float(row.trade_price) if row.trade_price is not None else None),
        trade_size=(float(row.trade_size) if row.trade_size is not None else None),
        trade_side=str(row.trade_side) if row.trade_side else None,
    )


class BookReplay:
    """DB-backed L2 replay over a window of time for one or more tokens.

    Caller provides an ``AsyncSession`` and a (start, end) range; the replay
    streams snapshots in chronological order, optionally filtered by
    snapshot type. Snapshots are loaded lazily in chunks to bound memory.
    """

    def __init__(
        self,
        *,
        session: AsyncSession | None = None,
        token_ids: Sequence[str],
        start: datetime,
        end: datetime,
        snapshot_type: Optional[str] = "book",
        chunk_size: int = 5000,
    ):
        self._session = session
        self._token_ids = list({tid for tid in token_ids if tid})
        self._start = _to_utc(start)
        self._end = _to_utc(end)
        self._snapshot_type = snapshot_type
        self._chunk_size = max(100, int(chunk_size))
        # Set when a chunk times out / errors and the replay is
        # truncated.  Caller reads this after iter_snapshots() to
        # surface a validation_warning.
        self.truncated: bool = False
        self.truncation_reason: Optional[str] = None
        self.snapshots_yielded: int = 0

    async def _raise_session_timeout(self) -> None:
        """Bump ``statement_timeout`` on this session for replay queries.

        Idempotent — runs once per BookReplay instance.  Backtests need
        a longer per-statement budget than the API default because each
        keyset-paginated chunk against ``market_microstructure_snapshots``
        can legitimately take >30s when the IN-list spans hundreds of
        tokens over a multi-day window AND the live system is loading
        the same Postgres.
        """
        if getattr(self, "_timeout_raised", False):
            return
        try:
            await self._session.execute(
                text(f"SET statement_timeout = {int(_BACKTEST_STATEMENT_TIMEOUT_MS)}")
            )
            self._timeout_raised = True
        except Exception as exc:
            # Non-fatal — if SET fails we still try the queries; they
            # just inherit whatever the connection-level default is.
            logger.warning("Failed to raise statement_timeout for backtest replay: %s", exc)
            self._timeout_raised = True  # don't retry every chunk

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        """Yield snapshots in (observed_at, sequence) order.

        When a session was passed at construction (legacy / test path)
        we use it.  Otherwise each chunk runs in its own short-lived
        session, avoiding the production pool reaper's 45s checkout
        limit — without that, multi-day replays get killed mid-stream.
        """
        if not self._token_ids:
            return
        from contextlib import asynccontextmanager
        from models.database import AsyncSessionLocal as _BTSession

        # When a caller-provided session is set we run all chunks
        # against it (legacy behaviour; preserves test stubs that
        # mock the session).  Production callers pass None (the new
        # default) to opt into the fresh-per-chunk path.
        @asynccontextmanager
        async def _session_for_chunk():
            if self._session is not None:
                yield self._session
                return
            async with _BTSession() as fresh:
                try:
                    await fresh.execute(
                        text(f"SET statement_timeout = {int(_BACKTEST_STATEMENT_TIMEOUT_MS)}")
                    )
                except Exception:
                    pass
                yield fresh

        last_observed = self._start
        last_id: Optional[str] = None
        chunk_index = 0
        total_yielded = 0
        if self._session is not None:
            await self._raise_session_timeout()
        while True:
            try:
                async with _session_for_chunk() as session:
                    stmt = (
                        select(MarketMicrostructureSnapshot)
                        .where(
                            MarketMicrostructureSnapshot.token_id.in_(self._token_ids),
                            MarketMicrostructureSnapshot.observed_at >= last_observed,
                            MarketMicrostructureSnapshot.observed_at <= self._end,
                        )
                        .order_by(
                            MarketMicrostructureSnapshot.observed_at.asc(),
                            MarketMicrostructureSnapshot.id.asc(),
                        )
                        .limit(self._chunk_size)
                    )
                    if self._snapshot_type:
                        stmt = stmt.where(
                            MarketMicrostructureSnapshot.snapshot_type == self._snapshot_type
                        )
                    if last_id is not None:
                        stmt = stmt.where(MarketMicrostructureSnapshot.id != last_id)
                    rows = (await session.execute(stmt)).scalars().all()
            except Exception as exc:
                # Per-chunk failure (statement_timeout, connection
                # drop, reaper kill).  Log + truncate; the matching
                # engine handles a truncated replay gracefully.
                logger.warning(
                    "BookReplay chunk %d failed after %d snapshots; truncating replay: %s",
                    chunk_index, total_yielded, exc,
                )
                self.truncated = True
                self.truncation_reason = str(exc)[:500]
                break
            if not rows:
                break
            for row in rows:
                snap = _row_to_snapshot(row)
                yield snap
                last_observed = snap.observed_at
                last_id = str(row.id)
                total_yielded += 1
            chunk_index += 1
            self.snapshots_yielded = total_yielded
            if len(rows) < self._chunk_size:
                break

    async def snapshot_at(
        self, *, token_id: str, ts: datetime
    ) -> Optional[BookSnapshot]:
        """Most-recent ``book`` snapshot at-or-before ``ts`` for this token."""
        target = _to_utc(ts)
        stmt = (
            select(MarketMicrostructureSnapshot)
            .where(
                MarketMicrostructureSnapshot.token_id == token_id,
                MarketMicrostructureSnapshot.observed_at <= target,
            )
            .order_by(MarketMicrostructureSnapshot.observed_at.desc())
            .limit(1)
        )
        if self._snapshot_type:
            stmt = stmt.where(
                MarketMicrostructureSnapshot.snapshot_type == self._snapshot_type
            )
        row = (await self._session.execute(stmt)).scalars().first()
        if row is None:
            return None
        return _row_to_snapshot(row)


class InMemoryBookReplay:
    """Synthetic replay seeded from in-memory snapshots.

    Used by tests and by callers that have already materialized a sequence
    of book states (e.g., when replaying a recorded backtest scenario).
    Mirrors the ``BookReplay`` interface but returns pre-loaded data.
    """

    def __init__(self, snapshots: Iterable[BookSnapshot]):
        snaps = sorted(
            (s for s in snapshots if isinstance(s, BookSnapshot)),
            key=lambda s: (s.observed_at, s.sequence or 0),
        )
        self._snaps = snaps
        # Pre-bucket per token for O(log n) point-in-time lookup
        self._by_token: dict[str, list[BookSnapshot]] = {}
        for s in snaps:
            self._by_token.setdefault(s.token_id, []).append(s)
        # already sorted by observed_at within each list because snaps was sorted

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        for s in self._snaps:
            yield s

    async def snapshot_at(
        self, *, token_id: str, ts: datetime
    ) -> Optional[BookSnapshot]:
        target = _to_utc(ts)
        bucket = self._by_token.get(token_id) or []
        if not bucket:
            return None
        # binary search for the rightmost snapshot with observed_at <= target
        keys = [s.observed_at for s in bucket]
        idx = bisect.bisect_right(keys, target) - 1
        if idx < 0:
            return None
        return bucket[idx]


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── BookDeltaReplay ──────────────────────────────────────────────────────
#
# Live-parity replay path: reconstruct full book state by walking
# ``book_delta_events`` (the live system's authoritative feed) from
# the most recent ``MarketMicrostructureSnapshot`` anchor.
#
# This is the answer to "why was the backtest reading from a sparse
# table when the live system populates a dense one?"  ``book_delta_events``
# carries 3-4M rows/week in steady state — every level change the live
# orchestrator sees lands here.  ``MarketMicrostructureSnapshot`` is
# now an ANCHOR table: the unified ingestor writes one snapshot per
# token every ~0.5s as the running state, plus external backfill /
# provider imports populate it with point-in-time books.
#
# Replay algorithm:
#
#   1. For each token in scope, find the most recent ``snapshot_type='book'``
#      row at-or-before ``start``.  This is the anchor.  If absent, the
#      bootstrap mode below is used.
#
#   2. Maintain per-token running state ``{(side, price): size}`` seeded
#      from the anchor.  Walk delta events in (observed_at, id) order.
#      For each delta, mutate the running state:
#        * ``trade`` events decrement the level by ``trade_size`` (or
#          remove it if depth_after=0)
#        * ``cancel`` events decrement the level by ``cancel_size``
#      For monotonicity, when ``queue_depth_after`` is provided we
#      authoritatively set the level to that value rather than
#      computing trade/cancel arithmetic — the ingestor already did
#      that work and including it bypasses cumulative drift.
#
#   3. After each delta, emit a ``BookSnapshot`` reflecting the new
#      state.  The matching engine consumes these as if they came
#      from the snapshot table — same ``BookSnapshot`` struct, same
#      ordering invariants.
#
# Bootstrap mode: when there is NO anchor for a token (most common
# situation when the unified ingestor only just started recording),
# we synthesize a starting state from the first delta's
# ``queue_depth_before`` per (price, side).  This produces a partial
# initial book — only price levels that have changed since the
# replay began are visible.  The matching engine handles this
# gracefully (untracked levels just don't get fills).  Coverage
# improves rapidly as more deltas land (typically within minutes).


@dataclass
class _RunningBook:
    """Mutable running per-token book state for delta replay."""

    bids: dict[float, float] = field(default_factory=dict)  # price → size
    asks: dict[float, float] = field(default_factory=dict)
    last_observed_at: Optional[datetime] = None

    def to_snapshot(
        self,
        *,
        token_id: str,
        observed_at: datetime,
        spread_bps: Optional[float],
    ) -> BookSnapshot:
        bids_sorted = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)
        asks_sorted = sorted(self.asks.items(), key=lambda kv: kv[0])
        return BookSnapshot(
            token_id=token_id,
            observed_at=observed_at,
            bids=tuple(PriceLevel(price=p, size=s) for p, s in bids_sorted),
            asks=tuple(PriceLevel(price=p, size=s) for p, s in asks_sorted),
            sequence=None,
            spread_bps=spread_bps,
            trade_price=None,
            trade_size=None,
            trade_side=None,
        )


class BookDeltaReplay:
    """Stream synthesized book snapshots from ``book_delta_events`` + anchors.

    Same public surface as ``BookReplay`` (``iter_snapshots`` +
    ``snapshot_at``) so callers can swap impls based on coverage.

    Memory bounded: per-token running state is at most ~50 entries (25
    levels × 2 sides).  No materialization of the full delta stream.

    Truncation safety inherited from BookReplay: if any chunk query
    fails, log + truncate rather than crash.  Truncation flags are
    surfaced on the instance for caller introspection.
    """

    def __init__(
        self,
        *,
        session: AsyncSession | None = None,
        token_ids: Sequence[str],
        start: datetime,
        end: datetime,
        chunk_size: int = 5000,
    ):
        from models.database import BookDeltaEvent  # local to avoid cycle
        self._BookDeltaEvent = BookDeltaEvent
        self._session = session
        self._token_ids = list({tid for tid in token_ids if tid})
        self._start = _to_utc(start)
        self._end = _to_utc(end)
        self._chunk_size = max(100, int(chunk_size))
        self._timeout_raised = False
        self.truncated: bool = False
        self.truncation_reason: Optional[str] = None
        self.snapshots_yielded: int = 0

    async def _raise_session_timeout(self) -> None:
        """Bump statement_timeout for this session.  Same rationale as
        BookReplay — the multi-day delta scan can legitimately take
        >30s under live DB load.
        """
        if self._timeout_raised:
            return
        try:
            await self._session.execute(
                text(f"SET statement_timeout = {int(_BACKTEST_STATEMENT_TIMEOUT_MS)}")
            )
        except Exception as exc:
            logger.warning("Failed to raise statement_timeout for delta replay: %s", exc)
        self._timeout_raised = True

    async def _load_anchors(self) -> dict[str, _RunningBook]:
        """For each token, fetch the most recent snapshot at-or-before
        ``start`` and seed a ``_RunningBook`` from it.  Tokens with no
        anchor get an empty book (bootstrap mode).  Uses caller-
        provided session when present (test path) or fresh sessions
        per chunk (production reaper-avoidance).
        """
        anchors: dict[str, _RunningBook] = {}
        if not self._token_ids:
            return anchors
        from contextlib import asynccontextmanager
        from models.database import AsyncSessionLocal as _BTSession

        @asynccontextmanager
        async def _session_for_chunk():
            if self._session is not None:
                yield self._session
                return
            async with _BTSession() as fresh:
                try:
                    await fresh.execute(
                        text(f"SET statement_timeout = {int(_BACKTEST_STATEMENT_TIMEOUT_MS)}")
                    )
                except Exception:
                    pass
                yield fresh

        CHUNK = 50
        for i in range(0, len(self._token_ids), CHUNK):
            chunk = self._token_ids[i : i + CHUNK]
            try:
                async with _session_for_chunk() as session:
                    stmt = (
                        select(MarketMicrostructureSnapshot)
                        .where(
                            MarketMicrostructureSnapshot.token_id.in_(chunk),
                            MarketMicrostructureSnapshot.observed_at <= self._start,
                            MarketMicrostructureSnapshot.snapshot_type == "book",
                        )
                        .order_by(
                            MarketMicrostructureSnapshot.token_id.asc(),
                            MarketMicrostructureSnapshot.observed_at.desc(),
                        )
                    )
                    rows = (await session.execute(stmt)).scalars().all()
            except Exception as exc:
                logger.warning("Anchor lookup failed for chunk; using bootstrap: %s", exc)
                continue
            # First row per token is the most recent (we sorted desc).
            seen: set[str] = set()
            for row in rows:
                tid = str(row.token_id or "")
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                rb = _RunningBook()
                for lvl in row.bids_json or []:
                    p = float(lvl.get("price") or 0.0)
                    s = float(lvl.get("size") or 0.0)
                    if p > 0 and s > 0:
                        rb.bids[round(p, 4)] = s
                for lvl in row.asks_json or []:
                    p = float(lvl.get("price") or 0.0)
                    s = float(lvl.get("size") or 0.0)
                    if p > 0 and s > 0:
                        rb.asks[round(p, 4)] = s
                rb.last_observed_at = row.observed_at
                anchors[tid] = rb
        return anchors

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        """Yield synthesized snapshots in (observed_at, id) order across
        all tokens.

        Memory model: maintains one ``_RunningBook`` per token; each is
        bounded to ~50 entries.  Total memory is O(tokens × 50) which
        for a 500-token universe is ~25k float pairs — trivial.
        """
        if not self._token_ids:
            return
        from contextlib import asynccontextmanager
        from models.database import AsyncSessionLocal as _BTSession

        # Seed running state from anchors.  Tokens without anchors get
        # empty books and are populated lazily from delta queue_depth_before.
        running: dict[str, _RunningBook] = await self._load_anchors()

        @asynccontextmanager
        async def _session_for_chunk():
            if self._session is not None:
                yield self._session
                return
            async with _BTSession() as fresh:
                try:
                    await fresh.execute(
                        text(f"SET statement_timeout = {int(_BACKTEST_STATEMENT_TIMEOUT_MS)}")
                    )
                except Exception:
                    pass
                yield fresh

        BookDeltaEvent = self._BookDeltaEvent
        last_observed = self._start
        last_id: Optional[str] = None
        chunk_index = 0
        total_yielded = 0
        if self._session is not None:
            await self._raise_session_timeout()
        while True:
            try:
                async with _session_for_chunk() as session:
                    stmt = (
                        select(BookDeltaEvent)
                        .where(
                            BookDeltaEvent.token_id.in_(self._token_ids),
                            BookDeltaEvent.observed_at >= last_observed,
                            BookDeltaEvent.observed_at <= self._end,
                        )
                        .order_by(
                            BookDeltaEvent.observed_at.asc(),
                            BookDeltaEvent.id.asc(),
                        )
                        .limit(self._chunk_size)
                    )
                    if last_id is not None:
                        stmt = stmt.where(BookDeltaEvent.id != last_id)
                    rows = (await session.execute(stmt)).scalars().all()
            except Exception as exc:
                logger.warning(
                    "BookDeltaReplay chunk %d failed after %d snapshots; truncating: %s",
                    chunk_index, total_yielded, exc,
                )
                self.truncated = True
                self.truncation_reason = str(exc)[:500]
                break
            if not rows:
                break

            for row in rows:
                tid = str(row.token_id or "")
                if not tid:
                    continue
                rb = running.setdefault(tid, _RunningBook())
                price = float(row.price or 0.0)
                if price <= 0:
                    continue
                key = round(price, 4)
                # Determine which side dict to mutate.  Delta events
                # use side='bid'|'ask' (lowercase per the schema).
                side_norm = (row.side or "").strip().lower()
                if side_norm not in {"bid", "ask"}:
                    continue
                book_side = rb.bids if side_norm == "bid" else rb.asks

                # Bootstrap: if we've never seen this level and the
                # delta carries queue_depth_before, seed it.  This
                # backfills levels we missed before the replay anchor.
                if key not in book_side and row.queue_depth_before is not None:
                    qb = float(row.queue_depth_before)
                    if qb > 0:
                        book_side[key] = qb

                # Authoritative state update: queue_depth_after is the
                # post-event size.  The ingestor recorded it; trust it.
                # Falling back to subtraction would accumulate drift.
                if row.queue_depth_after is not None:
                    qa = float(row.queue_depth_after)
                    if qa > 0:
                        book_side[key] = qa
                    else:
                        # Level emptied — remove it so it doesn't
                        # appear as zero-size in the snapshot.
                        book_side.pop(key, None)

                rb.last_observed_at = row.observed_at
                spread_bps = (
                    float(row.spread_bps_at_event)
                    if row.spread_bps_at_event is not None
                    else None
                )

                # Emit a snapshot at this delta's timestamp.
                snap = rb.to_snapshot(
                    token_id=tid,
                    observed_at=row.observed_at,
                    spread_bps=spread_bps,
                )
                yield snap
                last_observed = row.observed_at
                last_id = str(row.id)
                total_yielded += 1

            chunk_index += 1
            self.snapshots_yielded = total_yielded
            if len(rows) < self._chunk_size:
                break

    async def snapshot_at(
        self, *, token_id: str, ts: datetime
    ) -> Optional[BookSnapshot]:
        """Reconstruct the book state at ``ts`` for one token.

        Walks anchor + deltas up to ``ts``.  Used by point-in-time
        queries (e.g., the matching engine's _submit_entry).  Memory
        bounded to one running book.
        """
        BookDeltaEvent = self._BookDeltaEvent
        target = _to_utc(ts)

        # Anchor.
        rb = _RunningBook()
        try:
            anchor_stmt = (
                select(MarketMicrostructureSnapshot)
                .where(
                    MarketMicrostructureSnapshot.token_id == token_id,
                    MarketMicrostructureSnapshot.observed_at <= target,
                    MarketMicrostructureSnapshot.snapshot_type == "book",
                )
                .order_by(MarketMicrostructureSnapshot.observed_at.desc())
                .limit(1)
            )
            row = (await self._session.execute(anchor_stmt)).scalars().first()
            if row is not None:
                for lvl in row.bids_json or []:
                    p = float(lvl.get("price") or 0.0)
                    s = float(lvl.get("size") or 0.0)
                    if p > 0 and s > 0:
                        rb.bids[round(p, 4)] = s
                for lvl in row.asks_json or []:
                    p = float(lvl.get("price") or 0.0)
                    s = float(lvl.get("size") or 0.0)
                    if p > 0 and s > 0:
                        rb.asks[round(p, 4)] = s
                rb.last_observed_at = row.observed_at
        except Exception as exc:
            logger.warning("snapshot_at anchor lookup failed: %s", exc)
            try:
                await self._session.rollback()
            except Exception:
                pass

        # Apply deltas anchor → ts.
        anchor_ts = rb.last_observed_at or self._start
        try:
            stmt = (
                select(BookDeltaEvent)
                .where(
                    BookDeltaEvent.token_id == token_id,
                    BookDeltaEvent.observed_at > anchor_ts,
                    BookDeltaEvent.observed_at <= target,
                )
                .order_by(
                    BookDeltaEvent.observed_at.asc(),
                    BookDeltaEvent.id.asc(),
                )
            )
            rows = (await self._session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.warning("snapshot_at delta lookup failed: %s", exc)
            try:
                await self._session.rollback()
            except Exception:
                pass
            rows = []

        spread_bps: Optional[float] = None
        for row in rows:
            price = float(row.price or 0.0)
            if price <= 0:
                continue
            key = round(price, 4)
            side_norm = (row.side or "").strip().lower()
            if side_norm not in {"bid", "ask"}:
                continue
            book_side = rb.bids if side_norm == "bid" else rb.asks
            if key not in book_side and row.queue_depth_before is not None:
                qb = float(row.queue_depth_before)
                if qb > 0:
                    book_side[key] = qb
            if row.queue_depth_after is not None:
                qa = float(row.queue_depth_after)
                if qa > 0:
                    book_side[key] = qa
                else:
                    book_side.pop(key, None)
            if row.spread_bps_at_event is not None:
                spread_bps = float(row.spread_bps_at_event)

        if not rb.bids and not rb.asks:
            return None
        return rb.to_snapshot(
            token_id=token_id,
            observed_at=target,
            spread_bps=spread_bps,
        )


# ── HybridBookSource ────────────────────────────────────────────────
#
# Per-token source dispatcher.  The matcher's source-selection used to
# be a binary "snapshots vs deltas" choice for the entire run; that
# breaks down once a third source enters the picture (parquet) AND
# coverage is heterogeneous (some tokens covered by parquet, some by
# deltas, some only by snapshots).
#
# HybridBookSource holds one underlying source per backend
# (snapshots / deltas / parquet) plus a ``routing`` dict mapping
# ``token_id -> backend_name``.  ``snapshot_at`` looks up the routed
# backend and delegates; ``iter_snapshots`` heap-merges streams so
# global ordering on observed_at is preserved.
#
# The selection of which backend wins per token is the resolver's
# job (see ``services/backtest/replay_resolver.py``); HybridBookSource
# itself is a pure dispatcher.


class HybridBookSource:
    """Dispatches per-token book reads across multiple underlying
    BookSource implementations.  Same public surface as ``BookReplay``
    so the matching engine consumes it without knowing about the
    routing.

    Usage:

        backends = {
            "parquet":   ParquetBookReplay(per_token_files=..., ...),
            "deltas":    BookDeltaReplay(token_ids=parquet_miss_tokens, ...),
            "snapshots": BookReplay(token_ids=fallback_tokens, ...),
        }
        routing = {"<token_id>": "parquet" | "deltas" | "snapshots", ...}
        source = HybridBookSource(backends=backends, routing=routing)
        await engine.run(book_source=source, ...)
    """

    def __init__(
        self,
        *,
        backends: dict[str, Any],
        routing: dict[str, str],
    ) -> None:
        # Drop empty backends so iter_snapshots doesn't waste a heap
        # slot on a no-op stream.
        self._backends: dict[str, Any] = {
            name: b for name, b in (backends or {}).items() if b is not None
        }
        self._routing: dict[str, str] = {
            str(tid): name
            for tid, name in (routing or {}).items()
            if name in self._backends
        }
        # Truncation flags merge from any underlying that truncates —
        # the matcher surfaces a single warning regardless of which
        # backend hit the issue.
        self.truncated: bool = False
        self.truncation_reason: Optional[str] = None
        self.snapshots_yielded: int = 0

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        """Heap-merge ordered streams from every active backend.
        Works because each backend itself yields in observed_at order;
        we just need to interleave.
        """
        if not self._backends:
            return
        import asyncio as _asyncio

        # Spin up one async iterator per backend.
        iterators: dict[str, AsyncIterator[BookSnapshot]] = {
            name: backend.iter_snapshots() for name, backend in self._backends.items()
        }
        # Pre-fetch the first snapshot from each so the heap has
        # something to compare on.
        head_task: dict[str, _asyncio.Task | None] = {}
        head_snap: dict[str, BookSnapshot | None] = {}
        for name, it in iterators.items():
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                first = None
            head_snap[name] = first

        total_yielded = 0
        while True:
            # Find the backend with the smallest pending observed_at,
            # filtered to ones whose token is routed to it (this lets
            # backends preload snapshots for tokens they "own" without
            # us emitting duplicates from a different backend's stream).
            candidates = [
                (snap.observed_at, name, snap)
                for name, snap in head_snap.items()
                if snap is not None
                and self._routing.get(str(snap.token_id)) == name
            ]
            if not candidates:
                # Drain any non-routed pending snapshots silently —
                # they came from a backend whose stream covers more
                # tokens than the routing assigned (e.g. a shared
                # snapshots stream that includes tokens the parquet
                # path also covers).  We just advance them.
                advanced = False
                for name, snap in list(head_snap.items()):
                    if snap is not None:
                        try:
                            head_snap[name] = await iterators[name].__anext__()
                        except StopAsyncIteration:
                            head_snap[name] = None
                        advanced = True
                if not advanced:
                    break
                continue
            candidates.sort(key=lambda x: (x[0], x[1]))
            ts, winning_name, winning_snap = candidates[0]
            yield winning_snap
            total_yielded += 1
            self.snapshots_yielded = total_yielded
            # Advance only the winning backend's stream.
            try:
                head_snap[winning_name] = await iterators[winning_name].__anext__()
            except StopAsyncIteration:
                head_snap[winning_name] = None

        # Surface truncation on the merged source if any underlying
        # truncated mid-stream.
        for name, backend in self._backends.items():
            if getattr(backend, "truncated", False):
                self.truncated = True
                reason = getattr(backend, "truncation_reason", None) or "unknown"
                self.truncation_reason = f"{name}: {reason}"
                break

    async def snapshot_at(
        self, *, token_id: str, ts: datetime
    ) -> Optional[BookSnapshot]:
        """Route to the backend that owns this token, with fallback
        through the others if the primary returns nothing (covers the
        case where a token's parquet file ends mid-window and we need
        to fall back to the live recorder for the tail)."""
        primary = self._routing.get(str(token_id))
        order: list[str] = []
        if primary and primary in self._backends:
            order.append(primary)
        # Fallbacks: try every other backend whose ``token_ids`` (if
        # exposed) might cover this token.  Most backends accept any
        # token; let snapshot_at return None to signal absence.
        for name in self._backends:
            if name not in order:
                order.append(name)
        for name in order:
            backend = self._backends[name]
            try:
                snap = await backend.snapshot_at(token_id=token_id, ts=ts)
            except Exception as exc:
                logger.debug("HybridBookSource.snapshot_at(%s) failed: %s", name, exc)
                continue
            if snap is not None:
                return snap
        return None


__all__ = [
    "PriceLevel",
    "BookSnapshot",
    "BookReplay",
    "BookDeltaReplay",
    "InMemoryBookReplay",
    "HybridBookSource",
]
