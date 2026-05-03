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
from dataclasses import dataclass
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
        session: AsyncSession,
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
        """Yield snapshots in (observed_at, sequence) order."""
        if not self._token_ids:
            return
        await self._raise_session_timeout()
        last_observed = self._start
        last_id: Optional[str] = None
        chunk_index = 0
        total_yielded = 0
        while True:
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
            try:
                rows = (await self._session.execute(stmt)).scalars().all()
            except Exception as exc:
                # If a chunk fails (statement_timeout under load,
                # network blip, etc.) log and stop streaming rather
                # than nuking the entire backtest.  The matching
                # engine handles a truncated replay gracefully — open
                # positions just don't see further book updates and
                # the result records whatever fills already happened.
                logger.warning(
                    "BookReplay chunk %d failed after %d snapshots; truncating replay: %s",
                    chunk_index, total_yielded, exc,
                )
                self.truncated = True
                self.truncation_reason = str(exc)[:500]
                # Roll back the failed transaction so the session can
                # be returned cleanly when the caller's ``async with``
                # exits.  Without this, asyncpg keeps the session in
                # an aborted state.
                try:
                    await self._session.rollback()
                except Exception:
                    pass
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


__all__ = [
    "PriceLevel",
    "BookSnapshot",
    "BookReplay",
    "InMemoryBookReplay",
]
