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
from typing import AsyncIterator, Iterable, Optional



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
