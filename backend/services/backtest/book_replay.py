"""Canonical L2 book value types + an in-memory replay for tests.

Defines the immutable book primitives the whole backtester speaks in —
``PriceLevel`` and ``BookSnapshot`` — plus ``InMemoryBookReplay``, which lets
tests and sparse-data tokens seed synthetic snapshots directly.

Real point-in-time book access (reading recorded books for a window) now lives
in the unified ``services.marketdata`` layer: ``MarketDataView`` resolves the
canonical parquet plane and ``MarketDataViewSource`` adapts it to the matching
engine's ``snapshot_at`` / ``iter_snapshots`` contract.  The legacy
SQL-backed replay (``MarketMicrostructureSnapshot``) was retired in the
market-data clean cut, so this module no longer touches any database.
"""

from __future__ import annotations

import bisect
import logging
import os as _os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable, Optional, Sequence



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


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """One side of a single book level."""

    price: float
    size: float


class BookSnapshot:
    """Immutable L2 snapshot at a point in time.

    ``bids`` are descending by price (best bid first); ``asks`` are
    ascending (best ask first). ``mid`` is None when either side is empty.

    MEMORY: the full L2 ladders are materialised LAZILY.  A 4h × 177-token
    sub-second backtest loads ~1.2M snapshots; eagerly building a ``PriceLevel``
    object per level (~25/side) is tens of millions of objects (multi-GB) — yet
    the discovery loop only ever reads top-of-book (``best_bid``/``best_ask``),
    and the matcher only walks ladders for the handful of snapshots it actually
    fills against.  So we store the raw ``(price, size)`` arrays + the cached
    top-of-book and build ``PriceLevel`` tuples on first ``.bids``/``.asks``
    access.  Top-of-book (the hot path) costs nothing; ladders cost only when
    the matcher touches them.  Construction stays backward-compatible — callers
    that pass ``bids=``/``asks=`` PriceLevel tuples (tests, InMemoryBookReplay,
    the live path) work unchanged.
    """

    __slots__ = (
        "token_id", "observed_at", "sequence", "spread_bps",
        "trade_price", "trade_size", "trade_side",
        "_top_bid", "_top_ask", "_bids", "_asks", "_bids_raw", "_asks_raw",
    )

    def __init__(
        self,
        token_id: str,
        observed_at: datetime,
        bids: Optional[tuple[PriceLevel, ...]] = None,
        asks: Optional[tuple[PriceLevel, ...]] = None,
        sequence: Optional[int] = None,
        spread_bps: Optional[float] = None,
        trade_price: Optional[float] = None,
        trade_size: Optional[float] = None,
        trade_side: Optional[str] = None,
        *,
        top_bid: Optional[float] = None,
        top_ask: Optional[float] = None,
        bids_raw: Optional[Sequence[tuple[float, float]]] = None,
        asks_raw: Optional[Sequence[tuple[float, float]]] = None,
    ) -> None:
        self.token_id = token_id
        self.observed_at = observed_at
        self.sequence = sequence
        self.spread_bps = spread_bps
        self.trade_price = trade_price
        self.trade_size = trade_size
        self.trade_side = trade_side
        self._top_bid = top_bid
        self._top_ask = top_ask
        # Eager path (legacy): PriceLevel tuples handed in directly.
        # Lazy path: raw (price, size) arrays, materialised on first access.
        self._bids: Optional[tuple[PriceLevel, ...]] = tuple(bids) if bids is not None else None
        self._asks: Optional[tuple[PriceLevel, ...]] = tuple(asks) if asks is not None else None
        self._bids_raw = bids_raw
        self._asks_raw = asks_raw

    @property
    def bids(self) -> tuple[PriceLevel, ...]:
        if self._bids is None:
            self._bids = tuple(PriceLevel(price=p, size=s) for p, s in (self._bids_raw or ()))
        return self._bids

    @property
    def asks(self) -> tuple[PriceLevel, ...]:
        if self._asks is None:
            self._asks = tuple(PriceLevel(price=p, size=s) for p, s in (self._asks_raw or ()))
        return self._asks

    @property
    def best_bid(self) -> Optional[float]:
        if self._top_bid is not None:
            return self._top_bid
        b = self.bids
        return b[0].price if b else None

    @property
    def best_ask(self) -> Optional[float]:
        if self._top_ask is not None:
            return self._top_ask
        a = self.asks
        return a[0].price if a else None

    def __repr__(self) -> str:  # lightweight; avoids forcing ladder materialisation
        return (
            f"BookSnapshot(token_id={self.token_id!r}, observed_at={self.observed_at!r}, "
            f"best_bid={self.best_bid}, best_ask={self.best_ask})"
        )

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
