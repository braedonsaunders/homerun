"""The single point-in-time (as-of) lookup primitive.

Every backtest reader that answers "what was the value for key K at time tau"
historically reimplemented a ``bisect_right(observed_at_us, tau) - 1`` lookup
— at least six times (ParquetBookReplay, BusBookReplay, the crypto_update
synthesizer, _BulkBookIndex, InMemoryBookReplay, and the SQL ``DESC LIMIT 1``
paths). Each enforced the as-of (<= tau) barrier on its own, with subtly
different staleness handling and no shared tests.

``AsOfSeries`` is the one tested implementation. It:

  * enforces the point-in-time barrier (never returns an entry observed
    strictly after the query time) — the core anti-lookahead guarantee;
  * supports an optional staleness bound (treat data older than N micros as
    absent), the way live feeds expire stale top-of-book;
  * is monotonic-append + finalize, then read-only, so it is cheap to build
    once per token and query many times.

All timestamps are integer microseconds since the Unix epoch (the canonical
unit across the parquet plane: ``observed_at_us``).
"""
from __future__ import annotations

import bisect
from typing import Generic, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T")


def bisect_as_of(sorted_ts_us: Sequence[int], ts_us: int) -> int:
    """Return the index of the newest entry whose timestamp is <= ``ts_us``.

    ``sorted_ts_us`` must be ascending. Returns ``-1`` when every entry is
    strictly newer than ``ts_us`` (i.e. no data is visible as-of that time).
    This is the single primitive the as-of barrier is built on.
    """
    return bisect.bisect_right(sorted_ts_us, ts_us) - 1


class AsOfSeries(Generic[T]):
    """An ascending time series with point-in-time lookup.

    Build by ``add``-ing ``(ts_us, value)`` pairs (any order), then query.
    The series finalizes lazily on first read (sorts once, stably) so callers
    don't have to remember to call :meth:`finalize`; adding after a read
    raises, keeping the read-side immutable and cheap.

    Stable sort means that when two entries share a timestamp, the one added
    later wins the as-of lookup — matching "latest observation at this instant".
    """

    __slots__ = ("_ts", "_vals", "_final")

    def __init__(self) -> None:
        self._ts: list[int] = []
        self._vals: list[T] = []
        self._final: bool = False

    # ── build ────────────────────────────────────────────────────────
    def add(self, ts_us: int, value: T) -> None:
        if self._final:
            raise RuntimeError("AsOfSeries.add() after finalize()/read")
        self._ts.append(int(ts_us))
        self._vals.append(value)

    def extend(self, pairs: "Iterator[tuple[int, T]] | Sequence[tuple[int, T]]") -> None:
        for ts_us, value in pairs:
            self.add(ts_us, value)

    def finalize(self) -> "AsOfSeries[T]":
        """Sort ascending (stable) and freeze. Idempotent; returns self."""
        if self._final:
            return self
        if self._ts:
            order = sorted(range(len(self._ts)), key=self._ts.__getitem__)
            self._ts = [self._ts[i] for i in order]
            self._vals = [self._vals[i] for i in order]
        self._final = True
        return self

    # ── query ────────────────────────────────────────────────────────
    def as_of(self, ts_us: int, *, max_staleness_us: Optional[int] = None) -> Optional[T]:
        """Newest value observed at-or-before ``ts_us``.

        Returns ``None`` when nothing is visible as-of that time, or when the
        newest visible entry is older than ``max_staleness_us`` (when given).
        """
        if not self._final:
            self.finalize()
        if not self._ts:
            return None
        i = bisect_as_of(self._ts, int(ts_us))
        if i < 0:
            return None
        if max_staleness_us is not None and (int(ts_us) - self._ts[i]) > int(max_staleness_us):
            return None
        return self._vals[i]

    def as_of_entry(self, ts_us: int, *, max_staleness_us: Optional[int] = None) -> Optional[tuple[int, T]]:
        """Like :meth:`as_of` but returns the ``(observed_at_us, value)`` pair."""
        if not self._final:
            self.finalize()
        if not self._ts:
            return None
        i = bisect_as_of(self._ts, int(ts_us))
        if i < 0:
            return None
        if max_staleness_us is not None and (int(ts_us) - self._ts[i]) > int(max_staleness_us):
            return None
        return (self._ts[i], self._vals[i])

    def iter_range(self, start_us: int, end_us: int) -> Iterator[tuple[int, T]]:
        """Yield ``(ts_us, value)`` for entries with ``start_us <= ts <= end_us``."""
        if not self._final:
            self.finalize()
        lo = bisect.bisect_left(self._ts, int(start_us))
        hi = bisect.bisect_right(self._ts, int(end_us))
        for i in range(lo, hi):
            yield (self._ts[i], self._vals[i])

    # ── introspection ──────────────────────────────────────────────────
    @property
    def first_ts_us(self) -> Optional[int]:
        if not self._final:
            self.finalize()
        return self._ts[0] if self._ts else None

    @property
    def last_ts_us(self) -> Optional[int]:
        if not self._final:
            self.finalize()
        return self._ts[-1] if self._ts else None

    def __len__(self) -> int:
        return len(self._ts)

    def __bool__(self) -> bool:
        return bool(self._ts)


__all__ = ["AsOfSeries", "bisect_as_of"]
