"""``MarketDataView`` — the unified, point-in-time market-data access layer.

The keystone of the data refactor. One object answers, for a fixed token
universe + window, the questions every consumer (backtest discovery, the
matcher, coverage reporting, and — at ``as_of=now`` — live strategies) used to
answer through ~9 divergent code paths:

  * ``book_at(token, ts)``    — point-in-time top-of-book / full book.
  * ``iter_books(...)``       — globally-ordered snapshot stream.
  * ``coverage()``            — what data exists for the universe/window.
  * ``dataset_snapshot()``    — content-hashed pin for reproducibility.

Source selection (which provider's parquet, which window files) is resolved
once via :func:`resolve_coverage` and hidden from callers. Point-in-time
correctness is delegated to :class:`AsOfSeries`, so the anti-lookahead barrier
lives in exactly one tested place.

Phase 2 ships the book side over canonical parquet. The event side
(``events()`` — recorded bus + crypto_update projection) lands next, and the
engine/live migration onto this view is Phase 3. Until then the view runs in
parallel with the legacy readers (no behavior change).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable, Optional

from services.marketdata.asof import AsOfSeries
from services.marketdata.book import BookSnapshot, load_book_series, us_from_dt
from services.marketdata.coverage import CoverageMap, resolve_coverage
from services.marketdata.manifest import DatasetSnapshot, SnapshotEntry, build_snapshot
from services.marketdata.merge import ordered_merge

logger = logging.getLogger(__name__)


class MarketDataView:
    """Point-in-time view over a fixed token universe + window.

    Construct via :meth:`build` (async — resolves coverage). Book series are
    loaded lazily per token on first access and cached, so building a view over
    a large universe is cheap and only the tokens actually queried pay I/O.
    """

    def __init__(
        self,
        *,
        coverage: CoverageMap,
        start: datetime,
        end: datetime,
    ) -> None:
        self._coverage = coverage
        self._start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        self._end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        self._start_us = us_from_dt(self._start)
        self._end_us = us_from_dt(self._end)
        self._series_cache: dict[str, AsOfSeries[BookSnapshot]] = {}
        self._rows_read: dict[str, int] = {}

    # ── construction ───────────────────────────────────────────────────
    @classmethod
    async def build(
        cls,
        *,
        token_ids: Iterable[str],
        start: datetime,
        end: datetime,
        providers: Optional[Iterable[str]] = None,
        ensure_scan: bool = True,
    ) -> "MarketDataView":
        coverage = await resolve_coverage(
            token_ids=token_ids, start=start, end=end,
            providers=providers, ensure_scan=ensure_scan,
        )
        return cls(coverage=coverage, start=start, end=end)

    # ── coverage / reproducibility ─────────────────────────────────────
    def coverage(self) -> CoverageMap:
        return self._coverage

    def dataset_snapshot(self) -> DatasetSnapshot:
        """Content-hashed pin of every parquet file this view can read.

        Uses row counts already loaded for queried tokens; unqueried files are
        pinned by stat only (path/size/mtime) — enough for the content hash and
        the pruner guard.
        """
        entries: list[SnapshotEntry] = []
        seen_paths: set[str] = set()
        for tok, tc in self._coverage.by_token.items():
            rows = self._rows_read.get(tok, 0)
            # Distribute row count across files only when a single file; for
            # multi-file tokens leave per-file rows at 0 (hash uses size+mtime).
            single = len(tc.files) == 1
            for f in tc.files:
                if f in seen_paths:
                    continue
                seen_paths.add(f)
                from pathlib import Path as _P
                try:
                    st = _P(f).stat()
                except OSError:
                    continue
                entries.append(SnapshotEntry(
                    path=f,
                    size_bytes=int(st.st_size),
                    mtime_us=int(st.st_mtime * 1_000_000),
                    rows=rows if single else 0,
                    token_ids=(tok,),
                    start_us=tc.start_us,
                    end_us=tc.end_us,
                ))
        return build_snapshot(entries)

    # ── book access ────────────────────────────────────────────────────
    def _series_for(self, token_id: str) -> AsOfSeries[BookSnapshot]:
        tok = str(token_id)
        cached = self._series_cache.get(tok)
        if cached is not None:
            return cached
        files = self._coverage.files_for(tok)
        if not files:
            series: AsOfSeries[BookSnapshot] = AsOfSeries()
            series.finalize()
            self._series_cache[tok] = series
            self._rows_read[tok] = 0
            return series
        series, rows = load_book_series(tok, files, start_us=self._start_us, end_us=self._end_us)
        self._series_cache[tok] = series
        self._rows_read[tok] = rows
        return series

    async def book_at(
        self,
        token_id: str,
        ts: datetime,
        *,
        max_staleness_seconds: Optional[float] = None,
    ) -> Optional[BookSnapshot]:
        """Most-recent book snapshot at-or-before ``ts`` (point-in-time)."""
        series = self._series_for(token_id)
        max_stale_us = int(max_staleness_seconds * 1_000_000) if max_staleness_seconds is not None else None
        return series.as_of(us_from_dt(ts), max_staleness_us=max_stale_us)

    async def iter_books(
        self,
        token_ids: Optional[Iterable[str]] = None,
    ) -> AsyncIterator[BookSnapshot]:
        """Yield BookSnapshots across tokens in global ascending order.

        Merges per-token AsOfSeries via the single ordered-merge primitive so
        the matcher sees a chronologically-correct cross-token stream.
        """
        toks = [str(t) for t in token_ids] if token_ids is not None else list(self._coverage.covered_tokens)
        series_list = [self._series_for(t) for t in toks]
        # Each series exposes (ts_us, snap) in ascending order over the window.
        per_token_streams = [s.iter_range(self._start_us, self._end_us) for s in series_list]
        for _ts_us, snap in ordered_merge(per_token_streams, key=lambda pair: pair[0]):
            yield snap

    # ── introspection ──────────────────────────────────────────────────
    @property
    def window(self) -> tuple[datetime, datetime]:
        return (self._start, self._end)

    def loaded_token_count(self) -> int:
        return sum(1 for s in self._series_cache.values() if len(s) > 0)


__all__ = ["MarketDataView"]
