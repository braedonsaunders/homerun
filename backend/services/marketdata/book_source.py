"""``_BookSource`` adapter presenting a :class:`MarketDataView` to the engine.

The matching engine (``services.backtest.engine``) consumes any object with
the ``_BookSource`` protocol — ``iter_snapshots()`` + ``snapshot_at(token_id,
ts)`` returning ``BookSnapshot``s. This adapter lets the unified market-data
view drive the matcher directly, so the engine no longer needs the bespoke
``ParquetBookReplay`` (Phase 3 routes parquet-covered tokens through here;
Phase 4 routes everything through here once SQL is dropped).

Behavior is equivalent to ``ParquetBookReplay`` by construction: both read the
same canonical parquet, both do an as-of (<= ts) lookup, and both use the same
row -> BookSnapshot conversion (``marketdata.book.row_to_book_snapshot``).
"""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Iterable, Optional

from services.marketdata.book import BookSnapshot
from services.marketdata.view import MarketDataView


class MarketDataViewSource:
    """Adapt a :class:`MarketDataView` to the engine's ``_BookSource`` protocol."""

    def __init__(self, view: MarketDataView, *, token_ids: Optional[Iterable[str]] = None) -> None:
        self._view = view
        self._tokens: Optional[list[str]] = (
            [str(t) for t in token_ids] if token_ids is not None else None
        )
        # Mirror the replay sources' truncation surface so callers can render
        # the same warning regardless of source.
        self.truncated: bool = False
        self.truncation_reason: Optional[str] = None

    async def snapshot_at(self, *, token_id: str, ts: datetime) -> Optional[BookSnapshot]:
        return await self._view.book_at(token_id, ts)

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        async for snap in self._view.iter_books(self._tokens):
            yield snap


__all__ = ["MarketDataViewSource"]
