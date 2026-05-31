"""Canonical book-snapshot reading for the unified market-data layer.

One place that turns canonical ``SNAPSHOT_SCHEMA`` parquet rows into
``BookSnapshot`` value objects and loads a token's window into an
:class:`~services.marketdata.asof.AsOfSeries`. This is what ``MarketDataView``
builds on, and what eventually replaces the bespoke readers in
``parquet_replay.py`` / ``bus_book_replay.py`` / ``_BulkBookIndex`` (Phase 3
deletes those once the engine is migrated).

``BookSnapshot`` / ``PriceLevel`` remain the engine's data contract; we import
them rather than redefining so the matcher consumes the same type whether the
source is this layer or the legacy replays during the migration window.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from services.backtest.book_replay import BookSnapshot, PriceLevel
from services.marketdata.asof import AsOfSeries

logger = logging.getLogger(__name__)


def us_from_dt(dt: datetime) -> int:
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.astimezone(timezone.utc).timestamp() * 1_000_000)


def dt_from_us(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


def row_to_book_snapshot(token_id: str, row: dict[str, Any]) -> BookSnapshot:
    """Canonical parquet-row -> BookSnapshot conversion.

    Filters invalid levels (size <= 0, price <= 0 or >= 1) and sorts bids
    descending / asks ascending — identical semantics to the legacy
    ``parquet_replay._row_to_snapshot`` so the matcher sees the same books.

    MEMORY: stores the filtered/sorted ladders as raw ``(price, size)`` tuples
    + the cached top-of-book, NOT ``PriceLevel`` objects.  ``BookSnapshot``
    builds ``PriceLevel`` lazily only when ``.bids``/``.asks`` is accessed (the
    matcher), so a sub-second discovery replay reading only top-of-book never
    pays for tens of millions of ladder objects.  Top-of-book is derived from
    the same filtered ladder, so semantics are identical to the eager path.
    """
    def _raw_levels(prices: Any, sizes: Any, *, reverse: bool) -> tuple[tuple[float, float], ...]:
        out: list[tuple[float, float]] = []
        for p, s in zip(prices or [], sizes or []):
            try:
                pf = float(p) if p is not None else 0.0
                sf = float(s) if s is not None else 0.0
            except (TypeError, ValueError):
                continue
            if pf <= 0 or pf >= 1.0 or sf <= 0:
                continue
            out.append((pf, sf))
        out.sort(key=lambda lvl: lvl[0], reverse=reverse)
        return tuple(out)

    observed_us = int(row.get("observed_at_us") or 0)
    seq = row.get("sequence")
    spread = row.get("spread_bps")
    bids_raw = _raw_levels(row.get("bids_price"), row.get("bids_size"), reverse=True)
    asks_raw = _raw_levels(row.get("asks_price"), row.get("asks_size"), reverse=False)
    return BookSnapshot(
        token_id=token_id,
        observed_at=dt_from_us(observed_us),
        sequence=int(seq) if seq is not None else None,
        spread_bps=float(spread) if spread is not None else None,
        trade_price=(float(row["trade_price"]) if row.get("trade_price") is not None else None),
        trade_size=(float(row["trade_size"]) if row.get("trade_size") is not None else None),
        trade_side=(str(row["trade_side"]) if row.get("trade_side") else None),
        top_bid=bids_raw[0][0] if bids_raw else None,
        top_ask=asks_raw[0][0] if asks_raw else None,
        bids_raw=bids_raw,
        asks_raw=asks_raw,
    )


# Columns needed to materialise a BookSnapshot (projection keeps reads cheap).
_BOOK_COLUMNS = [
    "token_id", "observed_at_us", "sequence", "best_bid", "best_ask",
    "spread_bps", "bids_price", "bids_size", "asks_price", "asks_size",
    "trade_price", "trade_size", "trade_side",
]


def load_book_series(
    token_id: str,
    files: Sequence[str | Path],
    *,
    start_us: int,
    end_us: int,
) -> tuple[AsOfSeries[BookSnapshot], int]:
    """Load a token's snapshots across ``files`` into an AsOfSeries.

    Returns ``(series, rows_read)``. Rows outside ``[start_us, end_us]`` are
    dropped. Unreadable files are skipped (logged) — the caller decides
    whether partial coverage is acceptable.
    """
    import pyarrow.parquet as pq

    series: AsOfSeries[BookSnapshot] = AsOfSeries()
    rows_read = 0
    for fp in files:
        try:
            table = pq.read_table(str(fp), columns=_BOOK_COLUMNS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("marketdata.book: unreadable %s for %s: %s", fp, token_id, exc)
            continue
        cols = {name: table.column(name).to_pylist() for name in table.schema.names}
        obs = cols.get("observed_at_us") or []
        n = len(obs)
        for i in range(n):
            o = obs[i]
            if o is None:
                continue
            o = int(o)
            if o < start_us or o > end_us:
                continue
            row = {name: cols[name][i] for name in cols}
            series.add(o, row_to_book_snapshot(token_id, row))
            rows_read += 1
    series.finalize()
    return series, rows_read


__all__ = [
    "BookSnapshot",
    "PriceLevel",
    "row_to_book_snapshot",
    "load_book_series",
    "us_from_dt",
    "dt_from_us",
]
