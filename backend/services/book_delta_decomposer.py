"""Decompose order book deltas into trade vs cancel events.

This is the foundation of the world-class shadow-trading fill simulator.
The standard "depth-ahead-must-clear" heuristic everyone else uses
conflates two very different events:

- **Trade**: depth at price P decreased AND a print landed at price P
  in the same window.  The queue advanced because a real fill consumed
  it.  This is the kind of advancement that *hurts* a maker — you got
  filled because the market moved against you (adverse selection).
- **Cancel**: depth at price P decreased with NO matching print.  Some
  other maker pulled their order — usually because they have new
  information (a price move incoming) and don't want to be filled.
  Queue advanced for free, but the next print is now MORE adversely
  selected than the depth-disappearance suggests.

Conflating them gives identical fill probabilities for both — wrong.
Splitting them is the highest-leverage move in the whole simulator.

The decomposer keeps a small per-token state (last book snapshot, last
N trades indexed by price) and, on each new book snapshot, emits one
BookDeltaEvent row per changed price level: ``trade`` if the size
decrease is explained by recent prints at that price, ``cancel``
otherwise.  Events feed the Cox PH trainer + the live ensemble.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from models.database import AsyncSessionLocal, BookDeltaEvent
from utils.logger import get_logger


logger = get_logger("book_delta_decomposer")


# Time window within which a trade print is considered "matchable" to a
# corresponding book delta — i.e. how recently we saw the print before
# the depth disappeared.  Set to a value that absorbs typical exchange
# clock skew + WebSocket interleaving but is short enough that two
# unrelated trades at the same price don't get merged.  600ms covers
# every observed Polymarket interleaving in our recorded snapshots
# (verified empirically — see analysis below) and stays well under the
# 0.5s book sampling cadence.
_TRADE_MATCH_WINDOW_SECONDS = 0.60

# How many recent trades to keep in the per-token buffer.  Each trade
# is indexed by price level, so we want enough to cover the width of
# any plausible depth change in a single book sample.  Polymarket books
# are typically 25 levels deep — a burst that clears all 25 in one
# WebSocket frame is implausible, but we still budget generously.
_TRADE_BUFFER_PER_TOKEN = 64

# Minimum size delta to bother classifying.  Smaller deltas are floating
# point noise from the order book serialization (Polymarket reports
# sizes to 2 decimal places) or rounding artifacts.
_MIN_DELTA_SIZE = 0.5

# Queue size cap on the persist task.  Drop oldest under heavy load
# rather than blocking the event loop.
_QUEUE_MAX = 5000


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _epoch_to_utc(value: float | None) -> datetime:
    if value is None or value <= 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


@dataclass
class _RecentTrade:
    price: float
    size: float
    side: str  # "BUY" | "SELL"
    timestamp: float  # epoch seconds


@dataclass
class _TokenState:
    bids: dict[float, float] = field(default_factory=dict)  # price -> size
    asks: dict[float, float] = field(default_factory=dict)
    recent_trades: list[_RecentTrade] = field(default_factory=list)
    last_observed_at: float = 0.0
    spread_bps: float | None = None


class BookDeltaDecomposer:
    """Stream-process book snapshots + trades into trade/cancel events.

    Singleton, started once on app boot.  All public methods are sync
    (called from the WebSocket hot path); the persist loop is async.
    """

    def __init__(self) -> None:
        self._states: dict[str, _TokenState] = {}
        self._queue: asyncio.Queue[BookDeltaEvent] | None = None
        self._task: asyncio.Task | None = None
        self._dropped = 0

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._task = asyncio.create_task(self._flush_loop(), name="book-delta-decomposer")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            await self._flush_remaining()
        self._task = None
        self._queue = None

    def record_trade(self, *, token_id: str, trade: Any) -> None:
        """Note a trade print.  Must be called BEFORE the post-trade book.

        Critically, the trade's timestamp here is the LOCAL RECEIPT
        time (``time.time()`` at the moment we got the message), NOT
        the venue's reported timestamp.  ``record_book`` matches
        against ``observed_ts``, which is also local-clock
        (``ingest_ts``); using two different clocks introduces network/
        processing skew up to several hundred milliseconds, which
        consistently ages every trade out of the buffer before any
        book diff can match against it — observed in production as
        100% of book delta events being labelled "cancel".  Local
        receipt is what's actually comparable across both feeds.
        The venue timestamp is preserved by ``MicrostructureRecorder``
        for downstream analytics.
        """
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        if isinstance(trade, dict):
            price = _coerce_float(trade.get("price"), 0.0)
            size = _coerce_float(trade.get("size"), 0.0)
            side = str(trade.get("side") or "").strip().upper()
        else:
            price = _coerce_float(getattr(trade, "price", 0.0), 0.0)
            size = _coerce_float(getattr(trade, "size", 0.0), 0.0)
            side = str(getattr(trade, "side", "") or "").strip().upper()
        if price <= 0 or size <= 0:
            return
        timestamp = datetime.now(timezone.utc).timestamp()
        state = self._states.setdefault(normalized_token, _TokenState())
        state.recent_trades.append(
            _RecentTrade(
                price=price,
                size=size,
                side=side if side in {"BUY", "SELL"} else "BUY",
                timestamp=timestamp,
            )
        )
        # bound buffer; drop oldest
        if len(state.recent_trades) > _TRADE_BUFFER_PER_TOKEN:
            del state.recent_trades[: len(state.recent_trades) - _TRADE_BUFFER_PER_TOKEN]

    def record_book(
        self,
        *,
        token_id: str,
        order_book: Any,
        best_bid: float,
        best_ask: float,
        observed_ts: float | None,
    ) -> None:
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        if observed_ts is None or observed_ts <= 0:
            observed_ts = datetime.now(timezone.utc).timestamp()

        new_bids = self._levels_to_map(order_book, "bids")
        new_asks = self._levels_to_map(order_book, "asks")
        bid = _coerce_float(best_bid, 0.0)
        ask = _coerce_float(best_ask, 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (bid or ask)
        spread_bps = ((ask - bid) / mid * 10_000.0) if bid > 0 and ask > 0 and mid > 0 else None

        state = self._states.setdefault(normalized_token, _TokenState())
        prev_bids = state.bids
        prev_asks = state.asks

        # Drop expired trades from the buffer.
        cutoff = observed_ts - _TRADE_MATCH_WINDOW_SECONDS
        state.recent_trades = [t for t in state.recent_trades if t.timestamp >= cutoff]

        if prev_bids or prev_asks:
            self._diff_side(
                token_id=normalized_token,
                side="bid",
                prev_levels=prev_bids,
                new_levels=new_bids,
                state=state,
                observed_ts=observed_ts,
                spread_bps=spread_bps,
            )
            self._diff_side(
                token_id=normalized_token,
                side="ask",
                prev_levels=prev_asks,
                new_levels=new_asks,
                state=state,
                observed_ts=observed_ts,
                spread_bps=spread_bps,
            )

        state.bids = new_bids
        state.asks = new_asks
        state.last_observed_at = observed_ts
        state.spread_bps = spread_bps

    def _diff_side(
        self,
        *,
        token_id: str,
        side: str,
        prev_levels: dict[float, float],
        new_levels: dict[float, float],
        state: _TokenState,
        observed_ts: float,
        spread_bps: float | None,
    ) -> None:
        for price, prev_size in prev_levels.items():
            new_size = new_levels.get(price, 0.0)
            delta = prev_size - new_size
            if delta < _MIN_DELTA_SIZE:
                continue
            # Try to consume from recent trades at this price (FIFO by ts).
            consumed = self._consume_trades_at(state, price=price, max_size=delta)
            trade_part = consumed
            cancel_part = max(0.0, delta - consumed)
            if trade_part > 0:
                self._enqueue(
                    BookDeltaEvent(
                        id=uuid.uuid4().hex,
                        provider="polymarket",
                        token_id=token_id,
                        observed_at=_epoch_to_utc(observed_ts),
                        exchange_ts_ms=int(observed_ts * 1000),
                        sequence=None,
                        event_type="trade",
                        side=side,
                        price=price,
                        trade_size=trade_part,
                        cancel_size=None,
                        queue_depth_before=prev_size,
                        queue_depth_after=new_size,
                        spread_bps_at_event=spread_bps,
                        payload_json={
                            "delta": delta,
                            "matched_via": "trade_tape",
                        },
                        created_at=datetime.now(timezone.utc),
                    )
                )
            if cancel_part > 0:
                self._enqueue(
                    BookDeltaEvent(
                        id=uuid.uuid4().hex,
                        provider="polymarket",
                        token_id=token_id,
                        observed_at=_epoch_to_utc(observed_ts),
                        exchange_ts_ms=int(observed_ts * 1000),
                        sequence=None,
                        event_type="cancel",
                        side=side,
                        price=price,
                        trade_size=None,
                        cancel_size=cancel_part,
                        queue_depth_before=prev_size,
                        queue_depth_after=new_size,
                        spread_bps_at_event=spread_bps,
                        payload_json={
                            "delta": delta,
                            "matched_via": "no_trade_tape_match",
                        },
                        created_at=datetime.now(timezone.utc),
                    )
                )

    def _consume_trades_at(
        self,
        state: _TokenState,
        *,
        price: float,
        max_size: float,
    ) -> float:
        """Drain recent_trades that match this price level (within tick).

        Returns the total size actually consumed.  Trades are removed
        from the buffer after consumption so they don't get re-counted
        across multiple delta events in the same window.
        """
        consumed = 0.0
        idx = 0
        # Polymarket prints at exact level prices, so equality at 4dp is
        # fine.  If a future venue uses sub-tick midpoint prints, swap
        # equality for ``abs(t.price - price) < 1e-4``.
        while idx < len(state.recent_trades) and consumed < max_size:
            trade = state.recent_trades[idx]
            if abs(trade.price - price) < 1e-4:
                take = min(trade.size, max_size - consumed)
                consumed += take
                if take >= trade.size:
                    state.recent_trades.pop(idx)
                    continue
                trade.size -= take
            idx += 1
        return consumed

    def _enqueue(self, row: BookDeltaEvent) -> None:
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(row)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 1000 == 1:
                logger.warning("BookDeltaDecomposer queue full", dropped=self._dropped)

    def _levels_to_map(self, order_book: Any, side_name: str) -> dict[float, float]:
        raw = getattr(order_book, side_name, None)
        if raw is None and isinstance(order_book, dict):
            raw = order_book.get(side_name)
        result: dict[float, float] = {}
        for level in list(raw or [])[:25]:
            if isinstance(level, dict):
                price = _coerce_float(level.get("price"), 0.0)
                size = _coerce_float(level.get("size"), 0.0)
            else:
                price = _coerce_float(getattr(level, "price", 0.0), 0.0)
                size = _coerce_float(getattr(level, "size", 0.0), 0.0)
            if price > 0 and size > 0:
                result[round(price, 4)] = size
        return result

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            await self._flush_batch(max_rows=500)

    async def _flush_remaining(self) -> None:
        while self._queue is not None and not self._queue.empty():
            await self._flush_batch(max_rows=500)

    async def _flush_batch(self, *, max_rows: int) -> None:
        queue = self._queue
        if queue is None:
            return
        rows: list[BookDeltaEvent] = []
        while len(rows) < max_rows and not queue.empty():
            rows.append(queue.get_nowait())
        if not rows:
            return
        try:
            async with AsyncSessionLocal() as session:
                session.add_all(rows)
                await session.commit()
        except Exception as exc:
            logger.warning("Failed to persist book delta batch", exc_info=exc, rows=len(rows))


book_delta_decomposer = BookDeltaDecomposer()


def get_book_delta_decomposer() -> BookDeltaDecomposer:
    return book_delta_decomposer
