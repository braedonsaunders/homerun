"""Counterfactual order replay against historical book + delta history.

Given a historical period and a hypothetical maker order ``(token_id,
side, price, size, placed_at)``, this engine asks the question every
backtester ducks: *would this order actually have filled?*

The standard backtester walks the static book at decision time and
declares "fillable size = visible depth at price".  That's wrong for
maker orders: you join the queue at price P and only fill when (a)
enough trades clear at P or better to consume the queue ahead of you,
or (b) the queue ahead is cancelled away (in which case you advance
without trading — and you may never fill if the price moves).

This engine models that explicitly.

Algorithm:

1. At placement time, take the L2 book snapshot.  Initial queue
   ahead = sum of size at the same price level (you arrive at the
   back of that queue).  Initial depth behind = sum at worse prices.
2. Stream forward through ``book_delta_events`` for that token.
3. Each ``trade`` event at price <= P (for buys) consumes its
   ``trade_size`` from your queue ahead.  When queue_ahead reaches 0,
   any further trade at price <= P fills you.
4. Each ``cancel`` event at your price decreases queue ahead too,
   but does NOT advance the fill — it just shortens the line in
   front of you.
5. Stop when the order's ``time_in_force`` elapses, the queue is
   exhausted, or the size is fully filled.

Returns a ``CounterfactualResult`` with realized fill / time-to-fill
/ remaining queue.

Used by:
* the Cox PH trainer to bootstrap synthetic labels when real fill
  history is sparse;
* the backtester's replay engine to score hypothetical strategies
  against real book history;
* the UI's "what if I had placed at price X at time T?" panel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    BookDeltaEvent,
    MarketMicrostructureSnapshot,
)


logger = logging.getLogger("counterfactual_replay")

_MIN_DELTA_SIZE = 0.01


@dataclass
class CounterfactualResult:
    filled_shares: float
    average_fill_price: float | None
    time_to_fill_seconds: float | None  # None if not filled in window
    final_queue_ahead: float
    cancels_ahead_observed: float
    trades_ahead_observed: float
    events_processed: int
    expired: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filled_shares": self.filled_shares,
            "average_fill_price": self.average_fill_price,
            "time_to_fill_seconds": self.time_to_fill_seconds,
            "final_queue_ahead": self.final_queue_ahead,
            "cancels_ahead_observed": self.cancels_ahead_observed,
            "trades_ahead_observed": self.trades_ahead_observed,
            "events_processed": self.events_processed,
            "expired": self.expired,
            "notes": self.notes,
        }


@dataclass
class CounterfactualOrder:
    token_id: str
    side: str  # "buy" | "sell"
    price: float
    size_shares: float
    placed_at: datetime
    time_in_force_seconds: float = 60.0


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _initial_queue_ahead(
    snapshot: MarketMicrostructureSnapshot | None,
    *,
    side: str,
    price: float,
) -> float:
    """Sum of same-side, same-level depth in the book at placement time.

    For a BUY at price P, you join the bid queue at P; queue ahead is
    the existing bid size at P.  For a SELL, ask queue at P.
    """
    if snapshot is None:
        return 0.0
    levels = snapshot.bids_json if side.lower().startswith("buy") else snapshot.asks_json
    if not isinstance(levels, list):
        return 0.0
    total = 0.0
    for level in levels:
        try:
            lvl_price = float(level.get("price")) if isinstance(level, dict) else float(getattr(level, "price", 0.0))
            lvl_size = float(level.get("size")) if isinstance(level, dict) else float(getattr(level, "size", 0.0))
        except Exception:
            continue
        # Polymarket prices are reported to 4dp; equality on rounded value.
        if abs(lvl_price - round(price, 4)) < 1e-4:
            total += max(0.0, lvl_size)
    return total


async def _fetch_initial_snapshot(
    session: AsyncSession,
    *,
    token_id: str,
    placed_at: datetime,
) -> MarketMicrostructureSnapshot | None:
    placed_at = _aware(placed_at)
    result = await session.execute(
        select(MarketMicrostructureSnapshot)
        .where(MarketMicrostructureSnapshot.token_id == token_id)
        .where(MarketMicrostructureSnapshot.snapshot_type == "book")
        .where(MarketMicrostructureSnapshot.observed_at <= placed_at)
        .order_by(MarketMicrostructureSnapshot.observed_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _stream_delta_events(
    session: AsyncSession,
    *,
    token_id: str,
    start_at: datetime,
    end_at: datetime,
) -> list[BookDeltaEvent]:
    start_at = _aware(start_at)
    end_at = _aware(end_at)
    result = await session.execute(
        select(BookDeltaEvent)
        .where(BookDeltaEvent.token_id == token_id)
        .where(BookDeltaEvent.observed_at > start_at)
        .where(BookDeltaEvent.observed_at <= end_at)
        .order_by(BookDeltaEvent.observed_at.asc())
    )
    return list(result.scalars().all())


async def replay_counterfactual_order(
    order: CounterfactualOrder,
    *,
    session: AsyncSession | None = None,
) -> CounterfactualResult:
    """Simulate a single hypothetical order against historical book + tape."""
    own = session is None
    if own:
        session = AsyncSessionLocal()
        await session.__aenter__()
    try:
        snapshot = await _fetch_initial_snapshot(
            session, token_id=order.token_id, placed_at=order.placed_at
        )
        if snapshot is None:
            return CounterfactualResult(
                filled_shares=0.0,
                average_fill_price=None,
                time_to_fill_seconds=None,
                final_queue_ahead=0.0,
                cancels_ahead_observed=0.0,
                trades_ahead_observed=0.0,
                events_processed=0,
                expired=True,
                notes="no_book_snapshot_at_placement",
            )

        queue_ahead = _initial_queue_ahead(
            snapshot, side=order.side, price=order.price
        )

        from datetime import timedelta

        end_at = order.placed_at + timedelta(seconds=max(0.0, order.time_in_force_seconds))
        events = await _stream_delta_events(
            session,
            token_id=order.token_id,
            start_at=order.placed_at,
            end_at=end_at,
        )

        is_buy = order.side.lower().startswith("buy")
        target_price = round(order.price, 4)
        side_key = "bid" if is_buy else "ask"

        filled_shares = 0.0
        cancels_observed = 0.0
        trades_observed = 0.0
        first_fill_at: datetime | None = None
        for ev in events:
            if filled_shares >= order.size_shares:
                break
            ev_price = round(float(ev.price or 0.0), 4)
            ev_side = (ev.side or "").lower()
            ev_type = (ev.event_type or "").lower()
            # Only events at OUR price level on OUR side advance the queue.
            if ev_side != side_key:
                continue
            if abs(ev_price - target_price) > 1e-4:
                continue
            if ev_type == "trade":
                size = float(ev.trade_size or 0.0)
                if size < _MIN_DELTA_SIZE:
                    continue
                trades_observed += size
                # First clear the queue, then fill us.
                consumed = min(size, queue_ahead)
                queue_ahead -= consumed
                remaining = size - consumed
                if remaining > 0:
                    take = min(remaining, order.size_shares - filled_shares)
                    filled_shares += take
                    if first_fill_at is None:
                        first_fill_at = _aware(ev.observed_at)
            elif ev_type == "cancel":
                size = float(ev.cancel_size or 0.0)
                if size < _MIN_DELTA_SIZE:
                    continue
                cancels_observed += size
                # Cancels in front advance you for free (queue shortens)
                # but do NOT contribute to fill.
                queue_ahead = max(0.0, queue_ahead - size)

        time_to_fill_seconds: float | None = None
        if first_fill_at is not None:
            time_to_fill_seconds = max(0.0, (first_fill_at - _aware(order.placed_at)).total_seconds())

        return CounterfactualResult(
            filled_shares=filled_shares,
            average_fill_price=order.price if filled_shares > 0 else None,
            time_to_fill_seconds=time_to_fill_seconds,
            final_queue_ahead=queue_ahead,
            cancels_ahead_observed=cancels_observed,
            trades_ahead_observed=trades_observed,
            events_processed=len(events),
            expired=filled_shares < order.size_shares,
            notes=(
                f"queue_init={_initial_queue_ahead(snapshot, side=order.side, price=order.price):.1f} "
                f"trades_at_price={trades_observed:.1f} cancels_at_price={cancels_observed:.1f}"
            ),
        )
    finally:
        if own:
            await session.__aexit__(None, None, None)
