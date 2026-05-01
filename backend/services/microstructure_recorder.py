"""Microstructure recorder + data-quality controls.

Persists per-token book snapshots and trade prints to
``MarketMicrostructureSnapshot``.  Every accepted book passes a
structural validation gate (price bounds, level ordering, bid <= ask,
sequence monotonicity); rejects are counted by reason and surfaced
via ``get_data_quality_stats()`` for the BacktestStudio UI.

Polymarket binary tokens trade in [0, 1] (probabilities), so a level
priced at 1.5 is corrupt — silent acceptance pollutes the Cox PH
training set.  Multi-outcome markets stay in [0, 1] per outcome
token (the n outcome prices sum to ~1.0 across the parent market).
The same [0, 1] bound applies in both cases.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from models.database import AsyncSessionLocal, MarketMicrostructureSnapshot
from utils.logger import get_logger


logger = get_logger("microstructure_recorder")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _epoch_to_utc(value: float | None) -> datetime:
    if value is None or value <= 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


# Reject reasons surfaced via get_data_quality_stats().
_REJECT_REASONS = (
    "price_out_of_bounds",   # bid/ask outside [0, 1]
    "crossed_book",          # bid >= ask
    "unsorted_levels",       # bids not descending or asks not ascending
    "duplicate_price",       # same price appears twice on one side
    "stale_snapshot",        # ingest_ts older than threshold
    "clock_skew",            # ingest_ts in the future
    "sequence_regression",   # sequence number went backward
)


class MicrostructureRecorder:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[MarketMicrostructureSnapshot] | None = None
        self._task: asyncio.Task | None = None
        self._last_book_write: dict[str, float] = {}
        self._book_min_interval_seconds = 0.50
        self._max_levels = 25
        self._dropped = 0
        # Data quality tracking
        self._last_sequence: dict[str, int] = {}
        self._sequence_gaps: int = 0
        self._reject_counts: dict[str, int] = {r: 0 for r in _REJECT_REASONS}
        self._accepted_books: int = 0
        self._stale_seconds_threshold: float = 30.0
        self._clock_skew_seconds_tolerance: float = 5.0

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._queue = asyncio.Queue(maxsize=5000)
        self._task = asyncio.create_task(self._flush_loop(), name="microstructure-recorder")

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

    def record_book(
        self,
        *,
        token_id: str,
        order_book: Any,
        best_bid: float,
        best_ask: float,
        exchange_ts: float | None,
        ingest_ts: float | None,
        sequence: int | None,
    ) -> None:
        queue = self._queue
        if queue is None:
            return
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        now_epoch = _coerce_float(ingest_ts, 0.0)
        if now_epoch <= 0:
            now_epoch = datetime.now(timezone.utc).timestamp()
        last = self._last_book_write.get(normalized_token, 0.0)
        if now_epoch - last < self._book_min_interval_seconds:
            return
        self._last_book_write[normalized_token] = now_epoch
        bid = _coerce_float(best_bid, 0.0)
        ask = _coerce_float(best_ask, 0.0)
        bids = self._levels(order_book, "bids")
        asks = self._levels(order_book, "asks")

        # ── Data quality gate ─────────────────────────────────────
        # Validate before persistence: price bounds, level ordering,
        # crossed book, duplicate prices, sequence regression, clock
        # skew.  Rejects are counted by reason and surfaced via
        # get_data_quality_stats() for the BacktestStudio UI.  We
        # accept "thin" books (e.g., ask=0 means no asks visible)
        # because that's a real market state, not a corruption.
        reject = self._validate_book(
            token_id=normalized_token,
            bid=bid,
            ask=ask,
            bids=bids,
            asks=asks,
            ingest_ts=now_epoch,
            sequence=sequence,
        )
        if reject is not None:
            self._reject_counts[reject] = self._reject_counts.get(reject, 0) + 1
            return

        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid or ask
        spread_bps = ((ask - bid) / mid * 10_000.0) if bid > 0 and ask > 0 and mid > 0 else None
        self._accepted_books += 1
        row = MarketMicrostructureSnapshot(
            id=uuid.uuid4().hex,
            provider="polymarket",
            token_id=normalized_token,
            snapshot_type="book",
            observed_at=_epoch_to_utc(now_epoch),
            exchange_ts_ms=int(float(exchange_ts or now_epoch) * 1000),
            sequence=int(sequence or 0) or None,
            best_bid=bid or None,
            best_ask=ask or None,
            spread_bps=spread_bps,
            bids_json=bids,
            asks_json=asks,
            payload_json={
                "level_count": {"bids": len(bids), "asks": len(asks)},
                "ingest_ts": now_epoch,
            },
            created_at=datetime.now(timezone.utc),
        )
        self._enqueue(row)

    def record_trade(self, *, token_id: str, trade: Any) -> None:
        queue = self._queue
        if queue is None:
            return
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        if isinstance(trade, dict):
            price = _coerce_float(trade.get("price"), 0.0)
            size = _coerce_float(trade.get("size"), 0.0)
            side = str(trade.get("side") or "").strip().upper()
            timestamp = _coerce_float(trade.get("timestamp"), 0.0)
        else:
            price = _coerce_float(getattr(trade, "price", 0.0), 0.0)
            size = _coerce_float(getattr(trade, "size", 0.0), 0.0)
            side = str(getattr(trade, "side", "") or "").strip().upper()
            timestamp = _coerce_float(getattr(trade, "timestamp", 0.0), 0.0)
        if price <= 0 or size <= 0:
            return
        row = MarketMicrostructureSnapshot(
            id=uuid.uuid4().hex,
            provider="polymarket",
            token_id=normalized_token,
            snapshot_type="trade",
            observed_at=_epoch_to_utc(timestamp),
            exchange_ts_ms=int((timestamp if timestamp > 0 else datetime.now(timezone.utc).timestamp()) * 1000),
            trade_price=price,
            trade_size=size,
            trade_side=side if side in {"BUY", "SELL"} else "BUY",
            payload_json={},
            created_at=datetime.now(timezone.utc),
        )
        self._enqueue(row)

    def _enqueue(self, row: MarketMicrostructureSnapshot) -> None:
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(row)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 1000 == 1:
                logger.warning("Microstructure recorder queue is full", dropped=self._dropped)

    def _validate_book(
        self,
        *,
        token_id: str,
        bid: float,
        ask: float,
        bids: list[dict[str, float]],
        asks: list[dict[str, float]],
        ingest_ts: float,
        sequence: int | None,
    ) -> str | None:
        """Return reject reason or None (accept).  Cheap checks only —
        this runs in the WebSocket hot path on every book update.
        """
        # Price bounds: outcome tokens trade in [0, 1].  Allow 0 (no
        # bid/ask present means thin book) but reject anything outside.
        for level in bids + asks:
            price = float(level.get("price") or 0.0)
            if price < 0.0 or price > 1.0:
                return "price_out_of_bounds"
        if bid > 1.0 or ask > 1.0 or bid < 0.0 or ask < 0.0:
            return "price_out_of_bounds"

        # Crossed book: bid >= ask is corrupt unless one side is empty.
        if bid > 0 and ask > 0 and bid >= ask:
            return "crossed_book"

        # Level ordering: bids descending, asks ascending; no duplicates.
        if not self._levels_sorted_descending(bids):
            return "unsorted_levels"
        if not self._levels_sorted_ascending(asks):
            return "unsorted_levels"
        if self._has_duplicate_prices(bids) or self._has_duplicate_prices(asks):
            return "duplicate_price"

        # Stale snapshot: ingest_ts more than _stale_seconds_threshold
        # behind wall clock.  Distinguish from clock_skew (ahead).
        wall = datetime.now(timezone.utc).timestamp()
        if ingest_ts > 0:
            age = wall - ingest_ts
            if age > self._stale_seconds_threshold:
                return "stale_snapshot"
            if age < -self._clock_skew_seconds_tolerance:
                return "clock_skew"

        # Sequence regression: track last sequence per token; if a
        # snapshot arrives with a smaller seq than the last accepted,
        # reject.  Gaps (seq jumps forward by >1) are counted but
        # don't reject — the missing snapshot may be a genuine drop.
        if sequence is not None:
            try:
                seq_int = int(sequence)
            except (TypeError, ValueError):
                seq_int = 0
            if seq_int > 0:
                prev = self._last_sequence.get(token_id)
                if prev is not None:
                    if seq_int < prev:
                        return "sequence_regression"
                    if seq_int > prev + 1:
                        self._sequence_gaps += 1
                self._last_sequence[token_id] = seq_int

        return None

    @staticmethod
    def _levels_sorted_descending(levels: list[dict[str, float]]) -> bool:
        prev = None
        for level in levels:
            price = float(level.get("price") or 0.0)
            if prev is not None and price > prev:
                return False
            prev = price
        return True

    @staticmethod
    def _levels_sorted_ascending(levels: list[dict[str, float]]) -> bool:
        prev = None
        for level in levels:
            price = float(level.get("price") or 0.0)
            if prev is not None and price < prev:
                return False
            prev = price
        return True

    @staticmethod
    def _has_duplicate_prices(levels: list[dict[str, float]]) -> bool:
        seen: set[float] = set()
        for level in levels:
            price = float(level.get("price") or 0.0)
            if price in seen:
                return True
            seen.add(price)
        return False

    def get_data_quality_stats(self) -> dict[str, Any]:
        """Snapshot of data-quality counters for the UI.  Cheap call —
        returns the in-memory counters without DB access.
        """
        total_attempts = self._accepted_books + sum(self._reject_counts.values())
        accept_rate = (
            self._accepted_books / total_attempts if total_attempts > 0 else None
        )
        return {
            "accepted_books": self._accepted_books,
            "total_attempts": total_attempts,
            "accept_rate": accept_rate,
            "rejects_by_reason": dict(self._reject_counts),
            "sequence_gaps_observed": self._sequence_gaps,
            "tokens_tracked": len(self._last_sequence),
            "queue_dropped": self._dropped,
        }

    def _levels(self, order_book: Any, side_name: str) -> list[dict[str, float]]:
        raw = getattr(order_book, side_name, None)
        if raw is None and isinstance(order_book, dict):
            raw = order_book.get(side_name)
        rows: list[dict[str, float]] = []
        for level in list(raw or [])[: self._max_levels]:
            if isinstance(level, dict):
                price = _coerce_float(level.get("price"), 0.0)
                size = _coerce_float(level.get("size"), 0.0)
            else:
                price = _coerce_float(getattr(level, "price", 0.0), 0.0)
                size = _coerce_float(getattr(level, "size", 0.0), 0.0)
            if price > 0 and size > 0:
                rows.append({"price": price, "size": size})
        return rows

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(0.20)
            await self._flush_batch(max_rows=250)

    async def _flush_remaining(self) -> None:
        while self._queue is not None and not self._queue.empty():
            await self._flush_batch(max_rows=500)

    async def _flush_batch(self, *, max_rows: int) -> None:
        queue = self._queue
        if queue is None:
            return
        rows: list[MarketMicrostructureSnapshot] = []
        while len(rows) < max_rows and not queue.empty():
            rows.append(queue.get_nowait())
        if not rows:
            return
        try:
            async with AsyncSessionLocal() as session:
                session.add_all(rows)
                await session.commit()
        except Exception as exc:
            logger.warning("Failed to persist microstructure batch", exc_info=exc, rows=len(rows))


microstructure_recorder = MicrostructureRecorder()


def get_microstructure_recorder() -> MicrostructureRecorder:
    return microstructure_recorder
