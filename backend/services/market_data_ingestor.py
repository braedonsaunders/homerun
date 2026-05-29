"""Unified market-data ingestor: persists snapshots + deltas in one pass.

Replaces the previous split between ``microstructure_recorder.py``
(book snapshots) and ``book_delta_decomposer.py`` (book deltas).  Both
were called from the WS hot path, walking the book levels twice and
duplicating validation work.  Unifying them halves the hot-path cost
AND ensures the canonical book parquet is always populated whenever the
live system runs — no separate recorder process required.  The flush
writes canonical ``snapshots__`` / ``deltas__`` parquet (off Postgres);
the SQL book tables were retired in the market-data clean cut.

Design constraints (financial-institution-grade, sub-second loop
critical path):

  * **Hot path is sync, lock-free, zero I/O.**  ``record_book`` and
    ``record_trade`` are called from the Polymarket WebSocket
    callback.  Any await, any synchronous DB call, any unbounded loop
    here would compete with the orchestrator's sub-second decision
    loop on the same asyncio event loop.  All work is: validate (in-
    memory), update per-token state (dict mutation), enqueue (drop-on-
    full).  Microseconds, not milliseconds.

  * **Bounded memory + bounded queues.**  Per-token state caps the
    trade-tape buffer at 64 entries.  Both persist queues are sized
    (default 5000).  Overflow drops the oldest, never blocks the
    producer.

  * **Two queues, independent flush cadences.**  Snapshots write at
    most every ``snapshot_throttle_seconds`` per token (default 0.5s
    → 2/sec/token max).  Deltas write on every changed level — they
    can spike to thousands/sec across the universe during volatile
    moments, so their flush task batches up to 500 rows every 250ms.

  * **Single level-walk per book event.**  The previous split walked
    levels in both ``MicrostructureRecorder._levels`` AND
    ``BookDeltaDecomposer._levels_to_map``.  The unified ingestor
    walks once and reuses the parsed dicts for both delta diffing
    and snapshot serialization.

  * **Validation rejects propagate to BOTH paths.**  A book that
    fails the data-quality gate (price out of bounds, crossed,
    unsorted, dup price, stale, clock skew, sequence regression) is
    persisted nowhere — neither as a snapshot nor as a delta.  This
    matches institutional data-quality discipline: bad data should
    not contaminate the training set OR the backtest replay.

  * **Tradeoff: snapshot throttle vs delta fidelity.**  Deltas fire
    on every level change (no throttle).  Snapshots throttle to
    bound persistence load.  Backtest replay can use either:
    snapshots are good as anchors / for legacy code paths; the
    delta stream is the authoritative live source and ``BookDeltaReplay``
    reconstructs full book state by replaying deltas from the most
    recent snapshot anchor.

This module is a singleton — one instance per process, started once
at app boot from ``ws_feeds.FeedManager.start``.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from services.external_data.book_parquet_sink import (
    BookDeltaRow,
    BookSnapshotRow,
)
from services.live_pressure import (
    current_backpressure_level,
    is_db_pressure_active,
    publish_backpressure,
)
from utils.logger import get_logger


logger = get_logger("market_data_ingestor")


# ── Tuning constants ──────────────────────────────────────────────────────
#
# All thresholds here have institutional-grade rationale and are
# referenced from the surrounding code by name to make tuning explicit.

# Trade-tape buffer for delta classification.  Polymarket books are
# 25 levels deep; a single WS frame clearing every level is implausible
# but we budget generously for burst tolerance.
_TRADE_BUFFER_PER_TOKEN = 64

# Window within which a print can be matched to a corresponding depth
# delta.  600ms covers observed Polymarket WS interleaving and stays
# under the 0.5s book sampling cadence so two unrelated trades at the
# same price don't get conflated.
_TRADE_MATCH_WINDOW_SECONDS = 0.60

# Floor for delta classification.  Sizes are reported to 2 decimals; a
# 0.5-share delta is the smallest plausibly-real change.
_MIN_DELTA_SIZE = 0.5

# Snapshot throttle.  Cap one snapshot per token every 0.5s — matches
# Polymarket's natural book-update cadence and prevents the persistence
# layer from drowning in identical snapshots when the book is quiet.
_DEFAULT_SNAPSHOT_THROTTLE_SECONDS = 0.50

# Validation thresholds.
_STALE_SECONDS_THRESHOLD = 30.0       # ingest_ts older than this → reject
_CLOCK_SKEW_TOLERANCE_SECONDS = 5.0   # ingest_ts ahead of wall-clock by > this → reject

# Persistence queue depths.  Drop-on-full rather than block.
_SNAPSHOT_QUEUE_MAX = 5000
_DELTA_QUEUE_MAX = 10000              # 2x snapshot queue — deltas are higher freq

# Flush batch + cadence.  Tuned so each batch insert completes in
# <100ms p99 under typical load (asyncpg COPY-equivalent INSERT batch).
_SNAPSHOT_FLUSH_BATCH = 250
_DELTA_FLUSH_BATCH = 500
_FLUSH_INTERVAL_SECONDS = 0.20
_QUEUE_HIGH_WATER_FRACTION = 0.80
_PRESSURE_SHED_LEVEL = 0.70
_SLOW_FLUSH_BACKPRESSURE_MS = 1500.0
_PRESSURE_PROBE_INTERVAL_SECONDS = 0.25
_PRESSURE_SHED_CACHE_SECONDS = 0.50

# Reject reasons surfaced via get_data_quality_stats().
_REJECT_REASONS = (
    "price_out_of_bounds",
    "crossed_book",
    "unsorted_levels",
    "duplicate_price",
    "stale_snapshot",
    "clock_skew",
    "sequence_regression",
)

# Maximum levels per side persisted.  Polymarket exposes 25 levels per
# side; we clamp to that to bound the JSON payload size.
_MAX_LEVELS_PER_SIDE = 25


# ── Helpers ───────────────────────────────────────────────────────────────


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _epoch_to_utc(value: float | None) -> datetime:
    if value is None or value <= 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


# ── Per-token state ───────────────────────────────────────────────────────


@dataclass
class _RecentTrade:
    price: float
    size: float
    side: str  # "BUY" | "SELL"
    timestamp: float


@dataclass
class _TokenState:
    """In-memory state for a single token's running book + trade tape."""

    # Running book — keys are prices (rounded to 4dp), values are sizes.
    # Mutated on every record_book; used for delta diffing.
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    # Recent prints, FIFO ordered by timestamp, capped at
    # _TRADE_BUFFER_PER_TOKEN.  Used by delta classification to decide
    # whether a depth decrease was a trade or a cancel.
    recent_trades: list[_RecentTrade] = field(default_factory=list)
    # Last observed_ts seen for this token (epoch seconds).
    last_observed_at: float = 0.0
    # Most recent spread (bps).  Cached on the state so delta events
    # carry the prevailing spread without recomputing.
    spread_bps: float | None = None
    # Last snapshot persistence time (epoch seconds).  Throttle gate
    # for the snapshot write path.
    last_snapshot_write: float = 0.0
    # Last accepted sequence (for sequence-regression detection).
    last_sequence: int | None = None


# ── Ingestor ──────────────────────────────────────────────────────────────


class LiveMarketDataIngestor:
    """Singleton: validates, classifies, and persists every WS book/trade.

    Lifecycle:
      * ``start()``  — called once on app boot.  Creates queues + flush task.
      * ``stop()``   — called on shutdown.  Drains queues, persists tail.
      * ``record_book / record_trade`` — sync, hot-path, no awaits.

    Public stats (UI / diagnostics):
      * ``get_data_quality_stats()`` — counters: accepted/rejected by reason,
        sequence gaps, queue drop counts, flush latency p50/p95.
    """

    def __init__(
        self,
        *,
        snapshot_throttle_seconds: float = _DEFAULT_SNAPSHOT_THROTTLE_SECONDS,
    ) -> None:
        self._snapshot_throttle_seconds = float(snapshot_throttle_seconds)
        self._states: dict[str, _TokenState] = {}
        self._snapshot_queue: asyncio.Queue[BookSnapshotRow] | None = None
        self._delta_queue: asyncio.Queue[BookDeltaRow] | None = None
        self._snapshot_flush_task: asyncio.Task | None = None
        self._delta_flush_task: asyncio.Task | None = None

        # Counters (all in-memory, lock-free reads).
        self._snapshot_dropped = 0
        self._delta_dropped = 0
        self._recording_disabled_dropped = 0
        self._accepted_books = 0
        self._reject_counts: dict[str, int] = {r: 0 for r in _REJECT_REASONS}
        self._sequence_gaps = 0
        self._flush_latency_samples: list[float] = []  # bounded sliding window
        self._last_pressure_log_mono = 0.0
        self._last_pressure_probe_mono = 0.0
        self._pressure_shed_until_mono = 0.0
        # Columnar parquet sink — the OFF-Postgres persistence target for
        # books/deltas.  Instantiated in start(); see book_parquet_sink.
        self._book_sink: Any = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create queues + spawn flush tasks.  Idempotent."""
        if self._snapshot_flush_task is not None and not self._snapshot_flush_task.done():
            return
        self._snapshot_queue = asyncio.Queue(maxsize=_SNAPSHOT_QUEUE_MAX)
        self._delta_queue = asyncio.Queue(maxsize=_DELTA_QUEUE_MAX)
        # Columnar parquet sink — persistence runs off Postgres + off this
        # loop (its own flush loop uses to_thread).  Lazy import so pyarrow
        # only loads where the ingestor actually runs.
        try:
            from services.external_data.book_parquet_sink import BookParquetSink
            self._book_sink = BookParquetSink()
            asyncio.create_task(self._book_sink.start())
        except Exception:
            logger.exception("LiveMarketDataIngestor: failed to start book parquet sink")
            self._book_sink = None
        self._snapshot_flush_task = asyncio.create_task(
            self._flush_loop(
                queue=self._snapshot_queue,
                batch=_SNAPSHOT_FLUSH_BATCH,
                kind="snapshot",
            ),
            name="market-data-ingestor-snapshots",
        )
        self._delta_flush_task = asyncio.create_task(
            self._flush_loop(
                queue=self._delta_queue,
                batch=_DELTA_FLUSH_BATCH,
                kind="delta",
            ),
            name="market-data-ingestor-deltas",
        )
        logger.info(
            "LiveMarketDataIngestor started "
            f"(snapshot_throttle={self._snapshot_throttle_seconds}s, "
            f"snapshot_queue_max={_SNAPSHOT_QUEUE_MAX}, "
            f"delta_queue_max={_DELTA_QUEUE_MAX})"
        )

    async def stop(self) -> None:
        """Cancel flush tasks, drain remaining rows, clear state."""
        for task in (self._snapshot_flush_task, self._delta_flush_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Drain any remaining rows so we don't lose late data on shutdown.
        await self._drain(self._snapshot_queue, kind="snapshot")
        await self._drain(self._delta_queue, kind="delta")
        if self._book_sink is not None:
            try:
                await self._book_sink.stop()
            except Exception:
                logger.debug("book sink stop failed", exc_info=True)
            self._book_sink = None
        self._snapshot_flush_task = None
        self._delta_flush_task = None
        self._snapshot_queue = None
        self._delta_queue = None

    # ── Hot-path ingestion (sync, no awaits) ─────────────────────────────

    def record_trade(self, *, token_id: str, trade: Any) -> None:
        """Note a trade print.  Must be called BEFORE the post-trade book.

        The trade timestamp is the LOCAL receipt time (``time.time()``)
        — NOT the venue's reported ts — to match against the local-clock
        ``observed_ts`` of the book update arriving microseconds later.
        Mixing clocks consistently aged every trade out of the matching
        window in production, labeling 100% of deltas as ``cancel``.
        """
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        if isinstance(trade, dict):
            price = _coerce_float(trade.get("price"), 0.0)
            size = _coerce_float(trade.get("size"), 0.0)
            side = str(trade.get("side") or "").strip().upper()
            venue_ts = _coerce_float(trade.get("timestamp"), 0.0)
        else:
            price = _coerce_float(getattr(trade, "price", 0.0), 0.0)
            size = _coerce_float(getattr(trade, "size", 0.0), 0.0)
            side = str(getattr(trade, "side", "") or "").strip().upper()
            venue_ts = _coerce_float(getattr(trade, "timestamp", 0.0), 0.0)
        if price <= 0 or size <= 0:
            return

        now = time.time()
        state = self._states.setdefault(normalized_token, _TokenState())
        state.recent_trades.append(
            _RecentTrade(
                price=price,
                size=size,
                side=side if side in {"BUY", "SELL"} else "BUY",
                timestamp=now,
            )
        )
        # Bound buffer: drop oldest beyond the cap.
        if len(state.recent_trades) > _TRADE_BUFFER_PER_TOKEN:
            del state.recent_trades[: len(state.recent_trades) - _TRADE_BUFFER_PER_TOKEN]

        # Also persist as a separate ``snapshot_type='trade'`` row in mms.
        # This preserves the trade tape for downstream analytics
        # (counterfactual replay, latency-distribution capture) without
        # needing a separate trades table.  Same throttle as snapshots
        # — cheap because trades are far less frequent than book ticks.
        snap_queue = self._snapshot_queue
        if snap_queue is not None:
            row = BookSnapshotRow(
                id=uuid.uuid4().hex,
                provider="polymarket",
                token_id=normalized_token,
                snapshot_type="trade",
                observed_at=_epoch_to_utc(venue_ts if venue_ts > 0 else now),
                exchange_ts_ms=int((venue_ts if venue_ts > 0 else now) * 1000),
                trade_price=price,
                trade_size=size,
                trade_side=side if side in {"BUY", "SELL"} else "BUY",
                payload_json={},
                created_at=datetime.now(timezone.utc),
            )
            self._enqueue_snapshot(row)

    def record_book(
        self,
        *,
        token_id: str,
        order_book: Any,
        best_bid: float,
        best_ask: float,
        exchange_ts: float | None = None,
        ingest_ts: float | None = None,
        sequence: int | None = None,
    ) -> None:
        """Process one book update.

        Pipeline (all sync, all in-memory until the queue.put_nowait at
        the end):
          1. Coerce inputs, default ts to wall clock.
          2. Walk levels ONCE → ``new_bids``, ``new_asks`` dicts.
          3. Validate (cheap structural checks); reject without
             persisting if the book is malformed.
          4. Diff vs prev state → enqueue delta events.
          5. Update running state (bids, asks, last_observed_at).
          6. If snapshot throttle window has elapsed for this token,
             enqueue a full ``BookSnapshotRow`` row.

        Recorded-event bus contract: this hot path deliberately does
        NOT call ``bus.publish`` for ``polymarket.book.snapshot`` or
        ``polymarket.book.delta``.  Any await here would compete with
        the orchestrator's sub-second decision loop.  The bus exposes
        both topics as external_parquet adapters reading the same
        canonical ``snapshots__`` / ``deltas__`` parquet files this
        ingestor's flush writes — subscribers consuming via
        ``bus.replay`` see identical data, ~500ms behind live (the
        flush cadence) for the snapshot stream and within the delta
        flush window for the delta stream.  Strategies that need
        microsecond-fresh book state continue to use the existing
        feed-manager callback path; the bus is the unified API for
        everything that can tolerate the flush latency, which
        includes every backtest replay path."""
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return

        ingest = _coerce_float(ingest_ts, 0.0)
        if ingest <= 0:
            ingest = time.time()

        # Walk levels once.  These are the canonical parsed structures
        # for both delta diffing AND the snapshot serialization below.
        new_bids = self._levels_to_map(order_book, "bids")
        new_asks = self._levels_to_map(order_book, "asks")
        bid = _coerce_float(best_bid, 0.0)
        ask = _coerce_float(best_ask, 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (bid or ask)
        spread_bps = (
            ((ask - bid) / mid * 10_000.0)
            if bid > 0 and ask > 0 and mid > 0
            else None
        )

        state = self._states.setdefault(normalized_token, _TokenState())

        # Validation gate.  Cheap dict iteration + arithmetic; no I/O.
        reject = self._validate_book(
            bid=bid,
            ask=ask,
            new_bids=new_bids,
            new_asks=new_asks,
            ingest_ts=ingest,
            sequence=sequence,
            state=state,
        )
        if reject is not None:
            self._reject_counts[reject] = self._reject_counts.get(reject, 0) + 1
            return

        # Drop expired trades from the matching buffer.
        cutoff = ingest - _TRADE_MATCH_WINDOW_SECONDS
        state.recent_trades = [t for t in state.recent_trades if t.timestamp >= cutoff]

        # Diff + emit deltas (only after the first book — diffing
        # against an empty prev state would emit 50 spurious "cancels"
        # at startup as the recorder learned the initial book).
        if state.bids or state.asks:
            self._diff_and_emit(
                token_id=normalized_token,
                side="bid",
                prev_levels=state.bids,
                new_levels=new_bids,
                state=state,
                observed_ts=ingest,
                spread_bps=spread_bps,
            )
            self._diff_and_emit(
                token_id=normalized_token,
                side="ask",
                prev_levels=state.asks,
                new_levels=new_asks,
                state=state,
                observed_ts=ingest,
                spread_bps=spread_bps,
            )

        # Update running state.
        state.bids = new_bids
        state.asks = new_asks
        state.last_observed_at = ingest
        state.spread_bps = spread_bps
        self._accepted_books += 1

        # Snapshot persistence (throttled per token).  This is the path
        # that makes the snapshot table self-populating — no separate
        # recorder process required.
        if ingest - state.last_snapshot_write >= self._snapshot_throttle_seconds:
            state.last_snapshot_write = ingest
            self._enqueue_snapshot(
                BookSnapshotRow(
                    id=uuid.uuid4().hex,
                    provider="polymarket",
                    token_id=normalized_token,
                    snapshot_type="book",
                    observed_at=_epoch_to_utc(ingest),
                    exchange_ts_ms=int(float(exchange_ts or ingest) * 1000),
                    sequence=int(sequence or 0) or None,
                    best_bid=bid or None,
                    best_ask=ask or None,
                    spread_bps=spread_bps,
                    bids_json=self._levels_dict_to_list(new_bids, descending=True),
                    asks_json=self._levels_dict_to_list(new_asks, descending=False),
                    payload_json={
                        "level_count": {"bids": len(new_bids), "asks": len(new_asks)},
                        "ingest_ts": ingest,
                    },
                    created_at=datetime.now(timezone.utc),
                )
            )

    # ── Delta classification ────────────────────────────────────────────

    def _diff_and_emit(
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
        """For each price level whose size DECREASED, classify the cause
        (trade vs cancel) by attempting to consume from the recent-trade
        tape.  Emit one BookDeltaRow per non-zero classification.
        """
        for price, prev_size in prev_levels.items():
            new_size = new_levels.get(price, 0.0)
            delta = prev_size - new_size
            if delta < _MIN_DELTA_SIZE:
                continue
            consumed = self._consume_trades_at(state, price=price, max_size=delta)
            trade_part = consumed
            cancel_part = max(0.0, delta - consumed)
            if trade_part > 0:
                self._enqueue_delta(
                    BookDeltaRow(
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
                self._enqueue_delta(
                    BookDeltaRow(
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
        """FIFO drain of recent trades whose price equals ``price`` (4dp
        equality; Polymarket prints at exact tick prices).  Returns the
        total size consumed.  Trades are removed from the buffer after
        consumption to prevent double-counting across multiple deltas
        in the same window.
        """
        consumed = 0.0
        idx = 0
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

    # ── Validation ──────────────────────────────────────────────────────

    def _validate_book(
        self,
        *,
        bid: float,
        ask: float,
        new_bids: dict[float, float],
        new_asks: dict[float, float],
        ingest_ts: float,
        sequence: int | None,
        state: _TokenState,
    ) -> str | None:
        """Cheap structural checks.  Hot path, no I/O."""
        # Price bounds: outcome tokens trade in [0, 1].
        for price in new_bids:
            if price < 0.0 or price > 1.0:
                return "price_out_of_bounds"
        for price in new_asks:
            if price < 0.0 or price > 1.0:
                return "price_out_of_bounds"
        if bid > 1.0 or ask > 1.0 or bid < 0.0 or ask < 0.0:
            return "price_out_of_bounds"

        # Crossed book: bid >= ask is corrupt unless one side is empty.
        if bid > 0 and ask > 0 and bid >= ask:
            return "crossed_book"

        # Level ordering / dedup is implicit when we use a dict keyed
        # by price (duplicates collapse, ordering is recovered on
        # serialization).  Skipping the explicit checks here saves a
        # full level walk — the dict semantics already enforce it.
        # If raw incoming JSON has dups, the LAST wins (acceptable —
        # exchanges don't emit dups in practice).

        # Stale snapshot / clock skew.
        wall = time.time()
        age = wall - ingest_ts
        if age > _STALE_SECONDS_THRESHOLD:
            return "stale_snapshot"
        if age < -_CLOCK_SKEW_TOLERANCE_SECONDS:
            return "clock_skew"

        # Sequence regression (per-token monotonicity).
        if sequence is not None:
            try:
                seq_int = int(sequence)
            except (TypeError, ValueError):
                seq_int = 0
            if seq_int > 0:
                if state.last_sequence is not None:
                    if seq_int < state.last_sequence:
                        return "sequence_regression"
                    if seq_int > state.last_sequence + 1:
                        self._sequence_gaps += 1
                state.last_sequence = seq_int

        return None

    # ── Level parsing helpers ───────────────────────────────────────────

    def _levels_to_map(self, order_book: Any, side_name: str) -> dict[float, float]:
        """Walk the raw level list ONCE and return a price→size dict.
        Keys rounded to 4dp so equality comparisons are exact.
        """
        raw = getattr(order_book, side_name, None)
        if raw is None and isinstance(order_book, dict):
            raw = order_book.get(side_name)
        result: dict[float, float] = {}
        for level in list(raw or [])[:_MAX_LEVELS_PER_SIDE]:
            if isinstance(level, dict):
                price = _coerce_float(level.get("price"), 0.0)
                size = _coerce_float(level.get("size"), 0.0)
            else:
                price = _coerce_float(getattr(level, "price", 0.0), 0.0)
                size = _coerce_float(getattr(level, "size", 0.0), 0.0)
            if price > 0 and size > 0:
                result[round(price, 4)] = size
        return result

    @staticmethod
    def _levels_dict_to_list(
        levels: dict[float, float], *, descending: bool
    ) -> list[dict[str, float]]:
        """Serialize a price-keyed map back into the canonical
        ``[{"price": p, "size": s}, ...]`` shape that matches what
        downstream code (BookReplay, fill simulator) expects.
        """
        ordered = sorted(levels.items(), key=lambda kv: kv[0], reverse=descending)
        return [{"price": p, "size": s} for p, s in ordered]

    # ── Queue management ────────────────────────────────────────────────

    def _enqueue_snapshot(self, row: BookSnapshotRow) -> None:
        queue = self._snapshot_queue
        if queue is None:
            return
        if self._should_shed_persistence(queue):
            self._snapshot_dropped += 1
            self._publish_queue_pressure("snapshot")
            return
        try:
            queue.put_nowait(row)
        except asyncio.QueueFull:
            self._snapshot_dropped += 1
            self._publish_queue_pressure("snapshot")
            if self._snapshot_dropped % 1000 == 1:
                logger.warning(
                    "LiveMarketDataIngestor snapshot queue full",
                    dropped=self._snapshot_dropped,
                )

    def _enqueue_delta(self, row: BookDeltaRow) -> None:
        queue = self._delta_queue
        if queue is None:
            return
        if self._should_shed_persistence(queue):
            self._delta_dropped += 1
            self._publish_queue_pressure("delta")
            return
        try:
            queue.put_nowait(row)
        except asyncio.QueueFull:
            self._delta_dropped += 1
            self._publish_queue_pressure("delta")
            if self._delta_dropped % 1000 == 1:
                logger.warning(
                    "LiveMarketDataIngestor delta queue full",
                    dropped=self._delta_dropped,
                )

    def _should_shed_persistence(self, queue: asyncio.Queue) -> bool:
        now_mono = time.monotonic()
        if now_mono < self._pressure_shed_until_mono:
            return True
        try:
            maxsize = int(queue.maxsize or 0)
            if maxsize > 0 and queue.qsize() >= int(maxsize * _QUEUE_HIGH_WATER_FRACTION):
                self._pressure_shed_until_mono = max(
                    self._pressure_shed_until_mono,
                    now_mono + _PRESSURE_SHED_CACHE_SECONDS,
                )
                return True
        except Exception:
            return False
        if now_mono - self._last_pressure_probe_mono < _PRESSURE_PROBE_INTERVAL_SECONDS:
            return False
        self._last_pressure_probe_mono = now_mono
        if is_db_pressure_active() or current_backpressure_level() >= _PRESSURE_SHED_LEVEL:
            self._pressure_shed_until_mono = now_mono + _PRESSURE_SHED_CACHE_SECONDS
            return True
        return False

    def _publish_queue_pressure(self, kind: str) -> None:
        publish_backpressure(
            "market_data_ingestor",
            level=0.85,
            reason=f"{kind}_persistence_shed",
        )
        now_mono = time.monotonic()
        if now_mono - self._last_pressure_log_mono < 30.0:
            return
        self._last_pressure_log_mono = now_mono
        logger.warning(
            "LiveMarketDataIngestor shedding persistence under pressure",
            kind=kind,
            snapshot_dropped=self._snapshot_dropped,
            delta_dropped=self._delta_dropped,
            backpressure_level=current_backpressure_level(),
            db_pressure_active=is_db_pressure_active(),
        )

    # ── Persistence ─────────────────────────────────────────────────────

    async def _flush_loop(
        self,
        *,
        queue: asyncio.Queue,
        batch: int,
        kind: str,
    ) -> None:
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
            try:
                await self._flush_batch(queue=queue, batch=batch, kind=kind)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    f"LiveMarketDataIngestor flush ({kind}) iteration failed",
                    exc_info=exc,
                )

    async def _flush_batch(
        self,
        *,
        queue: asyncio.Queue,
        batch: int,
        kind: str,
    ) -> None:
        rows: list[Any] = []
        # Global recording master switch.  When recording is turned off, drain
        # the queue without persisting so nothing lands on disk and the queue
        # can't grow unbounded.  Read via a short-TTL cache (no per-batch DB
        # hit); the toggle takes effect within a few seconds, no restart.
        try:
            from services.recording_control import is_recording_enabled

            if not await is_recording_enabled():
                drained = 0
                while not queue.empty():
                    queue.get_nowait()
                    drained += 1
                if drained:
                    self._recording_disabled_dropped += drained
                return
        except Exception:  # pragma: no cover — never let the switch break flush
            pass
        if self._should_shed_persistence(queue):
            dropped = 0
            while dropped < batch and not queue.empty():
                queue.get_nowait()
                dropped += 1
            if dropped:
                if kind == "snapshot":
                    self._snapshot_dropped += dropped
                elif kind == "delta":
                    self._delta_dropped += dropped
                self._publish_queue_pressure(kind)
            return
        while len(rows) < batch and not queue.empty():
            rows.append(queue.get_nowait())
        if not rows:
            return
        t0 = time.perf_counter()
        # Shield the session work so a watchdog-driven cancel cannot
        # interrupt mid-asyncpg-protocol — that path leaves the
        # connection wedged ("cannot switch to state 12; another
        # operation (N) is in progress") and poisons later flushes.
        # The async-with itself handles rollback/invalidate on error;
        # do not call session.rollback() here, it races the close()
        # machinery (see trader_hot_state._audit_run_group_inner).
        # Persist OFF Postgres: hand the batch to the columnar parquet
        # sink, which buffers synchronously here and encodes + writes the
        # parquet on its OWN loop via ``to_thread``.  This removes ALL book
        # DB writes from the trading process — eliminating the hot-path DB
        # write pressure — while the unified ``MarketDataView`` reads the
        # result natively.  The live trader reads book state from the WS feed
        # cache + the ingestor's in-memory ``_states``, NOT this
        # persistence layer, so a write hiccup here can never affect it.
        sink = self._book_sink
        if sink is None:
            return
        try:
            sink.write(rows, kind=kind)
        except Exception as exc:
            if kind == "snapshot":
                self._snapshot_dropped += len(rows)
            elif kind == "delta":
                self._delta_dropped += len(rows)
            logger.warning(
                f"LiveMarketDataIngestor: book sink write failed for {kind} batch",
                exc_info=exc,
                rows=len(rows),
            )
            return
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Bounded sliding window of last 100 latencies for p50/p95 stats.
        self._flush_latency_samples.append(latency_ms)
        if len(self._flush_latency_samples) > 100:
            self._flush_latency_samples = self._flush_latency_samples[-100:]
        if latency_ms >= _SLOW_FLUSH_BACKPRESSURE_MS:
            publish_backpressure(
                "market_data_ingestor",
                level=min(1.0, latency_ms / 5000.0),
                reason=f"{kind}_flush_slow:{int(latency_ms)}ms",
            )

    async def _drain(self, queue: asyncio.Queue | None, *, kind: str) -> None:
        if queue is None:
            return
        while not queue.empty():
            await self._flush_batch(queue=queue, batch=_DELTA_FLUSH_BATCH, kind=kind)

    # ── Stats ───────────────────────────────────────────────────────────

    def get_data_quality_stats(self) -> dict[str, Any]:
        """Cheap call.  Returns in-memory counters for the UI."""
        total_attempts = self._accepted_books + sum(self._reject_counts.values())
        accept_rate = (
            self._accepted_books / total_attempts if total_attempts > 0 else None
        )
        samples = sorted(self._flush_latency_samples)
        p50 = samples[len(samples) // 2] if samples else None
        p95 = samples[int(len(samples) * 0.95)] if samples else None
        return {
            "accepted_books": self._accepted_books,
            "total_attempts": total_attempts,
            "accept_rate": accept_rate,
            "rejects_by_reason": dict(self._reject_counts),
            "sequence_gaps_observed": self._sequence_gaps,
            "tokens_tracked": len(self._states),
            "snapshot_queue_dropped": self._snapshot_dropped,
            "delta_queue_dropped": self._delta_dropped,
            # For backwards compat with the prior recorder API; the UI
            # rendered ``queue_dropped`` on a single counter.  Aggregate
            # of both queues now.
            "queue_dropped": self._snapshot_dropped + self._delta_dropped,
            # Rows discarded because the global recording master switch is OFF.
            "recording_disabled_dropped": self._recording_disabled_dropped,
            "flush_latency_ms_p50": p50,
            "flush_latency_ms_p95": p95,
        }


# ── Singleton accessor ──────────────────────────────────────────────────

_ingestor = LiveMarketDataIngestor()


def get_market_data_ingestor() -> LiveMarketDataIngestor:
    """Module-level singleton.  One instance per process."""
    return _ingestor


__all__ = [
    "LiveMarketDataIngestor",
    "get_market_data_ingestor",
]
