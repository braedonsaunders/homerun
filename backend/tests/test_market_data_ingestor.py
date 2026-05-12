"""Tests for the unified ``LiveMarketDataIngestor``.

Covers the hot-path contract (sync, no awaits, microsecond-scale) and
the persistence pipeline (queues, throttling, validation gates).  No
DB writes — we patch the AsyncSessionLocal to avoid touching real
Postgres in the test env.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest

from services.market_data_ingestor import (
    LiveMarketDataIngestor,
    _MAX_LEVELS_PER_SIDE,
)


@pytest.fixture(autouse=True)
def _disable_global_pressure(monkeypatch):
    monkeypatch.setattr("services.market_data_ingestor.current_backpressure_level", lambda: 0.0)
    monkeypatch.setattr("services.market_data_ingestor.is_db_pressure_active", lambda: False)
    monkeypatch.setattr("services.market_data_ingestor.maybe_mark_db_pressure", lambda *args, **kwargs: False)
    monkeypatch.setattr("services.market_data_ingestor.publish_backpressure", lambda *args, **kwargs: None)


# Synthetic order book object: supports both attr access and
# dict-style (the ingestor handles both).
class _Book:
    def __init__(self, bids, asks):
        self.bids = bids
        self.asks = asks


def _level(price, size):
    return {"price": price, "size": size}


def _fresh_ingestor() -> LiveMarketDataIngestor:
    """Construct a fresh ingestor; explicitly do NOT call start()
    because tests run sync hot-path methods only.  Without start(),
    the queues are None and the persistence path is a no-op.
    """
    return LiveMarketDataIngestor()


def _ingestor_with_queues() -> LiveMarketDataIngestor:
    """Construct an ingestor and manually create queues so the hot
    path can enqueue without spawning the flush task.  Tests inspect
    the queues directly to assert what would have been persisted.
    """
    ing = LiveMarketDataIngestor()
    ing._snapshot_queue = asyncio.Queue(maxsize=10000)
    ing._delta_queue = asyncio.Queue(maxsize=10000)
    return ing


# ── Hot-path contract: must be sync, must not block ────────────────────


def test_record_book_is_sync_and_fast():
    ing = _ingestor_with_queues()
    book = _Book(
        bids=[_level(0.50, 100), _level(0.49, 50)],
        asks=[_level(0.51, 80), _level(0.52, 40)],
    )
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        ing.record_book(
            token_id="tok",
            order_book=book,
            best_bid=0.50,
            best_ask=0.51,
            ingest_ts=time.time(),
            sequence=None,
        )
    elapsed_ns = time.perf_counter_ns() - t0
    # 1000 hot-path calls should complete in well under 100ms.  This is
    # the financial-institution-grade contract: the hot path cannot
    # compete with the sub-second orchestrator loop.
    assert elapsed_ns < 100_000_000, f"hot path too slow: {elapsed_ns/1e6:.2f}ms for 1000 calls"


def test_record_book_no_db_until_flush():
    """Hot path must not touch the DB.  We patch AsyncSessionLocal
    to a sentinel that raises if called — the test is that record_book
    completes without raising.
    """
    ing = _ingestor_with_queues()
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    with patch("services.market_data_ingestor.AsyncSessionLocal") as session_factory:
        session_factory.side_effect = RuntimeError("DB touched on hot path!")
        # If record_book awaits the session for any reason, this
        # raises and the test fails.
        ing.record_book(
            token_id="tok",
            order_book=book,
            best_bid=0.50,
            best_ask=0.51,
            ingest_ts=time.time(),
        )


# ── Snapshot throttling ────────────────────────────────────────────────


def test_snapshot_throttle_one_per_window():
    ing = _ingestor_with_queues()
    ing._snapshot_throttle_seconds = 0.5
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    base = time.time()
    # Three calls in rapid succession (well under throttle window).
    for delta in (0.0, 0.1, 0.2):
        ing.record_book(
            token_id="tok",
            order_book=book,
            best_bid=0.50,
            best_ask=0.51,
            ingest_ts=base + delta,
        )
    # Only the FIRST should have been enqueued as a snapshot; the
    # subsequent two are below the 0.5s throttle.
    assert ing._snapshot_queue.qsize() == 1


def test_snapshot_throttle_resets_after_window():
    ing = _ingestor_with_queues()
    ing._snapshot_throttle_seconds = 0.5
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    base = time.time()
    ing.record_book(token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51, ingest_ts=base)
    # Past the window — should be a fresh snapshot write.
    ing.record_book(token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51, ingest_ts=base + 0.6)
    assert ing._snapshot_queue.qsize() == 2


def test_snapshot_throttle_independent_per_token():
    ing = _ingestor_with_queues()
    ing._snapshot_throttle_seconds = 0.5
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    now = time.time()
    # Two different tokens — both should produce a snapshot.
    ing.record_book(token_id="t1", order_book=book, best_bid=0.50, best_ask=0.51, ingest_ts=now)
    ing.record_book(token_id="t2", order_book=book, best_bid=0.50, best_ask=0.51, ingest_ts=now)
    assert ing._snapshot_queue.qsize() == 2


# ── Validation rejects ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bid,ask,bids,asks,reason",
    [
        # Out of [0, 1] range — corrupt.
        (1.5, 1.6, [_level(1.5, 10)], [_level(1.6, 10)], "price_out_of_bounds"),
        # Crossed book.
        (0.60, 0.50, [_level(0.60, 10)], [_level(0.50, 10)], "crossed_book"),
    ],
)
def test_validation_rejects_with_reason(bid, ask, bids, asks, reason):
    ing = _ingestor_with_queues()
    book = _Book(bids=bids, asks=asks)
    ing.record_book(
        token_id="tok",
        order_book=book,
        best_bid=bid,
        best_ask=ask,
        ingest_ts=time.time(),
    )
    stats = ing.get_data_quality_stats()
    assert stats["rejects_by_reason"][reason] == 1
    assert stats["accepted_books"] == 0
    # Nothing enqueued for a rejected book.
    assert ing._snapshot_queue.qsize() == 0
    assert ing._delta_queue.qsize() == 0


def test_validation_rejects_stale_snapshot():
    ing = _ingestor_with_queues()
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    # 60s in the past — beyond the 30s stale threshold.
    ing.record_book(
        token_id="tok",
        order_book=book,
        best_bid=0.50,
        best_ask=0.51,
        ingest_ts=time.time() - 60,
    )
    stats = ing.get_data_quality_stats()
    assert stats["rejects_by_reason"]["stale_snapshot"] == 1


def test_sequence_regression_rejected():
    ing = _ingestor_with_queues()
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    now = time.time()
    ing.record_book(token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51,
                    ingest_ts=now, sequence=5)
    ing.record_book(token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51,
                    ingest_ts=now + 0.1, sequence=3)  # regression
    stats = ing.get_data_quality_stats()
    assert stats["rejects_by_reason"]["sequence_regression"] == 1


# ── Delta classification ───────────────────────────────────────────────


def test_first_book_emits_no_deltas():
    """First book on a new token cannot emit deltas because there's
    no prior state to diff against.  Without this guard, startup
    would emit 50 spurious 'cancel' events as the recorder learned
    the initial book.
    """
    ing = _ingestor_with_queues()
    book = _Book(
        bids=[_level(0.50, 100), _level(0.49, 80)],
        asks=[_level(0.51, 60), _level(0.52, 40)],
    )
    ing.record_book(
        token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51,
        ingest_ts=time.time(),
    )
    assert ing._delta_queue.qsize() == 0


def test_depth_decrease_classified_as_cancel_without_trade():
    ing = _ingestor_with_queues()
    book1 = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    book2 = _Book(bids=[_level(0.50, 50)], asks=[_level(0.51, 80)])  # bid shrunk
    base = time.time()
    ing.record_book(token_id="tok", order_book=book1, best_bid=0.50, best_ask=0.51,
                    ingest_ts=base)
    ing.record_book(token_id="tok", order_book=book2, best_bid=0.50, best_ask=0.51,
                    ingest_ts=base + 1.0)
    # No matching trade was seen → classified as cancel.
    assert ing._delta_queue.qsize() == 1
    delta = ing._delta_queue.get_nowait()
    assert delta.event_type == "cancel"
    assert delta.cancel_size == pytest.approx(50.0)


def test_depth_decrease_classified_as_trade_with_matching_print():
    ing = _ingestor_with_queues()
    book1 = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    book2 = _Book(bids=[_level(0.50, 60)], asks=[_level(0.51, 80)])  # 40 disappeared
    # Record a trade at the same price BEFORE the post-trade book.
    ing.record_trade(
        token_id="tok",
        trade={"price": 0.50, "size": 40, "side": "BUY", "timestamp": time.time()},
    )
    base = time.time()
    ing.record_book(token_id="tok", order_book=book1, best_bid=0.50, best_ask=0.51,
                    ingest_ts=base)
    ing.record_book(token_id="tok", order_book=book2, best_bid=0.50, best_ask=0.51,
                    ingest_ts=base + 0.1)  # within trade match window
    # The depth delta of 40 should match the 40-share trade → trade event.
    deltas = []
    while not ing._delta_queue.empty():
        deltas.append(ing._delta_queue.get_nowait())
    types = {d.event_type for d in deltas}
    # At least one trade event for the matched 40 shares.
    assert "trade" in types


# ── Level walking ─────────────────────────────────────────────────────


def test_levels_clamped_to_max_per_side():
    ing = _ingestor_with_queues()
    # 50 levels — should be clamped to _MAX_LEVELS_PER_SIDE.
    book = _Book(
        bids=[_level(0.50 - 0.001 * i, 10) for i in range(50)],
        asks=[_level(0.51 + 0.001 * i, 10) for i in range(50)],
    )
    ing.record_book(token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51,
                    ingest_ts=time.time())
    snap = ing._snapshot_queue.get_nowait()
    assert len(snap.bids_json) == _MAX_LEVELS_PER_SIDE
    assert len(snap.asks_json) == _MAX_LEVELS_PER_SIDE


def test_levels_dict_serializes_descending_bids_ascending_asks():
    ing = _ingestor_with_queues()
    book = _Book(
        bids=[_level(0.49, 10), _level(0.50, 20), _level(0.48, 30)],  # unordered
        asks=[_level(0.52, 10), _level(0.51, 20), _level(0.53, 30)],  # unordered
    )
    ing.record_book(token_id="tok", order_book=book, best_bid=0.50, best_ask=0.51,
                    ingest_ts=time.time())
    snap = ing._snapshot_queue.get_nowait()
    bid_prices = [b["price"] for b in snap.bids_json]
    ask_prices = [a["price"] for a in snap.asks_json]
    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ask_prices == sorted(ask_prices)


# ── Trade record ──────────────────────────────────────────────────────


def test_record_trade_enqueues_trade_snapshot():
    ing = _ingestor_with_queues()
    ing.record_trade(
        token_id="tok",
        trade={"price": 0.42, "size": 10, "side": "BUY", "timestamp": time.time()},
    )
    assert ing._snapshot_queue.qsize() == 1
    snap = ing._snapshot_queue.get_nowait()
    assert snap.snapshot_type == "trade"
    assert snap.trade_price == 0.42
    assert snap.trade_size == 10


def test_trade_buffer_bounded():
    ing = _ingestor_with_queues()
    # Stuff way more than _TRADE_BUFFER_PER_TOKEN trades in.
    for i in range(500):
        ing.record_trade(
            token_id="tok",
            trade={"price": 0.50, "size": 1, "side": "BUY", "timestamp": time.time()},
        )
    state = ing._states["tok"]
    assert len(state.recent_trades) <= 64  # _TRADE_BUFFER_PER_TOKEN


# ── Stats surface ─────────────────────────────────────────────────────


def test_record_trade_sheds_persistence_under_db_pressure(monkeypatch):
    ing = _ingestor_with_queues()
    monkeypatch.setattr("services.market_data_ingestor.is_db_pressure_active", lambda: True)

    ing.record_trade(
        token_id="tok",
        trade={"price": 0.42, "size": 10, "side": "BUY", "timestamp": time.time()},
    )

    stats = ing.get_data_quality_stats()
    assert ing._snapshot_queue.qsize() == 0
    assert stats["snapshot_queue_dropped"] == 1


def test_stats_after_mixed_traffic():
    ing = _ingestor_with_queues()
    book = _Book(bids=[_level(0.50, 100)], asks=[_level(0.51, 80)])
    ing.record_book(token_id="t1", order_book=book, best_bid=0.50, best_ask=0.51,
                    ingest_ts=time.time())
    ing.record_book(token_id="t2", order_book=book, best_bid=0.50, best_ask=0.51,
                    ingest_ts=time.time())
    # One reject.
    ing.record_book(token_id="t3", order_book=_Book(bids=[_level(1.5, 10)], asks=[_level(1.6, 10)]),
                    best_bid=1.5, best_ask=1.6, ingest_ts=time.time())
    stats = ing.get_data_quality_stats()
    assert stats["accepted_books"] == 2
    assert stats["rejects_by_reason"]["price_out_of_bounds"] == 1
    assert stats["accept_rate"] == pytest.approx(2 / 3)
    assert stats["tokens_tracked"] == 3  # t1, t2, t3 all created state entries


@pytest.mark.asyncio
async def test_flush_batch_drops_rows_after_commit_timeout(monkeypatch):
    ing = _ingestor_with_queues()
    row = object()
    ing._snapshot_queue.put_nowait(row)

    class _Session:
        def add_all(self, rows):
            self.rows = rows

        async def commit(self):
            raise TimeoutError("simulated commit timeout")

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("services.market_data_ingestor.AsyncSessionLocal", lambda: _SessionContext())

    await ing._flush_batch(queue=ing._snapshot_queue, batch=10, kind="snapshot")

    stats = ing.get_data_quality_stats()
    assert ing._snapshot_queue.qsize() == 0
    assert stats["snapshot_queue_dropped"] == 1
