"""Tests for BaseStrategy.on_timeframe_close().

The hook fires from the default ``on_event`` whenever a wall-clock
boundary for an opted-in timeframe was crossed since the strategy last
saw a MARKET_DATA_REFRESH event. Boundaries are unix-epoch aligned so
multiple workers / restarts agree on when a candle closed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from services.data_events import DataEvent, EventType
from services.strategies.base import (
    BaseStrategy,
    _last_boundary_at_or_before,
    _parse_timeframe_close_interval,
)


# ---------------------------------------------------------------------------
# Boundary math
# ---------------------------------------------------------------------------


def test_parse_canonical_labels():
    assert _parse_timeframe_close_interval("5m") == ("5m", 300)
    assert _parse_timeframe_close_interval("15m") == ("15m", 900)
    assert _parse_timeframe_close_interval("1h") == ("1h", 3600)
    assert _parse_timeframe_close_interval("4h") == ("4h", 14_400)


def test_parse_aliases_and_case_insensitive():
    assert _parse_timeframe_close_interval("5MIN") == ("5m", 300)
    assert _parse_timeframe_close_interval("1HR") == ("1h", 3600)
    assert _parse_timeframe_close_interval("60m") == ("1h", 3600)
    assert _parse_timeframe_close_interval("240m") == ("4h", 14_400)


def test_parse_unknown_returns_none():
    assert _parse_timeframe_close_interval("3m") == (None, 0)
    assert _parse_timeframe_close_interval("") == (None, 0)
    assert _parse_timeframe_close_interval(None) == (None, 0)


def test_boundary_floor_is_epoch_aligned():
    now = datetime(2026, 4, 30, 14, 32, 15, tzinfo=timezone.utc)
    assert _last_boundary_at_or_before(now, 300) == datetime(2026, 4, 30, 14, 30, tzinfo=timezone.utc)
    assert _last_boundary_at_or_before(now, 900) == datetime(2026, 4, 30, 14, 30, tzinfo=timezone.utc)
    assert _last_boundary_at_or_before(now, 3600) == datetime(2026, 4, 30, 14, 0, tzinfo=timezone.utc)
    assert _last_boundary_at_or_before(now, 14_400) == datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)


def test_boundary_naive_input_is_treated_as_utc():
    naive = datetime(2026, 4, 30, 14, 32, 15)
    out = _last_boundary_at_or_before(naive, 300)
    assert out.tzinfo == timezone.utc
    assert out == datetime(2026, 4, 30, 14, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# on_event integration — boundary detection
# ---------------------------------------------------------------------------


class _RecordingStrategy(BaseStrategy):
    """Test strategy that records every on_timeframe_close call."""

    strategy_type = "test_recording"
    name = "Test Recording Strategy"
    description = "Captures timeframe close events for tests"
    timeframe_close_intervals = ["5m", "1h"]

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, datetime]] = []

    def detect(self, events, markets, prices):
        return []

    def on_timeframe_close(self, timeframe, boundary_ts, events, markets, prices):
        self.calls.append((timeframe, boundary_ts))
        return []


def _market_refresh(ts: datetime) -> DataEvent:
    return DataEvent(
        event_type=EventType.MARKET_DATA_REFRESH,
        source="test",
        timestamp=ts,
        markets=[],
        events=[],
        prices={},
    )


def test_first_refresh_does_not_emit_close():
    """First refresh has no prior reference — we don't synthesise a
    close event from process start."""
    strat = _RecordingStrategy()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 0, 30, tzinfo=timezone.utc))))
    assert strat.calls == []


def test_close_fires_when_boundary_crossed():
    strat = _RecordingStrategy()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 30, tzinfo=timezone.utc))))
    # 14:04:30 — most recent 5m boundary is 14:00:00, most recent 1h is 14:00:00
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    # 14:05:30 — 5m boundary 14:05:00 just crossed; 1h boundary still 14:00:00 (unchanged)
    assert strat.calls == [("5m", datetime(2026, 4, 30, 14, 5, tzinfo=timezone.utc))]


def test_close_fires_only_once_per_boundary():
    """Even if the scanner ticks every 30s, one boundary -> one call."""
    strat = _RecordingStrategy()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 30, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 6, 0, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 7, 30, tzinfo=timezone.utc))))
    # Only one 5m close has happened (at 14:05:00) — the rest are intra-window
    assert strat.calls == [("5m", datetime(2026, 4, 30, 14, 5, tzinfo=timezone.utc))]


def test_skipped_cadence_still_emits_one_close_per_crossing():
    """If the scanner skipped a few cycles and we span multiple boundaries,
    each boundary should produce exactly one call (most recent wins)."""
    strat = _RecordingStrategy()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 0, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 17, 0, tzinfo=timezone.utc))))
    # 5m boundaries crossed: 14:05, 14:10, 14:15 — current contract emits the
    # most recent one (we don't replay old closes, just announce the latest).
    assert strat.calls == [("5m", datetime(2026, 4, 30, 14, 15, tzinfo=timezone.utc))]


def test_multiple_timeframes_independent():
    strat = _RecordingStrategy()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 13, 59, 0, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 0, 30, tzinfo=timezone.utc))))
    # Both 5m (14:00:00) and 1h (14:00:00) crossed simultaneously
    assert ("5m", datetime(2026, 4, 30, 14, 0, tzinfo=timezone.utc)) in strat.calls
    assert ("1h", datetime(2026, 4, 30, 14, 0, tzinfo=timezone.utc)) in strat.calls


def test_strategy_without_opt_in_never_calls_hook():
    class _NoOpt(_RecordingStrategy):
        timeframe_close_intervals = []

    strat = _NoOpt()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 0, 30, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    assert strat.calls == []


def test_async_on_timeframe_close_is_awaited():
    """on_timeframe_close may be async; default plumbing awaits it."""
    captured: list[tuple[str, datetime]] = []

    class _AsyncStrat(BaseStrategy):
        strategy_type = "test_async_close"
        name = "Test Async Close"
        description = "Async timeframe-close hook"
        timeframe_close_intervals = ["5m"]

        def detect(self, events, markets, prices):
            return []

        async def on_timeframe_close(self, timeframe, boundary_ts, events, markets, prices):
            await asyncio.sleep(0)
            captured.append((timeframe, boundary_ts))
            return []

    strat = _AsyncStrat()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 30, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    assert captured == [("5m", datetime(2026, 4, 30, 14, 5, tzinfo=timezone.utc))]


def test_hook_exceptions_do_not_break_detect():
    """Exceptions inside on_timeframe_close are swallowed so the rest of
    the dispatch pipeline continues. detect()'s opportunities still flow."""

    class _BoomStrat(BaseStrategy):
        strategy_type = "test_boom_close"
        name = "Test Boom Close"
        description = "Raises in close hook"
        timeframe_close_intervals = ["5m"]

        def detect(self, events, markets, prices):
            return ["sentinel"]  # type: ignore[return-value]

        def on_timeframe_close(self, timeframe, boundary_ts, events, markets, prices):
            raise RuntimeError("boom")

    strat = _BoomStrat()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 30, tzinfo=timezone.utc))))
    out = asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    assert out == ["sentinel"]


def test_close_hook_opportunities_merge_with_detect_results():
    class _MergeStrat(BaseStrategy):
        strategy_type = "test_merge_close"
        name = "Test Merge Close"
        description = "Both detect and close emit"
        timeframe_close_intervals = ["5m"]

        def detect(self, events, markets, prices):
            return ["from-detect"]  # type: ignore[return-value]

        def on_timeframe_close(self, timeframe, boundary_ts, events, markets, prices):
            return ["from-close"]  # type: ignore[return-value]

    strat = _MergeStrat()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 30, tzinfo=timezone.utc))))
    out = asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    assert out == ["from-detect", "from-close"]


def test_unknown_timeframe_label_is_skipped_silently():
    class _BadCfg(_RecordingStrategy):
        timeframe_close_intervals = ["5m", "3m"]  # 3m not supported

    strat = _BadCfg()
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 4, 30, tzinfo=timezone.utc))))
    asyncio.run(strat.on_event(_market_refresh(datetime(2026, 4, 30, 14, 5, 30, tzinfo=timezone.utc))))
    # Bad label silently skipped, valid 5m still fires
    assert strat.calls == [("5m", datetime(2026, 4, 30, 14, 5, tzinfo=timezone.utc))]
