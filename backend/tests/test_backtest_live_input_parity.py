"""Parity + no-silent-failure guards for the backtester's data plane.

Two fidelity invariants this suite pins:

1.  **Live/backtest input parity.** The live scanner feeds strategies a
    per-token price dict keyed
    ``{mid, bid, ask, ts, ingest_ts, exchange_ts, sequence, is_fresh}``
    (``scanner._snapshot_ws_prices``). The backtester's per-tick price grid
    (``strategy_backtester._build_per_tick_prices_grid``) must emit a
    *superset* of that shape — every live key PLUS the microstructure extras
    (``best_bid``/``best_ask``/``imbalance``/depth) — so a strategy written
    against the live dict reads real values in backtest, not ``None``.

2.  **No silent run on missing data.** The unified runner resolves parquet
    coverage for the run's token universe over ``[start, end]`` and, when a
    window has no recorded book data, surfaces a PROMINENT "NO DATA"
    validation_warning instead of silently producing zero fills.

Self-contained and fast: synthetic ``BookSnapshot``s + a monkeypatched
``MarketDataView`` (no parquet/DB), and a monkeypatched coverage resolver.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import services.strategy_backtester as strategy_backtester
from services.backtest.book_replay import BookSnapshot, PriceLevel

# The exact per-token key set the LIVE scanner emits
# (scanner._snapshot_ws_prices, ~line 845). The backtest grid must be a
# superset of these.
LIVE_PRICE_KEYS = {
    "mid",
    "bid",
    "ask",
    "ts",
    "ingest_ts",
    "exchange_ts",
    "sequence",
    "is_fresh",
}

# Microstructure extras the backtest carries on top of the live shape — a
# strategy that wants book pressure can read them, and they must not have
# displaced any live key.
MICROSTRUCTURE_EXTRA_KEYS = {"best_bid", "best_ask", "imbalance"}


def _make_snapshot(
    token_id: str,
    observed_at: datetime,
    *,
    best_bid: float,
    best_ask: float,
    sequence: int | None,
    bid_size: float = 100.0,
    ask_size: float = 60.0,
) -> BookSnapshot:
    """A minimal two-level synthetic book with eager PriceLevel ladders so
    ``best_bid``/``best_ask``/``depth`` resolve without parquet."""
    return BookSnapshot(
        token_id=token_id,
        observed_at=observed_at,
        bids=(
            PriceLevel(price=best_bid, size=bid_size),
            PriceLevel(price=round(best_bid - 0.01, 4), size=bid_size),
        ),
        asks=(
            PriceLevel(price=best_ask, size=ask_size),
            PriceLevel(price=round(best_ask + 0.01, 4), size=ask_size),
        ),
        sequence=sequence,
        spread_bps=(best_ask - best_bid) * 10000.0,
    )


class _FakeReplaySource:
    """Stands in for ``MarketDataViewSource`` — yields a fixed snapshot list
    in observed_at order, matching the real source's ``iter_snapshots`` shape.
    """

    def __init__(self, snaps: list[BookSnapshot]) -> None:
        self._snaps = snaps

    async def iter_snapshots(self):
        for s in self._snaps:
            yield s


def _patch_marketdata(monkeypatch, snaps: list[BookSnapshot]) -> None:
    """Patch the two symbols ``_build_per_tick_prices_grid`` imports so the
    grid builds against in-memory snapshots instead of parquet/DB."""
    import services.marketdata.view as view_mod
    import services.marketdata.book_source as source_mod

    class _FakeView:
        @classmethod
        async def build(cls, **_kwargs: Any) -> "_FakeView":
            return cls()

    def _fake_source(_view: Any, *, token_ids: Any = None) -> _FakeReplaySource:
        return _FakeReplaySource(snaps)

    monkeypatch.setattr(view_mod, "MarketDataView", _FakeView, raising=True)
    monkeypatch.setattr(
        source_mod, "MarketDataViewSource", _fake_source, raising=True
    )


@pytest.mark.asyncio
async def test_per_tick_price_grid_is_live_input_superset(monkeypatch):
    """The per-token price dict the backtest hands strategies must contain
    EVERY live key and the microstructure extras, with sane values."""
    t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    token = "tok_parity"
    # One snapshot observed just before the (single) tick boundary so it
    # freezes into the grid.
    snap = _make_snapshot(
        token,
        t0 + timedelta(seconds=1),
        best_bid=0.42,
        best_ask=0.44,
        sequence=7,
    )
    _patch_marketdata(monkeypatch, [snap])

    ticks = [t0 + timedelta(seconds=2)]
    grid = await strategy_backtester._build_per_tick_prices_grid(
        token_ids=[token],
        ticks=ticks,
        start_dt=t0,
        end_dt=t0 + timedelta(seconds=10),
    )

    assert token in grid, "covered token should appear in the grid"
    state = grid[token][0]
    assert state is not None, "state at-or-before the tick should be frozen"

    # (a) Every LIVE key is present — a strategy written for live works here.
    missing_live = LIVE_PRICE_KEYS - set(state.keys())
    assert not missing_live, f"backtest price dict missing live keys: {missing_live}"

    # (b) The microstructure extras are also present (superset, not a swap).
    missing_extra = MICROSTRUCTURE_EXTRA_KEYS - set(state.keys())
    assert not missing_extra, f"missing microstructure extras: {missing_extra}"

    # Values: live aliases mirror the book; ts is integer epoch seconds.
    assert state["bid"] == pytest.approx(0.42)
    assert state["ask"] == pytest.approx(0.44)
    assert state["best_bid"] == pytest.approx(0.42)
    assert state["best_ask"] == pytest.approx(0.44)
    assert state["mid"] == pytest.approx(0.43)
    assert state["bid"] == state["best_bid"]
    assert state["ask"] == state["best_ask"]

    observed = snap.observed_at
    expected_ts = int(observed.timestamp())
    assert isinstance(state["ts"], int)
    assert state["ts"] == expected_ts
    assert state["ingest_ts"] == expected_ts
    assert state["exchange_ts"] == expected_ts

    assert state["sequence"] == 7
    assert isinstance(state["sequence"], int)
    assert state["is_fresh"] is True

    # imbalance = bid_depth / (bid_depth + ask_depth); 200 / (200 + 120).
    assert state["imbalance"] == pytest.approx(200.0 / 320.0)


@pytest.mark.asyncio
async def test_per_tick_price_grid_sequence_defaults_to_zero(monkeypatch):
    """A snapshot with no sequence (live default is 0) must still produce an
    int ``sequence`` of 0, never ``None`` — matching the live shape."""
    t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    token = "tok_noseq"
    snap = _make_snapshot(
        token,
        t0 + timedelta(seconds=1),
        best_bid=0.30,
        best_ask=0.31,
        sequence=None,
    )
    _patch_marketdata(monkeypatch, [snap])

    grid = await strategy_backtester._build_per_tick_prices_grid(
        token_ids=[token],
        ticks=[t0 + timedelta(seconds=2)],
        start_dt=t0,
        end_dt=t0 + timedelta(seconds=10),
    )
    state = grid[token][0]
    assert state is not None
    assert state["sequence"] == 0
    assert isinstance(state["sequence"], int)


def test_coverage_warning_no_data():
    """Zero coverage → a PROMINENT NO-DATA warning naming the token count."""
    from services.backtest.unified_runner import _coverage_warning

    warn = _coverage_warning(requested=5, covered=0, fraction=0.0)
    assert warn is not None
    assert "NO DATA" in warn
    assert "0/5" in warn
    assert "no fills" in warn


def test_coverage_warning_low_coverage():
    """0 < fraction < 0.5 → a softer LOW-COVERAGE warning with a percentage."""
    from services.backtest.unified_runner import _coverage_warning

    warn = _coverage_warning(requested=10, covered=3, fraction=0.3)
    assert warn is not None
    assert "LOW COVERAGE" in warn
    assert "3/10" in warn
    assert "30%" in warn


def test_coverage_warning_adequate_is_silent():
    """>= 50% coverage → no warning (don't cry wolf on a healthy run)."""
    from services.backtest.unified_runner import _coverage_warning

    assert _coverage_warning(requested=10, covered=8, fraction=0.8) is None
    assert _coverage_warning(requested=10, covered=5, fraction=0.5) is None
    # No requested tokens → nothing to warn about.
    assert _coverage_warning(requested=0, covered=0, fraction=0.0) is None


@pytest.mark.asyncio
async def test_resolve_coverage_summary_zero_coverage_yields_no_data(monkeypatch):
    """A zero-coverage resolution produces both the structured data_coverage
    summary AND the NO-DATA warning — the runner's no-silent-failure guard."""
    import services.backtest.unified_runner as unified_runner
    from services.marketdata.coverage import CoverageMap

    requested = ["tok_a", "tok_b", "tok_c"]

    async def _fake_resolve(*, token_ids, start, end, **_kw) -> CoverageMap:
        # Empty by_token → covered_tokens == () → coverage_fraction == 0.0.
        return CoverageMap(by_token={}, requested=tuple(token_ids))

    monkeypatch.setattr(
        "services.marketdata.coverage.resolve_coverage", _fake_resolve, raising=True
    )

    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    summary, warning = await unified_runner._resolve_coverage_summary(
        token_ids=requested, start=start, end=end
    )

    # Structured summary shape (the UI banner contract).
    assert summary["requested_tokens"] == 3
    assert summary["covered_tokens"] == 0
    assert summary["coverage_fraction"] == 0.0
    assert summary["window_start"] == start.isoformat()
    assert summary["window_end"] == end.isoformat()

    # Prominent NO-DATA warning.
    assert warning is not None
    assert "NO DATA" in warning
    assert "0/3" in warning


@pytest.mark.asyncio
async def test_resolve_coverage_summary_partial_coverage_yields_low(monkeypatch):
    """1/3 covered (< 50%) → LOW COVERAGE warning + correct fraction."""
    import services.backtest.unified_runner as unified_runner
    from services.marketdata.coverage import CoverageMap, TokenCoverage

    requested = ("tok_a", "tok_b", "tok_c")

    async def _fake_resolve(*, token_ids, start, end, **_kw) -> CoverageMap:
        by_token = {
            "tok_a": TokenCoverage(token_id="tok_a", files=("/data/a.parquet",)),
            "tok_b": TokenCoverage(token_id="tok_b"),
            "tok_c": TokenCoverage(token_id="tok_c"),
        }
        return CoverageMap(by_token=by_token, requested=tuple(token_ids))

    monkeypatch.setattr(
        "services.marketdata.coverage.resolve_coverage", _fake_resolve, raising=True
    )

    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    summary, warning = await unified_runner._resolve_coverage_summary(
        token_ids=list(requested), start=start, end=end
    )

    assert summary["requested_tokens"] == 3
    assert summary["covered_tokens"] == 1
    assert summary["coverage_fraction"] == pytest.approx(1.0 / 3.0)
    assert warning is not None
    assert "LOW COVERAGE" in warning


@pytest.mark.asyncio
async def test_resolve_coverage_summary_never_crashes_on_resolver_error(monkeypatch):
    """If the resolver raises, the guard returns a best-effort summary and no
    warning — a backtest must never abort because the diagnostic failed."""
    import services.backtest.unified_runner as unified_runner

    async def _boom(*, token_ids, start, end, **_kw):
        raise RuntimeError("catalog unreachable")

    monkeypatch.setattr(
        "services.marketdata.coverage.resolve_coverage", _boom, raising=True
    )

    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    summary, warning = await unified_runner._resolve_coverage_summary(
        token_ids=["tok_a"], start=start, end=end
    )
    assert summary["requested_tokens"] == 1
    assert warning is None  # degrade gracefully, don't fabricate a warning


def test_recorded_market_hydration_preserves_end_date():
    """Markets leg of the invariant: recorded catalog-snapshot markets are
    ``Market.model_dump()``s (snake_case only).  The backtest must rehydrate them
    with ``model_validate`` — NOT ``from_gamma_response``, which parses the gamma
    API's camelCase and silently DROPS ``end_date`` (so a market-source strategy
    like tail_end_carry sees a null resolution date and emits nothing).
    ``_hydrate_market_model`` routes by shape: snake-only model_dumps ->
    model_validate; camelCase gamma payloads (incl. the projection) ->
    from_gamma_response."""
    from models.market import Market

    live = Market.from_gamma_response({
        "id": "m1",
        "question": "Will it resolve?",
        "conditionId": "0xcond",
        "clobTokenIds": '["tok_yes", "tok_no"]',
        "endDate": "2026-06-30T00:00:00Z",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.6", "0.4"]',
    })
    assert live.end_date is not None
    recorded = live.model_dump(mode="json")  # exactly what the catalog-snapshot tee stores
    assert "clob_token_ids" in recorded and "clobTokenIds" not in recorded  # snake-only

    # The bug this guards: from_gamma_response on a model_dump drops end_date.
    assert Market.from_gamma_response(recorded).end_date is None

    # The fix: the shape-aware hydrator restores it via model_validate.
    rehydrated = strategy_backtester._hydrate_market_model(Market, recorded)
    assert rehydrated is not None
    assert rehydrated.end_date is not None
    assert [str(t) for t in (rehydrated.clob_token_ids or [])] == ["tok_yes", "tok_no"]

    # Genuine gamma (camelCase) payloads still route through from_gamma_response.
    gamma = strategy_backtester._hydrate_market_model(Market, {
        "id": "m2", "question": "Q", "conditionId": "0xc2",
        "clobTokenIds": '["a", "b"]', "endDate": "2026-06-30T00:00:00Z",
    })
    assert gamma is not None and gamma.end_date is not None


@pytest.mark.asyncio
async def test_asof_catalog_first_seen_wins_keeps_active_state(monkeypatch):
    """P1b: book-driven backtests source the universe AS-OF the window from the
    recorded catalog.snapshot.  First-seen-wins per market key keeps each market
    in its in-window ACTIVE state — a market that resolves later in-window must
    NOT collapse to its resolved snapshot (which the closed/resolved filter would
    then drop), else the backtest excludes a market the live scanner could trade
    in-window."""
    import services.recorded_event_bus as reb

    class _Ev:
        def __init__(self, payload):
            self.payload = payload

    async def _fake_replay(window):
        yield _Ev({"markets": [
            {"id": "m1", "clob_token_ids": ["t1"], "closed": False, "end_date": "2026-06-30T00:00:00Z"},
        ], "events": [{"id": "e1"}]})
        yield _Ev({"markets": [
            {"id": "m1", "clob_token_ids": ["t1"], "closed": True, "resolved": True},
            {"id": "m2", "clob_token_ids": ["t2"], "closed": False},
        ], "events": []})

    class _FakeBus:
        def replay(self, window):
            return _fake_replay(window)

    monkeypatch.setattr(reb, "bus", _FakeBus())

    res = await strategy_backtester._load_asof_catalog_markets(
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, 1, tzinfo=timezone.utc),
    )
    assert res is not None
    _events, markets, meta = res
    by_id = {m["id"]: m for m in markets}
    assert set(by_id) == {"m1", "m2"}
    assert by_id["m1"].get("closed") is False, "first-seen-wins must keep m1's active in-window state"
    assert meta["source"] == "recorded_catalog_snapshot_asof"


@pytest.mark.asyncio
async def test_asof_catalog_empty_window_returns_none(monkeypatch):
    """No recorded catalog.snapshot covering the window -> None, so the caller
    falls back to the current catalog file (imported-only data still backtests)."""
    import services.recorded_event_bus as reb

    async def _empty_replay(window):
        return
        yield  # unreachable — makes this an async generator that yields nothing

    class _FakeBus:
        def replay(self, window):
            return _empty_replay(window)

    monkeypatch.setattr(reb, "bus", _FakeBus())
    res = await strategy_backtester._load_asof_catalog_markets(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 2, tzinfo=timezone.utc),
    )
    assert res is None
