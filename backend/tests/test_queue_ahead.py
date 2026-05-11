"""Regression tests for the canonical queue-ahead formula.

This is the single source of truth used by ``ExecutionEstimator``
(live shadow trader) and ``MatchingEngine`` (backtest matcher).  If
the formula changes here, both consumers shift in lockstep — that's
the whole point of having the util.

Tests pin:
  * Symmetry between BUY and SELL sides.
  * Better-priced same-side depth is fully counted.
  * Same-level depth contributes only ``maker_queue_ahead_fraction``
    (default 0.65) — the time-priority approximation.
  * Stale-book ``depth_factor`` scales the whole result.
  * Alias side strings (BUY_YES / BUY_NO) behave like BUY.
"""
from __future__ import annotations

import pytest

from services.optimization.queue_ahead import compute_queue_ahead_shares


# Convenience: a representative book for tests.  Bids descending, asks
# ascending — same convention as BookSnapshot.
_BIDS = [
    {"price": 0.52, "size": 40},
    {"price": 0.51, "size": 80},
    {"price": 0.50, "size": 60},
    {"price": 0.49, "size": 30},
]
_ASKS = [
    {"price": 0.53, "size": 30},
    {"price": 0.54, "size": 50},
    {"price": 0.55, "size": 100},
    {"price": 0.56, "size": 75},
]


def test_buy_at_top_of_book_no_better_queue():
    """BUY at 0.52 with bid level 0.52 size=40 → no better-priced
    same-side, just same-level fraction."""
    q = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY", limit_price=0.52
    )
    # 0 + 40 * 0.65 = 26.0
    assert q == pytest.approx(26.0)


def test_buy_below_top_includes_better_queue():
    """BUY at 0.50 — bids at 0.52 (40) and 0.51 (80) are better-priced
    (higher bids fill against incoming sells first), then same-level
    is 60 * 0.65 = 39."""
    q = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY", limit_price=0.50
    )
    # better = 40 + 80 = 120; same = 60 * 0.65 = 39 → 159
    assert q == pytest.approx(159.0)


def test_sell_at_top_of_book_no_better_queue():
    """SELL at 0.53 → no asks below 0.53; same-level 30 * 0.65."""
    q = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="SELL", limit_price=0.53
    )
    assert q == pytest.approx(30 * 0.65)


def test_sell_above_top_includes_better_queue():
    """SELL at 0.55 — asks at 0.53 (30) and 0.54 (50) are better-priced
    (lower asks fill against incoming buys first), then same-level
    100 * 0.65 = 65."""
    q = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="SELL", limit_price=0.55
    )
    # better = 30 + 50 = 80; same = 100 * 0.65 = 65 → 145
    assert q == pytest.approx(145.0)


def test_no_own_side_depth_at_price_returns_zero():
    """Quote at a fresh price level with nothing on our side → no
    queue, we are first-in-line.  Matches reality: if no one else is
    at this exact price, we get filled immediately when crossed."""
    # SELL at 0.535 — asks have no level at 0.535
    q = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="SELL", limit_price=0.535
    )
    # 0 better (no ask below 0.535 except 0.53 size 30) → wait, 0.53 < 0.535 IS better.
    # better = 30 (the 0.53 level); same = 0 → 30
    assert q == pytest.approx(30.0)


def test_depth_factor_scales_result():
    q_fresh = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY", limit_price=0.50, depth_factor=1.0,
    )
    q_stale = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY", limit_price=0.50, depth_factor=0.5,
    )
    assert q_stale == pytest.approx(q_fresh * 0.5)


def test_buy_yes_buy_no_aliases_behave_like_buy():
    """Strategy SDK uses BUY_YES / BUY_NO for Polymarket binaries; the
    util must treat them as BUY for queue-side determination."""
    q_buy = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY", limit_price=0.50
    )
    q_yes = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY_YES", limit_price=0.50
    )
    q_no = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="BUY_NO", limit_price=0.50
    )
    assert q_yes == pytest.approx(q_buy)
    assert q_no == pytest.approx(q_buy)


def test_maker_queue_ahead_fraction_override():
    """Pass a non-default fraction; verify only same-level is scaled."""
    # SELL at 0.55: better=80, same=100
    q_default = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="SELL", limit_price=0.55,
        maker_queue_ahead_fraction=0.65,
    )  # 80 + 65 = 145
    q_full = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="SELL", limit_price=0.55,
        maker_queue_ahead_fraction=1.0,
    )  # 80 + 100 = 180
    q_half = compute_queue_ahead_shares(
        bids=_BIDS, asks=_ASKS, side="SELL", limit_price=0.55,
        maker_queue_ahead_fraction=0.5,
    )  # 80 + 50 = 130
    assert q_default == pytest.approx(145.0)
    assert q_full == pytest.approx(180.0)
    assert q_half == pytest.approx(130.0)


def test_accepts_pricelevel_dataclass_inputs():
    """The util must accept BookSnapshot's PriceLevel objects (frozen
    dataclasses with .price / .size attrs) — that's what the backtest
    matcher hands it."""
    from services.backtest.book_replay import PriceLevel
    bids = [PriceLevel(price=0.50, size=100.0)]
    asks = [PriceLevel(price=0.51, size=50.0)]
    q = compute_queue_ahead_shares(
        bids=bids, asks=asks, side="BUY", limit_price=0.50
    )
    assert q == pytest.approx(100.0 * 0.65)


def test_empty_book_returns_zero():
    """Defensive: empty bids/asks → no queue ahead."""
    assert compute_queue_ahead_shares(
        bids=[], asks=[], side="BUY", limit_price=0.50
    ) == 0.0
    assert compute_queue_ahead_shares(
        bids=None, asks=None, side="SELL", limit_price=0.50
    ) == 0.0


def test_malformed_levels_skipped_gracefully():
    """Malformed level dicts (missing keys, zero size, etc.) should
    be silently dropped — same permissive behaviour as
    ExecutionEstimator's existing input handling."""
    bids = [
        {"price": 0.50, "size": 100},
        {"price": 0.49},                  # missing size
        {"size": 200},                    # missing price
        {"price": 0, "size": 50},         # zero price
        {"price": 0.48, "size": 0},       # zero size
        {"price": "not-a-number", "size": 50},
    ]
    q = compute_queue_ahead_shares(
        bids=bids, asks=[], side="BUY", limit_price=0.50
    )
    # Only the 0.50 × 100 level survives → 100 * 0.65 = 65
    assert q == pytest.approx(65.0)
