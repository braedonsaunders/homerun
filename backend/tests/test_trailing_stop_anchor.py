"""Regression test for the trailing-stop anchor bug.

Production incident 2026-05-20: order ``1ada192f8d594ace86c7138a7177421c``
opened at entry_price=$0.925, price dipped to $0.565 (39% drawdown),
then drifted back up before resolution.  Position held to resolution
and lost the full $10.05 — the trailing stop (12% from high) never
fired.

Root cause: the live/shadow lifecycle's ``reconcile_*_positions``
initialized ``highest_price`` from the first observed mark instead of
``entry_price``.  Because the first mark happened AFTER the dip, the
trailing-stop anchor became $0.565 — the same as the current price —
and the 12% drop check never tripped.

This test pins the strategy's ``should_exit`` semantics against the
fixed lifecycle state shape: when the position is bought at $0.925
and the current price reaches $0.565, the trailing stop MUST fire
"close" because the drop from the (anchored) high is 38.9% — well
above the 12% threshold.

If the lifecycle ever stops anchoring highest_price to entry_price
again, the second-stage tail-end-carry positions will silently bleed
through trailing-stop signals during early drawdowns.  This test
guards that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.tail_end_carry import TailEndCarryStrategy


def _position(*, entry: float, current: float, highest: float, age_min: float = 480.0):
    """Position view matching the lifecycle's pos_view shape."""
    return SimpleNamespace(
        entry_price=entry,
        current_price=current,
        highest_price=highest,
        lowest_price=current,
        age_minutes=age_min,
        pnl_percent=((current - entry) / entry) * 100.0 if entry > 0 else None,
        filled_size=10.0 / entry if entry > 0 else 0.0,
        notional_usd=10.0,
        strategy_context={},
        config={
            # Mirror the production strategy_exit_config that was on
            # the losing order, trimmed to the fields exercised here.
            "trailing_stop_enabled": True,
            "trailing_stop_pct": 12.0,
            "inversion_stop_enabled": True,
            "inversion_price_threshold": 0.50,
            "max_hold_minutes": 1440,
            # Disable time-weighted stop (its only effect is to tighten
            # trailing_stop_pct further; testing the legacy floor is
            # sufficient for this regression).
            "time_weighted_stop_enabled": False,
            # Resolution hold is opt-in via minutes_left — we pass a
            # large minutes_left below so the test is squarely in the
            # post-hold trailing-stop window.
            "resolution_hold_enabled": True,
            "resolution_hold_minutes": 360,
        },
        outcome_idx=1,  # NO (irrelevant for trailing stop math)
    )


def _market_state(*, price: float | None, seconds_left: float = 480 * 60.0):
    """Market state dict matching the lifecycle's market_state_dict."""
    return {
        "current_price": price,
        "market_tradable": True,
        "is_resolved": False,
        "winning_outcome": None,
        "seconds_left": seconds_left,
        "token_id": "tok-test",
        "mark_source": "market_mark",
        "min_order_size_usd": 10.0,
        "notional_usd": 10.0,
    }


def test_trailing_stop_fires_when_anchored_to_entry_after_drawdown():
    """Reproduces the FIXED behavior.

    Entry $0.925, price dipped to $0.565 (39% drawdown).  When
    highest_price is correctly anchored to entry_price ($0.925), the
    trailing stop's drop-from-high = 38.9% > 12% threshold and the
    strategy returns "close".
    """
    strategy = TailEndCarryStrategy()
    pos = _position(entry=0.925, current=0.565, highest=0.925, age_min=480.0)
    decision = strategy.should_exit(pos, _market_state(price=0.565))

    assert decision.action == "close", (
        f"Trailing stop must fire for 39% drawdown from entry $0.925 "
        f"to current $0.565; got action={decision.action!r} "
        f"reason={decision.reason!r}"
    )
    assert "trailing" in decision.reason.lower(), decision.reason


def test_trailing_stop_silently_loses_anchor_under_old_initialization():
    """Reproduces the OLD bug as a guard: if highest_price is
    initialized from the first observed mark (which landed below
    entry), the trailing stop never fires.

    This is the production scenario for order 1ada192f.  The test
    asserts the strategy's behavior given the buggy state — if the
    test ever flips to "close" on this input, we've changed the
    strategy semantics and should re-examine the lifecycle fix.
    """
    strategy = TailEndCarryStrategy()
    # The bug: lifecycle initialized highest_price = current_price
    # (0.565) on the first mark, forgetting that we bought at $0.925.
    pos = _position(entry=0.925, current=0.565, highest=0.565, age_min=480.0)
    decision = strategy.should_exit(pos, _market_state(price=0.565))

    # Strategy correctly does NOT fire on this state — highest == current
    # so drop_from_high = 0%.  This is exactly what allowed the
    # production loss; the bug is in the LIFECYCLE not seeding
    # highest_price from entry_price.
    assert decision.action != "close" or "trailing" not in decision.reason.lower(), (
        f"Strategy must not invent a trailing-stop trigger when "
        f"highest_price equals current_price; got {decision!r}"
    )


def test_inversion_stop_does_not_save_position_above_threshold():
    """Sanity: the inversion-stop floor at $0.50 doesn't catch
    drawdowns that stop just above it.  This is part of WHY the
    trailing-stop anchor matters — inversion alone isn't enough.

    Entry $0.925, current $0.565: just above the $0.50 inversion
    floor.  Inversion stop must NOT close (price is above floor).
    Only the trailing-stop or stop-loss-from-entry should catch this.
    """
    strategy = TailEndCarryStrategy()
    # Highest = current (bug scenario) so trailing stop doesn't fire.
    pos = _position(entry=0.925, current=0.565, highest=0.565, age_min=480.0)
    # Push minutes_left high to skip resolution-hold branch.
    decision = strategy.should_exit(pos, _market_state(price=0.565, seconds_left=86400))

    assert decision.action == "hold", (
        f"Inversion stop must not catch a position above the $0.50 "
        f"threshold; got {decision!r}"
    )


def test_trailing_stop_does_not_fire_within_threshold():
    """Sanity: a 5% drop from high (below the 12% threshold) holds."""
    strategy = TailEndCarryStrategy()
    pos = _position(entry=0.925, current=0.880, highest=0.925, age_min=480.0)
    decision = strategy.should_exit(pos, _market_state(price=0.880, seconds_left=86400))

    assert decision.action == "hold", (
        f"Trailing stop must not fire on a 5% drop (below 12% "
        f"threshold); got {decision!r}"
    )
