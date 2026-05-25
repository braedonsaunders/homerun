"""Regression test for the pre-resolution structural exit (gap cap).

Production incidents 2026-05: tail-end-carry positions carried ~0.9
favorites into binary resolution.  When the favored outcome lost, the
price gapped 0.9 -> 0 at the resolution instant with no declining price
for any stop to catch, costing the full notional.  The pre-resolution
structural exit closes the position WHILE STILL TRADABLE once the market
is within a configured window of resolution, so it never holds through
the gap.  Disabled (0) by default to preserve the pure carry edge until
the operator opts in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.tail_end_carry import TailEndCarryStrategy


def _decide(cfg_extra, minutes_left, *, sports=False):
    strat = TailEndCarryStrategy()
    cfg = {
        "resolution_hold_enabled": True,
        "resolution_hold_minutes": 360.0,
        "sports_resolution_hold_minutes": 150.0,
        "resolution_hold_max_loss_pct": 25.0,
    }
    cfg.update(cfg_extra)
    if sports:
        cfg["market_category"] = "sports"
    pos = SimpleNamespace(
        entry_price=0.94, highest_price=0.95, age_minutes=600.0,
        config=cfg, strategy_context={},
    )
    ms = {"current_price": 0.93, "seconds_left": minutes_left * 60.0,
          "is_resolved": False, "token_id": "x"}
    return strat.should_exit(pos, ms)


def test_disabled_by_default_holds_into_resolution():
    # 0 = disabled -> pure carry (resolution hold), behavior unchanged.
    d = _decide({}, minutes_left=8)
    assert d.action == "hold"


def test_pre_resolution_exit_fires_inside_window():
    d = _decide({"pre_resolution_force_exit_minutes": 15.0}, minutes_left=8)
    assert d.action == "close"
    assert "Pre-resolution structural exit" in d.reason


def test_pre_resolution_exit_holds_outside_window():
    # Within the resolution-hold window but outside the gap-cap window: carry.
    d = _decide({"pre_resolution_force_exit_minutes": 15.0}, minutes_left=30)
    assert d.action == "hold"


def test_pre_resolution_exit_overrides_resolution_hold():
    # The gap cap must override the resolution hold (it is checked first).
    d = _decide(
        {"pre_resolution_force_exit_minutes": 20.0, "resolution_hold_minutes": 360.0},
        minutes_left=10,
    )
    assert d.action == "close"


def test_sports_uses_sports_window():
    d = _decide({"sports_pre_resolution_force_exit_minutes": 12.0}, minutes_left=5, sports=True)
    assert d.action == "close"
