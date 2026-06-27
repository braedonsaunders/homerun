from utils.pnl import (
    canonical_terminal_net_pnl,
    feasible_net_pnl_bounds,
    is_implausible_pnl,
)


def test_bounds():
    lo, hi = feasible_net_pnl_bounds(cost=9.0, shares=10.0)
    assert lo == -9.0
    assert hi == 1.0


def test_resolved_win_is_net_not_notional():
    # buy_no 10 shares at 0.9 => cost 9.0; a win pays 10*$1 => net +1.0 (NOT 9.0)
    assert canonical_terminal_net_pnl("resolved_win", cost=9.0, shares=10.0) == 1.0


def test_resolved_loss_is_minus_cost():
    assert canonical_terminal_net_pnl("resolved_loss", cost=9.0, shares=10.0) == -9.0


def test_closed_early_clamped_into_bounds():
    # plausible stored value passes through
    assert canonical_terminal_net_pnl("closed_win", 9.0, 10.0, stored_pnl=0.5) == 0.5
    # impossible inflated value (the legacy bug: stored == notional) clamps to max
    assert canonical_terminal_net_pnl("closed_win", 9.0, 10.0, stored_pnl=9.0) == 1.0
    # impossible negative clamps to -cost
    assert canonical_terminal_net_pnl("closed_loss", 9.0, 10.0, stored_pnl=-50.0) == -9.0


def test_is_implausible_detects_notional_bug():
    # the exact legacy bug: a resolved_win storing actual_profit == notional(cost)
    assert is_implausible_pnl(9.0, cost=9.0, shares=10.0) is True
    # correct net is plausible
    assert is_implausible_pnl(1.0, cost=9.0, shares=10.0) is False
    # no cost basis -> cannot judge
    assert is_implausible_pnl(9.0, cost=0.0, shares=0.0) is False


def test_missing_cost_basis_returns_none():
    assert canonical_terminal_net_pnl("resolved_win", cost=0.0, shares=0.0) is None
