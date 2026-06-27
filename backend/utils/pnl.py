"""Canonical net-PnL math for terminal binary (CTF) positions.

A binary outcome token bought for ``cost`` USD over ``shares`` shares settles to
either $1/share (win) or $0/share (loss). Net realized PnL is therefore bounded:

    min = -cost              (total loss; token settled to $0)
    max = shares*$1 - cost   (every share won)

Several historical reconciliation paths recorded ``actual_profit`` as the gross
notional (== cost) instead of net, inflating every surface that sums realized
PnL (including ``daily_pnl``, which the risk guardian reads). These helpers are
the single source of truth for detecting and repairing those values; import
them anywhere that computes or validates realized PnL so the logic never drifts.
"""
from __future__ import annotations

WIN_STATES = frozenset({"resolved_win", "closed_win"})
LOSS_STATES = frozenset({"resolved_loss", "closed_loss"})
TERMINAL_STATES = WIN_STATES | LOSS_STATES


def feasible_net_pnl_bounds(cost: float, shares: float) -> tuple[float, float]:
    """Return ``(min_net, max_net)`` for a binary position."""
    return (-cost, shares - cost)


def is_implausible_pnl(stored_pnl: float, cost: float, shares: float, *, eps: float = 0.01) -> bool:
    """True when ``stored_pnl`` is physically impossible for the cost basis."""
    if cost <= 0.0 or shares <= 0.0:
        return False  # can't judge without a cost basis
    lo, hi = feasible_net_pnl_bounds(cost, shares)
    return not (lo - eps <= stored_pnl <= hi + eps)


def canonical_terminal_net_pnl(
    status: str,
    cost: float,
    shares: float,
    stored_pnl: float | None = None,
) -> float | None:
    """Deterministic net PnL for a terminal binary position.

    * ``resolved_win``  -> shares*$1 - cost   (held to a $1 settlement)
    * ``resolved_loss`` -> -cost              (settled to $0)
    * ``closed_*``      -> ``stored_pnl`` clamped into the feasible range
      (an early exit's exact proceeds aren't reconstructable here, so trust the
      recorded net but never let it exceed what's physically possible)

    Returns ``None`` when there is no cost basis or (for early exits) no stored
    value to anchor on.
    """
    if cost <= 0.0 or shares <= 0.0:
        return None
    lo, hi = feasible_net_pnl_bounds(cost, shares)
    if status == "resolved_win":
        return hi
    if status == "resolved_loss":
        return lo
    if status in ("closed_win", "closed_loss"):
        if stored_pnl is None:
            return None
        return max(lo, min(hi, stored_pnl))
    return None
