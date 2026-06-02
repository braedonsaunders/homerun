"""Golden-source market settlement for the backtester.

A position held into a market's resolution must settle at the binary
outcome — the winning outcome-token redeems at $1.00, the losing token at
$0.00 — NOT at the last observed order-book mid.  For BTC/ETH 5-minute
Up/Down markets, which exist to be held to expiry, that payoff IS the
dominant PnL term, so marking to the last mid (the legacy behavior)
systematically misstates returns.

This module defines the per-token settlement record the engine consumes.
Populating the actual winners is the settlement *store*'s job
(``services.backtest.settlement_store``), which sources outcomes OFFLINE
and reproducibly (no decision-time look-ahead): a winner is only ever
read at/after a position's resolution time, never fed into the strategy's
decision inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class TokenSettlement:
    """Resolved outcome for a single outcome-token, consumed at settlement.

    ``settle_price`` is the redemption value of ONE share of this token at
    resolution: ``1.0`` if this token's outcome won, ``0.0`` if it lost.
    ``resolution_time`` is when the market resolved (UTC).
    ``winning_outcome`` is the label of the winning outcome (surfaced to the
    strategy's ``market_state`` once resolved).  ``source`` records the
    provenance of the resolution (e.g. ``polybacktest`` / ``gamma`` /
    ``import``) for auditability.
    """

    token_id: str
    # ``None`` means the market is known to resolve (``resolution_time``
    # set) but the winner isn't available — the engine then surfaces
    # ``is_resolved`` to the strategy rather than auto-redeeming.
    settle_price: Optional[float] = None
    resolution_time: Optional[datetime] = None
    winning_outcome: Optional[str] = None
    condition_id: Optional[str] = None
    source: str = ""

    @property
    def won(self) -> Optional[bool]:
        if self.settle_price is None:
            return None
        return self.settle_price >= 0.5
