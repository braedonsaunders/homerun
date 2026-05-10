"""Portfolio engine for the backtest.

Tracks multiple concurrent positions, capital usage, per-strategy exposure
caps, and rolling realized correlations between active markets. Fills from
the matching engine are applied to positions; mark-to-market value is
maintained against the latest book mid; exits emit closing fills back
through the matching engine.

Design choices:

* **Cash + positions**: capital starts at ``initial_capital_usd`` and is
  drawn down by fills. A buy reserves ``size * price + fee`` from cash;
  fills release any over-reservation back. A sell adds the proceeds.
* **Exposure caps**: enforced at order *intent* time (before submission
  to the matching engine), so a strategy can't blow through the budget.
* **Rolling correlation**: maintained as a per-market price-return time
  series with a configurable window; portfolio-level metrics expose the
  realized correlation matrix at any point.
* **Position aggregation**: multiple opens on the same (token, side) are
  averaged into one logical ``Position`` with a weighted entry; exits
  reduce size. This matches Polymarket's wallet semantics.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from services.backtest.matching_engine import Fill, FeeModel

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class PortfolioConfig:
    initial_capital_usd: float = 10_000.0
    max_gross_exposure_usd: Optional[float] = None  # None → unlimited
    max_per_market_notional_usd: Optional[float] = None
    max_per_strategy_notional_usd: Optional[float] = None
    max_open_positions: Optional[int] = None
    correlation_window_size: int = 240  # samples (e.g., 240 * 0.5s = 2 min)
    enable_correlation: bool = True
    # Fees applied at position settlement (resolution / final mark-to-market).
    # The matching engine's ``FeeModel`` covers per-fill costs (gas + bps);
    # the portfolio adds resolution-time fees on top.
    fee_model: Optional[FeeModel] = None


# ── Position state ────────────────────────────────────────────────────


@dataclass
class Position:
    token_id: str
    side: str  # BUY or SELL — the side of the *opening* fill
    strategy_slug: Optional[str] = None
    entry_price: float = 0.0  # size-weighted
    size: float = 0.0
    cost_basis_usd: float = 0.0  # cash spent (BUY) or proceeds collected (SELL)
    realized_pnl_usd: float = 0.0
    fees_paid_usd: float = 0.0
    last_mark_price: Optional[float] = None
    last_mark_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    fill_count: int = 0

    def apply_open_fill(self, fill: Fill) -> None:
        """Apply a fill that increases position size."""
        new_size = self.size + fill.size
        self.entry_price = (
            (self.entry_price * self.size + fill.price * fill.size) / new_size
            if new_size > 0
            else fill.price
        )
        self.size = new_size
        if self.side == "BUY":
            self.cost_basis_usd += fill.notional_usd
        else:  # short / SELL open — proceeds taken in
            self.cost_basis_usd += fill.notional_usd  # treat as basis for symmetry
        self.fees_paid_usd += fill.fee_usd
        self.fill_count += 1
        if self.opened_at is None:
            self.opened_at = fill.occurred_at

    def apply_close_fill(self, fill: Fill) -> None:
        """Apply a fill that reduces position size."""
        if self.size <= 0:
            return
        take = min(fill.size, self.size)
        proportion = take / self.size if self.size > 0 else 0.0
        cost_chunk = self.cost_basis_usd * proportion
        proceeds = fill.price * take
        # Long position: pnl = proceeds − cost
        if self.side == "BUY":
            pnl = proceeds - cost_chunk
        else:
            # Short: pnl = cost - proceeds  (sold at entry, bought back at fill)
            pnl = cost_chunk - proceeds
        pnl -= fill.fee_usd
        self.realized_pnl_usd += pnl
        self.fees_paid_usd += fill.fee_usd
        self.size -= take
        self.cost_basis_usd -= cost_chunk
        self.fill_count += 1
        if self.size <= 1e-12:
            self.closed_at = fill.occurred_at

    def mark(self, price: float, at: datetime) -> None:
        self.last_mark_price = float(price)
        self.last_mark_at = at

    def unrealized_pnl_usd(self) -> float:
        if self.size <= 0 or self.last_mark_price is None:
            return 0.0
        if self.side == "BUY":
            return self.size * (self.last_mark_price - self.entry_price)
        return self.size * (self.entry_price - self.last_mark_price)

    def total_pnl_usd(self) -> float:
        return self.realized_pnl_usd + self.unrealized_pnl_usd()


# ── Rolling correlation ──────────────────────────────────────────────────


class _ReturnSeries:
    """Per-token rolling log-return series for correlation."""

    def __init__(self, window: int):
        self._window = max(8, int(window))
        self._prices: deque[float] = deque(maxlen=self._window + 1)
        self._returns: deque[float] = deque(maxlen=self._window)

    def push(self, price: float) -> None:
        if price <= 0:
            return
        if self._prices:
            prev = self._prices[-1]
            if prev > 0:
                self._returns.append(math.log(price / prev))
        self._prices.append(float(price))

    @property
    def returns(self) -> tuple[float, ...]:
        return tuple(self._returns)


def _pearson(a: Iterable[float], b: Iterable[float]) -> Optional[float]:
    xs = list(a)
    ys = list(b)
    n = min(len(xs), len(ys))
    if n < 4:
        return None
    xs = xs[-n:]
    ys = ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0 or sy <= 0:
        return None
    return num / (sx * sy)


# ── Portfolio engine ─────────────────────────────────────────────────────


class Portfolio:
    """Multi-position state machine driven by ``Fill`` events."""

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.cash_usd: float = float(self.config.initial_capital_usd)
        self.peak_equity_usd: float = self.cash_usd
        self.positions: dict[tuple[str, str, Optional[str]], Position] = {}
        self.closed_positions: list[Position] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self.cash_history: list[tuple[datetime, float]] = []
        self._return_series: dict[str, _ReturnSeries] = {}
        self._resolution_fees_paid_usd: float = 0.0

    # ── Pre-trade gate ────────────────────────────────────────────────

    def can_submit(
        self,
        *,
        token_id: str,
        side: str,
        price: float,
        size: float,
        strategy_slug: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """Return (ok, reason). ``reason`` is None when ok."""
        side_norm = (side or "").upper()
        notional = max(0.0, float(price) * float(size))

        # Capital cap (BUY only — SELL of an open position releases capital)
        if side_norm == "BUY":
            if self.cash_usd + 1e-9 < notional:
                return False, f"insufficient_cash:{self.cash_usd:.2f}<{notional:.2f}"

        # Gross exposure
        cap = self.config.max_gross_exposure_usd
        if cap is not None:
            current_gross = self.gross_exposure_usd()
            if current_gross + notional > cap + 1e-9:
                return False, f"gross_exposure_cap:{current_gross + notional:.2f}>{cap:.2f}"

        # Per-market
        per_market = self.config.max_per_market_notional_usd
        if per_market is not None:
            existing = sum(
                pos.size * (pos.last_mark_price or pos.entry_price)
                for key, pos in self.positions.items()
                if key[0] == token_id
            )
            if existing + notional > per_market + 1e-9:
                return False, f"per_market_cap:{existing + notional:.2f}>{per_market:.2f}"

        # Per-strategy
        per_strat = self.config.max_per_strategy_notional_usd
        if per_strat is not None and strategy_slug:
            existing = sum(
                pos.size * (pos.last_mark_price or pos.entry_price)
                for pos in self.positions.values()
                if pos.strategy_slug == strategy_slug
            )
            if existing + notional > per_strat + 1e-9:
                return False, f"per_strategy_cap:{existing + notional:.2f}>{per_strat:.2f}"

        # Max open positions
        if self.config.max_open_positions is not None:
            if len(self.positions) >= self.config.max_open_positions:
                # Allow only if we're closing an existing position
                key = (token_id, side_norm, strategy_slug)
                if key not in self.positions:
                    return False, "max_open_positions"

        return True, None

    # ── Fill application ──────────────────────────────────────────────

    def apply_fill(self, fill: Fill, *, strategy_slug: Optional[str]) -> None:
        side_norm = (fill.side or "").upper()
        # Try to match against an opposing-side position first (closing).
        opposing_side = "SELL" if side_norm == "BUY" else "BUY"
        opp_key = (fill.token_id, opposing_side, strategy_slug)
        if opp_key in self.positions:
            pos = self.positions[opp_key]
            prior_realized = pos.realized_pnl_usd
            pos.apply_close_fill(fill)
            self._on_cash_change_from_close(pos, fill)
            # If the closing fill flipped this slice into a winning state,
            # apply the venue's resolution fee on the *incremental* gain.
            if self.config.fee_model is not None:
                gained = pos.realized_pnl_usd - prior_realized
                if gained > 0:
                    res_fee = self.config.fee_model.resolution_fee(
                        gross_winnings_usd=gained
                    )
                    if res_fee > 0:
                        pos.fees_paid_usd += res_fee
                        pos.realized_pnl_usd -= res_fee
                        self.cash_usd -= res_fee
                        self._resolution_fees_paid_usd += res_fee
            if pos.size <= 1e-12:
                self.closed_positions.append(pos)
                del self.positions[opp_key]
            return

        # Otherwise this is an opening or scaling fill on the SAME side.
        key = (fill.token_id, side_norm, strategy_slug)
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(
                token_id=fill.token_id,
                side=side_norm,
                strategy_slug=strategy_slug,
            )
            self.positions[key] = pos
        pos.apply_open_fill(fill)
        if side_norm == "BUY":
            self.cash_usd -= fill.notional_usd + fill.fee_usd
        else:
            self.cash_usd += fill.notional_usd - fill.fee_usd

    def _on_cash_change_from_close(self, pos: Position, fill: Fill) -> None:
        """Cash effect of a closing fill: long sells receive proceeds; short
        covers pay cost. Fees always reduce cash.
        """
        if pos.side == "BUY":
            # Long, closing via SELL: cash += proceeds
            self.cash_usd += fill.notional_usd - fill.fee_usd
        else:
            # Short, closing via BUY: cash -= cost
            self.cash_usd -= fill.notional_usd + fill.fee_usd

    # ── Mark-to-market ────────────────────────────────────────────────

    def mark(self, *, token_id: str, price: float, at: datetime) -> None:
        for key, pos in self.positions.items():
            if key[0] == token_id:
                pos.mark(price, at)
        if self.config.enable_correlation:
            series = self._return_series.setdefault(
                token_id, _ReturnSeries(self.config.correlation_window_size)
            )
            series.push(price)
        equity = self.equity_usd()
        self.peak_equity_usd = max(self.peak_equity_usd, equity)
        self.equity_history.append((at, equity))
        self.cash_history.append((at, self.cash_usd))

    # ── Reporting ─────────────────────────────────────────────────────

    def equity_usd(self) -> float:
        """Net wealth: cash + marked-to-market value of open positions.

        For a BUY (long): we OWN ``size`` contracts worth ``size × mark``.
        For a SELL (short): we OWE ``size`` contracts costing ``size × mark``
        to buy back.  Cash already reflects the entry leg (debited on open
        for BUY, credited for SELL), so the open-position contribution is
        the current market value (long) or negative current cost (short).

        Falls back to ``entry_price`` when no mark has arrived yet — at
        the moment of opening the position is worth what we just paid for
        it, NOT zero.

        Old formula ``cash + sum(unrealized_pnl)`` only added the
        gain-relative-to-entry, so a freshly-opened (or mark==entry)
        position dropped equity by the full notional outlay until the
        position eventually closed and cash absorbed the proceeds.  That
        bled into ``equity_history`` and corrupted Sharpe / Sortino /
        Calmar / max-drawdown for any run that held positions mid-window
        (i.e. essentially every run).
        """
        total = self.cash_usd
        for p in self.positions.values():
            if p.size <= 0:
                continue
            mark = p.last_mark_price if p.last_mark_price is not None else p.entry_price
            position_value = p.size * mark
            if p.side == "BUY":
                total += position_value
            else:
                total -= position_value
        return total

    def gross_exposure_usd(self) -> float:
        return sum(
            p.size * (p.last_mark_price or p.entry_price) for p in self.positions.values()
        )

    def realized_pnl_usd(self) -> float:
        return sum(p.realized_pnl_usd for p in self.closed_positions) + sum(
            p.realized_pnl_usd for p in self.positions.values()
        )

    def unrealized_pnl_usd(self) -> float:
        return sum(p.unrealized_pnl_usd() for p in self.positions.values())

    def total_pnl_usd(self) -> float:
        return self.realized_pnl_usd() + self.unrealized_pnl_usd()

    def fees_paid_usd(self) -> float:
        return sum(p.fees_paid_usd for p in self.closed_positions) + sum(
            p.fees_paid_usd for p in self.positions.values()
        )

    @property
    def resolution_fees_paid_usd(self) -> float:
        return float(self._resolution_fees_paid_usd)

    def per_fill_fees_paid_usd(self) -> float:
        return max(0.0, self.fees_paid_usd() - self._resolution_fees_paid_usd)

    def max_drawdown_usd(self) -> float:
        if not self.equity_history:
            return 0.0
        peak = -math.inf
        worst = 0.0
        for _, eq in self.equity_history:
            peak = max(peak, eq)
            worst = max(worst, peak - eq)
        return worst

    def correlation_matrix(self) -> dict[tuple[str, str], float]:
        if not self.config.enable_correlation:
            return {}
        tokens = sorted(self._return_series.keys())
        out: dict[tuple[str, str], float] = {}
        for i, ta in enumerate(tokens):
            for tb in tokens[i + 1 :]:
                rho = _pearson(self._return_series[ta].returns, self._return_series[tb].returns)
                if rho is not None:
                    out[(ta, tb)] = rho
        return out

    def open_position_count(self) -> int:
        return len(self.positions)

    def closed_position_count(self) -> int:
        return len(self.closed_positions)


__all__ = [
    "Portfolio",
    "PortfolioConfig",
    "Position",
]
