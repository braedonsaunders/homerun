"""Order-book aware matching engine for the backtester.

This module is the heart of execution-realistic backtesting. Given:

* a stream of L2 snapshots (``BookReplay``),
* a venue model (``Venue`` — currently Polymarket),
* a latency model (``LatencyModel``), and
* a sequence of order intents emitted by a strategy,

it produces a deterministic ledger of fills, partial fills, rejections,
and cancel/replace transitions that mirror the live ``execute_live_order``
→ Polymarket CLOB path as closely as possible without sending traffic to
the venue.

Order lifecycle states
======================

::

    pending_submit → working   → filled | partial → done
                  ↘ rejected       ↓
                                  (cancel | reprice)
                                   ↓
                                  cancelled

* ``pending_submit`` — order intent emitted at strategy time T; will hit
  the venue at T + submit_latency.
* ``working`` — accepted by the venue and resting on the book.
* ``partial`` — has filled some size but not all; remainder still resting
  (only possible for GTC; IOC/FAK/FOK transition straight to done).
* ``filled`` / ``cancelled`` / ``rejected`` — terminal.

The engine is **single-threaded and deterministic**: given identical
inputs (including latency-model seed) it produces identical fills.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from services.backtest.book_replay import BookSnapshot
from services.backtest.latency_model import LatencyModel
from services.backtest.venue_model import (
    TIF_FAK,
    TIF_FOK,
    TIF_GTC,
    TIF_IOC,
    PolymarketVenue,
    Venue,
)

logger = logging.getLogger(__name__)


# ── Order / Fill data classes ────────────────────────────────────────────


class OrderState:
    PENDING = "pending_submit"
    WORKING = "working"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestOrder:
    """A simulated order living in the matching engine.

    ``trader_id``/``strategy_slug`` are bookkeeping fields that flow into
    the ``Fill`` records and metrics. ``meta`` is opaque caller state
    (e.g., the ``ExitPolicy`` child id when this order is part of a ladder).
    """

    order_id: str
    token_id: str
    side: str  # BUY | SELL
    price: float
    size: float
    tif: str
    post_only: bool
    submitted_at: datetime
    trader_id: Optional[str] = None
    strategy_slug: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    state: str = OrderState.PENDING
    venue_received_at: Optional[datetime] = None  # when ack arrived
    filled_size: float = 0.0
    avg_fill_price: float = 0.0  # size-weighted
    cancel_pending_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    reject_reason: Optional[str] = None
    fills: list["Fill"] = field(default_factory=list)

    @property
    def remaining_size(self) -> float:
        return max(0.0, float(self.size) - float(self.filled_size))

    @property
    def is_terminal(self) -> bool:
        return self.state in {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED}

    @property
    def filled_notional_usd(self) -> float:
        return float(self.filled_size) * float(self.avg_fill_price)


@dataclass(frozen=True)
class Fill:
    """A single match event, possibly one of many for a partial-fill order."""

    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    fee_usd: float
    occurred_at: datetime
    fill_index: int  # 0-based within the parent order
    venue_sequence: Optional[int] = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def notional_usd(self) -> float:
        return float(self.price) * float(self.size)


# ── Fee model ────────────────────────────────────────────────────────────
#
# Polymarket fees on a backtest break into three components:
#
#   1. **Per-fill gas** — every submitted order pays Polygon gas (proxy
#      wallet → CTF Exchange). Default ~$0.005/tx, configurable.
#   2. **Per-fill venue fee** — the CLOB itself charges 0 bps on trades;
#      kept as a parameter for non-Polymarket venues.
#   3. **Resolution fee** — at settlement the venue takes a fraction of
#      the winning side's proceeds (2% on Polymarket). This is *not* a
#      per-fill fee; it is applied by ``portfolio.py`` when a position
#      closes profitably (or settles at $1 / $0).
#
# This module exposes a single ``FeeModel`` that bundles all three, so
# both the matching engine and the portfolio reach for the same source of
# truth. Defaults are Polymarket-correct; pass overrides for other venues.


@dataclass
class FeeModel:
    """Bundled fee model for backtesting.

    Defaults match Polymarket as of 2026:

    * ``taker_bps = 0.0`` — CLOB trades pay no taker fee
    * ``maker_bps = 0.0`` — CLOB trades pay no maker fee
    * ``per_fill_gas_usd = 0.005`` — Polygon proxy-wallet tx cost
    * ``resolution_fee_rate = 0.02`` — 2% on winning proceeds at settlement
    * ``negrisk_conversion_gas_usd = 0.01`` — additional gas when a
      negative-risk multi-outcome leg requires CTF↔NegRisk conversion
    * ``maker_rebate_bps = 0.0`` — Polymarket-specific liquidity-reward
      approximation.  Real rewards are paid daily as LP tokens
      proportional to time-weighted maker depth inside a configurable
      spread band; modeling that fully requires per-tick time-on-book
      tracking which is too expensive in backtest.  This bps value is a
      conservative per-fill PROXY: enable it to estimate the realised
      maker P&L uplift from rebates, which on top crypto markets has
      historically been worth ~1-3 bps annualised on notional.  Treat
      anything over 5 bps as optimistic — real rebate yield erodes as
      makers compete for the same band.
    * ``maker_rebate_max_spread_bps = 50.0`` — only count maker fills
      whose limit price was within this distance of the snapshot mid
      as eligible for the rebate.  Approximates the actual program's
      "inside the spread band" requirement.

    Pass ``per_fill_gas_usd=0`` to disable gas modeling (unit tests do
    this so partial-fill counts don't get distorted by tx fees).
    """

    taker_bps: float = 0.0
    maker_bps: float = 0.0
    per_fill_gas_usd: float = 0.005
    resolution_fee_rate: float = 0.02
    negrisk_conversion_gas_usd: float = 0.01
    maker_rebate_bps: float = 0.0
    maker_rebate_max_spread_bps: float = 50.0

    def fill_fee(
        self,
        *,
        price: float,
        size: float,
        is_maker: bool,
        is_negrisk: bool = False,
        mid_price: float | None = None,
    ) -> float:
        """Per-fill fee in USD: bps fee on notional + gas, minus an
        optional maker rebate when the fill was a maker inside the
        configured spread band.

        ``mid_price`` is optional; if None, the rebate band check is
        skipped and the rebate (if configured) applies to every maker
        fill regardless of distance from mid.  Callers that have a
        snapshot context (the matching engine does) should pass it so
        the rebate model better matches the live program's
        inside-band-only payment.
        """
        bps = self.maker_bps if is_maker else self.taker_bps
        bps_fee = 0.0
        if bps > 0:
            bps_fee = float(price) * float(size) * (bps / 10_000.0)
        gas = max(0.0, float(self.per_fill_gas_usd))
        if is_negrisk:
            gas += max(0.0, float(self.negrisk_conversion_gas_usd))
        rebate_credit = 0.0
        if is_maker and float(self.maker_rebate_bps or 0.0) > 0:
            inside_band = True
            if mid_price is not None and float(mid_price) > 0:
                # Distance from mid as bps of mid.
                dist_bps = abs(float(price) - float(mid_price)) / float(mid_price) * 10_000.0
                inside_band = dist_bps <= float(self.maker_rebate_max_spread_bps or 0.0)
            if inside_band:
                rebate_credit = (
                    float(price) * float(size) * (float(self.maker_rebate_bps) / 10_000.0)
                )
        return bps_fee + gas - rebate_credit

    # ------------------------------------------------------------------
    # Adversarial book impact — see ``ImpactModel`` below.  The
    # FeeModel doesn't apply impact directly, but we hold a reference
    # so the matching engine can pull both from a single config point.
    # ------------------------------------------------------------------

    def resolution_fee(self, *, gross_winnings_usd: float) -> float:
        """Fee charged at settlement on a position's gross winnings.

        ``gross_winnings_usd`` is the proceeds *minus* the cost basis on
        the winning side — i.e., the realized profit before fees. Returns
        zero when the position lost (gross_winnings <= 0) since
        Polymarket only charges the winner.
        """
        if gross_winnings_usd <= 0:
            return 0.0
        rate = max(0.0, float(self.resolution_fee_rate))
        return float(gross_winnings_usd) * rate


# ── Adversarial book impact model ──────────────────────────────────────
#
# The standard backtest assumption is that your order does NOT move the
# book — you fill at the displayed price and other participants don't
# respond.  For tiny orders against deep books that's a reasonable
# approximation.  For institutional-size orders or thin books, it's
# significantly optimistic: real makers lift their quotes when they see
# an aggressive taker walking the book; real takers fade.
#
# This model adds a square-root-impact penalty to taker fills.  The
# penalty represents how much the book would have shifted AWAY from
# the taker over the course of consuming visible depth.  Calibration:
#
#   impact_bps = strength * sqrt(your_notional / total_visible_notional)
#
# where ``strength`` is a per-venue tunable.  Empirical literature on
# equities pins ``strength`` around 0.5-1.0; on prediction markets it
# is likely smaller (binary contracts have natural anchors at 0/1) —
# we default to 0.0 (off) so existing backtests are unchanged, and
# expose ``impact_bps`` as an additive override for the operator.
#
# The model is consulted by the matching engine on each taker fill;
# it adjusts the effective fill price by ``impact_bps`` BPS in the
# adverse direction (BUY pays more, SELL receives less).


@dataclass
class ImpactModel:
    """Almgren-Chriss square-root impact + capacity curve.

    Two-regime response function on the size/depth ratio ``r``:

      * ``r <= capacity_threshold``  →  ``strength_bps * sqrt(r)``
        (textbook Almgren-Chriss; impact grows concavely with size).
      * ``r >  capacity_threshold``  →  the same square-root term at
        the threshold PLUS a superlinear penalty
        ``strength_bps * (r - threshold) ** capacity_exponent``.

    The second regime models capacity decay: real books thin out
    under sustained pressure, late-arriving liquidity is asymmetric
    (other makers pull ahead of you), and sweep orders pay a worse
    blended price than depth-weighted average suggests.  Without this
    term the simulator says a $100k order in a $10k book pays the
    same impact as $1k in a $10k book — patently wrong.

    ``strength_bps`` is the impact (in bps of price) you'd pay if you
    consumed exactly the threshold fraction of visible side depth.
    ``cap_bps`` truncates the worst case so a single sweep can't
    eat infinite impact.

    ``strength_bps = 0`` disables impact entirely (existing backtests
    unchanged until operators opt in).  Calibration starting points:
        * 5-10 bps  — top-of-book on deep crypto markets (BTC/ETH 5m).
        * 25-50 bps — illiquid event markets, thin books, or off-hours.

    Defaults: capacity_threshold=0.5 (50% of book), capacity_exponent=1.5
    — kicks in when an order is half the visible book and grows fast
    from there.
    """

    strength_bps: float = 0.0
    cap_bps: float = 200.0  # never penalise a single fill more than 200 bps
    capacity_threshold: float = 0.5
    capacity_exponent: float = 1.5

    def adverse_bps(self, *, your_size_shares: float, side_visible_depth_shares: float) -> float:
        """Return the bps adverse adjustment (>= 0) applied to the fill
        price.  Caller decides direction (BUY adds bps, SELL subtracts).
        """
        if self.strength_bps <= 0:
            return 0.0
        if your_size_shares <= 0 or side_visible_depth_shares <= 0:
            return 0.0
        ratio = max(0.0, float(your_size_shares) / float(side_visible_depth_shares))
        threshold = max(1e-6, float(self.capacity_threshold))
        if ratio <= threshold:
            bps = float(self.strength_bps) * math.sqrt(ratio)
        else:
            base_bps = float(self.strength_bps) * math.sqrt(threshold)
            excess = ratio - threshold
            penalty_bps = float(self.strength_bps) * (excess ** float(self.capacity_exponent))
            bps = base_bps + penalty_bps
        return min(bps, max(0.0, float(self.cap_bps)))


# ── Matching engine ──────────────────────────────────────────────────────


@dataclass
class _ResidualBook:
    """Mutable per-token book state with the matching engine's view of
    consumed liquidity. We never mutate the shared ``BookSnapshot``; this
    tracks how much of each level we have already consumed in the current
    snapshot interval so successive maker-taker matches don't double-count.
    """

    snapshot: BookSnapshot
    consumed_bids: dict[float, float] = field(default_factory=dict)  # price -> qty
    consumed_asks: dict[float, float] = field(default_factory=dict)

    def remaining_bids(self) -> list[tuple[float, float]]:
        return [
            (lvl.price, max(0.0, lvl.size - self.consumed_bids.get(lvl.price, 0.0)))
            for lvl in self.snapshot.bids
        ]

    def remaining_asks(self) -> list[tuple[float, float]]:
        return [
            (lvl.price, max(0.0, lvl.size - self.consumed_asks.get(lvl.price, 0.0)))
            for lvl in self.snapshot.asks
        ]

    def consume(self, *, side_taker: str, price: float, size: float) -> None:
        """Record that a taker on ``side_taker`` consumed ``size`` at ``price``.
        SELL takers consume bids; BUY takers consume asks.
        """
        side = (side_taker or "").upper()
        if side == "SELL":
            self.consumed_bids[price] = self.consumed_bids.get(price, 0.0) + size
        elif side == "BUY":
            self.consumed_asks[price] = self.consumed_asks.get(price, 0.0) + size


class MatchingEngine:
    """Backtest matching engine with latency and full lifecycle.

    Usage pattern (driven by ``BacktestEngine``):

    1. Call ``advance_to(snapshot)`` for each new L2 snapshot in
       chronological order. The engine refreshes its internal book and
       attempts to fill any working orders that became marketable as a
       result of book movement, plus any orders whose submit-latency
       window has now elapsed.
    2. Call ``submit(order)`` to enqueue a strategy-emitted order. It is
       held in ``pending_submit`` until ``advance_to`` has crossed the
       submit-latency offset; only then does it interact with the book.
    3. Call ``cancel(order_id, requested_at)`` to enqueue a cancel.
    """

    def __init__(
        self,
        *,
        venue: Optional[Venue] = None,
        latency: Optional[LatencyModel] = None,
        fees: Optional[FeeModel] = None,
        impact: Optional[ImpactModel] = None,
    ):
        self.venue: Venue = venue or PolymarketVenue()
        self.latency: LatencyModel = latency or LatencyModel()
        self.fees: FeeModel = fees or FeeModel()
        self.impact: ImpactModel = impact or ImpactModel()

        self._orders: dict[str, BacktestOrder] = {}
        self._working_by_token: dict[str, list[str]] = {}
        # Per-token count of orders in ANY non-terminal state (PENDING /
        # WORKING / PARTIAL).  Used by the engine's fast-path bail to
        # decide whether a snapshot for ``token_id`` can be skipped — if
        # zero, no order on this token needs the matcher's attention.
        # Maintained alongside _working_by_token (which is WORKING-only)
        # because PENDING orders also need matcher.advance_to to run on
        # their token's next snapshot to be admitted.
        self._active_count_by_token: dict[str, int] = {}
        self._pending_cancels: dict[str, datetime] = {}  # order_id -> effective_at
        self._books: dict[str, _ResidualBook] = {}
        self._now: Optional[datetime] = None
        self._fills_by_order: dict[str, list[Fill]] = {}
        self._all_fills: list[Fill] = []

    # ── Public API ────────────────────────────────────────────────────

    @property
    def now(self) -> Optional[datetime]:
        return self._now

    def submit(self, order: BacktestOrder) -> BacktestOrder:
        """Register a new order intent. State stays ``pending_submit`` until
        the next ``advance_to`` call crosses its submit-latency offset.
        """
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order_id: {order.order_id}")
        # Pre-validate; reject early if the rule is purely static.
        decision = self.venue.validate_submit(
            side=order.side,
            price=order.price,
            size=order.size,
            tif=order.tif,
            post_only=order.post_only,
        )
        if not decision.ok:
            order.state = OrderState.REJECTED
            order.reject_reason = decision.reject_reason
            self._orders[order.order_id] = order
            self._fills_by_order[order.order_id] = []
            return order
        if decision.adjusted_price is not None:
            order.price = decision.adjusted_price
        if decision.adjusted_size is not None:
            order.size = decision.adjusted_size

        # Sample submit latency now so it's deterministic w.r.t. time of submit.
        # Pass the submit timestamp so multi-leg orders fired in the same
        # scheduling tick share a single network-path draw (correlated
        # latency); single-leg strategies are unaffected.
        latency_ms = self.latency.sample_submit_ms(at=order.submitted_at)
        order.venue_received_at = order.submitted_at + timedelta(milliseconds=latency_ms)
        order.state = OrderState.PENDING
        self._orders[order.order_id] = order
        self._fills_by_order[order.order_id] = []
        # Track this order in the per-token active-count index so the
        # engine's fast-path doesn't skip its token's next snapshot.
        self._active_count_by_token[order.token_id] = (
            self._active_count_by_token.get(order.token_id, 0) + 1
        )
        return order

    def _decrement_active(self, token_id: str) -> None:
        """Drop the active-count by 1 for a token that just had an order
        transition to a terminal state.  Removes the dict key when count
        hits zero so memory doesn't grow with completed runs.
        """
        n = self._active_count_by_token.get(token_id, 0) - 1
        if n <= 0:
            self._active_count_by_token.pop(token_id, None)
        else:
            self._active_count_by_token[token_id] = n

    def _terminate(self, order: "BacktestOrder", new_state: "OrderState") -> None:
        """Centralized terminal transition: set state + decrement the
        per-token active-count index.  Idempotent — calling twice is
        safe (the second call is a no-op).
        """
        if order.is_terminal:
            return
        order.state = new_state
        self._decrement_active(order.token_id)

    def has_active_orders_for_token(self, token_id: str) -> bool:
        """True if any order on this token is in a non-terminal state.
        Cheap O(1) lookup; used by the engine's fast-path bail.
        """
        return self._active_count_by_token.get(token_id, 0) > 0

    def cancel(self, *, order_id: str, requested_at: datetime) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.is_terminal:
            return False
        if order_id in self._pending_cancels:
            return True
        latency_ms = self.latency.sample_cancel_ms(at=requested_at)
        effective = requested_at + timedelta(milliseconds=latency_ms)
        self._pending_cancels[order_id] = effective
        order.cancel_pending_at = effective
        return True

    def advance_to(self, snapshot: BookSnapshot) -> list[Fill]:
        """Advance simulated time to ``snapshot.observed_at``.

        Steps:
          1. Apply any cancels whose latency window has now elapsed.
          2. Refresh the residual book for this token from the new
             snapshot. Liquidity consumed in the prior interval is wiped.
          3. Process orders whose ``venue_received_at`` is at-or-before
             the new time AND that are still pending.
          4. Re-evaluate working orders (they may have become marketable
             because the book moved into them).

        Returns the fills produced during this advance, in order.
        """
        now = snapshot.observed_at
        if self._now is not None and now < self._now:
            raise ValueError(
                f"non-monotonic advance: {now} < {self._now}"
            )
        self._now = now

        produced: list[Fill] = []

        # 1. Apply due cancels (before book update so we don't fill an
        #    order in the same tick we requested its cancel for).
        for order_id, effective_at in list(self._pending_cancels.items()):
            if effective_at <= now:
                self._apply_cancel(order_id, effective_at)
                self._pending_cancels.pop(order_id, None)

        # 2. Reset residual book for this token to the new snapshot
        self._books[snapshot.token_id] = _ResidualBook(snapshot=snapshot)

        # 3. Process orders newly admitted to the venue
        for order in list(self._orders.values()):
            if order.token_id != snapshot.token_id:
                continue
            if order.state != OrderState.PENDING:
                continue
            if order.venue_received_at is None:
                continue
            if order.venue_received_at > now:
                continue
            self._handle_admit(order, produced)

        # 4. Re-evaluate working orders against the updated book
        for order_id in list(self._working_by_token.get(snapshot.token_id, [])):
            order = self._orders.get(order_id)
            if order is None or order.is_terminal:
                continue
            self._try_match_resting(order, produced)

        self._all_fills.extend(produced)
        return produced

    def working_orders(self, token_id: Optional[str] = None) -> list[BacktestOrder]:
        if token_id is None:
            return [
                o for o in self._orders.values()
                if o.state in {OrderState.WORKING, OrderState.PARTIAL}
            ]
        ids = self._working_by_token.get(token_id, [])
        return [
            self._orders[i]
            for i in ids
            if i in self._orders and self._orders[i].state in {OrderState.WORKING, OrderState.PARTIAL}
        ]

    def order(self, order_id: str) -> Optional[BacktestOrder]:
        return self._orders.get(order_id)

    def all_orders(self) -> list[BacktestOrder]:
        return list(self._orders.values())

    def all_fills(self) -> list[Fill]:
        return list(self._all_fills)

    # ── Internals ─────────────────────────────────────────────────────

    def _apply_cancel(self, order_id: str, effective_at: datetime) -> None:
        order = self._orders.get(order_id)
        if order is None or order.is_terminal:
            return
        if order.state == OrderState.PARTIAL or order.state == OrderState.WORKING:
            self._terminate(order, OrderState.CANCELLED)
            order.cancelled_at = effective_at
            self._remove_from_working(order)
        elif order.state == OrderState.PENDING:
            # Cancelled before the venue received it — never made the book.
            self._terminate(order, OrderState.CANCELLED)
            order.cancelled_at = effective_at

    def _remove_from_working(self, order: BacktestOrder) -> None:
        ids = self._working_by_token.get(order.token_id) or []
        if order.order_id in ids:
            ids.remove(order.order_id)
            self._working_by_token[order.token_id] = ids

    def _handle_admit(self, order: BacktestOrder, produced: list[Fill]) -> None:
        """Process an order at the moment the venue accepts it.

        Decision matrix:
          * post_only and would cross book → reject
          * IOC / FAK → match what we can, cancel the remainder
          * FOK → match the full size atomically or reject
          * GTC / GTD → match what we can, rest the remainder
        """
        book = self._books.get(order.token_id)
        if book is None:
            # No snapshot for this token — defer; this should be rare but
            # can happen if a strategy submits before the book has any
            # observations. Park the order and re-attempt on next advance.
            return
        snap = book.snapshot

        if order.post_only and self.venue.crosses_book(
            side=order.side,
            price=order.price,
            best_bid=snap.best_bid,
            best_ask=snap.best_ask,
        ):
            order.reject_reason = "post_only_crosses_book"
            self._terminate(order, OrderState.REJECTED)
            return

        marketable = self.venue.is_marketable(
            side=order.side,
            price=order.price,
            best_bid=snap.best_bid,
            best_ask=snap.best_ask,
        )

        tif = (order.tif or TIF_GTC).upper()

        if tif == TIF_FOK:
            available = self._available_for_taker(book, order.side, order.price)
            if available + 1e-12 < order.size:
                order.reject_reason = "fok_insufficient_liquidity"
                self._terminate(order, OrderState.REJECTED)
                return
            self._consume_levels(order, book, produced)
            self._terminate(order, OrderState.FILLED)
            return

        if marketable:
            self._consume_levels(order, book, produced)
            if order.remaining_size > 0:
                if tif in {TIF_IOC, TIF_FAK}:
                    order.cancelled_at = order.venue_received_at
                    self._terminate(order, OrderState.CANCELLED)
                else:
                    # PARTIAL is non-terminal; no decrement.
                    order.state = OrderState.PARTIAL
                    self._add_to_working(order)
            else:
                self._terminate(order, OrderState.FILLED)
            return

        # Not marketable. IOC/FAK with no fill → cancel; GTC/GTD → rest.
        if tif in {TIF_IOC, TIF_FAK}:
            order.cancelled_at = order.venue_received_at
            self._terminate(order, OrderState.CANCELLED)
        else:
            # WORKING is non-terminal; no decrement.
            order.state = OrderState.WORKING
            self._add_to_working(order)

    def _add_to_working(self, order: BacktestOrder) -> None:
        ids = self._working_by_token.setdefault(order.token_id, [])
        if order.order_id not in ids:
            ids.append(order.order_id)

    def _available_for_taker(
        self, book: _ResidualBook, side: str, price: float
    ) -> float:
        """Total accessible size if a taker on ``side`` walks the book up
        to ``price``.
        """
        side_norm = (side or "").upper()
        if side_norm == "SELL":
            return sum(
                qty for (p, qty) in book.remaining_bids() if p >= price - 1e-12
            )
        if side_norm == "BUY":
            return sum(
                qty for (p, qty) in book.remaining_asks() if p <= price + 1e-12
            )
        return 0.0

    def _consume_levels(
        self,
        order: BacktestOrder,
        book: _ResidualBook,
        produced: list[Fill],
    ) -> None:
        """Walk the residual book and produce fills until ``order.remaining_size``
        is exhausted or no more levels match the limit.
        """
        side_norm = (order.side or "").upper()
        levels = book.remaining_bids() if side_norm == "SELL" else book.remaining_asks()
        remaining = order.remaining_size
        if remaining <= 0:
            return
        # Pre-compute total visible depth on the consumed side for the
        # impact model.  Excludes levels worse than the limit (the
        # taker can't actually reach those).
        total_visible_depth = 0.0
        for lvl_p, lvl_q in levels:
            if lvl_q <= 0:
                continue
            if side_norm == "SELL" and lvl_p + 1e-12 < order.price:
                continue
            if side_norm == "BUY" and lvl_p - 1e-12 > order.price:
                continue
            total_visible_depth += lvl_q
        impact_bps = self.impact.adverse_bps(
            your_size_shares=order.remaining_size,
            side_visible_depth_shares=total_visible_depth,
        )
        impact_factor = impact_bps / 10_000.0  # fraction of price
        for price, qty in levels:
            if remaining <= 0:
                break
            if qty <= 0:
                continue
            if side_norm == "SELL" and price + 1e-12 < order.price:
                break
            if side_norm == "BUY" and price - 1e-12 > order.price:
                break
            take = min(qty, remaining)
            # Apply adverse-direction impact: BUY pays more, SELL gets less.
            adjusted_price = price
            if impact_factor > 0:
                if side_norm == "BUY":
                    adjusted_price = min(0.9999, price * (1.0 + impact_factor))
                else:
                    adjusted_price = max(0.0001, price * (1.0 - impact_factor))
            fill = Fill(
                order_id=order.order_id,
                token_id=order.token_id,
                side=side_norm,
                price=adjusted_price,
                size=take,
                fee_usd=self.fees.fill_fee(
                    price=adjusted_price, size=take, is_maker=False,
                    mid_price=book.snapshot.mid if book.snapshot else None,
                ),
                occurred_at=order.venue_received_at or self._now or order.submitted_at,
                fill_index=len(self._fills_by_order[order.order_id]),
                venue_sequence=book.snapshot.sequence,
                notes={"impact_bps": impact_bps} if impact_bps > 0 else {},
            )
            produced.append(fill)
            order.fills.append(fill)
            self._fills_by_order[order.order_id].append(fill)
            new_filled = order.filled_size + take
            order.avg_fill_price = (
                (order.avg_fill_price * order.filled_size + adjusted_price * take) / new_filled
                if new_filled > 0
                else adjusted_price
            )
            order.filled_size = new_filled
            # Book consumption tracks displayed depth at the original
            # ladder price — the impact model adjusts what WE pay, not
            # what the resting maker received.
            book.consume(side_taker=side_norm, price=price, size=take)
            remaining -= take

    def _try_match_resting(self, order: BacktestOrder, produced: list[Fill]) -> None:
        """Re-evaluate a working/partial order against the latest snapshot.

        Resting orders can fill when the book moves into them (e.g., a
        SELL limit at 0.51 becomes marketable when the bid rises to 0.51+).
        We treat them as makers — a market order that consumed bid 0.51
        will have already updated the snapshot's best_bid; we simulate
        their match by checking the *current* snapshot's opposing side
        against the resting price.
        """
        book = self._books.get(order.token_id)
        if book is None:
            return
        snap = book.snapshot
        side_norm = (order.side or "").upper()
        remaining = order.remaining_size
        if remaining <= 0:
            return

        if side_norm == "SELL":
            # Resting sell at price P fills if the bid side now reaches P.
            for price, qty in book.remaining_bids():
                if remaining <= 0:
                    break
                if price + 1e-12 < order.price:
                    break
                take = min(qty, remaining)
                if take <= 0:
                    continue
                self._record_resting_fill(order, book, snap, take, price=order.price, produced=produced)
                remaining -= take
        elif side_norm == "BUY":
            for price, qty in book.remaining_asks():
                if remaining <= 0:
                    break
                if price - 1e-12 > order.price:
                    break
                take = min(qty, remaining)
                if take <= 0:
                    continue
                self._record_resting_fill(order, book, snap, take, price=order.price, produced=produced)
                remaining -= take

        if order.remaining_size <= 1e-12 and order.state != OrderState.FILLED:
            self._terminate(order, OrderState.FILLED)
            self._remove_from_working(order)

    def _record_resting_fill(
        self,
        order: BacktestOrder,
        book: _ResidualBook,
        snap: BookSnapshot,
        size: float,
        *,
        price: float,
        produced: list[Fill],
    ) -> None:
        """Record a fill for a resting order. Resting orders fill at their
        own limit price (they're the maker); the venue model treats them
        as maker so fees use the maker schedule.
        """
        fill = Fill(
            order_id=order.order_id,
            token_id=order.token_id,
            side=(order.side or "").upper(),
            price=price,
            size=size,
            fee_usd=self.fees.fill_fee(
                price=price, size=size, is_maker=True,
                mid_price=snap.mid if snap else None,
            ),
            occurred_at=snap.observed_at,
            fill_index=len(self._fills_by_order[order.order_id]),
            venue_sequence=snap.sequence,
            notes={"maker": True},
        )
        produced.append(fill)
        order.fills.append(fill)
        self._fills_by_order[order.order_id].append(fill)
        new_filled = order.filled_size + size
        order.avg_fill_price = (
            (order.avg_fill_price * order.filled_size + price * size) / new_filled
            if new_filled > 0
            else price
        )
        order.filled_size = new_filled
        # Mark the consumed liquidity on our residual book so other resting
        # orders at the same level don't double-count.
        side_norm = (order.side or "").upper()
        # Resting sell fills against bids; consume the bid at top.
        consume_side = "SELL" if side_norm == "SELL" else "BUY"
        # Use the resting price as the consumption point — it's the level
        # at which incoming aggressors paid.
        # For a partial-fill rest, we consume against the level that
        # provided ``size``. The book.remaining_bids() iterator above
        # passed us the level price; we approximate by using the resting
        # order's own price (the maker's price).
        book.consume(side_taker=consume_side, price=price, size=size)


def make_order_id() -> str:
    return f"bo_{uuid.uuid4().hex[:12]}"


__all__ = [
    "BacktestOrder",
    "Fill",
    "FeeModel",
    "MatchingEngine",
    "OrderState",
    "make_order_id",
]
