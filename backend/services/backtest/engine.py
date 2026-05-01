"""Top-level backtest engine: ties book replay, matching, portfolio,
and the exit_executor's laddered-exit planner into a deterministic
strategy-fitness simulator.

Inputs
======

* ``trade_intents``: a chronological list of ``TradeIntent`` records that
  represent strategy entries (the upstream DETECT + EVALUATE pipeline
  produces these). Each intent carries (token, side, size, limit price,
  TIF, post_only, strategy slug, opening trigger).
* ``strategy``: a ``BaseStrategy`` instance whose ``should_exit`` is
  invoked on each snapshot for every open position. Strategies that
  attach an ``ExitPolicy`` (per-decision or via ``exit_policies``) get
  the laddered/chunked execution path.
* ``book_replay``: any object with ``iter_snapshots`` returning
  chronological ``BookSnapshot`` objects (DB or in-memory).

Output
======

``BacktestResult`` with the equity history, full fill ledger, closed-
position summary, and ``BacktestMetrics`` (with bootstrap CIs).

Determinism
===========

Given identical inputs (same intents, same strategy state, same latency
seed, same book replay), the engine produces identical fills and metrics.
The matching engine uses an injectable ``LatencyModel`` whose RNG is
seeded; every other component is pure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, Protocol, Sequence

from services.backtest.book_replay import BookSnapshot
from services.backtest.latency_model import LatencyModel
from services.backtest.matching_engine import (
    BacktestOrder,
    Fill,
    FeeModel,
    ImpactModel,
    MatchingEngine,
    OrderState,
    make_order_id,
)
from services.backtest.metrics import (
    BacktestMetrics,
    TradeOutcome,
    compute_metrics,
)
from services.backtest.portfolio import Portfolio, PortfolioConfig
from services.backtest.venue_model import (
    PolymarketVenue,
    Venue,
    TIF_GTC,
    TIF_IOC,
)

logger = logging.getLogger(__name__)


# ── Inputs / outputs ────────────────────────────────────────────────────


@dataclass
class TradeIntent:
    """An entry signal emitted by the DETECT + EVALUATE pipeline."""

    intent_id: str
    emitted_at: datetime
    token_id: str
    side: str  # BUY (or SELL for shorts on supporting venues)
    size: float
    limit_price: float
    tif: str = TIF_GTC
    post_only: bool = False
    strategy_slug: Optional[str] = None
    trader_id: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    latency: Optional[LatencyModel] = None
    fees: Optional[FeeModel] = None
    venue: Optional[Venue] = None
    impact: Optional["ImpactModel"] = None
    final_close_at_last_mid: bool = True  # mark-to-market unclosed positions
    max_force_exit_attempts: int = 3
    seed: Optional[int] = 42
    log_progress_every: int = 0  # 0 = silent


@dataclass
class BacktestResult:
    config: BacktestConfig
    final_equity_usd: float
    initial_capital_usd: float
    metrics: BacktestMetrics
    closed_position_count: int
    open_position_count: int
    total_fills: int
    rejected_orders: int
    cancelled_orders: int
    equity_history: list[tuple[datetime, float]] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    trade_outcomes: list[TradeOutcome] = field(default_factory=list)
    correlation_matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    fees_per_fill_usd: float = 0.0
    fees_resolution_usd: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)


class _BookSource(Protocol):
    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]: ...
    async def snapshot_at(
        self, *, token_id: str, ts: datetime
    ) -> Optional[BookSnapshot]: ...


# ── Engine ───────────────────────────────────────────────────────────────


class BacktestEngine:
    """Drives a single backtest run from snapshots and trade intents."""

    def __init__(
        self,
        *,
        config: Optional[BacktestConfig] = None,
        strategy: Optional[Any] = None,
    ):
        self.config = config or BacktestConfig()
        self.strategy = strategy
        # Single FeeModel instance is shared between matching engine
        # (per-fill fees) and portfolio (resolution-time fees) so both
        # see consistent parameters.
        fees = self.config.fees or FeeModel()
        portfolio_config = self.config.portfolio
        if portfolio_config.fee_model is None:
            portfolio_config = replace(portfolio_config, fee_model=fees)
        self.portfolio = Portfolio(portfolio_config)
        self.matching = MatchingEngine(
            venue=self.config.venue or PolymarketVenue(),
            latency=self.config.latency or LatencyModel(seed=self.config.seed),
            fees=fees,
            impact=self.config.impact or ImpactModel(),
        )
        # Snapshot of the "current best" book per token, used by the
        # exit-decision hook to feed market_state into should_exit().
        self._latest_book: dict[str, BookSnapshot] = {}
        # Indexed view of pending intents, keyed by emitted_at, drained
        # as time advances.
        self._pending_intents: list[TradeIntent] = []
        self._intents_drained = 0
        # Maps (position_key, child_id) → matching engine order_id.
        self._exit_orders_by_position: dict[tuple, list[str]] = {}
        # Snapshot count for periodic progress logging.
        self._snapshots_processed = 0

    # ── Public driver ─────────────────────────────────────────────────

    async def run(
        self,
        *,
        book_source: _BookSource,
        trade_intents: Sequence[TradeIntent],
    ) -> BacktestResult:
        self._pending_intents = sorted(trade_intents, key=lambda t: t.emitted_at)
        self._intents_drained = 0
        self._snapshots_processed = 0

        async for snapshot in book_source.iter_snapshots():
            await self._on_snapshot(snapshot)
            self._snapshots_processed += 1
            if (
                self.config.log_progress_every > 0
                and self._snapshots_processed % self.config.log_progress_every == 0
            ):
                logger.info(
                    "backtest progress: %d snapshots, %d open, equity=%.2f",
                    self._snapshots_processed,
                    self.portfolio.open_position_count(),
                    self.portfolio.equity_usd(),
                )

        if self.config.final_close_at_last_mid:
            self._final_mark_to_market()

        return self._build_result()

    # ── Per-snapshot work ─────────────────────────────────────────────

    async def _on_snapshot(self, snapshot: BookSnapshot) -> None:
        # 1. Drain any pending intents whose emit time has now passed.
        while (
            self._intents_drained < len(self._pending_intents)
            and self._pending_intents[self._intents_drained].emitted_at <= snapshot.observed_at
        ):
            intent = self._pending_intents[self._intents_drained]
            self._intents_drained += 1
            self._submit_entry(intent, snapshot)

        # 2. Advance the matching engine — this picks up newly-admitted
        #    orders and re-evaluates resting orders against the new book.
        fills = self.matching.advance_to(snapshot)

        # 3. Apply fills to the portfolio (cash + positions update).
        for fill in fills:
            order = self.matching.order(fill.order_id)
            if order is None:
                continue
            self.portfolio.apply_fill(fill, strategy_slug=order.strategy_slug)

        # 4. Mark portfolio at the new mid (if available).
        self._latest_book[snapshot.token_id] = snapshot
        if snapshot.mid is not None:
            self.portfolio.mark(
                token_id=snapshot.token_id,
                price=float(snapshot.mid),
                at=snapshot.observed_at,
            )

        # 5. Run exit decisions on every position currently held in this
        #    token. We do this *after* applying fills so a position that
        #    just opened can also have its exit logic evaluated this tick
        #    if appropriate (matches live behavior).
        if self.strategy is not None:
            self._evaluate_exits_for_token(snapshot)

    def _submit_entry(self, intent: TradeIntent, snapshot: BookSnapshot) -> None:
        ok, reason = self.portfolio.can_submit(
            token_id=intent.token_id,
            side=intent.side,
            price=intent.limit_price,
            size=intent.size,
            strategy_slug=intent.strategy_slug,
        )
        if not ok:
            logger.debug(
                "entry intent rejected at portfolio gate: %s (intent=%s)",
                reason,
                intent.intent_id,
            )
            return
        order = BacktestOrder(
            order_id=intent.intent_id or make_order_id(),
            token_id=intent.token_id,
            side=intent.side,
            price=float(intent.limit_price),
            size=float(intent.size),
            tif=intent.tif,
            post_only=bool(intent.post_only),
            submitted_at=intent.emitted_at,
            trader_id=intent.trader_id,
            strategy_slug=intent.strategy_slug,
            meta={"role": "entry", **(intent.meta or {})},
        )
        self.matching.submit(order)

    # ── Exit decision + laddered execution ────────────────────────────

    def _evaluate_exits_for_token(self, snapshot: BookSnapshot) -> None:
        """For each open position whose token matches this snapshot, run
        ``strategy.should_exit`` and route the resulting decision through
        the exit-execution path (single order or laddered).
        """
        token_id = snapshot.token_id
        # Collect positions to evaluate (avoid mutating during iteration)
        targets = [
            (key, pos)
            for key, pos in list(self.portfolio.positions.items())
            if key[0] == token_id and pos.size > 1e-12
        ]
        for key, pos in targets:
            decision = self._call_should_exit(pos, snapshot)
            if decision is None:
                continue
            action = getattr(decision, "action", None)
            if action == "hold":
                continue
            if action == "reduce":
                fraction = float(getattr(decision, "reduce_fraction", 0.0) or 0.0)
                if fraction <= 0.0:
                    continue
                self._submit_exit_for(
                    position=pos,
                    decision=decision,
                    fraction=min(1.0, fraction),
                    snapshot=snapshot,
                )
                continue
            if action == "close":
                self._submit_exit_for(
                    position=pos,
                    decision=decision,
                    fraction=1.0,
                    snapshot=snapshot,
                )

    def _call_should_exit(self, pos, snapshot: BookSnapshot) -> Optional[Any]:
        """Run the strategy's exit hook with a position view + market state."""
        if self.strategy is None or not hasattr(self.strategy, "should_exit"):
            return None
        # Lightweight position view that mirrors what live trading passes.
        class _BTPosView:
            pass
        view = _BTPosView()
        view.entry_price = pos.entry_price
        view.current_price = float(snapshot.mid) if snapshot.mid is not None else pos.entry_price
        view.highest_price = max(pos.entry_price, float(snapshot.mid or pos.entry_price))
        view.lowest_price = min(pos.entry_price, float(snapshot.mid or pos.entry_price))
        # Age in minutes since position open
        if pos.opened_at is not None:
            view.age_minutes = max(
                0.0, (snapshot.observed_at - pos.opened_at).total_seconds() / 60.0
            )
        else:
            view.age_minutes = 0.0
        if pos.entry_price > 0:
            view.pnl_percent = (
                (view.current_price - pos.entry_price) / pos.entry_price * 100.0
            )
        else:
            view.pnl_percent = 0.0
        view.filled_size = pos.size
        view.notional_usd = pos.size * pos.entry_price
        view.strategy_context = pos and {} or {}
        view.config = {}
        view.outcome_idx = 0

        market_state = {
            "current_price": view.current_price,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "token_id": pos.token_id,
        }
        try:
            return self.strategy.should_exit(view, market_state)
        except Exception as exc:
            logger.warning(
                "strategy.should_exit raised in backtest: %s", exc, exc_info=exc
            )
            return None

    def _submit_exit_for(
        self,
        *,
        position,
        decision: Any,
        fraction: float,
        snapshot: BookSnapshot,
    ) -> None:
        """Convert an ExitDecision into one or more BacktestOrders.

        If the decision (or the strategy's ``exit_policies`` map) supplies
        an ``ExitPolicy``, this delegates to ``exit_executor.plan_children``
        to produce a laddered child plan that we submit individually.
        Otherwise we submit a single closing order at the decision price.
        """
        target_size = position.size * fraction
        if target_size <= 0:
            return
        close_price = (
            float(getattr(decision, "close_price", None) or snapshot.mid or position.entry_price)
        )
        close_trigger = str(getattr(decision, "reason", "exit") or "exit")
        # Side that closes the position
        close_side = "SELL" if position.side == "BUY" else "BUY"

        policy = self._resolve_exit_policy(decision, close_trigger)
        if policy is None:
            self._submit_single_exit(
                position=position,
                token_id=position.token_id,
                side=close_side,
                size=target_size,
                price=close_price,
                tif=TIF_IOC,
                post_only=False,
                snapshot=snapshot,
                role="exit",
            )
            return

        # Laddered exit — defer to the live planner so the same logic runs
        # in backtest as in production.
        from services.trader_orchestrator import exit_executor

        # Exit ladders are intentionally aggressive: their offset_ticks pulls
        # the inside-most rung past the trigger so each child is marketable
        # on submit. Default the planner to ``post_only=False`` so children
        # don't get post-only-crosses-book rejected by the venue model. For
        # specifically-passive exit policies (e.g., a take-profit limit
        # waiting for the book to come up), set chunk size and offset_ticks=0
        # in the policy and use a separate code path with post_only=True.
        plans = exit_executor.plan_children(
            target_size=target_size,
            trigger_price=close_price,
            side=close_side,
            policy=policy,
            tick_size=exit_executor.DEFAULT_TICK_SIZE,
            default_post_only=False,
            default_tif=TIF_IOC if close_trigger == "stop_loss" else TIF_GTC,
        )
        if not plans:
            self._submit_single_exit(
                position=position,
                token_id=position.token_id,
                side=close_side,
                size=target_size,
                price=close_price,
                tif=TIF_IOC,
                post_only=False,
                snapshot=snapshot,
                role="exit_fallback",
            )
            return

        for plan in plans:
            self._submit_single_exit(
                position=position,
                token_id=position.token_id,
                side=close_side,
                size=plan.size,
                price=plan.price,
                tif=plan.tif,
                post_only=plan.post_only,
                snapshot=snapshot,
                role="exit_child",
                child_meta={
                    "child_index": plan.index,
                    "bucket": plan.distribution_bucket,
                },
            )

    def _resolve_exit_policy(self, decision: Any, close_trigger: str):
        """Resolve an ExitPolicy from (decision override, strategy map)."""
        if self.strategy is not None and hasattr(self.strategy, "resolve_exit_policy"):
            try:
                return self.strategy.resolve_exit_policy(decision, close_trigger)
            except Exception:
                return None
        # Fallback: just look at the decision-level field.
        return getattr(decision, "exit_policy", None)

    def _submit_single_exit(
        self,
        *,
        position,
        token_id: str,
        side: str,
        size: float,
        price: float,
        tif: str,
        post_only: bool,
        snapshot: BookSnapshot,
        role: str,
        child_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        order = BacktestOrder(
            order_id=make_order_id(),
            token_id=token_id,
            side=side,
            price=float(price),
            size=float(size),
            tif=tif,
            post_only=post_only,
            submitted_at=snapshot.observed_at,
            trader_id=None,
            strategy_slug=position.strategy_slug,
            meta={"role": role, "exit_for_position": position.side, **(child_meta or {})},
        )
        self.matching.submit(order)

    # ── Finalization ──────────────────────────────────────────────────

    def _final_mark_to_market(self) -> None:
        """Force-close remaining open positions at the last seen mid.

        Production positions resolve at expiration (1.0 or 0.0); for a
        backtest that ends mid-life we mark to the last observed mid.
        This is a *book-keeping close* and does not produce fills — it
        just realizes the unrealized PnL into the ledger. Resolution
        fees apply to any winning positions, mirroring live behavior.
        """
        fee_model = self.portfolio.config.fee_model
        for key, pos in list(self.portfolio.positions.items()):
            if pos.size <= 1e-12:
                continue
            mark = pos.last_mark_price or pos.entry_price
            close_at = pos.last_mark_at or pos.opened_at or datetime.now(timezone.utc)
            synthetic_fill = Fill(
                order_id=f"final_mtm_{key[0]}",
                token_id=pos.token_id,
                side=("SELL" if pos.side == "BUY" else "BUY"),
                price=float(mark),
                size=pos.size,
                fee_usd=0.0,
                occurred_at=close_at,
                fill_index=0,
                notes={"final_mark_to_market": True},
            )
            prior_realized = pos.realized_pnl_usd
            pos.apply_close_fill(synthetic_fill)
            self.portfolio._on_cash_change_from_close(pos, synthetic_fill)
            if fee_model is not None:
                gained = pos.realized_pnl_usd - prior_realized
                if gained > 0:
                    res_fee = fee_model.resolution_fee(gross_winnings_usd=gained)
                    if res_fee > 0:
                        pos.fees_paid_usd += res_fee
                        pos.realized_pnl_usd -= res_fee
                        self.portfolio.cash_usd -= res_fee
                        self.portfolio._resolution_fees_paid_usd += res_fee
            self.portfolio.closed_positions.append(pos)
            del self.portfolio.positions[key]

    def _build_result(self) -> BacktestResult:
        all_fills = self.matching.all_fills()
        rejected = sum(
            1 for o in self.matching.all_orders() if o.state == OrderState.REJECTED
        )
        cancelled = sum(
            1 for o in self.matching.all_orders() if o.state == OrderState.CANCELLED
        )
        # Build per-trade outcome list from closed positions
        outcomes: list[TradeOutcome] = []
        for pos in self.portfolio.closed_positions:
            entry_notional = max(0.0, pos.entry_price * (pos.size + pos.realized_pnl_usd / max(0.0001, pos.entry_price)))
            # Use cost basis at open as the denominator for return %
            denom = pos.cost_basis_usd + pos.realized_pnl_usd if pos.cost_basis_usd > 0 else max(1e-9, entry_notional)
            return_pct = (pos.realized_pnl_usd / denom) * 100.0 if denom > 0 else 0.0
            holding_s = 0.0
            if pos.opened_at and pos.closed_at:
                holding_s = max(0.0, (pos.closed_at - pos.opened_at).total_seconds())
            outcomes.append(
                TradeOutcome(
                    pnl_usd=pos.realized_pnl_usd,
                    return_pct=return_pct,
                    holding_seconds=holding_s,
                    won=pos.realized_pnl_usd > 0,
                )
            )

        metrics = compute_metrics(
            initial_capital_usd=self.config.portfolio.initial_capital_usd,
            final_equity_usd=self.portfolio.equity_usd(),
            equity_history=self.portfolio.equity_history,
            trades=outcomes,
            fees_paid_usd=self.portfolio.fees_paid_usd(),
            seed=self.config.seed,
        )

        return BacktestResult(
            config=self.config,
            final_equity_usd=self.portfolio.equity_usd(),
            initial_capital_usd=self.config.portfolio.initial_capital_usd,
            metrics=metrics,
            closed_position_count=self.portfolio.closed_position_count(),
            open_position_count=self.portfolio.open_position_count(),
            total_fills=len(all_fills),
            rejected_orders=rejected,
            cancelled_orders=cancelled,
            equity_history=list(self.portfolio.equity_history),
            fills=all_fills,
            trade_outcomes=outcomes,
            correlation_matrix=self.portfolio.correlation_matrix(),
            fees_per_fill_usd=self.portfolio.per_fill_fees_paid_usd(),
            fees_resolution_usd=self.portfolio.resolution_fees_paid_usd,
            notes={"snapshots_processed": self._snapshots_processed},
        )


__all__ = [
    "TradeIntent",
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
]
