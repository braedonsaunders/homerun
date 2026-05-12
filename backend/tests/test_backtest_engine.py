"""Tests for the production-grade backtester (services.backtest).

Covers each module in isolation plus an end-to-end run against a synthetic
order-book tape that exercises:

* Entry intent submission with portfolio capital gate.
* L2-aware fills with partial fills and book consumption.
* Post-only crosses-book rejection.
* IOC / FOK / GTC TIF semantics.
* Cancel/replace lifecycle with sampled latency.
* Laddered exits via the live ``exit_executor.plan_children``.
* Walk-forward split (rolling and anchored).
* Bootstrap CIs on Sharpe, hit-rate, profit-factor.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest

from services.backtest import (
    BacktestConfig,
    BacktestEngine,
    BookSnapshot,
    InMemoryBookReplay,
    LatencyModel,
    LatencyProfile,
    MatchingEngine,
    BacktestOrder,
    OrderState,
    PolymarketVenue,
    Portfolio,
    PortfolioConfig,
    PriceLevel,
    TradeIntent,
    bootstrap_ci,
    compute_metrics,
    walk_forward_split,
    WalkForwardConfig,
)
from services.backtest.matching_engine import FeeModel, make_order_id
from services.backtest.metrics import (
    TradeOutcome,
    max_drawdown,
)
from services.backtest.venue_model import TIF_FOK, TIF_GTC, TIF_IOC
from services.strategies.base import BaseStrategy, ExitDecision
from services.strategy_sdk import StrategySDK


T0 = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)


def _snap(t: datetime, *, bid: float, ask: float, depth: int = 5, qty: float = 50) -> BookSnapshot:
    return BookSnapshot(
        token_id="tok",
        observed_at=t,
        bids=tuple(PriceLevel(bid - 0.01 * k, qty) for k in range(depth)),
        asks=tuple(PriceLevel(ask + 0.01 * k, qty) for k in range(depth)),
    )


class _ConstantFillProbabilityModel:
    family = "cox_ph"
    strata_key = "test"
    n_events = 100
    concordance_index = 0.75

    def __init__(self, probability: float):
        self.probability = probability

    def evaluate(self, *, covariates, horizon_seconds):
        return self.probability


# ── Venue model ──────────────────────────────────────────────────────────


class TestVenueModel:
    def test_rounds_sell_down(self):
        v = PolymarketVenue()
        assert v.round_price(0.5079, side="SELL") == pytest.approx(0.50)

    def test_rounds_buy_up(self):
        v = PolymarketVenue()
        assert v.round_price(0.5021, side="BUY") == pytest.approx(0.51)

    def test_rejects_below_min_notional(self):
        v = PolymarketVenue()
        d = v.validate_submit(
            side="SELL", price=0.05, size=2, tif=TIF_GTC, post_only=False,
        )
        assert d.ok is False
        assert "min_notional" in (d.reject_reason or "")

    def test_rejects_post_only_with_taker_tif(self):
        v = PolymarketVenue()
        d = v.validate_submit(
            side="SELL", price=0.50, size=10, tif=TIF_IOC, post_only=True,
        )
        assert d.ok is False
        assert "post_only_with_taker_tif" in (d.reject_reason or "")

    def test_crosses_book_sell_at_or_below_bid(self):
        v = PolymarketVenue()
        assert v.crosses_book(side="SELL", price=0.50, best_bid=0.50, best_ask=0.51) is True
        assert v.crosses_book(side="SELL", price=0.51, best_bid=0.50, best_ask=0.51) is False


# ── Latency model ────────────────────────────────────────────────────────


class TestLatencyModel:
    def test_quantile_recovers_p50(self):
        prof = LatencyProfile.from_quantiles(p50_ms=350, p95_ms=900)
        assert prof.quantile(0.5) == pytest.approx(350.0, rel=0.01)

    def test_quantile_recovers_p95(self):
        prof = LatencyProfile.from_quantiles(p50_ms=350, p95_ms=900)
        assert prof.quantile(0.95) == pytest.approx(900.0, rel=0.01)

    def test_deterministic_returns_constant(self):
        m = LatencyModel.deterministic(submit_ms=400, cancel_ms=200)
        for _ in range(10):
            assert m.sample_submit_ms() == pytest.approx(400.0, rel=0.001)
            assert m.sample_cancel_ms() == pytest.approx(200.0, rel=0.001)


# ── Matching engine ──────────────────────────────────────────────────────


class TestMatchingEngine:
    def _engine(self, *, fill_model_snapshot=None) -> MatchingEngine:
        return MatchingEngine(
            venue=PolymarketVenue(),
            latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
            fees=FeeModel(),
            fill_model_snapshot=fill_model_snapshot,
        )

    def test_ioc_fills_marketable_size_and_cancels_remainder(self):
        me = self._engine()
        me.advance_to(_snap(T0, bid=0.50, ask=0.51, depth=2, qty=20))
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.49, size=80, tif=TIF_IOC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(_snap(T0 + timedelta(milliseconds=200), bid=0.50, ask=0.51, depth=2, qty=20))
        # Filled 20 at 0.50 + 20 at 0.49 = 40
        assert order.state == OrderState.CANCELLED
        assert order.filled_size == pytest.approx(40.0)

    def test_fok_full_fill_or_reject(self):
        me = self._engine()
        me.advance_to(_snap(T0, bid=0.50, ask=0.51, depth=2, qty=10))
        # Need 50, only 20 available → reject
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.40, size=50, tif=TIF_FOK, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(_snap(T0 + timedelta(milliseconds=200), bid=0.50, ask=0.51, depth=2, qty=10))
        assert order.state == OrderState.REJECTED
        assert order.reject_reason == "fok_insufficient_liquidity"

    def test_post_only_rejects_when_crosses(self):
        me = self._engine()
        me.advance_to(_snap(T0, bid=0.50, ask=0.51, depth=1, qty=20))
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.50, size=10, tif=TIF_GTC, post_only=True,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(_snap(T0 + timedelta(milliseconds=200), bid=0.50, ask=0.51, depth=1, qty=20))
        assert order.state == OrderState.REJECTED
        assert order.reject_reason == "post_only_crosses_book"

    def test_gtc_rests_then_fills_when_book_moves(self):
        me = self._engine()
        # Place a SELL limit at 0.52 — above the bid → rests
        me.advance_to(_snap(T0, bid=0.50, ask=0.51, depth=1, qty=20))
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.52, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(_snap(T0 + timedelta(milliseconds=200), bid=0.50, ask=0.51, depth=1, qty=20))
        assert order.state == OrderState.WORKING
        # Bid rises to 0.52 → resting sell now fills
        me.advance_to(_snap(T0 + timedelta(seconds=1), bid=0.52, ask=0.53, depth=1, qty=20))
        assert order.state == OrderState.FILLED
        assert order.filled_size == pytest.approx(10.0)

    def test_resting_queue_clears_from_same_level_depth_removed(self):
        me = self._engine()
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0,
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.51, 20.0), PriceLevel(0.52, 100.0)),
            )
        )
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.52, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(milliseconds=200),
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.51, 20.0), PriceLevel(0.52, 100.0)),
            )
        )
        assert order.state == OrderState.WORKING
        assert order.queue_ahead_shares == pytest.approx(85.0)

        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(seconds=1),
                bids=(PriceLevel(0.52, 20.0),),
                asks=(PriceLevel(0.52, 35.0),),
            )
        )
        assert order.state == OrderState.FILLED
        assert order.filled_size == pytest.approx(10.0)

    def test_resting_buy_queue_clears_from_same_level_depth_removed(self):
        me = self._engine()
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0,
                bids=(PriceLevel(0.48, 100.0), PriceLevel(0.47, 20.0)),
                asks=(PriceLevel(0.50, 20.0),),
            )
        )
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="BUY",
            price=0.48, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(milliseconds=200),
                bids=(PriceLevel(0.48, 100.0), PriceLevel(0.47, 20.0)),
                asks=(PriceLevel(0.50, 20.0),),
            )
        )
        assert order.state == OrderState.WORKING
        assert order.queue_ahead_shares == pytest.approx(65.0)

        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(seconds=1),
                bids=(PriceLevel(0.48, 35.0),),
                asks=(PriceLevel(0.48, 20.0),),
            )
        )
        assert order.state == OrderState.FILLED
        assert order.filled_size == pytest.approx(10.0)

    def test_resting_sell_queue_does_not_advance_on_better_level_reprice(self):
        me = self._engine()
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0,
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.51, 20.0), PriceLevel(0.52, 100.0)),
            )
        )
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.52, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(milliseconds=200),
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.51, 20.0), PriceLevel(0.52, 100.0)),
            )
        )
        assert order.queue_ahead_shares == pytest.approx(85.0)

        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(seconds=1),
                bids=(PriceLevel(0.52, 20.0),),
                asks=(PriceLevel(0.50, 20.0), PriceLevel(0.52, 100.0)),
            )
        )
        assert order.state == OrderState.WORKING
        assert order.queue_ahead_shares == pytest.approx(85.0)

    def test_resting_buy_queue_does_not_advance_on_better_level_reprice(self):
        me = self._engine()
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0,
                bids=(PriceLevel(0.48, 100.0), PriceLevel(0.47, 20.0)),
                asks=(PriceLevel(0.50, 20.0),),
            )
        )
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="BUY",
            price=0.48, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(milliseconds=200),
                bids=(PriceLevel(0.48, 100.0), PriceLevel(0.47, 20.0)),
                asks=(PriceLevel(0.50, 20.0),),
            )
        )
        assert order.queue_ahead_shares == pytest.approx(65.0)

        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(seconds=1),
                bids=(PriceLevel(0.49, 65.0), PriceLevel(0.48, 35.0)),
                asks=(PriceLevel(0.48, 20.0),),
            )
        )
        assert order.state == OrderState.WORKING
        assert order.queue_ahead_shares == pytest.approx(65.0)

    def test_fill_probability_model_blocks_resting_maker_fill(self):
        me = self._engine(fill_model_snapshot=_ConstantFillProbabilityModel(0.0))
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0,
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.53, 20.0),),
            )
        )
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.52, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(milliseconds=200),
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.53, 20.0),),
            )
        )
        assert order.survival_covariates is not None
        assert order.survival_covariates["queue_ahead_shares"] == pytest.approx(0.0)

        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(seconds=1),
                bids=(PriceLevel(0.52, 20.0),),
                asks=(PriceLevel(0.53, 20.0),),
            )
        )
        assert order.state == OrderState.WORKING
        assert order.filled_size == pytest.approx(0.0)
        assert me.all_fills() == []

    def test_fill_probability_model_caps_resting_maker_fill_size(self):
        me = self._engine(fill_model_snapshot=_ConstantFillProbabilityModel(0.5))
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0,
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.53, 20.0),),
            )
        )
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.52, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(milliseconds=200),
                bids=(PriceLevel(0.50, 20.0),),
                asks=(PriceLevel(0.53, 20.0),),
            )
        )
        me.advance_to(
            BookSnapshot(
                token_id="tok",
                observed_at=T0 + timedelta(seconds=1),
                bids=(PriceLevel(0.52, 20.0),),
                asks=(PriceLevel(0.53, 20.0),),
            )
        )
        fills = me.all_fills()
        assert order.state == OrderState.WORKING
        assert order.filled_size == pytest.approx(5.0)
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(5.0)
        assert fills[0].notes["fill_probability"] == pytest.approx(0.5)

    def test_cancel_with_latency_fills_in_window(self):
        me = self._engine()
        me.advance_to(_snap(T0, bid=0.50, ask=0.51, depth=1, qty=20))
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.52, size=10, tif=TIF_GTC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        # Order admits at T0 + 100ms; book moves into it BEFORE cancel takes effect
        me.advance_to(_snap(T0 + timedelta(milliseconds=200), bid=0.52, ask=0.53, depth=1, qty=20))
        # Order should fill before cancel even fires
        assert order.state == OrderState.FILLED


# ── Portfolio ────────────────────────────────────────────────────────────


class TestPortfolio:
    def test_capital_gate_blocks_oversize_buy(self):
        p = Portfolio(PortfolioConfig(initial_capital_usd=10.0))
        ok, reason = p.can_submit(
            token_id="t", side="BUY", price=0.50, size=100, strategy_slug="x",
        )
        assert ok is False
        assert "insufficient_cash" in (reason or "")

    def test_max_per_market_caps_concurrent_size(self):
        p = Portfolio(
            PortfolioConfig(initial_capital_usd=1000.0, max_per_market_notional_usd=20.0)
        )
        # First fill: BUY 30 at 0.50 = $15 — within $20 cap
        from services.backtest.matching_engine import Fill
        f = Fill(order_id="o1", token_id="t", side="BUY", price=0.50, size=30,
                 fee_usd=0.0, occurred_at=T0, fill_index=0)
        p.apply_fill(f, strategy_slug="x")
        ok, reason = p.can_submit(
            token_id="t", side="BUY", price=0.50, size=20, strategy_slug="x",
        )
        assert ok is False
        assert "per_market_cap" in (reason or "")

    def test_equity_usd_marks_open_buy_at_market_value(self):
        """Regression: opening a position must NOT drop equity by notional.

        Production bug: a +4.83% return run rendered "$1.00K → $556.90"
        on the equity chart with Sharpe -2.01.  Root cause: equity_usd()
        = cash + sum(unrealized_pnl), where unrealized_pnl was
        size*(mark-entry).  At open (or when mark==entry) unrealized=0,
        so equity dropped by the FULL cost basis until the position
        eventually closed and cash absorbed the proceeds.  This polluted
        equity_history and corrupted Sharpe / drawdown / Calmar.
        """
        from services.backtest.matching_engine import Fill
        p = Portfolio(PortfolioConfig(initial_capital_usd=1000.0))
        assert p.equity_usd() == pytest.approx(1000.0)

        # Open BUY 100 @ $0.50 = $50 notional outlay.
        p.apply_fill(
            Fill(order_id="o1", token_id="t", side="BUY", price=0.50, size=100,
                 fee_usd=0.0, occurred_at=T0, fill_index=0),
            strategy_slug="x",
        )
        # Cash dropped by $50 — but the position is WORTH $50, so net
        # wealth must remain $1000.  Old broken formula returned $950.
        assert p.cash_usd == pytest.approx(950.0)
        assert p.equity_usd() == pytest.approx(1000.0), (
            "freshly-opened position must contribute its market value, "
            "not zero — otherwise opening a position is misread as a loss"
        )

        # After mark at entry: still $1000 (no PnL yet).
        p.mark(token_id="t", price=0.50, at=T0 + timedelta(seconds=1))
        assert p.equity_usd() == pytest.approx(1000.0)

        # Mark moves up to $0.60: +$10 unrealized gain.
        p.mark(token_id="t", price=0.60, at=T0 + timedelta(seconds=2))
        assert p.equity_usd() == pytest.approx(1010.0)

        # Mark drops to $0.40: -$10 unrealized loss.
        p.mark(token_id="t", price=0.40, at=T0 + timedelta(seconds=3))
        assert p.equity_usd() == pytest.approx(990.0)

    def test_equity_usd_marks_open_short_at_buy_back_cost(self):
        """SELL (short) at $0.50 collects $50 cash but creates a $50
        liability.  Equity must remain at initial capital, not jump to
        $1050 (old formula's overstatement).
        """
        from services.backtest.matching_engine import Fill
        p = Portfolio(PortfolioConfig(initial_capital_usd=1000.0))

        p.apply_fill(
            Fill(order_id="o1", token_id="t", side="SELL", price=0.50, size=100,
                 fee_usd=0.0, occurred_at=T0, fill_index=0),
            strategy_slug="x",
        )
        # Cash went UP by $50 (entry proceeds).
        assert p.cash_usd == pytest.approx(1050.0)
        # But we OWE 100 contracts at current price ≈ $0.50, so equity
        # must remain $1000.  Old broken formula returned $1050.
        assert p.equity_usd() == pytest.approx(1000.0), (
            "freshly-opened short must subtract buy-back liability, "
            "not zero — otherwise shorting is misread as instant gain"
        )

        # Mark drops to $0.40 (favorable for short): +$10 gain.
        p.mark(token_id="t", price=0.40, at=T0 + timedelta(seconds=1))
        assert p.equity_usd() == pytest.approx(1010.0)

        # Mark rises to $0.60 (adverse): -$10 loss.
        p.mark(token_id="t", price=0.60, at=T0 + timedelta(seconds=2))
        assert p.equity_usd() == pytest.approx(990.0)


# ── Walk-forward ─────────────────────────────────────────────────────────


class TestWalkForward:
    def test_rolling_produces_n_folds(self):
        end = T0 + timedelta(days=30)
        windows = walk_forward_split(
            start=T0, end=end, config=WalkForwardConfig(mode="rolling", n_folds=5),
        )
        assert len(windows) == 5
        # Folds are chronological and non-overlapping in test windows
        for w in windows:
            assert w.train_end <= w.test_start
            assert w.train_start < w.train_end

    def test_anchored_train_starts_always_at_t0(self):
        end = T0 + timedelta(days=30)
        windows = walk_forward_split(
            start=T0, end=end, config=WalkForwardConfig(mode="anchored", n_folds=4, train_ratio=0.5),
        )
        assert all(w.train_start == T0 for w in windows)
        # Train end grows monotonically
        ends = [w.train_end for w in windows]
        assert ends == sorted(ends)


# ── Metrics + bootstrap ─────────────────────────────────────────────────


class TestMetrics:
    def test_bootstrap_ci_returns_none_for_small_samples(self):
        lo, hi = bootstrap_ci([1.0, 2.0, 3.0], statistic=lambda xs: sum(xs) / len(xs))
        assert lo is None and hi is None

    def test_bootstrap_ci_brackets_point_estimate(self):
        xs = [0.05, -0.02, 0.03, 0.01, -0.01, 0.04, 0.02, 0.06, -0.03, 0.05] * 5
        point = sum(xs) / len(xs)
        lo, hi = bootstrap_ci(xs, statistic=lambda xs_: sum(xs_) / len(xs_), seed=42)
        assert lo is not None and hi is not None
        assert lo <= point <= hi

    def test_bootstrap_ci_caps_large_sample_work(self):
        xs = [float(i % 17) for i in range(50_000)]
        calls = 0

        def statistic(xs_):
            nonlocal calls
            calls += 1
            return sum(xs_) / len(xs_)

        lo, hi = bootstrap_ci(xs, statistic=statistic, seed=42)
        assert lo is not None and hi is not None
        assert calls <= 125

    def test_max_drawdown_basic(self):
        eq = [
            (T0, 1000.0),
            (T0 + timedelta(seconds=1), 1100.0),
            (T0 + timedelta(seconds=2), 900.0),
            (T0 + timedelta(seconds=3), 950.0),
            (T0 + timedelta(seconds=4), 1200.0),
        ]
        dd_usd, dd_pct, dur = max_drawdown(eq)
        assert dd_usd == pytest.approx(200.0)
        assert dd_pct == pytest.approx(200.0 / 1100.0)

    def test_compute_metrics_full_pipeline(self):
        eq = [
            (T0 + timedelta(seconds=i), 1000.0 + i)
            for i in range(60)
        ]
        trades = [TradeOutcome(pnl_usd=2, return_pct=2, holding_seconds=30, won=True) for _ in range(8)]
        trades += [TradeOutcome(pnl_usd=-1, return_pct=-1, holding_seconds=30, won=False) for _ in range(4)]
        m = compute_metrics(
            initial_capital_usd=1000.0,
            final_equity_usd=1059.0,
            equity_history=eq,
            trades=trades,
            fees_paid_usd=0.5,
            seed=1,
        )
        assert m.trade_count == 12
        assert m.hit_rate.value == pytest.approx(8 / 12)
        assert m.profit_factor.value == pytest.approx(16 / 4)
        assert m.total_return_usd == pytest.approx(59.0)


# ── End-to-end engine tests ─────────────────────────────────────────────


def _build_drifting_book(seconds: int = 60) -> InMemoryBookReplay:
    snaps = []
    for i in range(seconds):
        bid = 0.50 - 0.002 * i
        ask = bid + 0.01
        snaps.append(_snap(T0 + timedelta(seconds=i), bid=bid, ask=ask, depth=10, qty=30))
    return InMemoryBookReplay(snaps)


class _LegacyExitStrategy(BaseStrategy):
    strategy_type = "legacy_test"
    name = "legacy"
    description = "single-order exit"
    def should_exit(self, position, market_state):
        if position.pnl_percent <= -5.0:
            return ExitDecision("close", "stop_loss", close_price=market_state["current_price"])
        return ExitDecision("hold", "no exit")


class _LadderExitStrategy(BaseStrategy):
    strategy_type = "ladder_test"
    name = "ladder"
    description = "laddered stop-loss"
    exit_policies = {
        "stop_loss": StrategySDK.build_ladder_exit_policy(
            levels=5, step_ticks=1, offset_ticks=2, chunk_size=10.0,
            distribution="back_loaded", escalation_after_seconds=None,
        ),
    }
    def should_exit(self, position, market_state):
        if position.pnl_percent <= -5.0:
            return ExitDecision("close", "stop_loss", close_price=market_state["current_price"])
        return ExitDecision("hold", "no exit")


class TestFees:
    def test_per_fill_gas_charged_on_each_fill(self):
        from services.backtest.matching_engine import FeeModel
        fees = FeeModel(per_fill_gas_usd=0.10, resolution_fee_rate=0.0)
        # 10 contracts × $0.50 with 5 contracts on top, 5 next level → 2 fills
        me = MatchingEngine(
            venue=PolymarketVenue(),
            latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
            fees=fees,
        )
        me.advance_to(_snap(T0, bid=0.50, ask=0.51, depth=2, qty=5))
        order = BacktestOrder(
            order_id=make_order_id(), token_id="tok", side="SELL",
            price=0.49, size=10, tif=TIF_IOC, post_only=False,
            submitted_at=T0,
        )
        me.submit(order)
        me.advance_to(_snap(T0 + timedelta(milliseconds=200), bid=0.50, ask=0.51, depth=2, qty=5))
        # Two distinct fills (one per level)
        assert len(order.fills) == 2
        # Each pays $0.10 gas
        for f in order.fills:
            assert f.fee_usd == pytest.approx(0.10)

    def test_resolution_fee_charged_on_winners_only(self):
        from services.backtest.matching_engine import FeeModel
        fees = FeeModel(per_fill_gas_usd=0.0, resolution_fee_rate=0.02)
        # Winning $10 gross → $0.20 fee
        assert fees.resolution_fee(gross_winnings_usd=10.0) == pytest.approx(0.20)
        # Loss → no fee
        assert fees.resolution_fee(gross_winnings_usd=-5.0) == 0.0
        assert fees.resolution_fee(gross_winnings_usd=0.0) == 0.0

    @pytest.mark.asyncio
    async def test_resolution_fee_applied_at_close(self):
        from services.backtest.matching_engine import FeeModel
        # Simple drift: bid rises so a BUY → SELL closes at higher price
        snaps = []
        for i in range(30):
            bid = 0.40 + 0.001 * i
            ask = bid + 0.01
            snaps.append(_snap(T0 + timedelta(seconds=i), bid=bid, ask=ask, depth=5, qty=50))
        br = InMemoryBookReplay(snaps)

        class _ProfitTaker(BaseStrategy):
            strategy_type = "profit_test"
            name = "profit"
            description = "simple"
            def should_exit(self, position, market_state):
                if position.pnl_percent >= 5.0:
                    return ExitDecision("close", "take_profit",
                                        close_price=market_state["current_price"])
                return ExitDecision("hold", "wait")

        intent = TradeIntent(
            intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
            size=20, limit_price=0.42, tif=TIF_IOC, post_only=False,
            strategy_slug="profit_test",
        )
        engine = BacktestEngine(
            config=BacktestConfig(
                portfolio=PortfolioConfig(initial_capital_usd=1000.0),
                latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
                fees=FeeModel(per_fill_gas_usd=0.0, resolution_fee_rate=0.02),
                seed=42,
            ),
            strategy=_ProfitTaker(),
        )
        result = await engine.run(book_source=br, trade_intents=[intent])
        # Some fees should have been paid (gas=0, so all fees are resolution fees)
        assert result.fees_resolution_usd > 0
        assert result.fees_per_fill_usd == pytest.approx(0.0)


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_legacy_single_order_exit_runs_to_completion(self):
        br = _build_drifting_book(60)
        intent = TradeIntent(
            intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
            size=20, limit_price=0.52, tif=TIF_IOC, post_only=False,
            strategy_slug="legacy_test",
        )
        engine = BacktestEngine(
            config=BacktestConfig(
                portfolio=PortfolioConfig(initial_capital_usd=1000.0),
                latency=LatencyModel.deterministic(submit_ms=200, cancel_ms=100),
                seed=42,
            ),
            strategy=_LegacyExitStrategy(),
        )
        result = await engine.run(book_source=br, trade_intents=[intent])
        assert result.notes["snapshots_processed"] == 60
        assert result.closed_position_count == 1
        # Stop-loss firing produces both an entry and an exit fill (at minimum).
        assert result.total_fills >= 2

    @pytest.mark.asyncio
    async def test_laddered_exit_simulates_multiple_child_fills(self):
        br = _build_drifting_book(60)
        intent = TradeIntent(
            intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
            size=50, limit_price=0.52, tif=TIF_IOC, post_only=False,
            strategy_slug="ladder_test",
        )
        engine = BacktestEngine(
            config=BacktestConfig(
                portfolio=PortfolioConfig(initial_capital_usd=1000.0),
                latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
                seed=42,
            ),
            strategy=_LadderExitStrategy(),
        )
        await engine.run(book_source=br, trade_intents=[intent])
        # Engine must produce > 1 exit fill (laddered children)
        exit_orders = [o for o in engine.matching.all_orders() if o.meta.get("role") == "exit_child"]
        assert len(exit_orders) > 1
        filled = sum(1 for o in exit_orders if o.filled_size > 0)
        assert filled >= 1
        # Total filled across exit children should match the position size
        total_exit_filled = sum(o.filled_size for o in exit_orders)
        assert total_exit_filled == pytest.approx(50.0, rel=0.05)

    @pytest.mark.asyncio
    async def test_capital_gate_blocks_oversized_intent(self):
        br = _build_drifting_book(10)
        intent = TradeIntent(
            intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
            size=10000, limit_price=0.52, tif=TIF_IOC, post_only=False,
            strategy_slug="legacy_test",
        )
        engine = BacktestEngine(
            config=BacktestConfig(
                portfolio=PortfolioConfig(initial_capital_usd=100.0),
                latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
                seed=42,
            ),
            strategy=_LegacyExitStrategy(),
        )
        result = await engine.run(book_source=br, trade_intents=[intent])
        # Intent rejected at portfolio gate → no fills
        assert result.total_fills == 0

    @pytest.mark.asyncio
    async def test_metrics_bootstrap_present(self):
        br = _build_drifting_book(60)
        intent = TradeIntent(
            intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
            size=20, limit_price=0.52, tif=TIF_IOC, post_only=False,
            strategy_slug="legacy_test",
        )
        engine = BacktestEngine(
            config=BacktestConfig(
                portfolio=PortfolioConfig(initial_capital_usd=1000.0),
                latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
                seed=42,
            ),
            strategy=_LegacyExitStrategy(),
        )
        result = await engine.run(book_source=br, trade_intents=[intent])
        assert result.metrics.trade_count >= 1
        # Bootstrap returns None for tiny samples — that's expected for a
        # 1-trade backtest. Verify the structure.
        assert hasattr(result.metrics.sharpe, "value")
        assert hasattr(result.metrics.hit_rate, "ci_low")

    @pytest.mark.asyncio
    async def test_equity_history_anchors_to_final_equity(self):
        """Regression: chart's last point must match the headline return.

        Bug observed in production: a -16.32% max drawdown run that
        recovered to -0.18% return rendered the equity curve ending at
        $836 (pre-recovery trough sample) while the headline showed
        -0.18%.  Root cause: ``_final_mark_to_market`` converts
        unrealized PnL to realized cash + pays resolution fees but
        never appended a final ``equity_history`` entry, so the curve
        ended at the last mark BEFORE close-out — diverging from
        ``portfolio.equity_usd()`` (the source of total_return_pct).
        """
        br = _build_drifting_book(60)
        intent = TradeIntent(
            intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
            size=20, limit_price=0.52, tif=TIF_IOC, post_only=False,
            strategy_slug="legacy_test",
        )
        # Strategy that NEVER exits — guarantees a position is open at
        # window end and exercises _final_mark_to_market.
        class _NeverExit(BaseStrategy):
            strategy_type = "never_exit"
            name = "never"
            description = "hold"
            def should_exit(self, position, market_state):
                return ExitDecision("hold", "always")

        engine = BacktestEngine(
            config=BacktestConfig(
                portfolio=PortfolioConfig(initial_capital_usd=1000.0),
                latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
                final_close_at_last_mid=True,
                seed=42,
            ),
            strategy=_NeverExit(),
        )
        result = await engine.run(book_source=br, trade_intents=[intent])
        assert result.metrics.trade_count >= 1, "intent should have filled"
        assert len(result.equity_history) > 0, "equity_history must be populated"
        # The LAST equity_history entry must equal final_equity_usd to
        # within float tolerance — otherwise the rendered chart's
        # ending value will disagree with total_return_pct.
        last_at, last_equity = result.equity_history[-1]
        assert last_equity == pytest.approx(result.final_equity_usd, abs=1e-6), (
            f"equity_history[-1]={last_equity:.6f} must match "
            f"final_equity_usd={result.final_equity_usd:.6f} so the "
            f"rendered chart matches the headline return"
        )
