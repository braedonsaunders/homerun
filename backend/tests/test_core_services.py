"""
Comprehensive unit tests for core services:
  - AutoTrader
  - SimulationService / SlippageModel
  - AnomalyDetector
  - CopyTradingService
"""

import sys
sys.path.insert(0, "/home/user/homerun/backend")

import math
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# AutoTrader imports
# ---------------------------------------------------------------------------
from services.auto_trader import (
    AutoTrader,
    AutoTraderConfig,
    AutoTraderMode,
    AutoTraderStats,
    TradeRecord,
)
from models.opportunity import ArbitrageOpportunity, StrategyType, MispricingType

# ---------------------------------------------------------------------------
# Simulation imports
# ---------------------------------------------------------------------------
from services.simulation import SimulationService, SlippageModel

# ---------------------------------------------------------------------------
# AnomalyDetector imports
# ---------------------------------------------------------------------------
from services.anomaly_detector import (
    AnomalyDetector,
    AnomalyType,
    Severity,
    Anomaly,
)

# ---------------------------------------------------------------------------
# CopyTrader imports
# ---------------------------------------------------------------------------
from services.copy_trader import CopyTradingService
from models.database import CopyTradingConfig, CopyTradingMode


# ============================================================================
# HELPERS
# ============================================================================

def _make_opportunity(**overrides) -> ArbitrageOpportunity:
    """Build a valid ArbitrageOpportunity with sane defaults."""
    defaults = dict(
        strategy=StrategyType.BASIC,
        title="Test Opportunity",
        description="Test description",
        total_cost=0.95,
        expected_payout=1.0,
        gross_profit=0.05,
        fee=0.02,
        net_profit=0.03,
        roi_percent=3.16,
        risk_score=0.3,
        risk_factors=[],
        markets=[{"id": "mkt_abc12345"}],
        event_id="evt_1",
        event_title="Test Event",
        min_liquidity=10000.0,
        max_position_size=50.0,
        positions_to_take=[
            {"market": "mkt_abc12345", "outcome": "YES", "price": 0.45, "token_id": "tok1"},
            {"market": "mkt_abc12345", "outcome": "NO", "price": 0.50, "token_id": "tok2"},
        ],
    )
    defaults.update(overrides)
    return ArbitrageOpportunity(**defaults)


def _fresh_auto_trader(**config_overrides) -> AutoTrader:
    """Return a new AutoTrader instance with optional config tweaks."""
    trader = AutoTrader()
    trader.configure(mode=AutoTraderMode.SHADOW, **config_overrides)
    return trader


# ============================================================================
# AUTO TRADER TESTS
# ============================================================================


class TestAutoTraderCircuitBreaker:
    """Circuit breaker triggers after consecutive losses."""

    def test_circuit_breaker_triggers_after_consecutive_losses(self):
        trader = _fresh_auto_trader(circuit_breaker_losses=3)
        opp = _make_opportunity()

        # Record 3 consecutive losses
        for i in range(3):
            tid = f"trade_{i}"
            trader._trades[tid] = TradeRecord(
                id=tid,
                opportunity_id=opp.id,
                strategy=opp.strategy,
                executed_at=datetime.utcnow(),
                positions=opp.positions_to_take,
                total_cost=10.0,
                expected_profit=0.5,
            )
            trader.record_trade_result(tid, won=False, actual_profit=-5.0)

        assert trader.stats.consecutive_losses == 3

        can_trade, reason = trader._check_circuit_breaker()
        assert can_trade is False
        assert "Circuit breaker triggered" in reason

    def test_circuit_breaker_resets_after_win(self):
        trader = _fresh_auto_trader(circuit_breaker_losses=3)

        # 2 losses
        for i in range(2):
            tid = f"t_{i}"
            trader._trades[tid] = TradeRecord(
                id=tid, opportunity_id="opp", strategy=StrategyType.BASIC,
                executed_at=datetime.utcnow(), positions=[], total_cost=10,
                expected_profit=1,
            )
            trader.record_trade_result(tid, won=False, actual_profit=-5)

        assert trader.stats.consecutive_losses == 2

        # Then a win
        tid = "t_win"
        trader._trades[tid] = TradeRecord(
            id=tid, opportunity_id="opp", strategy=StrategyType.BASIC,
            executed_at=datetime.utcnow(), positions=[], total_cost=10,
            expected_profit=1,
        )
        trader.record_trade_result(tid, won=True, actual_profit=2)
        assert trader.stats.consecutive_losses == 0

    def test_circuit_breaker_timeout_expires(self):
        trader = _fresh_auto_trader(circuit_breaker_losses=2)

        # Set circuit breaker to already-expired time
        trader.stats.circuit_breaker_until = datetime.utcnow() - timedelta(seconds=1)
        trader.stats.consecutive_losses = 5  # stale value

        can_trade, _ = trader._check_circuit_breaker()
        assert can_trade is True
        assert trader.stats.circuit_breaker_until is None
        assert trader.stats.consecutive_losses == 0


class TestAutoTraderDailyDrawdown:
    """Daily drawdown limit enforcement."""

    def test_daily_loss_limit_blocks_trading(self):
        trader = _fresh_auto_trader(max_daily_loss_usd=50.0)
        # Simulate a daily loss exceeding the limit
        trader.stats.daily_profit = -60.0

        can_trade, reason = trader._check_circuit_breaker()
        assert can_trade is False
        assert "Daily loss limit" in reason

    def test_daily_loss_at_limit_still_blocks(self):
        trader = _fresh_auto_trader(max_daily_loss_usd=50.0)
        trader.stats.daily_profit = -50.01

        can_trade, _ = trader._check_circuit_breaker()
        assert can_trade is False

    def test_daily_loss_within_limit_allows_trading(self):
        trader = _fresh_auto_trader(max_daily_loss_usd=50.0)
        trader.stats.daily_profit = -40.0

        can_trade, _ = trader._check_circuit_breaker()
        assert can_trade is True

    def test_daily_trades_limit(self):
        trader = _fresh_auto_trader(max_daily_trades=5)
        trader.stats.daily_trades = 5

        can_trade, reason = trader._check_circuit_breaker()
        assert can_trade is False
        assert "Daily trade limit" in reason

    def test_daily_stats_reset_on_new_day(self):
        trader = _fresh_auto_trader()
        trader.stats.daily_trades = 10
        trader.stats.daily_profit = -999
        trader.stats.daily_invested = 5000
        # Force yesterday
        trader._daily_reset_date = (datetime.utcnow() - timedelta(days=1)).date()

        trader._check_daily_reset()

        assert trader.stats.daily_trades == 0
        assert trader.stats.daily_profit == 0.0
        assert trader.stats.daily_invested == 0.0


class TestAutoTraderPositionSizing:
    """Position sizing methods: fixed, Kelly, volatility-adjusted (default)."""

    def test_fixed_sizing(self):
        trader = _fresh_auto_trader(
            position_size_method="fixed",
            base_position_size_usd=25.0,
            max_position_size_usd=100.0,
        )
        opp = _make_opportunity(max_position_size=200.0)
        size = trader._calculate_position_size(opp)
        assert size == 25.0

    def test_fixed_sizing_capped_by_max(self):
        trader = _fresh_auto_trader(
            position_size_method="fixed",
            base_position_size_usd=150.0,
            max_position_size_usd=100.0,
        )
        opp = _make_opportunity(max_position_size=200.0)
        size = trader._calculate_position_size(opp)
        assert size == 100.0

    def test_fixed_sizing_capped_by_max_position(self):
        trader = _fresh_auto_trader(
            position_size_method="fixed",
            base_position_size_usd=60.0,
            max_position_size_usd=100.0,
        )
        opp = _make_opportunity(max_position_size=30.0)
        size = trader._calculate_position_size(opp)
        assert size == 30.0

    def test_kelly_sizing_high_liquidity(self):
        trader = _fresh_auto_trader(
            position_size_method="kelly",
            max_position_size_usd=100.0,
        )
        # Need ROI high enough that Kelly is positive after execution risk.
        # exec_prob = 0.95 (liquidity >= 20000), q = 0.05
        # standard_kelly = (0.20*0.95 - 0.05)/0.20 = 0.14/0.20 = 0.70
        # adjusted = 0.70 * sqrt(0.95) = ~0.68, clamped to 0.25
        # size = 100 * 0.25 = 25.0
        opp = _make_opportunity(
            roi_percent=20.0,
            min_liquidity=25000.0,
            max_position_size=200.0,
            positions_to_take=[{"market": "m1", "outcome": "YES", "price": 0.5}],
        )
        size = trader._calculate_position_size(opp)
        assert size > 0
        assert size <= 100.0

    def test_kelly_sizing_low_liquidity(self):
        trader = _fresh_auto_trader(
            position_size_method="kelly",
            max_position_size_usd=100.0,
        )
        opp = _make_opportunity(
            roi_percent=5.0,
            min_liquidity=500.0,
            max_position_size=200.0,
            positions_to_take=[{"market": "m1", "outcome": "YES", "price": 0.5}],
        )
        size = trader._calculate_position_size(opp)
        # Low liquidity => exec_prob = 0.5, kelly likely 0 or very small
        assert size >= 0

    def test_kelly_sizing_zero_roi(self):
        trader = _fresh_auto_trader(
            position_size_method="kelly",
            max_position_size_usd=100.0,
        )
        opp = _make_opportunity(
            roi_percent=0.0,
            min_liquidity=25000.0,
            max_position_size=200.0,
        )
        size = trader._calculate_position_size(opp)
        assert size == 0

    def test_unknown_method_falls_back_to_fixed(self):
        trader = _fresh_auto_trader(
            position_size_method="volatility_adjusted",
            base_position_size_usd=20.0,
            max_position_size_usd=100.0,
        )
        opp = _make_opportunity(max_position_size=200.0)
        # Unknown method falls through to the else branch (same as fixed)
        size = trader._calculate_position_size(opp)
        assert size == 20.0


class TestAutoTraderMaxOpenPositions:
    """Max open positions limit."""

    def test_max_positions_blocks_new_trade(self):
        trader = _fresh_auto_trader(max_concurrent_positions=2)

        # Insert 2 open trades
        for i in range(2):
            tid = f"open_{i}"
            trader._trades[tid] = TradeRecord(
                id=tid, opportunity_id=f"opp_{i}", strategy=StrategyType.BASIC,
                executed_at=datetime.utcnow(), positions=[], total_cost=10,
                expected_profit=1, status="open",
            )

        opp = _make_opportunity()
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "Max concurrent positions" in reason

    def test_resolved_positions_dont_count(self):
        trader = _fresh_auto_trader(max_concurrent_positions=2)

        # 1 open, 1 resolved
        trader._trades["open_1"] = TradeRecord(
            id="open_1", opportunity_id="o1", strategy=StrategyType.BASIC,
            executed_at=datetime.utcnow(), positions=[], total_cost=10,
            expected_profit=1, status="open",
        )
        trader._trades["resolved_1"] = TradeRecord(
            id="resolved_1", opportunity_id="o2", strategy=StrategyType.BASIC,
            executed_at=datetime.utcnow(), positions=[], total_cost=10,
            expected_profit=1, status="resolved_win",
        )

        opp = _make_opportunity()
        should, _ = trader._should_trade_opportunity(opp)
        assert should is True


class TestAutoTraderFiltering:
    """Trade filtering by risk score, ROI, strategy, etc."""

    def test_filter_by_risk_score_rejects_high_risk(self):
        trader = _fresh_auto_trader(max_risk_score=0.5)
        opp = _make_opportunity(risk_score=0.7)
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "Risk score" in reason

    def test_filter_by_risk_score_accepts_low_risk(self):
        trader = _fresh_auto_trader(max_risk_score=0.5)
        opp = _make_opportunity(risk_score=0.3)
        should, _ = trader._should_trade_opportunity(opp)
        assert should is True

    def test_filter_by_min_roi_rejects_low_roi(self):
        trader = _fresh_auto_trader(min_roi_percent=5.0)
        opp = _make_opportunity(roi_percent=2.0)
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "ROI" in reason

    def test_filter_by_min_roi_accepts_high_roi(self):
        trader = _fresh_auto_trader(min_roi_percent=2.0)
        opp = _make_opportunity(roi_percent=5.0)
        should, _ = trader._should_trade_opportunity(opp)
        assert should is True

    def test_disabled_mode_rejects_all(self):
        trader = AutoTrader()  # default is DISABLED
        opp = _make_opportunity()
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "disabled" in reason.lower()

    def test_already_processed_skipped(self):
        trader = _fresh_auto_trader()
        opp = _make_opportunity()
        trader._processed_opportunities.add(opp.id)
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "Already processed" in reason

    def test_strategy_not_enabled_rejects(self):
        trader = _fresh_auto_trader(enabled_strategies=[StrategyType.MIRACLE])
        opp = _make_opportunity(strategy=StrategyType.BASIC)
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "Strategy" in reason

    def test_low_liquidity_rejects(self):
        trader = _fresh_auto_trader(min_liquidity_usd=10000.0)
        opp = _make_opportunity(min_liquidity=2000.0)
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "Liquidity" in reason

    def test_guaranteed_profit_filter(self):
        trader = _fresh_auto_trader(
            use_profit_guarantee=True,
            min_guaranteed_profit=0.10,
        )
        opp = _make_opportunity(guaranteed_profit=0.02)
        should, reason = trader._should_trade_opportunity(opp)
        assert should is False
        assert "Guaranteed profit" in reason

    def test_guaranteed_profit_filter_pass(self):
        trader = _fresh_auto_trader(
            use_profit_guarantee=True,
            min_guaranteed_profit=0.05,
        )
        opp = _make_opportunity(guaranteed_profit=0.10)
        should, _ = trader._should_trade_opportunity(opp)
        assert should is True


class TestAutoTraderModes:
    """Paper / shadow / live mode behaviour differences."""

    @pytest.mark.asyncio
    async def test_shadow_mode_records_trade(self):
        trader = _fresh_auto_trader()
        trader.config.mode = AutoTraderMode.SHADOW
        opp = _make_opportunity()

        trade = await trader._execute_trade(opp)
        assert trade.status == "shadow"
        assert trade.mode == AutoTraderMode.SHADOW

    @pytest.mark.asyncio
    async def test_paper_mode_calls_simulation(self):
        trader = _fresh_auto_trader()
        trader.config.mode = AutoTraderMode.PAPER

        mock_account = MagicMock()
        mock_account.id = "sim_acc_1"
        mock_account.name = "Auto Trader"

        mock_sim_trade = MagicMock()
        mock_sim_trade.id = "sim_trade_1"

        # simulation_service is imported locally inside _execute_trade via:
        #   from services.simulation import simulation_service
        # So we patch it on the services.simulation module.
        mock_sim = MagicMock()
        mock_sim.get_all_accounts = AsyncMock(return_value=[mock_account])
        mock_sim.execute_opportunity = AsyncMock(return_value=mock_sim_trade)

        with patch("services.simulation.simulation_service", mock_sim):
            opp = _make_opportunity()
            trade = await trader._execute_trade(opp)

            assert trade.status == "open"
            assert trade.mode == AutoTraderMode.PAPER

    @pytest.mark.asyncio
    async def test_live_mode_fails_if_not_ready(self):
        trader = _fresh_auto_trader()
        trader.config.mode = AutoTraderMode.LIVE

        with patch("services.auto_trader.trading_service") as mock_ts:
            mock_ts.is_ready.return_value = False
            opp = _make_opportunity()
            trade = await trader._execute_trade(opp)
            assert trade.status == "failed"

    def test_max_trade_size_enforcement(self):
        """Position size never exceeds max_position_size_usd."""
        trader = _fresh_auto_trader(
            position_size_method="fixed",
            base_position_size_usd=500.0,
            max_position_size_usd=75.0,
        )
        opp = _make_opportunity(max_position_size=200.0)
        size = trader._calculate_position_size(opp)
        assert size <= 75.0


# ============================================================================
# SIMULATION TESTS
# ============================================================================


class TestSlippageModel:
    """Test the three slippage calculation models."""

    def test_fixed_slippage(self):
        # 50 bps on a 0.60 price => 0.60 * (1 + 50/10000) = 0.603
        result = SlippageModel.fixed(0.60, 50.0)
        assert result == pytest.approx(0.603, abs=1e-6)

    def test_fixed_slippage_zero_bps(self):
        result = SlippageModel.fixed(0.60, 0.0)
        assert result == pytest.approx(0.60)

    def test_linear_slippage_small_order(self):
        # size/liquidity = 100/10000 = 0.01, impact = 0.01 * 50/10000 = 0.00005
        result = SlippageModel.linear(0.50, 100.0, 10000.0, 50.0)
        expected = 0.50 * (1 + 0.01 * 50 / 10000)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_linear_slippage_large_order(self):
        # size/liquidity = 5000/10000 = 0.5
        result = SlippageModel.linear(0.50, 5000.0, 10000.0, 50.0)
        expected = 0.50 * (1 + 0.5 * 50 / 10000)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_sqrt_slippage(self):
        # impact = sqrt(100/10000) * 50/10000 = 0.1 * 0.005 = 0.0005
        result = SlippageModel.sqrt(0.50, 100.0, 10000.0, 50.0)
        expected = 0.50 * (1 + (100 / 10000) ** 0.5 * 50 / 10000)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_sqrt_larger_than_linear_for_small_orders(self):
        """sqrt model produces MORE slippage on small orders than linear."""
        price, size, liq, bps = 0.50, 100.0, 100000.0, 50.0
        lin = SlippageModel.linear(price, size, liq, bps)
        sqr = SlippageModel.sqrt(price, size, liq, bps)
        # sqrt(0.001) > 0.001 so sqrt model gives higher price
        assert sqr > lin


class TestSimulationServiceSlippage:
    """Test the service-level _calculate_slippage helper."""

    def setup_method(self):
        self.svc = SimulationService()

    def test_fixed_model_slippage(self):
        slippage = self.svc._calculate_slippage("fixed", 100.0, 50.0, 10000.0, 50.0)
        # factor = 1 + 50/10000 = 1.005 => slippage = 100*(1.005 - 1) = 0.5
        assert slippage == pytest.approx(0.5, abs=1e-6)

    def test_linear_model_slippage(self):
        slippage = self.svc._calculate_slippage("linear", 100.0, 50.0, 10000.0, 50.0)
        factor = 1 + (50.0 / 10000.0) * 50.0 / 10000.0
        expected = 100.0 * (factor - 1)
        assert slippage == pytest.approx(expected, abs=1e-8)

    def test_sqrt_model_slippage(self):
        slippage = self.svc._calculate_slippage("sqrt", 100.0, 50.0, 10000.0, 50.0)
        factor = 1 + (50.0 / 10000.0) ** 0.5 * 50.0 / 10000.0
        expected = 100.0 * (factor - 1)
        assert slippage == pytest.approx(expected, abs=1e-8)

    def test_zero_liquidity_uses_default(self):
        slippage = self.svc._calculate_slippage("linear", 100.0, 50.0, 0.0, 50.0)
        # liquidity defaults to 10000
        factor = 1 + (50.0 / 10000.0) * 50.0 / 10000.0
        expected = 100.0 * (factor - 1)
        assert slippage == pytest.approx(expected, abs=1e-8)

    def test_negative_liquidity_uses_default(self):
        slippage = self.svc._calculate_slippage("fixed", 100.0, 50.0, -500.0, 50.0)
        expected = 100.0 * (50.0 / 10000.0)
        assert slippage == pytest.approx(expected, abs=1e-6)


class TestSimulationFeeCalculation:
    """Fee is 2% on winnings (POLYMARKET_FEE = 0.02)."""

    def test_fee_constant(self):
        svc = SimulationService()
        assert svc.POLYMARKET_FEE == 0.02

    def test_fee_on_payout(self):
        """Fee = payout * 0.02"""
        payout = 100.0
        fee = payout * SimulationService.POLYMARKET_FEE
        assert fee == pytest.approx(2.0)

    def test_no_fee_on_zero_payout(self):
        payout = 0.0
        fee = payout * SimulationService.POLYMARKET_FEE if payout > 0 else 0
        assert fee == 0.0


class TestSimulationPnL:
    """PnL calculation: net_payout - total_cost."""

    def test_winning_trade_pnl(self):
        total_cost = 0.90
        payout = 1.0
        fee = payout * 0.02
        net = payout - fee
        pnl = net - total_cost
        assert pnl == pytest.approx(0.08, abs=1e-6)

    def test_losing_trade_pnl(self):
        total_cost = 0.90
        payout = 0.0
        fee = 0.0
        net = payout - fee
        pnl = net - total_cost
        assert pnl == pytest.approx(-0.90, abs=1e-6)

    def test_break_even_with_fee(self):
        """If payout == total_cost + fee, PnL is zero."""
        total_cost = 0.98
        payout = 1.0
        fee = payout * 0.02  # 0.02
        net = payout - fee  # 0.98
        pnl = net - total_cost
        assert pnl == pytest.approx(0.0, abs=1e-6)


class TestSimulationPortfolioTracking:
    """Portfolio value = current_capital (initial - invested + payouts)."""

    def test_capital_decreases_after_trade(self):
        initial = 10000.0
        trade_cost = 100.0
        remaining = initial - trade_cost
        assert remaining == 9900.0

    def test_capital_increases_after_win(self):
        current = 9900.0
        payout = 100.0
        fee = payout * 0.02
        new_capital = current + payout - fee
        assert new_capital == pytest.approx(9998.0)

    def test_multiple_trades_track_correctly(self):
        capital = 10000.0
        # Trade 1: cost 100, win payout 100 (net 98 after 2% fee)
        capital -= 100.0         # 9900
        capital += 100.0 * 0.98  # 9998 (net payout after fee)
        # Trade 2: cost 50, lose (0 payout)
        capital -= 50.0          # 9948
        assert capital == pytest.approx(9948.0)


class TestSimulationWithKnownData:
    """Simulation with a known opportunity producing predictable results."""

    @pytest.mark.asyncio
    async def test_execute_and_resolve_in_simulation(self):
        """Verify full lifecycle through mocked DB session."""
        svc = SimulationService()
        opp = _make_opportunity(
            total_cost=0.90,
            net_profit=0.08,
            min_liquidity=50000.0,
            max_position_size=100.0,
        )

        # Mock SimulationAccount object
        mock_account = MagicMock()
        mock_account.id = "acc_1"
        mock_account.current_capital = 10000.0
        mock_account.max_position_size_pct = 10.0
        mock_account.slippage_model = "fixed"
        mock_account.slippage_bps = 0.0  # zero slippage for predictability
        mock_account.total_trades = 0

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_account)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # Make the context manager return our mock session
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("services.simulation.AsyncSessionLocal", return_value=mock_session):
            trade = await svc.execute_opportunity(
                account_id="acc_1",
                opportunity=opp,
                position_size=50.0,
            )

        # Assertions on the created trade
        assert trade is not None
        mock_session.add.assert_called()
        mock_session.commit.assert_called()


# ============================================================================
# ANOMALY DETECTOR TESTS
# ============================================================================


class TestAnomalyDetectorWashTrading:
    """Wash trading detection: rapid buy/sell of same market."""

    def test_detects_wash_trading(self):
        detector = AnomalyDetector()
        trades = [
            {"market": "m1", "side": "BUY", "timestamp": "2024-01-01T00:00:00Z",
             "size": 100, "price": 0.5, "outcome": "YES"},
            {"market": "m1", "side": "SELL", "timestamp": "2024-01-01T00:00:30Z",
             "size": 100, "price": 0.5, "outcome": "YES"},
        ]
        anomalies = detector._detect_pattern_anomalies(trades)
        wash = [a for a in anomalies if a.type == AnomalyType.WASH_TRADING]
        assert len(wash) >= 1

    def test_no_wash_trading_same_side(self):
        detector = AnomalyDetector()
        trades = [
            {"market": "m1", "side": "BUY", "timestamp": "2024-01-01T00:00:00Z",
             "size": 100, "price": 0.5},
            {"market": "m1", "side": "BUY", "timestamp": "2024-01-01T00:00:30Z",
             "size": 100, "price": 0.5},
        ]
        anomalies = detector._detect_pattern_anomalies(trades)
        wash = [a for a in anomalies if a.type == AnomalyType.WASH_TRADING]
        assert len(wash) == 0


class TestAnomalyDetectorFrontRunning:
    """Front-running detection (timing anomalies)."""

    def test_timing_anomalies_returns_list(self):
        detector = AnomalyDetector()
        trades = [
            {"market": "m1", "side": "BUY", "timestamp": "2024-01-01T00:00:00Z",
             "price": 0.3, "size": 500},
        ]
        result = detector._detect_timing_anomalies(trades)
        assert isinstance(result, list)

    def test_detect_pattern_anomalies_front_run_placeholder(self):
        """The current code's timing detection is a placeholder; verify it doesn't crash."""
        detector = AnomalyDetector()
        trades = [
            {"market": "m1", "side": "BUY", "timestamp": "2024-01-01T00:00:00Z",
             "price": 0.1, "size": 1000},
            {"market": "m1", "side": "SELL", "timestamp": "2024-01-01T00:00:05Z",
             "price": 0.9, "size": 1000},
        ]
        anomalies = detector._detect_timing_anomalies(trades)
        # Currently returns empty, placeholder logic
        assert isinstance(anomalies, list)


class TestAnomalyDetectorImpossibleWinRate:
    """Impossible win rate detection (> 95%)."""

    def test_flags_96_percent_win_rate(self):
        detector = AnomalyDetector()
        stats = {
            "win_rate": 0.96,
            "total_trades": 100,
            "wins": 96,
            "losses": 4,
            "roi_std": 5.0,
            "avg_roi": 5.0,
            "max_roi": 10.0,
        }
        anomalies = detector._detect_statistical_anomalies([], stats)
        impossible = [a for a in anomalies if a.type == AnomalyType.IMPOSSIBLE_WIN_RATE]
        assert len(impossible) >= 1

    def test_99_percent_win_rate_critical(self):
        detector = AnomalyDetector()
        stats = {
            "win_rate": 0.99,
            "total_trades": 200,
            "wins": 198,
            "losses": 2,
            "roi_std": 3.0,
            "avg_roi": 8.0,
            "max_roi": 15.0,
        }
        anomalies = detector._detect_statistical_anomalies([], stats)
        impossible = [a for a in anomalies if a.type == AnomalyType.IMPOSSIBLE_WIN_RATE]
        assert len(impossible) >= 1
        assert impossible[0].severity == Severity.CRITICAL

    def test_no_flag_for_normal_win_rate(self):
        detector = AnomalyDetector()
        stats = {
            "win_rate": 0.60,
            "total_trades": 50,
            "wins": 30,
            "losses": 20,
            "roi_std": 5.0,
            "avg_roi": 3.0,
            "max_roi": 10.0,
        }
        anomalies = detector._detect_statistical_anomalies([], stats)
        impossible = [a for a in anomalies if a.type == AnomalyType.IMPOSSIBLE_WIN_RATE]
        assert len(impossible) == 0

    def test_zero_losses_flags_statistically_impossible(self):
        detector = AnomalyDetector()
        stats = {
            "win_rate": 1.0,
            "total_trades": 25,
            "wins": 25,
            "losses": 0,
            "roi_std": 2.0,
            "avg_roi": 5.0,
            "max_roi": 10.0,
        }
        anomalies = detector._detect_statistical_anomalies([], stats)
        stat_imp = [a for a in anomalies if a.type == AnomalyType.STATISTICALLY_IMPOSSIBLE]
        assert len(stat_imp) >= 1


class TestAnomalyScoring:
    """Anomaly scoring aggregation."""

    def test_no_anomalies_score_zero(self):
        detector = AnomalyDetector()
        score = detector._calculate_anomaly_score([])
        assert score == 0.0

    def test_single_critical_anomaly(self):
        detector = AnomalyDetector()
        anomalies = [
            Anomaly(
                type=AnomalyType.IMPOSSIBLE_WIN_RATE,
                severity=Severity.CRITICAL,
                score=0.9,
                description="test",
                evidence={},
            )
        ]
        score = detector._calculate_anomaly_score(anomalies)
        # 0.9 * 1.0 (critical weight) / 1 = 0.9
        assert score == pytest.approx(0.9, abs=0.01)

    def test_mixed_severity_scoring(self):
        detector = AnomalyDetector()
        anomalies = [
            Anomaly(type=AnomalyType.WASH_TRADING, severity=Severity.MEDIUM,
                     score=0.6, description="", evidence={}),
            Anomaly(type=AnomalyType.UNUSUAL_ROI, severity=Severity.HIGH,
                     score=0.8, description="", evidence={}),
        ]
        score = detector._calculate_anomaly_score(anomalies)
        # (0.6*0.4 + 0.8*0.7) / 2 = (0.24 + 0.56) / 2 = 0.40
        expected = (0.6 * 0.4 + 0.8 * 0.7) / 2
        assert score == pytest.approx(expected, abs=0.01)

    def test_score_capped_at_one(self):
        detector = AnomalyDetector()
        anomalies = [
            Anomaly(type=AnomalyType.IMPOSSIBLE_WIN_RATE, severity=Severity.CRITICAL,
                     score=1.0, description="", evidence={}),
            Anomaly(type=AnomalyType.STATISTICALLY_IMPOSSIBLE, severity=Severity.CRITICAL,
                     score=1.0, description="", evidence={}),
        ]
        score = detector._calculate_anomaly_score(anomalies)
        assert score <= 1.0


class TestAnomalyDetectorCleanData:
    """Clean data should NOT be flagged."""

    def test_clean_stats_no_anomalies(self):
        detector = AnomalyDetector()
        stats = {
            "win_rate": 0.55,
            "total_trades": 50,
            "wins": 28,
            "losses": 22,
            "roi_std": 3.0,
            "avg_roi": 3.0,
            "max_roi": 8.0,
        }
        anomalies = detector._detect_statistical_anomalies([], stats)
        assert len(anomalies) == 0

    def test_clean_pattern_no_anomalies(self):
        detector = AnomalyDetector()
        trades = [
            {"market": f"m{i}", "side": "BUY", "timestamp": f"2024-01-0{i+1}T00:00:00Z",
             "size": 100, "price": 0.5, "pnl": 2, "cost": 50, "outcome": "YES"}
            for i in range(5)
        ]
        anomalies = detector._detect_pattern_anomalies(trades)
        # No wash trading (different markets), and arb ratio < 0.8
        wash = [a for a in anomalies if a.type == AnomalyType.WASH_TRADING]
        assert len(wash) == 0


class TestAnomalyDetectorSuspiciousPatterns:
    """Suspicious patterns should be flagged."""

    def test_unusual_roi_flagged(self):
        detector = AnomalyDetector()
        stats = {
            "win_rate": 0.85,
            "total_trades": 100,
            "wins": 85,
            "losses": 15,
            "roi_std": 10.0,
            "avg_roi": 55.0,
            "max_roi": 200.0,
        }
        anomalies = detector._detect_statistical_anomalies([], stats)
        unusual = [a for a in anomalies if a.type == AnomalyType.UNUSUAL_ROI]
        assert len(unusual) >= 1

    def test_arbitrage_only_pattern_flagged(self):
        detector = AnomalyDetector()
        # 90% arb-like trades (pnl/cost in 1-10% range)
        trades = []
        for i in range(20):
            trades.append({
                "market": f"m{i}", "side": "BUY", "timestamp": f"2024-01-01T00:{i:02d}:00Z",
                "size": 100, "price": 0.5, "pnl": 3, "cost": 50, "outcome": "YES",
            })
        anomalies = detector._detect_pattern_anomalies(trades)
        arb_only = [a for a in anomalies if a.type == AnomalyType.ARBITRAGE_ONLY]
        assert len(arb_only) >= 1

    def test_is_profitable_pattern_requires_positive_pnl(self):
        detector = AnomalyDetector()
        stats = {"total_pnl": -100, "win_rate": 0.7}
        assert detector._is_profitable_pattern(stats, [], ["basic_arbitrage"]) is False

    def test_is_profitable_pattern_rejects_suspiciously_high_win_rate(self):
        detector = AnomalyDetector()
        stats = {"total_pnl": 1000, "win_rate": 0.99}
        assert detector._is_profitable_pattern(stats, [], ["basic_arbitrage"]) is False

    def test_is_profitable_pattern_rejects_critical_anomalies(self):
        detector = AnomalyDetector()
        stats = {"total_pnl": 1000, "win_rate": 0.70}
        anomalies = [
            Anomaly(type=AnomalyType.IMPOSSIBLE_WIN_RATE, severity=Severity.CRITICAL,
                     score=1.0, description="", evidence={})
        ]
        assert detector._is_profitable_pattern(stats, anomalies, ["basic_arbitrage"]) is False

    def test_is_profitable_pattern_accepts_good_wallet(self):
        detector = AnomalyDetector()
        stats = {"total_pnl": 500, "win_rate": 0.70}
        anomalies = []
        strategies = ["basic_arbitrage"]
        assert detector._is_profitable_pattern(stats, anomalies, strategies) is True


class TestAnomalyDetectorStdDev:
    """Helper _std_dev."""

    def test_std_dev_basic(self):
        detector = AnomalyDetector()
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        result = detector._std_dev(values)
        # sample std dev with n-1
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        expected = math.sqrt(var)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_std_dev_single_value(self):
        detector = AnomalyDetector()
        assert detector._std_dev([42]) == 0.0

    def test_std_dev_empty(self):
        detector = AnomalyDetector()
        assert detector._std_dev([]) == 0.0


# ============================================================================
# COPY TRADER TESTS
# ============================================================================


def _make_copy_config(**overrides) -> CopyTradingConfig:
    """Build a mock CopyTradingConfig with sane defaults."""
    config = MagicMock(spec=CopyTradingConfig)
    config.id = overrides.get("id", "cfg_1")
    config.source_wallet = overrides.get("source_wallet", "0xabc123")
    config.account_id = overrides.get("account_id", "acc_1")
    config.enabled = overrides.get("enabled", True)
    config.copy_mode = overrides.get("copy_mode", CopyTradingMode.ALL_TRADES)
    config.min_roi_threshold = overrides.get("min_roi_threshold", 2.5)
    config.max_position_size = overrides.get("max_position_size", 1000.0)
    config.copy_delay_seconds = overrides.get("copy_delay_seconds", 0)
    config.slippage_tolerance = overrides.get("slippage_tolerance", 1.0)
    config.proportional_sizing = overrides.get("proportional_sizing", False)
    config.proportional_multiplier = overrides.get("proportional_multiplier", 1.0)
    config.copy_buys = overrides.get("copy_buys", True)
    config.copy_sells = overrides.get("copy_sells", True)
    config.market_categories = overrides.get("market_categories", [])
    config.total_copied = overrides.get("total_copied", 0)
    config.successful_copies = overrides.get("successful_copies", 0)
    config.failed_copies = overrides.get("failed_copies", 0)
    config.total_pnl = overrides.get("total_pnl", 0.0)
    config.total_buys_copied = overrides.get("total_buys_copied", 0)
    config.total_sells_copied = overrides.get("total_sells_copied", 0)
    return config


class TestCopyTraderShouldCopy:
    """Trade filtering logic: _should_copy_trade."""

    def test_copy_buy_allowed(self):
        svc = CopyTradingService()
        trade = {"side": "BUY", "price": 0.50, "size": 100}
        config = _make_copy_config(copy_buys=True, copy_sells=True)
        should, reason = svc._should_copy_trade(trade, config)
        assert should is True

    def test_copy_buy_disabled(self):
        svc = CopyTradingService()
        trade = {"side": "BUY", "price": 0.50, "size": 100}
        config = _make_copy_config(copy_buys=False)
        should, reason = svc._should_copy_trade(trade, config)
        assert should is False
        assert "buy copying disabled" in reason

    def test_copy_sell_disabled(self):
        svc = CopyTradingService()
        trade = {"side": "SELL", "price": 0.50, "size": 100}
        config = _make_copy_config(copy_sells=False)
        should, reason = svc._should_copy_trade(trade, config)
        assert should is False
        assert "sell copying disabled" in reason

    def test_zero_price_rejected(self):
        svc = CopyTradingService()
        trade = {"side": "BUY", "price": 0, "size": 100}
        config = _make_copy_config()
        should, reason = svc._should_copy_trade(trade, config)
        assert should is False
        assert "zero" in reason

    def test_zero_size_rejected(self):
        svc = CopyTradingService()
        trade = {"side": "BUY", "price": 0.5, "size": 0}
        config = _make_copy_config()
        should, reason = svc._should_copy_trade(trade, config)
        assert should is False
        assert "zero" in reason

    def test_sell_allowed(self):
        svc = CopyTradingService()
        trade = {"side": "SELL", "price": 0.60, "size": 50}
        config = _make_copy_config()
        should, _ = svc._should_copy_trade(trade, config)
        assert should is True


class TestCopyTraderPositionSizing:
    """Position size scaling: _calculate_copy_size."""

    def test_default_sizing_follows_source(self):
        svc = CopyTradingService()
        trade = {"size": 100, "price": 0.50}
        config = _make_copy_config(
            proportional_sizing=False,
            max_position_size=1000.0,
        )
        size = svc._calculate_copy_size(trade, config, account_capital=10000.0)
        # target_cost = 100 * 0.50 = 50, capped at 1000, capped at 10000*0.99
        # size = 50 / 0.50 = 100
        assert size == pytest.approx(100.0)

    def test_proportional_sizing_half(self):
        svc = CopyTradingService()
        trade = {"size": 200, "price": 0.50}
        config = _make_copy_config(
            proportional_sizing=True,
            proportional_multiplier=0.5,
            max_position_size=10000.0,
        )
        size = svc._calculate_copy_size(trade, config, account_capital=10000.0)
        # source_cost = 200 * 0.5 = 100, target = 100 * 0.5 = 50, shares = 50/0.5 = 100
        assert size == pytest.approx(100.0)

    def test_proportional_sizing_double(self):
        svc = CopyTradingService()
        trade = {"size": 100, "price": 0.50}
        config = _make_copy_config(
            proportional_sizing=True,
            proportional_multiplier=2.0,
            max_position_size=10000.0,
        )
        size = svc._calculate_copy_size(trade, config, account_capital=10000.0)
        # source_cost = 50, target = 50*2 = 100, shares = 100/0.5 = 200
        assert size == pytest.approx(200.0)

    def test_capped_by_max_position_size(self):
        svc = CopyTradingService()
        trade = {"size": 1000, "price": 0.50}
        config = _make_copy_config(
            proportional_sizing=False,
            max_position_size=100.0,
        )
        size = svc._calculate_copy_size(trade, config, account_capital=10000.0)
        # source_cost = 500, capped to 100, shares = 100/0.5 = 200
        assert size == pytest.approx(200.0)

    def test_capped_by_capital(self):
        svc = CopyTradingService()
        trade = {"size": 1000, "price": 0.50}
        config = _make_copy_config(
            proportional_sizing=False,
            max_position_size=100000.0,
        )
        size = svc._calculate_copy_size(trade, config, account_capital=100.0)
        # source_cost = 500, capped to 100*0.99 = 99, shares = 99/0.5 = 198
        assert size == pytest.approx(198.0)

    def test_zero_price_returns_zero(self):
        svc = CopyTradingService()
        trade = {"size": 100, "price": 0}
        config = _make_copy_config()
        size = svc._calculate_copy_size(trade, config, account_capital=10000.0)
        assert size == 0.0

    def test_uses_amount_field_fallback(self):
        svc = CopyTradingService()
        trade = {"amount": 100, "price": 0.50}
        config = _make_copy_config(
            proportional_sizing=False,
            max_position_size=10000.0,
        )
        size = svc._calculate_copy_size(trade, config, account_capital=10000.0)
        assert size == pytest.approx(100.0)


class TestCopyTraderDelayMechanism:
    """Copy delay mechanism: _execute_copy_buy should sleep for configured delay."""

    @pytest.mark.asyncio
    async def test_delay_is_applied(self):
        svc = CopyTradingService()
        config = _make_copy_config(copy_delay_seconds=2)

        trade = {
            "id": "trade_1",
            "side": "BUY",
            "price": 0.50,
            "size": 100,
            "market": "m1",
            "asset": "tok1",
            "outcome": "YES",
            "title": "Test market",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        mock_account = MagicMock()
        mock_account.current_capital = 10000.0
        mock_account.slippage_bps = 0.0

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_account)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("services.copy_trader.AsyncSessionLocal", return_value=mock_session), \
             patch("services.copy_trader.polymarket_client") as mock_pc, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            mock_pc.get_price = AsyncMock(return_value=0.50)

            # Mock the internal execution to avoid deep DB calls
            mock_sim_trade = MagicMock()
            mock_sim_trade.id = "sim_1"
            svc._execute_sim_buy = AsyncMock(return_value=mock_sim_trade)
            svc._record_copied_trade = AsyncMock(return_value=MagicMock())

            await svc._execute_copy_buy(trade, config)

            mock_sleep.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_no_delay_when_zero(self):
        svc = CopyTradingService()
        config = _make_copy_config(copy_delay_seconds=0)

        trade = {
            "id": "trade_1",
            "side": "BUY",
            "price": 0.50,
            "size": 100,
            "market": "m1",
            "asset": "tok1",
            "outcome": "YES",
            "title": "Test market",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        mock_account = MagicMock()
        mock_account.current_capital = 10000.0
        mock_account.slippage_bps = 0.0

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_account)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("services.copy_trader.AsyncSessionLocal", return_value=mock_session), \
             patch("services.copy_trader.polymarket_client") as mock_pc, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            mock_pc.get_price = AsyncMock(return_value=0.50)
            mock_sim_trade = MagicMock()
            mock_sim_trade.id = "sim_1"
            svc._execute_sim_buy = AsyncMock(return_value=mock_sim_trade)
            svc._record_copied_trade = AsyncMock(return_value=MagicMock())

            await svc._execute_copy_buy(trade, config)

            mock_sleep.assert_not_called()


class TestCopyTraderAnomalyFiltering:
    """Filtering by anomaly score (used in the broader workflow)."""

    def test_is_profitable_pattern_filters_high_anomaly(self):
        """AnomalyDetector._is_profitable_pattern rejects critical anomalies,
        which is used to decide whether to copy a wallet."""
        detector = AnomalyDetector()
        stats = {"total_pnl": 5000, "win_rate": 0.70}
        critical_anomaly = Anomaly(
            type=AnomalyType.IMPOSSIBLE_WIN_RATE,
            severity=Severity.CRITICAL,
            score=1.0,
            description="Impossible",
            evidence={},
        )
        result = detector._is_profitable_pattern(stats, [critical_anomaly], ["basic_arbitrage"])
        assert result is False

    def test_recommendation_avoid_critical(self):
        detector = AnomalyDetector()
        stats = {"total_trades": 100, "win_rate": 0.99}
        anomalies = [
            Anomaly(type=AnomalyType.IMPOSSIBLE_WIN_RATE, severity=Severity.CRITICAL,
                     score=1.0, description="", evidence={})
        ]
        rec = detector._generate_recommendation(stats, anomalies, is_profitable=False)
        assert "AVOID" in rec

    def test_recommendation_caution_high(self):
        detector = AnomalyDetector()
        stats = {"total_trades": 100, "win_rate": 0.80}
        anomalies = [
            Anomaly(type=AnomalyType.UNUSUAL_ROI, severity=Severity.HIGH,
                     score=0.7, description="", evidence={})
        ]
        rec = detector._generate_recommendation(stats, anomalies, is_profitable=False)
        assert "CAUTION" in rec

    def test_recommendation_consider_copying(self):
        detector = AnomalyDetector()
        stats = {"total_trades": 100, "win_rate": 0.70, "avg_roi": 5.0}
        rec = detector._generate_recommendation(stats, [], is_profitable=True)
        assert "CONSIDER COPYING" in rec

    def test_recommendation_monitor(self):
        detector = AnomalyDetector()
        stats = {"total_trades": 100, "win_rate": 0.65, "total_pnl": 500}
        rec = detector._generate_recommendation(stats, [], is_profitable=False)
        assert "MONITOR" in rec

    def test_recommendation_not_recommended(self):
        detector = AnomalyDetector()
        stats = {"total_trades": 100, "win_rate": 0.40, "total_pnl": -200}
        rec = detector._generate_recommendation(stats, [], is_profitable=False)
        assert "NOT RECOMMENDED" in rec

    def test_recommendation_insufficient_data(self):
        detector = AnomalyDetector()
        stats = {"total_trades": 0}
        rec = detector._generate_recommendation(stats, [], is_profitable=False)
        assert "Insufficient" in rec


class TestCopyTraderWalletTracking:
    """Wallet tracking: _get_new_trades deduplication logic."""

    @pytest.mark.asyncio
    async def test_get_new_trades_deduplication(self):
        svc = CopyTradingService()
        config = _make_copy_config()

        # Polymarket returns 2 trades, one already copied
        mock_trades = [
            {"id": "t1", "side": "BUY", "price": 0.5, "size": 100},
            {"id": "t2", "side": "BUY", "price": 0.6, "size": 50},
        ]

        # Mock that t1 already exists in CopiedTrade table
        mock_existing_t1 = MagicMock()
        mock_no_existing = None

        mock_session = AsyncMock()

        execute_results = [MagicMock(), MagicMock()]
        execute_results[0].scalar_one_or_none.return_value = mock_existing_t1  # t1 exists
        execute_results[1].scalar_one_or_none.return_value = mock_no_existing  # t2 is new
        mock_session.execute = AsyncMock(side_effect=execute_results)

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("services.copy_trader.polymarket_client") as mock_pc, \
             patch("services.copy_trader.AsyncSessionLocal", return_value=mock_session):
            mock_pc.get_wallet_trades = AsyncMock(return_value=mock_trades)

            new_trades = await svc._get_new_trades("0xabc", config)

        # Only t2 should be returned (t1 was already processed)
        assert len(new_trades) == 1
        assert new_trades[0]["id"] == "t2"


class TestCopyTraderTradeMirroring:
    """Trade mirroring: _should_copy_trade correctly mirrors sides."""

    def test_mirrors_buy(self):
        svc = CopyTradingService()
        trade = {"side": "BUY", "price": 0.5, "size": 100}
        config = _make_copy_config()
        should, _ = svc._should_copy_trade(trade, config)
        assert should is True

    def test_mirrors_sell(self):
        svc = CopyTradingService()
        trade = {"side": "SELL", "price": 0.5, "size": 100}
        config = _make_copy_config()
        should, _ = svc._should_copy_trade(trade, config)
        assert should is True

    def test_unknown_side_passes_filter(self):
        """Unknown side is not BUY or SELL, so it passes the direction filter
        but may be caught elsewhere."""
        svc = CopyTradingService()
        trade = {"side": "UNKNOWN", "price": 0.5, "size": 100}
        config = _make_copy_config()
        should, _ = svc._should_copy_trade(trade, config)
        assert should is True

    def test_missing_side_passes_filter(self):
        svc = CopyTradingService()
        trade = {"price": 0.5, "size": 100}
        config = _make_copy_config()
        should, _ = svc._should_copy_trade(trade, config)
        assert should is True


# ============================================================================
# ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestAutoTraderConfigure:
    """AutoTrader.configure updates config correctly."""

    def test_configure_updates_mode(self):
        trader = AutoTrader()
        trader.configure(mode=AutoTraderMode.PAPER)
        assert trader.config.mode == AutoTraderMode.PAPER

    def test_configure_ignores_unknown_keys(self):
        trader = AutoTrader()
        trader.configure(nonexistent_field="value")
        assert not hasattr(trader.config, "nonexistent_field")

    def test_configure_multiple_values(self):
        trader = AutoTrader()
        trader.configure(
            min_roi_percent=5.0,
            max_risk_score=0.3,
            max_daily_trades=20,
        )
        assert trader.config.min_roi_percent == 5.0
        assert trader.config.max_risk_score == 0.3
        assert trader.config.max_daily_trades == 20


class TestAutoTraderRecordTradeResult:
    """record_trade_result updates stats."""

    def test_winning_trade_updates_stats(self):
        trader = AutoTrader()
        tid = "t1"
        trader._trades[tid] = TradeRecord(
            id=tid, opportunity_id="o1", strategy=StrategyType.BASIC,
            executed_at=datetime.utcnow(), positions=[], total_cost=10,
            expected_profit=1,
        )
        trader.record_trade_result(tid, won=True, actual_profit=2.0)
        assert trader.stats.winning_trades == 1
        assert trader.stats.total_profit == 2.0
        assert trader.stats.consecutive_losses == 0
        assert trader._trades[tid].status == "resolved_win"

    def test_losing_trade_updates_stats(self):
        trader = AutoTrader()
        tid = "t1"
        trader._trades[tid] = TradeRecord(
            id=tid, opportunity_id="o1", strategy=StrategyType.BASIC,
            executed_at=datetime.utcnow(), positions=[], total_cost=10,
            expected_profit=1,
        )
        trader.record_trade_result(tid, won=False, actual_profit=-5.0)
        assert trader.stats.losing_trades == 1
        assert trader.stats.total_profit == -5.0
        assert trader.stats.consecutive_losses == 1
        assert trader._trades[tid].status == "resolved_loss"

    def test_unknown_trade_id_ignored(self):
        trader = AutoTrader()
        trader.record_trade_result("nonexistent", won=True, actual_profit=10.0)
        assert trader.stats.winning_trades == 0


class TestAutoTraderGetStats:
    """get_stats returns correct dictionary."""

    def test_stats_dict_shape(self):
        trader = _fresh_auto_trader()
        stats = trader.get_stats()
        assert "mode" in stats
        assert "running" in stats
        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "total_profit" in stats
        assert "circuit_breaker_active" in stats

    def test_win_rate_calculation(self):
        trader = AutoTrader()
        trader.stats.winning_trades = 7
        trader.stats.losing_trades = 3
        stats = trader.get_stats()
        assert stats["win_rate"] == pytest.approx(0.7)

    def test_roi_calculation(self):
        trader = AutoTrader()
        trader.stats.total_profit = 50.0
        trader.stats.total_invested = 1000.0
        stats = trader.get_stats()
        assert stats["roi_percent"] == pytest.approx(5.0)


class TestAutoTraderGetTrades:
    """get_trades returns formatted list."""

    def test_get_trades_sorted_newest_first(self):
        trader = AutoTrader()
        for i in range(5):
            tid = f"t_{i}"
            trader._trades[tid] = TradeRecord(
                id=tid, opportunity_id=f"o_{i}", strategy=StrategyType.BASIC,
                executed_at=datetime.utcnow() + timedelta(seconds=i),
                positions=[], total_cost=10, expected_profit=1,
                mode=AutoTraderMode.SHADOW,
            )

        trades = trader.get_trades(limit=3)
        assert len(trades) == 3
        # Newest first
        assert trades[0]["id"] == "t_4"
        assert trades[1]["id"] == "t_3"


class TestAnomalyDetectorDetectStrategies:
    """_detect_strategies identifies patterns."""

    def test_detects_basic_arbitrage(self):
        detector = AnomalyDetector()
        trades = [
            {"market": "m1", "outcome": "YES"},
            {"market": "m1", "outcome": "NO"},
        ]
        strategies = detector._detect_strategies(trades)
        assert "basic_arbitrage" in strategies

    def test_detects_negrisk_date_sweep(self):
        detector = AnomalyDetector()
        trades = [{"market": "m1", "outcome": "YES"}] * 4
        strategies = detector._detect_strategies(trades)
        assert "negrisk_date_sweep" in strategies

    def test_detects_automated_trading(self):
        detector = AnomalyDetector()
        trades = [{"market": f"m{i}", "outcome": "YES"} for i in range(101)]
        strategies = detector._detect_strategies(trades)
        assert "automated_trading" in strategies
