import sys
from pathlib import Path
from datetime import datetime

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.fifo_pnl import (
    TradeLot,
    FIFOResult,
    compute_fifo_pnl,
    compute_fifo_pnl_multi_market,
    _parse_ts,
)


class TestParseTimestamp:
    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        assert _parse_ts(dt) == dt

    def test_epoch_seconds(self):
        result = _parse_ts(1700000000)
        assert isinstance(result, datetime)
        assert result.year == 2023

    def test_epoch_millis(self):
        result = _parse_ts(1700000000000)
        assert isinstance(result, datetime)
        assert result.year == 2023

    def test_iso_string_z(self):
        result = _parse_ts("2024-01-15T12:00:00Z")
        assert result == datetime(2024, 1, 15, 12, 0, 0)

    def test_iso_string_fractional(self):
        result = _parse_ts("2024-01-15T12:00:00.123456Z")
        assert result is not None
        assert result.year == 2024

    def test_numeric_string(self):
        result = _parse_ts("1700000000")
        assert isinstance(result, datetime)

    def test_none(self):
        assert _parse_ts(None) is None

    def test_empty_string(self):
        assert _parse_ts("") is None

    def test_garbage_string(self):
        assert _parse_ts("not-a-date") is None


class TestComputeFifoPnl:
    def test_empty_trades(self):
        result = compute_fifo_pnl([], "m1")
        assert result.total_realized_pnl == 0.0
        assert result.closed_lots == []
        assert result.open_lots == []
        assert result.total_trades == 0
        assert result.accuracy == 0.0

    def test_single_buy_no_sell(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 0
        assert len(result.open_lots) == 1
        assert result.open_lots[0].size == 100
        assert result.open_lots[0].entry_price == 0.40
        assert result.total_realized_pnl == 0.0
        assert result.total_trades == 1

    def test_simple_buy_sell_profit(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"timestamp": 1700003600, "price": 0.60, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 1
        assert len(result.open_lots) == 0
        assert result.total_realized_pnl == pytest.approx(20.0)  # (0.60 - 0.40) * 100
        assert result.winning_lots == 1
        assert result.losing_lots == 0
        assert result.accuracy == 1.0
        assert result.avg_hold_seconds == 3600.0

    def test_simple_buy_sell_loss(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.60, "size": 50, "side": "BUY"},
            {"timestamp": 1700003600, "price": 0.30, "size": 50, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 1
        assert result.total_realized_pnl == pytest.approx(-15.0)  # (0.30 - 0.60) * 50
        assert result.winning_lots == 0
        assert result.losing_lots == 1
        assert result.accuracy == 0.0

    def test_fifo_order_multiple_buys(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.30, "size": 50, "side": "BUY"},
            {"timestamp": 1700001000, "price": 0.50, "size": 50, "side": "BUY"},
            {"timestamp": 1700002000, "price": 0.60, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 2
        assert len(result.open_lots) == 0
        # First lot: (0.60 - 0.30) * 50 = 15.0
        assert result.closed_lots[0].realized_pnl == pytest.approx(15.0)
        assert result.closed_lots[0].entry_price == 0.30
        # Second lot: (0.60 - 0.50) * 50 = 5.0
        assert result.closed_lots[1].realized_pnl == pytest.approx(5.0)
        assert result.closed_lots[1].entry_price == 0.50
        assert result.total_realized_pnl == pytest.approx(20.0)

    def test_partial_fill(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"timestamp": 1700003600, "price": 0.60, "size": 60, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 1
        assert result.closed_lots[0].size == pytest.approx(60)
        assert result.closed_lots[0].realized_pnl == pytest.approx(12.0)  # (0.60 - 0.40) * 60
        assert len(result.open_lots) == 1
        assert result.open_lots[0].size == pytest.approx(40)
        assert result.open_lots[0].entry_price == 0.40

    def test_sell_matches_multiple_buys(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.30, "size": 30, "side": "BUY"},
            {"timestamp": 1700001000, "price": 0.40, "size": 30, "side": "BUY"},
            {"timestamp": 1700002000, "price": 0.50, "size": 30, "side": "BUY"},
            {"timestamp": 1700003000, "price": 0.70, "size": 80, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        # Sell 80 matches: 30 from first buy, 30 from second, 20 from third
        assert len(result.closed_lots) == 3
        assert result.closed_lots[0].size == pytest.approx(30)
        assert result.closed_lots[0].realized_pnl == pytest.approx((0.70 - 0.30) * 30)
        assert result.closed_lots[1].size == pytest.approx(30)
        assert result.closed_lots[1].realized_pnl == pytest.approx((0.70 - 0.40) * 30)
        assert result.closed_lots[2].size == pytest.approx(20)
        assert result.closed_lots[2].realized_pnl == pytest.approx((0.70 - 0.50) * 20)
        # 10 shares remain open from the third buy
        assert len(result.open_lots) == 1
        assert result.open_lots[0].size == pytest.approx(10)
        assert result.open_lots[0].entry_price == 0.50

    def test_multiple_sells(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"timestamp": 1700001000, "price": 0.50, "size": 40, "side": "SELL"},
            {"timestamp": 1700002000, "price": 0.60, "size": 60, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 2
        # First sell: (0.50 - 0.40) * 40 = 4.0
        assert result.closed_lots[0].realized_pnl == pytest.approx(4.0)
        # Second sell: (0.60 - 0.40) * 60 = 12.0
        assert result.closed_lots[1].realized_pnl == pytest.approx(12.0)
        assert result.total_realized_pnl == pytest.approx(16.0)
        assert len(result.open_lots) == 0

    def test_sell_with_no_matching_buys_ignored(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.50, "size": 50, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 0
        assert len(result.open_lots) == 0
        assert result.total_realized_pnl == 0.0
        assert result.total_trades == 1

    def test_prediction_market_win_settlement(self):
        # Buy YES at 0.40, market resolves YES (sell at 1.0)
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 200, "side": "BUY"},
            {"timestamp": 1700100000, "price": 1.00, "size": 200, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.total_realized_pnl == pytest.approx(120.0)  # (1.0 - 0.4) * 200
        assert result.winning_lots == 1

    def test_prediction_market_loss_settlement(self):
        # Buy YES at 0.70, market resolves NO (sell at 0.0)
        trades = [
            {"timestamp": 1700000000, "price": 0.70, "size": 100, "side": "BUY"},
            {"timestamp": 1700100000, "price": 0.00, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.total_realized_pnl == pytest.approx(-70.0)
        assert result.losing_lots == 1

    def test_breakeven_lot(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.50, "size": 100, "side": "BUY"},
            {"timestamp": 1700003600, "price": 0.50, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.total_realized_pnl == pytest.approx(0.0)
        assert result.winning_lots == 0
        assert result.losing_lots == 0
        assert result.accuracy == 0.0  # 0 winners / 1 closed = 0

    def test_mixed_wins_and_losses(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 50, "side": "BUY"},
            {"timestamp": 1700001000, "price": 0.60, "size": 50, "side": "SELL"},
            {"timestamp": 1700002000, "price": 0.70, "size": 50, "side": "BUY"},
            {"timestamp": 1700003000, "price": 0.50, "size": 50, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.winning_lots == 1
        assert result.losing_lots == 1
        assert result.accuracy == pytest.approx(0.5)
        assert result.total_realized_pnl == pytest.approx(0.0)  # +10 - 10 = 0

    def test_iso_timestamps(self):
        trades = [
            {"timestamp": "2024-01-15T10:00:00Z", "price": 0.40, "size": 100, "side": "BUY"},
            {"timestamp": "2024-01-15T11:00:00Z", "price": 0.60, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.total_realized_pnl == pytest.approx(20.0)
        assert result.avg_hold_seconds == 3600.0

    def test_created_at_key(self):
        trades = [
            {"created_at": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"created_at": 1700003600, "price": 0.60, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.total_realized_pnl == pytest.approx(20.0)

    def test_amount_key(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "amount": 100, "side": "BUY"},
            {"timestamp": 1700003600, "price": 0.60, "amount": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert result.total_realized_pnl == pytest.approx(20.0)

    def test_skips_invalid_trades(self):
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"price": 0.50, "size": 50, "side": "SELL"},  # no timestamp
            {"timestamp": 1700003600, "price": 0.60, "size": 0, "side": "SELL"},  # zero size
            {"timestamp": 1700003600, "price": 0.60, "size": 100, "side": "HOLD"},  # bad side
            {"timestamp": 1700007200, "price": 0.60, "size": 100, "side": "SELL"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 1
        assert result.total_realized_pnl == pytest.approx(20.0)

    def test_trades_sorted_by_timestamp(self):
        # Pass trades out of order; FIFO should sort them
        trades = [
            {"timestamp": 1700003600, "price": 0.60, "size": 100, "side": "SELL"},
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
        ]
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 1
        assert result.total_realized_pnl == pytest.approx(20.0)

    def test_large_number_of_trades(self):
        trades = []
        for i in range(500):
            ts = 1700000000 + i * 100
            trades.append({"timestamp": ts, "price": 0.40, "size": 10, "side": "BUY"})
            trades.append({"timestamp": ts + 50, "price": 0.60, "size": 10, "side": "SELL"})
        result = compute_fifo_pnl(trades, "m1")
        assert len(result.closed_lots) == 500
        assert result.total_realized_pnl == pytest.approx(500 * 2.0)  # (0.60-0.40)*10 per lot
        assert result.winning_lots == 500
        assert result.accuracy == 1.0


class TestComputeFifoPnlMultiMarket:
    def test_empty_dict(self):
        result = compute_fifo_pnl_multi_market({})
        assert result == {}

    def test_multiple_markets(self):
        trades_by_market = {
            "m1": [
                {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
                {"timestamp": 1700003600, "price": 0.60, "size": 100, "side": "SELL"},
            ],
            "m2": [
                {"timestamp": 1700000000, "price": 0.50, "size": 50, "side": "BUY"},
                {"timestamp": 1700003600, "price": 0.30, "size": 50, "side": "SELL"},
            ],
        }
        results = compute_fifo_pnl_multi_market(trades_by_market)
        assert len(results) == 2
        assert results["m1"].total_realized_pnl == pytest.approx(20.0)
        assert results["m2"].total_realized_pnl == pytest.approx(-10.0)
        assert results["m1"].winning_lots == 1
        assert results["m2"].losing_lots == 1

    def test_market_ids_preserved(self):
        trades_by_market = {
            "0xabc123": [
                {"timestamp": 1700000000, "price": 0.40, "size": 10, "side": "BUY"},
            ],
        }
        results = compute_fifo_pnl_multi_market(trades_by_market)
        assert "0xabc123" in results
        assert results["0xabc123"].open_lots[0].market_id == "0xabc123"


class TestEnrichStatsWithFifo:
    def test_fifo_enrichment_updates_stats(self):
        from services.wallet_discovery import WalletDiscoveryEngine

        engine = WalletDiscoveryEngine()
        stats = engine._empty_stats()
        trades = [
            {"market": "m1", "timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"market": "m1", "timestamp": 1700003600, "price": 0.60, "size": 100, "side": "SELL"},
            {"market": "m2", "timestamp": 1700000000, "price": 0.50, "size": 50, "side": "BUY"},
            {"market": "m2", "timestamp": 1700003600, "price": 0.30, "size": 50, "side": "SELL"},
        ]
        enriched = engine._enrich_stats_with_fifo(stats, trades)
        assert enriched["wins"] == 1
        assert enriched["losses"] == 1
        assert enriched["win_rate"] == pytest.approx(0.5)
        assert enriched["fifo_realized_pnl"] == pytest.approx(10.0)
        assert enriched["fifo_closed_lots"] == 2
        assert enriched["fifo_open_lots"] == 0
        assert enriched["fifo_accuracy"] == pytest.approx(0.5)
        assert enriched["avg_hold_time_hours"] == pytest.approx(1.0)

    def test_fifo_enrichment_empty_trades(self):
        from services.wallet_discovery import WalletDiscoveryEngine

        engine = WalletDiscoveryEngine()
        stats = engine._empty_stats()
        enriched = engine._enrich_stats_with_fifo(stats, [])
        assert enriched["wins"] == 0
        assert enriched["losses"] == 0
        assert "fifo_realized_pnl" not in enriched

    def test_fifo_enrichment_no_market_id(self):
        from services.wallet_discovery import WalletDiscoveryEngine

        engine = WalletDiscoveryEngine()
        stats = engine._empty_stats()
        trades = [
            {"timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
        ]
        enriched = engine._enrich_stats_with_fifo(stats, trades)
        assert enriched["wins"] == 0
        assert "fifo_realized_pnl" not in enriched

    def test_fifo_enrichment_only_buys(self):
        from services.wallet_discovery import WalletDiscoveryEngine

        engine = WalletDiscoveryEngine()
        stats = engine._empty_stats()
        trades = [
            {"market": "m1", "timestamp": 1700000000, "price": 0.40, "size": 100, "side": "BUY"},
            {"market": "m2", "timestamp": 1700001000, "price": 0.50, "size": 50, "side": "BUY"},
        ]
        enriched = engine._enrich_stats_with_fifo(stats, trades)
        # No closed lots, so wins/losses stay at 0 from empty_stats
        assert enriched["fifo_closed_lots"] == 0
        assert enriched["fifo_open_lots"] == 2
        assert enriched["fifo_realized_pnl"] == pytest.approx(0.0)
        assert enriched["wins"] == 0
        assert enriched["losses"] == 0
