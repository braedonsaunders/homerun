"""Unit tests for accuracy-first wallet discovery metrics synthesis."""

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.wallet_discovery import (  # noqa: E402
    METRICS_SOURCE_VERSION,
    WalletDiscoveryEngine,
)


class TestWalletDiscoveryAccuracy:
    def test_accuracy_first_stats_override_inconsistent_trade_defaults(self):
        engine = WalletDiscoveryEngine()
        base_stats = engine._empty_stats()
        base_stats.update(
            {
                "wins": 19,
                "losses": 1,
                "win_rate": 0.95,
                "total_pnl": -20000.0,
                "total_invested": 40000.0,
                "days_active": 30,
            }
        )
        pnl_snapshot = {
            "total_trades": 120,
            "open_positions": 3,
            "total_invested": 12000.0,
            "total_returned": 14800.0,
            "realized_pnl": 2600.0,
            "unrealized_pnl": 200.0,
            "total_pnl": 2800.0,
        }
        closed_positions = [
            {"market": "m1", "realizedPnl": 300.0, "initialValue": 1000.0},
            {"market": "m2", "realizedPnl": 250.0, "initialValue": 900.0},
            {"market": "m3", "realizedPnl": -100.0, "initialValue": 800.0},
            {"market": "m4", "realizedPnl": 150.0, "initialValue": 700.0},
        ]

        merged = engine._build_accuracy_first_stats(
            base_stats=base_stats,
            pnl_snapshot=pnl_snapshot,
            closed_positions=closed_positions,
        )

        assert merged["total_pnl"] == 2800.0
        assert merged["total_trades"] == 120
        assert merged["wins"] == 3
        assert merged["losses"] == 1
        assert pytest.approx(merged["win_rate"], rel=1e-6) == 0.75

    @pytest.mark.asyncio
    async def test_analyze_wallet_sets_accuracy_metrics_source_version(self):
        engine = WalletDiscoveryEngine()

        async def fake_get_wallet_trades(_address: str, limit: int = 100):
            assert limit >= 500
            return [
                {"market": "m1", "side": "BUY", "size": 10, "price": 0.4, "timestamp": 1700000000},
                {"market": "m1", "side": "SELL", "size": 10, "price": 0.6, "timestamp": 1700003600},
                {"market": "m2", "side": "BUY", "size": 12, "price": 0.35, "timestamp": 1700007200},
                {"market": "m2", "side": "SELL", "size": 12, "price": 0.55, "timestamp": 1700010800},
                {"market": "m3", "side": "BUY", "size": 8, "price": 0.5, "timestamp": 1700014400},
            ]

        async def fake_get_wallet_positions(_address: str):
            return []

        async def fake_get_user_profile(_address: str):
            return {"username": "alpha_wallet"}

        async def fake_get_wallet_pnl(_address: str, time_period: str = "ALL"):
            assert time_period == "ALL"
            return {
                "total_trades": 75,
                "open_positions": 0,
                "total_invested": 10000.0,
                "total_returned": 11400.0,
                "realized_pnl": 1400.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 1400.0,
            }

        async def fake_get_closed_positions_paginated(_address: str, max_positions: int = 200):
            assert max_positions >= 200
            return [
                {"market": "m1", "realizedPnl": 250.0, "initialValue": 1200.0},
                {"market": "m2", "realizedPnl": 190.0, "initialValue": 1000.0},
                {"market": "m3", "realizedPnl": -60.0, "initialValue": 950.0},
                {"market": "m4", "realizedPnl": 210.0, "initialValue": 900.0},
                {"market": "m5", "realizedPnl": 110.0, "initialValue": 820.0},
            ]

        engine.client.get_wallet_trades = fake_get_wallet_trades
        engine.client.get_wallet_positions = fake_get_wallet_positions
        engine.client.get_user_profile = fake_get_user_profile
        engine.client.get_wallet_pnl = fake_get_wallet_pnl
        engine.client.get_closed_positions_paginated = fake_get_closed_positions_paginated

        profile = await engine.analyze_wallet("0xabc123")

        assert profile is not None
        assert profile["metrics_source_version"] == METRICS_SOURCE_VERSION
        assert profile["username"] == "alpha_wallet"
        assert profile["total_pnl"] == 1400.0
        assert profile["total_trades"] == 75
