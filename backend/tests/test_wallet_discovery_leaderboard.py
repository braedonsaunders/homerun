import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, DiscoveredWallet  # noqa: E402
import services.wallet_discovery as wallet_discovery_module  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


@pytest.mark.asyncio
async def test_get_leaderboard_accepts_decimal_rolling_trade_counts(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "wallet_discovery_leaderboard")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    now = utcnow()

    try:
        async with session_factory() as session:
            session.add_all(
                [
                    DiscoveredWallet(
                        address="wallet_alpha",
                        discovered_at=now,
                        total_trades=120,
                        total_pnl=25000.0,
                        rolling_trade_count={"30d": 49.0},
                    ),
                    DiscoveredWallet(
                        address="wallet_beta",
                        discovered_at=now,
                        total_trades=90,
                        total_pnl=18000.0,
                        rolling_trade_count={"30d": "7.0"},
                    ),
                    DiscoveredWallet(
                        address="wallet_gamma",
                        discovered_at=now,
                        total_trades=40,
                        total_pnl=15000.0,
                        rolling_trade_count={"30d": 0},
                    ),
                ]
            )
            await session.commit()

        result = await service.get_leaderboard(
            limit=10,
            offset=0,
            min_trades=10,
            min_pnl=10000.0,
            sort_by="total_trades",
            sort_dir="desc",
            window_key="30d",
        )

        assert result["total"] == 2
        assert result["window_key"] == "30d"
        assert [wallet["address"] for wallet in result["wallets"]] == ["wallet_alpha", "wallet_beta"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_leaderboard_win_rate_sort_requires_resolved_sample(monkeypatch):
    engine, session_factory = await build_postgres_session_factory(Base, "wallet_discovery_win_rate_sample")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    now = utcnow()

    try:
        async with session_factory() as session:
            session.add_all(
                [
                    DiscoveredWallet(
                        address="wallet_low_sample",
                        discovered_at=now,
                        last_analyzed_at=now,
                        total_trades=194,
                        wins=1,
                        losses=0,
                        win_rate=1.0,
                        total_pnl=27330.95,
                    ),
                    DiscoveredWallet(
                        address="wallet_real_sample",
                        discovered_at=now,
                        last_analyzed_at=now,
                        total_trades=362,
                        wins=102,
                        losses=21,
                        win_rate=102 / 123,
                        total_pnl=22893.88,
                    ),
                ]
            )
            await session.commit()

        result = await service.get_leaderboard(
            limit=10,
            offset=0,
            min_trades=100,
            min_pnl=10000.0,
            sort_by="win_rate",
            sort_dir="desc",
        )

        assert [wallet["address"] for wallet in result["wallets"]] == ["wallet_real_sample"]
        assert result["wallets"][0]["resolved_positions"] == 123
    finally:
        await engine.dispose()
