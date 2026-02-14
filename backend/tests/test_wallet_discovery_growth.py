"""Unit tests for discovery frontier growth and maintenance-oriented scanning."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.wallet_discovery import WalletDiscoveryEngine  # noqa: E402


def _mk_market(condition_id: str):
    return SimpleNamespace(condition_id=condition_id, id=condition_id, slug=condition_id)


class TestWalletDiscoveryGrowth:
    @pytest.mark.asyncio
    async def test_market_selection_wraps_and_advances_cursor(self):
        engine = WalletDiscoveryEngine()
        engine._market_scan_offset = 3

        async def fake_fetch_active_market_slice(*, offset: int, limit: int, delay_between_requests: float):
            if offset == 3:
                return [_mk_market("m3")]
            if offset == 0:
                return [_mk_market("m0"), _mk_market("m1"), _mk_market("m2"), _mk_market("m3")][:limit]
            return []

        async def fake_recent_markets(since_minutes: int = 180, active: bool = True):
            return [_mk_market("m_new"), _mk_market("m1")]

        engine._fetch_active_market_slice = fake_fetch_active_market_slice
        engine.client.get_recent_markets = fake_recent_markets

        markets, stats = await engine._select_markets_for_discovery(
            max_markets=5,
            delay_between_requests=0.0,
        )

        ids = [m.condition_id for m in markets]
        assert ids == ["m_new", "m1", "m3", "m0", "m2"]
        assert stats["market_scan_offset_start"] == 3
        assert stats["markets_primary"] == 1
        assert stats["markets_wrapped"] == 4
        assert stats["markets_recent"] == 2
        assert stats["markets_selected"] == 5
        assert engine._market_scan_offset == 4

    @pytest.mark.asyncio
    async def test_market_discovery_paginates_trade_pages(self):
        engine = WalletDiscoveryEngine()
        market = _mk_market("cond-1")
        calls: list[tuple[int, int]] = []

        async def fake_market_trades(condition_id: str, limit: int = 100, offset: int = 0):
            assert condition_id == "cond-1"
            calls.append((limit, offset))
            if offset == 0:
                # Two unique addresses + duplicates
                return [
                    {"user": "0xaaa0000000000000000000000000000000000001"},
                    {"taker": "0xbbb0000000000000000000000000000000000002"},
                    {"maker": "0xaaa0000000000000000000000000000000000001"},
                    {"user": "0xbbb0000000000000000000000000000000000002"},
                ]
            if offset == 4:
                return [
                    {"wallet": "0xccc0000000000000000000000000000000000003"},
                    {"wallet_address": "0xddd0000000000000000000000000000000000004"},
                ]
            return []

        engine.client.get_market_trades = fake_market_trades

        discovered = await engine._discover_wallets_from_market(market, max_wallets=4)
        assert len(discovered) == 4
        assert (50, 0) in calls  # max_wallets*2 page limit floors to a minimum of 50 per request
        assert any(offset == 4 for (_limit, offset) in calls)

    @pytest.mark.asyncio
    async def test_leaderboard_discovery_rotates_combos_and_extracts_multiple_fields(self):
        engine = WalletDiscoveryEngine()
        calls: list[tuple[str, str, str, int]] = []

        async def fake_get_leaderboard(
            limit: int = 50,
            time_period: str = "ALL",
            order_by: str = "PNL",
            category: str = "OVERALL",
            offset: int = 0,
        ):
            calls.append((order_by, time_period, category, offset))
            base = f"0x{order_by[:1]}{time_period[:1]}{category[:3]}{offset:04d}abcdef1234567890"
            idx = len(calls) % 3
            if idx == 0:
                return [{"proxyWallet": base}]
            if idx == 1:
                return [{"wallet": base}]
            return [{"address": base}]

        engine.client.get_leaderboard = fake_get_leaderboard

        first = await engine._discover_wallets_from_leaderboard(scan_count=20)
        second = await engine._discover_wallets_from_leaderboard(scan_count=20)

        # Minimum request budget per run is 6 leaderboard slices.
        assert len(first) == 6
        assert len(second) == 6
        assert len(calls) == 12
        assert len({(order_by, time_period, category) for order_by, time_period, category, _ in calls[:6]}) == 6
        assert engine._leaderboard_scan_cursor == 12 % len(engine._leaderboard_discovery_combinations())
