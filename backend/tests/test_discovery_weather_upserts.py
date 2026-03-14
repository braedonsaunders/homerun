import sys
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, DiscoveredWallet, WeatherTradeIntent  # noqa: E402
import services.smart_wallet_pool as smart_wallet_pool_module  # noqa: E402
import services.wallet_discovery as wallet_discovery_module  # noqa: E402
from services.weather import shared_state as weather_shared_state  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


async def _build_session_factory(name: str):
    return await build_postgres_session_factory(Base, name)


@pytest.mark.asyncio
async def test_wallet_discovery_placeholder_insert_is_idempotent(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("wallet_discovery_placeholders")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    address = "0xabc0000000000000000000000000000000000001"

    try:
        created_first = await service._upsert_discovered_placeholders({address}, discovery_source="scan")
        created_second = await service._upsert_discovered_placeholders({address}, discovery_source="scan")

        async with session_factory() as session:
            count = int((await session.execute(select(func.count(DiscoveredWallet.address)))).scalar() or 0)
            row = await session.get(DiscoveredWallet, address)

        assert created_first == 1
        assert created_second == 0
        assert count == 1
        assert row is not None
        assert row.discovery_source == "scan"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_wallet_discovery_placeholder_insert_supports_large_batch_without_bind_overflow(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("wallet_discovery_placeholders_large_batch")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    addresses = {f"0xabc{i:037x}" for i in range(900)}

    try:
        created = await service._upsert_discovered_placeholders(addresses, discovery_source="scan")

        async with session_factory() as session:
            count = int((await session.execute(select(func.count(DiscoveredWallet.address)))).scalar() or 0)

        assert created == len(addresses)
        assert count == len(addresses)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_wallet_discovery_upsert_wallet_updates_existing_row_without_duplicate(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("wallet_discovery_wallet_upsert")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    address = "0xabc0000000000000000000000000000000000002"

    try:
        async with session_factory() as session:
            session.add(
                DiscoveredWallet(
                    address=address,
                    discovered_at=utcnow(),
                    discovery_source="manual_pool",
                    username="seeded",
                    source_flags={"pool_manual_include": True},
                )
            )
            await session.commit()

        await service._upsert_wallet(
            {
                "address": address,
                "username": "alpha",
                "total_trades": 11,
                "win_rate": 0.57,
                "total_pnl": 125.0,
                "recommendation": "monitor",
            }
        )
        await service._upsert_wallet(
            {
                "address": address,
                "username": "beta",
                "total_trades": 12,
                "win_rate": 0.61,
                "total_pnl": 225.0,
                "recommendation": "copy_candidate",
            }
        )

        async with session_factory() as session:
            rows = (
                await session.execute(select(DiscoveredWallet).where(DiscoveredWallet.address == address))
            ).scalars().all()

        assert len(rows) == 1
        row = rows[0]
        assert row.username == "beta"
        assert row.total_trades == 12
        assert row.total_pnl == 225.0
        assert row.recommendation == "copy_candidate"
        assert row.source_flags == {"pool_manual_include": True}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_smart_wallet_pool_candidate_upsert_merges_flags_without_duplicate(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("smart_wallet_pool_candidates")
    monkeypatch.setattr(smart_wallet_pool_module, "AsyncSessionLocal", session_factory)
    service = smart_wallet_pool_module.SmartWalletPoolService()
    address = "0xabc0000000000000000000000000000000000003"

    try:
        await service._upsert_candidate_wallets(
            {address: {"leaderboard": True}},
            candidate_usernames={address: "alpha"},
        )
        await service._upsert_candidate_wallets(
            {address: {"wallet_trades": True}},
            candidate_usernames={address: "beta"},
        )

        async with session_factory() as session:
            rows = (
                await session.execute(select(DiscoveredWallet).where(DiscoveredWallet.address == address))
            ).scalars().all()

        assert len(rows) == 1
        row = rows[0]
        assert row.username == "beta"
        assert row.source_flags["leaderboard"] is True
        assert row.source_flags["wallet_trades"] is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_smart_wallet_pool_candidate_upsert_supports_large_batch_without_bind_overflow(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("smart_wallet_pool_candidates_large_batch")
    monkeypatch.setattr(smart_wallet_pool_module, "AsyncSessionLocal", session_factory)
    service = smart_wallet_pool_module.SmartWalletPoolService()
    addresses = [f"0xdef{i:037x}" for i in range(900)]

    try:
        await service._upsert_candidate_wallets(
            {address: {"leaderboard": True, "wallet_trades": True} for address in addresses},
            candidate_usernames={address: f"user_{idx}" for idx, address in enumerate(addresses)},
        )

        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(DiscoveredWallet).where(DiscoveredWallet.address.in_(addresses))
                )
            ).scalars().all()

        assert len(rows) == len(addresses)
        assert all(row.discovery_source == "smart_pool" for row in rows)
        assert all((row.source_flags or {}).get("leaderboard") is True for row in rows)
        assert all((row.source_flags or {}).get("wallet_trades") is True for row in rows)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_weather_intent_upsert_updates_pending_row_without_duplicate(tmp_path):
    engine, session_factory = await _build_session_factory("weather_intent_upsert_pending")
    intent_id = "weather-intent-pending"
    created_at = utcnow()

    try:
        async with session_factory() as session:
            await weather_shared_state.upsert_weather_intent(
                session,
                {
                    "id": intent_id,
                    "market_id": "market-1",
                    "market_question": "Will it rain?",
                    "direction": "buy_yes",
                    "entry_price": 0.31,
                    "take_profit_price": 0.8,
                    "stop_loss_pct": 25.0,
                    "model_probability": 0.74,
                    "edge_percent": 14.0,
                    "confidence": 0.72,
                    "model_agreement": 0.8,
                    "suggested_size_usd": 12.0,
                    "metadata_json": {"run": 1},
                    "status": "pending",
                    "created_at": created_at,
                },
            )
            await session.commit()

        async with session_factory() as session:
            await weather_shared_state.upsert_weather_intent(
                session,
                {
                    "id": intent_id,
                    "market_id": "market-1",
                    "market_question": "Will it rain tomorrow?",
                    "direction": "buy_no",
                    "entry_price": 0.41,
                    "take_profit_price": 0.85,
                    "stop_loss_pct": 20.0,
                    "model_probability": 0.66,
                    "edge_percent": 11.0,
                    "confidence": 0.77,
                    "model_agreement": 0.82,
                    "suggested_size_usd": 22.0,
                    "metadata_json": {"run": 2},
                    "status": "pending",
                    "created_at": utcnow(),
                },
            )
            rows = await weather_shared_state.list_weather_intents(session, status_filter="pending", limit=10)
            await session.commit()

        assert len(rows) == 1
        row = rows[0]
        assert row.id == intent_id
        assert row.direction == "buy_no"
        assert row.market_question == "Will it rain tomorrow?"
        assert row.suggested_size_usd == 22.0
        assert row.metadata_json == {"run": 2}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_weather_intent_upsert_does_not_overwrite_terminal_row(tmp_path):
    engine, session_factory = await _build_session_factory("weather_intent_upsert_terminal")
    intent_id = "weather-intent-terminal"
    created_at = utcnow()

    try:
        async with session_factory() as session:
            session.add(
                WeatherTradeIntent(
                    id=intent_id,
                    market_id="market-2",
                    market_question="Will it snow?",
                    direction="buy_yes",
                    entry_price=0.22,
                    take_profit_price=0.78,
                    stop_loss_pct=35.0,
                    model_probability=0.71,
                    edge_percent=9.0,
                    confidence=0.69,
                    model_agreement=0.74,
                    suggested_size_usd=18.0,
                    metadata_json={"seed": True},
                    status="executed",
                    created_at=created_at,
                )
            )
            await session.commit()

        async with session_factory() as session:
            await weather_shared_state.upsert_weather_intent(
                session,
                {
                    "id": intent_id,
                    "market_id": "market-2",
                    "market_question": "Will it snow hard?",
                    "direction": "buy_no",
                    "entry_price": 0.52,
                    "take_profit_price": 0.91,
                    "stop_loss_pct": 15.0,
                    "model_probability": 0.61,
                    "edge_percent": 13.0,
                    "confidence": 0.81,
                    "model_agreement": 0.85,
                    "suggested_size_usd": 28.0,
                    "metadata_json": {"seed": False},
                    "status": "pending",
                    "created_at": utcnow(),
                },
            )
            row = await session.get(WeatherTradeIntent, intent_id)

        assert row is not None
        assert row.status == "executed"
        assert row.direction == "buy_yes"
        assert row.market_question == "Will it snow?"
        assert row.metadata_json == {"seed": True}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_wallet_discovery_priority_queue_excludes_top_pool_without_large_not_in(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("wallet_discovery_priority_queue")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    service._runtime_discovery_settings = {
        "analysis_priority_batch_limit": 100000,
    }
    now = utcnow()

    try:
        async with session_factory() as session:
            session.add_all(
                [
                    DiscoveredWallet(
                        address="top_only",
                        discovered_at=now,
                        in_top_pool=True,
                        last_analyzed_at=None,
                        trades_24h=50,
                    ),
                    DiscoveredWallet(
                        address="top_and_smart",
                        discovered_at=now,
                        discovery_source="smart_pool",
                        in_top_pool=True,
                        last_analyzed_at=None,
                        trades_24h=40,
                    ),
                    DiscoveredWallet(
                        address="smart_only_false",
                        discovered_at=now,
                        discovery_source="smart_pool",
                        in_top_pool=False,
                        last_analyzed_at=None,
                        trades_24h=30,
                    ),
                    DiscoveredWallet(
                        address="smart_only_null",
                        discovered_at=now,
                        discovery_source="smart_pool",
                        in_top_pool=None,
                        last_analyzed_at=None,
                        trades_24h=20,
                    ),
                ]
            )
            await session.commit()

        ordered, counts = await service._priority_analysis_queue()

        assert counts["top_pool_unanalyzed"] == 2
        assert counts["smart_pool_unanalyzed"] == 2
        assert counts["priority_total"] == 4
        assert ordered == ["top_only", "top_and_smart", "smart_only_false", "smart_only_null"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_wallet_discovery_maintenance_refresh_prioritizes_active_stale_wallets(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("wallet_discovery_maintenance_refresh")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    now = utcnow()

    try:
        async with session_factory() as session:
            session.add_all(
                [
                    DiscoveredWallet(
                        address="active_stale",
                        discovered_at=now - timedelta(days=2),
                        last_analyzed_at=now - timedelta(days=2),
                        last_trade_at=now - timedelta(minutes=15),
                        trades_24h=45,
                        trades_1h=6,
                    ),
                    DiscoveredWallet(
                        address="inactive_older",
                        discovered_at=now - timedelta(days=10),
                        last_analyzed_at=now - timedelta(days=10),
                        last_trade_at=now - timedelta(days=4),
                        trades_24h=0,
                        trades_1h=0,
                    ),
                    DiscoveredWallet(
                        address="mid_stale",
                        discovered_at=now - timedelta(days=3),
                        last_analyzed_at=now - timedelta(days=3),
                        last_trade_at=now - timedelta(hours=3),
                        trades_24h=12,
                        trades_1h=1,
                    ),
                ]
            )
            await session.commit()

        ordered = await service._maintenance_refresh_queue(
            stale_cutoff=now - timedelta(hours=12),
            limit=3,
        )

        assert ordered == ["active_stale", "mid_stale", "inactive_older"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_wallet_discovery_priority_queue_reanalyzes_large_trade_gap_wallets(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory("wallet_discovery_priority_trade_gap")
    monkeypatch.setattr(wallet_discovery_module, "AsyncSessionLocal", session_factory)
    service = wallet_discovery_module.WalletDiscoveryEngine()
    now = utcnow()

    try:
        async with session_factory() as session:
            session.add_all(
                [
                    DiscoveredWallet(
                        address="gap_wallet",
                        discovered_at=now - timedelta(days=3),
                        last_analyzed_at=now - timedelta(hours=6),
                        last_trade_at=now - timedelta(minutes=20),
                        total_trades=194,
                        wins=1,
                        losses=0,
                        trades_24h=31,
                        trades_1h=5,
                        metrics_source_version=wallet_discovery_module.METRICS_SOURCE_VERSION,
                    ),
                    DiscoveredWallet(
                        address="fresh_gap_wallet",
                        discovered_at=now - timedelta(days=2),
                        last_analyzed_at=now - timedelta(hours=1),
                        last_trade_at=now - timedelta(minutes=10),
                        total_trades=220,
                        wins=1,
                        losses=0,
                        trades_24h=40,
                        trades_1h=7,
                        metrics_source_version=wallet_discovery_module.METRICS_SOURCE_VERSION,
                    ),
                    DiscoveredWallet(
                        address="normal_sample_wallet",
                        discovered_at=now - timedelta(days=4),
                        last_analyzed_at=now - timedelta(hours=8),
                        last_trade_at=now - timedelta(hours=2),
                        total_trades=362,
                        wins=102,
                        losses=21,
                        trades_24h=18,
                        trades_1h=2,
                        metrics_source_version=wallet_discovery_module.METRICS_SOURCE_VERSION,
                    ),
                    DiscoveredWallet(
                        address="backfill_wallet",
                        discovered_at=now - timedelta(days=5),
                        last_analyzed_at=now - timedelta(days=1),
                        total_trades=75,
                        wins=14,
                        losses=6,
                        metrics_source_version="legacy_v1",
                    ),
                ]
            )
            await session.commit()

        ordered, counts = await service._priority_analysis_queue()

        assert counts["trade_gap_reanalysis"] == 1
        assert ordered == ["gap_wallet", "backfill_wallet"]
    finally:
        await engine.dispose()
