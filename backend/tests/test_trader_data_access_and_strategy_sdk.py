import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import services.trader_data_access as trader_data_access
import services.traders_firehose_pipeline as traders_firehose_pipeline
from services.strategy_sdk import StrategySDK
from services.traders_sdk import TradersSDK


@pytest.mark.asyncio
async def test_trader_data_access_strategy_filtered_signals_delegates_pipeline(monkeypatch):
    rows = [{"id": "sig-1"}]
    loader = AsyncMock(return_value=rows)
    monkeypatch.setattr(
        traders_firehose_pipeline,
        "get_strategy_filtered_trader_opportunities",
        loader,
    )

    result = await trader_data_access.get_strategy_filtered_trader_signals(
        limit=23,
        include_filtered=True,
    )

    loader.assert_awaited_once_with(limit=23, include_filtered=True)
    assert result == rows


@pytest.mark.asyncio
async def test_strategy_sdk_trader_signal_wrappers_passthrough(monkeypatch):
    firehose_rows = [{"id": "firehose-row"}]
    strategy_rows = [{"id": "strategy-row"}]
    confluence_rows = [{"id": "confluence-row"}]

    firehose_loader = AsyncMock(return_value=firehose_rows)
    strategy_loader = AsyncMock(return_value=strategy_rows)
    confluence_loader = AsyncMock(return_value=confluence_rows)

    monkeypatch.setattr(trader_data_access, "get_trader_firehose_signals", firehose_loader)
    monkeypatch.setattr(trader_data_access, "get_strategy_filtered_trader_signals", strategy_loader)
    monkeypatch.setattr(trader_data_access, "get_trader_confluence_signals", confluence_loader)

    firehose = await StrategySDK.get_trader_firehose_signals(
        limit=111,
        include_filtered=True,
        include_source_context=False,
    )
    strategy = await StrategySDK.get_trader_strategy_signals(
        limit=77,
        include_filtered=False,
    )
    confluence = await StrategySDK.get_trader_confluence_signals(
        min_strength=0.42,
        min_tier="HIGH",
        limit=12,
    )

    firehose_loader.assert_awaited_once_with(
        limit=111,
        include_filtered=True,
        include_source_context=False,
    )
    strategy_loader.assert_awaited_once_with(limit=77, include_filtered=False)
    confluence_loader.assert_awaited_once_with(min_strength=0.42, min_tier="HIGH", limit=12)
    assert firehose == firehose_rows
    assert strategy == strategy_rows
    assert confluence == confluence_rows


@pytest.mark.asyncio
async def test_traders_sdk_signal_wrappers_passthrough(monkeypatch):
    firehose_rows = [{"id": "firehose-row"}]
    strategy_rows = [{"id": "strategy-row"}]
    confluence_rows = [{"id": "confluence-row"}]

    firehose_loader = AsyncMock(return_value=firehose_rows)
    strategy_loader = AsyncMock(return_value=strategy_rows)
    confluence_loader = AsyncMock(return_value=confluence_rows)

    monkeypatch.setattr(trader_data_access, "get_trader_firehose_signals", firehose_loader)
    monkeypatch.setattr(trader_data_access, "get_strategy_filtered_trader_signals", strategy_loader)
    monkeypatch.setattr(trader_data_access, "get_trader_confluence_signals", confluence_loader)

    firehose = await TradersSDK.get_firehose_signals(
        limit=111,
        include_filtered=True,
        include_source_context=False,
    )
    strategy = await TradersSDK.get_strategy_filtered_signals(
        limit=77,
        include_filtered=False,
    )
    confluence = await TradersSDK.get_confluence_signals(
        min_strength=0.42,
        min_tier="HIGH",
        limit=12,
    )

    firehose_loader.assert_awaited_once_with(
        limit=111,
        include_filtered=True,
        include_source_context=False,
    )
    strategy_loader.assert_awaited_once_with(limit=77, include_filtered=False)
    confluence_loader.assert_awaited_once_with(min_strength=0.42, min_tier="HIGH", limit=12)
    assert firehose == firehose_rows
    assert strategy == strategy_rows
    assert confluence == confluence_rows


@pytest.mark.asyncio
async def test_strategy_sdk_trader_directory_wrappers_passthrough(monkeypatch):
    pooled_rows = [{"address": "0xpool"}]
    tracked_rows = [{"address": "0xtracked"}]
    group_rows = [{"id": "grp-1"}]
    tag_rows = [{"name": "whale", "wallet_count": 2}]
    wallets_by_tag = [{"address": "0xtag"}]

    pooled_loader = AsyncMock(return_value=pooled_rows)
    tracked_loader = AsyncMock(return_value=tracked_rows)
    groups_loader = AsyncMock(return_value=group_rows)
    tags_loader = AsyncMock(return_value=tag_rows)
    by_tag_loader = AsyncMock(return_value=wallets_by_tag)

    monkeypatch.setattr(trader_data_access, "get_pooled_traders", pooled_loader)
    monkeypatch.setattr(trader_data_access, "get_tracked_traders", tracked_loader)
    monkeypatch.setattr(trader_data_access, "get_trader_groups", groups_loader)
    monkeypatch.setattr(trader_data_access, "get_trader_tags", tags_loader)
    monkeypatch.setattr(trader_data_access, "get_traders_by_tag", by_tag_loader)

    pooled = await StrategySDK.get_pooled_traders(
        limit=19,
        tier="high",
        include_blacklisted=False,
        tracked_only=True,
    )
    tracked = await StrategySDK.get_tracked_traders(
        limit=29,
        include_recent_activity=True,
        activity_hours=12,
    )
    groups = await StrategySDK.get_trader_groups(
        include_members=True,
        member_limit=7,
    )
    tags = await StrategySDK.get_trader_tags()
    by_tag = await StrategySDK.get_traders_by_tag("whale", limit=33)

    pooled_loader.assert_awaited_once_with(
        limit=19,
        tier="high",
        include_blacklisted=False,
        tracked_only=True,
    )
    tracked_loader.assert_awaited_once_with(
        limit=29,
        include_recent_activity=True,
        activity_hours=12,
    )
    groups_loader.assert_awaited_once_with(include_members=True, member_limit=7)
    tags_loader.assert_awaited_once_with()
    by_tag_loader.assert_awaited_once_with(tag_name="whale", limit=33)

    assert pooled == pooled_rows
    assert tracked == tracked_rows
    assert groups == group_rows
    assert tags == tag_rows
    assert by_tag == wallets_by_tag


@pytest.mark.asyncio
async def test_traders_sdk_directory_wrappers_passthrough(monkeypatch):
    pooled_rows = [{"address": "0xpool"}]
    tracked_rows = [{"address": "0xtracked"}]
    group_rows = [{"id": "grp-1"}]
    tag_rows = [{"name": "whale", "wallet_count": 2}]
    wallets_by_tag = [{"address": "0xtag"}]

    pooled_loader = AsyncMock(return_value=pooled_rows)
    tracked_loader = AsyncMock(return_value=tracked_rows)
    groups_loader = AsyncMock(return_value=group_rows)
    tags_loader = AsyncMock(return_value=tag_rows)
    by_tag_loader = AsyncMock(return_value=wallets_by_tag)

    monkeypatch.setattr(trader_data_access, "get_pooled_traders", pooled_loader)
    monkeypatch.setattr(trader_data_access, "get_tracked_traders", tracked_loader)
    monkeypatch.setattr(trader_data_access, "get_trader_groups", groups_loader)
    monkeypatch.setattr(trader_data_access, "get_trader_tags", tags_loader)
    monkeypatch.setattr(trader_data_access, "get_traders_by_tag", by_tag_loader)

    pooled = await TradersSDK.get_pooled_traders(
        limit=19,
        tier="high",
        include_blacklisted=False,
        tracked_only=True,
    )
    tracked = await TradersSDK.get_tracked_traders(
        limit=29,
        include_recent_activity=True,
        activity_hours=12,
    )
    groups = await TradersSDK.get_groups(
        include_members=True,
        member_limit=7,
    )
    tags = await TradersSDK.get_tags()
    by_tag = await TradersSDK.get_traders_by_tag("whale", limit=33)

    pooled_loader.assert_awaited_once_with(
        limit=19,
        tier="high",
        include_blacklisted=False,
        tracked_only=True,
    )
    tracked_loader.assert_awaited_once_with(
        limit=29,
        include_recent_activity=True,
        activity_hours=12,
    )
    groups_loader.assert_awaited_once_with(include_members=True, member_limit=7)
    tags_loader.assert_awaited_once_with()
    by_tag_loader.assert_awaited_once_with(tag_name="whale", limit=33)

    assert pooled == pooled_rows
    assert tracked == tracked_rows
    assert groups == group_rows
    assert tags == tag_rows
    assert by_tag == wallets_by_tag


def test_strategy_sdk_news_filter_validation_and_defaults():
    defaults = StrategySDK.news_filter_defaults()
    assert defaults["min_edge_percent"] == 5.0
    assert defaults["min_confidence"] == 0.45
    assert defaults["orchestrator_min_edge"] == 10.0
    assert defaults["require_verifier"] is True
    assert defaults["require_second_source"] is False
    assert defaults["min_supporting_articles"] == 2
    assert defaults["min_supporting_sources"] == 2

    validated = StrategySDK.validate_news_filter_config(
        {
            "min_edge_percent": "12.5",
            "min_confidence": "0.62",
            "orchestrator_min_edge": "9",
            "require_verifier": "false",
            "require_second_source": "1",
            "min_supporting_articles": "3",
            "min_supporting_sources": "4",
        }
    )
    assert validated["min_edge_percent"] == 12.5
    assert validated["min_confidence"] == 0.62
    assert validated["orchestrator_min_edge"] == 9.0
    assert validated["require_verifier"] is False
    assert validated["require_second_source"] is True
    assert validated["min_supporting_articles"] == 3
    assert validated["min_supporting_sources"] == 4


def test_strategy_sdk_extract_trader_signal_wallets_reads_nested_context():
    payload = {
        "wallets": ["0x1111111111111111111111111111111111111111"],
        "strategy_context": {
            "wallet_addresses": ["0x2222222222222222222222222222222222222222"],
            "firehose": {
                "top_wallets": [
                    {"address": "0x3333333333333333333333333333333333333333"},
                ]
            },
        },
    }

    wallets = StrategySDK.extract_trader_signal_wallets(payload)
    assert wallets == {
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333",
    }


def test_strategy_sdk_match_trader_signal_scope_uses_runtime_context():
    signal = {
        "strategy_context": {
            "firehose": {
                "wallets": [
                    "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ]
            }
        }
    }
    scope_context = StrategySDK.build_trader_scope_runtime_context(
        {"modes": ["pool"], "individual_wallets": [], "group_ids": []},
        pool_wallets={"0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    )

    matched, payload = StrategySDK.match_trader_signal_scope(signal, scope_context)
    assert matched is True
    assert payload["matched_modes"] == ["pool"]
