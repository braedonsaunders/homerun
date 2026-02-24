import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_discovery


def _base_graph_payload() -> dict:
    return {
        "generated_at": "2026-02-23T00:00:00+00:00",
        "lookback_days": 30,
        "nodes": [
            {
                "id": "wallet:0xaaa",
                "kind": "wallet",
                "address": "0xaaa",
                "label": "Alpha",
                "degree": 3,
                "composite_score": 0.91,
                "rank_score": 0.88,
            },
            {
                "id": "wallet:0xbbb",
                "kind": "wallet",
                "address": "0xbbb",
                "label": "Bravo",
                "degree": 2,
                "composite_score": 0.79,
                "rank_score": 0.66,
            },
            {
                "id": "wallet:0xccc",
                "kind": "wallet",
                "address": "0xccc",
                "label": "Charlie",
                "degree": 1,
                "composite_score": 0.55,
                "rank_score": 0.44,
            },
            {
                "id": "cluster:cluster-1",
                "kind": "cluster",
                "cluster_id": "cluster-1",
                "label": "Cluster One",
                "member_count": 2,
            },
        ],
        "edges": [
            {
                "id": "edge:co_trade:ab",
                "kind": "co_trade",
                "source": "wallet:0xaaa",
                "target": "wallet:0xbbb",
                "combined_score": 0.9,
                "weight": 0.9,
            },
            {
                "id": "edge:co_trade:bc",
                "kind": "co_trade",
                "source": "wallet:0xbbb",
                "target": "wallet:0xccc",
                "combined_score": 0.61,
                "weight": 0.61,
            },
            {
                "id": "edge:cluster:aaa",
                "kind": "cluster_membership",
                "source": "cluster:cluster-1",
                "target": "wallet:0xaaa",
                "weight": 1.0,
            },
            {
                "id": "edge:cluster:bbb",
                "kind": "cluster_membership",
                "source": "cluster:cluster-1",
                "target": "wallet:0xbbb",
                "weight": 1.0,
            },
        ],
        "cohorts": [
            {
                "id": "cohort_1",
                "wallet_addresses": ["0xaaa", "0xbbb", "0xddd"],
                "avg_combined_score": 0.83,
                "shared_market_count": 7,
                "direction_agreement": 0.81,
            }
        ],
    }


@pytest.mark.asyncio
async def test_get_traders_network_graph_includes_group_nodes(monkeypatch):
    base_graph = _base_graph_payload()
    get_network_graph = AsyncMock(return_value=base_graph)
    analyze_cohorts = AsyncMock(return_value=[])
    monkeypatch.setattr(
        routes_discovery.wallet_intelligence.cohort_analyzer,
        "get_network_graph",
        get_network_graph,
    )
    monkeypatch.setattr(
        routes_discovery.wallet_intelligence.cohort_analyzer,
        "analyze_cohorts",
        analyze_cohorts,
    )
    monkeypatch.setattr(
        routes_discovery,
        "_fetch_group_payload",
        AsyncMock(
            return_value=[
                {
                    "id": "group-1",
                    "name": "Core Traders",
                    "description": "Graph test group",
                    "source_type": "manual",
                    "member_count": 3,
                    "auto_track_members": True,
                    "members": [
                        {"wallet_address": "0xaaa"},
                        {"wallet_address": "0xbbb"},
                        {"wallet_address": "0x999"},
                    ],
                }
            ]
        ),
    )

    payload = await routes_discovery.get_traders_network_graph(
        min_pair_score=0.55,
        limit_wallets=20,
        include_groups=True,
        include_clusters=True,
        session=SimpleNamespace(),
    )

    assert payload["summary"]["wallet_nodes"] == 3
    assert payload["summary"]["cluster_nodes"] == 1
    assert payload["summary"]["group_nodes"] == 1
    assert payload["summary"]["group_membership_edges"] == 2
    assert any(node.get("kind") == "group" and node.get("group_id") == "group-1" for node in payload["nodes"])
    assert any(edge.get("kind") == "group_membership" for edge in payload["edges"])
    assert payload["cohorts"][0]["visible_wallet_count"] == 2
    analyze_cohorts.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_traders_network_graph_applies_score_and_cluster_filters(monkeypatch):
    base_graph = _base_graph_payload()
    monkeypatch.setattr(
        routes_discovery.wallet_intelligence.cohort_analyzer,
        "get_network_graph",
        AsyncMock(return_value=base_graph),
    )
    monkeypatch.setattr(
        routes_discovery.wallet_intelligence.cohort_analyzer,
        "analyze_cohorts",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_fetch_group_payload",
        AsyncMock(return_value=[]),
    )

    payload = await routes_discovery.get_traders_network_graph(
        min_pair_score=0.8,
        limit_wallets=20,
        include_groups=False,
        include_clusters=False,
        session=SimpleNamespace(),
    )

    assert payload["summary"]["co_trade_edges"] == 1
    assert payload["summary"]["cluster_nodes"] == 0
    assert payload["summary"]["group_nodes"] == 0
    assert all(node.get("kind") != "cluster" for node in payload["nodes"])
    assert all(edge.get("kind") == "co_trade" for edge in payload["edges"])


@pytest.mark.asyncio
async def test_get_traders_network_graph_refreshes_when_cache_empty(monkeypatch):
    base_graph = _base_graph_payload()
    get_network_graph = AsyncMock(
        side_effect=[
            {"generated_at": None, "lookback_days": 30, "nodes": [], "edges": [], "cohorts": []},
            base_graph,
        ]
    )
    analyze_cohorts = AsyncMock(return_value=[{"id": "cohort_1"}])
    monkeypatch.setattr(
        routes_discovery.wallet_intelligence.cohort_analyzer,
        "get_network_graph",
        get_network_graph,
    )
    monkeypatch.setattr(
        routes_discovery.wallet_intelligence.cohort_analyzer,
        "analyze_cohorts",
        analyze_cohorts,
    )
    monkeypatch.setattr(
        routes_discovery,
        "_fetch_group_payload",
        AsyncMock(return_value=[]),
    )

    payload = await routes_discovery.get_traders_network_graph(
        min_pair_score=0.55,
        limit_wallets=20,
        include_groups=False,
        include_clusters=True,
        session=SimpleNamespace(),
    )

    assert payload["summary"]["wallet_nodes"] == 3
    assert get_network_graph.await_count == 2
    analyze_cohorts.assert_awaited_once()
