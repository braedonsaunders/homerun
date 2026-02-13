import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.polymarket import PolymarketClient  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_extract_market_info_from_trades_prefers_matching_condition():
    requested = "0xabc"
    info = PolymarketClient._extract_market_info_from_trades(
        requested_condition_id=requested,
        trades=[
            {
                "conditionId": "0xdef",
                "title": "Wrong market",
                "slug": "wrong-market",
                "asset": "100",
            },
            {
                "conditionId": "0xAbC",
                "title": "Pacers vs. Nets",
                "slug": "nba-ind-bkn-2026-02-11",
                "eventSlug": "nba-ind-bkn-2026-02-11",
                "asset": "200",
                "asset_id": "201",
            },
        ],
    )

    assert info is not None
    assert info["condition_id"] == requested
    assert info["question"] == "Pacers vs. Nets"
    assert info["slug"] == "nba-ind-bkn-2026-02-11"
    assert info["event_slug"] == "nba-ind-bkn-2026-02-11"
    assert sorted(info["token_ids"]) == ["200", "201"]


def test_get_market_by_condition_id_falls_back_to_market_trades(monkeypatch):
    client = PolymarketClient()
    requested = "0x168b010a13936e827d9f1407afbfcfd915120f31246e95e9e20441e31011c3b0"

    async def _fake_rate_limited_get(url: str, **kwargs):
        # Simulate Gamma returning unrelated rows for condition filters.
        return _FakeResponse(
            [
                {
                    "id": "12",
                    "question": "Will Joe Biden get Coronavirus before the election?",
                    "conditionId": "0xe3b423dfad8c22ff75c9899c4e8176f628cf4ad4caa00481764d320e7415f7a9",
                    "slug": "will-joe-biden-get-coronavirus-before-the-election",
                }
            ]
        )

    async def _fake_get_market_trades(condition_id: str, limit: int = 100):
        assert condition_id == requested
        return [
            {
                "conditionId": requested,
                "title": "Pacers vs. Nets",
                "slug": "nba-ind-bkn-2026-02-11",
                "eventSlug": "nba-ind-bkn-2026-02-11",
                "asset": "2104009334376064720665425320836536669709149939945916240432665923815485331158",
            }
        ]

    async def _fake_cache():
        return None

    monkeypatch.setattr(client, "_rate_limited_get", _fake_rate_limited_get)
    monkeypatch.setattr(client, "get_market_trades", _fake_get_market_trades)
    monkeypatch.setattr(client, "_get_persistent_cache", _fake_cache)

    info = asyncio.run(client.get_market_by_condition_id(requested))

    assert info is not None
    assert info["question"] == "Pacers vs. Nets"
    assert info["slug"] == "nba-ind-bkn-2026-02-11"
    assert info["condition_id"] == requested
    assert client._market_cache[requested]["question"] == "Pacers vs. Nets"
    token_key = "token:2104009334376064720665425320836536669709149939945916240432665923815485331158"
    assert client._market_cache[token_key]["question"] == "Pacers vs. Nets"


def test_get_market_by_condition_id_uses_condition_ids_query_param(monkeypatch):
    client = PolymarketClient()
    requested = "0xe39e84e10a538e4dea9f999409f9b54fff4037b2125d3b7efc44d3eeb9f1ea39"
    seen_params = []

    async def _fake_rate_limited_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        seen_params.append(params)
        if params.get("condition_ids") == requested:
            return _FakeResponse(
                [
                    {
                        "id": "1339834",
                        "question": "Grizzlies vs. Nuggets",
                        "conditionId": requested,
                        "slug": "nba-mem-den-2026-02-11",
                        "endDate": "2026-02-12T02:00:00Z",
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    }
                ]
            )
        return _FakeResponse([])

    async def _fake_get_market_trades(*args, **kwargs):
        raise AssertionError("should not fall back to trade lookup when Gamma match exists")

    async def _fake_cache():
        return None

    monkeypatch.setattr(client, "_rate_limited_get", _fake_rate_limited_get)
    monkeypatch.setattr(client, "get_market_trades", _fake_get_market_trades)
    monkeypatch.setattr(client, "_get_persistent_cache", _fake_cache)

    info = asyncio.run(client.get_market_by_condition_id(requested))

    assert info is not None
    assert info["condition_id"] == requested
    assert info["question"] == "Grizzlies vs. Nuggets"
    assert any(params.get("condition_ids") == requested for params in seen_params)


def test_get_market_by_condition_id_rejects_cache_without_tradability(monkeypatch):
    client = PolymarketClient()
    requested = "0xf23a18c26127d9b153ef1ca40ec94f7603b18170c474d0415cdca71ac52dbebd"
    seen_params = []

    # Simulate a stale fallback cache row (question/slug only, no tradability metadata).
    client._market_cache[requested] = {
        "condition_id": requested,
        "question": "Spurs vs. Warriors",
        "slug": "nba-sas-gsw-2026-02-11",
    }

    async def _fake_rate_limited_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        seen_params.append(params)
        return _FakeResponse(
            [
                {
                    "id": "1339837",
                    "question": "Spurs vs. Warriors",
                    "conditionId": requested,
                    "slug": "nba-sas-gsw-2026-02-11",
                    "endDate": "2026-02-12T03:00:00Z",
                    "active": True,
                    "closed": False,
                    "acceptingOrders": True,
                }
            ]
        )

    async def _fake_get_market_trades(*args, **kwargs):
        raise AssertionError("should not hit fallback trade lookup")

    async def _fake_cache():
        return None

    monkeypatch.setattr(client, "_rate_limited_get", _fake_rate_limited_get)
    monkeypatch.setattr(client, "get_market_trades", _fake_get_market_trades)
    monkeypatch.setattr(client, "_get_persistent_cache", _fake_cache)

    info = asyncio.run(client.get_market_by_condition_id(requested))

    assert info is not None
    assert info["condition_id"] == requested
    assert info["end_date"] == "2026-02-12T03:00:00Z"
    assert info["active"] is True
    assert any(params.get("condition_ids") == requested for params in seen_params)


def test_get_market_by_condition_id_force_refresh_bypasses_cache(monkeypatch):
    client = PolymarketClient()
    requested = "0x4f7ca19f1c0f2dc7d7b8b2a52981a2eea9b0f56ef31a31f9c9e64eeb424f06f0"
    seen_params = []

    client._market_cache[requested] = {
        "condition_id": requested,
        "question": "Cached row",
        "slug": "cached-row",
        "outcomes": ["Yes", "No"],
        "outcome_prices": [0.5, 0.5],
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "end_date": "2026-12-31T00:00:00Z",
    }

    async def _fake_rate_limited_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        seen_params.append(params)
        return _FakeResponse(
            [
                {
                    "id": "2000001",
                    "question": "Fresh row",
                    "conditionId": requested,
                    "slug": "fresh-row",
                    "endDate": "2026-12-31T00:00:00Z",
                    "active": True,
                    "closed": True,
                    "acceptingOrders": False,
                    "outcomes": ["Yes", "No"],
                    "outcomePrices": ["0.99", "0.01"],
                }
            ]
        )

    async def _fake_get_market_trades(*args, **kwargs):
        raise AssertionError("should not hit fallback trade lookup")

    async def _fake_cache():
        return None

    monkeypatch.setattr(client, "_rate_limited_get", _fake_rate_limited_get)
    monkeypatch.setattr(client, "get_market_trades", _fake_get_market_trades)
    monkeypatch.setattr(client, "_get_persistent_cache", _fake_cache)

    info = asyncio.run(client.get_market_by_condition_id(requested, force_refresh=True))

    assert info is not None
    assert info["condition_id"] == requested
    assert info["question"] == "Fresh row"
    assert info["closed"] is True
    assert info["accepting_orders"] is False
    assert any(params.get("condition_ids") == requested for params in seen_params)


def test_get_market_by_token_id_skips_invalid_ids(monkeypatch):
    client = PolymarketClient()

    async def _unexpected_rate_call(*_args, **_kwargs):
        raise AssertionError("should not call Gamma for non-token IDs")

    monkeypatch.setattr(client, "_rate_limited_get", _unexpected_rate_call)

    result = asyncio.run(client.get_market_by_token_id("kasimpasa vs karagumruk winner?"))
    assert result is None


def test_token_id_shape_classifier():
    assert (
        PolymarketClient._looks_like_condition_id("0x168b010a13936e827d9f1407afbfcfd915120f31246e95e9e20441e31011c3b0")
        is True
    )
    assert (
        PolymarketClient._looks_like_token_id(
            "2104009334376064720665425320836536669709149939945916240432665923815485331158"
        )
        is True
    )
    assert PolymarketClient._looks_like_token_id("kxmvesportsmultigameextended") is False
    assert PolymarketClient._looks_like_token_id("KXINXSPXW-26FEB11-B5910_yes") is False


def test_is_market_tradable_false_for_closed_or_resolved():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": True,
                "active": True,
                "resolved": False,
                "end_date": (now + timedelta(hours=1)).isoformat(),
            },
            now=now,
        )
        is False
    )
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": True,
                "end_date": (now + timedelta(hours=1)).isoformat(),
            },
            now=now,
        )
        is False
    )


def test_is_market_tradable_false_for_past_end_date():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "end_date": (now - timedelta(minutes=1)).isoformat(),
            },
            now=now,
        )
        is False
    )


def test_is_market_tradable_true_for_active_future_market():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "end_date": (now + timedelta(hours=2)).isoformat(),
            },
            now=now,
        )
        is True
    )


def test_is_market_tradable_false_when_order_book_disabled():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "enable_order_book": False,
                "end_date": (now + timedelta(hours=2)).isoformat(),
            },
            now=now,
        )
        is False
    )


def test_is_market_tradable_false_for_review_or_dispute_status():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "status": "in review",
                "end_date": (now + timedelta(hours=2)).isoformat(),
            },
            now=now,
        )
        is False
    )


def test_is_market_tradable_false_for_uma_resolution_proposed():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "uma_resolution_status": "proposed",
                "end_date": (now + timedelta(hours=2)).isoformat(),
            },
            now=now,
        )
        is False
    )
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "uma_resolution_statuses": ["proposed"],
                "end_date": (now + timedelta(hours=2)).isoformat(),
            },
            now=now,
        )
        is False
    )
    assert (
        PolymarketClient.is_market_tradable(
            {
                "closed": False,
                "active": True,
                "resolved": False,
                "accepting_orders": True,
                "status": "in dispute",
                "end_date": (now + timedelta(hours=2)).isoformat(),
            },
            now=now,
        )
        is False
    )
