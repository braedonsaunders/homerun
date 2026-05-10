import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

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


class _FakeStreamingResponse:
    def __init__(self, read_error=None):
        self.status_code = 200
        self.headers = {}
        self._read_error = read_error
        self.read_calls = 0

    async def aread(self):
        self.read_calls += 1
        if self._read_error is not None:
            raise self._read_error
        return b"{}"


class _FakeStreamingClient:
    def __init__(self, failures_before_success):
        self.failures_before_success = failures_before_success
        self.calls = 0

    async def get(self, _url, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            return _FakeStreamingResponse(
                read_error=httpx.RemoteProtocolError("peer closed connection without sending complete message body")
            )
        return _FakeStreamingResponse()


class _ClosedThenHealthyClient:
    def __init__(self, fail_once: bool = False):
        self._fail_once = fail_once
        self.calls = 0
        self.is_closed = False

    async def get(self, _url, **_kwargs):
        self.calls += 1
        if self._fail_once and self.calls == 1:
            self.is_closed = True
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return _FakeStreamingResponse()

    async def aclose(self):
        self.is_closed = True


def test_rate_limited_get_retries_when_body_read_fails(monkeypatch):
    client = PolymarketClient()

    async def _fake_acquire(_endpoint):
        return None

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr("services.polymarket.rate_limiter.acquire", _fake_acquire)
    monkeypatch.setattr("services.polymarket.asyncio.sleep", _fake_sleep)

    fake_client = _FakeStreamingClient(failures_before_success=2)
    response = asyncio.run(client._rate_limited_get("https://example.com/test", client=fake_client))

    assert response.status_code == 200
    assert fake_client.calls == 3


def test_rate_limited_get_raises_after_body_read_failures(monkeypatch):
    client = PolymarketClient()

    async def _fake_acquire(_endpoint):
        return None

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr("services.polymarket.rate_limiter.acquire", _fake_acquire)
    monkeypatch.setattr("services.polymarket.asyncio.sleep", _fake_sleep)

    fake_client = _FakeStreamingClient(failures_before_success=10)
    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(client._rate_limited_get("https://example.com/test", client=fake_client))

    assert fake_client.calls == 4


def test_rate_limited_get_recovers_when_client_closed_runtime_error(monkeypatch):
    client = PolymarketClient()

    async def _fake_acquire(_endpoint):
        return None

    async def _fake_sleep(_seconds):
        return None

    first_client = _ClosedThenHealthyClient(fail_once=True)
    second_client = _ClosedThenHealthyClient(fail_once=False)
    client._client = first_client

    async def _fake_get_client():
        client._client = second_client
        return second_client

    monkeypatch.setattr("services.polymarket.rate_limiter.acquire", _fake_acquire)
    monkeypatch.setattr("services.polymarket.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(client, "_get_client", _fake_get_client)

    response = asyncio.run(client._rate_limited_get("https://example.com/test", client=first_client))

    assert response.status_code == 200
    assert first_client.calls == 1
    assert second_client.calls == 1
    assert first_client.is_closed is True


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


def test_get_market_by_condition_id_falls_back_to_closed_when_first_probe_empty(monkeypatch):
    client = PolymarketClient()
    requested = "0xc2cdb3b65e84e1d2fb2ee43e09f7d2cf66e36c9b0d76d6adafefd6cb16d10aa1"
    seen_params: list[dict] = []

    async def _fake_rate_limited_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        seen_params.append(dict(params))
        # First probe (active markets) is empty: gamma defaults to closed=false.
        if "closed" not in params:
            return _FakeResponse([])
        # Second probe (closed=true) returns a resolved market row.
        return _FakeResponse(
            [
                {
                    "id": "1488421",
                    "question": "UFC 328: Du Plessis vs. Chimaev — Du Plessis to win",
                    "conditionId": requested,
                    "slug": "ufc-328-du-plessis-vs-chimaev-du-plessis",
                    "endDate": "2026-05-09T05:00:00Z",
                    "active": False,
                    "closed": True,
                    "archived": False,
                    "acceptingOrders": False,
                    "resolved": True,
                    "winningOutcome": "No",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0", "1"]',
                    "umaResolutionStatus": "resolved",
                }
            ]
        )

    async def _fake_get_market_trades(*args, **kwargs):
        raise AssertionError("should not fall back to trade lookup when closed probe matches")

    async def _fake_cache():
        return None

    monkeypatch.setattr(client, "_rate_limited_get", _fake_rate_limited_get)
    monkeypatch.setattr(client, "get_market_trades", _fake_get_market_trades)
    monkeypatch.setattr(client, "_get_persistent_cache", _fake_cache)

    info = asyncio.run(client.get_market_by_condition_id(requested))

    assert info is not None
    assert info["condition_id"] == requested
    assert info["closed"] is True
    assert info["resolved"] is True
    assert info["winning_outcome"] == "No"
    assert len(seen_params) == 2
    assert seen_params[0].get("condition_ids") == requested
    assert seen_params[0].get("closed") is None
    assert seen_params[1].get("condition_ids") == requested
    assert seen_params[1].get("closed") == "true"
    assert client._market_cache[requested]["resolved"] is True


def test_get_market_by_condition_id_does_not_probe_closed_when_active_match(monkeypatch):
    client = PolymarketClient()
    requested = "0x3a1b8e7d5b2f4c6a9e8b1d2c4a6b8e0f1a2d4c5b6e7f8a9b0c1d2e3f4a5b6c7d"
    call_count = 0

    async def _fake_rate_limited_get(url: str, **kwargs):
        nonlocal call_count
        call_count += 1
        params = kwargs.get("params") or {}
        if call_count == 1:
            return _FakeResponse(
                [
                    {
                        "id": "9000123",
                        "question": "Active market lookup happy path",
                        "conditionId": requested,
                        "slug": "active-market-happy-path",
                        "endDate": "2026-12-31T23:59:00Z",
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                    }
                ]
            )
        raise AssertionError(f"closed-probe should not run when active match exists; got params={params}")

    async def _fake_get_market_trades(*args, **kwargs):
        raise AssertionError("should not hit trade fallback")

    async def _fake_cache():
        return None

    monkeypatch.setattr(client, "_rate_limited_get", _fake_rate_limited_get)
    monkeypatch.setattr(client, "get_market_trades", _fake_get_market_trades)
    monkeypatch.setattr(client, "_get_persistent_cache", _fake_cache)

    info = asyncio.run(client.get_market_by_condition_id(requested))

    assert info is not None
    assert info["condition_id"] == requested
    assert call_count == 1


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


def test_filter_by_time_period_handles_mixed_timezone_timestamps():
    client = PolymarketClient()
    now = datetime.now(timezone.utc)
    trades = [
        {"timestamp": (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"), "id": "recent"},
        {"timestamp": (now - timedelta(days=2)).replace(tzinfo=None).isoformat(), "id": "stale"},
    ]

    filtered = client._filter_by_time_period(trades, "DAY")

    assert [trade["id"] for trade in filtered] == ["recent"]


def test_is_market_tradable_false_for_dispute_status():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
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


# ---------------------------------------------------------------------------
# 2026-05-05: per-endpoint 429 cooldown (sustained-throttle safety net)
# ---------------------------------------------------------------------------


class _Fake429Client:
    """Returns a 429 response; aread() yields an empty body."""

    def __init__(self):
        self.calls = 0

    async def get(self, _url, **_kwargs):
        self.calls += 1
        return _Fake429Response(retry_after_seconds=None)


class _Fake429Response:
    def __init__(self, *, retry_after_seconds):
        self.status_code = 429
        self.headers = (
            {"Retry-After": str(int(retry_after_seconds))}
            if retry_after_seconds is not None
            else {}
        )

    def raise_for_status(self):
        # The wrapper checks status_code directly for 429; raise_for_status
        # is only consulted on transport-level errors that escape into
        # the except branch — so we no-op here.
        return None

    async def aread(self):
        return b""


def test_endpoint_cooldown_remaining_returns_zero_when_no_cooldown_set():
    client = PolymarketClient()
    assert client._endpoint_cooldown_remaining("data_positions") == 0.0


def test_stamp_endpoint_cooldown_marks_endpoint_cooled():
    client = PolymarketClient()
    cooldown = client._stamp_endpoint_cooldown("data_positions")
    assert cooldown >= 30.0
    assert client._endpoint_cooldown_remaining("data_positions") > 0.0


def test_stamp_endpoint_cooldown_backs_off_exponentially():
    client = PolymarketClient()
    first = client._stamp_endpoint_cooldown("data_positions")
    second = client._stamp_endpoint_cooldown("data_positions")
    third = client._stamp_endpoint_cooldown("data_positions")
    assert second > first
    assert third > second


def test_stamp_endpoint_cooldown_honors_retry_after_floor():
    client = PolymarketClient()
    response = httpx.Response(429, headers={"Retry-After": "120"})
    cooldown = client._stamp_endpoint_cooldown("data_positions", response=response)
    assert cooldown >= 120.0


def test_stamp_endpoint_cooldown_caps_at_max():
    client = PolymarketClient()
    # Sufficient consecutive trips to blow past the cap.
    for _ in range(20):
        cooldown = client._stamp_endpoint_cooldown("data_positions")
    assert cooldown <= 300.0


def test_clear_endpoint_cooldown_resets_attempts_and_window():
    client = PolymarketClient()
    client._stamp_endpoint_cooldown("data_positions")
    client._stamp_endpoint_cooldown("data_positions")
    client._clear_endpoint_cooldown("data_positions")
    assert client._endpoint_cooldown_remaining("data_positions") == 0.0
    # Next 429 starts from the base window again.
    next_cooldown = client._stamp_endpoint_cooldown("data_positions")
    assert next_cooldown >= 30.0
    assert next_cooldown < 90.0


def test_rate_limited_get_short_circuits_when_endpoint_cooled(monkeypatch):
    """Cooldown set → next request returns synthetic 429 without network IO."""

    client = PolymarketClient()
    # Skip the rate-limiter and clock for determinism.
    async def _fake_acquire(_endpoint):
        return None

    monkeypatch.setattr("services.polymarket.rate_limiter.acquire", _fake_acquire)

    # Stamp a cooldown on the same endpoint key the wrapper derives.
    from services.polymarket import endpoint_for_url

    endpoint_key = endpoint_for_url("https://data-api.polymarket.com/closed-positions")
    client._stamp_endpoint_cooldown(endpoint_key)

    fake_client = _FakeStreamingClient(failures_before_success=0)
    response = asyncio.run(
        client._rate_limited_get(
            "https://data-api.polymarket.com/closed-positions?user=0xabc",
            client=fake_client,
        )
    )

    # Request short-circuited → no calls were made through the HTTP client.
    assert fake_client.calls == 0
    assert response.status_code == 429


def test_rate_limited_get_stamps_cooldown_when_429_retries_exhausted(monkeypatch):
    """After all retries return 429, the wrapper stamps a cooldown so the
    next caller's request short-circuits instead of repeating the storm."""

    client = PolymarketClient()

    async def _fake_acquire(_endpoint):
        return None

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr("services.polymarket.rate_limiter.acquire", _fake_acquire)
    monkeypatch.setattr("services.polymarket.asyncio.sleep", _fake_sleep)

    fake_client = _Fake429Client()
    response = asyncio.run(
        client._rate_limited_get(
            "https://data-api.polymarket.com/closed-positions?user=0xabc",
            client=fake_client,
        )
    )

    # All retries returned 429 — the final response is still surfaced…
    assert response.status_code == 429
    # …and the endpoint is cooled-down for the next caller.
    from services.polymarket import endpoint_for_url

    endpoint_key = endpoint_for_url("https://data-api.polymarket.com/closed-positions")
    assert client._endpoint_cooldown_remaining(endpoint_key) > 0.0
