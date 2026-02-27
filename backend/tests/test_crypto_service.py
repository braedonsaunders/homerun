from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
SERVICES_ROOT = BACKEND_ROOT / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

import crypto_service


class _FakeResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class _FakeClient:
    def __init__(
        self,
        events: list[dict],
        *,
        midpoint_by_token: dict[str, float] | None = None,
        price_by_token_side: dict[tuple[str, str], float] | None = None,
    ):
        self._events = events
        self._midpoint_by_token = midpoint_by_token or {}
        self._price_by_token_side = price_by_token_side or {}
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append((url, dict(params)))
        if url.endswith("/events"):
            return _FakeResponse(self._events)
        if url.endswith("/midpoint"):
            token_id = str(params.get("token_id", ""))
            if token_id in self._midpoint_by_token:
                return _FakeResponse({"mid": str(self._midpoint_by_token[token_id])})
            return _FakeResponse({"mid": None}, status_code=404)
        if url.endswith("/price"):
            token_id = str(params.get("token_id", ""))
            side = str(params.get("side", "")).lower()
            key = (token_id, side)
            if key in self._price_by_token_side:
                return _FakeResponse({"price": str(self._price_by_token_side[key])})
            return _FakeResponse({"price": None}, status_code=404)
        return _FakeResponse({}, status_code=404)


class _FakePriceToBeatClient:
    def __init__(
        self,
        *,
        response_data: dict | list | None = None,
        status_code: int = 200,
        responses_by_url: dict[str, tuple[dict | list | None, int]] | None = None,
    ):
        self._response_data = response_data
        self._status_code = status_code
        self._responses_by_url = responses_by_url or {}
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        return None

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append((url, dict(params)))
        if url in self._responses_by_url:
            payload, status = self._responses_by_url[url]
            return _FakeResponse(payload, status_code=status)
        return _FakeResponse(self._response_data, status_code=self._status_code)


def _event(slug: str, end_date: str) -> dict:
    return {
        "slug": slug,
        "title": "Bitcoin Up or Down - Test Window",
        "startTime": "2099-01-01T00:00:00Z",
        "endDate": end_date,
        "closed": False,
        "series": [{"volume24hr": 1234.5, "liquidity": 678.9}],
        "markets": [
            {
                "id": "m1",
                "conditionId": "cond1",
                "slug": slug,
                "question": "Bitcoin Up or Down - Test Window",
                "eventStartTime": "2099-01-01T00:00:00Z",
                "endDate": end_date,
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0.52", "0.48"]',
                "liquidityNum": 1000,
                "volumeNum": 100,
                "volume24hr": 50,
                "clobTokenIds": '["tok_up", "tok_down"]',
                "bestBid": 0.51,
                "bestAsk": 0.53,
                "spread": 0.02,
                "lastTradePrice": 0.52,
                "feesEnabled": True,
            }
        ],
    }


def test_fetch_all_requests_events_sorted_by_latest_end_date(monkeypatch):
    fake_events = [_event("btc-updown-5m-test", "2099-01-01T00:05:00Z")]
    fake_client = _FakeClient(fake_events)
    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=10.0: fake_client)
    monkeypatch.setattr(crypto_service, "_get_series_configs", lambda: [("10684", "BTC", "5min")])

    svc = crypto_service.CryptoService(gamma_url="https://gamma-api.polymarket.com")
    markets = svc._fetch_all()

    assert len(markets) == 1
    assert fake_client.calls, "expected Gamma events request"

    url, params = fake_client.calls[0]
    assert url.endswith("/events")
    assert params["series_id"] == "10684"
    assert params["active"] == "true"
    assert params["closed"] == "false"
    assert "end_date_min" in params
    assert isinstance(params["end_date_min"], str) and params["end_date_min"].endswith("Z")
    assert params["order"] == "endDate"
    assert params["ascending"] == "true"


def test_fetch_all_keeps_gamma_prices_and_skips_clob_http_on_hot_path(monkeypatch):
    fake_events = [_event("btc-updown-5m-test", "2099-01-01T00:05:00Z")]
    fake_client = _FakeClient(fake_events)
    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=10.0: fake_client)
    monkeypatch.setattr(crypto_service, "_get_series_configs", lambda: [("10684", "BTC", "5min")])

    svc = crypto_service.CryptoService(gamma_url="https://gamma-api.polymarket.com")
    markets = svc._fetch_all()

    assert len(markets) == 1
    market = markets[0]
    assert market.up_price == 0.52
    assert market.down_price == 0.48
    assert market.best_bid == 0.51
    assert market.best_ask == 0.53
    assert all(url.endswith("/events") for url, _ in fake_client.calls)


def test_update_price_to_beat_uses_crypto_price_api_for_cold_start(monkeypatch):
    start_time = (
        (datetime.now(timezone.utc) - timedelta(minutes=2)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    market = crypto_service.CryptoMarket(
        slug="btc-updown-15m-test",
        asset="BTC",
        timeframe="15min",
        start_time=start_time,
    )

    class _Feed:
        @staticmethod
        def get_price_at_time(asset: str, timestamp_s: float):
            return None

        @staticmethod
        def get_price(asset: str):
            return None

    fake_client = _FakePriceToBeatClient(response_data={"openPrice": 71234.56})
    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=2.0: fake_client)
    monkeypatch.setattr("services.chainlink_feed.get_chainlink_feed", lambda: _Feed())

    svc = crypto_service.CryptoService()
    svc._update_price_to_beat([market])

    assert svc._price_to_beat["btc-updown-15m-test"] == 71234.56
    assert fake_client.calls, "expected crypto-price API call"
    url, params = fake_client.calls[0]
    assert url == "https://polymarket.com/api/crypto/crypto-price"
    assert params["symbol"] == "BTC"
    assert params["variant"] == "fifteen"
    assert params["eventStartTime"] == start_time


def test_update_price_to_beat_falls_back_to_chainlink_history_when_api_missing(monkeypatch):
    start_time = (
        (datetime.now(timezone.utc) - timedelta(minutes=3)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    market = crypto_service.CryptoMarket(
        slug="eth-updown-15m-test",
        asset="ETH",
        timeframe="15min",
        start_time=start_time,
    )

    class _Feed:
        @staticmethod
        def get_price_at_time(asset: str, timestamp_s: float):
            return 1999.25

        @staticmethod
        def get_price(asset: str):
            return None

    fake_client = _FakePriceToBeatClient(response_data={"openPrice": None}, status_code=500)
    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=2.0: fake_client)
    monkeypatch.setattr("services.chainlink_feed.get_chainlink_feed", lambda: _Feed())

    svc = crypto_service.CryptoService()
    svc._update_price_to_beat([market])

    assert svc._price_to_beat["eth-updown-15m-test"] == 1999.25


def test_update_price_to_beat_uses_delayed_chainlink_history_when_exact_point_missing(monkeypatch):
    start_time = (
        (datetime.now(timezone.utc) - timedelta(minutes=3)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    market = crypto_service.CryptoMarket(
        slug="btc-updown-15m-delayed-history",
        asset="BTC",
        timeframe="15min",
        start_time=start_time,
    )

    class _Feed:
        @staticmethod
        def get_price_at_time(asset: str, timestamp_s: float):
            return None

        @staticmethod
        def get_price_at_or_after_time(asset: str, timestamp_s: float, *, max_delay_seconds: float = 300.0):
            return 70654.12

        @staticmethod
        def get_price(asset: str):
            return None

    fake_client = _FakePriceToBeatClient(response_data={"openPrice": None}, status_code=500)
    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=2.0: fake_client)
    monkeypatch.setattr("services.chainlink_feed.get_chainlink_feed", lambda: _Feed())

    svc = crypto_service.CryptoService()
    svc._update_price_to_beat([market])

    assert svc._price_to_beat["btc-updown-15m-delayed-history"] == 70654.12


def test_update_price_to_beat_uses_binance_kline_when_api_rate_limited(monkeypatch):
    base_now = 1_800_000_000.0
    start_dt = datetime.fromtimestamp(base_now - 120.0, tz=timezone.utc)
    start_time = start_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    market = crypto_service.CryptoMarket(
        slug="btc-updown-15m-binance-fallback",
        asset="BTC",
        timeframe="15min",
        start_time=start_time,
    )

    class _Feed:
        @staticmethod
        def get_price_at_time(asset: str, timestamp_s: float):
            return None

        @staticmethod
        def get_price_at_or_after_time(asset: str, timestamp_s: float, *, max_delay_seconds: float = 300.0):
            return None

        @staticmethod
        def get_price(asset: str):
            return None

    start_ms = int(start_dt.timestamp() * 1000)
    fake_client = _FakePriceToBeatClient(
        responses_by_url={
            crypto_service._CRYPTO_PRICE_TO_BEAT_API_URL: ({}, 429),
            crypto_service._BINANCE_KLINES_API_URL: ([[start_ms, "70234.98"]], 200),
        }
    )

    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=2.0: fake_client)
    monkeypatch.setattr("services.chainlink_feed.get_chainlink_feed", lambda: _Feed())
    monkeypatch.setattr(crypto_service.time, "time", lambda: base_now)

    svc = crypto_service.CryptoService()
    svc._update_price_to_beat([market])

    assert svc._price_to_beat["btc-updown-15m-binance-fallback"] == 70234.98


def test_update_price_to_beat_does_not_requery_every_cycle_when_unresolved(monkeypatch):
    base_now = 1_800_000_000.0
    start_dt = datetime.fromtimestamp(base_now - 120.0, tz=timezone.utc)
    start_time = start_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    market = crypto_service.CryptoMarket(
        slug="btc-updown-15m-retry-window",
        asset="BTC",
        timeframe="15min",
        start_time=start_time,
    )

    class _Feed:
        @staticmethod
        def get_price_at_time(asset: str, timestamp_s: float):
            return None

        @staticmethod
        def get_price_at_or_after_time(asset: str, timestamp_s: float, *, max_delay_seconds: float = 300.0):
            return None

        @staticmethod
        def get_price(asset: str):
            return None

    fake_client = _FakePriceToBeatClient(
        responses_by_url={
            crypto_service._CRYPTO_PRICE_TO_BEAT_API_URL: ({}, 429),
            crypto_service._BINANCE_KLINES_API_URL: ([], 500),
        }
    )

    monkeypatch.setattr(crypto_service.httpx, "Client", lambda timeout=2.0: fake_client)
    monkeypatch.setattr("services.chainlink_feed.get_chainlink_feed", lambda: _Feed())
    monkeypatch.setattr(crypto_service.time, "time", lambda: base_now)

    svc = crypto_service.CryptoService()
    svc._update_price_to_beat([market])
    svc._update_price_to_beat([market])

    crypto_api_calls = [call for call in fake_client.calls if call[0] == crypto_service._CRYPTO_PRICE_TO_BEAT_API_URL]
    assert len(crypto_api_calls) == 1
