import sys
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
                "outcomes": "[\"Up\", \"Down\"]",
                "outcomePrices": "[\"0.52\", \"0.48\"]",
                "liquidityNum": 1000,
                "volumeNum": 100,
                "volume24hr": 50,
                "clobTokenIds": "[\"tok_up\", \"tok_down\"]",
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
    monkeypatch.setattr(
        crypto_service.httpx, "Client", lambda timeout=10.0: fake_client
    )
    monkeypatch.setattr(
        crypto_service, "_get_series_configs", lambda: [("10684", "BTC", "5min")]
    )

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


def test_fetch_all_overlays_live_clob_prices_on_current_market(monkeypatch):
    fake_events = [_event("btc-updown-5m-test", "2099-01-01T00:05:00Z")]
    fake_client = _FakeClient(
        fake_events,
        midpoint_by_token={"tok_up": 0.98, "tok_down": 0.02},
        price_by_token_side={
            ("tok_up", "sell"): 0.97,
            ("tok_up", "buy"): 0.99,
        },
    )
    monkeypatch.setattr(
        crypto_service.httpx, "Client", lambda timeout=10.0: fake_client
    )
    monkeypatch.setattr(
        crypto_service, "_get_series_configs", lambda: [("10684", "BTC", "5min")]
    )

    svc = crypto_service.CryptoService(gamma_url="https://gamma-api.polymarket.com")
    markets = svc._fetch_all()

    assert len(markets) == 1
    market = markets[0]
    assert market.up_price == 0.98
    assert market.down_price == 0.02
    assert market.best_bid == 0.97
    assert market.best_ask == 0.99
