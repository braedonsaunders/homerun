import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import crypto_worker


class _FakePriceCache:
    def __init__(self, mids: dict[str, float], fresh: set[str]):
        self._mids = dict(mids)
        self._fresh = set(fresh)

    def is_fresh(self, token_id: str) -> bool:
        return token_id in self._fresh

    def get_mid(self, token_id: str):
        return self._mids.get(token_id)


def test_collect_ws_prices_for_markets_uses_only_fresh_valid_tokens():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    stale_token = "stale_token_12345678901234567890"

    markets = [
        SimpleNamespace(clob_token_ids=[up_token, down_token, stale_token, "short_token"]),
    ]
    feed_manager = SimpleNamespace(
        _started=True,
        price_cache=_FakePriceCache(
            mids={
                up_token: 0.623,
                down_token: 1.009,
                stale_token: 0.42,
            },
            fresh={up_token, down_token},
        ),
    )

    out = crypto_worker._collect_ws_prices_for_markets(feed_manager, markets)

    assert out == {
        up_token: 0.623,
        down_token: 1.0,
    }


def test_overlay_ws_prices_on_market_row_prefers_ws_for_both_legs():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    market = SimpleNamespace(
        clob_token_ids=[up_token, down_token],
        up_token_index=0,
        down_token_index=1,
    )
    row = {"up_price": 0.41, "down_price": 0.59, "combined": 1.0}

    crypto_worker._overlay_ws_prices_on_market_row(
        row,
        market,
        {up_token: 0.71, down_token: 0.29},
    )

    assert row["up_price"] == 0.71
    assert row["down_price"] == 0.29
    assert row["combined"] == 1.0


def test_overlay_ws_prices_on_market_row_derives_missing_opposite_leg():
    up_token = "up_token_123456789012345678901234"
    down_token = "down_token_1234567890123456789012"
    market = SimpleNamespace(
        clob_token_ids=[up_token, down_token],
        up_token_index=0,
        down_token_index=1,
    )
    row = {"up_price": 0.41, "down_price": 0.59, "combined": 1.0}

    crypto_worker._overlay_ws_prices_on_market_row(
        row,
        market,
        {up_token: 0.82},
    )

    assert row["up_price"] == 0.82
    assert row["down_price"] == pytest.approx(0.18)
    assert row["combined"] == 1.0
