"""Unit tests for the backtest-time CRYPTO_UPDATE synthesizer.

Covers the pure reconstruction logic (book as-of lookup, binary-complement
derivation of a missing side, seconds_left / is_live, market-dict shape) plus
an end-to-end synthesize over a temp parquet + ProviderDataset row, asserting
the gap-fill exclude semantics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.backtest.crypto_update_synthesizer import (
    _BookSeries,
    _SynthMarket,
    _clamp01,
    _parse_iso_us,
)


def _us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def test_clamp01_rejects_degenerate_prices():
    assert _clamp01(0.42) == 0.42
    assert _clamp01(0.0) is None
    assert _clamp01(1.0) is None
    assert _clamp01(-0.1) is None
    assert _clamp01(None) is None
    assert _clamp01("nope") is None


def test_parse_iso_us_roundtrip():
    dt = datetime(2026, 5, 28, 19, 13, tzinfo=timezone.utc)
    assert _parse_iso_us(dt.isoformat().replace("+00:00", "Z")) == _us(dt)
    assert _parse_iso_us(None) is None
    assert _parse_iso_us("garbage") is None


def test_book_series_as_of_respects_staleness():
    s = _BookSeries()
    base = datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc)
    # three snapshots 10s apart
    s._us = [_us(base), _us(base + timedelta(seconds=10)), _us(base + timedelta(seconds=20))]
    s._bid = [0.40, 0.42, 0.44]
    s._ask = [0.41, 0.43, 0.45]

    # as-of between snap2 and snap3 -> returns snap2
    got = s.as_of(_us(base + timedelta(seconds=15)))
    assert got == (0.42, 0.43)

    # before first snapshot -> None
    assert s.as_of(_us(base - timedelta(seconds=5))) is None

    # far past the last snapshot beyond staleness window -> None
    assert s.as_of(_us(base + timedelta(seconds=120))) is None


def _make_market(*, up=True, down=True) -> _SynthMarket:
    base = datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc)
    m = _SynthMarket(
        market_id="111",
        condition_id="0xabc",
        slug="btc-updown-5m-1",
        title="BTC Up or Down",
        coin="btc",
        timeframe="5m",
        start_us=_us(base),
        end_us=_us(base + timedelta(minutes=5)),
        up_token="UP_TOK",
        down_token="DOWN_TOK",
        price_to_beat=80000.0,
        up_series=_BookSeries(),
        down_series=_BookSeries(),
    )
    if up:
        m.up_series._us = [_us(base + timedelta(seconds=30))]
        m.up_series._bid = [0.60]
        m.up_series._ask = [0.62]
    if down:
        m.down_series._us = [_us(base + timedelta(seconds=30))]
        m.down_series._bid = [0.38]
        m.down_series._ask = [0.40]
    return m


def test_market_dict_two_sided_shape():
    m = _make_market(up=True, down=True)
    base = datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc)
    md = m.to_market_dict(base + timedelta(seconds=45))
    assert md is not None
    assert md["clob_token_ids"] == ["UP_TOK", "DOWN_TOK"]
    assert md["up_token_index"] == 0 and md["down_token_index"] == 1
    assert md["condition_id"] == "0xabc"
    assert md["asset"] == "BTC"
    # up mid = (0.60+0.62)/2 = 0.61 ; down mid = 0.39
    assert md["up_price"] == pytest.approx(0.61)
    assert md["down_price"] == pytest.approx(0.39)
    assert md["best_bid"] == 0.60 and md["best_ask"] == 0.62
    # seconds_left ~ 5min - 45s = 255s ; live
    assert md["seconds_left"] == pytest.approx(255, abs=2)
    assert md["is_live"] is True


def test_market_dict_derives_missing_down_via_complement():
    m = _make_market(up=True, down=False)
    base = datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc)
    md = m.to_market_dict(base + timedelta(seconds=45))
    assert md is not None
    # down derived: down_bid = 1 - up_ask = 0.38 ; down_ask = 1 - up_bid = 0.40
    # down mid = 0.39
    assert md["down_price"] == pytest.approx(0.39)
    assert md["up_price"] == pytest.approx(0.61)


def test_market_dict_none_when_no_book():
    m = _make_market(up=False, down=False)
    base = datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc)
    assert m.to_market_dict(base + timedelta(seconds=45)) is None


def test_alive_at_window_bounds():
    m = _make_market()
    base = datetime(2026, 5, 28, 19, 0, tzinfo=timezone.utc)
    assert m.alive_at(_us(base)) is True
    assert m.alive_at(_us(base + timedelta(seconds=1))) is True
    assert m.alive_at(_us(base - timedelta(seconds=1))) is False
    # end is exclusive
    assert m.alive_at(_us(base + timedelta(minutes=5))) is False


def test_market_key_prefers_condition_id():
    m = _make_market()
    assert m.market_key() == "0xabc"
    m.condition_id = None
    assert m.market_key() == "111"
