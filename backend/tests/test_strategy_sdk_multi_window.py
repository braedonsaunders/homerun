"""Tests for MultiWindow and the multi-timeframe confirmation helpers.

These primitives let strategies fan one price stream into multiple
rolling lookbacks (e.g. 5m / 15m / 1h / 4h) and check whether the
windows agree on direction — the building block for compound-movement
strategies.
"""

from __future__ import annotations

import math

import pytest

from services.strategy_helpers.price_window import (
    MultiWindow,
    PriceWindow,
    timeframes_agree,
    weighted_signal,
)


# ---------------------------------------------------------------------------
# MultiWindow construction + basic mechanics
# ---------------------------------------------------------------------------


def test_multi_window_rejects_empty_lookbacks():
    with pytest.raises(ValueError):
        MultiWindow(lookbacks={})


def test_multi_window_rejects_non_positive_lookback():
    with pytest.raises(ValueError):
        MultiWindow(lookbacks={"bad": 0})
    with pytest.raises(ValueError):
        MultiWindow(lookbacks={"bad": -10})


def test_multi_window_creates_one_pricewindow_per_label():
    mw = MultiWindow(lookbacks={"5m": 300, "15m": 900, "1h": 3600})
    assert mw.labels() == ["5m", "15m", "1h"]
    assert len(mw) == 3
    assert isinstance(mw["5m"], PriceWindow)
    assert mw["5m"].window_seconds == 300.0
    assert mw["15m"].window_seconds == 900.0
    assert mw["1h"].window_seconds == 3600.0


def test_multi_window_record_fans_into_all_children():
    mw = MultiWindow(lookbacks={"5m": 300, "15m": 900})
    mw.record(100.0, ts_ms=1_000)
    mw.record(101.0, ts_ms=2_000)
    for label in mw.labels():
        assert len(mw[label]) == 2
        assert mw[label].latest() == (2_000, 101.0)


def test_multi_window_record_uses_shared_ts_for_all_children():
    """Each tick should land at the same ts in every window so
    cross-window math doesn't see staggered timestamps."""
    mw = MultiWindow(lookbacks={"a": 60, "b": 600})
    mw.record(50.0, ts_ms=10_000)
    assert mw["a"].latest()[0] == mw["b"].latest()[0] == 10_000


def test_multi_window_eviction_is_per_window():
    """The longest window should not bound the shortest window's
    eviction — each tracks its own cutoff."""
    mw = MultiWindow(lookbacks={"short": 10, "long": 1000})
    mw.record(100.0, ts_ms=1_000)
    mw.record(101.0, ts_ms=5_000)
    mw.record(102.0, ts_ms=20_000)
    # short window: cutoff = 20_000 - 10_000 = 10_000 → only the 20_000 sample
    assert len(mw["short"]) == 1
    assert mw["short"].samples[0] == (20_000, 102.0)
    # long window: cutoff = 20_000 - 1_000_000 → keeps everything
    assert len(mw["long"]) == 3


def test_multi_window_has_data_requires_all_children_populated():
    mw = MultiWindow(lookbacks={"5m": 300, "15m": 900})
    assert mw.has_data is False
    mw.record(100.0, ts_ms=1_000)
    assert mw.has_data is False  # only 1 sample
    mw.record(101.0, ts_ms=2_000)
    assert mw.has_data is True


def test_multi_window_contains_and_iter():
    mw = MultiWindow(lookbacks={"5m": 300, "15m": 900})
    assert "5m" in mw
    assert "missing" not in mw
    assert list(iter(mw)) == ["5m", "15m"]


def test_multi_window_windows_returns_copy():
    mw = MultiWindow(lookbacks={"5m": 300})
    snapshot = mw.windows()
    assert "5m" in snapshot
    snapshot["injected"] = PriceWindow(window_seconds=60)
    # mutating returned dict must not detach windows from the MultiWindow
    assert "injected" not in mw


# ---------------------------------------------------------------------------
# log_returns / aligned_count / all_agree
# ---------------------------------------------------------------------------


def _seed_uptrend(mw: MultiWindow, *, start: float = 100.0, step: float = 1.0) -> None:
    """Drop a monotonic uptrend across all windows, spaced 1s apart so
    each window has the prior observation available for log_return."""
    for i in range(11):
        mw.record(start + i * step, ts_ms=1_000 + i * 1_000)


def test_log_returns_returns_one_value_per_label():
    mw = MultiWindow(lookbacks={"5s": 5, "10s": 10})
    _seed_uptrend(mw)
    rets = mw.log_returns()
    assert set(rets.keys()) == {"5s", "10s"}
    # 11 samples total, last at ts=11_000. 5s lookback anchors at ts=6_000
    # (sample value 105). 10s anchors at ts=1_000 (sample value 100).
    assert rets["5s"] == pytest.approx(math.log(110 / 105))
    assert rets["10s"] == pytest.approx(math.log(110 / 100))


def test_log_returns_returns_none_when_window_lacks_data():
    mw = MultiWindow(lookbacks={"5m": 300})
    rets = mw.log_returns()
    assert rets == {"5m": None}


def test_aligned_count_counts_only_sign_matching_returns():
    mw = MultiWindow(lookbacks={"5s": 5, "10s": 10})
    _seed_uptrend(mw)
    assert mw.aligned_count(direction="up") == 2
    assert mw.aligned_count(direction="down") == 0


def test_aligned_count_respects_min_return():
    mw = MultiWindow(lookbacks={"5s": 5, "10s": 10})
    _seed_uptrend(mw, step=1.0)
    # log(110/105) ~= 0.0465, log(110/100) ~= 0.0953
    assert mw.aligned_count(direction="up", min_return=0.05) == 1  # only 10s clears
    assert mw.aligned_count(direction="up", min_return=0.20) == 0


def test_all_agree_requires_full_population_and_alignment():
    mw = MultiWindow(lookbacks={"5s": 5, "10s": 10})
    assert mw.all_agree(direction="up") is False  # empty
    _seed_uptrend(mw)
    assert mw.all_agree(direction="up") is True
    assert mw.all_agree(direction="down") is False


def test_all_agree_with_partial_data_is_false():
    """Even if every populated window agrees, all_agree is False unless
    every child has data — partial agreement is misleading."""
    mw = MultiWindow(lookbacks={"5s": 5, "30s": 30})
    # Only enough samples to populate the short window's lookback
    mw.record(100.0, ts_ms=1_000)
    mw.record(105.0, ts_ms=6_000)
    # 30s window has 2 samples but no anchor 30s ago → log_return None
    assert mw.has_data is True  # both have ≥2 samples...
    assert mw.log_returns()["30s"] is None  # ...but the 30s lookback is empty
    assert mw.all_agree(direction="up") is False


def test_direction_aliases_normalize():
    mw = MultiWindow(lookbacks={"5s": 5, "10s": 10})
    _seed_uptrend(mw)
    for alias in ("up", "long", "buy", "yes", "+", "1"):
        assert mw.aligned_count(direction=alias) == 2
    for alias in ("down", "short", "sell", "no", "-", "-1"):
        assert mw.aligned_count(direction=alias) == 0


def test_unknown_direction_raises():
    mw = MultiWindow(lookbacks={"5s": 5})
    with pytest.raises(ValueError):
        mw.aligned_count(direction="sideways")


# ---------------------------------------------------------------------------
# timeframes_agree (module-level helper, accepts plain dicts)
# ---------------------------------------------------------------------------


def test_timeframes_agree_min_count_threshold():
    rets = {"5m": 0.02, "15m": 0.03, "1h": -0.01, "4h": None}
    assert timeframes_agree(rets, direction="up", min_count=1) is True
    assert timeframes_agree(rets, direction="up", min_count=2) is True
    assert timeframes_agree(rets, direction="up", min_count=3) is False  # only 2 up


def test_timeframes_agree_skips_none():
    rets = {"5m": None, "15m": None}
    assert timeframes_agree(rets, direction="up", min_count=1) is False


def test_timeframes_agree_min_return_filter():
    rets = {"5m": 0.005, "15m": 0.05}
    assert timeframes_agree(rets, direction="up", min_count=1, min_return=0.01) is True
    assert timeframes_agree(rets, direction="up", min_count=2, min_return=0.01) is False


def test_timeframes_agree_min_count_must_be_positive():
    with pytest.raises(ValueError):
        timeframes_agree({"5m": 0.01}, min_count=0)


# ---------------------------------------------------------------------------
# weighted_signal
# ---------------------------------------------------------------------------


def test_weighted_signal_basic():
    rets = {"5m": 0.01, "15m": 0.02, "1h": 0.03, "4h": 0.04}
    weights = {"5m": 1, "15m": 2, "1h": 3, "4h": 4}
    expected = (1 * 0.01 + 2 * 0.02 + 3 * 0.03 + 4 * 0.04) / (1 + 2 + 3 + 4)
    assert weighted_signal(rets, weights) == pytest.approx(expected)


def test_weighted_signal_renormalizes_when_some_labels_missing():
    rets = {"5m": 0.10, "15m": None, "1h": 0.20, "4h": None}
    weights = {"5m": 1, "15m": 2, "1h": 3, "4h": 4}
    # Renormalise over 5m+1h only: (1*0.10 + 3*0.20) / (1+3) = 0.175
    assert weighted_signal(rets, weights) == pytest.approx((0.10 + 0.60) / 4)


def test_weighted_signal_returns_none_when_nothing_contributes():
    assert weighted_signal({"a": None, "b": None}, {"a": 1, "b": 2}) is None
    assert weighted_signal({"a": 1.0}, {"a": 0, "b": 0}) is None


def test_weighted_signal_ignores_negative_or_zero_weights():
    rets = {"a": 0.10, "b": 0.20}
    # Negative weights are dropped (treated like zero) — caller's contract
    # is positive weights only.
    assert weighted_signal(rets, {"a": 1, "b": -1}) == pytest.approx(0.10)
    assert weighted_signal(rets, {"a": 1, "b": 0}) == pytest.approx(0.10)
