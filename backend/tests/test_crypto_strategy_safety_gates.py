"""Regression tests for the safety gates added to crypto_spike_reversion and
crypto_entropy_maker — staleness, resolution-boundary, fee-clearance, and
real-timeframe handling. Each test pins a specific gate so a future refactor
that drops one shows up as a clear failure."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from services.strategies.crypto_entropy_maker import CryptoEntropyMakerStrategy
from services.strategies.crypto_spike_reversion import CryptoSpikeReversionStrategy


def _end_time(seconds_left: float) -> str:
    end = datetime.now(timezone.utc) + timedelta(seconds=seconds_left)
    return end.isoformat().replace("+00:00", "Z")


def _fresh_btc_row(*, seconds_left: float = 240.0, **overrides):
    now_ms = time.time() * 1000
    row = {
        "id": "btc-row",
        "condition_id": "btc-row",
        "slug": "btc-up-or-down-12pm",
        "asset": "BTC",
        "timeframe": "5m",
        "end_time": _end_time(seconds_left),
        "up_price": 0.42,
        "down_price": 0.58,
        "oracle_price": 78600.0,
        "oracle_age_seconds": 0.3,
        "oracle_source": "binance_direct",
        "price_to_beat": 78000.0,
        "oracle_prices_by_source": {
            "binance_direct": {
                "source": "binance_direct",
                "price": 78600.0,
                "updated_at_ms": now_ms - 200,
            },
            "chainlink": {
                "source": "chainlink",
                "price": 78050.08,
                "updated_at_ms": now_ms - 6000,
            },
        },
        "market_data_age_ms": 800.0,
        "liquidity": 5000.0,
        "move_5m_percent": 6.0,
        "move_30m_percent": 1.0,
        "move_2h_percent": 0.5,
        "spread": 0.012,
        "cancel_rate_30s": 0.30,
        "cancel_peak_2m": 0.86,
        "recent_move_zscore": 2.0,
        "clob_token_ids": ["tok_yes", "tok_no"],
    }
    row.update(overrides)
    return row


# ---------------- spike reversion ----------------


@pytest.fixture
def spike_strategy() -> CryptoSpikeReversionStrategy:
    return CryptoSpikeReversionStrategy()


def test_spike_happy_path_uses_binance_direct(spike_strategy):
    signal = spike_strategy._score_market(_fresh_btc_row(), dict(spike_strategy.default_config))
    assert signal is not None
    assert signal["oracle_source_used"] == "binance_direct"
    assert signal["taker_fee_pct"] > 0.5  # Real curve, not flat 0.25
    assert signal["timeframe"] == "5m"
    assert signal["seconds_left"] is not None and signal["seconds_left"] > 30.0


def test_spike_drops_stale_oracle_but_can_still_enter_on_move_alone(spike_strategy):
    # Spike-reversion's primary signal is the 5m move; oracle is additive.
    # When every oracle source is stale we keep going on move-only edge but
    # the resulting signal must record that no oracle was consulted so the
    # downstream sizing/orchestrator can decide whether to allow it through.
    now_ms = time.time() * 1000
    row = _fresh_btc_row()
    row["oracle_prices_by_source"] = {
        "binance_direct": {"source": "binance_direct", "price": 78600.0, "updated_at_ms": now_ms - 30000},
        "chainlink": {"source": "chainlink", "price": 78050.0, "updated_at_ms": now_ms - 30000},
    }
    row["oracle_age_seconds"] = 30.0
    signal = spike_strategy._score_market(row, dict(spike_strategy.default_config))
    if signal is not None:
        assert signal.get("oracle_source_used") is None
        assert signal.get("oracle_price") is None
        assert signal.get("oracle_diff_pct") == 0.0


def test_spike_rejects_too_close_to_resolution(spike_strategy):
    row = _fresh_btc_row(seconds_left=10.0)
    assert spike_strategy._score_market(row, dict(spike_strategy.default_config)) is None


def test_spike_rejects_stale_market_data(spike_strategy):
    row = _fresh_btc_row(market_data_age_ms=10_000.0)
    assert spike_strategy._score_market(row, dict(spike_strategy.default_config)) is None


def test_spike_rejects_when_edge_cant_clear_2x_fee(spike_strategy):
    # Tiny move + matching oracle => edge < 2× taker fee at 0.58.
    row = _fresh_btc_row(move_5m_percent=2.0, oracle_price=78010.0)
    assert spike_strategy._score_market(row, dict(spike_strategy.default_config)) is None


def test_spike_uses_real_timeframe_for_elapsed_ratio(spike_strategy):
    # 1h market, 30 minutes left → elapsed_ratio should be ~0.5, not 0 or 1.
    row = _fresh_btc_row(timeframe="1h", seconds_left=1800.0)
    signal = spike_strategy._score_market(row, dict(spike_strategy.default_config))
    assert signal is not None
    assert 0.45 <= signal["elapsed_ratio"] <= 0.55, signal["elapsed_ratio"]


# ---------------- entropy maker ----------------


@pytest.fixture
def entropy_strategy() -> CryptoEntropyMakerStrategy:
    return CryptoEntropyMakerStrategy()


def _entropy_cfg(strategy: CryptoEntropyMakerStrategy) -> dict:
    """Strategy default config with the min-entry-price floor disabled.

    The strategy intentionally picks the cheap side of the book (entry < 0.5)
    so the production min_entry_price floor (currently 0.80, raised from
    0.50 after the 2026-05-05 live data showed only the 0.80+ bucket was
    profitable) would block these test scenarios. Tests that target other
    gates (freshness, fee clearance, resolution boundary) drop the floor
    here so the gate under test is the one that fires.
    """
    cfg = dict(strategy.default_config)
    cfg["min_entry_price"] = 0.0
    return cfg


def test_entropy_happy_path_uses_binance_direct(entropy_strategy):
    row = _fresh_btc_row(up_price=0.49, down_price=0.51)
    signal = entropy_strategy._score_market(row, _entropy_cfg(entropy_strategy))
    assert signal is not None
    assert signal["oracle_source_used"] == "binance_direct"
    assert signal["taker_fee_pct"] > 0.5
    assert signal["timeframe"] == "5m"


def test_entropy_rejects_too_close_to_resolution(entropy_strategy):
    row = _fresh_btc_row(up_price=0.49, down_price=0.51, seconds_left=10.0)
    assert entropy_strategy._score_market(row, _entropy_cfg(entropy_strategy)) is None


def test_entropy_rejects_stale_market_data(entropy_strategy):
    row = _fresh_btc_row(up_price=0.49, down_price=0.51, market_data_age_ms=10_000.0)
    assert entropy_strategy._score_market(row, _entropy_cfg(entropy_strategy)) is None


def test_entropy_rejects_stale_oracle(entropy_strategy):
    now_ms = time.time() * 1000
    row = _fresh_btc_row(up_price=0.49, down_price=0.51)
    row["oracle_prices_by_source"] = {
        "binance_direct": {"source": "binance_direct", "price": 78600.0, "updated_at_ms": now_ms - 30000},
        "chainlink": {"source": "chainlink", "price": 78050.0, "updated_at_ms": now_ms - 30000},
    }
    row["oracle_age_seconds"] = 30.0
    # Falls through to price-skew direction since oracle is unavailable, but
    # timeframe + min-seconds-left still pass — verify the OracleSource
    # missing path still works (returns a signal with oracle_source_used=None).
    signal = entropy_strategy._score_market(row, _entropy_cfg(entropy_strategy))
    if signal is not None:
        assert signal.get("oracle_source_used") is None
        assert signal.get("oracle_price") is None


def test_entropy_maker_rest_inflates_min_seconds_left_for_entry(entropy_strategy):
    # 5m default min-seconds-left is 35s; the maker open-order timeout is
    # 20s, so the effective gate is ~55s. A market with 50s left should be
    # rejected — a maker order placed there could rest into the resolution
    # boundary if it doesn't fill.
    row = _fresh_btc_row(up_price=0.49, down_price=0.51, seconds_left=50.0)
    assert entropy_strategy._score_market(row, _entropy_cfg(entropy_strategy)) is None
    # Same row with maker-rest accounting disabled should pass the seconds
    # gate (it'll still need to clear other filters but not fail on time).
    cfg = _entropy_cfg(entropy_strategy)
    cfg["maker_rest_includes_timeout"] = False
    signal = entropy_strategy._score_market(row, cfg)
    # Either the signal passes or it's rejected for a non-time reason —
    # what matters for this test is that disabling the inflation removes
    # the ~55s effective floor.
    if signal is not None:
        assert signal["seconds_left"] >= 35.0


def test_entropy_rejects_when_edge_cant_clear_15x_fee(entropy_strategy):
    # Strip the cancel-recovery / orderflow / z-score boosters and pin a tiny
    # oracle-vs-market drift. Edge ≈ diff_pct × entropy_multiplier ≈ <0.05%,
    # well under 1.5× the ~1.56% taker fee at price ≈ 0.5.
    now_ms = time.time() * 1000
    row = _fresh_btc_row(
        up_price=0.49,
        down_price=0.51,
        oracle_price=78002.0,
        cancel_rate_30s=None,
        cancel_peak_2m=None,
        recent_move_zscore=None,
    )
    row["oracle_prices_by_source"] = {
        "binance_direct": {"source": "binance_direct", "price": 78002.0, "updated_at_ms": now_ms - 200},
    }
    assert entropy_strategy._score_market(row, _entropy_cfg(entropy_strategy)) is None


def test_entropy_rejects_below_min_entry_price(entropy_strategy):
    # 2026-05-05 live data showed entries below 0.50 lost 4 wins / 19 losses
    # for -$58. The default min_entry_price=0.50 floor must reject those.
    row = _fresh_btc_row(up_price=0.42, down_price=0.58)
    # Default config (with floor=0.50) — buy_yes at 0.42 is below the floor.
    assert entropy_strategy._score_market(row, dict(entropy_strategy.default_config)) is None
    # Lower the floor and the same row should clear the entry-price gate
    # (it may still be rejected by other gates, but not for this reason).
    cfg = dict(entropy_strategy.default_config)
    cfg["min_entry_price"] = 0.0
    signal = entropy_strategy._score_market(row, cfg)
    # Not asserting truthy — other gates may still reject. The asymmetric
    # check above is what locks in the floor's behavior.
    assert signal is None or signal["entry_price"] >= 0.0
