"""Locks down the docs-accurate Polymarket taker-fee curve in utils.kelly.

Polymarket's Fee Structure V2 (live 2026-03-30) charges takers:

    fee_per_share = feeRate * p * (1 - p)

with a category-dependent ``feeRate`` (crypto 0.07, sports/economics/
culture/weather/other 0.05, finance/politics/mentions/tech 0.04,
geopolitics 0.0).  Makers pay zero.

Two earlier shapes were wrong and are guarded against here:

  * ``p * (1-p) * 0.0625`` — right shape, hardcoded blended rate.
  * ``p * 0.25 * (p * (1-p))**2`` — a *quadratic* that understated the
    real crypto fee at p=0.50 by 2.24x (1.56% of notional vs the true
    3.5%) and collapsed far too fast toward the tails.

Source: https://docs.polymarket.com/trading/fees
"""

from __future__ import annotations

import math

import pytest

from utils.kelly import (
    POLYMARKET_DEFAULT_TAKER_FEE_RATE,
    POLYMARKET_TAKER_FEE_RATES,
    polymarket_fee_rate_for_category,
    polymarket_taker_fee,
    polymarket_taker_fee_pct,
)


@pytest.mark.parametrize(
    "price, expected_per_share, expected_pct",
    [
        # Crypto (default) rate 0.07: fee = 0.07 * p * (1-p)
        (0.10, 0.0063, 6.30),
        (0.30, 0.0147, 4.90),
        (0.50, 0.0175, 3.50),
        (0.70, 0.0147, 2.10),
        (0.90, 0.0063, 0.70),
    ],
)
def test_curve_matches_polymarket_docs(price, expected_per_share, expected_pct):
    fee = polymarket_taker_fee(price)
    pct_decimal = polymarket_taker_fee_pct(price)
    assert math.isclose(fee, expected_per_share, abs_tol=1e-6), fee
    assert math.isclose(pct_decimal * 100.0, expected_pct, abs_tol=0.01), pct_decimal


def test_max_absolute_fee_is_at_half_price():
    """The fee peaks at p=0.50 — 1.75 cents/share on crypto ($1.75/100)."""
    peak = polymarket_taker_fee(0.5)
    assert math.isclose(peak, 0.0175, abs_tol=1e-9)
    for p in (0.05, 0.2, 0.35, 0.65, 0.8, 0.95):
        assert polymarket_taker_fee(p) < peak


@pytest.mark.parametrize(
    "category, rate",
    [
        ("crypto", 0.07),
        ("Crypto", 0.07),
        ("  SPORTS  ", 0.05),
        ("politics", 0.04),
        ("finance", 0.04),
        ("tech", 0.04),
        ("mentions", 0.04),
        ("weather", 0.05),
        ("geopolitics", 0.0),
    ],
)
def test_category_rates(category, rate):
    assert polymarket_fee_rate_for_category(category) == rate
    assert math.isclose(
        polymarket_taker_fee(0.5, category=category), rate * 0.25, abs_tol=1e-12
    )


def test_geopolitics_is_fee_free():
    assert polymarket_taker_fee(0.5, category="geopolitics") == 0.0
    assert polymarket_taker_fee_pct(0.5, category="geopolitics") == 0.0


def test_unknown_and_missing_category_fall_back_to_conservative_default():
    assert polymarket_fee_rate_for_category(None) == POLYMARKET_DEFAULT_TAKER_FEE_RATE
    assert polymarket_fee_rate_for_category("") == POLYMARKET_DEFAULT_TAKER_FEE_RATE
    assert (
        polymarket_fee_rate_for_category("no-such-category")
        == POLYMARKET_DEFAULT_TAKER_FEE_RATE
    )
    # The default must be the highest non-zero published rate so an
    # unknown category can never under-charge a fee-aware gate.
    assert POLYMARKET_DEFAULT_TAKER_FEE_RATE == max(POLYMARKET_TAKER_FEE_RATES.values())


def test_explicit_fee_rate_overrides_category():
    assert math.isclose(
        polymarket_taker_fee(0.5, 0.04, category="crypto"), 0.01, abs_tol=1e-12
    )


def test_fee_pct_zero_at_zero_price():
    assert polymarket_taker_fee_pct(0.0) == 0.0
    assert polymarket_taker_fee_pct(-0.5) == 0.0


def test_fee_clamps_outside_unit_interval():
    # >1 and <0 are clamped — strategies can pass slightly-out-of-range
    # values from arithmetic without blowing up.
    assert polymarket_taker_fee(1.5) == polymarket_taker_fee(1.0)
    assert polymarket_taker_fee(-0.2) == polymarket_taker_fee(0.0)


def test_not_the_old_quadratic_shape():
    """Regression guard: the retired quadratic must not creep back."""
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        quadratic = p * 0.25 * (p * (1.0 - p)) ** 2
        assert not math.isclose(polymarket_taker_fee(p), quadratic, abs_tol=1e-6)
