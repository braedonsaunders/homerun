"""Tests for ``services.execution_safety``.

These pin the defense-in-depth pre-submit floors so a future refactor
that drops, weakens, or accidentally bypasses one shows up as a clear
test failure. The 2026-05-05 entropy-maker incident motivated this
layer: a strategy-level fix in the .py file failed to propagate to
the live system because strategy source_code lives in the database
and isn't overwritten on reseed.

The submit-boundary integration is exercised in
``test_order_manager_safety_floor.py``; these tests cover the
pure-Python safety primitives in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.execution_safety import (
    SafetyAssessment,
    assert_buy_entry_price_within_safety_bounds,
    get_strategy_entry_price_ceiling,
    get_strategy_entry_price_floor,
)


# ---------------------------------------------------------------------------
# Pin-the-floor regression tests. Changing these means changing a hard
# safety bound — a code review must verify the new value matches the
# strategy's empirical profitability window.
# ---------------------------------------------------------------------------


def test_crypto_entropy_maker_floor_is_pinned_at_0_80():
    """The 2026-05-05 review cited 0.80+ as the ONLY profitable bucket
    for ``crypto_entropy_maker``. Loosen at your own peril."""
    assert get_strategy_entry_price_floor("crypto_entropy_maker") == pytest.approx(0.80)


def test_unknown_strategy_has_no_floor():
    assert get_strategy_entry_price_floor("nonexistent_strategy_xyz") is None
    assert get_strategy_entry_price_ceiling("nonexistent_strategy_xyz") is None


# ---------------------------------------------------------------------------
# assert_buy_entry_price_within_safety_bounds — happy path / edge cases
# ---------------------------------------------------------------------------


def test_passes_when_strategy_slug_is_none():
    """Unknown strategy → no floor enforced, never blocks."""
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug=None,
        entry_price=0.10,  # well below any known floor
    )
    assert result.passed is True
    assert result.reason == "ok"


def test_passes_when_strategy_slug_is_empty():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="",
        entry_price=0.10,
    )
    assert result.passed is True


def test_passes_when_strategy_slug_unknown():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="some_user_authored_strategy",
        entry_price=0.05,
    )
    assert result.passed is True


def test_passes_when_entry_price_is_none():
    """Missing entry_price → don't fabricate a violation. The existing
    upstream missing-price gate handles that case."""
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=None,
    )
    assert result.passed is True


def test_passes_when_entry_price_is_garbage():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=float("nan"),  # NaN < anything is False, so passes
    )
    # NaN comparisons return False — explicit safety: don't reject
    # on a malformed price (other gates catch it first).
    assert result.passed is True


def test_passes_at_exactly_the_floor():
    """Edge: floor is INCLUSIVE — entry exactly at the floor is OK."""
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=0.80,
    )
    assert result.passed is True
    assert result.observed == pytest.approx(0.80)
    assert result.floor == pytest.approx(0.80)


def test_passes_above_the_floor():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=0.85,
    )
    assert result.passed is True


# ---------------------------------------------------------------------------
# Rejection path
# ---------------------------------------------------------------------------


def test_rejects_below_floor_for_crypto_entropy_maker():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=0.42,  # 2026-05-05 actual losing trades clustered here
    )
    assert result.passed is False
    assert result.reason == "entry_price_below_safety_floor"
    assert result.floor == pytest.approx(0.80)
    assert result.observed == pytest.approx(0.42)
    assert "crypto_entropy_maker" in result.message
    assert "0.4200" in result.message
    assert "0.8000" in result.message


def test_rejects_just_below_floor():
    """Even a tiny amount below the floor must fail — there's no fudge factor."""
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=0.7999,
    )
    assert result.passed is False
    assert result.reason == "entry_price_below_safety_floor"


def test_rejects_at_zero():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="crypto_entropy_maker",
        entry_price=0.0,
    )
    assert result.passed is False
    assert result.reason == "entry_price_below_safety_floor"


# ---------------------------------------------------------------------------
# Slug normalization
# ---------------------------------------------------------------------------


def test_slug_normalization_is_lower_and_stripped():
    """Slug match is case-insensitive and whitespace-tolerant."""
    result_upper = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="CRYPTO_ENTROPY_MAKER",
        entry_price=0.50,
    )
    result_padded = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="  crypto_entropy_maker  ",
        entry_price=0.50,
    )
    assert result_upper.passed is False
    assert result_padded.passed is False
    assert result_upper.reason == "entry_price_below_safety_floor"
    assert result_padded.reason == "entry_price_below_safety_floor"


def test_safety_assessment_is_immutable_dataclass():
    """Frozen dataclass — callers can't mutate the result post-hoc."""
    result = SafetyAssessment(passed=True, reason="ok", message="all clear")
    with pytest.raises(Exception):  # FrozenInstanceError
        result.passed = False  # type: ignore[misc]
