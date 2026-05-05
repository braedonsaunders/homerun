"""Tests for ``services.execution_safety``.

These exercise the strategy-agnostic mechanism for installing and
enforcing operator pre-submit safety floors/ceilings. The platform
makes no assumption about which strategy slugs exist (strategies are
DB-backed and user-managed), so these tests use synthetic slugs
("alpha", "beta", ...) — they verify the mechanism, not any specific
operator policy.

The submit-boundary integration is exercised by the order-manager
tests; these tests cover the pure-Python safety primitives in
isolation.
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
    clear_all_strategy_entry_price_bounds,
    get_strategy_entry_price_ceiling,
    get_strategy_entry_price_floor,
    register_strategy_entry_price_ceiling,
    register_strategy_entry_price_floor,
    unregister_strategy_entry_price_ceiling,
    unregister_strategy_entry_price_floor,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty floor/ceiling registry. The
    registry is module-level state — without this fixture cross-test
    pollution would silently mask bugs. Cleared again on teardown so
    we don't leak state into other test files."""
    clear_all_strategy_entry_price_bounds()
    yield
    clear_all_strategy_entry_price_bounds()


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


def test_registry_is_empty_by_default():
    """Platform makes no assumption about which strategies exist."""
    assert get_strategy_entry_price_floor("alpha") is None
    assert get_strategy_entry_price_ceiling("alpha") is None


def test_unknown_strategy_passes_when_no_floor_registered():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.05,  # would be very low for any sensible floor
    )
    assert result.passed is True
    assert result.reason == "ok"


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------


def test_register_floor_installs_the_floor():
    register_strategy_entry_price_floor("alpha", 0.50)
    assert get_strategy_entry_price_floor("alpha") == pytest.approx(0.50)


def test_register_floor_overwrites_existing_value():
    register_strategy_entry_price_floor("alpha", 0.50)
    register_strategy_entry_price_floor("alpha", 0.75)
    assert get_strategy_entry_price_floor("alpha") == pytest.approx(0.75)


def test_register_ceiling_installs_the_ceiling():
    register_strategy_entry_price_ceiling("beta", 0.99)
    assert get_strategy_entry_price_ceiling("beta") == pytest.approx(0.99)


def test_register_floor_ignores_empty_slug():
    register_strategy_entry_price_floor("", 0.50)
    register_strategy_entry_price_floor("   ", 0.50)
    assert get_strategy_entry_price_floor("") is None
    assert get_strategy_entry_price_floor("   ") is None


def test_register_floor_ignores_non_numeric_value():
    register_strategy_entry_price_floor("alpha", "not_a_number")  # type: ignore[arg-type]
    assert get_strategy_entry_price_floor("alpha") is None


def test_register_floor_ignores_nan_and_inf():
    register_strategy_entry_price_floor("alpha", float("nan"))
    register_strategy_entry_price_floor("alpha", float("inf"))
    register_strategy_entry_price_floor("alpha", float("-inf"))
    assert get_strategy_entry_price_floor("alpha") is None


def test_unregister_floor_removes_it():
    register_strategy_entry_price_floor("alpha", 0.50)
    unregister_strategy_entry_price_floor("alpha")
    assert get_strategy_entry_price_floor("alpha") is None


def test_unregister_floor_is_idempotent_for_unknown_slug():
    # No exception, no side effect.
    unregister_strategy_entry_price_floor("never_registered")
    assert get_strategy_entry_price_floor("never_registered") is None


def test_unregister_ceiling_removes_it():
    register_strategy_entry_price_ceiling("beta", 0.99)
    unregister_strategy_entry_price_ceiling("beta")
    assert get_strategy_entry_price_ceiling("beta") is None


def test_clear_all_removes_floors_and_ceilings():
    register_strategy_entry_price_floor("alpha", 0.50)
    register_strategy_entry_price_ceiling("alpha", 0.95)
    register_strategy_entry_price_floor("beta", 0.30)
    clear_all_strategy_entry_price_bounds()
    assert get_strategy_entry_price_floor("alpha") is None
    assert get_strategy_entry_price_ceiling("alpha") is None
    assert get_strategy_entry_price_floor("beta") is None


def test_slug_normalization_is_case_insensitive_and_stripped():
    register_strategy_entry_price_floor("Alpha", 0.50)
    assert get_strategy_entry_price_floor("alpha") == pytest.approx(0.50)
    assert get_strategy_entry_price_floor("ALPHA") == pytest.approx(0.50)
    assert get_strategy_entry_price_floor("  alpha  ") == pytest.approx(0.50)


def test_floors_and_ceilings_are_independent():
    register_strategy_entry_price_floor("alpha", 0.50)
    register_strategy_entry_price_ceiling("alpha", 0.90)
    assert get_strategy_entry_price_floor("alpha") == pytest.approx(0.50)
    assert get_strategy_entry_price_ceiling("alpha") == pytest.approx(0.90)
    unregister_strategy_entry_price_floor("alpha")
    assert get_strategy_entry_price_floor("alpha") is None
    assert get_strategy_entry_price_ceiling("alpha") == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Bounds-checking — happy path
# ---------------------------------------------------------------------------


def test_passes_when_strategy_slug_is_none():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug=None,
        entry_price=0.10,
    )
    assert result.passed is True
    assert result.reason == "ok"


def test_passes_when_strategy_slug_is_empty():
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="",
        entry_price=0.10,
    )
    assert result.passed is True


def test_passes_when_no_bounds_registered_for_slug():
    """Empty registry → never blocks, regardless of price."""
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="some_user_authored_strategy",
        entry_price=0.05,
    )
    assert result.passed is True


def test_passes_when_entry_price_is_none():
    """Missing entry_price → don't fabricate a violation. Upstream
    missing-price gate handles that case."""
    register_strategy_entry_price_floor("alpha", 0.80)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=None,
    )
    assert result.passed is True


def test_passes_when_entry_price_is_nan():
    """NaN comparisons return False — explicit safety: don't reject
    on a malformed price (other gates catch it first)."""
    register_strategy_entry_price_floor("alpha", 0.80)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=float("nan"),
    )
    assert result.passed is True


def test_passes_when_entry_price_not_numeric():
    register_strategy_entry_price_floor("alpha", 0.80)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price="not_a_number",  # type: ignore[arg-type]
    )
    assert result.passed is True


def test_passes_at_exactly_the_floor():
    """Edge: floor is INCLUSIVE — entry exactly at the floor is OK."""
    register_strategy_entry_price_floor("alpha", 0.50)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.50,
    )
    assert result.passed is True
    assert result.observed == pytest.approx(0.50)
    assert result.floor == pytest.approx(0.50)


def test_passes_above_the_floor():
    register_strategy_entry_price_floor("alpha", 0.50)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.85,
    )
    assert result.passed is True


def test_passes_at_exactly_the_ceiling():
    """Edge: ceiling is INCLUSIVE — entry exactly at the ceiling is OK."""
    register_strategy_entry_price_ceiling("alpha", 0.99)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.99,
    )
    assert result.passed is True


def test_passes_below_the_ceiling():
    register_strategy_entry_price_ceiling("alpha", 0.95)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.50,
    )
    assert result.passed is True


# ---------------------------------------------------------------------------
# Bounds-checking — rejection path
# ---------------------------------------------------------------------------


def test_rejects_below_floor():
    register_strategy_entry_price_floor("alpha", 0.50)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.42,
    )
    assert result.passed is False
    assert result.reason == "entry_price_below_safety_floor"
    assert result.floor == pytest.approx(0.50)
    assert result.observed == pytest.approx(0.42)
    assert "alpha" in result.message
    assert "0.4200" in result.message
    assert "0.5000" in result.message


def test_rejects_just_below_floor():
    """No fudge factor: even a tiny amount below the floor must fail."""
    register_strategy_entry_price_floor("alpha", 0.80)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.7999,
    )
    assert result.passed is False
    assert result.reason == "entry_price_below_safety_floor"


def test_rejects_at_zero():
    register_strategy_entry_price_floor("alpha", 0.50)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.0,
    )
    assert result.passed is False
    assert result.reason == "entry_price_below_safety_floor"


def test_rejects_above_ceiling():
    register_strategy_entry_price_ceiling("alpha", 0.95)
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.97,
    )
    assert result.passed is False
    assert result.reason == "entry_price_above_safety_ceiling"
    assert result.ceiling == pytest.approx(0.95)
    assert result.observed == pytest.approx(0.97)


def test_floor_check_runs_before_ceiling():
    """When BOTH bounds are violated (impossible if floor < ceiling, but
    well-defined for misconfigured pairs), the floor message wins so
    the operator sees the more conservative violation first.

    The check is well-defined for floor > ceiling configs too: such a
    config is operator error, but the function still returns a stable
    SafetyAssessment rather than crashing."""
    register_strategy_entry_price_floor("alpha", 0.95)
    register_strategy_entry_price_ceiling("alpha", 0.50)  # misconfigured
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="alpha",
        entry_price=0.70,  # above ceiling AND below floor
    )
    assert result.passed is False
    # Implementation order: floor checked first.
    assert result.reason == "entry_price_below_safety_floor"


# ---------------------------------------------------------------------------
# Slug normalization on the check path
# ---------------------------------------------------------------------------


def test_check_path_slug_normalization_is_lower_and_stripped():
    register_strategy_entry_price_floor("alpha", 0.80)
    result_upper = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="ALPHA",
        entry_price=0.50,
    )
    result_padded = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="  alpha  ",
        entry_price=0.50,
    )
    assert result_upper.passed is False
    assert result_padded.passed is False
    assert result_upper.reason == "entry_price_below_safety_floor"
    assert result_padded.reason == "entry_price_below_safety_floor"


def test_one_strategys_floor_does_not_affect_another():
    """Per-slug isolation — the registry is keyed by slug."""
    register_strategy_entry_price_floor("alpha", 0.80)
    # 'beta' has no floor; the alpha floor must not leak across.
    result = assert_buy_entry_price_within_safety_bounds(
        strategy_slug="beta",
        entry_price=0.10,
    )
    assert result.passed is True


def test_safety_assessment_is_immutable_dataclass():
    """Frozen dataclass — callers can't mutate the result post-hoc."""
    result = SafetyAssessment(passed=True, reason="ok", message="all clear")
    with pytest.raises(Exception):  # FrozenInstanceError
        result.passed = False  # type: ignore[misc]
