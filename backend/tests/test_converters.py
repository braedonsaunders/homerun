"""Tests for utils.converters shared helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

from utils.converters import (
    safe_float,
    safe_int,
    clamp,
    to_confidence,
    to_iso,
    normalize_market_id,
)


# --- safe_float ---

class TestSafeFloat:
    def test_valid_string(self):
        assert safe_float("3.14") == 3.14

    def test_valid_int(self):
        assert safe_float(42) == 42.0

    def test_invalid_returns_default_none(self):
        assert safe_float("abc") is None

    def test_invalid_returns_custom_default(self):
        assert safe_float("abc", 0.0) == 0.0

    def test_none_returns_default(self):
        assert safe_float(None) is None
        assert safe_float(None, 5.0) == 5.0

    def test_nan_allowed_by_default(self):
        result = safe_float(float("nan"))
        assert result is not None and math.isnan(result)

    def test_nan_rejected(self):
        assert safe_float(float("nan"), reject_nan_inf=True) is None
        assert safe_float(float("nan"), 0.0, reject_nan_inf=True) == 0.0

    def test_inf_rejected(self):
        assert safe_float(float("inf"), reject_nan_inf=True) is None
        assert safe_float(float("-inf"), 0.0, reject_nan_inf=True) == 0.0

    def test_inf_allowed_by_default(self):
        assert safe_float(float("inf")) == float("inf")


# --- safe_int ---

class TestSafeInt:
    def test_valid_string(self):
        assert safe_int("42") == 42

    def test_valid_float(self):
        assert safe_int(3.9) == 3

    def test_invalid_returns_default(self):
        assert safe_int("abc") == 0
        assert safe_int("abc", 99) == 99

    def test_none_returns_default(self):
        assert safe_int(None) == 0


# --- clamp ---

class TestClamp:
    def test_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_low(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_high(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_boundaries(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


# --- to_confidence ---

class TestToConfidence:
    def test_normal_value(self):
        assert to_confidence(0.75) == 0.75

    def test_percentage_converted(self):
        assert to_confidence(75) == 0.75

    def test_over_100_clamped(self):
        assert to_confidence(150) == 1.0

    def test_negative_clamped(self):
        assert to_confidence(-0.5) == 0.0

    def test_invalid_returns_default(self):
        assert to_confidence("abc", 0.5) == 0.5

    def test_none_returns_default(self):
        assert to_confidence(None) == 0.0


# --- to_iso ---

class TestToIso:
    def test_none(self):
        assert to_iso(None) is None

    def test_naive_datetime(self):
        dt = datetime(2025, 1, 15, 12, 30, 0)
        assert to_iso(dt) == "2025-01-15T12:30:00Z"

    def test_utc_datetime(self):
        dt = datetime(2025, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        assert to_iso(dt) == "2025-01-15T12:30:00Z"

    def test_offset_datetime(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=eastern)
        assert to_iso(dt) == "2025-01-15T17:00:00Z"


# --- normalize_market_id ---

class TestNormalizeMarketId:
    def test_strips_and_lowercases(self):
        assert normalize_market_id("  ABC-123  ") == "abc-123"

    def test_none(self):
        assert normalize_market_id(None) == ""

    def test_empty(self):
        assert normalize_market_id("") == ""

    def test_already_normalized(self):
        assert normalize_market_id("abc") == "abc"
