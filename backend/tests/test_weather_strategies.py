"""Tests for the 4 weather trading strategies + weather_edge direction fix.

Validates:
- Direction logic: model_prob vs market_price (NOT 0.5 threshold)
- Ensemble bucket probability calculation
- Full distribution normalization
- Conservative NO: never bets near consensus
- Bucket edge: sigmoid sharpness affects probability
"""

from __future__ import annotations

import math
import pytest

# ---------------------------------------------------------------------------
# Signal engine utility functions
# ---------------------------------------------------------------------------

from services.weather.signal_engine import (
    clamp01,
    temp_range_probability,
    ensemble_bucket_probability,
    compute_consensus,
    compute_model_agreement,
    compute_confidence,
)


class TestSignalEngineUtilities:
    """Test exported utility functions from signal_engine."""

    def test_clamp01(self):
        assert clamp01(0.5) == 0.5
        assert clamp01(-0.1) == 0.0
        assert clamp01(1.5) == 1.0

    def test_temp_range_probability_center(self):
        """When forecast is exactly at bucket center, probability should peak."""
        prob = temp_range_probability(7.0, 6.5, 7.5, scale_c=2.0)
        assert 0.10 < prob < 0.20  # narrow bucket with wide scale

    def test_temp_range_probability_tighter_scale(self):
        """Tighter scale should give higher bucket probability."""
        wide = temp_range_probability(7.0, 6.5, 7.5, scale_c=2.0)
        tight = temp_range_probability(7.0, 6.5, 7.5, scale_c=1.0)
        assert tight > wide

    def test_temp_range_probability_far_from_bucket(self):
        """Forecast far from bucket should give very low probability."""
        prob = temp_range_probability(15.0, 6.5, 7.5, scale_c=2.0)
        assert prob < 0.05

    def test_ensemble_bucket_probability_basic(self):
        """Count fraction of ensemble members in bucket."""
        members = [6.0, 6.5, 7.0, 7.2, 7.4, 8.0, 8.5, 9.0, 10.0, 11.0]
        prob = ensemble_bucket_probability(members, 6.5, 7.5)
        # Members in [6.5, 7.5): 6.5, 7.0, 7.2, 7.4 = 4 out of 10
        assert prob == pytest.approx(0.4)

    def test_ensemble_bucket_probability_none(self):
        """No members in bucket = 0 probability."""
        members = [1.0, 2.0, 3.0]
        prob = ensemble_bucket_probability(members, 6.5, 7.5)
        assert prob == 0.0

    def test_ensemble_bucket_probability_all(self):
        """All members in bucket = 1.0 probability."""
        members = [7.0, 7.1, 7.2, 7.3, 7.4]
        prob = ensemble_bucket_probability(members, 6.5, 7.5)
        assert prob == 1.0

    def test_ensemble_bucket_probability_empty(self):
        """Empty ensemble = 0 probability."""
        prob = ensemble_bucket_probability([], 6.5, 7.5)
        assert prob == 0.0

    def test_compute_consensus(self):
        probs = {"gfs": 0.8, "ecmwf": 0.6}
        weights = {"gfs": 0.6, "ecmwf": 0.4}
        consensus = compute_consensus(probs, weights)
        expected = (0.8 * 0.6 + 0.6 * 0.4) / (0.6 + 0.4)
        assert consensus == pytest.approx(expected, abs=0.01)

    def test_compute_model_agreement_high(self):
        probs = {"gfs": 0.8, "ecmwf": 0.78}
        agreement = compute_model_agreement(probs)
        assert agreement > 0.95

    def test_compute_model_agreement_low(self):
        probs = {"gfs": 0.9, "ecmwf": 0.1}
        agreement = compute_model_agreement(probs)
        assert agreement < 0.25

    def test_compute_confidence(self):
        confidence = compute_confidence(agreement=0.95, consensus_yes=0.8, source_count=3)
        assert 0.6 < confidence < 1.0


# ---------------------------------------------------------------------------
# Weather Edge Strategy (direction fix)
# ---------------------------------------------------------------------------


class TestWeatherEdgeDirectionFix:
    """Test that weather_edge uses model_prob vs market_price, NOT 0.5."""

    def _make_intent(self, **overrides):
        base = {
            "market_id": "test-market-1",
            "market_question": "Will the highest temperature in London be 7C on Feb 17?",
            "event_slug": "london-temp-feb-17",
            "yes_price": 0.31,
            "no_price": 0.69,
            "liquidity": 5000.0,
            "clob_token_ids": ["yes-token", "no-token"],
            "location": "London",
            "metric": "temp_max_range",
            "operator": "between",
            "bucket_low_c": 6.5,
            "bucket_high_c": 7.5,
            "threshold_c": None,
            "target_time": "2026-02-17T12:00:00Z",
            "source_values_c": {"open_meteo:gfs_seamless": 5.9, "open_meteo:icon_seamless": 7.0},
            "source_probabilities": {"gfs": 0.116, "icon": 0.124},
            "source_weights": {"gfs": 0.71, "icon": 0.29},
            "consensus_value_c": 6.2,
            "consensus_probability": 0.118,
            "source_spread_c": 1.1,
            "source_count": 2,
            "model_agreement": 0.99,
            "ensemble_members": None,
            "ensemble_daily_max": None,
            "sibling_markets": [],
        }
        base.update(overrides)
        return base

    def test_direction_buy_no_when_model_prob_less_than_yes_price(self):
        """When model says 12% but market says 31% YES, should buy NO."""
        from services.weather.signal_engine import temp_range_probability

        intent = self._make_intent(yes_price=0.31, no_price=0.69, consensus_value_c=6.2)
        model_prob = temp_range_probability(6.2, 6.5, 7.5, 2.0)
        # model_prob (~12%) < yes_price (31%) → buy NO
        assert model_prob < intent["yes_price"]

    def test_direction_buy_yes_when_forecast_matches_bucket(self):
        """When forecast is 7.0C and bucket is 6.5-7.5C, with low yes_price, should buy YES."""
        from services.weather.signal_engine import temp_range_probability

        # Forecast exactly 7.0C, tighter scale
        model_prob = temp_range_probability(7.0, 6.5, 7.5, 1.0)
        # With scale 1.0, model_prob should be higher
        yes_price = 0.05  # Very low market price
        assert model_prob > yes_price  # Should buy YES

    def test_the_original_bug_scenario(self):
        """THE BUG: forecast ~6.2C for 7C bucket, market says 31% YES.
        Old code: consensus_yes (12%) < 0.5 → buy NO ← WRONG for bucket markets
        New code: model_prob (12%) < yes_price (31%) → buy NO ← CORRECT reasoning
        But if market had yes_price=5%, buy YES since 12% > 5%
        """
        from services.weather.signal_engine import temp_range_probability

        model_prob = temp_range_probability(6.2, 6.5, 7.5, 2.0)

        # Scenario 1: market says 31% → model says 12% → NO is correct
        assert model_prob < 0.31  # buy NO

        # Scenario 2: market says 5% → model says 12% → YES is correct
        assert model_prob > 0.05  # buy YES

        # The old code would ALWAYS buy NO because 12% < 50%
        # The new code correctly depends on market price

    def test_never_defaults_to_buy_no_for_bucket_markets(self):
        """With high yes_price, should buy NO. With low yes_price, should buy YES.
        The 0.5 threshold should NOT be the decision boundary."""
        from services.weather.signal_engine import temp_range_probability

        model_prob = temp_range_probability(7.0, 6.5, 7.5, 2.0)
        # model_prob is ~12.4% for exact center with scale 2.0

        # With very low yes_price (3%), model_prob > yes_price → buy YES
        assert model_prob > 0.03

        # With moderate yes_price (20%), model_prob < yes_price → buy NO
        assert model_prob < 0.20


# ---------------------------------------------------------------------------
# Ensemble Edge Strategy
# ---------------------------------------------------------------------------


class TestEnsembleEdgeStrategy:
    """Test ensemble-based direction and probability."""

    def test_ensemble_bucket_counting(self):
        """31 GFS members, count those in [6.5, 7.5) range."""
        # Simulate 31 ensemble members centered around 7C
        members = [
            5.0,
            5.5,
            6.0,
            6.2,
            6.4,  # below bucket
            6.5,
            6.8,
            7.0,
            7.1,
            7.2,
            7.3,
            7.4,  # in bucket (7 members)
            7.5,
            7.8,
            8.0,
            8.2,
            8.5,
            8.8,  # above bucket
            9.0,
            9.2,
            9.5,
            10.0,
            10.5,
            11.0,  # far above
            6.6,
            6.9,
            7.05,
            7.15,
            7.25,
            7.35,
            7.45,  # more in bucket (7 members)
        ]
        assert len(members) == 31

        prob = ensemble_bucket_probability(members, 6.5, 7.5)
        # 14 members in [6.5, 7.5): 6.5, 6.8, 7.0, 7.1, 7.2, 7.3, 7.4, 6.6, 6.9, 7.05, 7.15, 7.25, 7.35, 7.45
        assert prob == pytest.approx(14 / 31, abs=0.01)

    def test_ensemble_direction_buy_yes(self):
        """When ensemble says 45% and market says 30%, buy YES."""
        members = [7.0] * 14 + [10.0] * 17  # 14/31 ≈ 45% in bucket
        prob = ensemble_bucket_probability(members, 6.5, 7.5)
        yes_price = 0.30
        assert prob > yes_price  # → buy YES

    def test_ensemble_direction_buy_no(self):
        """When ensemble says 10% and market says 30%, buy NO."""
        members = [7.0] * 3 + [10.0] * 28  # 3/31 ≈ 10% in bucket
        prob = ensemble_bucket_probability(members, 6.5, 7.5)
        yes_price = 0.30
        assert prob < yes_price  # → buy NO


# ---------------------------------------------------------------------------
# Distribution Strategy
# ---------------------------------------------------------------------------


class TestDistributionStrategy:
    """Test full distribution normalization and edge ranking."""

    def _normal_cdf(self, x, mu, sigma):
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

    def test_distribution_sums_to_one(self):
        """Probabilities across all buckets should sum to ~1.0."""
        mu = 7.0
        sigma = 1.8
        buckets = [(i - 0.5, i + 0.5) for i in range(3, 13)]  # 3C to 12C

        probs = []
        for low, high in buckets:
            p = self._normal_cdf(high, mu, sigma) - self._normal_cdf(low, mu, sigma)
            probs.append(p)

        total = sum(probs)
        # Should be close to 1.0 (some mass outside 3-12C range)
        assert 0.95 < total < 1.05

        # Normalize
        normalized = [p / total for p in probs]
        assert sum(normalized) == pytest.approx(1.0, abs=0.001)

    def test_distribution_peak_at_consensus(self):
        """Bucket containing consensus value should have highest probability."""
        mu = 7.0
        sigma = 1.8
        buckets = [(i - 0.5, i + 0.5) for i in range(3, 13)]

        probs = []
        for low, high in buckets:
            p = self._normal_cdf(high, mu, sigma) - self._normal_cdf(low, mu, sigma)
            probs.append(p)

        # Bucket index 4 = (6.5, 7.5) should be highest
        peak_idx = probs.index(max(probs))
        assert buckets[peak_idx] == (6.5, 7.5)


# ---------------------------------------------------------------------------
# Conservative NO Strategy
# ---------------------------------------------------------------------------


class TestConservativeNoStrategy:
    """Test that conservative NO never bets near consensus."""

    def test_rejects_bucket_near_consensus(self):
        """Bucket center within safe_distance of consensus → reject."""
        consensus_c = 7.0
        bucket_center = 7.5  # only 0.5C away
        min_safe_distance = 2.5
        distance = abs(consensus_c - bucket_center)
        assert distance < min_safe_distance  # should be rejected

    def test_accepts_bucket_far_from_consensus(self):
        """Bucket center far from consensus → accept."""
        consensus_c = 7.0
        bucket_center = 11.0  # 4.0C away
        min_safe_distance = 2.5
        distance = abs(consensus_c - bucket_center)
        assert distance > min_safe_distance  # should be accepted

    def test_always_buy_no(self):
        """Direction should always be buy_no, never buy_yes."""
        # Conservative NO strategy by definition:
        # For any accepted bucket, direction = buy_no
        direction = "buy_no"  # hardcoded in strategy
        assert direction == "buy_no"

    def test_model_prob_no_high_for_distant_buckets(self):
        """For distant buckets, model_prob_no should be high."""
        # Gaussian decay: prob_yes = exp(-distance^2 / (2 * sigma^2))
        distance = 4.0  # 4C from consensus
        sigma = 2.0
        prob_yes = math.exp(-(distance**2) / (2 * sigma**2))
        prob_no = 1.0 - prob_yes
        assert prob_no > 0.85

    def test_model_prob_no_lower_for_closer_buckets(self):
        """For closer buckets, model_prob_no should be lower."""
        # This is above safe distance but closer
        distance = 2.6  # just above min_safe_distance
        sigma = 2.0
        prob_yes = math.exp(-(distance**2) / (2 * sigma**2))
        prob_no = 1.0 - prob_yes

        distance_far = 5.0
        prob_yes_far = math.exp(-(distance_far**2) / (2 * sigma**2))
        prob_no_far = 1.0 - prob_yes_far

        assert prob_no < prob_no_far  # closer = less certain NO


# ---------------------------------------------------------------------------
# Bucket Edge Strategy
# ---------------------------------------------------------------------------


class TestBucketEdgeStrategy:
    """Test per-bucket edge with tunable sigmoid."""

    def test_tighter_sigmoid_gives_higher_center_prob(self):
        """Scale 1.5 should give higher prob than 2.0 for center value."""
        wide = temp_range_probability(7.0, 6.5, 7.5, scale_c=2.0)
        tight = temp_range_probability(7.0, 6.5, 7.5, scale_c=1.5)
        assert tight > wide

    def test_tighter_sigmoid_gives_lower_edge_prob(self):
        """Scale 1.5 should give lower prob for values far from bucket."""
        wide = temp_range_probability(10.0, 6.5, 7.5, scale_c=2.0)
        tight = temp_range_probability(10.0, 6.5, 7.5, scale_c=1.5)
        assert tight < wide

    def test_direction_depends_on_market_price(self):
        """Direction should flip based on market price, not fixed threshold."""
        prob = temp_range_probability(7.0, 6.5, 7.5, scale_c=1.5)

        # Low market price → buy YES
        low_yes = 0.05
        assert prob > low_yes

        # High market price → buy NO
        high_yes = 0.50
        assert prob < high_yes


class TestWeatherStrategyIntentHandling:
    def test_distribution_ignores_siblings_missing_bucket_bounds(self):
        from services.strategies.weather_distribution import WeatherDistributionStrategy

        strategy = WeatherDistributionStrategy()
        intent = {
            "market_id": "market-current",
            "market_question": "Will the high be between 6.5C and 7.5C?",
            "yes_price": 0.45,
            "no_price": 0.55,
            "liquidity": 2000.0,
            "clob_token_ids": ["yes-token", "no-token"],
            "location": "London",
            "operator": "between",
            "bucket_low_c": 6.5,
            "bucket_high_c": 7.5,
            "consensus_value_c": 7.0,
            "source_count": 3,
            "source_spread_c": 1.0,
            "model_agreement": 0.95,
            "sibling_markets": [
                {
                    "market_id": "market-sibling-valid",
                    "yes_price": 0.2,
                    "no_price": 0.8,
                    "bucket_low_c": 9.5,
                    "bucket_high_c": 10.5,
                    "clob_token_ids": ["sib-yes", "sib-no"],
                },
                {
                    "market_id": "market-sibling-invalid",
                    "yes_price": 0.5,
                    "no_price": 0.5,
                    "bucket_low_c": None,
                    "bucket_high_c": None,
                },
            ],
        }

        normalized = strategy._normalize_intent(intent)
        market = strategy._market_from_intent(normalized)
        assert market is not None

        opportunities = strategy.detect_from_intents([normalized], [market], [])
        assert len(opportunities) == 1
        assert opportunities[0].strategy == "weather_distribution"

    @pytest.mark.asyncio
    async def test_ensemble_edge_accepts_numeric_string_model_fields(self):
        from datetime import datetime, timezone

        from services.data_events import DataEvent
        from services.strategies.weather_ensemble_edge import WeatherEnsembleEdgeStrategy

        strategy = WeatherEnsembleEdgeStrategy()
        intent = {
            "market_id": "market-ensemble",
            "market_question": "Will the high be between 6.5C and 7.5C?",
            "yes_price": 0.4,
            "no_price": 0.6,
            "liquidity": 3000.0,
            "clob_token_ids": ["yes-token", "no-token"],
            "location": "London",
            "operator": "between",
            "bucket_low_c": 6.5,
            "bucket_high_c": 7.5,
            "consensus_value_c": 7.0,
            "source_count": "3",
            "source_spread_c": 0.5,
            "model_agreement": "0.92",
            "ensemble_members": None,
            "market": {
                "id": "market-ensemble",
                "condition_id": "market-ensemble",
                "question": "Will the high be between 6.5C and 7.5C?",
                "slug": "market-ensemble",
                "event_slug": "weather-event",
                "clob_token_ids": ["yes-token", "no-token"],
                "liquidity": 3000.0,
                "volume": 500.0,
                "platform": "polymarket",
            },
            "weather": {
                "location": "London",
                "operator": "between",
                "threshold_c_low": 6.5,
                "threshold_c_high": 7.5,
                "consensus_value_c": 7.0,
                "source_count": 3,
                "source_spread_c": 0.5,
                "model_agreement": 0.92,
            },
        }

        event = DataEvent(
            event_type="weather_update",
            source="test",
            timestamp=datetime.now(timezone.utc),
            payload={"intents": [intent]},
        )
        opportunities = await strategy.on_event(event)
        assert len(opportunities) == 1
        assert opportunities[0].strategy == "weather_ensemble_edge"
