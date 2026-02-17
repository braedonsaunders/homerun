"""
Weather Edge Strategy

Evaluates weather workflow intent signals to detect actionable mispricings
in temperature-based prediction markets. The weather workflow generates
trade intents by comparing multi-source forecast consensus against current
market prices. This strategy applies configurable thresholds before
converting intents into ArbitrageOpportunity objects.

Unlike scanner strategies that detect structural mispricings, this strategy
detects INFORMATIONAL mispricings based on meteorological forecast data.

Pipeline:
  1. Weather workflow fetches forecasts from multiple sources
  2. Consensus engine computes agreement, spread, and dislocation
  3. Enriched intents are built with raw forecast data
  4. This strategy evaluates direction by comparing model probability
     to market price (NOT using a fixed 0.5 threshold)
"""

from __future__ import annotations

import logging
from typing import Optional

from services.strategies.weather_base import BaseWeatherStrategy
from services.weather.signal_engine import (
    compute_confidence,
    temp_range_probability,
)

logger = logging.getLogger(__name__)


class WeatherEdgeStrategy(BaseWeatherStrategy):
    """
    Weather Edge Strategy

    Detects weather-driven mispricings by evaluating forecast consensus
    against prediction market prices. Compares model probability directly
    to market price for direction (buy YES if underpriced, NO if overpriced).
    """

    strategy_type = "weather_edge"
    name = "Weather Edge"
    description = "Detect weather-driven mispricings via multi-source forecast consensus"

    DEFAULT_CONFIG = {
        "min_edge_percent": 6.0,
        "min_confidence": 0.55,
        "min_model_agreement": 0.60,
        "min_source_count": 2,
        "max_source_spread_c": 4.5,
        "max_entry_price": 0.82,
        "risk_base_score": 0.35,
        "probability_scale_c": 2.0,
    }

    # ------------------------------------------------------------------
    # Hook: quality_gates
    # ------------------------------------------------------------------

    def quality_gates(self, intent: dict, cfg: dict) -> bool:
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(
            intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0
        )
        agreement = float(intent.get("model_agreement", 0))

        if agreement < cfg["min_model_agreement"]:
            return False
        if source_count < cfg["min_source_count"]:
            return False
        if source_spread_c > cfg["max_source_spread_c"]:
            return False
        return True

    # ------------------------------------------------------------------
    # Hook: compute_model_probability
    # ------------------------------------------------------------------

    def compute_model_probability(
        self, intent: dict, cfg: dict,
    ) -> tuple[Optional[float], dict]:
        bucket_low = intent.get("bucket_low_c")
        bucket_high = intent.get("bucket_high_c")
        consensus_value_c = intent.get("consensus_value_c")

        scale_c = float(cfg.get("probability_scale_c", 2.0))
        if bucket_low is not None and bucket_high is not None and consensus_value_c is not None:
            model_prob = temp_range_probability(
                float(consensus_value_c), float(bucket_low), float(bucket_high), scale_c,
            )
        else:
            model_prob = float(intent.get("consensus_probability", 0.5) or 0.5)

        return model_prob, {}

    # ------------------------------------------------------------------
    # Hook: post_direction_gates  (confidence check AFTER direction/edge)
    # ------------------------------------------------------------------

    def post_direction_gates(
        self,
        intent: dict,
        cfg: dict,
        model_prob: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> bool:
        agreement = float(intent.get("model_agreement", 0))
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(
            intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0
        )
        confidence = compute_confidence(agreement, model_prob, source_count, source_spread_c)
        if confidence < cfg["min_confidence"]:
            return False
        return True

    # ------------------------------------------------------------------
    # Hook: risk_scoring
    # ------------------------------------------------------------------

    def risk_scoring(
        self,
        cfg: dict,
        intent: dict,
        model_prob: float,
        confidence: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> tuple[float, list[str]]:
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(
            intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0
        )
        agreement = float(intent.get("model_agreement", 0))

        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Weather-driven directional bet (forecast vs market)",
            f"Model agreement: {agreement:.0%}",
            f"Source spread: {source_spread_c:.1f}C across {source_count} sources",
        ]
        if confidence < 0.65:
            risk_score += 0.15
            risk_factors.append("Moderate confidence")
        if source_spread_c > 3.0:
            risk_score += 0.1
            risk_factors.append("High source disagreement")
        risk_score = min(risk_score, 1.0)
        return risk_score, risk_factors

    # ------------------------------------------------------------------
    # Hook: build_metadata
    # ------------------------------------------------------------------

    def build_metadata(
        self,
        intent: dict,
        cfg: dict,
        model_prob: float,
        direction: str,
        edge_percent: float,
        confidence: float,
        extra_metadata: dict,
    ) -> dict:
        city = intent.get("location", "Unknown")
        consensus_temp = intent.get("consensus_value_c")
        market_temp = intent.get("market_implied_temp_c")
        agreement = float(intent.get("model_agreement", 0))
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(
            intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0
        )

        return {
            "_weather_edge": {
                "city": city,
                "consensus_temp_c": consensus_temp,
                "market_implied_temp_c": market_temp,
                "model_agreement": agreement,
                "source_count": source_count,
                "source_spread_c": source_spread_c,
                "edge_percent": edge_percent,
                "confidence": confidence,
                "direction": direction,
                "model_probability": model_prob,
            },
        }

    # ------------------------------------------------------------------
    # Hook: build_title_description
    # ------------------------------------------------------------------

    def build_title_description(
        self,
        city: str,
        question: str,
        intent: dict,
        model_prob: float,
        direction: str,
        side: str,
        entry_price: float,
        yes_price: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> tuple[str, str]:
        source_count = int(intent.get("source_count", 0))
        agreement = float(intent.get("model_agreement", 0))

        title = f"Weather Edge: {city} - {question[:40]}"
        description = (
            f"Forecast consensus ({source_count} sources, {agreement:.0%} agreement) "
            f"suggests {side} at ${entry_price:.2f} "
            f"(model: {model_prob:.0%} vs market: {yes_price:.0%}, edge: {edge_percent:.1f}%)"
        )
        return title, description
