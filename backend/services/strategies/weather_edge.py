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
from typing import Any, Optional

from services.strategies.base import DecisionCheck, StrategyDecision
from services.strategies.weather_base import BaseWeatherStrategy
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload, weather_metadata
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
    # evaluate()  (unified strategy interface — ported from
    # WeatherConsensusStrategy in trader_orchestrator)
    # ------------------------------------------------------------------

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)
        weather = weather_metadata(payload)

        min_edge = to_float(params.get("min_edge_percent", 6.0), 6.0)
        min_conf = to_confidence(params.get("min_confidence", 0.58), 0.58)
        min_agreement = to_confidence(params.get("min_model_agreement", 0.62), 0.62)
        min_source_count = max(1, int(to_float(params.get("min_source_count", 2), 2)))
        max_source_spread = max(0.0, to_float(params.get("max_source_spread_c", 4.0), 4.0))
        max_entry_price = max(0.05, min(0.98, to_float(params.get("max_entry_price", 0.8), 0.8)))
        base_size = max(1.0, to_float(params.get("base_size_usd", 14.0), 14.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 90.0), 90.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in {"weather"}
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        agreement = to_confidence(
            weather.get("agreement", payload.get("model_agreement", 0.0)),
            0.0,
        )
        source_count = max(0, int(to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, to_float(weather.get("source_spread_c", 0.0), 0.0))

        checks = [
            DecisionCheck("source", "Weather source", source_ok, detail="Requires source=weather."),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "agreement",
                "Model agreement threshold",
                agreement >= min_agreement,
                score=agreement,
                detail=f"min={min_agreement:.2f}",
            ),
            DecisionCheck(
                "source_count",
                "Forecast source depth",
                source_count >= min_source_count,
                score=float(source_count),
                detail=f"min={min_source_count}",
            ),
            DecisionCheck(
                "source_spread",
                "Model spread ceiling (C)",
                source_spread_c <= max_source_spread,
                score=source_spread_c,
                detail=f"max={max_source_spread:.2f}",
            ),
            DecisionCheck(
                "entry_price",
                "Entry price ceiling",
                0.0 < entry_price <= max_entry_price,
                score=entry_price,
                detail=f"max={max_entry_price:.2f}",
            ),
        ]

        score = (
            (edge * 0.6)
            + (confidence * 30.0)
            + (agreement * 12.0)
            + (min(4, source_count) * 1.5)
            - (source_spread_c * 1.2)
        )
        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Weather consensus filters not met",
                score=score,
                checks=checks,
                payload={
                    "source": source,
                    "edge": edge,
                    "confidence": confidence,
                    "agreement": agreement,
                    "source_count": source_count,
                    "source_spread_c": source_spread_c,
                    "entry_price": entry_price,
                },
            )

        spread_scale = max(0.55, 1.0 - min(0.4, source_spread_c / 10.0))
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * (0.8 + agreement) * spread_scale
        size = max(1.0, min(max_size, size))
        return StrategyDecision(
            decision="selected",
            reason="Weather consensus signal selected",
            score=score,
            size_usd=size,
            checks=checks,
            payload={
                "source": source,
                "edge": edge,
                "confidence": confidence,
                "agreement": agreement,
                "source_count": source_count,
                "source_spread_c": source_spread_c,
                "entry_price": entry_price,
                "size_usd": size,
            },
        )

    # ------------------------------------------------------------------
    # Hook: quality_gates
    # ------------------------------------------------------------------

    def quality_gates(self, intent: dict, cfg: dict) -> bool:
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0)
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
        self,
        intent: dict,
        cfg: dict,
    ) -> tuple[Optional[float], dict]:
        bucket_low = intent.get("bucket_low_c")
        bucket_high = intent.get("bucket_high_c")
        consensus_value_c = intent.get("consensus_value_c")

        scale_c = float(cfg.get("probability_scale_c", 2.0))
        if bucket_low is not None and bucket_high is not None and consensus_value_c is not None:
            model_prob = temp_range_probability(
                float(consensus_value_c),
                float(bucket_low),
                float(bucket_high),
                scale_c,
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
        source_spread_c = float(intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0)
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
        source_spread_c = float(intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0)
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
        source_spread_c = float(intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0)

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
