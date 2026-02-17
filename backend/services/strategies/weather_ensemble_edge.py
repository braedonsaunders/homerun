"""
Weather Ensemble Edge Strategy

Uses GFS ensemble member data (31 members) to compute per-bucket probabilities
via Monte Carlo counting. Each ensemble member represents a plausible future
atmospheric state; the fraction of members landing in a temperature bucket is
a direct, non-parametric probability estimate.

When ensemble data is unavailable, an optional deterministic fallback uses a
sigmoid-based probability model identical to the standard weather edge strategy.

Pipeline:
  1. Weather workflow fetches GFS ensemble forecasts (31 members)
  2. Intent builder attaches raw ensemble_members array to each intent
  3. This strategy counts the fraction of members in [bucket_low, bucket_high)
  4. Direction is chosen by comparing ensemble probability to market price
     (NOT using a fixed 0.5 threshold)
"""

from __future__ import annotations

import logging
from typing import Optional

from services.strategies.weather_base import BaseWeatherStrategy
from services.weather.signal_engine import (
    compute_confidence,
    ensemble_bucket_probability,
    temp_range_probability,
)

logger = logging.getLogger(__name__)


class WeatherEnsembleEdgeStrategy(BaseWeatherStrategy):
    """
    Weather Ensemble Edge Strategy

    Counts the fraction of GFS ensemble members that fall within each
    temperature bucket to derive a non-parametric model probability.
    Compares that probability directly to market price for direction
    (buy YES if model says underpriced, buy NO if overpriced).
    """

    strategy_type = "weather_ensemble_edge"
    name = "Weather Ensemble Edge"
    description = "Ensemble Monte Carlo: count fraction of GFS ensemble members in each temperature bucket"

    DEFAULT_CONFIG = {
        "min_edge_percent": 5.0,
        "min_ensemble_members": 10,
        "min_ensemble_agreement": 0.15,  # at least 15% of members in bucket
        "max_entry_price": 0.85,
        "risk_base_score": 0.30,
        "deterministic_fallback": True,  # fall back to sigmoid when no ensemble data
        "probability_scale_c": 2.0,  # sigmoid sharpness for fallback
    }

    # ------------------------------------------------------------------
    # Hook: compute_model_probability
    # ------------------------------------------------------------------

    def compute_model_probability(
        self, intent: dict, cfg: dict,
    ) -> tuple[Optional[float], dict]:
        bucket_low = intent.get("bucket_low_c")
        bucket_high = intent.get("bucket_high_c")
        consensus_value_c = intent.get("consensus_value_c")

        # Ensemble data extraction
        ensemble_data = intent.get("ensemble_members") or intent.get("ensemble_daily_max")

        model_prob = None
        ensemble_count = 0
        ensemble_fraction = 0.0
        used_ensemble = False

        if ensemble_data and len(ensemble_data) >= int(cfg["min_ensemble_members"]):
            if bucket_low is not None and bucket_high is not None:
                ensemble_fraction = ensemble_bucket_probability(
                    [float(v) for v in ensemble_data],
                    float(bucket_low),
                    float(bucket_high),
                )
                ensemble_count = len(ensemble_data)
                model_prob = ensemble_fraction
                used_ensemble = True

        # Deterministic fallback (sigmoid)
        if model_prob is None and cfg.get("deterministic_fallback"):
            scale_c = float(cfg.get("probability_scale_c", 2.0))
            if (
                bucket_low is not None
                and bucket_high is not None
                and consensus_value_c is not None
            ):
                model_prob = temp_range_probability(
                    float(consensus_value_c),
                    float(bucket_low),
                    float(bucket_high),
                    scale_c,
                )

        if model_prob is None:
            return None, {}

        # Ensemble agreement gate
        if used_ensemble and ensemble_fraction < float(cfg["min_ensemble_agreement"]):
            return None, {}

        extra = {
            "used_ensemble": used_ensemble,
            "ensemble_count": ensemble_count,
            "ensemble_fraction": ensemble_fraction,
            "bucket_low_c": bucket_low,
            "bucket_high_c": bucket_high,
            "consensus_value_c": consensus_value_c,
        }
        return model_prob, extra

    # ------------------------------------------------------------------
    # Hook: _compute_confidence_value  (ensemble-aware override)
    # ------------------------------------------------------------------

    def _compute_confidence_value(
        self, intent: dict, model_prob: float, extra_metadata: dict,
    ) -> float:
        used_ensemble = extra_metadata.get("used_ensemble", False)
        ensemble_fraction = extra_metadata.get("ensemble_fraction", 0.0)
        agreement_proxy = ensemble_fraction if used_ensemble else 0.5
        source_count = 1 if used_ensemble else 0
        source_spread_c = float(intent.get("source_spread_c") or 0)
        return compute_confidence(agreement_proxy, model_prob, source_count, source_spread_c)

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
        used_ensemble = extra_metadata.get("used_ensemble", False)
        ensemble_count = extra_metadata.get("ensemble_count", 0)
        ensemble_fraction = extra_metadata.get("ensemble_fraction", 0.0)
        entry_price = float(intent.get("yes_price", 0.5))
        # Reconstruct entry_price from direction context
        if model_prob <= entry_price:
            entry_price = float(intent.get("no_price", 0.5))

        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Weather-driven directional bet (ensemble forecast vs market)",
        ]
        if used_ensemble:
            risk_factors.append(
                f"Ensemble: {ensemble_count} members, {ensemble_fraction:.0%} in bucket"
            )
        else:
            risk_factors.append("Deterministic fallback (no ensemble data)")
            risk_score += 0.10

        if edge_percent < 8.0:
            risk_score += 0.05
            risk_factors.append("Thin edge")
        # For the entry_price check, we need the actual entry_price from compute_direction
        # We approximate via the intent prices; this matches the original logic
        yes_price = float(intent.get("yes_price", 0.5))
        no_price = float(intent.get("no_price", 0.5))
        actual_entry = yes_price if model_prob > yes_price else no_price
        if actual_entry > 0.75:
            risk_score += 0.05
            risk_factors.append("High entry price")
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
        return {
            "_ensemble_edge": {
                "city": city,
                "used_ensemble": extra_metadata.get("used_ensemble", False),
                "ensemble_count": extra_metadata.get("ensemble_count", 0),
                "ensemble_fraction": extra_metadata.get("ensemble_fraction", 0.0),
                "model_probability": model_prob,
                "bucket_low_c": extra_metadata.get("bucket_low_c"),
                "bucket_high_c": extra_metadata.get("bucket_high_c"),
                "consensus_value_c": extra_metadata.get("consensus_value_c"),
                "edge_percent": edge_percent,
                "confidence": confidence,
                "direction": direction,
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
        used_ensemble = extra_metadata.get("used_ensemble", False)
        ensemble_count = extra_metadata.get("ensemble_count", 0)
        ensemble_fraction = extra_metadata.get("ensemble_fraction", 0.0)

        title = f"Ensemble Edge: {city} - {question[:40]}"
        if used_ensemble:
            desc = (
                f"Ensemble ({ensemble_count} members, {ensemble_fraction:.0%} in bucket) "
                f"suggests {side} at ${entry_price:.2f} "
                f"(model: {model_prob:.0%} vs market: {yes_price:.0%}, edge: {edge_percent:.1f}%)"
            )
        else:
            desc = (
                f"Sigmoid fallback suggests {side} at ${entry_price:.2f} "
                f"(model: {model_prob:.0%} vs market: {yes_price:.0%}, edge: {edge_percent:.1f}%)"
            )
        return title, desc
