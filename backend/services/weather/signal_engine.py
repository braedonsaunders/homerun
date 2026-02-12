from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .adapters.base import WeatherForecastResult


@dataclass
class WeatherSignal:
    market_id: str
    direction: str  # buy_yes | buy_no
    market_price: float
    model_probability: float
    edge_percent: float
    confidence: float
    model_agreement: float
    gfs_probability: float
    ecmwf_probability: float
    source_count: int
    source_spread_c: Optional[float]
    consensus_temperature_c: Optional[float]
    market_implied_temperature_c: Optional[float]
    should_trade: bool
    reasons: list[str] = field(default_factory=list)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_weights(source_probs: dict[str, float], raw_weights: dict[str, float]) -> dict[str, float]:
    keys = [k for k in source_probs.keys() if k in raw_weights]
    if not keys:
        count = len(source_probs)
        if count == 0:
            return {}
        return {k: 1.0 / count for k in source_probs.keys()}
    total = sum(max(0.0, float(raw_weights[k])) for k in keys)
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: max(0.0, float(raw_weights[k])) / total for k in keys}


def _logit(p: float) -> float:
    bounded = max(0.001, min(0.999, p))
    return math.log(bounded / (1.0 - bounded))


def _infer_market_temp_c(
    yes_price: float,
    operator: Optional[str],
    threshold_c: Optional[float],
    threshold_c_low: Optional[float],
    threshold_c_high: Optional[float],
) -> Optional[float]:
    op = (operator or "").lower()
    # For one-sided thresholds, invert the same logistic mapping used by the adapter.
    if threshold_c is not None and op in {"gt", "gte", "lt", "lte"}:
        if op in {"gt", "gte"}:
            return threshold_c + (2.0 * _logit(yes_price))
        return threshold_c - (2.0 * _logit(yes_price))

    # For bucket/range contracts, use midpoint proxy when inversion is underdetermined.
    if threshold_c_low is not None and threshold_c_high is not None:
        return (float(threshold_c_low) + float(threshold_c_high)) / 2.0
    return None


def build_weather_signal(
    market_id: str,
    yes_price: float,
    no_price: float,
    forecast: WeatherForecastResult,
    entry_max_price: float,
    min_edge_percent: float,
    min_confidence: float,
    min_model_agreement: float,
    operator: Optional[str] = None,
    threshold_c: Optional[float] = None,
    threshold_c_low: Optional[float] = None,
    threshold_c_high: Optional[float] = None,
) -> WeatherSignal:
    gfs = _clamp01(forecast.gfs_probability)
    ecmwf = _clamp01(forecast.ecmwf_probability)

    meta_probs = forecast.metadata.get("source_probabilities")
    source_probs: dict[str, float] = {}
    if isinstance(meta_probs, dict):
        for k, v in meta_probs.items():
            try:
                source_probs[str(k)] = _clamp01(float(v))
            except Exception:
                continue
    if not source_probs:
        source_probs = {
            "open_meteo:gfs_seamless": gfs,
            "open_meteo:ecmwf_ifs04": ecmwf,
        }

    raw_weights = forecast.metadata.get("source_weights")
    source_weights: dict[str, float] = {}
    if isinstance(raw_weights, dict):
        for k, v in raw_weights.items():
            try:
                source_weights[str(k)] = max(0.0, float(v))
            except Exception:
                continue
    norm_weights = _normalize_weights(source_probs, source_weights)

    source_count = len(source_probs)
    consensus_yes = sum(source_probs[k] * norm_weights.get(k, 0.0) for k in source_probs)
    consensus_yes = _clamp01(consensus_yes)

    probs = list(source_probs.values())
    agreement = 1.0
    if len(probs) >= 2:
        agreement = 1.0 - (max(probs) - min(probs))
    agreement = _clamp01(agreement)

    # Confidence blends agreement with separation from 50/50.
    separation = abs(consensus_yes - 0.5) * 2.0
    source_depth = min(1.0, source_count / 3.0)
    confidence = _clamp01((agreement * 0.45) + (separation * 0.35) + (source_depth * 0.20))

    source_spread_c = None
    if forecast.source_spread_c is not None:
        source_spread_c = max(0.0, float(forecast.source_spread_c))
    elif isinstance(forecast.metadata.get("source_spread_c"), (int, float)):
        source_spread_c = max(0.0, float(forecast.metadata["source_spread_c"]))
    if source_spread_c is not None and source_spread_c > 0:
        # Penalize confidence as source disagreement grows.
        spread_penalty = min(0.35, source_spread_c / 20.0)
        confidence = _clamp01(confidence * (1.0 - spread_penalty))

    yes_price = max(0.01, min(0.99, yes_price))
    no_price = max(0.01, min(0.99, no_price))

    market_implied_temp_c = _infer_market_temp_c(
        yes_price=yes_price,
        operator=operator,
        threshold_c=threshold_c,
        threshold_c_low=threshold_c_low,
        threshold_c_high=threshold_c_high,
    )
    consensus_temp_c = forecast.consensus_value_c
    if consensus_temp_c is None and isinstance(forecast.metadata.get("consensus_value_c"), (int, float)):
        consensus_temp_c = float(forecast.metadata.get("consensus_value_c"))

    if consensus_yes >= 0.5:
        direction = "buy_yes"
        market_price = yes_price
        model_probability = consensus_yes
        edge_percent = (model_probability - market_price) * 100.0
    else:
        direction = "buy_no"
        market_price = no_price
        model_probability = 1.0 - consensus_yes
        edge_percent = (model_probability - market_price) * 100.0

    reasons: list[str] = []
    if market_price > entry_max_price:
        reasons.append(f"entry_price {market_price:.3f} > max {entry_max_price:.3f}")
    if edge_percent < min_edge_percent:
        reasons.append(f"edge {edge_percent:.2f}% < min {min_edge_percent:.2f}%")
    if confidence < min_confidence:
        reasons.append(f"confidence {confidence:.2f} < min {min_confidence:.2f}")
    if agreement < min_model_agreement:
        reasons.append(f"agreement {agreement:.2f} < min {min_model_agreement:.2f}")
    if source_count < 2:
        reasons.append("insufficient source diversity (<2 forecast sources)")

    return WeatherSignal(
        market_id=market_id,
        direction=direction,
        market_price=market_price,
        model_probability=model_probability,
        edge_percent=edge_percent,
        confidence=confidence,
        model_agreement=agreement,
        gfs_probability=gfs,
        ecmwf_probability=ecmwf,
        source_count=source_count,
        source_spread_c=source_spread_c,
        consensus_temperature_c=consensus_temp_c,
        market_implied_temperature_c=market_implied_temp_c,
        should_trade=len(reasons) == 0,
        reasons=reasons,
    )
