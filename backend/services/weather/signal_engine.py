from __future__ import annotations

from dataclasses import dataclass, field

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
    should_trade: bool
    reasons: list[str] = field(default_factory=list)


def build_weather_signal(
    market_id: str,
    yes_price: float,
    no_price: float,
    forecast: WeatherForecastResult,
    entry_max_price: float,
    min_edge_percent: float,
    min_confidence: float,
    min_model_agreement: float,
) -> WeatherSignal:
    gfs = max(0.0, min(1.0, forecast.gfs_probability))
    ecmwf = max(0.0, min(1.0, forecast.ecmwf_probability))

    consensus_yes = (gfs + ecmwf) / 2.0
    agreement = 1.0 - abs(gfs - ecmwf)

    # Confidence blends agreement with separation from 50/50.
    separation = abs(consensus_yes - 0.5) * 2.0
    confidence = max(0.0, min(1.0, (agreement * 0.55) + (separation * 0.45)))

    yes_price = max(0.01, min(0.99, yes_price))
    no_price = max(0.01, min(0.99, no_price))

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
        should_trade=len(reasons) == 0,
        reasons=reasons,
    )
