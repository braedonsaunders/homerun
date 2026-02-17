from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import BaseTraderStrategy, DecisionCheck, StrategyDecision


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_confidence(value: Any, default: float = 0.0) -> float:
    parsed = _to_float(value, default)
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.0, min(1.0, parsed))


def _safe_payload(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _weather_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    market = metadata.get("market")
    if not isinstance(market, dict):
        return {}
    weather = market.get("weather")
    if not isinstance(weather, dict):
        return {}
    return weather


def _hours_to_target(target_time: Any) -> float | None:
    text = str(target_time or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)
        return delta.total_seconds() / 3600.0
    except Exception:
        return None


class WeatherConsensusStrategy(BaseTraderStrategy):
    key = "weather_consensus"
    SOURCES = {"weather"}

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = _safe_payload(signal)
        weather = _weather_metadata(payload)

        min_edge = _to_float(params.get("min_edge_percent", 6.0), 6.0)
        min_conf = _to_confidence(params.get("min_confidence", 0.58), 0.58)
        min_agreement = _to_confidence(params.get("min_model_agreement", 0.62), 0.62)
        min_source_count = max(1, int(_to_float(params.get("min_source_count", 2), 2)))
        max_source_spread = max(0.0, _to_float(params.get("max_source_spread_c", 4.0), 4.0))
        max_entry_price = max(0.05, min(0.98, _to_float(params.get("max_entry_price", 0.8), 0.8)))
        base_size = max(1.0, _to_float(params.get("base_size_usd", 14.0), 14.0))
        max_size = max(base_size, _to_float(params.get("max_size_usd", 90.0), 90.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES
        edge = max(0.0, _to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = _to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        entry_price = _to_float(getattr(signal, "entry_price", 0.0), 0.0)
        agreement = _to_confidence(
            weather.get("agreement", payload.get("model_agreement", 0.0)),
            0.0,
        )
        source_count = max(0, int(_to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, _to_float(weather.get("source_spread_c", 0.0), 0.0))

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


class WeatherAlertsStrategy(BaseTraderStrategy):
    key = "weather_alerts"
    SOURCES = {"weather"}

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = _safe_payload(signal)
        weather = _weather_metadata(payload)

        min_edge = _to_float(params.get("min_edge_percent", 8.0), 8.0)
        min_conf = _to_confidence(params.get("min_confidence", 0.46), 0.46)
        min_temp_dislocation = max(0.0, _to_float(params.get("min_temp_dislocation_c", 1.5), 1.5))
        min_source_count = max(1, int(_to_float(params.get("min_source_count", 1), 1)))
        max_target_hours = max(1.0, _to_float(params.get("max_target_hours", 96.0), 96.0))
        max_source_spread = max(0.0, _to_float(params.get("max_source_spread_c", 6.0), 6.0))
        base_size = max(1.0, _to_float(params.get("base_size_usd", 12.0), 12.0))
        max_size = max(base_size, _to_float(params.get("max_size_usd", 80.0), 80.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES
        edge = max(0.0, _to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = _to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        source_count = max(0, int(_to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, _to_float(weather.get("source_spread_c", 0.0), 0.0))
        consensus_temp = _to_float(weather.get("consensus_temp_c"), 0.0)
        implied_temp = _to_float(weather.get("market_implied_temp_c"), 0.0)
        temp_dislocation = abs(consensus_temp - implied_temp)
        hours_to_target = _hours_to_target(weather.get("target_time"))
        target_window_ok = hours_to_target is None or (0.0 <= hours_to_target <= max_target_hours)

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
                "temp_dislocation",
                "Temperature dislocation (C)",
                temp_dislocation >= min_temp_dislocation,
                score=temp_dislocation,
                detail=f"min={min_temp_dislocation:.2f}",
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
                "target_window",
                "Target window horizon",
                target_window_ok,
                score=hours_to_target,
                detail=f"max={max_target_hours:.0f}h",
            ),
        ]

        score = (
            (edge * 0.58)
            + (confidence * 28.0)
            + (temp_dislocation * 2.5)
            + (min(3, source_count) * 1.5)
            - (source_spread_c * 1.1)
        )
        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Weather alerts filters not met",
                score=score,
                checks=checks,
                payload={
                    "source": source,
                    "edge": edge,
                    "confidence": confidence,
                    "source_count": source_count,
                    "source_spread_c": source_spread_c,
                    "consensus_temp_c": consensus_temp,
                    "market_implied_temp_c": implied_temp,
                    "temp_dislocation_c": temp_dislocation,
                    "hours_to_target": hours_to_target,
                },
            )

        dislocation_scale = 1.0 + min(0.45, temp_dislocation / 8.0)
        size = base_size * (1.0 + (edge / 100.0)) * (0.7 + confidence) * dislocation_scale
        size = max(1.0, min(max_size, size))
        return StrategyDecision(
            decision="selected",
            reason="Weather alerts signal selected",
            score=score,
            size_usd=size,
            checks=checks,
            payload={
                "source": source,
                "edge": edge,
                "confidence": confidence,
                "source_count": source_count,
                "source_spread_c": source_spread_c,
                "consensus_temp_c": consensus_temp,
                "market_implied_temp_c": implied_temp,
                "temp_dislocation_c": temp_dislocation,
                "hours_to_target": hours_to_target,
                "size_usd": size,
            },
        )
