from __future__ import annotations

from typing import Any

from .base import BaseTraderStrategy, DecisionCheck, StrategyDecision


class OpportunityWeatherStrategy(BaseTraderStrategy):
    key = "opportunity_weather"

    SOURCES = {"scanner", "weather", "news", "world_intelligence"}

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        min_edge = float(params.get("min_edge_percent", 6.0) or 6.0)
        min_conf = float(params.get("min_confidence", 0.5) or 0.5)
        base_size = float(params.get("base_size_usd", 18.0) or 18.0)

        source = str(getattr(signal, "source", ""))
        source_ok = source in self.SOURCES
        edge = float(getattr(signal, "edge_percent", 0.0) or 0.0)
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)

        checks = [
            DecisionCheck("source", "Supported source", source_ok, detail="scanner/weather/news/world_intelligence"),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf}",
            ),
        ]

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Opportunity-weather filters not met",
                score=(edge + confidence) / 2.0,
                checks=checks,
                payload={"source": source, "edge": edge, "confidence": confidence},
            )

        size = max(1.0, base_size * (1.0 + (edge / 100.0)))
        return StrategyDecision(
            decision="selected",
            reason="Opportunity-weather signal selected",
            score=(edge * 0.6) + (confidence * 40.0),
            size_usd=size,
            checks=checks,
            payload={"source": source, "edge": edge, "confidence": confidence},
        )
