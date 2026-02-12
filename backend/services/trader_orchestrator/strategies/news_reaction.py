from __future__ import annotations

from typing import Any

from .base import BaseTraderStrategy, DecisionCheck, StrategyDecision


class NewsReactionStrategy(BaseTraderStrategy):
    key = "news_reaction"

    NEWS_SOURCES = {"news", "insider", "world_intelligence"}

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        min_edge = float(params.get("min_edge_percent", 8.0) or 8.0)
        min_conf = float(params.get("min_confidence", 0.55) or 0.55)
        base_size = float(params.get("base_size_usd", 20.0) or 20.0)

        source = str(getattr(signal, "source", ""))
        source_ok = source in self.NEWS_SOURCES
        edge = float(getattr(signal, "edge_percent", 0.0) or 0.0)
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)

        checks = [
            DecisionCheck("source", "News-capable source", source_ok, detail="news/insider/world_intelligence"),
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
                reason="News reaction filters not met",
                score=(edge + confidence) / 2.0,
                checks=checks,
                payload={"source": source, "edge": edge, "confidence": confidence},
            )

        size = max(1.0, base_size * (1.0 + confidence))
        return StrategyDecision(
            decision="selected",
            reason="News reaction signal selected",
            score=(edge * 0.55) + (confidence * 45.0),
            size_usd=size,
            checks=checks,
            payload={"source": source, "edge": edge, "confidence": confidence},
        )
