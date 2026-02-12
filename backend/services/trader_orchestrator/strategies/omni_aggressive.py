from __future__ import annotations

from typing import Any

from .base import BaseTraderStrategy, DecisionCheck, StrategyDecision


class OmniAggressiveStrategy(BaseTraderStrategy):
    key = "omni_aggressive"

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        aggressiveness = float(params.get("aggressiveness", 1.5) or 1.5)
        min_edge = float(params.get("min_edge_percent", max(1.0, 5.0 / aggressiveness)) or 2.0)
        min_conf = float(params.get("min_confidence", max(0.25, 0.45 / aggressiveness)) or 0.3)
        base_size = float(params.get("base_size_usd", 20.0) or 20.0)
        max_size = float(params.get("max_size_usd", 120.0) or 120.0)

        edge = float(getattr(signal, "edge_percent", 0.0) or 0.0)
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)

        checks = [
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
                reason="Omni aggressive thresholds not met",
                score=(edge + confidence) / 2.0,
                checks=checks,
                payload={"edge": edge, "confidence": confidence, "aggressiveness": aggressiveness},
            )

        raw_size = base_size * aggressiveness * (1.0 + max(0.0, edge) / 100.0)
        size = min(max_size, max(1.0, raw_size))
        return StrategyDecision(
            decision="selected",
            reason="Omni aggressive selected signal",
            score=(edge * 0.65) + (confidence * 35.0),
            size_usd=size,
            checks=checks,
            payload={"edge": edge, "confidence": confidence, "aggressiveness": aggressiveness},
        )
