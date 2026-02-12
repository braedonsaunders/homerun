from __future__ import annotations

from typing import Any

from .base import BaseTraderStrategy, DecisionCheck, StrategyDecision


class Crypto15mStrategy(BaseTraderStrategy):
    key = "crypto_15m"

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        min_edge = float(params.get("min_edge_percent", 3.0) or 3.0)
        min_conf = float(params.get("min_confidence", 0.45) or 0.45)
        base_size = float(params.get("base_size_usd", 25.0) or 25.0)

        source_ok = str(getattr(signal, "source", "")) == "crypto"
        edge = float(getattr(signal, "edge_percent", 0.0) or 0.0)
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)

        checks = [
            DecisionCheck("source", "Crypto source", source_ok, detail="Requires crypto signals."),
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
                reason="Crypto 15m filters not met",
                score=(edge + confidence) / 2.0,
                checks=checks,
                payload={"edge": edge, "confidence": confidence},
            )

        size = max(1.0, base_size * (1.0 + max(0.0, edge - min_edge) / 100.0))
        return StrategyDecision(
            decision="selected",
            reason="Crypto 15m setup validated",
            score=(edge * 0.6) + (confidence * 40.0),
            size_usd=size,
            checks=checks,
            payload={"edge": edge, "confidence": confidence},
        )
