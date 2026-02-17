from __future__ import annotations

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


def _signal_payload(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


class OpportunityGeneralStrategy(BaseTraderStrategy):
    key = "opportunity_general"
    SOURCES = {"scanner"}

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = _signal_payload(signal)

        min_edge = _to_float(params.get("min_edge_percent", 4.0), 4.0)
        min_conf = _to_confidence(params.get("min_confidence", 0.45), 0.45)
        max_risk = _to_confidence(params.get("max_risk_score", 0.78), 0.78)
        min_liquidity = max(0.0, _to_float(params.get("min_liquidity", 25.0), 25.0))
        min_markets = max(1, int(_to_float(params.get("min_markets", 1), 1)))
        base_size = max(1.0, _to_float(params.get("base_size_usd", 18.0), 18.0))
        max_size = max(base_size, _to_float(params.get("max_size_usd", 150.0), 150.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES

        edge = max(0.0, _to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = _to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, _to_float(getattr(signal, "liquidity", 0.0), 0.0))
        risk_score = _to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])
        position_count = len(payload.get("positions_to_take") or [])
        is_guaranteed = bool(payload.get("is_guaranteed", False))

        checks = [
            DecisionCheck("source", "Scanner source", source_ok, detail="Requires source=scanner."),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk_score",
                "Risk score ceiling",
                risk_score <= max_risk,
                score=risk_score,
                detail=f"max={max_risk:.2f}",
            ),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity >= min_liquidity,
                score=liquidity,
                detail=f"min={min_liquidity:.0f}",
            ),
            DecisionCheck(
                "markets",
                "Market leg coverage",
                market_count >= min_markets and position_count > 0,
                score=float(market_count),
                detail=f"markets={market_count}, positions={position_count}",
            ),
        ]

        score = (
            (edge * 0.55)
            + (confidence * 30.0)
            + (min(1.0, liquidity / 5000.0) * 8.0)
            + (min(4, market_count) * 1.5)
            + (2.0 if is_guaranteed else 0.0)
            - (risk_score * 8.0)
        )

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="General opportunity filters not met",
                score=score,
                checks=checks,
                payload={
                    "source": source,
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "liquidity": liquidity,
                    "market_count": market_count,
                    "position_count": position_count,
                    "is_guaranteed": is_guaranteed,
                },
            )

        market_scale = 1.0 + (min(4, max(0, market_count - 1)) * 0.08)
        risk_scale = max(0.55, 1.0 - (risk_score * 0.35))
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * market_scale * risk_scale
        size = max(1.0, min(max_size, size))

        return StrategyDecision(
            decision="selected",
            reason="General opportunity signal selected",
            score=score,
            size_usd=size,
            checks=checks,
            payload={
                "source": source,
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "liquidity": liquidity,
                "market_count": market_count,
                "position_count": position_count,
                "is_guaranteed": is_guaranteed,
                "size_usd": size,
            },
        )


class OpportunityStructuralStrategy(BaseTraderStrategy):
    key = "opportunity_structural"
    SOURCES = {"scanner"}
    STRUCTURAL_TYPES = {"within_market", "cross_market", "settlement_lag"}

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = _signal_payload(signal)

        min_edge = _to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = _to_confidence(params.get("min_confidence", 0.42), 0.42)
        max_risk = _to_confidence(params.get("max_risk_score", 0.68), 0.68)
        min_markets = max(1, int(_to_float(params.get("min_markets", 2), 2)))
        base_size = max(1.0, _to_float(params.get("base_size_usd", 20.0), 20.0))
        max_size = max(base_size, _to_float(params.get("max_size_usd", 180.0), 180.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES
        edge = max(0.0, _to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = _to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        risk_score = _to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])
        mispricing_type = str(payload.get("mispricing_type", "") or "").strip().lower()
        guaranteed = bool(payload.get("is_guaranteed", False))
        structural_ok = guaranteed or mispricing_type in self.STRUCTURAL_TYPES

        checks = [
            DecisionCheck("source", "Scanner source", source_ok, detail="Requires source=scanner."),
            DecisionCheck(
                "structural",
                "Structural opportunity type",
                structural_ok,
                detail="is_guaranteed or structural mispricing type",
            ),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk_score",
                "Risk score ceiling",
                risk_score <= max_risk,
                score=risk_score,
                detail=f"max={max_risk:.2f}",
            ),
            DecisionCheck(
                "markets",
                "Multi-leg structure",
                market_count >= min_markets,
                score=float(market_count),
                detail=f"min={min_markets}",
            ),
        ]

        score = (edge * 0.65) + (confidence * 35.0) - (risk_score * 10.0) + (min(6, market_count) * 1.2)
        if structural_ok:
            score += 4.0

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Structural opportunity filters not met",
                score=score,
                checks=checks,
                payload={
                    "source": source,
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "market_count": market_count,
                    "mispricing_type": mispricing_type,
                    "is_guaranteed": guaranteed,
                },
            )

        size = base_size * (1.0 + (edge / 120.0)) * (0.8 + confidence) * (1.0 + min(0.45, market_count * 0.06))
        size = max(1.0, min(max_size, size))
        return StrategyDecision(
            decision="selected",
            reason="Structural opportunity selected",
            score=score,
            size_usd=size,
            checks=checks,
            payload={
                "source": source,
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "market_count": market_count,
                "mispricing_type": mispricing_type,
                "is_guaranteed": guaranteed,
                "size_usd": size,
            },
        )
