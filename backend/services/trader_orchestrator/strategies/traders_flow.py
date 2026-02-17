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


def _payload_dict(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _channel(signal: Any, payload: dict[str, Any]) -> str:
    payload_channel = str(payload.get("traders_channel") or "").strip().lower()
    if payload_channel:
        return payload_channel
    signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
    if signal_type == "confluence":
        return "confluence"
    return signal_type or "unknown"


class TradersFlowStrategy(BaseTraderStrategy):
    key = "traders_flow"

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        source = str(getattr(signal, "source", "") or "").strip().lower()
        payload = _payload_dict(signal)
        channel = _channel(signal, payload)

        min_edge = _to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = _to_confidence(params.get("min_confidence", 0.48), 0.48)
        min_confluence_strength = _to_confidence(
            params.get("min_confluence_strength", 0.55),
            0.55,
        )
        base_size = _to_float(params.get("base_size_usd", 18.0), 18.0)
        max_size = max(1.0, _to_float(params.get("max_size_usd", 120.0), 120.0))

        edge = max(0.0, _to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = _to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        confluence_strength = _to_confidence(
            payload.get("strength", payload.get("conviction_score", 0.0)),
            0.0,
        )

        source_ok = source == "traders"
        channel_ok = channel == "confluence"
        channel_threshold_ok = channel == "confluence" and confluence_strength >= min_confluence_strength

        checks = [
            DecisionCheck(
                "source",
                "Unified traders source",
                source_ok,
                detail="Requires source=traders.",
            ),
            DecisionCheck(
                "channel",
                "Supported traders channel",
                channel_ok,
                detail="Requires confluence channel.",
            ),
            DecisionCheck(
                "channel_threshold",
                "Channel strength threshold",
                channel_threshold_ok,
                score=confluence_strength,
                detail=f"confluence>={min_confluence_strength:.2f}",
            ),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
        ]

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Traders flow filters not met",
                score=(edge * 0.6) + (confidence * 40.0),
                checks=checks,
                payload={
                    "channel": channel,
                    "edge": edge,
                    "confidence": confidence,
                    "confluence_strength": confluence_strength,
                    "min_edge_percent": min_edge,
                    "min_confidence": min_conf,
                    "min_confluence_strength": min_confluence_strength,
                },
            )

        channel_score = confluence_strength
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * (0.9 + channel_score)
        size = max(1.0, min(max_size, size))

        return StrategyDecision(
            decision="selected",
            reason=f"Traders flow {channel} signal selected",
            score=(edge * 0.55) + (confidence * 35.0) + (channel_score * 10.0),
            size_usd=size,
            checks=checks,
            payload={
                "channel": channel,
                "edge": edge,
                "confidence": confidence,
                "confluence_strength": confluence_strength,
                "size_usd": size,
            },
        )
