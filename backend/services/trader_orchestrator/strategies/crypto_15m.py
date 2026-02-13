from __future__ import annotations

from typing import Any, Callable

from .base import BaseTraderStrategy, DecisionCheck, StrategyDecision


_ALLOWED_MODES = {"auto", "directional", "pure_arb", "rebalance"}
_REGIMES = {"opening", "mid", "closing"}

_EDGE_MODE_FACTORS: dict[str, dict[str, float]] = {
    "opening": {"auto": 1.0, "directional": 1.05, "pure_arb": 0.90, "rebalance": 1.05},
    "mid": {"auto": 1.0, "directional": 1.00, "pure_arb": 0.85, "rebalance": 1.00},
    "closing": {"auto": 0.9, "directional": 0.90, "pure_arb": 0.80, "rebalance": 0.85},
}

_CONF_MODE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "directional": 1.0,
    "pure_arb": 0.9,
    "rebalance": 0.95,
}

_REGIME_CONF_FACTORS: dict[str, float] = {
    "opening": 1.0,
    "mid": 1.0,
    "closing": 0.95,
}

_MODE_SIZE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "directional": 1.0,
    "pure_arb": 0.85,
    "rebalance": 0.9,
}

_REGIME_SIZE_FACTORS: dict[str, float] = {
    "opening": 0.95,
    "mid": 1.0,
    "closing": 1.1,
}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_confidence(value: Any, default: float) -> float:
    parsed = _to_float(value, default)
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.0, min(1.0, parsed))


def _normalize_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in _ALLOWED_MODES:
        return "auto"
    return mode


def _normalize_regime(value: Any) -> str:
    regime = str(value or "mid").strip().lower()
    if regime not in _REGIMES:
        return "mid"
    return regime


def _normalize_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").strip().lower()
    if tf in {"5m", "5min", "5"}:
        return "5m"
    if tf in {"15m", "15min", "15"}:
        return "15m"
    if tf in {"1h", "1hr", "60m", "60min"}:
        return "1h"
    if tf in {"4h", "4hr", "240m", "240min"}:
        return "4h"
    return tf


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",")]
    return []


def _normalize_scope(value: Any, normalizer: Callable[[Any], str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in _as_list(value):
        normalized = normalizer(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _payload_dict(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _get_component_edge(payload: dict[str, Any], direction: str, mode: str) -> float:
    component_edges = payload.get("component_edges")
    if not isinstance(component_edges, dict):
        return 0.0
    side_edges = component_edges.get(direction)
    if not isinstance(side_edges, dict):
        return 0.0
    return max(0.0, _to_float(side_edges.get(mode), 0.0))


def _get_net_edge(payload: dict[str, Any], direction: str, fallback: float) -> float:
    net_edges = payload.get("net_edges")
    if not isinstance(net_edges, dict):
        return fallback
    return _to_float(net_edges.get(direction), fallback)


class Crypto15mStrategy(BaseTraderStrategy):
    key = "crypto_15m"

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        min_edge = _to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = _normalize_confidence(params.get("min_confidence", 0.45), 0.45)
        base_size = _to_float(params.get("base_size_usd", 25.0), 25.0)
        max_size = max(1.0, _to_float(params.get("max_size_usd", base_size * 3.0), base_size * 3.0))

        requested_mode = _normalize_mode(params.get("strategy_mode") or params.get("mode"))
        payload = _payload_dict(signal)
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        regime = _normalize_regime(payload.get("regime"))
        signal_asset = _normalize_asset(
            payload.get("asset") or payload.get("coin") or payload.get("symbol")
        )
        signal_timeframe = _normalize_timeframe(
            payload.get("timeframe") or payload.get("cadence") or payload.get("interval")
        )
        target_assets = _normalize_scope(
            _first_present(
                params.get("target_assets"),
                params.get("allowed_assets"),
                params.get("assets"),
                params.get("coins"),
            ),
            _normalize_asset,
        )
        target_timeframes = _normalize_scope(
            _first_present(
                params.get("target_timeframes"),
                params.get("allowed_timeframes"),
                params.get("timeframes"),
                params.get("cadence"),
            ),
            _normalize_timeframe,
        )
        asset_scope_ok = (not target_assets) or (
            bool(signal_asset) and signal_asset in target_assets
        )
        timeframe_scope_ok = (not target_timeframes) or (
            bool(signal_timeframe) and signal_timeframe in target_timeframes
        )
        dominant_mode = _normalize_mode(payload.get("dominant_strategy"))
        active_mode = dominant_mode if requested_mode == "auto" and dominant_mode != "auto" else requested_mode
        if active_mode == "auto":
            active_mode = "directional"

        source_ok = str(getattr(signal, "source", "")) == "crypto"
        signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
        origin_ok = (
            str(payload.get("strategy_origin") or "").strip().lower() == "crypto_worker"
            or signal_type.startswith("crypto_worker")
        )
        edge = max(0.0, _to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = _normalize_confidence(getattr(signal, "confidence", 0.0), 0.0)
        mode_edge = _get_component_edge(payload, direction, active_mode)
        net_edge = _get_net_edge(payload, direction, edge)

        required_edge = min_edge * _EDGE_MODE_FACTORS.get(regime, {}).get(active_mode, 1.0)
        required_conf = (
            min_conf
            * _CONF_MODE_FACTORS.get(active_mode, 1.0)
            * _REGIME_CONF_FACTORS.get(regime, 1.0)
        )

        edge_for_gate = min(edge, mode_edge) if mode_edge > 0.0 else edge
        checks = [
            DecisionCheck("source", "Crypto source", source_ok, detail="Requires crypto worker signals."),
            DecisionCheck(
                "signal_origin",
                "Dedicated crypto worker signal",
                origin_ok,
                detail="Legacy scanner crypto opportunities are unsupported.",
            ),
            DecisionCheck(
                "asset_scope",
                "Asset target scope",
                asset_scope_ok,
                detail=(
                    f"asset={signal_asset or 'unknown'} targets={','.join(target_assets) or 'all'}"
                ),
            ),
            DecisionCheck(
                "timeframe_scope",
                "Cadence target scope",
                timeframe_scope_ok,
                detail=(
                    f"timeframe={signal_timeframe or 'unknown'} targets={','.join(target_timeframes) or 'all'}"
                ),
            ),
            DecisionCheck(
                "edge",
                "Edge threshold",
                edge_for_gate >= required_edge,
                score=edge_for_gate,
                detail=f"mode={active_mode} regime={regime} min={required_edge:.2f}",
            ),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= required_conf,
                score=confidence,
                detail=f"min={required_conf:.2f}",
            ),
            DecisionCheck(
                "execution_edge",
                "Execution-adjusted edge",
                net_edge > 0.0,
                score=net_edge,
                detail="Requires positive post-penalty edge.",
            ),
        ]

        if requested_mode != "auto":
            checks.append(
                DecisionCheck(
                    "mode_signal",
                    "Requested strategy mode has signal",
                    mode_edge > 0.0,
                    score=mode_edge,
                    detail=f"requested={requested_mode}",
                )
            )

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Crypto worker filters not met",
                score=(edge_for_gate * 0.7) + (confidence * 30.0),
                checks=checks,
                payload={
                    "requested_mode": requested_mode,
                    "active_mode": active_mode,
                    "regime": regime,
                    "edge": edge,
                    "mode_edge": mode_edge,
                    "net_edge": net_edge,
                    "confidence": confidence,
                    "required_edge": required_edge,
                    "required_confidence": required_conf,
                    "asset": signal_asset,
                    "timeframe": signal_timeframe,
                    "target_assets": target_assets,
                    "target_timeframes": target_timeframes,
                },
            )

        edge_boost = 1.0 + max(0.0, edge_for_gate - required_edge) / 30.0
        conf_boost = 0.8 + (confidence * 0.8)
        size = (
            base_size
            * _MODE_SIZE_FACTORS.get(active_mode, 1.0)
            * _REGIME_SIZE_FACTORS.get(regime, 1.0)
            * edge_boost
            * conf_boost
        )
        size = max(1.0, min(max_size, size))

        return StrategyDecision(
            decision="selected",
            reason=f"Crypto {active_mode} setup validated ({regime})",
            score=(edge_for_gate * 0.7) + (confidence * 30.0),
            size_usd=size,
            checks=checks,
            payload={
                "requested_mode": requested_mode,
                "active_mode": active_mode,
                "dominant_mode": dominant_mode,
                "regime": regime,
                "edge": edge,
                "mode_edge": mode_edge,
                "net_edge": net_edge,
                "confidence": confidence,
                "required_edge": required_edge,
                "required_confidence": required_conf,
                "asset": signal_asset,
                "timeframe": signal_timeframe,
                "target_assets": target_assets,
                "target_timeframes": target_timeframes,
            },
        )
