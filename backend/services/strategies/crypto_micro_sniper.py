"""Micro-sniper execution strategy for crypto short-horizon markets.

This strategy is evaluate-only. It consumes crypto worker signals and only
selects entries when the selected-side price sits in a tight, user-defined
window with acceptable spread/liquidity.
"""

from __future__ import annotations

import logging
from typing import Any

from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.data_events import DataEvent
from services.quality_filter import QualityFilterOverrides
from services.trader_orchestrator.strategies.sizing import compute_position_size
from utils.converters import safe_float, to_confidence, to_float
from utils.signal_helpers import selected_probability, signal_payload

logger = logging.getLogger(__name__)


class CryptoMicroSniperStrategy(BaseStrategy):
    strategy_type = "crypto_micro_sniper"
    name = "Crypto Micro Sniper"
    description = "Tight price-window sniper execution for crypto worker signals"
    mispricing_type = "within_market"
    source_key = "crypto"
    worker_affinity = "crypto"
    market_categories = ["crypto"]
    subscriptions = ["crypto_update"]
    accepted_signal_strategy_types = ["btc_eth_highfreq"]
    requires_live_market_context = True

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.5,
        max_resolution_months=0.1,
    )

    default_config = {
        "min_edge_percent": 1.2,
        "min_confidence": 0.42,
        "entry_price_min": 0.47,
        "entry_price_max": 0.53,
        "max_spread_bps": 220.0,
        "min_liquidity_usd": 1200.0,
        "min_order_size_usd": 2.0,
        "base_size_usd": 18.0,
        "max_size_usd": 110.0,
        "sizing_policy": "adaptive",
        "kelly_fractional_scale": 0.40,
        "take_profit_pct": 6.0,
        "stop_loss_pct": 4.0,
        "max_hold_minutes": 12.0,
    }

    async def on_event(self, event: DataEvent) -> list:
        return []

    def _extract_live_selected_price(self, live_market: dict[str, Any], payload: dict[str, Any]) -> float | None:
        for candidate in (
            live_market.get("live_selected_price"),
            live_market.get("selected_price"),
            payload.get("live_selected_price"),
            payload.get("selected_price"),
            (
                payload.get("live_market", {}).get("live_selected_price")
                if isinstance(payload.get("live_market"), dict)
                else None
            ),
            payload.get("entry_price"),
        ):
            parsed = safe_float(candidate, None)
            if parsed is not None and parsed > 0.0:
                return float(parsed)
        return None

    def _extract_live_spread_bps(self, live_market: dict[str, Any], payload: dict[str, Any]) -> float | None:
        for candidate in (
            live_market.get("live_selected_spread_bps"),
            live_market.get("spread_bps"),
            payload.get("live_selected_spread_bps"),
            payload.get("spread_bps"),
        ):
            parsed = safe_float(candidate, None)
            if parsed is not None and parsed >= 0.0:
                return float(parsed)
        return None

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market") if isinstance(payload.get("live_market"), dict) else {}

        min_edge = max(0.0, to_float(params.get("min_edge_percent", 1.2), 1.2))
        min_conf = to_confidence(params.get("min_confidence", 0.42), 0.42)
        entry_price_min = max(0.001, to_float(params.get("entry_price_min", 0.47), 0.47))
        entry_price_max = min(0.999, to_float(params.get("entry_price_max", 0.53), 0.53))
        if entry_price_max < entry_price_min:
            entry_price_max = entry_price_min

        max_spread_bps = max(0.0, to_float(params.get("max_spread_bps", 220.0), 220.0))
        min_liquidity_usd = max(0.0, to_float(params.get("min_liquidity_usd", 1200.0), 1200.0))
        min_order_size_usd = max(0.01, to_float(params.get("min_order_size_usd", 2.0), 2.0))

        base_size = max(min_order_size_usd, to_float(params.get("base_size_usd", 18.0), 18.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 110.0), 110.0))
        sizing_policy = str(params.get("sizing_policy", "adaptive") or "adaptive")
        kelly_fractional_scale = to_float(params.get("kelly_fractional_scale", 0.40), 0.40)

        source = str(getattr(signal, "source", "") or "").strip().lower()
        signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))

        selected_price = self._extract_live_selected_price(live_market, payload)
        spread_bps = self._extract_live_spread_bps(live_market, payload)

        source_ok = source == "crypto"
        origin_ok = bool(payload.get("strategy_origin") == "crypto_worker") or signal_type.startswith("crypto_worker")
        edge_ok = edge >= min_edge
        confidence_ok = confidence >= min_conf
        price_window_ok = selected_price is not None and entry_price_min <= selected_price <= entry_price_max
        spread_ok = spread_bps is None or spread_bps <= max_spread_bps
        liquidity_ok = liquidity >= min_liquidity_usd

        checks = [
            DecisionCheck("source", "Crypto source", source_ok, detail="Requires source=crypto"),
            DecisionCheck(
                "origin", "Crypto worker origin", origin_ok, detail="Requires worker-generated crypto signal"
            ),
            DecisionCheck("edge", "Edge threshold", edge_ok, score=edge, detail=f"min={min_edge:.2f}%"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence_ok,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "entry_window",
                "Entry price window",
                price_window_ok,
                score=selected_price,
                detail=f"target=[{entry_price_min:.3f}, {entry_price_max:.3f}]",
            ),
            DecisionCheck(
                "spread",
                "Spread guard",
                spread_ok,
                score=spread_bps,
                detail=f"max={max_spread_bps:.1f} bps",
            ),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity_ok,
                score=liquidity,
                detail=f"min=${min_liquidity_usd:.0f}",
            ),
        ]

        score = (
            (edge * 0.45)
            + (confidence * 34.0)
            + (min(1.0, liquidity / 20_000.0) * 5.0)
            + (2.5 if price_window_ok else 0.0)
        )

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Micro-sniper filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "selected_price": selected_price,
                    "spread_bps": spread_bps,
                    "liquidity": liquidity,
                    "min_order_size_usd": min_order_size_usd,
                },
            )

        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        probability = selected_probability(signal, payload, direction)
        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry_price <= 0.0 and selected_price is not None:
            entry_price = selected_price

        sizing = compute_position_size(
            base_size_usd=base_size,
            max_size_usd=max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=entry_price if entry_price > 0 else None,
            kelly_fractional_scale=kelly_fractional_scale,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=0.08,
            min_size_usd=min_order_size_usd,
        )

        size_usd = max(min_order_size_usd, float(sizing.get("size_usd") or min_order_size_usd))

        return StrategyDecision(
            decision="selected",
            reason="Micro-sniper signal selected",
            score=score,
            size_usd=size_usd,
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "selected_price": selected_price,
                "spread_bps": spread_bps,
                "liquidity": liquidity,
                "sizing": sizing,
                "min_order_size_usd": min_order_size_usd,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        config = getattr(position, "config", None) or {}
        config = dict(config)

        configured_tp = (getattr(self, "config", None) or {}).get("take_profit_pct", 6.0)
        configured_sl = (getattr(self, "config", None) or {}).get("stop_loss_pct", 4.0)
        configured_hold = (getattr(self, "config", None) or {}).get("max_hold_minutes", 12.0)

        try:
            config.setdefault("take_profit_pct", float(configured_tp))
        except Exception:
            config.setdefault("take_profit_pct", 6.0)
        try:
            config.setdefault("stop_loss_pct", float(configured_sl))
        except Exception:
            config.setdefault("stop_loss_pct", 4.0)
        try:
            config.setdefault("max_hold_minutes", float(configured_hold))
        except Exception:
            config.setdefault("max_hold_minutes", 12.0)

        position.config = config
        return self.default_exit_check(position, market_state)

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
