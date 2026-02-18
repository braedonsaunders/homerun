"""Tail-end carry opportunity filter.

Portable pattern adapted from public tail-end Polymarket bots: filter for
high-probability outcomes close to resolution, then only keep entries with
liquidity/spread quality and non-trivial expected repricing room.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import (
    BaseStrategy,
    DecisionCheck,
    StrategyDecision,
    ExitDecision,
    ScoringWeights,
    SizingConfig,
    make_aware,
    utcnow,
)
import logging

from utils.converters import to_float, to_confidence, clamp
from utils.signal_helpers import signal_payload, days_to_resolution, selected_probability
from utils.converters import safe_float
from services.quality_filter import QualityFilterOverrides

logger = logging.getLogger(__name__)


class TailEndCarryStrategy(BaseStrategy):
    """Find near-resolution high-probability outcomes with executable carry."""

    strategy_type = "tail_end_carry"
    name = "Tail-End Carry"
    description = "Near-expiry high-probability carry with liquidity and spread gates"
    mispricing_type = "within_market"
    requires_resolution_date = True
    subscriptions = ["market_data_refresh"]
    realtime_processing_mode = "incremental"

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=1.0,
    )

    pipeline_defaults = {
        "min_edge_percent": 1.6,
        "min_confidence": 0.35,
        "max_risk_score": 0.78,
        "base_size_usd": 14.0,
        "max_size_usd": 90.0,
    }

    # Composable evaluate pipeline: score = edge*0.55 + conf*28 + entry*6 - risk*9 + time_bonus
    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=28.0,
        risk_penalty=9.0,
    )
    # Sizing uses adaptive policy -- override compute_size in evaluate
    sizing_config = SizingConfig()

    default_config = {
        "min_probability": 0.88,
        "max_probability": 0.98,
        "min_days_to_resolution": 0.15,
        "max_days_to_resolution": 10.0,
        "min_liquidity": 5000.0,
        "max_spread": 0.03,
        "min_repricing_buffer": 0.015,
        "repricing_weight": 0.45,
        "max_opportunities": 35,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    @staticmethod
    def _book_value(payload: Optional[dict], key: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        val = payload.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _extract_side_book(
        self,
        market: Market,
        prices: dict[str, dict],
        outcome: str,
    ) -> tuple[float, Optional[float], Optional[float], Optional[str]]:
        yes = safe_float(getattr(market, "yes_price", 0.0))
        no = safe_float(getattr(market, "no_price", 0.0))
        tokens = list(getattr(market, "clob_token_ids", []) or [])

        if outcome == "YES":
            token_id = tokens[0] if len(tokens) > 0 else None
            payload = prices.get(token_id) if token_id else None
            mid = self._book_value(payload, "mid")
            if mid is None:
                mid = self._book_value(payload, "price")
            price = mid if mid is not None else yes
            bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
            ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
            return price, bid, ask, token_id

        token_id = tokens[1] if len(tokens) > 1 else None
        payload = prices.get(token_id) if token_id else None
        mid = self._book_value(payload, "mid")
        if mid is None:
            mid = self._book_value(payload, "price")
        price = mid if mid is not None else no
        bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
        ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
        return price, bid, ask, token_id

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        min_probability = clamp(safe_float(cfg.get("min_probability"), 0.88), 0.5, 0.99)
        max_probability = clamp(safe_float(cfg.get("max_probability"), 0.98), min_probability + 0.01, 0.995)
        min_days = max(0.0, safe_float(cfg.get("min_days_to_resolution"), 0.15))
        max_days = max(min_days + 0.01, safe_float(cfg.get("max_days_to_resolution"), 10.0))
        min_liquidity = max(100.0, safe_float(cfg.get("min_liquidity"), 5000.0))
        max_spread = clamp(safe_float(cfg.get("max_spread"), 0.03), 0.005, 0.20)
        min_repricing_buffer = clamp(safe_float(cfg.get("min_repricing_buffer"), 0.015), 0.005, 0.10)
        repricing_weight = clamp(safe_float(cfg.get("repricing_weight"), 0.45), 0.10, 0.90)
        max_opportunities = max(1, int(safe_float(cfg.get("max_opportunities"), 35)))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                event_by_market[event_market.id] = event

        now = utcnow().astimezone(timezone.utc)
        candidates: list[tuple[float, Opportunity]] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if safe_float(getattr(market, "liquidity", 0.0)) < min_liquidity:
                continue
            if (
                len(list(getattr(market, "clob_token_ids", []) or [])) < 2
                and len(list(getattr(market, "outcome_prices", []) or [])) < 2
            ):
                continue

            end_date = make_aware(getattr(market, "end_date", None))
            if end_date is None:
                continue

            days_to_resolution = (end_date - now).total_seconds() / 86400.0
            if days_to_resolution < min_days or days_to_resolution > max_days:
                continue

            for outcome in ("YES", "NO"):
                price, bid, ask, token_id = self._extract_side_book(market, prices, outcome)
                if not (min_probability <= price <= max_probability):
                    continue

                spread = 0.0
                if isinstance(bid, float) and isinstance(ask, float) and bid > 0.0 and ask > 0.0:
                    spread = max(0.0, ask - bid)
                    if spread > max_spread:
                        continue

                target_move = max(min_repricing_buffer, (1.0 - price) * repricing_weight)
                target_price = min(0.999, price + target_move)
                if target_price <= (price + 1e-6):
                    continue

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": price,
                        "token_id": token_id,
                        "entry_style": "tail_carry",
                        "_tail_end": {
                            "days_to_resolution": days_to_resolution,
                            "spread": spread,
                            "target_price": target_price,
                            "probability": price,
                        },
                    }
                ]

                opp = self.create_opportunity(
                    title=f"Tail Carry: {outcome} {price:.1%} into resolution",
                    description=(
                        f"{outcome} at {price:.3f} with {days_to_resolution:.1f} days to resolution; "
                        f"target repricing to {target_price:.3f}."
                    ),
                    total_cost=price,
                    expected_payout=target_price,
                    markets=[market],
                    positions=positions,
                    event=event_by_market.get(market.id),
                    is_guaranteed=False,
                    min_liquidity_hard=min_liquidity,
                    min_position_size=max(settings.MIN_POSITION_SIZE, 5.0),
                    skip_fee_model=True,
                    custom_roi_percent=((target_price - price) / price) * 100.0 if price > 0 else 0.0,
                )
                if not opp:
                    continue

                time_score = 1.0 - (days_to_resolution / max_days)
                black_swan_penalty = clamp((price - 0.90) * 0.40, 0.0, 0.12)
                spread_penalty = min(0.14, spread * 2.8)
                risk = 0.56 - (time_score * 0.10) + black_swan_penalty + spread_penalty
                opp.risk_score = clamp(risk, 0.35, 0.82)
                opp.risk_factors = [
                    f"Near-expiry carry ({days_to_resolution:.1f} days)",
                    f"Selected probability {price:.1%}",
                    "Non-zero tail-risk remains until resolution",
                ]
                opp.mispricing_type = MispricingType.WITHIN_MARKET

                strength = (target_price - price) * (1.0 - opp.risk_score)
                candidates.append((strength, opp))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[Opportunity] = []
        seen: set[tuple[str, str]] = set()
        for _, opp in candidates:
            market_id = str((opp.markets or [{}])[0].get("id") or "")
            outcome = str((opp.positions_to_take or [{}])[0].get("outcome") or "")
            key = (market_id, outcome)
            if key in seen:
                continue
            seen.add(key)
            out.append(opp)
            if len(out) >= max_opportunities:
                break
        return out

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        """Tail carry: source, strategy type, entry band, resolution window checks."""
        min_entry = clamp(to_float(params.get("min_entry_price", 0.85), 0.85), 0.01, 0.99)
        max_entry = clamp(to_float(params.get("max_entry_price", 0.985), 0.985), min_entry, 0.999)
        min_days = max(0.0, to_float(params.get("min_days_to_resolution", 0.03), 0.03))
        max_days = max(min_days + 0.01, to_float(params.get("max_days_to_resolution", 7.0), 7.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        strategy_type = str(payload.get("strategy") or payload.get("strategy_type") or "").strip().lower()
        strategy_ok = strategy_type == "tail_end_carry"

        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry_price <= 0.0:
            positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
            if positions:
                entry_price = to_float((positions[0] or {}).get("price"), 0.0)

        dtr = days_to_resolution(payload)
        days_ok = dtr is not None and min_days <= dtr <= max_days

        # Stash for compute_score / evaluate
        payload["_entry_price"] = entry_price
        payload["_dtr"] = dtr
        payload["_max_days"] = max_days
        payload["_strategy_type"] = strategy_type

        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail="Requires source=scanner."),
            DecisionCheck("strategy", "Tail carry strategy type", strategy_ok, detail="strategy=tail_end_carry"),
            DecisionCheck(
                "entry",
                "Entry probability band",
                min_entry <= entry_price <= max_entry,
                score=entry_price,
                detail=f"[{min_entry:.3f}, {max_entry:.3f}]",
            ),
            DecisionCheck(
                "resolution_window",
                "Resolution window",
                days_ok,
                score=dtr,
                detail=f"[{min_days:.2f}, {max_days:.2f}] days",
            ),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        """Tail carry: edge*0.55 + conf*28 + entry*6 - risk*9 + time_bonus."""
        entry_price = float(payload.get("_entry_price", 0) or 0)
        dtr = payload.get("_dtr")
        max_days = float(payload.get("_max_days", 7.0) or 7.0)

        score = (edge * 0.55) + (confidence * 28.0) + (entry_price * 6.0) - (risk_score * 9.0)
        if dtr is not None:
            score += max(0.0, (max_days - dtr) * 0.4)
        return score

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        """Tail carry: composable pipeline with adaptive sizing and custom payload."""
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 1.6), 1.6)
        min_conf = to_confidence(params.get("min_confidence", 0.35), 0.35)
        max_risk = to_confidence(params.get("max_risk_score", 0.78), 0.78)
        base_size = max(1.0, to_float(params.get("base_size_usd", 14.0), 14.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 90.0), 90.0))
        sizing_policy = str(params.get("sizing_policy", "adaptive") or "adaptive")

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])

        # Standard checks
        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk", "Risk ceiling", risk_score <= max_risk, score=risk_score, detail=f"max={max_risk:.2f}"
            ),
        ]

        # Strategy-specific checks (also stashes entry_price/dtr/etc in payload)
        checks.extend(self.custom_checks(signal, context, params, payload))

        score = self.compute_score(edge, confidence, risk_score, market_count, payload)

        entry_price = float(payload.get("_entry_price", 0) or 0)
        dtr = payload.get("_dtr")
        strategy_type = str(payload.get("_strategy_type", "") or "")

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Tail carry filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "entry_price": entry_price,
                    "days_to_resolution": dtr,
                },
            )

        direction = str(getattr(signal, "direction", "") or "")
        probability = selected_probability(signal, payload, direction)

        from services.trader_orchestrator.strategies.sizing import compute_position_size

        sizing = compute_position_size(
            base_size_usd=base_size,
            max_size_usd=max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=entry_price if entry_price > 0 else None,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=0.06,
        )

        return StrategyDecision(
            decision="selected",
            reason="Tail carry signal selected",
            score=score,
            size_usd=float(sizing["size_usd"]),
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "entry_price": entry_price,
                "days_to_resolution": dtr,
                "sizing": sizing,
                "strategy_type": strategy_type,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Near-expiry carry: hold to resolution with tight trailing stop."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        config = dict(config)
        config.setdefault("trailing_stop_pct", 3.0)
        config.setdefault("resolve_only", False)
        position.config = config
        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
