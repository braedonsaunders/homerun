"""
Weather Edge Strategy

Evaluates weather workflow intent signals to detect actionable mispricings
in temperature-based prediction markets. The weather workflow generates
trade intents by comparing multi-source forecast consensus against current
market prices. This strategy applies configurable thresholds before
converting intents into ArbitrageOpportunity objects.

Unlike scanner strategies that detect structural mispricings, this strategy
detects INFORMATIONAL mispricings based on meteorological forecast data.

Pipeline:
  1. Weather workflow fetches forecasts from multiple sources
  2. Consensus engine computes agreement, spread, and dislocation
  3. Enriched intents are built with raw forecast data
  4. This strategy evaluates direction by comparing model probability
     to market price (NOT using a fixed 0.5 threshold)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload, weather_metadata
from services.weather.signal_engine import (
    compute_confidence,
    temp_range_probability,
)

logger = logging.getLogger(__name__)


class WeatherEdgeStrategy(BaseStrategy):
    """
    Weather Edge Strategy

    Detects weather-driven mispricings by evaluating forecast consensus
    against prediction market prices. Compares model probability directly
    to market price for direction (buy YES if underpriced, NO if overpriced).
    """

    strategy_type = "weather_edge"
    name = "Weather Edge"
    description = "Detect weather-driven mispricings via multi-source forecast consensus"
    mispricing_type = "news_information"
    source_key = "weather"
    worker_affinity = "weather"
    allow_deduplication = False

    DEFAULT_CONFIG = {
        "min_edge_percent": 6.0,
        "min_confidence": 0.55,
        "min_model_agreement": 0.60,
        "min_source_count": 2,
        "max_source_spread_c": 4.5,
        "max_entry_price": 0.82,
        "risk_base_score": 0.35,
        "probability_scale_c": 2.0,
    }

    # ------------------------------------------------------------------
    # Init / configure
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._config: dict = dict(self.DEFAULT_CONFIG)

    def configure(self, config: dict) -> None:
        """Apply user config overrides from the DB config column."""
        if config:
            for key in self.DEFAULT_CONFIG:
                if key in config:
                    self._config[key] = config[key]

    # ------------------------------------------------------------------
    # detect  (sync – always returns []; weather uses detect_from_intents)
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        return []

    # ------------------------------------------------------------------
    # detect_from_intents  (main entry-point for weather pipeline)
    # ------------------------------------------------------------------

    def detect_from_intents(
        self,
        intents: list[dict],
        markets: list[Market],
        events: list[Event],
    ) -> list[ArbitrageOpportunity]:
        """Convert weather trade intents into ArbitrageOpportunity objects."""
        if not intents:
            return []

        cfg = self._config
        opportunities: list[ArbitrageOpportunity] = []
        market_map = {m.id: m for m in markets}
        event_map: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                event_map[m.id] = event

        for intent in intents:
            try:
                opp = self._evaluate_intent(intent, market_map, event_map, cfg)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug("%s: skipped intent: %s", self.name, e)

        if opportunities:
            logger.info(
                "%s: %d opportunities from %d intents",
                self.name,
                len(opportunities),
                len(intents),
            )
        return opportunities

    # ------------------------------------------------------------------
    # _evaluate_intent  (full evaluation pipeline)
    # ------------------------------------------------------------------

    def _evaluate_intent(
        self,
        intent: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluate a single weather intent for opportunity detection.

        Pipeline:
        1. Quality gates (agreement, source count, spread)
        2. Compute model probability via sigmoid
        3. Choose direction (compare model prob to market price)
        4. Apply edge and price filters
        5. Post-direction confidence check
        6. Market lookup
        7. Position sizing + profit calc
        8. Risk scoring
        9. Build opportunity
        """
        yes_price = float(intent.get("yes_price", 0.5))
        no_price = float(intent.get("no_price", 0.5))

        # --- 1. Quality gates ---
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(
            intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0
        )
        agreement = float(intent.get("model_agreement", 0))

        if agreement < cfg["min_model_agreement"]:
            return None
        if source_count < cfg["min_source_count"]:
            return None
        if source_spread_c > cfg["max_source_spread_c"]:
            return None

        # --- 2. Model probability (sigmoid) ---
        bucket_low = intent.get("bucket_low_c")
        bucket_high = intent.get("bucket_high_c")
        consensus_value_c = intent.get("consensus_value_c")

        scale_c = float(cfg.get("probability_scale_c", 2.0))
        if bucket_low is not None and bucket_high is not None and consensus_value_c is not None:
            model_prob = temp_range_probability(
                float(consensus_value_c), float(bucket_low), float(bucket_high), scale_c,
            )
        else:
            model_prob = float(intent.get("consensus_probability", 0.5) or 0.5)

        # --- 3. Direction: compare model_prob to yes_price ---
        if model_prob > yes_price:
            direction = "buy_yes"
            entry_price = yes_price
            target_price = model_prob
            edge_percent = (model_prob - yes_price) * 100.0
        else:
            direction = "buy_no"
            entry_price = no_price
            target_price = 1.0 - model_prob
            edge_percent = ((1.0 - model_prob) - no_price) * 100.0

        # --- Edge and price filters ---
        if edge_percent < float(cfg.get("min_edge_percent", 0)):
            return None
        max_entry = cfg.get("max_entry_price")
        if max_entry is not None and entry_price > float(max_entry):
            return None

        # --- Post-direction confidence check ---
        confidence = compute_confidence(agreement, model_prob, source_count, source_spread_c)
        if confidence < cfg["min_confidence"]:
            return None

        # --- 4. Market lookup ---
        market_id = intent.get("market_id")
        market = market_map.get(market_id) if market_id else None
        event = event_map.get(market_id) if market_id else None
        if not market:
            return None

        city = intent.get("location", "Unknown")
        question = market.question or ""

        # --- 5. Position sizing + profit calc ---
        side = "YES" if direction == "buy_yes" else "NO"
        token_id = None
        if market.clob_token_ids:
            idx = 0 if direction == "buy_yes" else (
                1 if len(market.clob_token_ids) > 1 else 0
            )
            token_id = market.clob_token_ids[idx]

        expected_payout = target_price
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee_amount = expected_payout * self.fee
        net_profit = gross_profit - fee_amount
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        # --- Min ROI ---
        if roi < float(cfg.get("min_edge_percent", 0)) / 2:
            return None

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.05, 400.0)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # --- 6. Risk scoring ---
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Weather-driven directional bet (forecast vs market)",
            f"Model agreement: {agreement:.0%}",
            f"Source spread: {source_spread_c:.1f}C across {source_count} sources",
        ]
        if confidence < 0.65:
            risk_score += 0.15
            risk_factors.append("Moderate confidence")
        if source_spread_c > 3.0:
            risk_score += 0.1
            risk_factors.append("High source disagreement")
        risk_score = min(risk_score, 1.0)

        # --- 7. Build opportunity ---
        consensus_temp = intent.get("consensus_value_c")
        market_temp = intent.get("market_implied_temp_c")

        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                "_weather_edge": {
                    "city": city,
                    "consensus_temp_c": consensus_temp,
                    "market_implied_temp_c": market_temp,
                    "model_agreement": agreement,
                    "source_count": source_count,
                    "source_spread_c": source_spread_c,
                    "edge_percent": edge_percent,
                    "confidence": confidence,
                    "direction": direction,
                    "model_probability": model_prob,
                },
            }
        ]

        market_dict = {
            "id": market.id,
            "slug": market.slug,
            "question": market.question,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
        }

        title = f"Weather Edge: {city} - {question[:40]}"
        description = (
            f"Forecast consensus ({source_count} sources, {agreement:.0%} agreement) "
            f"suggests {side} at ${entry_price:.2f} "
            f"(model: {model_prob:.0%} vs market: {yes_price:.0%}, edge: {edge_percent:.1f}%)"
        )

        return ArbitrageOpportunity(
            strategy=self.strategy_type,
            title=title,
            description=description,
            total_cost=total_cost,
            expected_payout=expected_payout,
            gross_profit=gross_profit,
            fee=fee_amount,
            net_profit=net_profit,
            roi_percent=roi,
            is_guaranteed=False,
            roi_type="directional_payout",
            risk_score=risk_score,
            risk_factors=risk_factors,
            markets=[market_dict],
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=market.end_date,
            mispricing_type=MispricingType.NEWS_INFORMATION,
            positions_to_take=positions,
        )

    # ------------------------------------------------------------------
    # evaluate()  (unified strategy interface)
    # ------------------------------------------------------------------

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        """Weather consensus evaluation with agreement and source gating."""
        params = context.get("params") or {}
        payload = signal_payload(signal)
        weather = weather_metadata(payload)

        min_edge = to_float(params.get("min_edge_percent", 6.0), 6.0)
        min_conf = to_confidence(params.get("min_confidence", 0.58), 0.58)
        min_agreement = to_confidence(params.get("min_model_agreement", 0.62), 0.62)
        min_source_count = max(1, int(to_float(params.get("min_source_count", 2), 2)))
        max_source_spread = max(0.0, to_float(params.get("max_source_spread_c", 4.0), 4.0))
        max_entry_price = max(0.05, min(0.98, to_float(params.get("max_entry_price", 0.8), 0.8)))
        base_size = max(1.0, to_float(params.get("base_size_usd", 14.0), 14.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 90.0), 90.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in {"weather"}
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        agreement = to_confidence(
            weather.get("agreement", payload.get("model_agreement", 0.0)),
            0.0,
        )
        source_count = max(0, int(to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, to_float(weather.get("source_spread_c", 0.0), 0.0))

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

    # ------------------------------------------------------------------
    # should_exit()  (weather markets resolve at fixed time)
    # ------------------------------------------------------------------

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Weather markets resolve at fixed time — hold to resolution."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        if config.get("resolve_only", True):
            return ExitDecision("hold", "Weather — holding to forecast resolution")
        return self.default_exit_check(position, market_state)
