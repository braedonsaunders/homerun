"""
Weather Conservative NO Strategy

Conservative NO-betting strategy for Polymarket temperature markets.
ONLY bets NO on temperature buckets that are FAR from the consensus
forecast. High win rate, lower per-trade profit.

Inspired by the jazzmine-p LSTM approach that achieved 20% weekly returns.
The key insight: when the forecast consensus is far from a bucket range,
the probability of the actual temperature landing in that bucket is very
low, making NO a high-probability bet.

Pipeline:
  1. Weather workflow generates trade intents with forecast consensus data
  2. This strategy filters for buckets far from forecast consensus
  3. ALWAYS buys NO (never YES) -- only on distant buckets
  4. Gaussian decay or ensemble counting for model probability
  5. Conservative position sizing (3% of liquidity, capped at $200)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload, weather_metadata
from services.weather.signal_engine import (
    compute_confidence,
    ensemble_bucket_probability,
)

logger = logging.getLogger(__name__)


class WeatherConservativeNoStrategy(BaseStrategy):
    """
    Conservative NO-betting: bet NO on buckets far from forecast consensus
    for high win rate.

    This strategy exploits the fact that temperature buckets far from the
    forecast consensus have a very low probability of resolving YES. By
    buying NO on these distant buckets, we capture small but highly
    reliable profits. The tradeoff is lower per-trade profit in exchange
    for a significantly higher win rate.
    """

    strategy_type = "weather_conservative_no"
    name = "Weather Conservative NO"
    description = "Conservative NO-betting: bet NO on buckets far from forecast consensus for high win rate"

    DEFAULT_CONFIG = {
        "min_safe_distance_c": 2.5,   # min degrees C away from consensus to bet NO
        "max_no_price": 0.92,         # don't bet NO if it costs more than this
        "min_model_agreement": 0.60,
        "max_source_spread_c": 4.0,
        "min_source_count": 2,
        "max_positions_per_event": 3,
        "risk_base_score": 0.20,      # lower risk since these are high-probability bets
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
    # detect_from_intents  (with event_position_counts tracking)
    # ------------------------------------------------------------------

    def detect_from_intents(
        self,
        intents: list[dict],
        markets: list[Market],
        events: list[Event],
    ) -> list[ArbitrageOpportunity]:
        """Convert weather trade intents into conservative NO opportunities.

        Filters for buckets far from the forecast consensus and builds
        NO-side ArbitrageOpportunity objects.
        """
        if not intents:
            return []

        cfg = self._config
        opportunities: list[ArbitrageOpportunity] = []
        market_map = {m.id: m for m in markets}
        event_map: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                event_map[m.id] = event

        # Track positions per event for max_positions_per_event limit
        event_position_counts: dict[str, int] = {}

        for intent in intents:
            try:
                opp = self._evaluate_intent(intent, market_map, event_map, cfg)
                if opp:
                    # Enforce max positions per event
                    event_id = opp.event_id or "unknown"
                    count = event_position_counts.get(event_id, 0)
                    if count < cfg["max_positions_per_event"]:
                        opportunities.append(opp)
                        event_position_counts[event_id] = count + 1
            except Exception as e:
                logger.debug("Weather Conservative NO: skipped intent: %s", e)

        if opportunities:
            logger.info(
                "Weather Conservative NO: %d opportunities from %d intents",
                len(opportunities),
                len(intents),
            )
        return opportunities

    # ------------------------------------------------------------------
    # _evaluate_intent  (conservative NO flow)
    # ------------------------------------------------------------------

    def _evaluate_intent(
        self,
        intent: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluate a single weather intent for conservative NO betting.

        Only produces NO-side bets on buckets that are far from the
        forecast consensus temperature.

        Pipeline:
        1. Extract forecast and bucket data
        2. Compute distance from consensus to bucket center
        3. Distance filter (too close = risky for NO)
        4. Quality gates (agreement, source count, spread)
        5. Direction is ALWAYS buy_no
        6. Price filter (NO side too expensive)
        7. Model probability of NOT being in this bucket
        8. Edge calculation
        9. Market lookup
        10. Position sizing (conservative)
        11. Profit calculations
        12. Risk scoring
        13. Build opportunity
        """
        # --- 1. Extract forecast and bucket data ---
        consensus_value_c = intent.get("consensus_value_c")
        bucket_low_c = intent.get("bucket_low_c")
        bucket_high_c = intent.get("bucket_high_c")

        if consensus_value_c is None or bucket_low_c is None or bucket_high_c is None:
            return None

        consensus_value_c = float(consensus_value_c)
        bucket_low_c = float(bucket_low_c)
        bucket_high_c = float(bucket_high_c)

        # --- 2. Distance from consensus to bucket center ---
        bucket_center = (bucket_low_c + bucket_high_c) / 2.0
        distance = abs(consensus_value_c - bucket_center)

        # --- 3. Distance filter ---
        if distance < cfg["min_safe_distance_c"]:
            return None

        # --- 4. Quality gates ---
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0)
        agreement = float(intent.get("model_agreement", 0))

        if agreement < cfg["min_model_agreement"]:
            return None
        if source_count < cfg["min_source_count"]:
            return None
        if source_spread_c > cfg["max_source_spread_c"]:
            return None

        # --- 5. Direction is ALWAYS buy_no ---
        direction = "buy_no"
        no_price = float(intent.get("no_price", 0.5))
        entry_price = no_price

        # --- 6. Price filter ---
        if entry_price > cfg["max_no_price"]:
            return None

        # --- 7. Model probability of NOT being in this bucket ---
        ensemble_members = intent.get("ensemble_members")
        if ensemble_members and isinstance(ensemble_members, list) and len(ensemble_members) > 0:
            bucket_prob = ensemble_bucket_probability(
                ensemble_members, bucket_low_c, bucket_high_c
            )
            model_prob_no = 1.0 - bucket_prob
        else:
            # Gaussian decay: probability decays with distance squared
            gaussian_prob = max(0.01, math.exp(-(distance ** 2) / (2 * 2.0 ** 2)))
            model_prob_no = 1.0 - gaussian_prob

        # --- 8. Edge calculation ---
        edge = (model_prob_no - no_price) * 100.0

        min_edge = 3.0
        if edge < min_edge:
            return None

        # --- 9. Market lookup ---
        market_id = intent.get("market_id")
        market = market_map.get(market_id) if market_id else None
        event = event_map.get(market_id) if market_id else None
        if not market:
            return None

        city = intent.get("location", "Unknown")
        question = intent.get("question", market.question or "")

        # --- 10. Position sizing (conservative) ---
        token_id = None
        if market.clob_token_ids:
            idx = 1 if len(market.clob_token_ids) > 1 else 0
            token_id = market.clob_token_ids[idx]

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.03, 200.0)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # --- 11. Profit calculations ---
        expected_payout = model_prob_no
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee_amount = expected_payout * self.fee
        net_profit = gross_profit - fee_amount
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if roi < 1.0:
            return None

        # --- 12. Risk scoring ---
        confidence = compute_confidence(agreement, model_prob_no, source_count, source_spread_c)
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Conservative NO bet on distant temperature bucket",
            f"Distance from consensus: {distance:.1f}C",
            f"Model agreement: {agreement:.0%}",
            f"Source spread: {source_spread_c:.1f}C across {source_count} sources",
        ]
        if confidence < 0.65:
            risk_score += 0.10
            risk_factors.append("Moderate confidence")
        if source_spread_c > 3.0:
            risk_score += 0.05
            risk_factors.append("Elevated source disagreement")
        if distance < 3.5:
            risk_score += 0.10
            risk_factors.append(f"Moderate distance ({distance:.1f}C) from consensus")
        risk_score = min(risk_score, 1.0)

        # --- 13. Build opportunity ---
        consensus_temp = intent.get("consensus_value_c")
        market_temp = intent.get("market_implied_temp_c")

        positions = [
            {
                "action": "BUY",
                "outcome": "NO",
                "price": entry_price,
                "token_id": token_id,
                "_weather_conservative_no": {
                    "city": city,
                    "consensus_temp_c": consensus_temp,
                    "market_implied_temp_c": market_temp,
                    "bucket_low_c": bucket_low_c,
                    "bucket_high_c": bucket_high_c,
                    "bucket_center_c": bucket_center,
                    "distance_from_consensus_c": distance,
                    "model_prob_no": model_prob_no,
                    "model_agreement": agreement,
                    "source_count": source_count,
                    "source_spread_c": source_spread_c,
                    "edge_percent": edge,
                    "confidence": confidence,
                    "direction": direction,
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

        title = f"Conservative NO: {city} - {question[:40]}"
        description = (
            f"Bet NO on {bucket_center:.0f}C "
            f"(distance {distance:.1f}C from {consensus_value_c:.1f}C consensus)"
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

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        agreement = to_confidence(weather.get("agreement", payload.get("model_agreement", 0.0)), 0.0)
        source_count = max(0, int(to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, to_float(weather.get("source_spread_c", 0.0), 0.0))

        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck("confidence", "Confidence threshold", confidence >= min_conf, score=confidence, detail=f"min={min_conf:.2f}"),
            DecisionCheck("agreement", "Model agreement", agreement >= min_agreement, score=agreement, detail=f"min={min_agreement:.2f}"),
            DecisionCheck("source_count", "Forecast source depth", source_count >= min_source_count, score=float(source_count), detail=f"min={min_source_count}"),
            DecisionCheck("source_spread", "Model spread ceiling (C)", source_spread_c <= max_source_spread, score=source_spread_c, detail=f"max={max_source_spread:.2f}"),
            DecisionCheck("entry_price", "Entry price ceiling", 0.0 < entry_price <= max_entry_price, score=entry_price, detail=f"max={max_entry_price:.2f}"),
        ]

        score = (edge * 0.6) + (confidence * 30.0) + (agreement * 12.0) + (min(4, source_count) * 1.5) - (source_spread_c * 1.2)

        if not all(c.passed for c in checks):
            return StrategyDecision("skipped", "Weather filters not met", score=score, checks=checks)

        spread_scale = max(0.55, 1.0 - min(0.4, source_spread_c / 10.0))
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * (0.8 + agreement) * spread_scale
        size = max(1.0, min(max_size, size))

        return StrategyDecision("selected", "Weather signal selected", score=score, size_usd=size, checks=checks)

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
