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
from typing import Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from services.strategies.weather_base import BaseWeatherStrategy
from services.weather.signal_engine import (
    compute_confidence,
    ensemble_bucket_probability,
)

logger = logging.getLogger(__name__)


class WeatherConservativeNoStrategy(BaseWeatherStrategy):
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
    # Override detect_from_intents for event_position_counts tracking
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
    # Override _evaluate_intent -- conservative NO flow is distinct
    # enough to warrant a full override (distance filter, always NO,
    # Gaussian decay, different token idx logic, different min_edge).
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
        """
        # ------------------------------------------------------------------
        # 1. Extract forecast and bucket data
        # ------------------------------------------------------------------
        consensus_value_c = intent.get("consensus_value_c")
        bucket_low_c = intent.get("bucket_low_c")
        bucket_high_c = intent.get("bucket_high_c")

        if consensus_value_c is None or bucket_low_c is None or bucket_high_c is None:
            return None

        consensus_value_c = float(consensus_value_c)
        bucket_low_c = float(bucket_low_c)
        bucket_high_c = float(bucket_high_c)

        # ------------------------------------------------------------------
        # 2. Compute distance from consensus to bucket center
        # ------------------------------------------------------------------
        bucket_center = (bucket_low_c + bucket_high_c) / 2.0
        distance = abs(consensus_value_c - bucket_center)

        # ------------------------------------------------------------------
        # 3. Distance filter: too close to forecast = risky for NO
        # ------------------------------------------------------------------
        if distance < cfg["min_safe_distance_c"]:
            return None

        # ------------------------------------------------------------------
        # 4. Quality gates: model agreement, source count, source spread
        # ------------------------------------------------------------------
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(intent.get("source_spread_c") or intent.get("source_spread_c", 0) or 0)
        agreement = float(intent.get("model_agreement", 0))

        if agreement < cfg["min_model_agreement"]:
            return None
        if source_count < cfg["min_source_count"]:
            return None
        if source_spread_c > cfg["max_source_spread_c"]:
            return None

        # ------------------------------------------------------------------
        # 5. Direction is ALWAYS buy_no
        # ------------------------------------------------------------------
        direction = "buy_no"
        no_price = float(intent.get("no_price", 0.5))
        entry_price = no_price

        # ------------------------------------------------------------------
        # 6. Price filter: NO side too expensive
        # ------------------------------------------------------------------
        if entry_price > cfg["max_no_price"]:
            return None

        # ------------------------------------------------------------------
        # 7. Compute model probability of NOT being in this bucket
        # ------------------------------------------------------------------
        ensemble_members = intent.get("ensemble_members")
        if ensemble_members and isinstance(ensemble_members, list) and len(ensemble_members) > 0:
            # Ensemble approach: count fraction outside the bucket
            bucket_prob = ensemble_bucket_probability(
                ensemble_members, bucket_low_c, bucket_high_c
            )
            model_prob_no = 1.0 - bucket_prob
        else:
            # Deterministic approach: Gaussian decay
            # Probability of being in this bucket decays with distance squared
            gaussian_prob = max(0.01, math.exp(-(distance ** 2) / (2 * 2.0 ** 2)))
            model_prob_no = 1.0 - gaussian_prob

        # ------------------------------------------------------------------
        # 8. Edge calculation
        # ------------------------------------------------------------------
        edge = (model_prob_no - no_price) * 100.0

        # Minimum edge threshold (lower than weather_edge since these are
        # high-probability bets with smaller individual returns)
        min_edge = 3.0
        if edge < min_edge:
            return None

        # ------------------------------------------------------------------
        # 9. Look up market and event
        # ------------------------------------------------------------------
        market_id = intent.get("market_id")
        market, event = self._lookup_market(market_id, market_map, event_map)
        if not market:
            return None

        city = intent.get("location", "Unknown")
        question = intent.get("question", market.question or "")

        # ------------------------------------------------------------------
        # 10. Position sizing (conservative)
        # ------------------------------------------------------------------
        token_id = None
        if market.clob_token_ids:
            # NO token is typically index 1
            idx = 1 if len(market.clob_token_ids) > 1 else 0
            token_id = market.clob_token_ids[idx]

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.03, 200.0)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # ------------------------------------------------------------------
        # 11. Profit calculations
        # ------------------------------------------------------------------
        profit = self._compute_profit(entry_price, model_prob_no)
        roi = profit["roi"]

        if roi < 1.0:
            return None

        # ------------------------------------------------------------------
        # 12. Risk scoring
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 13. Build opportunity
        # ------------------------------------------------------------------
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

        market_dict = self._build_market_dict(market)

        title = f"Conservative NO: {city} - {question[:40]}"
        description = (
            f"Bet NO on {bucket_center:.0f}C "
            f"(distance {distance:.1f}C from {consensus_value_c:.1f}C consensus)"
        )

        return self._build_opportunity(
            title=title,
            description=description,
            total_cost=profit["total_cost"],
            expected_payout=profit["expected_payout"],
            gross_profit=profit["gross_profit"],
            fee_amount=profit["fee_amount"],
            net_profit=profit["net_profit"],
            roi=roi,
            risk_score=risk_score,
            risk_factors=risk_factors,
            market_dict=market_dict,
            event=event,
            min_liquidity=min_liquidity,
            max_position=max_position,
            market=market,
            positions=positions,
        )

    # ------------------------------------------------------------------
    # Stubs for abstract hooks (not used since _evaluate_intent is
    # overridden, but required by the base class interface).
    # ------------------------------------------------------------------

    def compute_model_probability(
        self, intent: dict, cfg: dict,
    ) -> tuple[Optional[float], dict]:
        # Not called -- _evaluate_intent is fully overridden.
        return None, {}

    def risk_scoring(
        self,
        cfg: dict,
        intent: dict,
        model_prob: float,
        confidence: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> tuple[float, list[str]]:
        # Not called -- _evaluate_intent is fully overridden.
        return 0.0, []

    def build_metadata(
        self,
        intent: dict,
        cfg: dict,
        model_prob: float,
        direction: str,
        edge_percent: float,
        confidence: float,
        extra_metadata: dict,
    ) -> dict:
        # Not called -- _evaluate_intent is fully overridden.
        return {}

    def build_title_description(
        self,
        city: str,
        question: str,
        intent: dict,
        model_prob: float,
        direction: str,
        side: str,
        entry_price: float,
        yes_price: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> tuple[str, str]:
        # Not called -- _evaluate_intent is fully overridden.
        return "", ""
