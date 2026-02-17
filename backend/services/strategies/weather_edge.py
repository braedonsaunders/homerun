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
  3. Intent generator creates WeatherTradeIntent signals
  4. This strategy filters intents and converts to ArbitrageOpportunity
"""

from __future__ import annotations

import logging
from typing import Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class WeatherEdgeStrategy(BaseStrategy):
    """
    Weather Edge Strategy

    Detects weather-driven mispricings by evaluating forecast consensus
    against prediction market prices with configurable quality gates.
    """

    strategy_type = "weather_edge"
    name = "Weather Edge"
    description = "Detect weather-driven mispricings via multi-source forecast consensus"

    # Default config thresholds (overridden by DB config column)
    DEFAULT_CONFIG = {
        "min_edge_percent": 6.0,
        "min_confidence": 0.55,
        "min_model_agreement": 0.60,
        "min_source_count": 2,
        "max_source_spread_c": 4.5,
        "max_entry_price": 0.82,
        "risk_base_score": 0.35,
    }

    def __init__(self):
        super().__init__()
        self._config = dict(self.DEFAULT_CONFIG)

    def configure(self, config: dict) -> None:
        """Apply user config overrides from the DB config column."""
        if config:
            for key in self.DEFAULT_CONFIG:
                if key in config:
                    self._config[key] = config[key]

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Sync detect - returns empty.

        WeatherEdgeStrategy is fed by the weather workflow pipeline,
        not the main scanner loop. Weather intents are converted to
        opportunities by detect_from_intents().
        """
        return []

    def detect_from_intents(
        self,
        intents: list[dict],
        markets: list[Market],
        events: list[Event],
    ) -> list[ArbitrageOpportunity]:
        """Convert weather trade intents into ArbitrageOpportunity objects.

        Each intent dict should contain:
          - market_id: str
          - direction: "buy_yes" | "buy_no"
          - edge_percent: float
          - confidence: float
          - model_agreement: float (0-1)
          - source_count: int
          - source_spread_c: float
          - consensus_temp_c: float
          - market_implied_temp_c: float
          - entry_price: float
          - target_price: float (model-estimated fair value)
          - city: str
          - forecast_summary: str
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

        for intent in intents:
            try:
                opp = self._evaluate_intent(intent, market_map, event_map, cfg)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug("Weather Edge: skipped intent: %s", e)

        if opportunities:
            logger.info("Weather Edge: %d opportunities from %d intents", len(opportunities), len(intents))
        return opportunities

    def _evaluate_intent(
        self,
        intent: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluate a single weather trade intent against config thresholds."""
        edge = float(intent.get("edge_percent", 0))
        confidence = float(intent.get("confidence", 0))
        agreement = float(intent.get("model_agreement", 0))
        source_count = int(intent.get("source_count", 0))
        spread_c = float(intent.get("source_spread_c", 999))
        entry_price = float(intent.get("entry_price", 1))

        # Quality gates
        if edge < cfg["min_edge_percent"]:
            return None
        if confidence < cfg["min_confidence"]:
            return None
        if agreement < cfg["min_model_agreement"]:
            return None
        if source_count < cfg["min_source_count"]:
            return None
        if spread_c > cfg["max_source_spread_c"]:
            return None
        if entry_price > cfg["max_entry_price"]:
            return None

        market_id = intent.get("market_id")
        market = market_map.get(market_id) if market_id else None
        if not market:
            return None

        event = event_map.get(market_id)
        direction = intent.get("direction", "buy_yes")
        target_price = float(intent.get("target_price", entry_price))
        city = intent.get("city", "Unknown")
        forecast = intent.get("forecast_summary", "")

        # Position sizing
        side = "YES" if direction == "buy_yes" else "NO"
        token_id = None
        if market.clob_token_ids:
            idx = 0 if direction == "buy_yes" else (1 if len(market.clob_token_ids) > 1 else 0)
            token_id = market.clob_token_ids[idx]

        expected_payout = target_price
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee_amount = expected_payout * self.fee
        net_profit = gross_profit - fee_amount
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if roi < cfg["min_edge_percent"] / 2:
            return None

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.05, 400.0)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # Risk scoring
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Weather-driven directional bet (forecast vs market)",
            f"Model agreement: {agreement:.0%}",
            f"Source spread: {spread_c:.1f}C across {source_count} sources",
        ]
        if confidence < 0.65:
            risk_score += 0.15
            risk_factors.append("Moderate confidence")
        if spread_c > 3.0:
            risk_score += 0.1
            risk_factors.append("High source disagreement")
        risk_score = min(risk_score, 1.0)

        consensus_temp = intent.get("consensus_temp_c")
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
                    "source_spread_c": spread_c,
                    "edge_percent": edge,
                    "confidence": confidence,
                    "direction": direction,
                    "forecast_summary": forecast[:200],
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

        return ArbitrageOpportunity(
            strategy=self.strategy_type,
            title=f"Weather Edge: {city} - {market.question[:40]}",
            description=(
                f"Forecast consensus ({source_count} sources, {agreement:.0%} agreement) "
                f"suggests {side} at ${entry_price:.2f} "
                f"(edge: {edge:.1f}%, confidence: {confidence:.0%}). {forecast[:100]}"
            ),
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
