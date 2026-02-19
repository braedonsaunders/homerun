"""
Weather Ensemble Edge Strategy

Uses GFS ensemble member data (31 members) to compute per-bucket probabilities
via Monte Carlo counting. Each ensemble member represents a plausible future
atmospheric state; the fraction of members landing in a temperature bucket is
a direct, non-parametric probability estimate.

When ensemble data is unavailable, an optional deterministic fallback uses a
sigmoid-based probability model identical to the standard weather edge strategy.

Pipeline:
  1. Weather workflow fetches GFS ensemble forecasts (31 members)
  2. Intent builder attaches raw ensemble_members array to each intent
  3. This strategy counts the fraction of members in [bucket_low, bucket_high)
  4. Direction is chosen by comparing ensemble probability to market price
     (NOT using a fixed 0.5 threshold)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, ScoringWeights, SizingConfig, ExitDecision
from services.data_events import DataEvent
from services.quality_filter import QualityFilterOverrides
from services.strategy_sdk import StrategySDK
from utils.converters import to_float, to_confidence
from utils.signal_helpers import weather_metadata
from services.weather.signal_engine import (
    compute_confidence,
    ensemble_bucket_probability,
    temp_range_probability,
)

logger = logging.getLogger(__name__)


class WeatherEnsembleEdgeStrategy(BaseStrategy):
    """
    Weather Ensemble Edge Strategy

    Counts the fraction of GFS ensemble members that fall within each
    temperature bucket to derive a non-parametric model probability.
    Compares that probability directly to market price for direction
    (buy YES if model says underpriced, buy NO if overpriced).
    """

    strategy_type = "weather_ensemble_edge"
    name = "Weather Ensemble Edge"
    description = "Ensemble Monte Carlo: count fraction of GFS ensemble members in each temperature bucket"
    mispricing_type = "news_information"
    source_key = "weather"
    worker_affinity = "weather"
    subscriptions = ["weather_update"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
        max_resolution_months=0.5,
    )

    DEFAULT_CONFIG = {
        "min_edge_percent": 5.0,
        "min_ensemble_members": 10,
        "min_ensemble_agreement": 0.15,  # at least 15% of members in bucket
        "max_entry_price": 0.85,
        "risk_base_score": 0.30,
        "deterministic_fallback": True,  # fall back to sigmoid when no ensemble data
        "probability_scale_c": 2.0,  # sigmoid sharpness for fallback
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

    @staticmethod
    def _normalize_intent(intent: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(intent)

        weather_payload = normalized.get("weather")
        if isinstance(weather_payload, dict):
            for key in (
                "source_count",
                "source_spread_c",
                "consensus_probability",
                "consensus_value_c",
                "model_agreement",
                "target_time",
                "location",
                "metric",
                "operator",
                "ensemble_members",
            ):
                if normalized.get(key) is None and weather_payload.get(key) is not None:
                    normalized[key] = weather_payload.get(key)
            if normalized.get("bucket_low_c") is None and weather_payload.get("threshold_c_low") is not None:
                normalized["bucket_low_c"] = weather_payload.get("threshold_c_low")
            if normalized.get("bucket_high_c") is None and weather_payload.get("threshold_c_high") is not None:
                normalized["bucket_high_c"] = weather_payload.get("threshold_c_high")

        market_payload = normalized.get("market")
        if isinstance(market_payload, dict):
            if not str(normalized.get("market_id") or "").strip():
                normalized["market_id"] = market_payload.get("condition_id") or market_payload.get("id")
            if not str(normalized.get("market_slug") or "").strip() and market_payload.get("slug") is not None:
                normalized["market_slug"] = market_payload.get("slug")
            if not str(normalized.get("event_slug") or "").strip() and market_payload.get("event_slug") is not None:
                normalized["event_slug"] = market_payload.get("event_slug")
            if normalized.get("liquidity") is None and market_payload.get("liquidity") is not None:
                normalized["liquidity"] = market_payload.get("liquidity")
            if normalized.get("volume") is None and market_payload.get("volume") is not None:
                normalized["volume"] = market_payload.get("volume")
            if not normalized.get("clob_token_ids") and isinstance(market_payload.get("clob_token_ids"), list):
                normalized["clob_token_ids"] = list(market_payload.get("clob_token_ids") or [])

        direction = str(normalized.get("direction") or "").strip().lower()
        entry_price = normalized.get("entry_price")
        try:
            entry = float(entry_price) if entry_price is not None else None
        except Exception:
            entry = None

        yes_raw = normalized.get("yes_price")
        no_raw = normalized.get("no_price")
        try:
            yes_price = float(yes_raw) if yes_raw is not None else None
        except Exception:
            yes_price = None
        try:
            no_price = float(no_raw) if no_raw is not None else None
        except Exception:
            no_price = None

        if entry is not None:
            if direction == "buy_yes" and yes_price is None:
                yes_price = entry
            if direction == "buy_no" and no_price is None:
                no_price = entry

        if yes_price is None and no_price is not None:
            yes_price = 1.0 - no_price
        if no_price is None and yes_price is not None:
            no_price = 1.0 - yes_price

        if yes_price is None:
            yes_price = 0.5
        if no_price is None:
            no_price = 0.5

        normalized["yes_price"] = max(0.0, min(1.0, float(yes_price)))
        normalized["no_price"] = max(0.0, min(1.0, float(no_price)))
        return normalized

    @staticmethod
    def _market_from_intent(intent: dict[str, Any]) -> Optional[Market]:
        market_payload = intent.get("market")
        if not isinstance(market_payload, dict):
            market_payload = {}

        condition_id = str(
            intent.get("market_id") or market_payload.get("condition_id") or market_payload.get("id") or ""
        ).strip()
        if not condition_id:
            return None

        question = str(intent.get("market_question") or market_payload.get("question") or condition_id).strip()
        slug = str(market_payload.get("slug") or intent.get("market_slug") or condition_id).strip()
        event_slug = str(market_payload.get("event_slug") or intent.get("event_slug") or "").strip()
        platform = str(market_payload.get("platform") or intent.get("platform") or "polymarket").strip() or "polymarket"

        raw_token_ids = market_payload.get("clob_token_ids")
        if not isinstance(raw_token_ids, list):
            raw_token_ids = []
        clob_token_ids = [str(token_id).strip() for token_id in raw_token_ids if str(token_id).strip()]

        try:
            liquidity = float(market_payload.get("liquidity") or intent.get("liquidity") or 0.0)
        except Exception:
            liquidity = 0.0
        try:
            volume = float(market_payload.get("volume") or intent.get("volume") or 0.0)
        except Exception:
            volume = 0.0

        yes_price = float(intent.get("yes_price", 0.5))
        no_price = float(intent.get("no_price", 0.5))
        return Market(
            id=condition_id,
            condition_id=condition_id,
            question=question or condition_id,
            slug=slug or condition_id,
            event_slug=event_slug,
            clob_token_ids=clob_token_ids,
            outcome_prices=[yes_price, no_price],
            liquidity=max(0.0, liquidity),
            volume=max(0.0, volume),
            platform=platform,
        )

    # ------------------------------------------------------------------
    # detect  (sync – always returns []; weather uses detect_from_intents)
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        return []

    # ------------------------------------------------------------------
    # detect_from_intents  (main entry-point for weather pipeline)
    # ------------------------------------------------------------------

    def detect_from_intents(
        self,
        intents: list[dict],
        markets: list[Market],
        events: list[Event],
    ) -> list[Opportunity]:
        """Convert weather trade intents into Opportunity objects."""
        if not intents:
            return []

        cfg = self._config
        opportunities: list[Opportunity] = []
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
    ) -> Optional[Opportunity]:
        """Evaluate a single weather intent using ensemble data.

        Pipeline:
        1. Extract bucket and ensemble data
        2. Compute model probability via ensemble counting (or sigmoid fallback)
        3. Ensemble agreement gate
        4. Choose direction (compare model prob to market price)
        5. Edge and price filters
        6. Market lookup
        7. Position sizing + profit calc
        8. Risk scoring
        9. Build opportunity
        """
        yes_price = float(intent.get("yes_price", 0.5))
        no_price = float(intent.get("no_price", 0.5))

        bucket_low = intent.get("bucket_low_c")
        bucket_high = intent.get("bucket_high_c")
        consensus_value_c = intent.get("consensus_value_c")

        # --- 1. Ensemble data extraction ---
        ensemble_data = intent.get("ensemble_members") or intent.get("ensemble_daily_max")

        model_prob = None
        ensemble_count = 0
        ensemble_fraction = 0.0
        used_ensemble = False

        if ensemble_data and len(ensemble_data) >= int(cfg["min_ensemble_members"]):
            if bucket_low is not None and bucket_high is not None:
                ensemble_fraction = ensemble_bucket_probability(
                    [float(v) for v in ensemble_data],
                    float(bucket_low),
                    float(bucket_high),
                )
                ensemble_count = len(ensemble_data)
                model_prob = ensemble_fraction
                used_ensemble = True

        # --- 2. Deterministic fallback (sigmoid) ---
        if model_prob is None and cfg.get("deterministic_fallback"):
            scale_c = float(cfg.get("probability_scale_c", 2.0))
            if bucket_low is not None and bucket_high is not None and consensus_value_c is not None:
                model_prob = temp_range_probability(
                    float(consensus_value_c),
                    float(bucket_low),
                    float(bucket_high),
                    scale_c,
                )

        if model_prob is None:
            return None

        # --- 3. Ensemble agreement gate ---
        if used_ensemble and ensemble_fraction < float(cfg["min_ensemble_agreement"]):
            return None

        # --- 4. Direction: compare model_prob to yes_price ---
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

        # --- 5. Market lookup ---
        market_id = intent.get("market_id")
        market = market_map.get(market_id) if market_id else None
        event = event_map.get(market_id) if market_id else None
        if not market:
            return None

        city = intent.get("location", "Unknown")
        question = market.question or ""

        # --- 6. Position sizing + profit calc ---
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

        # --- Min ROI ---
        if roi < float(cfg.get("min_edge_percent", 0)) / 2:
            return None

        sizing = StrategySDK.resolve_position_sizing(
            liquidity_usd=market.liquidity,
            liquidity_fraction=0.05,
            hard_cap_usd=400.0,
            signal=intent,
            default_min_size=float(settings.MIN_POSITION_SIZE),
        )
        min_liquidity = float(sizing.get("liquidity_usd", 0.0) or 0.0)
        max_position = float(sizing.get("max_position_size", 0.0) or 0.0)
        if not bool(sizing.get("is_tradeable", False)):
            return None

        # --- 7. Confidence ---
        agreement_proxy = ensemble_fraction if used_ensemble else 0.5
        source_count = 1 if used_ensemble else 0
        source_spread_c = float(intent.get("source_spread_c") or 0)
        confidence = compute_confidence(agreement_proxy, model_prob, source_count, source_spread_c)

        # --- 8. Risk scoring ---
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Weather-driven directional bet (ensemble forecast vs market)",
        ]
        if used_ensemble:
            risk_factors.append(f"Ensemble: {ensemble_count} members, {ensemble_fraction:.0%} in bucket")
        else:
            risk_factors.append("Deterministic fallback (no ensemble data)")
            risk_score += 0.10

        if edge_percent < 8.0:
            risk_score += 0.05
            risk_factors.append("Thin edge")
        actual_entry = yes_price if model_prob > yes_price else no_price
        if actual_entry > 0.75:
            risk_score += 0.05
            risk_factors.append("High entry price")
        risk_score = min(risk_score, 1.0)

        # --- 9. Build opportunity ---
        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                "_ensemble_edge": {
                    "city": city,
                    "used_ensemble": used_ensemble,
                    "ensemble_count": ensemble_count,
                    "ensemble_fraction": ensemble_fraction,
                    "model_probability": model_prob,
                    "bucket_low_c": bucket_low,
                    "bucket_high_c": bucket_high,
                    "consensus_value_c": consensus_value_c,
                    "edge_percent": edge_percent,
                    "confidence": confidence,
                    "direction": direction,
                },
            }
        ]

        title = f"Ensemble Edge: {city} - {question[:40]}"
        if used_ensemble:
            desc = (
                f"Ensemble ({ensemble_count} members, {ensemble_fraction:.0%} in bucket) "
                f"suggests {side} at ${entry_price:.2f} "
                f"(model: {model_prob:.0%} vs market: {yes_price:.0%}, edge: {edge_percent:.1f}%)"
            )
        else:
            desc = (
                f"Sigmoid fallback suggests {side} at ${entry_price:.2f} "
                f"(model: {model_prob:.0%} vs market: {yes_price:.0%}, edge: {edge_percent:.1f}%)"
            )

        opp = self.create_opportunity(
            title=title,
            description=desc,
            total_cost=total_cost,
            expected_payout=expected_payout,
            markets=[market],
            positions=positions,
            event=event,
            is_guaranteed=False,
            custom_roi_percent=roi,
            custom_risk_score=risk_score,
            confidence=confidence,
        )
        if opp is not None:
            opp.risk_factors = risk_factors
            opp.min_liquidity = min_liquidity
            opp.max_position_size = max_position
            opp.mispricing_type = MispricingType.NEWS_INFORMATION
        return opp

    # ------------------------------------------------------------------
    # on_event()  (event-driven detection from weather worker)
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != "weather_update":
            return []
        raw_intents = event.payload.get("intents") or []
        if not raw_intents:
            return []
        intents: list[dict[str, Any]] = []
        markets: list[Market] = []
        for raw in raw_intents:
            if not isinstance(raw, dict):
                continue
            normalized = self._normalize_intent(raw)
            market = self._market_from_intent(normalized)
            if market is None:
                continue
            intents.append(normalized)
            markets.append(market)
        if not intents:
            return []
        return self.detect_from_intents(intents, markets, [])

    # ------------------------------------------------------------------
    # evaluate()  (composable pipeline)
    # ------------------------------------------------------------------

    scoring_weights = ScoringWeights()
    sizing_config = SizingConfig()

    _weather_agreement: float = 0.0
    _weather_source_count: int = 0
    _weather_source_spread_c: float = 0.0

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        weather = weather_metadata(payload)

        min_agreement = to_confidence(params.get("min_model_agreement", 0.62), 0.62)
        min_source_count = max(1, int(to_float(params.get("min_source_count", 2), 2)))
        max_source_spread = max(0.0, to_float(params.get("max_source_spread_c", 4.0), 4.0))
        max_entry_price = max(0.05, min(0.98, to_float(params.get("max_entry_price", 0.8), 0.8)))

        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        agreement = to_confidence(weather.get("agreement", payload.get("model_agreement", 0.0)), 0.0)
        source_count = max(0, int(to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, to_float(weather.get("source_spread_c", 0.0), 0.0))

        self._weather_agreement = agreement
        self._weather_source_count = source_count
        self._weather_source_spread_c = source_spread_c

        return [
            DecisionCheck(
                "agreement",
                "Model agreement",
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

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        return (
            (edge * 0.6)
            + (confidence * 30.0)
            + (self._weather_agreement * 12.0)
            + (min(4, self._weather_source_count) * 1.5)
            - (self._weather_source_spread_c * 1.2)
        )

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        spread_scale = max(0.55, 1.0 - min(0.4, self._weather_source_spread_c / 10.0))
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * (0.8 + self._weather_agreement) * spread_scale
        return max(1.0, min(max_size, size))

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

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
