"""
Weather Distribution Strategy

Builds a full probability distribution across ALL temperature buckets in an
event and compares each bucket's model probability to its market price. Uses
cross-bucket awareness via sibling market data so that probabilities are
normalized (sum to 1.0) before computing edge, preventing inflation from
treating each bucket independently.

Pipeline:
  1. Weather workflow provides enriched intents with sibling_markets list
  2. Strategy collects all buckets (current + siblings)
  3. For each bucket, compute model probability:
     a. Ensemble members available -> fraction of members in bucket
     b. Else -> normal CDF over bucket range using consensus + sigma_c
  4. Normalize probabilities to sum to 1.0
  5. Rank buckets by edge (model_prob - yes_price) descending
  6. Return opportunity for the CURRENT market's bucket only
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, ScoringWeights, SizingConfig, ExitDecision
from services.data_events import DataEvent
from services.quality_filter import QualityFilterOverrides
from services.strategy_sdk import StrategySDK
from utils.converters import to_float
from utils.signal_helpers import weather_signal_context, hours_to_target
from services.weather.signal_engine import (
    ensemble_bucket_probability,
    compute_confidence,
    compute_model_agreement,
)

logger = logging.getLogger(__name__)


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Normal distribution CDF using the error function."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


class WeatherDistributionStrategy(BaseStrategy):
    """
    Full distribution comparison: build probability across all buckets,
    buy the most underpriced.

    Instead of evaluating each bucket independently (which inflates edge
    when multiple buckets overlap the consensus), this strategy normalizes
    the model distribution across all sibling markets and only surfaces
    the current bucket's edge relative to that normalized view.
    """

    strategy_type = "weather_distribution"
    name = "Weather Distribution"
    description = "Full distribution comparison: build probability across all buckets, buy the most underpriced"
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
        "sigma_c": 1.8,  # std dev for normal distribution (when no ensemble)
        "min_confidence": 0.50,
        "max_entry_price": 0.85,
        "max_buckets_per_event": 2,  # max simultaneous positions in one event
        "risk_base_score": 0.30,
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
            threshold_raw = (
                normalized.get("threshold_c")
                if normalized.get("threshold_c") is not None
                else weather_payload.get("threshold_c")
            )
            try:
                threshold_c = float(threshold_raw) if threshold_raw is not None else None
            except Exception:
                threshold_c = None
            if threshold_c is not None and (
                normalized.get("bucket_low_c") is None or normalized.get("bucket_high_c") is None
            ):
                operator = str(
                    normalized.get("operator")
                    if normalized.get("operator") is not None
                    else weather_payload.get("operator") or ""
                ).strip().lower()
                span_c = 25.0
                if normalized.get("bucket_low_c") is None and normalized.get("bucket_high_c") is None:
                    if operator in {"gt", "gte", ">", ">="}:
                        normalized["bucket_low_c"] = threshold_c
                        normalized["bucket_high_c"] = threshold_c + span_c
                    else:
                        normalized["bucket_low_c"] = threshold_c - span_c
                        normalized["bucket_high_c"] = threshold_c
                elif normalized.get("bucket_low_c") is None:
                    normalized["bucket_low_c"] = threshold_c - span_c
                elif normalized.get("bucket_high_c") is None:
                    normalized["bucket_high_c"] = threshold_c + span_c

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
    # _evaluate_intent  (full distribution-based evaluation pipeline)
    # ------------------------------------------------------------------

    def _evaluate_intent(
        self,
        intent: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[Opportunity]:
        """Evaluate a single weather trade intent using full distribution normalization.

        Builds model probabilities for ALL buckets (current + siblings),
        normalizes them to sum to 1.0, then computes edge for the current
        market's bucket only.
        """
        yes_price = float(intent.get("yes_price", 0.5))
        no_price = float(intent.get("no_price", 0.5))
        bucket_low = intent.get("bucket_low_c")
        bucket_high = intent.get("bucket_high_c")
        consensus_value_c = intent.get("consensus_value_c")
        ensemble_members = intent.get("ensemble_members")
        market_id = intent.get("market_id")
        sibling_markets = intent.get("sibling_markets") or []

        if bucket_low is None or bucket_high is None or consensus_value_c is None:
            return None

        bucket_low_f = float(bucket_low)
        bucket_high_f = float(bucket_high)
        consensus_f = float(consensus_value_c)
        sigma = float(cfg["sigma_c"])

        # -----------------------------------------------------------
        # 1. Build list of ALL buckets: current market + siblings
        # -----------------------------------------------------------
        current_bucket = {
            "bucket_low_c": bucket_low_f,
            "bucket_high_c": bucket_high_f,
            "yes_price": yes_price,
            "no_price": no_price,
            "market_id": market_id,
            "clob_token_ids": intent.get("clob_token_ids"),
            "is_current": True,
        }

        all_buckets = [current_bucket]
        for sib in sibling_markets:
            low_raw = sib.get("bucket_low_c")
            high_raw = sib.get("bucket_high_c")
            if low_raw is None or high_raw is None:
                continue
            try:
                low = float(low_raw)
                high = float(high_raw)
            except Exception:
                continue
            if high <= low:
                continue
            try:
                sib_yes = float(sib.get("yes_price", 0.5))
            except Exception:
                sib_yes = 0.5
            try:
                sib_no = float(sib.get("no_price", 0.5))
            except Exception:
                sib_no = 0.5
            all_buckets.append(
                {
                    "bucket_low_c": low,
                    "bucket_high_c": high,
                    "yes_price": sib_yes,
                    "no_price": sib_no,
                    "market_id": sib.get("market_id"),
                    "clob_token_ids": sib.get("clob_token_ids"),
                    "is_current": False,
                }
            )

        # -----------------------------------------------------------
        # 2. Compute raw model probability for each bucket
        # -----------------------------------------------------------
        raw_probs = []
        for bucket in all_buckets:
            low = bucket["bucket_low_c"]
            high = bucket["bucket_high_c"]

            if ensemble_members and len(ensemble_members) > 0:
                prob = ensemble_bucket_probability(ensemble_members, low, high)
            else:
                prob = _norm_cdf(high, consensus_f, sigma) - _norm_cdf(low, consensus_f, sigma)
                prob = max(0.0, prob)

            raw_probs.append(prob)

        # -----------------------------------------------------------
        # 3. Normalize probabilities so they sum to 1.0
        # -----------------------------------------------------------
        total_prob = sum(raw_probs)
        if total_prob > 0:
            normalized_probs = [p / total_prob for p in raw_probs]
        else:
            n = len(all_buckets)
            normalized_probs = [1.0 / n for _ in range(n)]

        for i, bucket in enumerate(all_buckets):
            bucket["model_prob"] = normalized_probs[i]
            bucket["edge"] = normalized_probs[i] - bucket["yes_price"]

        # -----------------------------------------------------------
        # 4. Rank buckets by edge (descending)
        # -----------------------------------------------------------
        ranked = sorted(all_buckets, key=lambda b: b["edge"], reverse=True)

        current_rank = None
        current_data = None
        for rank_idx, bucket in enumerate(ranked):
            if bucket.get("is_current"):
                current_rank = rank_idx + 1
                current_data = bucket
                break

        if current_data is None:
            return None

        total_buckets = len(ranked)
        model_prob = current_data["model_prob"]
        edge = current_data["edge"]
        edge_percent = edge * 100.0

        # -----------------------------------------------------------
        # 5. Direction: compare normalized model_prob to yes_price
        # -----------------------------------------------------------
        if model_prob > yes_price:
            direction = "buy_yes"
            entry_price = yes_price
            target_price = model_prob
        else:
            direction = "buy_no"
            entry_price = no_price
            target_price = 1.0 - model_prob
            edge_percent = ((1.0 - model_prob) - no_price) * 100.0

        # -----------------------------------------------------------
        # 6. Apply filters
        # -----------------------------------------------------------
        if edge_percent < cfg["min_edge_percent"]:
            return None
        if entry_price > cfg["max_entry_price"]:
            return None

        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(intent.get("source_spread_c") or 0)
        agreement = float(intent.get("model_agreement", 0))

        if agreement == 0 and ensemble_members:
            agreement = compute_model_agreement({"ensemble": model_prob})

        confidence = compute_confidence(agreement, model_prob, source_count, source_spread_c)
        if confidence < cfg["min_confidence"]:
            return None

        # -----------------------------------------------------------
        # 7. Build opportunity
        # -----------------------------------------------------------
        market = market_map.get(market_id) if market_id else None
        event = event_map.get(market_id) if market_id else None
        if not market:
            return None

        city = intent.get("location", "Unknown")
        question = market.question or ""

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

        # Risk scoring
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Weather distribution: full cross-bucket normalization",
            f"Bucket rank: {current_rank}/{total_buckets}",
            f"Normalized model prob: {model_prob:.1%}",
        ]
        if confidence < 0.65:
            risk_score += 0.15
            risk_factors.append("Moderate confidence")
        if source_spread_c > 3.0:
            risk_score += 0.1
            risk_factors.append("High source disagreement")
        if current_rank and current_rank > cfg["max_buckets_per_event"]:
            risk_score += 0.1
            risk_factors.append(f"Bucket rank {current_rank} exceeds max_buckets_per_event")
        risk_score = min(risk_score, 1.0)

        consensus_temp = intent.get("consensus_value_c")
        market_temp = intent.get("market_implied_temp_c")

        distribution_snapshot = []
        for bucket in ranked:
            distribution_snapshot.append(
                {
                    "bucket_low_c": bucket["bucket_low_c"],
                    "bucket_high_c": bucket["bucket_high_c"],
                    "model_prob": round(bucket["model_prob"], 4),
                    "yes_price": bucket["yes_price"],
                    "edge": round(bucket["edge"], 4),
                    "market_id": bucket.get("market_id"),
                }
            )

        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                "_weather_distribution": {
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
                    "bucket_rank": current_rank,
                    "total_buckets": total_buckets,
                    "distribution": distribution_snapshot,
                    "used_ensemble": bool(ensemble_members),
                },
            }
        ]

        title = f"Distribution: {city} - {question[:40]}"
        description = (
            f"Rank {current_rank}/{total_buckets} bucket | "
            f"Normalized model: {model_prob:.1%} vs market: {yes_price:.0%} | "
            f"{side} at ${entry_price:.2f} (edge: {edge_percent:.1f}%)"
        )

        opp = self.create_opportunity(
            title=title,
            description=description,
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
            opp.strategy_context = {
                "source_key": "weather",
                "strategy_slug": self.strategy_type,
                "weather": {
                    "agreement": agreement,
                    "model_agreement": agreement,
                    "source_count": source_count,
                    "source_spread_c": source_spread_c,
                    "consensus_temp_c": consensus_temp,
                    "market_implied_temp_c": market_temp,
                    "model_probability": model_prob,
                    "edge_percent": edge_percent,
                    "target_time": intent.get("target_time"),
                    "used_ensemble": bool(ensemble_members),
                },
            }
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

    _dist_temp_dislocation: float = 0.0
    _dist_source_count: int = 0
    _dist_source_spread_c: float = 0.0

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        weather = weather_signal_context(signal)

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in {"weather"}
        min_temp_dislocation = max(0.0, to_float(params.get("min_temp_dislocation_c", 1.5), 1.5))
        min_source_count = max(1, int(to_float(params.get("min_source_count", 1), 1)))
        max_target_hours = max(1.0, to_float(params.get("max_target_hours", 96.0), 96.0))
        max_source_spread = max(0.0, to_float(params.get("max_source_spread_c", 6.0), 6.0))

        source_count = max(0, int(to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, to_float(weather.get("source_spread_c", 0.0), 0.0))
        consensus_temp = to_float(weather.get("consensus_temp_c"), 0.0)
        implied_temp = to_float(weather.get("market_implied_temp_c"), 0.0)
        temp_dislocation = abs(consensus_temp - implied_temp)
        htt = hours_to_target(weather.get("target_time"))
        target_window_ok = htt is None or (0.0 <= htt <= max_target_hours)

        self._dist_temp_dislocation = temp_dislocation
        self._dist_source_count = source_count
        self._dist_source_spread_c = source_spread_c

        return [
            DecisionCheck("source", "Weather source", source_ok, detail="Requires source=weather."),
            DecisionCheck(
                "temp_dislocation",
                "Temperature dislocation (C)",
                temp_dislocation >= min_temp_dislocation,
                score=temp_dislocation,
                detail=f"min={min_temp_dislocation:.2f}",
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
                "target_window",
                "Target window horizon",
                target_window_ok,
                score=htt,
                detail=f"max={max_target_hours:.0f}h",
            ),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        return (
            (edge * 0.58)
            + (confidence * 28.0)
            + (self._dist_temp_dislocation * 2.5)
            + (min(3, self._dist_source_count) * 1.5)
            - (self._dist_source_spread_c * 1.1)
        )

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        dislocation_scale = 1.0 + min(0.45, self._dist_temp_dislocation / 8.0)
        size = base_size * (1.0 + (edge / 100.0)) * (0.7 + confidence) * dislocation_scale
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
