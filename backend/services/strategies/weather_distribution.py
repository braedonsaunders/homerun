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
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload, weather_metadata, hours_to_target
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

    DEFAULT_CONFIG = {
        "min_edge_percent": 5.0,
        "sigma_c": 1.8,           # std dev for normal distribution (when no ensemble)
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
    # _evaluate_intent  (full distribution-based evaluation pipeline)
    # ------------------------------------------------------------------

    def _evaluate_intent(
        self,
        intent: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[ArbitrageOpportunity]:
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
            all_buckets.append({
                "bucket_low_c": float(sib.get("bucket_low_c", 0)),
                "bucket_high_c": float(sib.get("bucket_high_c", 0)),
                "yes_price": float(sib.get("yes_price", 0.5)),
                "no_price": float(sib.get("no_price", 0.5)),
                "market_id": sib.get("market_id"),
                "clob_token_ids": sib.get("clob_token_ids"),
                "is_current": False,
            })

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

        if roi < cfg["min_edge_percent"] / 2:
            return None

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.05, 400.0)

        if max_position < settings.MIN_POSITION_SIZE:
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
            distribution_snapshot.append({
                "bucket_low_c": bucket["bucket_low_c"],
                "bucket_high_c": bucket["bucket_high_c"],
                "model_prob": round(bucket["model_prob"], 4),
                "yes_price": bucket["yes_price"],
                "edge": round(bucket["edge"], 4),
                "market_id": bucket.get("market_id"),
            })

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

        market_dict = {
            "id": market.id,
            "slug": market.slug,
            "question": market.question,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
        }

        title = f"Distribution: {city} - {question[:40]}"
        description = (
            f"Rank {current_rank}/{total_buckets} bucket | "
            f"Normalized model: {model_prob:.1%} vs market: {yes_price:.0%} | "
            f"{side} at ${entry_price:.2f} (edge: {edge_percent:.1f}%)"
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
        """Weather distribution evaluation with dislocation and source gating."""
        params = context.get("params") or {}
        payload = signal_payload(signal)
        weather = weather_metadata(payload)

        min_edge = to_float(params.get("min_edge_percent", 8.0), 8.0)
        min_conf = to_confidence(params.get("min_confidence", 0.46), 0.46)
        min_temp_dislocation = max(0.0, to_float(params.get("min_temp_dislocation_c", 1.5), 1.5))
        min_source_count = max(1, int(to_float(params.get("min_source_count", 1), 1)))
        max_target_hours = max(1.0, to_float(params.get("max_target_hours", 96.0), 96.0))
        max_source_spread = max(0.0, to_float(params.get("max_source_spread_c", 6.0), 6.0))
        base_size = max(1.0, to_float(params.get("base_size_usd", 12.0), 12.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 80.0), 80.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in {"weather"}
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        source_count = max(0, int(to_float(weather.get("source_count", 0), 0)))
        source_spread_c = max(0.0, to_float(weather.get("source_spread_c", 0.0), 0.0))
        consensus_temp = to_float(weather.get("consensus_temp_c"), 0.0)
        implied_temp = to_float(weather.get("market_implied_temp_c"), 0.0)
        temp_dislocation = abs(consensus_temp - implied_temp)
        htt = hours_to_target(weather.get("target_time"))
        target_window_ok = htt is None or (0.0 <= htt <= max_target_hours)

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

        score = (
            (edge * 0.58)
            + (confidence * 28.0)
            + (temp_dislocation * 2.5)
            + (min(3, source_count) * 1.5)
            - (source_spread_c * 1.1)
        )
        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Weather alerts filters not met",
                score=score,
                checks=checks,
                payload={
                    "source": source,
                    "edge": edge,
                    "confidence": confidence,
                    "source_count": source_count,
                    "source_spread_c": source_spread_c,
                    "consensus_temp_c": consensus_temp,
                    "market_implied_temp_c": implied_temp,
                    "temp_dislocation_c": temp_dislocation,
                    "hours_to_target": htt,
                },
            )

        dislocation_scale = 1.0 + min(0.45, temp_dislocation / 8.0)
        size = base_size * (1.0 + (edge / 100.0)) * (0.7 + confidence) * dislocation_scale
        size = max(1.0, min(max_size, size))
        return StrategyDecision(
            decision="selected",
            reason="Weather alerts signal selected",
            score=score,
            size_usd=size,
            checks=checks,
            payload={
                "source": source,
                "edge": edge,
                "confidence": confidence,
                "source_count": source_count,
                "source_spread_c": source_spread_c,
                "consensus_temp_c": consensus_temp,
                "market_implied_temp_c": implied_temp,
                "temp_dislocation_c": temp_dislocation,
                "hours_to_target": htt,
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
