"""
Base Weather Strategy

Extracts the shared boilerplate from all weather strategies into a single
base class.  Each concrete weather strategy only needs to define:

  - DEFAULT_CONFIG, strategy_type, name, description
  - Hook methods that customise probability computation, quality gates,
    direction logic, risk scoring, metadata, and title/description.

Strategies whose evaluation flow is *fundamentally* different (e.g.
weather_distribution with its cross-bucket normalisation) can override
``_evaluate_intent`` entirely and call the protected helper methods
(``_lookup_market``, ``_compute_profit``, ``_size_position``,
``_build_opportunity``) directly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, ScoringWeights, SizingConfig, ExitDecision
from services.quality_filter import QualityFilterOverrides
from utils.converters import to_float, to_confidence
from utils.signal_helpers import weather_signal_context
from services.weather.signal_engine import (
    compute_confidence,
)

logger = logging.getLogger(__name__)


class BaseWeatherStrategy(BaseStrategy):
    """Common base for all weather-driven strategies.

    Provides the shared ``__init__``/``configure`` pattern, a default
    ``detect_from_intents`` loop, a standard ``_evaluate_intent``
    scaffold, and a set of protected helpers that subclasses (especially
    those that override ``_evaluate_intent``) can reuse.

    Hook methods (override in subclasses):
        compute_model_probability   - model probability from intent data
        quality_gates               - pre-direction filters
        compute_direction           - pick YES/NO, prices, edge
        position_sizing_params      - (liquidity_fraction, max_cap)
        min_roi_threshold           - minimum ROI to proceed
        risk_scoring                - risk_score + risk_factors
        build_metadata              - position metadata dict
        build_title_description     - (title, description) strings
    """

    # Subclasses MUST define these.
    DEFAULT_CONFIG: dict = {}

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
        max_resolution_months=0.5,
    )

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
    # detect  (sync – always returns [])
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
    # _evaluate_intent  (standard scaffold – override for exotic flows)
    # ------------------------------------------------------------------

    def _evaluate_intent(
        self,
        intent: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[Opportunity]:
        """Standard evaluation scaffold.

        1. quality_gates
        2. compute_model_probability
        3. compute_direction  (edge/price filters)
        4. market lookup
        5. position sizing + profit calc
        6. min ROI check
        7. risk scoring
        8. build metadata + opportunity
        """
        yes_price = float(intent.get("yes_price", 0.5))
        no_price = float(intent.get("no_price", 0.5))

        # --- 1. Quality gates ---
        if not self.quality_gates(intent, cfg):
            return None

        # --- 2. Model probability ---
        model_prob, extra_metadata = self.compute_model_probability(intent, cfg)
        if model_prob is None:
            return None

        # --- 3. Direction ---
        direction_result = self.compute_direction(
            model_prob,
            yes_price,
            no_price,
            intent,
            cfg,
            extra_metadata,
        )
        if direction_result is None:
            return None

        direction, entry_price, target_price, edge_percent = direction_result

        # --- Edge and price filters ---
        if edge_percent < float(cfg.get("min_edge_percent", 0)):
            return None
        max_entry = cfg.get("max_entry_price")
        if max_entry is not None and entry_price > float(max_entry):
            return None

        # --- Post-direction gates (e.g. confidence check) ---
        if not self.post_direction_gates(
            intent,
            cfg,
            model_prob,
            edge_percent,
            extra_metadata,
        ):
            return None

        # --- 4. Market lookup ---
        market_id = intent.get("market_id")
        market, event = self._lookup_market(market_id, market_map, event_map)
        if not market:
            return None

        city = intent.get("location", "Unknown")
        question = market.question or ""

        # --- 5. Position sizing + profit calc ---
        side = "YES" if direction == "buy_yes" else "NO"
        token_id = self._resolve_token_id(market, direction)

        profit = self._compute_profit(entry_price, target_price)
        roi = profit["roi"]

        # --- 6. Min ROI ---
        if roi < self.min_roi_threshold(cfg):
            return None

        liq_frac, max_cap = self.position_sizing_params(cfg)
        min_liquidity = market.liquidity
        max_position = min(min_liquidity * liq_frac, max_cap)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # --- 7. Risk scoring ---
        confidence = self._compute_confidence_value(intent, model_prob, extra_metadata)
        risk_score, risk_factors = self.risk_scoring(
            cfg,
            intent,
            model_prob,
            confidence,
            edge_percent,
            extra_metadata,
        )

        # --- 8. Build metadata + opportunity ---
        position_meta = self.build_metadata(
            intent,
            cfg,
            model_prob,
            direction,
            edge_percent,
            confidence,
            extra_metadata,
        )

        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                **position_meta,
            }
        ]

        market_dict = self._build_market_dict(market)

        title, description = self.build_title_description(
            city,
            question,
            intent,
            model_prob,
            direction,
            side,
            entry_price,
            yes_price,
            edge_percent,
            extra_metadata,
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

    # ==================================================================
    # HOOK METHODS  (override in subclasses)
    # ==================================================================

    def compute_model_probability(
        self,
        intent: dict,
        cfg: dict,
    ) -> tuple[Optional[float], dict]:
        """Return (model_prob, extra_metadata).

        Must be overridden by every subclass.
        """
        raise NotImplementedError

    def quality_gates(self, intent: dict, cfg: dict) -> bool:
        """Pre-direction filters.  Return True to proceed, False to skip."""
        return True

    def post_direction_gates(
        self,
        intent: dict,
        cfg: dict,
        model_prob: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> bool:
        """Post-direction filters (e.g. confidence check).  Default: pass."""
        return True

    def compute_direction(
        self,
        model_prob: float,
        yes_price: float,
        no_price: float,
        intent: dict,
        cfg: dict,
        extra_metadata: dict,
    ) -> Optional[tuple[str, float, float, float]]:
        """Pick direction.  Default: compare model_prob to yes_price.

        Returns (direction, entry_price, target_price, edge_pct) or None.
        """
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
        return (direction, entry_price, target_price, edge_percent)

    def position_sizing_params(self, cfg: dict) -> tuple[float, float]:
        """Return (liquidity_fraction, max_cap).  Default: 5%, $400."""
        return (0.05, 400.0)

    def min_roi_threshold(self, cfg: dict) -> float:
        """Minimum ROI to proceed.  Default: min_edge_percent / 2."""
        return float(cfg.get("min_edge_percent", 0)) / 2

    def risk_scoring(
        self,
        cfg: dict,
        intent: dict,
        model_prob: float,
        confidence: float,
        edge_percent: float,
        extra_metadata: dict,
    ) -> tuple[float, list[str]]:
        """Return (risk_score, risk_factors).  Must be overridden."""
        raise NotImplementedError

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
        """Return the position metadata dict.  Must be overridden."""
        raise NotImplementedError

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
        """Return (title, description).  Must be overridden."""
        raise NotImplementedError

    # ==================================================================
    # PROTECTED HELPERS  (callable by subclasses, especially those that
    # override _evaluate_intent entirely like weather_distribution)
    # ==================================================================

    def _compute_confidence_value(
        self,
        intent: dict,
        model_prob: float,
        extra_metadata: dict,
    ) -> float:
        """Compute confidence from intent data.  Subclasses may override."""
        agreement = float(intent.get("model_agreement", 0))
        source_count = int(intent.get("source_count", 0))
        source_spread_c = float(intent.get("source_spread_c") or 0)
        return compute_confidence(agreement, model_prob, source_count, source_spread_c)

    @staticmethod
    def _lookup_market(
        market_id: Optional[str],
        market_map: dict[str, Market],
        event_map: dict[str, Event],
    ) -> tuple[Optional[Market], Optional[Event]]:
        """Look up market and event by ID."""
        market = market_map.get(market_id) if market_id else None
        event = event_map.get(market_id) if market_id else None
        return market, event

    @staticmethod
    def _resolve_token_id(market: Market, direction: str) -> Optional[str]:
        """Resolve the CLOB token ID for the given direction."""
        token_id = None
        if market.clob_token_ids:
            idx = 0 if direction == "buy_yes" else (1 if len(market.clob_token_ids) > 1 else 0)
            token_id = market.clob_token_ids[idx]
        return token_id

    def _compute_profit(
        self,
        entry_price: float,
        target_price: float,
    ) -> dict:
        """Compute profit numbers.  Returns dict with all fields."""
        expected_payout = target_price
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee_amount = expected_payout * self.fee
        net_profit = gross_profit - fee_amount
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        return {
            "expected_payout": expected_payout,
            "total_cost": total_cost,
            "gross_profit": gross_profit,
            "fee_amount": fee_amount,
            "net_profit": net_profit,
            "roi": roi,
        }

    @staticmethod
    def _build_market_dict(market: Market) -> dict:
        """Build the standard market dict for an Opportunity."""
        return {
            "id": market.id,
            "slug": market.slug,
            "question": market.question,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
        }

    def _build_opportunity(
        self,
        *,
        title: str,
        description: str,
        total_cost: float,
        expected_payout: float,
        gross_profit: float,
        fee_amount: float,
        net_profit: float,
        roi: float,
        risk_score: float,
        risk_factors: list[str],
        market_dict: dict,
        event: Optional[Event],
        min_liquidity: float,
        max_position: float,
        market: Market,
        positions: list[dict],
    ) -> Opportunity:
        """Construct the final Opportunity object."""
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
        )
        if opp is not None:
            opp.risk_factors = risk_factors
            opp.min_liquidity = min_liquidity
            opp.max_position_size = max_position
            opp.mispricing_type = MispricingType.NEWS_INFORMATION
        return opp

    # ==================================================================
    # EVALUATE / SHOULD_EXIT  (unified strategy interface)
    # ==================================================================

    # Composable pipeline opt-in
    scoring_weights = ScoringWeights()
    sizing_config = SizingConfig()

    # Per-evaluate transient weather fields (set in custom_checks, read in compute_score/compute_size)
    _weather_agreement: float = 0.0
    _weather_source_count: int = 0
    _weather_source_spread_c: float = 0.0

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        weather = weather_signal_context(signal)

        min_agreement = to_confidence(params.get("min_model_agreement", 0.62), 0.62)
        min_source_count = max(1, int(to_float(params.get("min_source_count", 2), 2)))
        max_source_spread = max(0.0, to_float(params.get("max_source_spread_c", 4.0), 4.0))
        max_entry_price = max(0.05, min(0.98, to_float(params.get("max_entry_price", 0.8), 0.8)))

        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        agreement = to_confidence(weather.get("agreement", weather.get("model_agreement", 0.0)), 0.0)
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
