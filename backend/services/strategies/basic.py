# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
from __future__ import annotations

import logging
from typing import Any

from models import Event, ExecutionConstraints, ExecutionLeg, ExecutionPlan, Market, Opportunity
from services.fee_model import fee_model
from services.quality_filter import QualityFilterOverrides
from services.strategies.base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig
from utils.converters import safe_float, to_float
logger = logging.getLogger(__name__)


def _book_value(payload: dict[str, Any] | None, *keys: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = safe_float(payload.get(key), None)
        if value is not None and value > 0.0:
            return float(value)
    return None


def _ask_depth(payload: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    for key in ("best_ask_size", "ask_size", "ask_depth"):
        value = safe_float(payload.get(key), None)
        if value is not None and value > 0.0:
            return float(value), key
    return None, None


class BasicArbStrategy(BaseStrategy):
    """Executable same-market binary arbitrage using live ask prices only."""

    strategy_type = "basic"
    name = "Basic Arbitrage"
    description = "Buy executable YES+NO pairs on the same market only when the live book locks net profit"
    mispricing_type = "within_market"
    source_key = "scanner"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.2,
        min_annualized_roi=0.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.70,
        confidence_weight=32.0,
        risk_penalty=10.0,
        liquidity_weight=6.0,
        liquidity_divisor=5000.0,
        market_count_bonus=0.0,
        structural_bonus=3.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.80,
        risk_scale_factor=0.30,
        risk_floor=0.60,
        market_scale_factor=0.0,
        market_scale_cap=0,
    )
    default_config = {
        "min_edge_percent": 0.75,
        "min_confidence": 0.55,
        "max_risk_score": 0.32,
        "min_liquidity": 1000.0,
        "min_book_depth": 25.0,
        "max_leg_spread": 0.03,
        "require_accepting_orders": True,
        "require_order_book": True,
        "require_polymarket": True,
        "retention_window": "2m",
    }
    pipeline_defaults = {
        "min_edge_percent": 0.75,
        "min_confidence": 0.55,
        "max_risk_score": 0.32,
    }

    @staticmethod
    def _quotes_for_market(market: Market, prices: dict[str, dict]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        token_ids = list(getattr(market, "clob_token_ids", []) or [])
        yes_payload = prices.get(token_ids[0]) if len(token_ids) > 0 else None
        no_payload = prices.get(token_ids[1]) if len(token_ids) > 1 else None
        return yes_payload if isinstance(yes_payload, dict) else None, no_payload if isinstance(no_payload, dict) else None

    def _eligible_market(self, market: Market, config: dict[str, Any]) -> bool:
        if len(list(getattr(market, "outcome_prices", []) or [])) != 2:
            return False
        if market.closed or not market.active:
            return False
        if bool(getattr(market, "resolved", False)) or bool(getattr(market, "archived", False)):
            return False
        if bool(config.get("require_polymarket", True)) and str(getattr(market, "platform", "") or "").strip().lower() != "polymarket":
            return False
        if len(list(getattr(market, "clob_token_ids", []) or [])) < 2:
            return False
        if bool(config.get("require_accepting_orders", True)) and getattr(market, "accepting_orders", None) is not True:
            return False
        if bool(config.get("require_order_book", True)) and getattr(market, "enable_order_book", None) is not True:
            return False
        return True

    def _pair_risk_score(
        self,
        *,
        max_leg_spread: float,
        max_allowed_spread: float,
        min_depth_seen: float,
        min_depth_required: float,
        liquidity: float,
    ) -> float:
        spread_component = 0.0 if max_allowed_spread <= 0.0 else min(max_leg_spread / max_allowed_spread, 1.0) * 0.18
        depth_component = 0.0 if min_depth_required <= 0.0 else max(0.0, 1.0 - min(min_depth_seen / min_depth_required, 1.0)) * 0.12
        liquidity_credit = min(liquidity / 25000.0, 1.0) * 0.06
        return max(0.08, min(0.45, 0.18 + spread_component + depth_component - liquidity_credit))

    def _pair_confidence(
        self,
        *,
        edge_percent: float,
        max_leg_spread: float,
        liquidity: float,
    ) -> float:
        confidence = 0.62
        confidence += min(max(edge_percent, 0.0) / 8.0, 1.0) * 0.18
        confidence += min(liquidity / 25000.0, 1.0) * 0.10
        confidence += max(0.0, 0.03 - max_leg_spread) / 0.03 * 0.05
        return max(0.5, min(0.95, confidence))

    def _build_execution_plan(
        self,
        *,
        positions: list[dict[str, Any]],
        markets: list[Market],
        market_roster: dict[str, Any] | None = None,
        is_guaranteed: bool = True,
    ) -> ExecutionPlan | None:
        if len(positions) != 2 or len(markets) != 1:
            return None
        market = markets[0]
        market_id = str(market.id)
        roster_scope = str(market_roster.get("scope") or "").strip().lower() if isinstance(market_roster, dict) else ""
        roster_hash = (
            str(market_roster.get("roster_hash") or "").strip() or None
            if isinstance(market_roster, dict)
            else None
        )
        roster_market_ids = []
        if isinstance(market_roster, dict):
            for raw_market in market_roster.get("markets") or []:
                if not isinstance(raw_market, dict):
                    continue
                raw_market_id = str(raw_market.get("id") or raw_market.get("market_id") or "").strip()
                if raw_market_id and raw_market_id not in roster_market_ids:
                    roster_market_ids.append(raw_market_id)

        legs: list[ExecutionLeg] = []
        total_cost = sum(max(0.0, to_float(position.get("price"), 0.0)) for position in positions)
        for index, position in enumerate(positions, start=1):
            limit_price = max(0.0, to_float(position.get("price"), 0.0))
            if limit_price <= 0.0:
                return None
            token_id = str(position.get("token_id") or "").strip() or None
            outcome = str(position.get("outcome") or "").strip().lower() or None
            if token_id is None or outcome not in {"yes", "no"}:
                return None
            notional_weight = limit_price / total_cost if total_cost > 0.0 else 0.5
            legs.append(
                ExecutionLeg(
                    leg_id=f"pair_leg_{index}",
                    market_id=market_id,
                    market_question=market.question,
                    token_id=token_id,
                    side="buy",
                    outcome=outcome,
                    limit_price=limit_price,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=max(0.0001, notional_weight),
                    min_fill_ratio=1.0,
                    metadata={
                        "position_index": index - 1,
                        "same_market_pair": True,
                    },
                )
            )

        return ExecutionPlan(
            policy="PAIR_LOCK",
            time_in_force="IOC",
            legs=legs,
            constraints=ExecutionConstraints(
                max_unhedged_notional_usd=0.0,
                hedge_timeout_seconds=5,
                session_timeout_seconds=30,
                max_reprice_attempts=0,
                pair_lock=True,
                leg_fill_tolerance_ratio=0.0,
            ),
            metadata={
                "strategy_type": self.strategy_type,
                "generated_by": "basic.same_market_pair",
                "market_coverage": {
                    "scope": roster_scope or None,
                    "market_roster_hash": roster_hash,
                    "roster_market_count": len(roster_market_ids),
                    "planned_market_count": 1,
                    "planned_market_ids": [market_id],
                    "missing_market_ids": [mid for mid in roster_market_ids if mid != market_id],
                    "event_internal_bundle": False,
                    "requires_full_market_coverage": False,
                },
            },
        )

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:
        del events
        config = dict(self.default_config)
        config.update(getattr(self, "config", {}) or {})
        min_edge_percent = max(0.0, to_float(config.get("min_edge_percent", 0.75), 0.75))
        min_liquidity = max(0.0, to_float(config.get("min_liquidity", 1000.0), 1000.0))
        min_book_depth = max(0.0, to_float(config.get("min_book_depth", 25.0), 25.0))
        max_leg_spread_allowed = max(0.0, to_float(config.get("max_leg_spread", 0.03), 0.03))
        opportunities: list[Opportunity] = []

        for market in markets:
            if not self._eligible_market(market, config):
                continue
            if safe_float(getattr(market, "liquidity", 0.0), 0.0) < min_liquidity:
                continue

            yes_payload, no_payload = self._quotes_for_market(market, prices)
            yes_bid = _book_value(yes_payload, "bid", "best_bid")
            yes_ask = _book_value(yes_payload, "ask", "best_ask")
            no_bid = _book_value(no_payload, "bid", "best_bid")
            no_ask = _book_value(no_payload, "ask", "best_ask")
            if yes_bid is None or yes_ask is None or no_bid is None or no_ask is None:
                continue
            if yes_ask < yes_bid or no_ask < no_bid:
                continue

            yes_depth, yes_depth_key = _ask_depth(yes_payload)
            no_depth, no_depth_key = _ask_depth(no_payload)
            if yes_depth is None or no_depth is None:
                continue
            if min(yes_depth, no_depth) < min_book_depth:
                continue

            max_leg_spread = max(yes_ask - yes_bid, no_ask - no_bid)
            if max_leg_spread > max_leg_spread_allowed:
                continue

            executable_total_cost = yes_ask + no_ask
            if executable_total_cost <= 0.0 or executable_total_cost >= 1.0:
                continue

            gross_profit = 1.0 - executable_total_cost
            net_profit, fee_breakdown = fee_model.net_profit_after_all_fees(
                gross_profit=gross_profit,
                expected_payout=1.0,
                num_legs=2,
                is_negrisk=False,
                spread_bps=0.0,
                total_cost=executable_total_cost,
                maker_mode=False,
                total_liquidity=max(safe_float(getattr(market, "liquidity", 0.0), 0.0), min_liquidity),
                platform="polymarket",
                entry_prices=[yes_ask, no_ask],
            )
            if net_profit <= 0.0:
                continue

            edge_percent = (net_profit / executable_total_cost) * 100.0
            if edge_percent < min_edge_percent:
                continue

            confidence = self._pair_confidence(
                edge_percent=edge_percent,
                max_leg_spread=max_leg_spread,
                liquidity=max(safe_float(getattr(market, "liquidity", 0.0), 0.0), min_liquidity),
            )
            risk_score = self._pair_risk_score(
                max_leg_spread=max_leg_spread,
                max_allowed_spread=max_leg_spread_allowed,
                min_depth_seen=min(yes_depth, no_depth),
                min_depth_required=min_book_depth,
                liquidity=max(safe_float(getattr(market, "liquidity", 0.0), 0.0), min_liquidity),
            )

            positions = [
                {
                    "action": "BUY",
                    "market_id": market.id,
                    "market_question": market.question,
                    "outcome": "YES",
                    "price": yes_ask,
                    "token_id": market.clob_token_ids[0],
                    "price_policy": "taker_limit",
                    "time_in_force": "IOC",
                    "notional_weight": yes_ask / executable_total_cost,
                    "min_fill_ratio": 1.0,
                },
                {
                    "action": "BUY",
                    "market_id": market.id,
                    "market_question": market.question,
                    "outcome": "NO",
                    "price": no_ask,
                    "token_id": market.clob_token_ids[1],
                    "price_policy": "taker_limit",
                    "time_in_force": "IOC",
                    "notional_weight": no_ask / executable_total_cost,
                    "min_fill_ratio": 1.0,
                },
            ]

            opportunity = self.create_opportunity(
                title=f"Basic Pair Arb: {market.question[:60]}",
                description=(
                    f"Executable same-market pair: buy YES at ${yes_ask:.3f} and NO at ${no_ask:.3f}. "
                    f"Conservative net after taker fees and gas: ${net_profit:.4f} per paired share."
                ),
                total_cost=executable_total_cost,
                expected_payout=1.0,
                markets=[market],
                positions=positions,
                is_guaranteed=True,
                vwap_total_cost=executable_total_cost,
                spread_bps=0.0,
                fill_probability=1.0,
                confidence=confidence,
                custom_risk_score=risk_score,
                fee_model_maker_mode=False,
            )
            if opportunity is None:
                continue

            capacity_notional = None
            if yes_depth_key in {"best_ask_size", "ask_size"} and no_depth_key in {"best_ask_size", "ask_size"}:
                capacity_notional = min(yes_depth, no_depth) * executable_total_cost
            if capacity_notional is not None and capacity_notional > 0.0:
                opportunity.max_position_size = min(
                    float(opportunity.max_position_size or capacity_notional),
                    float(capacity_notional),
                )

            opportunity.risk_factors = [
                "Guaranteed same-market binary payout only if both IOC buy legs fully fill.",
                f"Executable pair cost uses live asks only: YES ${yes_ask:.3f}, NO ${no_ask:.3f}.",
                f"Leg spreads: YES {(yes_ask - yes_bid):.3f}, NO {(no_ask - no_bid):.3f}.",
                f"Taker fee + gas haircut applied: ${fee_breakdown.total_fees:.4f} per paired share.",
            ]
            opportunity.strategy_context.update(
                {
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "no_bid": no_bid,
                    "no_ask": no_ask,
                    "yes_ask_depth": yes_depth,
                    "yes_ask_depth_source": yes_depth_key,
                    "no_ask_depth": no_depth,
                    "no_ask_depth_source": no_depth_key,
                    "gross_pair_cost": executable_total_cost,
                    "net_pair_profit": net_profit,
                    "fee_total_usd": fee_breakdown.total_fees,
                    "entry_mode": "paired_ioc_taker",
                }
            )
            opportunities.append(opportunity)

        return opportunities

    SOURCES = {"scanner"}

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 1000.0), 1000.0))
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        market_count = len(payload.get("markets") or [])
        position_count = len(payload.get("positions_to_take") or [])
        execution_plan = payload.get("execution_plan") if isinstance(payload.get("execution_plan"), dict) else {}
        execution_policy = str(execution_plan.get("policy") or "").strip().upper()
        payload["liquidity"] = liquidity
        return [
            DecisionCheck("source", "Scanner source", source_ok, detail="Requires source=scanner."),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity >= min_liquidity,
                score=liquidity,
                detail=f"min={min_liquidity:.0f}",
            ),
            DecisionCheck(
                "pair_shape",
                "Same-market pair contract",
                market_count == 1 and position_count == 2 and execution_policy == "PAIR_LOCK",
                score=float(position_count),
                detail=f"markets={market_count}, positions={position_count}, policy={execution_policy or 'n/a'}",
            ),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        if not config.get("resolve_only", True):
            return self.default_exit_check(position, market_state)
        return ExitDecision("hold", "Guaranteed same-market pair - hold to resolution")

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked - %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f -> $%.0f - %s", self.name, original_size, capped_size, reason)
