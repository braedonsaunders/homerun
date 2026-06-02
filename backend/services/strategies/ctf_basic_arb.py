# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
from __future__ import annotations

import logging
from typing import Any, Optional

from models import Event, ExecutionConstraints, ExecutionLeg, ExecutionPlan, Market, Opportunity
from services.quality_filter import QualityFilterOverrides
from services.strategies.base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig
from utils.converters import safe_float, to_float
from utils.kelly import polymarket_taker_fee

logger = logging.getLogger(__name__)


def _book_value(payload: dict[str, Any] | None, *keys: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = safe_float(payload.get(key), None)
        if value is not None and value > 0.0:
            return float(value)
    return None


def _book_capacity(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("best_bid_size", "bid_size", "bid_depth", "best_ask_size", "ask_size", "ask_depth"):
        value = safe_float(payload.get(key), None)
        if value is not None and value > 0.0:
            return float(value)
    return None


class CTFBasicArbStrategy(BaseStrategy):
    """Immediate structural arb using CTF split/merge plus the order book."""

    strategy_type = "ctf_basic_arb"
    name = "CTF Basic Arb"
    description = "Immediate split/sell or buy/merge arbitrage using Polymarket CTF"
    mispricing_type = "within_market"
    source_key = "scanner"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.2,
        min_annualized_roi=0.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=30.0,
        risk_penalty=7.5,
        liquidity_weight=8.0,
        liquidity_divisor=5000.0,
        structural_bonus=2.5,
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
        "min_edge_percent": 0.60,
        "min_confidence": 0.45,
        "max_risk_score": 0.55,
        "min_liquidity": 500.0,
        "gas_buffer_usd": 0.30,
        "assumed_trade_size_usd": 25.0,
        "require_accepting_orders": True,
        "require_order_book": True,
        "allow_split_sell": True,
        "allow_buy_merge": True,
        "min_position_size": 10.0,
    }

    @staticmethod
    def _quotes_for_market(market: Market, prices: dict[str, dict]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        token_ids = list(getattr(market, "clob_token_ids", []) or [])
        yes_payload = prices.get(token_ids[0]) if len(token_ids) > 0 else None
        no_payload = prices.get(token_ids[1]) if len(token_ids) > 1 else None
        return yes_payload if isinstance(yes_payload, dict) else None, no_payload if isinstance(no_payload, dict) else None

    @staticmethod
    def _condition_market_id(market: Market) -> str:
        condition_id = str(getattr(market, "condition_id", "") or "").strip()
        if condition_id.startswith("0x") and len(condition_id) == 66:
            return condition_id
        return ""

    def _gas_per_share(self, config: dict[str, Any]) -> float:
        gas_buffer = max(0.0, to_float(config.get("gas_buffer_usd", 0.30), 0.30))
        assumed_trade_size = max(1.0, to_float(config.get("assumed_trade_size_usd", 25.0), 25.0))
        return gas_buffer / assumed_trade_size

    def _build_execution_plan(
        self,
        *,
        positions: list[dict[str, Any]],
        markets: list[Market],
        market_roster: dict[str, Any] | None = None,
        is_guaranteed: bool = True,
    ) -> ExecutionPlan | None:
        del market_roster, is_guaranteed
        if len(markets) != 1 or not positions:
            return None
        market = markets[0]
        mode: str | None = None
        yes_price: float | None = None
        no_price: float | None = None
        for position in positions:
            action = str(position.get("action") or position.get("side") or "").strip().upper()
            outcome = str(position.get("outcome") or "").strip().upper()
            price = safe_float(position.get("price"), None)
            if action == "SPLIT":
                mode = "split_sell"
                continue
            if action == "MERGE":
                mode = "buy_merge"
                continue
            if action in {"BUY", "SELL"} and outcome == "YES" and price is not None:
                yes_price = float(price)
                continue
            if action in {"BUY", "SELL"} and outcome == "NO" and price is not None:
                no_price = float(price)
        if mode not in {"split_sell", "buy_merge"} or yes_price is None or no_price is None:
            return None
        condition_id = self._condition_market_id(market)
        yes_token = market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None
        no_token = market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
        if mode == "split_sell":
            legs = [
                ExecutionLeg(
                    leg_id="split_leg",
                    market_id=condition_id,
                    market_question=market.question,
                    side="buy",
                    limit_price=1.0,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=1.0,
                    metadata={"ctf_action": "split", "condition_id": condition_id},
                ),
                ExecutionLeg(
                    leg_id="sell_yes_leg",
                    market_id=condition_id,
                    market_question=market.question,
                    token_id=yes_token,
                    side="sell",
                    outcome="YES",
                    limit_price=yes_price,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=max(0.0001, yes_price),
                    metadata={"condition_id": condition_id},
                ),
                ExecutionLeg(
                    leg_id="sell_no_leg",
                    market_id=condition_id,
                    market_question=market.question,
                    token_id=no_token,
                    side="sell",
                    outcome="NO",
                    limit_price=no_price,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=max(0.0001, no_price),
                    metadata={"condition_id": condition_id},
                ),
            ]
        else:
            legs = [
                ExecutionLeg(
                    leg_id="buy_yes_leg",
                    market_id=condition_id,
                    market_question=market.question,
                    token_id=yes_token,
                    side="buy",
                    outcome="YES",
                    limit_price=yes_price,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=max(0.0001, yes_price),
                    metadata={"condition_id": condition_id},
                ),
                ExecutionLeg(
                    leg_id="buy_no_leg",
                    market_id=condition_id,
                    market_question=market.question,
                    token_id=no_token,
                    side="buy",
                    outcome="NO",
                    limit_price=no_price,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=max(0.0001, no_price),
                    metadata={"condition_id": condition_id},
                ),
                ExecutionLeg(
                    leg_id="merge_leg",
                    market_id=condition_id,
                    market_question=market.question,
                    side="buy",
                    limit_price=1.0,
                    price_policy="taker_limit",
                    time_in_force="IOC",
                    post_only=False,
                    notional_weight=1.0,
                    metadata={"ctf_action": "merge", "condition_id": condition_id},
                ),
            ]
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
                "ctf_bundle": mode,
                "market_coverage": {"requires_full_market_coverage": False},
            },
        )

    def _annotate_capacity(
        self,
        *,
        opportunity: Opportunity,
        yes_payload: dict[str, Any] | None,
        no_payload: dict[str, Any] | None,
    ) -> None:
        yes_capacity = _book_capacity(yes_payload)
        no_capacity = _book_capacity(no_payload)
        capacities = [value for value in (yes_capacity, no_capacity) if value is not None and value > 0.0]
        if capacities:
            opportunity.max_position_size = min(float(opportunity.max_position_size or 0.0) or capacities[0], min(capacities))

    def _create_split_sell_opportunity(
        self,
        *,
        market: Market,
        event: Event | None,
        config: dict[str, Any],
        yes_payload: dict[str, Any] | None,
        no_payload: dict[str, Any] | None,
    ) -> Optional[Opportunity]:
        yes_bid = _book_value(yes_payload, "bid", "best_bid")
        no_bid = _book_value(no_payload, "bid", "best_bid")
        if yes_bid is None or no_bid is None:
            return None

        proceeds = yes_bid + no_bid
        fee_per_share = polymarket_taker_fee(yes_bid) + polymarket_taker_fee(no_bid)
        net_proceeds = proceeds - fee_per_share - self._gas_per_share(config)
        edge_percent = (net_proceeds - 1.0) * 100.0
        min_edge_percent = max(0.0, to_float(config.get("min_edge_percent", 0.60), 0.60))
        if edge_percent < min_edge_percent:
            return None

        positions = [
            {
                "action": "SPLIT",
                "market_id": market.condition_id,
                "market": market.question[:60],
                "price": 1.0,
                "condition_id": market.condition_id,
            },
            {
                "action": "SELL",
                "market_id": market.condition_id,
                "market": market.question[:60],
                "outcome": "YES",
                "price": yes_bid,
                "token_id": market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None,
            },
            {
                "action": "SELL",
                "market_id": market.condition_id,
                "market": market.question[:60],
                "outcome": "NO",
                "price": no_bid,
                "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None,
            },
        ]

        opportunity = self.create_opportunity(
            title=f"CTF Split Arb: {market.question[:60]}",
            description=(
                f"Split 1 USDC into YES+NO, then sell both at the current bids. "
                f"YES bid ${yes_bid:.3f} + NO bid ${no_bid:.3f} -> conservative net ${net_proceeds:.3f}."
            ),
            total_cost=1.0,
            expected_payout=net_proceeds,
            markets=[market],
            positions=positions,
            event=event,
            is_guaranteed=True,
            skip_fee_model=True,
            confidence=0.78,
            custom_risk_score=0.18,
            min_liquidity_hard=max(0.0, to_float(config.get("min_liquidity", 500.0), 500.0)),
            min_position_size=max(1.0, to_float(config.get("min_position_size", 10.0), 10.0)),
        )
        if opportunity is None:
            return None

        opportunity.risk_factors = [
            "Structural CTF split/sell arbitrage with full-bundle execution required.",
            f"Conservative proceeds use live bids only: YES ${yes_bid:.3f}, NO ${no_bid:.3f}.",
            f"Taker fee + gas buffer haircut applied: ${fee_per_share + self._gas_per_share(config):.3f} per share.",
        ]
        opportunity.strategy_context.update(
            {
                "sub_strategy": "split_sell",
                "condition_id": market.condition_id,
                "yes_bid": yes_bid,
                "no_bid": no_bid,
                "gross_proceeds": proceeds,
                "net_proceeds": net_proceeds,
                "per_share_fee_and_gas": fee_per_share + self._gas_per_share(config),
            }
        )
        self._annotate_capacity(opportunity=opportunity, yes_payload=yes_payload, no_payload=no_payload)
        return opportunity

    def _create_buy_merge_opportunity(
        self,
        *,
        market: Market,
        event: Event | None,
        config: dict[str, Any],
        yes_payload: dict[str, Any] | None,
        no_payload: dict[str, Any] | None,
    ) -> Optional[Opportunity]:
        yes_ask = _book_value(yes_payload, "ask", "best_ask")
        no_ask = _book_value(no_payload, "ask", "best_ask")
        if yes_ask is None or no_ask is None:
            return None

        buy_cost = yes_ask + no_ask
        fee_per_share = polymarket_taker_fee(yes_ask) + polymarket_taker_fee(no_ask)
        total_cost = buy_cost + fee_per_share + self._gas_per_share(config)
        edge_percent = (1.0 - total_cost) * 100.0
        min_edge_percent = max(0.0, to_float(config.get("min_edge_percent", 0.60), 0.60))
        if edge_percent < min_edge_percent:
            return None

        positions = [
            {
                "action": "BUY",
                "market_id": market.condition_id,
                "market": market.question[:60],
                "outcome": "YES",
                "price": yes_ask,
                "token_id": market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None,
            },
            {
                "action": "BUY",
                "market_id": market.condition_id,
                "market": market.question[:60],
                "outcome": "NO",
                "price": no_ask,
                "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None,
            },
            {
                "action": "MERGE",
                "market_id": market.condition_id,
                "market": market.question[:60],
                "price": 1.0,
                "condition_id": market.condition_id,
            },
        ]

        opportunity = self.create_opportunity(
            title=f"CTF Merge Arb: {market.question[:60]}",
            description=(
                f"Buy both sides at the current asks, then merge immediately into 1 USDC. "
                f"YES ask ${yes_ask:.3f} + NO ask ${no_ask:.3f} -> conservative cost ${total_cost:.3f}."
            ),
            total_cost=total_cost,
            expected_payout=1.0,
            markets=[market],
            positions=positions,
            event=event,
            is_guaranteed=True,
            skip_fee_model=True,
            confidence=0.76,
            custom_risk_score=0.16,
            min_liquidity_hard=max(0.0, to_float(config.get("min_liquidity", 500.0), 500.0)),
            min_position_size=max(1.0, to_float(config.get("min_position_size", 10.0), 10.0)),
        )
        if opportunity is None:
            return None

        opportunity.risk_factors = [
            "Structural CTF buy/merge arbitrage with full-bundle execution required.",
            f"Conservative cost uses live asks only: YES ${yes_ask:.3f}, NO ${no_ask:.3f}.",
            f"Taker fee + gas buffer added to entry cost: ${fee_per_share + self._gas_per_share(config):.3f} per share.",
        ]
        opportunity.strategy_context.update(
            {
                "sub_strategy": "buy_merge",
                "condition_id": market.condition_id,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "gross_buy_cost": buy_cost,
                "total_cost": total_cost,
                "per_share_fee_and_gas": fee_per_share + self._gas_per_share(config),
            }
        )
        self._annotate_capacity(opportunity=opportunity, yes_payload=yes_payload, no_payload=no_payload)
        return opportunity

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:
        config = dict(self.default_config)
        config.update(getattr(self, "config", {}) or {})
        require_accepting_orders = bool(config.get("require_accepting_orders", True))
        require_order_book = bool(config.get("require_order_book", True))
        min_liquidity = max(0.0, to_float(config.get("min_liquidity", 500.0), 500.0))

        market_to_event: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                market_to_event[str(event_market.id)] = event

        opportunities: list[Opportunity] = []
        for market in markets:
            if market.closed or not market.active:
                continue
            if str(getattr(market, "platform", "") or "polymarket").strip().lower() != "polymarket":
                continue
            if len(market.outcome_prices) != 2:
                continue
            if market.liquidity < min_liquidity:
                continue
            if not self._condition_market_id(market):
                continue
            if len(market.clob_token_ids) < 2:
                continue
            if require_accepting_orders and market.accepting_orders is False:
                continue
            if require_order_book and market.enable_order_book is False:
                continue

            yes_payload, no_payload = self._quotes_for_market(market, prices)
            if config.get("allow_split_sell", True):
                split_opportunity = self._create_split_sell_opportunity(
                    market=market,
                    event=market_to_event.get(str(market.id)),
                    config=config,
                    yes_payload=yes_payload,
                    no_payload=no_payload,
                )
                if split_opportunity is not None:
                    opportunities.append(split_opportunity)
            if config.get("allow_buy_merge", True):
                merge_opportunity = self._create_buy_merge_opportunity(
                    market=market,
                    event=market_to_event.get(str(market.id)),
                    config=config,
                    yes_payload=yes_payload,
                    no_payload=no_payload,
                )
                if merge_opportunity is not None:
                    opportunities.append(merge_opportunity)

        return opportunities

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        source = str(getattr(signal, "source", "") or "").strip().lower()
        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail=f"got={source}"),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        return self.default_exit_check(position, market_state)
