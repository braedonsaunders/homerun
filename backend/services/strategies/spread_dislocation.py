"""Spread-dislocation opportunity filter.

Portable pattern adapted from open-source spread scalping / micro-spread bots:
scan raw markets for wide book dislocations and emit only executions with
reasonable spread-capture targets and liquidity constraints.
"""

from __future__ import annotations

from typing import Any, Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision, make_aware, utcnow
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload, clamp, days_to_resolution, selected_probability, live_move
from utils.converters import safe_float



class SpreadDislocationStrategy(BaseStrategy):
    """Find spread capture entries on liquid markets with atypical bid/ask gaps."""

    strategy_type = "spread_dislocation"
    name = "Spread Dislocation"
    description = "Wide-spread dislocation filter for passive/limit-style captures"
    mispricing_type = "within_market"
    requires_order_book = True

    default_config = {
        "min_spread": 0.03,
        "max_spread": 0.18,
        "min_mid_price": 0.55,
        "max_mid_price": 0.92,
        "capture_fraction": 0.55,
        "min_target_move": 0.01,
        "min_liquidity": 6000.0,
        "min_days_to_resolution": 0.5,
        "max_days_to_resolution": 60.0,
        "max_opportunities": 50,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    @staticmethod
    def _book_value(payload: Optional[dict], key: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _extract_side_book(
        self,
        market: Market,
        prices: dict[str, dict],
        outcome: str,
    ) -> tuple[Optional[float], Optional[float], Optional[str]]:
        tokens = list(getattr(market, "clob_token_ids", []) or [])
        token_id = None
        if outcome == "YES" and len(tokens) > 0:
            token_id = tokens[0]
        elif outcome == "NO" and len(tokens) > 1:
            token_id = tokens[1]
        payload = prices.get(token_id) if token_id else None
        bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
        ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
        return bid, ask, token_id

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        min_spread = clamp(safe_float(cfg.get("min_spread"), 0.03), 0.005, 0.5)
        max_spread = max(min_spread, clamp(safe_float(cfg.get("max_spread"), 0.18), 0.01, 0.6))
        min_mid_price = clamp(safe_float(cfg.get("min_mid_price"), 0.55), 0.05, 0.98)
        max_mid_price = clamp(safe_float(cfg.get("max_mid_price"), 0.92), min_mid_price + 0.01, 0.99)
        capture_fraction = clamp(safe_float(cfg.get("capture_fraction"), 0.55), 0.1, 0.95)
        min_target_move = clamp(safe_float(cfg.get("min_target_move"), 0.01), 0.002, 0.15)
        min_liquidity = max(100.0, safe_float(cfg.get("min_liquidity"), 6000.0))
        min_days = max(0.0, safe_float(cfg.get("min_days_to_resolution"), 0.5))
        max_days = max(min_days + 0.1, safe_float(cfg.get("max_days_to_resolution"), 60.0))
        max_opportunities = max(1, int(safe_float(cfg.get("max_opportunities"), 50)))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                event_by_market[event_market.id] = event

        now = utcnow()
        ranked: list[tuple[float, ArbitrageOpportunity]] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if safe_float(getattr(market, "liquidity", 0.0)) < min_liquidity:
                continue
            if len(list(getattr(market, "clob_token_ids", []) or [])) < 2:
                continue

            end_date = make_aware(getattr(market, "end_date", None))
            if end_date is not None:
                days = (end_date - now).total_seconds() / 86400.0
                if days < min_days or days > max_days:
                    continue

            for outcome in ("YES", "NO"):
                bid, ask, token_id = self._extract_side_book(market, prices, outcome)
                if bid is None or ask is None:
                    continue
                if bid <= 0.0 or ask <= 0.0 or ask <= bid:
                    continue

                spread = ask - bid
                if spread < min_spread or spread > max_spread:
                    continue

                mid = (ask + bid) / 2.0
                if mid < min_mid_price or mid > max_mid_price:
                    continue

                entry = bid
                target_move = max(min_target_move, spread * capture_fraction)
                target = min(0.99, entry + target_move)
                target = min(target, ask)
                if target <= (entry + 1e-6):
                    continue

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": entry,
                        "token_id": token_id,
                        "entry_style": "passive_bid",
                        "_spread_dislocation": {
                            "bid": bid,
                            "ask": ask,
                            "spread": spread,
                            "mid": mid,
                            "target_price": target,
                            "capture_fraction": capture_fraction,
                        },
                    }
                ]

                opp = self.create_opportunity(
                    title=f"Spread Capture: {outcome} in {market.question[:64]}",
                    description=(
                        f"{outcome} book spread {spread:.3f} ({bid:.3f}/{ask:.3f}); "
                        f"entry {entry:.3f} targeting {target:.3f}."
                    ),
                    total_cost=entry,
                    expected_payout=target,
                    markets=[market],
                    positions=positions,
                    event=event_by_market.get(market.id),
                    is_guaranteed=False,
                    min_liquidity_hard=min_liquidity,
                    min_position_size=max(settings.MIN_POSITION_SIZE, 5.0),
                )
                if not opp:
                    continue

                liquidity = safe_float(getattr(market, "liquidity", 0.0))
                liquidity_discount = min(0.12, liquidity / 120000.0)
                risk = 0.58 + min(0.18, spread * 1.8) - liquidity_discount
                opp.risk_score = clamp(risk, 0.32, 0.82)
                opp.risk_factors = [
                    f"Book spread {spread:.2%}",
                    "Execution depends on passive/limit queue quality",
                    f"Liquidity ${liquidity:,.0f}",
                ]
                opp.mispricing_type = MispricingType.WITHIN_MARKET

                quality = (target - entry) * (1.0 - opp.risk_score)
                ranked.append((quality, opp))

        if not ranked:
            return []

        ranked.sort(key=lambda item: item[0], reverse=True)
        out: list[ArbitrageOpportunity] = []
        seen: set[tuple[str, str]] = set()
        for _, opp in ranked:
            market_id = str((opp.markets or [{}])[0].get("id") or "")
            outcome = str((opp.positions_to_take or [{}])[0].get("outcome") or "")
            key = (market_id, outcome)
            if key in seen:
                continue
            seen.add(key)
            out.append(opp)
            if len(out) >= max_opportunities:
                break
        return out

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = to_confidence(params.get("min_confidence", 0.40), 0.40)
        max_risk = to_confidence(params.get("max_risk_score", 0.80), 0.80)
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 500.0), 500.0))
        base_size = max(1.0, to_float(params.get("base_size_usd", 16.0), 16.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 120.0), 120.0))

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)

        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck("confidence", "Confidence threshold", confidence >= min_conf, score=confidence, detail=f"min={min_conf:.2f}"),
            DecisionCheck("risk", "Risk ceiling", risk_score <= max_risk, score=risk_score, detail=f"max={max_risk:.2f}"),
            DecisionCheck("liquidity", "Liquidity floor", liquidity >= min_liquidity, score=liquidity, detail=f"min={min_liquidity:.0f}"),
        ]

        score = (edge * 0.60) + (confidence * 28.0) + (min(1.0, liquidity / 8000.0) * 6.0) - (risk_score * 9.0)

        if not all(c.passed for c in checks):
            return StrategyDecision("skipped", "Spread dislocation filters not met", score=score, checks=checks)

        size = base_size * (1.0 + (edge / 100.0)) * (0.70 + confidence)
        size = max(1.0, min(max_size, size))

        return StrategyDecision("selected", "Spread dislocation signal selected", score=score, size_usd=size, checks=checks)

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Exit when spread closes or on time decay."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        age_minutes = float(getattr(position, "age_minutes", 0) or 0)
        max_hold = float(config.get("max_hold_minutes", 180) or 180)
        if age_minutes > max_hold:
            current_price = market_state.get("current_price")
            return ExitDecision("close", f"Spread dislocation time decay ({age_minutes:.0f} > {max_hold:.0f} min)", close_price=current_price)
        return self.default_exit_check(position, market_state)
