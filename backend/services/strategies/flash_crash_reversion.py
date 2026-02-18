"""Flash-crash reversion opportunity filter.

Portable pattern adapted from open-source Polymarket flash-crash bots:
track short-horizon price drops and surface rebound entries only when
liquidity/spread constraints make execution realistic.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision, ScoringWeights, SizingConfig
from services.data_events import DataEvent
from utils.converters import to_float, to_confidence, clamp
from utils.signal_helpers import signal_payload, days_to_resolution, selected_probability, live_move
from utils.converters import safe_float



class FlashCrashReversionStrategy(BaseStrategy):
    """Detect abrupt short-window probability crashes and buy reversion."""

    strategy_type = "flash_crash_reversion"
    name = "Flash Crash Reversion"
    description = "Short-horizon crash detector with liquidity/spread gating"
    mispricing_type = "within_market"
    subscriptions = ["market_data_refresh"]

    async def on_event(self, event: DataEvent) -> list[ArbitrageOpportunity]:
        if event.event_type == "market_data_refresh":
            return self.detect(event.events or [], event.markets or [], event.prices or {})
        return []
    requires_historical_prices = True

    pipeline_defaults = {
        "min_edge_percent": 3.0,
        "min_confidence": 0.40,
        "max_risk_score": 0.80,
        "base_size_usd": 16.0,
        "max_size_usd": 130.0,
    }

    # Composable evaluate pipeline: score = edge*0.65 + conf*30 + liq_score*8 - risk*10
    scoring_weights = ScoringWeights(
        edge_weight=0.65,
        confidence_weight=30.0,
        risk_penalty=10.0,
        liquidity_weight=8.0,
        liquidity_divisor=10000.0,
    )
    # Sizing uses Kelly criterion -- override compute_size
    sizing_config = SizingConfig()

    default_config = {
        "lookback_seconds": 240.0,
        "stale_history_seconds": 1800.0,
        "drop_threshold": 0.08,
        "min_rebound_fraction": 0.45,
        "min_target_move": 0.015,
        "max_entry_price": 0.82,
        "max_spread": 0.07,
        "min_liquidity": 2500.0,
        "max_opportunities": 40,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)
        # market_id -> deque[(ts, yes, no, yes_bid, yes_ask, no_bid, no_ask)]
        self._history: dict[
            str, deque[tuple[float, float, float, Optional[float], Optional[float], Optional[float], Optional[float]]]
        ] = {}

    @staticmethod
    def _extract_book_value(payload: Optional[dict], key: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        val = payload.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _extract_yes_no_snapshot(
        self,
        market: Market,
        prices: dict[str, dict],
    ) -> tuple[float, float, Optional[float], Optional[float], Optional[float], Optional[float]]:
        yes = safe_float(market.yes_price)
        no = safe_float(market.no_price)
        yes_bid = None
        yes_ask = None
        no_bid = None
        no_ask = None

        token_ids = list(getattr(market, "clob_token_ids", []) or [])
        if token_ids:
            yes_raw = prices.get(token_ids[0])
            yes_mid = self._extract_book_value(yes_raw, "mid")
            if yes_mid is None:
                yes_mid = self._extract_book_value(yes_raw, "price")
            if yes_mid is not None:
                yes = yes_mid
            yes_bid = self._extract_book_value(yes_raw, "bid") or self._extract_book_value(yes_raw, "best_bid")
            yes_ask = self._extract_book_value(yes_raw, "ask") or self._extract_book_value(yes_raw, "best_ask")

        if len(token_ids) > 1:
            no_raw = prices.get(token_ids[1])
            no_mid = self._extract_book_value(no_raw, "mid")
            if no_mid is None:
                no_mid = self._extract_book_value(no_raw, "price")
            if no_mid is not None:
                no = no_mid
            no_bid = self._extract_book_value(no_raw, "bid") or self._extract_book_value(no_raw, "best_bid")
            no_ask = self._extract_book_value(no_raw, "ask") or self._extract_book_value(no_raw, "best_ask")

        if (yes <= 0.0 or no <= 0.0) and len(getattr(market, "outcome_prices", []) or []) >= 2:
            yes = yes if yes > 0.0 else safe_float(market.outcome_prices[0])
            no = no if no > 0.0 else safe_float(market.outcome_prices[1])

        return yes, no, yes_bid, yes_ask, no_bid, no_ask

    def _side_spread(
        self,
        bid: Optional[float],
        ask: Optional[float],
    ) -> float:
        if bid is None or ask is None:
            return 0.0
        if bid <= 0.0 or ask <= 0.0:
            return 0.0
        return max(0.0, ask - bid)

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        lookback_seconds = max(30.0, safe_float(cfg.get("lookback_seconds"), 240.0))
        stale_history_seconds = max(lookback_seconds * 2.0, safe_float(cfg.get("stale_history_seconds"), 1800.0))
        drop_threshold = clamp(safe_float(cfg.get("drop_threshold"), 0.08), 0.01, 0.50)
        min_rebound_fraction = clamp(safe_float(cfg.get("min_rebound_fraction"), 0.45), 0.10, 0.95)
        min_target_move = clamp(safe_float(cfg.get("min_target_move"), 0.015), 0.005, 0.15)
        max_entry_price = clamp(safe_float(cfg.get("max_entry_price"), 0.82), 0.2, 0.99)
        max_spread = clamp(safe_float(cfg.get("max_spread"), 0.07), 0.005, 0.25)
        min_liquidity = max(100.0, safe_float(cfg.get("min_liquidity"), 2500.0))
        max_opportunities = max(1, int(safe_float(cfg.get("max_opportunities"), 40)))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                event_by_market[event_market.id] = event

        now = time.time()
        candidates: list[tuple[float, ArbitrageOpportunity]] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if (
                len(list(getattr(market, "outcome_prices", []) or [])) < 2
                and len(list(getattr(market, "clob_token_ids", []) or [])) < 2
            ):
                continue
            if safe_float(getattr(market, "liquidity", 0.0)) < min_liquidity:
                continue

            yes, no, yes_bid, yes_ask, no_bid, no_ask = self._extract_yes_no_snapshot(market, prices)
            if not (0.0 < yes < 1.0 and 0.0 < no < 1.0):
                continue

            row = (now, yes, no, yes_bid, yes_ask, no_bid, no_ask)
            history = self._history.setdefault(market.id, deque(maxlen=180))
            history.append(row)

            # Keep only recent rows for crash detection.
            while history and (now - history[0][0]) > stale_history_seconds:
                history.popleft()

            if len(history) < 2:
                continue

            baseline = None
            cutoff = now - lookback_seconds
            for point in history:
                if point[0] >= cutoff:
                    baseline = point
                    break
            if baseline is None:
                continue

            for outcome, idx, bid, ask in (
                ("YES", 1, yes_bid, yes_ask),
                ("NO", 2, no_bid, no_ask),
            ):
                old_price = safe_float(baseline[idx])
                current_price = yes if outcome == "YES" else no
                if old_price <= 0.0 or current_price <= 0.0:
                    continue

                drop = old_price - current_price
                if drop < drop_threshold:
                    continue
                if current_price > max_entry_price:
                    continue

                spread = self._side_spread(bid, ask)
                if spread > max_spread:
                    continue

                target_move = max(min_target_move, drop * min_rebound_fraction)
                target_price = min(0.99, current_price + target_move)
                target_price = min(target_price, old_price - 0.001)
                if target_price <= (current_price + 1e-6):
                    continue

                token_ids = list(getattr(market, "clob_token_ids", []) or [])
                token_id = token_ids[0] if outcome == "YES" and len(token_ids) > 0 else None
                if outcome == "NO" and len(token_ids) > 1:
                    token_id = token_ids[1]

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": current_price,
                        "token_id": token_id,
                        "entry_style": "reversion",
                        "_flash_crash": {
                            "old_price": old_price,
                            "new_price": current_price,
                            "drop": drop,
                            "lookback_seconds": lookback_seconds,
                            "target_price": target_price,
                            "spread": spread,
                        },
                    }
                ]

                opp = self.create_opportunity(
                    title=f"Flash Reversion: {outcome} dip in {market.question[:64]}",
                    description=(
                        f"{outcome} dropped {drop:.3f} ({old_price:.3f}->{current_price:.3f}) "
                        f"over {int(lookback_seconds)}s; targeting rebound to {target_price:.3f}."
                    ),
                    total_cost=current_price,
                    expected_payout=target_price,
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
                liquidity_penalty = 0.0 if liquidity >= 10000.0 else 0.08
                risk_score = 0.66 - min(0.20, drop * 1.25) + min(0.16, spread * 2.5) + liquidity_penalty
                opp.risk_score = clamp(risk_score, 0.35, 0.86)
                opp.risk_factors = [
                    f"Short-window crash magnitude {drop:.1%}",
                    f"Targeting partial rebound ({target_move:.1%})",
                    f"Book spread {spread:.2%}",
                ]
                opp.mispricing_type = MispricingType.WITHIN_MARKET
                candidates.append((drop, opp))

        if not candidates:
            return []

        # Keep strongest crashes first; de-duplicate by (market, outcome).
        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[ArbitrageOpportunity] = []
        seen: set[tuple[str, str]] = set()
        for _, opp in candidates:
            position = (opp.positions_to_take or [{}])[0]
            outcome = str(position.get("outcome") or "")
            market_id = str((opp.markets or [{}])[0].get("id") or "")
            key = (market_id, outcome)
            if key in seen:
                continue
            seen.add(key)
            out.append(opp)
            if len(out) >= max_opportunities:
                break
        return out

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def custom_checks(self, signal: Any, context: dict, params: dict,
                      payload: dict) -> list[DecisionCheck]:
        """Flash crash reversion: source, strategy type, liquidity, crash alignment checks."""
        live_market = context.get("live_market") or {}
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 1500.0), 1500.0))
        min_abs_move_5m = max(0.1, to_float(params.get("min_abs_move_5m", 1.5), 1.5))
        require_alignment = bool(params.get("require_crash_alignment", True))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        strategy_type = str(payload.get("strategy") or payload.get("strategy_type") or "").strip().lower()
        strategy_ok = strategy_type == "flash_crash_reversion"

        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        move_5m_pct = live_move(live_market, "move_5m")

        alignment_ok = True
        if require_alignment:
            if move_5m_pct is None:
                alignment_ok = False
            elif direction == "buy_yes":
                alignment_ok = move_5m_pct <= -min_abs_move_5m
            elif direction == "buy_no":
                alignment_ok = move_5m_pct >= min_abs_move_5m
            else:
                alignment_ok = False

        # Stash for compute_score / evaluate
        payload["_signal_liquidity"] = liquidity
        payload["_move_5m_pct"] = move_5m_pct
        payload["_direction"] = direction
        payload["_strategy_type"] = strategy_type

        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail="Requires source=scanner."),
            DecisionCheck("strategy", "Flash reversion strategy type", strategy_ok, detail="strategy=flash_crash_reversion"),
            DecisionCheck("liquidity", "Liquidity floor", liquidity >= min_liquidity,
                          score=liquidity, detail=f"min={min_liquidity:.0f}"),
            DecisionCheck(
                "alignment_5m",
                "Crash alignment (5m move)",
                alignment_ok,
                score=move_5m_pct,
                detail=f"abs move >= {min_abs_move_5m:.2f}% in signal direction",
            ),
        ]

    def compute_score(self, edge: float, confidence: float, risk_score: float,
                      market_count: int, payload: dict) -> float:
        """Flash crash: edge*0.65 + conf*30 + liq_score*8 - risk*10."""
        liquidity = float(payload.get("_signal_liquidity", 0) or 0)
        return (
            (edge * 0.65)
            + (confidence * 30.0)
            + (min(1.0, liquidity / 10000.0) * 8.0)
            - (risk_score * 10.0)
        )

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        """Flash crash reversion: composable pipeline with Kelly sizing and custom payload."""
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = to_confidence(params.get("min_confidence", 0.40), 0.40)
        max_risk = to_confidence(params.get("max_risk_score", 0.80), 0.80)
        base_size = max(1.0, to_float(params.get("base_size_usd", 16.0), 16.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 130.0), 130.0))
        sizing_policy = str(params.get("sizing_policy", "kelly") or "kelly")
        kelly_fractional_scale = to_float(params.get("kelly_fractional_scale", 0.5), 0.5)

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])

        # Standard checks
        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge,
                          score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck("confidence", "Confidence threshold", confidence >= min_conf,
                          score=confidence, detail=f"min={min_conf:.2f}"),
            DecisionCheck("risk", "Risk ceiling", risk_score <= max_risk,
                          score=risk_score, detail=f"max={max_risk:.2f}"),
        ]

        # Strategy-specific checks (also stashes liquidity/direction/etc in payload)
        checks.extend(self.custom_checks(signal, context, params, payload))

        score = self.compute_score(edge, confidence, risk_score, market_count, payload)

        liquidity = float(payload.get("_signal_liquidity", 0) or 0)
        move_5m_pct = payload.get("_move_5m_pct")
        direction = str(payload.get("_direction", "") or "")
        strategy_type = str(payload.get("_strategy_type", "") or "")

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Flash reversion filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "liquidity": liquidity,
                    "move_5m_pct": move_5m_pct,
                },
            )

        probability = selected_probability(signal, payload, direction)
        entry_price = to_float(getattr(signal, "entry_price", None), 0.0)

        from services.trader_orchestrator.strategies.sizing import compute_position_size
        sizing = compute_position_size(
            base_size_usd=base_size,
            max_size_usd=max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=entry_price if entry_price > 0 else None,
            kelly_fractional_scale=kelly_fractional_scale,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=0.08,
        )

        return StrategyDecision(
            decision="selected",
            reason="Flash reversion signal selected",
            score=score,
            size_usd=float(sizing["size_usd"]),
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "liquidity": liquidity,
                "move_5m_pct": move_5m_pct,
                "sizing": sizing,
                "strategy_type": strategy_type,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Exit on reversion target hit, time decay, or trailing stop."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        ctx = getattr(position, "strategy_context", None) or {}
        current_price = market_state.get("current_price")
        age_minutes = float(getattr(position, "age_minutes", 0) or 0)

        target_price = ctx.get("target_price") or config.get("target_price")
        if target_price and current_price and current_price >= float(target_price):
            return ExitDecision("close", f"Reversion target hit ({current_price:.4f} >= {target_price})", close_price=current_price)

        max_hold = float(config.get("max_hold_minutes", 120) or 120)
        if age_minutes > max_hold:
            return ExitDecision("close", f"Flash reversion time decay ({age_minutes:.0f} > {max_hold:.0f} min)", close_price=current_price)

        return self.default_exit_check(position, market_state)
