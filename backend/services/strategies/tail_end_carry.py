"""Tail-end carry opportunity filter.

Portable pattern adapted from public tail-end Polymarket bots: filter for
high-probability outcomes close to resolution, then only keep entries with
liquidity/spread quality and non-trivial expected repricing room.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import timezone
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import (
    BaseStrategy,
    DecisionCheck,
    StrategyDecision,
    ExitDecision,
    ScoringWeights,
    SizingConfig,
    make_aware,
    utcnow,
)
import logging

from utils.converters import to_float, to_confidence, clamp
from utils.signal_helpers import signal_payload, days_to_resolution, selected_probability
from utils.converters import safe_float
from services.quality_filter import QualityFilterOverrides

logger = logging.getLogger(__name__)


class TailEndCarryStrategy(BaseStrategy):
    """Find near-resolution high-probability outcomes with executable carry."""

    strategy_type = "tail_end_carry"
    name = "Tail-End Carry"
    description = "Near-expiry high-probability carry with liquidity and spread gates"
    mispricing_type = "within_market"
    requires_resolution_date = True
    subscriptions = ["market_data_refresh"]
    realtime_processing_mode = "full_snapshot"

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=1.0,
    )

    pipeline_defaults = {
        "min_edge_percent": 1.0,
        "min_confidence": 0.35,
        "max_risk_score": 0.78,
        "base_size_usd": 14.0,
        "max_size_usd": 90.0,
    }

    # Composable evaluate pipeline: score = edge*0.55 + conf*28 + entry*6 - risk*9 + time_bonus
    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=28.0,
        risk_penalty=9.0,
    )
    # Sizing uses adaptive policy -- override compute_size in evaluate
    sizing_config = SizingConfig()

    default_config = {
        "min_probability": 0.85,
        "max_probability": 0.999,
        "min_upside_percent": 10.0,
        "min_days_to_resolution": 0.01,
        "max_days_to_resolution": 2.0,
        "min_liquidity": 3500.0,
        "max_spread": 0.05,
        "min_repricing_buffer": 0.015,
        "repricing_weight": 0.45,
        "max_opportunities": 120,
        "exclude_market_keywords": "",
        "panic_drop_threshold": 0.08,
        "panic_window_points": 6,
        "panic_recovery_ratio_max": 0.20,
        "take_profit_pct": 10.0,
        "smart_take_profit_enabled": True,
        "smart_take_profit_min_pnl_pct": 10.0,
        "smart_take_profit_max_price_headroom": 0.03,
        "inversion_stop_enabled": True,
        "inversion_price_threshold": 0.50,
        "price_policy": "taker_limit",
        "time_in_force": "IOC",
        "session_timeout_seconds": 90,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)
        self.min_profit = 0.01

    @staticmethod
    def _book_value(payload: Optional[dict], key: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        val = payload.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _extract_side_book(
        self,
        market: Market,
        prices: dict[str, dict],
        outcome: str,
    ) -> tuple[float, Optional[float], Optional[float], Optional[str]]:
        yes = safe_float(getattr(market, "yes_price", 0.0))
        no = safe_float(getattr(market, "no_price", 0.0))
        tokens = list(getattr(market, "clob_token_ids", []) or [])

        if outcome == "YES":
            token_id = tokens[0] if len(tokens) > 0 else None
            payload = prices.get(token_id) if token_id else None
            mid = self._book_value(payload, "mid")
            if mid is None:
                mid = self._book_value(payload, "price")
            price = mid if mid is not None else yes
            bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
            ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
            return price, bid, ask, token_id

        token_id = tokens[1] if len(tokens) > 1 else None
        payload = prices.get(token_id) if token_id else None
        mid = self._book_value(payload, "mid")
        if mid is None:
            mid = self._book_value(payload, "price")
        price = mid if mid is not None else no
        bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
        ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
        return price, bid, ask, token_id

    # ------------------------------------------------------------------
    # Keyword exclusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_text(chunks: list[str], value: Any) -> None:
        text = str(value or "").strip().lower()
        if text:
            chunks.append(text)

    @staticmethod
    def _normalize_excluded_keywords(value: Any) -> list[str]:
        if isinstance(value, str):
            candidates: list[Any] = [token.strip() for token in value.split(",")]
        elif isinstance(value, list):
            candidates = list(value)
        else:
            candidates = []
        out: list[str] = []
        seen: set[str] = set()
        for raw in candidates:
            token = str(raw or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _keyword_in_text(keyword: str, text: str) -> bool:
        if not keyword or not text:
            return False
        if len(keyword) <= 4 and keyword.replace("-", "").replace("_", "").isalnum():
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text

    @classmethod
    def _first_blocked_keyword(cls, text: str, excluded_keywords: list[str]) -> str | None:
        for keyword in excluded_keywords:
            if cls._keyword_in_text(keyword, text):
                return keyword
        return None

    @classmethod
    def _market_text(cls, market: Market, event: Optional[Event]) -> str:
        chunks: list[str] = []
        for value in (market.id, market.question, market.slug, market.group_item_title, market.event_slug):
            cls._append_text(chunks, value)
        for tag in list(getattr(market, "tags", []) or []):
            cls._append_text(chunks, tag)
        if event is not None:
            for value in (event.id, event.slug, event.title, event.category):
                cls._append_text(chunks, value)
            for tag in list(getattr(event, "tags", []) or []):
                cls._append_text(chunks, tag)
        return " | ".join(chunks)

    @classmethod
    def _signal_market_text(cls, signal: Any, payload: dict[str, Any]) -> str:
        chunks: list[str] = []
        for value in (
            getattr(signal, "market_question", None),
            payload.get("market_question"),
            payload.get("title"),
            payload.get("description"),
            payload.get("market_id"),
            payload.get("market_slug"),
            payload.get("event_title"),
            payload.get("event_slug"),
            payload.get("category"),
        ):
            cls._append_text(chunks, value)
        markets = payload.get("markets")
        if isinstance(markets, list):
            for raw_market in markets[:2]:
                if not isinstance(raw_market, dict):
                    continue
                for value in (
                    raw_market.get("id"),
                    raw_market.get("question"),
                    raw_market.get("slug"),
                    raw_market.get("event_slug"),
                    raw_market.get("group_item_title"),
                ):
                    cls._append_text(chunks, value)
                tags = raw_market.get("tags")
                if isinstance(tags, list):
                    for tag in tags:
                        cls._append_text(chunks, tag)
                else:
                    cls._append_text(chunks, tags)
        event = payload.get("event")
        if isinstance(event, dict):
            for value in (event.get("id"), event.get("slug"), event.get("title"), event.get("category")):
                cls._append_text(chunks, value)
            tags = event.get("tags")
            if isinstance(tags, list):
                for tag in tags:
                    cls._append_text(chunks, tag)
            else:
                cls._append_text(chunks, tags)
        return " | ".join(chunks)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        min_probability = clamp(safe_float(cfg.get("min_probability"), 0.85), 0.5, 0.995)
        max_probability = clamp(safe_float(cfg.get("max_probability"), 0.999), min_probability + 0.005, 0.999)
        min_upside_percent = clamp(safe_float(cfg.get("min_upside_percent"), 10.0), 10.0, 100.0)
        min_days = max(0.0, safe_float(cfg.get("min_days_to_resolution"), 0.01))
        max_days = max(min_days + 0.005, safe_float(cfg.get("max_days_to_resolution"), 2.0))
        min_liquidity = max(100.0, safe_float(cfg.get("min_liquidity"), 3500.0))
        max_spread = clamp(safe_float(cfg.get("max_spread"), 0.05), 0.005, 0.20)
        min_repricing_buffer = clamp(safe_float(cfg.get("min_repricing_buffer"), 0.015), 0.005, 0.10)
        repricing_weight = clamp(safe_float(cfg.get("repricing_weight"), 0.45), 0.10, 0.90)
        max_opportunities = max(1, int(safe_float(cfg.get("max_opportunities"), 120)))
        panic_drop_threshold = clamp(safe_float(cfg.get("panic_drop_threshold"), 0.08), 0.02, 0.30)
        panic_window_points = max(3, int(safe_float(cfg.get("panic_window_points"), 6)))
        panic_recovery_ratio_max = clamp(safe_float(cfg.get("panic_recovery_ratio_max"), 0.20), 0.0, 0.9)
        excluded_keywords = self._normalize_excluded_keywords(cfg.get("exclude_market_keywords"))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                event_by_market[event_market.id] = event

        now = utcnow().astimezone(timezone.utc)
        candidates: list[tuple[float, Opportunity]] = []
        history_by_key = self.state.setdefault("tail_carry_price_history", {})

        for market in markets:
            if market.closed or not market.active:
                continue
            if excluded_keywords:
                market_text = self._market_text(market, event_by_market.get(market.id))
                if self._first_blocked_keyword(market_text, excluded_keywords) is not None:
                    continue
            if safe_float(getattr(market, "liquidity", 0.0)) < min_liquidity:
                continue
            if (
                len(list(getattr(market, "clob_token_ids", []) or [])) < 2
                and len(list(getattr(market, "outcome_prices", []) or [])) < 2
            ):
                continue

            end_date = make_aware(getattr(market, "end_date", None))
            if end_date is None:
                continue

            days_to_resolution = (end_date - now).total_seconds() / 86400.0
            if days_to_resolution < min_days or days_to_resolution > max_days:
                continue

            for outcome in ("YES", "NO"):
                price, bid, ask, token_id = self._extract_side_book(market, prices, outcome)
                if not (min_probability <= price <= max_probability):
                    continue
                if price <= 0.0:
                    continue
                max_settlement_upside_pct = ((1.0 - price) / price) * 100.0
                if max_settlement_upside_pct < min_upside_percent:
                    continue

                history_key = f"{market.id}:{outcome}"
                history = history_by_key.get(history_key)
                if not isinstance(history, deque):
                    history = deque(maxlen=max(12, panic_window_points * 2))
                    history_by_key[history_key] = history
                history.append(float(price))
                price_window = list(history)[-panic_window_points:]
                if len(price_window) >= panic_window_points:
                    window_high = max(price_window)
                    window_low = min(price_window)
                    drop_from_high = (window_high - price) / max(window_high, 1e-6)
                    recovery_ratio = (price - window_low) / max(window_high - window_low, 1e-6)
                    if drop_from_high >= panic_drop_threshold and recovery_ratio <= panic_recovery_ratio_max:
                        continue

                spread = 0.0
                if isinstance(bid, float) and isinstance(ask, float) and bid > 0.0 and ask > 0.0:
                    spread = max(0.0, ask - bid)
                    if spread > max_spread:
                        continue

                target_move = max(min_repricing_buffer, (1.0 - price) * repricing_weight)
                target_price = min(0.999, price + target_move)
                if target_price <= (price + 1e-6):
                    continue

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": price,
                        "token_id": token_id,
                        "entry_style": "tail_carry",
                        "_tail_end": {
                            "days_to_resolution": days_to_resolution,
                            "spread": spread,
                            "target_price": target_price,
                            "probability": price,
                            "max_settlement_upside_pct": max_settlement_upside_pct,
                        },
                    }
                ]

                opp = self.create_opportunity(
                    title=f"Tail Carry: {outcome} {price:.1%} into resolution",
                    description=(
                        f"{outcome} at {price:.3f} with {days_to_resolution:.1f} days to resolution; "
                        f"target repricing to {target_price:.3f}."
                    ),
                    total_cost=price,
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

                time_score = 1.0 - (days_to_resolution / max_days)
                black_swan_penalty = clamp((price - 0.90) * 0.40, 0.0, 0.12)
                spread_penalty = min(0.14, spread * 2.8)
                risk = 0.56 - (time_score * 0.10) + black_swan_penalty + spread_penalty
                opp.risk_score = clamp(risk, 0.35, 0.82)
                opp.risk_factors = [
                    f"Near-expiry carry ({days_to_resolution:.1f} days)",
                    f"Selected probability {price:.1%}",
                    "Non-zero tail-risk remains until resolution",
                ]
                opp.mispricing_type = MispricingType.WITHIN_MARKET

                strength = (target_price - price) * (1.0 - opp.risk_score)
                candidates.append((strength, opp))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[Opportunity] = []
        seen: set[tuple[str, str]] = set()
        for _, opp in candidates:
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

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        """Tail carry: source, strategy type, entry band, resolution window, keyword exclusion checks."""
        min_entry = clamp(to_float(params.get("min_entry_price", 0.85), 0.85), 0.01, 0.995)
        max_entry = clamp(to_float(params.get("max_entry_price", 0.999), 0.999), min_entry, 0.999)
        min_upside_percent = clamp(to_float(params.get("min_upside_percent", 10.0), 10.0), 10.0, 100.0)
        min_days = max(0.0, to_float(params.get("min_days_to_resolution", 0.01), 0.01))
        max_days = max(min_days + 0.005, to_float(params.get("max_days_to_resolution", 2.0), 2.0))

        excluded_keywords = self._normalize_excluded_keywords(
            params.get("exclude_market_keywords", self.config.get("exclude_market_keywords", ""))
        )
        keyword_ok = True
        blocked_keyword: str | None = None
        if excluded_keywords:
            signal_text = self._signal_market_text(signal, payload)
            blocked_keyword = self._first_blocked_keyword(signal_text, excluded_keywords)
            keyword_ok = blocked_keyword is None

        source = str(getattr(signal, "source", "") or "").strip().lower()
        strategy_type = (
            str(payload.get("strategy") or payload.get("strategy_type") or getattr(signal, "strategy_type", "") or "")
            .strip()
            .lower()
        )
        strategy_ok = strategy_type == "tail_end_carry"

        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry_price <= 0.0:
            positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
            if positions:
                entry_price = to_float((positions[0] or {}).get("price"), 0.0)

        dtr = days_to_resolution(payload)
        days_ok = dtr is not None and min_days <= dtr <= max_days
        max_settlement_upside_pct = ((1.0 - entry_price) / entry_price * 100.0) if entry_price > 0.0 else 0.0
        upside_ok = max_settlement_upside_pct >= min_upside_percent

        # Stash for compute_score / evaluate
        payload["_entry_price"] = entry_price
        payload["_dtr"] = dtr
        payload["_max_days"] = max_days
        payload["_strategy_type"] = strategy_type
        payload["_max_settlement_upside_pct"] = max_settlement_upside_pct

        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail="Requires source=scanner."),
            DecisionCheck("strategy", "Tail carry strategy type", strategy_ok, detail="strategy=tail_end_carry"),
            DecisionCheck(
                "keyword_exclusion",
                "Market keyword exclusion",
                keyword_ok,
                detail=f"blocked by '{blocked_keyword}'" if blocked_keyword else "no excluded keywords matched",
            ),
            DecisionCheck(
                "entry",
                "Entry probability band",
                min_entry <= entry_price <= max_entry,
                score=entry_price,
                detail=f"[{min_entry:.3f}, {max_entry:.3f}]",
            ),
            DecisionCheck(
                "resolution_window",
                "Resolution window",
                days_ok,
                score=dtr,
                detail=f"[{min_days:.2f}, {max_days:.2f}] days",
            ),
            DecisionCheck(
                "upside",
                "Max settlement upside floor",
                upside_ok,
                score=max_settlement_upside_pct,
                detail=f">= {min_upside_percent:.2f}%",
            ),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        """Tail carry: edge*0.55 + conf*28 + entry*6 - risk*9 + time_bonus."""
        entry_price = float(payload.get("_entry_price", 0) or 0)
        dtr = payload.get("_dtr")
        max_days = float(payload.get("_max_days", 7.0) or 7.0)

        score = (edge * 0.55) + (confidence * 28.0) + (entry_price * 6.0) - (risk_score * 9.0)
        if dtr is not None:
            score += max(0.0, (max_days - dtr) * 0.4)
        return score

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        """Tail carry: composable pipeline with adaptive sizing and custom payload."""
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 1.0), 1.0)
        min_conf = to_confidence(params.get("min_confidence", 0.35), 0.35)
        max_risk = to_confidence(params.get("max_risk_score", 0.78), 0.78)
        base_size = max(1.0, to_float(params.get("base_size_usd", 14.0), 14.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 90.0), 90.0))
        sizing_policy = str(params.get("sizing_policy", "adaptive") or "adaptive")

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])

        # Standard checks
        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk", "Risk ceiling", risk_score <= max_risk, score=risk_score, detail=f"max={max_risk:.2f}"
            ),
        ]

        # Strategy-specific checks (also stashes entry_price/dtr/etc in payload)
        checks.extend(self.custom_checks(signal, context, params, payload))

        score = self.compute_score(edge, confidence, risk_score, market_count, payload)

        entry_price = float(payload.get("_entry_price", 0) or 0)
        dtr = payload.get("_dtr")
        max_settlement_upside_pct = float(payload.get("_max_settlement_upside_pct", 0.0) or 0.0)
        strategy_type = str(payload.get("_strategy_type", "") or "")

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Tail carry filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "entry_price": entry_price,
                    "days_to_resolution": dtr,
                    "max_settlement_upside_pct": max_settlement_upside_pct,
                },
            )

        direction = str(getattr(signal, "direction", "") or "")
        probability = selected_probability(signal, payload, direction)

        from services.trader_orchestrator.strategies.sizing import compute_position_size

        sizing = compute_position_size(
            base_size_usd=base_size,
            max_size_usd=max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=entry_price if entry_price > 0 else None,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=0.06,
        )

        return StrategyDecision(
            decision="selected",
            reason="Tail carry signal selected",
            score=score,
            size_usd=float(sizing["size_usd"]),
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "entry_price": entry_price,
                "days_to_resolution": dtr,
                "max_settlement_upside_pct": max_settlement_upside_pct,
                "sizing": sizing,
                "strategy_type": strategy_type,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Near-expiry carry: hold unless market resolves or fully inverts."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        config = getattr(position, "config", None) or {}
        config = dict(config)

        smart_enabled_raw = config.get("smart_take_profit_enabled")
        if isinstance(smart_enabled_raw, bool):
            smart_enabled = smart_enabled_raw
        elif smart_enabled_raw is None:
            smart_enabled = True
        else:
            smart_enabled = str(smart_enabled_raw).strip().lower() in {"1", "true", "yes", "on"}

        smart_min_pnl_pct = clamp(safe_float(config.get("smart_take_profit_min_pnl_pct"), 10.0), 0.0, 100.0)
        smart_max_headroom = clamp(safe_float(config.get("smart_take_profit_max_price_headroom"), 0.03), 0.001, 0.20)
        inversion_enabled_raw = config.get("inversion_stop_enabled")
        if isinstance(inversion_enabled_raw, bool):
            inversion_stop_enabled = inversion_enabled_raw
        elif inversion_enabled_raw is None:
            inversion_stop_enabled = True
        else:
            inversion_stop_enabled = str(inversion_enabled_raw).strip().lower() in {"1", "true", "yes", "on"}
        inversion_price_threshold = clamp(safe_float(config.get("inversion_price_threshold"), 0.50), 0.05, 0.95)

        entry_price = safe_float(getattr(position, "entry_price", 0.0), 0.0)
        current_price = safe_float(market_state.get("current_price"), 0.0)
        if inversion_stop_enabled and entry_price > 0.0 and current_price > 0.0:
            if current_price <= inversion_price_threshold:
                return ExitDecision(
                    "close",
                    (
                        f"Inversion stop triggered ({current_price:.4f} <= {inversion_price_threshold:.4f}; "
                        f"entry={entry_price:.4f})"
                    ),
                    close_price=current_price,
                )
        if smart_enabled and entry_price > 0.0 and current_price > 0.0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
            headroom = max(0.0, 1.0 - current_price)
            if pnl_pct >= smart_min_pnl_pct and headroom <= smart_max_headroom:
                return ExitDecision(
                    "close",
                    (
                        f"Smart take profit near max ({pnl_pct:.1f}% >= {smart_min_pnl_pct:.1f}%, "
                        f"headroom={headroom:.3f})"
                    ),
                    close_price=current_price,
                )

        return ExitDecision(
            "hold",
            "Tail carry hold — awaiting resolution or full inversion",
            payload={"skip_default_exit": True},
        )

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
