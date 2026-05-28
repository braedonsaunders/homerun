"""
Strategy: Temporal Decay

Detects markets whose price has deviated from an expected sqrt-time decay
curve — the assumption being that price should approach 0 (or 1) as the
deadline approaches. When actual price diverges from the expected curve,
buy in the direction of expected mean reversion.

The decay model is heuristic: ``expected_price = initial * (ratio ** 0.5)``
where ``ratio = days_remaining / total_days``. This is a working
hypothesis, not a derived model — it has no theoretical basis in
prediction-market microstructure. Treat outputs accordingly: small sizes,
tight stops, and require a strong deviation before firing.

Was previously a sub-branch of stat_arb. Split out so its (weak)
performance is tracked independently and the ensemble main branch stays
clean.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from typing import Any, Optional

from models import Event, Market, Opportunity
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig, make_aware, utcnow
from services.quality_filter import QualityFilterOverrides

logger = logging.getLogger(__name__)


_DEADLINE_PATTERNS = [
    re.compile(r"\bby\s+(\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})", re.IGNORECASE),
    re.compile(r"\bbefore\s+(\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})", re.IGNORECASE),
    re.compile(r"\bin\s+(\w+\s+\d{4})", re.IGNORECASE),
    re.compile(r"\bby\s+(?:the\s+)?end\s+of\s+(\d{4})", re.IGNORECASE),
]

_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# All numeric tunables live in default_config below — module-level
# constants are reserved for algorithmic structure (regex, month maps).


class TemporalDecayStrategy(BaseStrategy):
    """Bet on mean reversion when price deviates from a sqrt-time decay prior."""

    strategy_type = "temporal_decay"
    name = "Temporal Decay"
    description = "Trade markets where price diverges from an expected decay curve"
    mispricing_type = "within_market"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.5,
        max_resolution_months=2.0,
    )

    default_config = {
        "min_edge_percent": 5.0,
        "min_confidence": 0.50,
        "max_risk_score": 0.70,
        "max_days_to_deadline": 30.0,
        "min_days_to_deadline": 1.0,
        # Decay-curve fit parameters.
        "min_deviation": 0.07,            # only fire on big price-vs-curve gaps
        "min_history_points": 8,          # need a stable baseline
        "decay_rate": 0.5,                # sqrt-time exponent (heuristic, tunable)
        # Entry-side gates: keep contracts off the price extremes where
        # spread/slippage dominate.
        "min_entry_price": 0.10,
        "max_entry_price": 0.90,
        "min_expected_move": 0.04,        # reject if target re-price < this
        # Bundle-level gates passed into create_opportunity.
        "min_liquidity_hard": 2000.0,
        "min_position_size": 50.0,
        # ROI ceiling — anything claiming more than this is almost
        # certainly mispriced model output.
        "max_realistic_roi_pct": 30.0,
        "take_profit_pct": 10.0,
        "stop_loss_pct": 5.0,
        "trailing_stop_pct": 7.0,
        "exclude_market_keywords": [
            "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp",
            "doge", "crypto",
        ],
    }

    pipeline_defaults = {
        "min_edge_percent": 4.0,
        "min_confidence": 0.50,
        "max_risk_score": 0.70,
    }

    scoring_weights = ScoringWeights(
        edge_weight=0.50,
        confidence_weight=28.0,
        risk_penalty=10.0,
    )
    sizing_config = SizingConfig(
        base_divisor=130.0,
        confidence_offset=0.65,
        risk_scale_factor=0.40,
        risk_floor=0.50,
        market_scale_factor=0.0,
        market_scale_cap=0,
    )

    @staticmethod
    def _normalize_excluded_keywords(value: Any) -> list[str]:
        if isinstance(value, str):
            candidates = [token.strip() for token in value.split(",")]
        elif isinstance(value, list):
            candidates = list(value)
        else:
            return []
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
    def _market_text(market: Market) -> str:
        chunks: list[str] = []
        for value in (market.id, market.question, getattr(market, "slug", None),
                      getattr(market, "event_slug", None)):
            text = str(value or "").strip().lower()
            if text:
                chunks.append(text)
        return " | ".join(chunks)

    def _live_yes_price(self, market: Market, prices: dict[str, dict]) -> float:
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            yes_token = market.clob_token_ids[0]
            if yes_token in prices:
                yes_price = prices[yes_token].get("mid", yes_price)
        return yes_price

    def _live_no_price(self, market: Market, prices: dict[str, dict]) -> float:
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            no_token = market.clob_token_ids[1]
            if no_token in prices:
                no_price = prices[no_token].get("mid", no_price)
        return no_price

    def _extract_deadline(self, market: Market) -> Optional[datetime]:
        if market.end_date:
            return make_aware(market.end_date)
        for pattern in _DEADLINE_PATTERNS:
            match = pattern.search(market.question or "")
            if not match:
                continue
            parsed = self._parse_date_string(match.group(1).strip().rstrip(","))
            if parsed is not None:
                return parsed
        return None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        parts = date_str.lower().split()
        if len(parts) == 1:
            try:
                year = int(parts[0])
            except ValueError:
                return None
            return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        if len(parts) >= 2:
            month = _MONTH_MAP.get(parts[0])
            if month is None:
                return None
            if len(parts) == 2:
                try:
                    year = int(parts[1])
                except ValueError:
                    return None
                import calendar
                _, day = calendar.monthrange(year, month)
                return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
            try:
                day = int(parts[1].rstrip(","))
                year = int(parts[2])
            except (IndexError, ValueError):
                return None
            return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
        return None

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------
    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        config = dict(self.default_config)
        config.update(getattr(self, "config", {}) or {})

        excluded = self._normalize_excluded_keywords(config.get("exclude_market_keywords"))
        max_days = max(0.0, float(config.get("max_days_to_deadline", 30.0) or 30.0))
        min_days = max(0.0, float(config.get("min_days_to_deadline", 1.0) or 1.0))
        min_dev = max(0.01, float(config.get("min_deviation", 0.07) or 0.07))
        min_hist = max(3, int(config.get("min_history_points", 8) or 8))
        decay_rate = max(0.05, min(2.0, float(config.get("decay_rate", 0.5) or 0.5)))
        min_entry = float(config.get("min_entry_price", 0.10) or 0.10)
        max_entry = float(config.get("max_entry_price", 0.90) or 0.90)
        min_expected_move = float(config.get("min_expected_move", 0.04) or 0.04)
        liquidity_hard = float(config.get("min_liquidity_hard", 2000.0) or 2000.0)
        position_size = float(config.get("min_position_size", 50.0) or 50.0)
        max_realistic_roi = float(config.get("max_realistic_roi_pct", 30.0) or 30.0)

        price_history = self.state.setdefault("price_history", {})
        market_baselines = self.state.setdefault("market_baselines", {})
        opportunities: list[Opportunity] = []
        now = utcnow()
        scan_time = utcnow().timestamp()

        for market in markets:
            if len(market.outcome_prices) != 2:
                continue
            if market.closed or not market.active:
                continue

            if excluded:
                text = self._market_text(market)
                blocked = False
                for kw in excluded:
                    if kw in text:
                        blocked = True
                        break
                if blocked:
                    continue

            yes_price = self._live_yes_price(market, prices)

            history = price_history.setdefault(market.id, [])
            history.append((scan_time, yes_price))
            if len(history) > 200:
                price_history[market.id] = history[-200:]
                history = price_history[market.id]

            deadline = self._extract_deadline(market)
            if deadline is None:
                continue
            days_remaining = (deadline - now).total_seconds() / 86400.0
            if days_remaining < min_days or days_remaining > max_days:
                continue

            if market.id not in market_baselines:
                market_baselines[market.id] = (deadline, max(yes_price, 0.10))
            else:
                stored_deadline, stored_price = market_baselines[market.id]
                market_baselines[market.id] = (stored_deadline, max(stored_price, yes_price))

            if len(history) < min_hist:
                continue

            first_seen_dt = datetime.fromtimestamp(history[0][0], tz=timezone.utc)
            total_days = max((deadline - first_seen_dt).total_seconds() / 86400.0, 1.0)
            ratio = min(days_remaining / total_days, 1.0)
            initial_price = market_baselines[market.id][1]
            expected_price = initial_price * (ratio ** decay_rate)
            deviation = yes_price - expected_price
            if abs(deviation) < min_dev:
                continue

            no_price = self._live_no_price(market, prices)
            if deviation > 0:
                outcome = "NO"
                entry_price = no_price
                token_id = market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
                direction_desc = "overpriced"
            else:
                outcome = "YES"
                entry_price = yes_price
                token_id = market.clob_token_ids[0] if market.clob_token_ids else None
                direction_desc = "underpriced"

            if entry_price < min_entry or entry_price > max_entry:
                continue

            target_exit_price = max(
                0.01,
                min(0.99, 1.0 - expected_price if deviation > 0 else expected_price),
            )
            expected_move = target_exit_price - entry_price
            if expected_move < min_expected_move:
                continue

            realistic_roi = (expected_move / max(entry_price, 0.001)) * 100.0

            positions = [{
                "action": "BUY",
                "outcome": outcome,
                "market": market.question[:50],
                "price": entry_price,
                "token_id": token_id,
                "rationale": (
                    f"Expected decay price: ${expected_price:.3f}, actual: ${yes_price:.3f} "
                    f"({direction_desc} by {abs(deviation):.3f})"
                ),
            }]

            opp = self.create_opportunity(
                title=f"Temporal Decay: {market.question[:50]}...",
                description=(
                    f"Deadline market {direction_desc} vs expected decay curve. "
                    f"YES actual: ${yes_price:.3f}, expected: ${expected_price:.3f} "
                    f"(deviation: {abs(deviation):.3f}). {days_remaining:.1f} days to deadline. "
                    f"Buy {outcome} at ${entry_price:.3f}, target re-price ${target_exit_price:.3f}."
                ),
                total_cost=entry_price,
                expected_payout=target_exit_price,
                markets=[market],
                positions=positions,
                is_guaranteed=False,
                min_liquidity_hard=liquidity_hard,
                min_position_size=position_size,
                custom_roi_percent=realistic_roi,
            )
            if opp is None or opp.roi_percent > max_realistic_roi:
                continue

            deviation_adjustment = min(abs(deviation) * 1.5, 0.10)
            opp.risk_score = max(0.55, 0.65 - deviation_adjustment)
            # Conviction signal = magnitude of the decay-curve deviation in
            # price-cents (× 100). Naturally bounded by [0, 100].
            conviction_edge = round(abs(deviation) * 100.0, 3)
            try:
                opp.edge_percent = conviction_edge
            except (AttributeError, ValueError):
                pass
            opp.strategy_context["edge_percent"] = conviction_edge
            opp.risk_factors.insert(
                0,
                "DIRECTIONAL BET — decay deviation may reflect new information instead of mispricing.",
            )
            opp.risk_factors.append(f"Statistical edge: decay deviation {abs(deviation):.1%}")
            opp.risk_factors.append(f"Deadline in {days_remaining:.0f} days ({deadline.strftime('%Y-%m-%d')})")
            opp.risk_factors.append(f"Expected repricing target: +${expected_move:.3f} per share")
            if days_remaining < 7:
                opp.risk_factors.append("Near-deadline: steep decay but higher event uncertainty")
            opportunities.append(opp)

        return opportunities

    def custom_checks(self, signal, context, params, payload):
        source = str(getattr(signal, "source", "") or "").strip().lower()
        return [
            DecisionCheck("source", "Signal source", source == "scanner",
                          detail=f"got={source}"),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        config = dict(config)
        for key, fallback in (
            ("take_profit_pct", 10.0),
            ("stop_loss_pct", 5.0),
            ("trailing_stop_pct", 7.0),
        ):
            try:
                default_value = float((getattr(self, "config", None) or {}).get(key, fallback))
            except (TypeError, ValueError):
                default_value = fallback
            config.setdefault(key, default_value)
        position.config = config
        return self.default_exit_check(position, market_state)

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason,
                    getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name,
                    original_size, capped_size, reason)
