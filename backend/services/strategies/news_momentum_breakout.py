"""News momentum / breakout strategy.

When a Polymarket binary spikes upward on fresh news (e.g. a cabinet
resignation market jumping from 0.20 to 0.40 in minutes), this strategy
enters in the *direction* of the spike and rides it with scale-out
take-profits, a trailing stop, and a breakout-invalidation cutoff.

Distinct from ``flash_crash_reversion`` — the inverse trade. Flash-crash
fades short-window spikes on the thesis they are liquidity events;
this strategy backs the spike on the thesis it is news that has not
yet been fully priced in. The two should not run on the same market at
the same time, which is why both excludes apply (crypto, sports
overlap with ``sports_overreaction_fader``).
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any, Optional

from utils.utcnow import utcnow  # replay-clock-aware "now" (honors backtest sim time)
from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import (
    BaseStrategy,
    DecisionCheck,
    StrategyDecision,
    ExitDecision,
    ScaleOutConfig,
    ScaleOutTarget,
    ScoringWeights,
    SizingConfig,
    _trader_size_limits,
)
import logging

from utils.converters import to_float, to_confidence, clamp, safe_float
from utils.signal_helpers import signal_payload, selected_probability
from services.quality_filter import QualityFilterOverrides
from services.strategies.reversion_helpers import (
    breakout_shape_ok,
    direction_aligns_impulse,
    market_move_pct,
)

logger = logging.getLogger(__name__)


class NewsMomentumBreakoutStrategy(BaseStrategy):
    """Detect short-window upside breakouts and ride them with scale-outs."""

    strategy_type = "news_momentum_breakout"
    name = "News Momentum Breakout"
    description = (
        "Enter on fresh upward spikes aligned with a building trend; "
        "ride with scale-outs and a breakout-invalidation stop"
    )
    mispricing_type = "within_market"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
        max_resolution_months=6.0,
    )

    requires_historical_prices = True

    pipeline_defaults = {
        "min_edge_percent": 4.0,
        "min_confidence": 0.40,
        "max_risk_score": 0.78,
    }

    scoring_weights = ScoringWeights(
        edge_weight=0.65,
        confidence_weight=28.0,
        risk_penalty=12.0,
        liquidity_weight=10.0,
        liquidity_divisor=10000.0,
    )
    sizing_config = SizingConfig()

    scale_out_config = ScaleOutConfig(
        targets=[
            ScaleOutTarget(trigger_bps=30.0, exit_fraction=0.33),
            ScaleOutTarget(trigger_bps=60.0, exit_fraction=0.50),
        ],
        trailing_stop_bps=120.0,
        near_resolution_exit=True,
        near_resolution_hours=6.0,
        near_resolution_spread_widen_bps=80.0,
    )

    default_config = {
        # Detection window
        "lookback_seconds": 300.0,
        "stale_history_seconds": 1800.0,
        "breakout_threshold": 0.10,
        "min_target_move": 0.04,
        "target_distance_to_one_fraction": 0.55,
        # Entry-price band
        "min_entry_price": 0.18,
        "max_entry_price": 0.78,
        "max_spread": 0.06,
        "min_liquidity": 3000.0,
        # Anti-chase guards
        "max_retrace_from_peak_fraction": 0.35,
        "min_5m_share_of_30m": 0.45,
        "max_abs_move_2h_pct": 80.0,
        "require_breakout_shape": True,
        "require_breakout_alignment": True,
        "min_abs_move_5m": 4.0,
        "emit_cooldown_seconds": 300.0,
        "max_opportunities": 25,
        # Market-scope filters
        "exclude_crypto_markets": True,
        "exclude_sports_markets": True,
        "exclude_market_keywords": [],
        # Sizing
        "sizing_policy": "kelly",
        "kelly_fractional_scale": 0.4,
        # Exit
        "take_profit_pct": 70.0,
        "stop_loss_pct": 25.0,
        "stop_loss_policy": "always",
        "trailing_stop_pct": 18.0,
        "trailing_stop_activation_profit_pct": 25.0,
        "max_hold_minutes": 240.0,
        "min_hold_minutes": 1.0,
        "momentum_stall_minutes": 45.0,
    }

    _CRYPTO_MARKET_HINTS = (
        "btc",
        "bitcoin",
        "eth",
        "ethereum",
        "sol",
        "solana",
        "xrp",
        "doge",
        "crypto",
        "cryptocurrency",
        "coinbase",
    )
    _SPORTS_MARKET_HINTS = (
        "nba",
        "nfl",
        "nhl",
        "mlb",
        "epl",
        "ucl",
        "ufc",
        "boxing",
        "f1",
        "formula 1",
        "tennis",
        "soccer",
        "football",
        "basketball",
        "baseball",
        "hockey",
        "match",
        "fixture",
    )

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

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
    def _is_crypto_market_text(cls, text: str) -> bool:
        return any(cls._keyword_in_text(hint, text) for hint in cls._CRYPTO_MARKET_HINTS)

    @classmethod
    def _is_sports_market_text(cls, text: str) -> bool:
        return any(cls._keyword_in_text(hint, text) for hint in cls._SPORTS_MARKET_HINTS)

    @classmethod
    def _first_blocked_keyword(cls, text: str, excluded_keywords: list[str]) -> str | None:
        for keyword in excluded_keywords:
            if cls._keyword_in_text(keyword, text):
                return keyword
        return None

    @classmethod
    def _detect_market_text(cls, market: Market, event: Optional[Event]) -> str:
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
    ) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        lookback_seconds = max(60.0, safe_float(cfg.get("lookback_seconds"), 300.0))
        stale_history_seconds = max(lookback_seconds * 2.0, safe_float(cfg.get("stale_history_seconds"), 1800.0))
        breakout_threshold = clamp(safe_float(cfg.get("breakout_threshold"), 0.10), 0.02, 0.50)
        min_target_move = clamp(safe_float(cfg.get("min_target_move"), 0.04), 0.005, 0.30)
        target_distance_to_one_fraction = clamp(
            safe_float(cfg.get("target_distance_to_one_fraction"), 0.55), 0.10, 0.95
        )
        min_entry_price = clamp(safe_float(cfg.get("min_entry_price"), 0.18), 0.05, 0.50)
        max_entry_price = clamp(safe_float(cfg.get("max_entry_price"), 0.78), 0.50, 0.95)
        max_spread = clamp(safe_float(cfg.get("max_spread"), 0.06), 0.005, 0.25)
        min_liquidity = max(100.0, safe_float(cfg.get("min_liquidity"), 3000.0))
        max_retrace = clamp(safe_float(cfg.get("max_retrace_from_peak_fraction"), 0.35), 0.05, 0.90)
        emit_cooldown = max(0.0, safe_float(cfg.get("emit_cooldown_seconds"), 300.0))
        max_opportunities = max(1, int(safe_float(cfg.get("max_opportunities"), 25)))
        exclude_crypto = self._to_bool(cfg.get("exclude_crypto_markets"), True)
        exclude_sports = self._to_bool(cfg.get("exclude_sports_markets"), True)
        exclude_market_keywords = self._normalize_excluded_keywords(cfg.get("exclude_market_keywords"))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                event_by_market[event_market.id] = event

        now = utcnow().timestamp()
        candidates: list[tuple[float, Opportunity]] = []
        last_emit: dict[tuple[str, str], float] = self.state.setdefault("last_emit", {})

        for market in markets:
            if market.closed or not market.active:
                continue
            event = event_by_market.get(market.id)
            market_text = self._detect_market_text(market, event)
            if exclude_crypto and self._is_crypto_market_text(market_text):
                continue
            if exclude_sports and self._is_sports_market_text(market_text):
                continue
            blocked_keyword = self._first_blocked_keyword(market_text, exclude_market_keywords)
            if blocked_keyword is not None:
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
            all_history = self.state.setdefault("price_history", {})
            if market.id not in all_history:
                all_history[market.id] = deque(maxlen=240)
            history = all_history[market.id]
            history.append(row)

            while history and (now - history[0][0]) > stale_history_seconds:
                history.popleft()

            if len(history) < 2:
                continue

            cutoff = now - lookback_seconds
            baseline = None
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

                rise = current_price - old_price
                if rise < breakout_threshold:
                    continue
                if current_price < min_entry_price:
                    continue
                if current_price > max_entry_price:
                    continue

                key = (str(market.id), outcome)
                last_ts = last_emit.get(key)
                if last_ts is not None and (now - last_ts) < emit_cooldown:
                    continue

                spread = self._side_spread(bid, ask)
                if spread > max_spread:
                    continue

                peak = current_price
                for point in history:
                    if point[0] < cutoff:
                        continue
                    candidate_peak = safe_float(point[idx])
                    if candidate_peak > peak:
                        peak = candidate_peak
                if peak > old_price:
                    retraced = (peak - current_price) / max(1e-6, peak - old_price)
                    if retraced > max_retrace:
                        continue

                distance_to_one = max(0.0, 1.0 - current_price)
                target_move = max(min_target_move, distance_to_one * target_distance_to_one_fraction)
                target_price = min(0.985, current_price + target_move)
                if target_price <= (current_price + 1e-6):
                    continue

                token_ids = list(getattr(market, "clob_token_ids", []) or [])
                token_id = None
                if outcome == "YES" and len(token_ids) > 0:
                    token_id = token_ids[0]
                elif outcome == "NO" and len(token_ids) > 1:
                    token_id = token_ids[1]

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": current_price,
                        "token_id": token_id,
                        "entry_style": "breakout",
                        "_breakout": {
                            "old_price": old_price,
                            "new_price": current_price,
                            "peak_price": peak,
                            "rise": rise,
                            "lookback_seconds": lookback_seconds,
                            "target_price": target_price,
                            "spread": spread,
                        },
                    }
                ]

                opp = self.create_opportunity(
                    title=f"News Momentum: {outcome} breakout in {market.question[:64]}",
                    description=(
                        f"{outcome} surged {rise:.3f} ({old_price:.3f}->{current_price:.3f}) "
                        f"over {int(lookback_seconds)}s; targeting {target_price:.3f}."
                    ),
                    total_cost=current_price,
                    expected_payout=target_price,
                    markets=[market],
                    positions=positions,
                    event=event,
                    is_guaranteed=False,
                    min_liquidity_hard=min_liquidity,
                    min_position_size=max(settings.MIN_POSITION_SIZE, 5.0),
                )
                if not opp:
                    continue

                liquidity = safe_float(getattr(market, "liquidity", 0.0))
                liquidity_penalty = 0.0 if liquidity >= 10000.0 else 0.10
                price_penalty = max(0.0, (current_price - 0.5) * 0.20)
                risk_score = (
                    0.62
                    - min(0.18, rise * 1.0)
                    + min(0.16, spread * 2.5)
                    + liquidity_penalty
                    + price_penalty
                )
                opp.risk_score = clamp(risk_score, 0.30, 0.85)
                opp.risk_factors = [
                    f"Short-window breakout magnitude {rise:.1%}",
                    f"Targeting +{target_move:.1%} run to {target_price:.3f}",
                    f"Book spread {spread:.2%}",
                    f"Entry price {current_price:.3f} (room to 1.0: {distance_to_one:.2f})",
                ]
                opp.mispricing_type = MispricingType.WITHIN_MARKET
                last_emit[key] = now
                candidates.append((rise, opp))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[Opportunity] = []
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

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        live_market = context.get("live_market") or {}
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 1500.0), 1500.0))
        min_abs_move_5m = max(0.0, to_float(params.get("min_abs_move_5m", 4.0), 4.0))
        max_abs_move_2h = max(5.0, to_float(params.get("max_abs_move_2h_pct", 80.0), 80.0))
        require_alignment = self._to_bool(params.get("require_breakout_alignment"), True)
        require_shape = self._to_bool(params.get("require_breakout_shape"), True)
        min_5m_share_of_30m = clamp(to_float(params.get("min_5m_share_of_30m", 0.45), 0.45), 0.0, 1.0)
        exclude_crypto = self._to_bool(params.get("exclude_crypto_markets"), True)
        exclude_sports = self._to_bool(params.get("exclude_sports_markets"), True)
        exclude_market_keywords = self._normalize_excluded_keywords(params.get("exclude_market_keywords"))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        strategy_type = (
            str(payload.get("strategy") or payload.get("strategy_type") or getattr(signal, "strategy_type", "") or "")
            .strip()
            .lower()
        )
        strategy_ok = strategy_type == "news_momentum_breakout"

        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        move_5m_pct = market_move_pct(live_market, payload, "move_5m")
        move_30m_pct = market_move_pct(live_market, payload, "move_30m")
        move_2h_pct = market_move_pct(live_market, payload, "move_2h")

        if move_5m_pct is None:
            positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
            first_position = positions[0] if positions and isinstance(positions[0], dict) else {}
            breakout = first_position.get("_breakout") if isinstance(first_position.get("_breakout"), dict) else {}
            old_price = to_float(breakout.get("old_price"), None)
            new_price = to_float(breakout.get("new_price"), None)
            if old_price is not None and old_price > 0.0 and new_price is not None and new_price > 0.0:
                selected_leg_move_pct = ((new_price - old_price) / old_price) * 100.0
                if direction == "buy_yes":
                    move_5m_pct = selected_leg_move_pct
                elif direction == "buy_no":
                    move_5m_pct = -selected_leg_move_pct

        alignment_ok = True
        if require_alignment:
            alignment_ok = direction_aligns_impulse(direction, move_5m_pct, min_abs_move_5m)

        shape_ok = breakout_shape_ok(
            move_5m_pct,
            move_30m_pct,
            move_2h_pct,
            require_shape=require_shape,
            max_abs_move_2h=max_abs_move_2h,
            min_5m_share_of_30m=min_5m_share_of_30m,
        )

        market_text = self._signal_market_text(signal, payload)
        is_crypto_market = self._is_crypto_market_text(market_text)
        is_sports_market = self._is_sports_market_text(market_text)
        blocked_keyword = self._first_blocked_keyword(market_text, exclude_market_keywords)
        market_scope_ok = (not exclude_crypto or not is_crypto_market) and (
            not exclude_sports or not is_sports_market
        )
        keyword_filter_ok = blocked_keyword is None

        payload["_signal_liquidity"] = liquidity
        payload["_move_5m_pct"] = move_5m_pct
        payload["_move_30m_pct"] = move_30m_pct
        payload["_move_2h_pct"] = move_2h_pct
        payload["_direction"] = direction
        payload["_strategy_type"] = strategy_type
        payload["_is_crypto_market"] = is_crypto_market
        payload["_is_sports_market"] = is_sports_market
        payload["_blocked_keyword"] = blocked_keyword

        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail="Requires source=scanner."),
            DecisionCheck(
                "strategy",
                "Momentum breakout strategy type",
                strategy_ok,
                detail="strategy=news_momentum_breakout",
            ),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity >= min_liquidity,
                score=liquidity,
                detail=f"min={min_liquidity:.0f}",
            ),
            DecisionCheck(
                "alignment_5m",
                "Breakout alignment (5m move)",
                alignment_ok,
                score=move_5m_pct,
                detail=f"|5m| >= {min_abs_move_5m:.2f}% in signal direction",
            ),
            DecisionCheck(
                "breakout_shape",
                "Fresh breakout shape (5m/30m/2h)",
                shape_ok,
                detail=f"5m share >= {min_5m_share_of_30m:.2f}, |2h| <= {max_abs_move_2h:.0f}%",
            ),
            DecisionCheck(
                "market_scope",
                "Exclude crypto/sports markets",
                market_scope_ok,
                detail=f"crypto={exclude_crypto},sports={exclude_sports}",
            ),
            DecisionCheck(
                "keyword_filter",
                "Exclude keyword matches",
                keyword_filter_ok,
                detail=f"blocked={blocked_keyword}" if blocked_keyword else "no blocked keyword match",
            ),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        liquidity = float(payload.get("_signal_liquidity", 0) or 0)
        return (edge * 0.65) + (confidence * 28.0) + (min(1.0, liquidity / 10000.0) * 10.0) - (risk_score * 12.0)

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 4.0), 4.0)
        min_conf = to_confidence(params.get("min_confidence", 0.40), 0.40)
        max_risk = to_confidence(params.get("max_risk_score", 0.78), 0.78)
        base_size, max_size = _trader_size_limits(context)
        sizing_policy = str(params.get("sizing_policy", "kelly") or "kelly")
        kelly_fractional_scale = to_float(params.get("kelly_fractional_scale", 0.4), 0.4)

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])

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
        checks.extend(self.custom_checks(signal, context, params, payload))

        score = self.compute_score(edge, confidence, risk_score, market_count, payload)

        liquidity = float(payload.get("_signal_liquidity", 0) or 0)
        move_5m_pct = payload.get("_move_5m_pct")
        move_30m_pct = payload.get("_move_30m_pct")
        move_2h_pct = payload.get("_move_2h_pct")
        direction = str(payload.get("_direction", "") or "")
        strategy_type = str(payload.get("_strategy_type", "") or "")
        is_crypto_market = bool(payload.get("_is_crypto_market", False))
        is_sports_market = bool(payload.get("_is_sports_market", False))
        blocked_keyword = payload.get("_blocked_keyword")

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Momentum breakout filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "liquidity": liquidity,
                    "move_5m_pct": move_5m_pct,
                    "move_30m_pct": move_30m_pct,
                    "move_2h_pct": move_2h_pct,
                    "is_crypto_market": is_crypto_market,
                    "is_sports_market": is_sports_market,
                    "blocked_keyword": blocked_keyword,
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
            liquidity_cap_fraction=0.06,
        )

        return StrategyDecision(
            decision="selected",
            reason="Momentum breakout signal selected",
            score=score,
            size_usd=float(sizing["size_usd"]),
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "liquidity": liquidity,
                "move_5m_pct": move_5m_pct,
                "move_30m_pct": move_30m_pct,
                "move_2h_pct": move_2h_pct,
                "is_crypto_market": is_crypto_market,
                "is_sports_market": is_sports_market,
                "blocked_keyword": blocked_keyword,
                "sizing": sizing,
                "strategy_type": strategy_type,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        config = getattr(position, "config", None) or {}
        config = dict(config)
        defaults = getattr(self, "config", None) or {}
        for key, fallback in (
            ("take_profit_pct", 70.0),
            ("stop_loss_pct", 25.0),
            ("trailing_stop_pct", 18.0),
            ("trailing_stop_activation_profit_pct", 25.0),
            ("max_hold_minutes", 240.0),
            ("min_hold_minutes", 1.0),
            ("momentum_stall_minutes", 45.0),
        ):
            try:
                config.setdefault(key, float(defaults.get(key, fallback)))
            except (TypeError, ValueError):
                config.setdefault(key, fallback)
        config.setdefault("stop_loss_policy", str(defaults.get("stop_loss_policy", "always")))
        position.config = config

        ctx = getattr(position, "strategy_context", None)
        if not isinstance(ctx, dict):
            ctx = {}
            position.strategy_context = ctx

        current_price = market_state.get("current_price")
        entry_price = float(getattr(position, "entry_price", 0) or 0)
        age_minutes = float(getattr(position, "age_minutes", 0) or 0)

        if current_price is not None and entry_price > 0.0:
            stall_minutes = float(config.get("momentum_stall_minutes") or 0.0)
            if stall_minutes > 0:
                tracked_high = float(ctx.get("_tracked_high", entry_price) or entry_price)
                tracked_high_age = float(ctx.get("_tracked_high_age", age_minutes) or age_minutes)
                if current_price > tracked_high:
                    ctx["_tracked_high"] = current_price
                    ctx["_tracked_high_age"] = age_minutes
                else:
                    stall_age = age_minutes - tracked_high_age
                    if stall_age >= stall_minutes and current_price <= entry_price:
                        return ExitDecision(
                            "close",
                            f"Momentum stalled ({stall_age:.0f}min since new high, no follow-through)",
                            close_price=current_price,
                        )

        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)

    # ------------------------------------------------------------------
    # Phase 3: declarative pre-submit gates
    # ------------------------------------------------------------------
    #
    # Cheap (L0_MEMORY) subset of custom_checks: liquidity floor, entry
    # band, exclude-crypto/sports/keyword scoping.  Inline checks remain
    # — these gates short-circuit earlier in the pipeline only.

    def get_pre_submit_gates(self):
        from services.strategy_sdk import Gate, CostClass, GateContext, GateResult

        def _payload(ctx: GateContext) -> dict:
            payload = (ctx.extras or {}).get("signal_payload")
            if isinstance(payload, dict):
                return payload
            payload = getattr(ctx.runtime_signal, "payload_json", None)
            return payload if isinstance(payload, dict) else {}

        def _merged_params(ctx: GateContext) -> dict:
            merged = dict(self.default_config or {})
            merged.update(ctx.strategy_params or {})
            return merged

        def liquidity_floor_predicate(ctx: GateContext) -> GateResult:
            cfg = _merged_params(ctx)
            min_liq = float(safe_float(cfg.get("min_liquidity"), 0.0) or 0.0)
            if min_liq <= 0.0:
                return GateResult(passed=True)
            liq = safe_float(getattr(ctx.runtime_signal, "liquidity", None), None)
            if liq is None:
                liq = safe_float(_payload(ctx).get("liquidity"), None)
            if liq is None:
                return GateResult(passed=True)
            if float(liq) < min_liq:
                return GateResult(
                    passed=False,
                    reason="news_momentum_min_liquidity",
                    error_message=f"Liquidity {liq:.0f} < min {min_liq:.0f}",
                    detail={"liquidity": float(liq), "min_liquidity": min_liq},
                )
            return GateResult(passed=True)

        def market_scope_predicate(ctx: GateContext) -> GateResult:
            payload = _payload(ctx)
            cfg = _merged_params(ctx)
            exclude_crypto = self._to_bool(cfg.get("exclude_crypto_markets"), True)
            exclude_sports = self._to_bool(cfg.get("exclude_sports_markets"), True)
            excluded_keywords = self._normalize_excluded_keywords(
                cfg.get("exclude_market_keywords")
            )
            market_text = self._signal_market_text(ctx.runtime_signal, payload)
            if exclude_crypto and self._is_crypto_market_text(market_text):
                return GateResult(
                    passed=False,
                    reason="news_momentum_market_scope",
                    error_message="Crypto market excluded by news_momentum scope.",
                    detail={"market_scope": "crypto"},
                )
            if exclude_sports and self._is_sports_market_text(market_text):
                return GateResult(
                    passed=False,
                    reason="news_momentum_market_scope",
                    error_message="Sports market excluded by news_momentum scope.",
                    detail={"market_scope": "sports"},
                )
            blocked = self._first_blocked_keyword(market_text, excluded_keywords)
            if blocked:
                return GateResult(
                    passed=False,
                    reason="news_momentum_excluded_keyword",
                    error_message=f"Market contains excluded keyword: {blocked!r}",
                    detail={"keyword": blocked},
                )
            return GateResult(passed=True)

        def entry_band_predicate(ctx: GateContext) -> GateResult:
            cfg = _merged_params(ctx)
            min_entry = float(safe_float(cfg.get("min_entry_price"), 0.0) or 0.0)
            max_entry = float(safe_float(cfg.get("max_entry_price"), 1.0) or 1.0)
            leg = ctx.leg if isinstance(ctx.leg, dict) else {}
            limit_price = safe_float(leg.get("limit_price"), None)
            if limit_price is None:
                payload = _payload(ctx)
                positions = payload.get("positions_to_take")
                if isinstance(positions, list) and positions:
                    first = positions[0]
                    if isinstance(first, dict):
                        limit_price = safe_float(first.get("price"), None)
            if limit_price is None or float(limit_price) <= 0.0:
                return GateResult(passed=True)
            price = float(limit_price)
            if price < min_entry or price > max_entry:
                return GateResult(
                    passed=False,
                    reason="news_momentum_entry_band",
                    error_message=(
                        f"Entry price {price:.3f} outside band "
                        f"[{min_entry:.3f}, {max_entry:.3f}]"
                    ),
                    detail={
                        "entry_price": price,
                        "min_entry_price": min_entry,
                        "max_entry_price": max_entry,
                    },
                )
            return GateResult(passed=True)

        return [
            Gate(
                name="news_momentum_market_scope",
                cost_class=CostClass.L0_MEMORY,
                predicate=market_scope_predicate,
            ),
            Gate(
                name="news_momentum_min_liquidity",
                cost_class=CostClass.L0_MEMORY,
                predicate=liquidity_floor_predicate,
            ),
            Gate(
                name="news_momentum_entry_band",
                cost_class=CostClass.L0_MEMORY,
                predicate=entry_band_predicate,
            ),
        ]
