"""Standalone BTC 5m tail-end sniper strategy.

Detects opportunities directly from ``crypto_update`` worker payloads near the
final seconds of the market window and does not depend on other crypto
strategies for signal generation.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from models import Market, Opportunity
from services.data_events import DataEvent, EventType
from services.quality_filter import QualityFilterOverrides
from services.strategies.base import BaseStrategy, DecisionCheck, ExitDecision, StrategyDecision
from services.trader_orchestrator.strategies.sizing import compute_position_size
from utils.converters import clamp, safe_float, to_confidence, to_float
from utils.signal_helpers import selected_probability, signal_payload

logger = logging.getLogger(__name__)


def _parse_datetime_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_timeframe(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"5m", "5min", "5-minute", "5minutes", "five", "fivemin", "fiveminute"}:
        return "5min"
    if text in {"15m", "15min", "15-minute", "15minutes", "fifteen", "fifteenmin"}:
        return "15min"
    if text in {"1h", "1hr", "1hour", "60m"}:
        return "1hr"
    if text in {"4h", "4hr", "4hour", "240m"}:
        return "4hr"
    return text


def _timeframe_seconds(value: Any) -> int:
    tf = _normalize_timeframe(value)
    if tf == "5min":
        return 300
    if tf == "15min":
        return 900
    if tf == "1hr":
        return 3600
    if tf == "4hr":
        return 14400
    return 300


def _taker_fee_pct(entry_price: float) -> float:
    price = clamp(float(entry_price), 0.0001, 0.9999)
    return 0.25 * ((price * (1.0 - price)) ** 2)


def _sigmoid(z: float) -> float:
    bounded = clamp(float(z), -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-bounded))


class CryptoMicroSniperStrategy(BaseStrategy):
    strategy_type = "crypto_micro_sniper"
    name = "Crypto Micro Sniper"
    description = "Standalone BTC 5m tail-end sniper driven by price-to-beat/oracle divergence"
    mispricing_type = "within_market"
    source_key = "crypto"
    worker_affinity = "crypto"
    market_categories = ["crypto"]
    subscriptions = [EventType.CRYPTO_UPDATE]
    requires_live_market_context = True
    supports_entry_take_profit_exit = True
    default_open_order_timeout_seconds = 20.0

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.4,
        max_resolution_months=0.1,
    )

    default_config = {
        "asset": "BTC",
        "timeframe": "5min",
        "entry_window_start_seconds": 110.0,
        "entry_window_end_seconds": 12.0,
        "target_entry_seconds": 60.0,
        "window_tolerance_seconds": 45.0,
        "min_oracle_move_pct": 0.18,
        "max_oracle_age_seconds": 8.0,
        "expected_prob_scale_pct": 0.28,
        "time_pressure_weight": 0.75,
        "min_repricing_buffer": 0.015,
        "min_net_edge_percent": 0.35,
        "min_confidence": 0.50,
        "max_entry_price": 0.90,
        "min_liquidity_usd": 2500.0,
        "max_spread_pct": 0.03,
        "slippage_spread_multiplier": 0.22,
        "min_slippage_pct": 0.08,
        "max_markets_per_event": 24,
        "min_order_size_usd": 2.0,
        "base_size_usd": 24.0,
        "max_size_usd": 180.0,
        "sizing_policy": "kelly",
        "kelly_fractional_scale": 0.35,
        "take_profit_pct": 7.0,
        "stop_loss_pct": 3.8,
        "max_hold_minutes": 6.0,
        "liquidity_cap_fraction": 0.07,
        "min_entry_seconds_5m": 12.0,
        "max_entry_seconds_5m": 110.0,
        "min_entry_seconds_15m": 45.0,
        "max_entry_seconds_15m": 180.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.min_profit = 0.0

    @staticmethod
    def _row_market(row: dict[str, Any]) -> Market | None:
        market_id = str(row.get("condition_id") or row.get("id") or "").strip()
        if not market_id:
            return None

        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        if up_price is None or down_price is None:
            return None

        end_date = _parse_datetime_utc(row.get("end_time"))
        token_ids = [
            str(token).strip()
            for token in list(row.get("clob_token_ids") or [])
            if str(token).strip() and len(str(token).strip()) > 20
        ]

        return Market(
            id=market_id,
            condition_id=market_id,
            question=str(row.get("question") or row.get("slug") or market_id),
            slug=str(row.get("slug") or market_id),
            outcome_prices=[float(up_price), float(down_price)],
            liquidity=max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0)),
            end_date=end_date,
            platform="polymarket",
            clob_token_ids=token_ids,
        )

    @staticmethod
    def _seconds_left(row: dict[str, Any], timeframe_value: Any) -> float:
        seconds_left = safe_float(row.get("seconds_left"), None)
        if seconds_left is not None and seconds_left >= 0.0:
            return float(seconds_left)
        end_date = _parse_datetime_utc(row.get("end_time"))
        if end_date is not None:
            return max(0.0, (end_date - datetime.now(timezone.utc)).total_seconds())
        return float(_timeframe_seconds(timeframe_value))

    @staticmethod
    def _spread_pct(row: dict[str, Any]) -> float:
        spread = safe_float(row.get("spread"), None)
        if spread is None or spread < 0.0:
            best_bid = safe_float(row.get("best_bid"), None)
            best_ask = safe_float(row.get("best_ask"), None)
            if best_bid is not None and best_ask is not None and best_ask >= best_bid:
                spread = best_ask - best_bid
        if spread is None:
            spread = 0.0
        return clamp(float(spread), 0.0, 1.0)

    @staticmethod
    def _adaptive_min_oracle_move_pct(
        base_min_oracle_move_pct: float,
        *,
        seconds_left: float | None,
        window_start_seconds: float,
    ) -> float:
        """Tight at window open, relaxed near expiry to avoid starvation in quiet tapes."""
        base = max(0.01, float(base_min_oracle_move_pct))
        if seconds_left is None:
            return base
        window_start = max(1.0, float(window_start_seconds))
        time_pressure = clamp(1.0 - (max(0.0, float(seconds_left)) / window_start), 0.0, 1.0)
        return max(0.01, base * (1.0 - time_pressure))

    def _score_market(self, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
        asset = str(row.get("asset") or "").strip().upper()
        target_asset = str(cfg.get("asset", "BTC") or "BTC").strip().upper()
        if asset != target_asset:
            return None

        timeframe = _normalize_timeframe(row.get("timeframe"))
        target_timeframe = _normalize_timeframe(cfg.get("timeframe", "5min"))
        if timeframe != target_timeframe:
            return None

        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        oracle_price = safe_float(row.get("oracle_price"), None)
        price_to_beat = safe_float(row.get("price_to_beat"), None)
        if up_price is None or down_price is None or oracle_price is None or price_to_beat is None:
            return None
        if not (0.0 < up_price < 1.0 and 0.0 < down_price < 1.0 and price_to_beat > 0.0):
            return None

        oracle_age_seconds = safe_float(row.get("oracle_age_seconds"), None)
        max_oracle_age_seconds = max(0.1, to_float(cfg.get("max_oracle_age_seconds", 8.0), 8.0))
        if oracle_age_seconds is None or oracle_age_seconds > max_oracle_age_seconds:
            return None

        timeframe_seconds = _timeframe_seconds(timeframe)
        seconds_left = self._seconds_left(row, timeframe)
        window_start = max(1.0, to_float(cfg.get("entry_window_start_seconds", 110.0), 110.0))
        window_end = max(1.0, to_float(cfg.get("entry_window_end_seconds", 12.0), 12.0))
        if window_end > window_start:
            window_end = window_start
        if seconds_left > window_start or seconds_left < window_end:
            return None

        target_entry_seconds = clamp(
            to_float(cfg.get("target_entry_seconds", 60.0), 60.0),
            window_end,
            window_start,
        )
        tolerance_seconds = max(1.0, to_float(cfg.get("window_tolerance_seconds", 45.0), 45.0))
        tail_alignment = clamp(1.0 - (abs(seconds_left - target_entry_seconds) / tolerance_seconds), 0.0, 1.0)
        if tail_alignment <= 0.0:
            return None

        liquidity_usd = max(0.0, safe_float(row.get("liquidity"), 0.0) or 0.0)
        min_liquidity_usd = max(0.0, to_float(cfg.get("min_liquidity_usd", 2500.0), 2500.0))
        if liquidity_usd < min_liquidity_usd:
            return None

        spread_pct = self._spread_pct(row)
        max_spread_pct = clamp(to_float(cfg.get("max_spread_pct", 0.03), 0.03), 0.001, 0.2)
        if spread_pct > max_spread_pct:
            return None

        diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0
        min_oracle_move_pct = max(0.01, to_float(cfg.get("min_oracle_move_pct", 0.18), 0.18))
        adaptive_min_oracle_move_pct = self._adaptive_min_oracle_move_pct(
            min_oracle_move_pct,
            seconds_left=seconds_left,
            window_start_seconds=window_start,
        )
        if abs(diff_pct) < adaptive_min_oracle_move_pct:
            return None

        if diff_pct > 0.0:
            direction = "buy_yes"
            outcome = "YES"
            selected_price = float(up_price)
            model_side_prob_key = "model_up"
        else:
            direction = "buy_no"
            outcome = "NO"
            selected_price = float(down_price)
            model_side_prob_key = "model_down"

        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.90), 0.90), 0.05, 0.99)
        if selected_price <= 0.0 or selected_price >= 1.0 or selected_price > max_entry_price:
            return None

        elapsed_ratio = clamp(1.0 - (seconds_left / float(max(1, timeframe_seconds))), 0.0, 1.0)
        base_scale_pct = max(0.05, to_float(cfg.get("expected_prob_scale_pct", 0.28), 0.28))
        time_pressure_weight = clamp(to_float(cfg.get("time_pressure_weight", 0.75), 0.75), 0.0, 1.0)
        scale_multiplier = max(0.25, 1.0 - (time_pressure_weight * 0.65 * elapsed_ratio))
        adaptive_scale_pct = base_scale_pct * scale_multiplier

        z_value = diff_pct / adaptive_scale_pct
        model_up = clamp(_sigmoid(z_value), 0.03, 0.97)
        model_down = 1.0 - model_up
        expected_prob = model_up if direction == "buy_yes" else model_down

        repricing_buffer = expected_prob - selected_price
        min_repricing_buffer = clamp(to_float(cfg.get("min_repricing_buffer", 0.015), 0.015), 0.001, 0.20)
        if repricing_buffer < min_repricing_buffer:
            return None

        raw_edge_percent = max(0.0, repricing_buffer * 100.0)
        fee_percent = _taker_fee_pct(selected_price) * 100.0
        slippage_spread_multiplier = max(0.01, to_float(cfg.get("slippage_spread_multiplier", 0.22), 0.22))
        min_slippage_pct = max(0.0, to_float(cfg.get("min_slippage_pct", 0.08), 0.08))
        slippage_percent = max(min_slippage_pct, (spread_pct * 100.0) * slippage_spread_multiplier)

        net_edge_percent = raw_edge_percent - fee_percent - slippage_percent
        min_net_edge_percent = max(0.0, to_float(cfg.get("min_net_edge_percent", 0.35), 0.35))
        if net_edge_percent < min_net_edge_percent:
            return None

        signal_strength = clamp(abs(diff_pct) / max(adaptive_min_oracle_move_pct * 3.0, 0.01), 0.0, 1.0)
        freshness = clamp(1.0 - (float(oracle_age_seconds) / max_oracle_age_seconds), 0.0, 1.0)
        spread_quality = clamp(1.0 - (spread_pct / max_spread_pct), 0.0, 1.0)
        confidence = clamp(
            0.42 + (signal_strength * 0.24) + (tail_alignment * 0.16) + (freshness * 0.10) + (spread_quality * 0.08),
            0.35,
            0.96,
        )

        min_confidence = to_confidence(cfg.get("min_confidence", 0.50), 0.50)
        if confidence < min_confidence:
            return None

        risk_score = clamp(
            0.68
            - (signal_strength * 0.25)
            - (tail_alignment * 0.12)
            + ((1.0 - spread_quality) * 0.12)
            + ((1.0 - freshness) * 0.10),
            0.12,
            0.88,
        )

        score = (
            (net_edge_percent * 2.8)
            + (confidence * 40.0)
            + (tail_alignment * 10.0)
            + (spread_quality * 6.0)
            + (freshness * 4.0)
        )

        target_exit_price = clamp(selected_price + (net_edge_percent / 100.0), selected_price + 0.0001, 0.999)

        return {
            "asset": asset,
            "timeframe": timeframe,
            "direction": direction,
            "outcome": outcome,
            "selected_price": selected_price,
            "seconds_left": float(seconds_left),
            "oracle_price": float(oracle_price),
            "price_to_beat": float(price_to_beat),
            "oracle_diff_pct": float(diff_pct),
            "adaptive_min_oracle_move_pct": float(adaptive_min_oracle_move_pct),
            "raw_edge_percent": float(raw_edge_percent),
            "net_edge_percent": float(net_edge_percent),
            "fee_percent": float(fee_percent),
            "slippage_percent": float(slippage_percent),
            "confidence": float(confidence),
            "risk_score": float(risk_score),
            "tail_alignment": float(tail_alignment),
            "signal_strength": float(signal_strength),
            "freshness": float(freshness),
            "spread_quality": float(spread_quality),
            "spread_pct": float(spread_pct),
            "liquidity_usd": float(liquidity_usd),
            "expected_prob": float(expected_prob),
            "model_up": float(model_up),
            "model_down": float(model_down),
            "target_exit_price": float(target_exit_price),
            "score": float(score),
            "model_side_prob_key": model_side_prob_key,
            "oracle_age_seconds": float(oracle_age_seconds),
        }

    def _build_opportunity(self, row: dict[str, Any], signal: dict[str, Any]) -> Opportunity | None:
        typed_market = self._row_market(row)
        if typed_market is None:
            return None

        strategy_cfg = dict(getattr(self, "config", None) or {})
        direction = str(signal["direction"])
        outcome = str(signal["outcome"])
        selected_price = float(signal["selected_price"])
        token_ids = list(typed_market.clob_token_ids or [])
        token_idx = 0 if direction == "buy_yes" else 1
        token_id = token_ids[token_idx] if len(token_ids) > token_idx else None

        sniper_context = {
            "strategy_origin": "crypto_worker",
            "asset": signal["asset"],
            "timeframe": signal["timeframe"],
            "direction": direction,
            "seconds_left": signal["seconds_left"],
            "oracle_price": signal["oracle_price"],
            "price_to_beat": signal["price_to_beat"],
            "oracle_diff_pct": signal["oracle_diff_pct"],
            "raw_edge_percent": signal["raw_edge_percent"],
            "net_edge_percent": signal["net_edge_percent"],
            "fee_percent": signal["fee_percent"],
            "slippage_percent": signal["slippage_percent"],
            "confidence": signal["confidence"],
            "risk_score": signal["risk_score"],
            "tail_alignment": signal["tail_alignment"],
            "signal_strength": signal["signal_strength"],
            "freshness": signal["freshness"],
            "spread_quality": signal["spread_quality"],
            "spread_pct": signal["spread_pct"],
            "liquidity_usd": signal["liquidity_usd"],
            "expected_prob": signal["expected_prob"],
            "model_up": signal["model_up"],
            "model_down": signal["model_down"],
            "target_exit_price": signal["target_exit_price"],
            "oracle_age_seconds": signal["oracle_age_seconds"],
            "market_data_age_ms": safe_float(row.get("market_data_age_ms"), None),
            "fetched_at": row.get("fetched_at"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "regime": row.get("regime"),
            "regime_params": row.get("regime_params"),
        }
        context_flags = dict(sniper_context)

        seconds_left = float(signal["seconds_left"])
        timeframe = str(signal["timeframe"] or "5min").lower()
        timeframe_suffix_map = {
            "5min": "5m",
            "15min": "15m",
            "1hr": "1h",
            "4hr": "4h",
        }
        timeframe_suffix = timeframe_suffix_map.get(timeframe, timeframe)

        # Phase 2: Heuristic Intent Setting (for Orchestrator guidance)
        min_maker_s = to_float(strategy_cfg.get(f"min_entry_seconds_{timeframe_suffix}", 12.0), 12.0)
        max_maker_s = to_float(strategy_cfg.get(f"max_entry_seconds_{timeframe_suffix}", 110.0), 110.0)
        if min_maker_s > max_maker_s:
            min_maker_s, max_maker_s = max_maker_s, min_maker_s

        rescue_s_end = to_float(strategy_cfg.get("entry_window_end_seconds", 12.0), 12.0)

        if min_maker_s <= seconds_left <= max_maker_s:
            context_flags["execution_intent"] = "maker_preferred"
        elif seconds_left <= rescue_s_end:
            context_flags["execution_intent"] = "taker_rescue"
        else:
            context_flags["execution_intent"] = "outside_window"

        positions = [
            {
                "action": "BUY",
                "outcome": outcome,
                "price": selected_price,
                "token_id": token_id,
                "price_policy": "taker_limit",  # Keep default; intent is stored in context
                "time_in_force": "IOC",
                "_sniper": context_flags,
            }
        ]

        opp = self.create_opportunity(
            title=f"Micro Sniper: {outcome} @ {selected_price:.3f} ({context_flags['execution_intent']})",
            description=(
                f"{seconds_left:.0f}s left, oracle diff {signal['oracle_diff_pct']:+.3f}%, "
                f"net edge {signal['net_edge_percent']:.2f}%"
            ),
            total_cost=selected_price,
            expected_payout=signal["target_exit_price"],
            markets=[typed_market],
            positions=positions,
            is_guaranteed=False,
            skip_fee_model=True,
            custom_roi_percent=signal["net_edge_percent"],
            custom_risk_score=signal["risk_score"],
            confidence=signal["confidence"],
            min_liquidity_hard=max(100.0, to_float(strategy_cfg.get("min_liquidity_usd", 2500.0), 2500.0)),
            min_position_size=max(1.0, to_float(strategy_cfg.get("min_order_size_usd", 2.0), 2.0)),
        )
        if opp is None:
            return None

        opp.risk_factors = [
            f"Tail-end entry ({seconds_left:.0f}s left)",
            f"Oracle diff={signal['oracle_diff_pct']:+.3f}% vs price-to-beat",
            f"Net edge after fees/slippage={signal['net_edge_percent']:.2f}%",
        ]
        opp.strategy_context = {
            "source_key": "crypto",
            "strategy_slug": self.strategy_type,
            **sniper_context,
        }
        return opp

    def _detect_from_rows(self, rows: list[dict[str, Any]]) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        candidates: list[tuple[float, Opportunity]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            signal = self._score_market(row, cfg)
            if signal is None:
                continue
            opp = self._build_opportunity(row, signal)
            if opp is None:
                continue
            candidates.append((float(signal["score"]), opp))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        max_markets_per_event = max(1, int(to_float(cfg.get("max_markets_per_event", 24), 24.0)))

        out: list[Opportunity] = []
        seen_markets: set[str] = set()
        for _, opp in candidates:
            market_id = str((opp.markets or [{}])[0].get("id") or "")
            if market_id in seen_markets:
                continue
            seen_markets.add(market_id)
            out.append(opp)
            if len(out) >= max_markets_per_event:
                break
        return out

    def detect(self, events: list, markets: list, prices: dict[str, dict]) -> list[Opportunity]:
        return []

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != EventType.CRYPTO_UPDATE:
            return []
        rows = event.payload.get("markets") if isinstance(event.payload, dict) else None
        if not isinstance(rows, list) or not rows:
            return []
        return self._detect_from_rows(rows)

    def _extract_live_selected_price(self, live_market: dict[str, Any], payload: dict[str, Any]) -> float | None:
        for candidate in (
            live_market.get("live_selected_price"),
            live_market.get("selected_price"),
            payload.get("live_selected_price"),
            payload.get("selected_price"),
            payload.get("entry_price"),
        ):
            parsed = safe_float(candidate, None)
            if parsed is not None and 0.0 < parsed < 1.0:
                return float(parsed)
        return None

    def _extract_seconds_left(self, live_market: dict[str, Any], payload: dict[str, Any]) -> float | None:
        direct = safe_float(live_market.get("seconds_left"), safe_float(payload.get("seconds_left"), None))
        if direct is not None and direct >= 0.0:
            return float(direct)
        end_time = (
            live_market.get("market_end_time")
            or live_market.get("end_time")
            or payload.get("end_time")
            or payload.get("strategy_context", {}).get("end_time")
        )
        parsed_end = _parse_datetime_utc(end_time)
        if parsed_end is None:
            return None
        return max(0.0, (parsed_end - datetime.now(timezone.utc)).total_seconds())

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})
        cfg.update(params)

        payload = signal_payload(signal)
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market") if isinstance(payload.get("live_market"), dict) else {}

        source = str(getattr(signal, "source", "") or "").strip().lower()
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)

        min_edge = max(0.0, to_float(cfg.get("min_net_edge_percent", 0.35), 0.35))
        min_confidence = to_confidence(cfg.get("min_confidence", 0.50), 0.50)

        seconds_left = self._extract_seconds_left(live_market, payload)
        timeframe = payload.get("timeframe", "5min").lower()
        timeframe_key = timeframe if timeframe.endswith("min") else f"{timeframe}min"

        window_start = max(1.0, to_float(cfg.get(f"entry_window_start_seconds_{timeframe_key}", 110.0), 110.0))
        window_end = max(1.0, to_float(cfg.get(f"entry_window_end_seconds_{timeframe_key}", 12.0), 12.0))
        if window_end > window_start:
            window_end = window_start
        window_ok = seconds_left is not None and window_end <= seconds_left <= window_start

        selected_price = self._extract_live_selected_price(live_market, payload)
        if selected_price is None:
            selected_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
            if selected_price <= 0.0:
                selected_price = None

        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.90), 0.90), 0.05, 0.99)
        entry_price_ok = selected_price is not None and selected_price <= max_entry_price

        oracle_age_seconds = safe_float(
            live_market.get("oracle_age_seconds"),
            safe_float(
                payload.get("oracle_age_seconds"),
                safe_float(payload.get("strategy_context", {}).get("oracle_age_seconds"), None),
            ),
        )
        max_oracle_age_seconds = max(0.1, to_float(cfg.get("max_oracle_age_seconds", 8.0), 8.0))
        oracle_fresh_ok = oracle_age_seconds is not None and oracle_age_seconds <= max_oracle_age_seconds

        spread_pct = safe_float(
            live_market.get("spread"),
            safe_float(payload.get("spread"), safe_float(payload.get("strategy_context", {}).get("spread_pct"), None)),
        )
        if spread_pct is None:
            spread_pct = 0.0
        max_spread_pct = clamp(to_float(cfg.get("max_spread_pct", 0.03), 0.03), 0.001, 0.2)
        spread_ok = spread_pct <= max_spread_pct

        oracle_diff_pct = safe_float(
            payload.get("oracle_diff_pct"),
            safe_float(payload.get("strategy_context", {}).get("oracle_diff_pct"), None),
        )
        min_oracle_move_pct = max(0.01, to_float(cfg.get("min_oracle_move_pct", 0.18), 0.18))
        adaptive_min_oracle_move_pct = self._adaptive_min_oracle_move_pct(
            min_oracle_move_pct,
            seconds_left=seconds_left,
            window_start_seconds=window_start,
        )
        oracle_move_ok = oracle_diff_pct is not None and abs(oracle_diff_pct) >= adaptive_min_oracle_move_pct

        min_liquidity_usd = max(0.0, to_float(cfg.get("min_liquidity_usd", 2500.0), 2500.0))
        liquidity = max(
            0.0,
            to_float(
                live_market.get("liquidity"),
                to_float(
                    payload.get("liquidity"),
                    to_float(
                        payload.get("strategy_context", {}).get("liquidity_usd"),
                        to_float(getattr(signal, "liquidity", 0.0), 0.0),
                    ),
                ),
            ),
        )
        liquidity_ok = liquidity >= min_liquidity_usd

        checks = [
            DecisionCheck("source", "Crypto source", source == "crypto", detail="Requires source=crypto."),
            DecisionCheck("edge", "Net edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}%"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_confidence,
                score=confidence,
                detail=f"min={min_confidence:.2f}",
            ),
            DecisionCheck(
                "entry_window",
                "Tail entry window",
                window_ok,
                score=seconds_left,
                detail=f"[{window_end:.0f}, {window_start:.0f}]s",
            ),
            DecisionCheck(
                "oracle_freshness",
                "Oracle freshness",
                oracle_fresh_ok,
                score=oracle_age_seconds,
                detail=f"max={max_oracle_age_seconds:.1f}s",
            ),
            DecisionCheck(
                "oracle_move",
                "Oracle move magnitude",
                oracle_move_ok,
                score=oracle_diff_pct,
                detail=f"min_abs={adaptive_min_oracle_move_pct:.3f}%",
            ),
            DecisionCheck(
                "spread",
                "Spread cap",
                spread_ok,
                score=spread_pct,
                detail=f"max={max_spread_pct:.3f}",
            ),
            DecisionCheck(
                "entry_price",
                "Entry price cap",
                entry_price_ok,
                score=selected_price,
                detail=f"max={max_entry_price:.3f}",
            ),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity_ok,
                score=liquidity,
                detail=f"min=${min_liquidity_usd:.0f}",
            ),
        ]

        target_entry_seconds = clamp(
            to_float(cfg.get("target_entry_seconds", 60.0), 60.0),
            window_end,
            window_start,
        )
        tolerance_seconds = max(1.0, to_float(cfg.get("window_tolerance_seconds", 45.0), 45.0))
        tail_alignment = 0.0
        if seconds_left is not None:
            tail_alignment = clamp(1.0 - (abs(seconds_left - target_entry_seconds) / tolerance_seconds), 0.0, 1.0)

        spread_quality = clamp(1.0 - (spread_pct / max_spread_pct), 0.0, 1.0)
        score = (edge * 2.4) + (confidence * 38.0) + (tail_alignment * 10.0) + (spread_quality * 6.0)

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Standalone sniper filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "seconds_left": seconds_left,
                    "oracle_age_seconds": oracle_age_seconds,
                    "oracle_diff_pct": oracle_diff_pct,
                    "adaptive_min_oracle_move_pct": adaptive_min_oracle_move_pct,
                    "selected_price": selected_price,
                    "spread_pct": spread_pct,
                    "liquidity": liquidity,
                },
            )

        base_size = max(1.0, to_float(cfg.get("base_size_usd", 24.0), 24.0))
        max_size = max(base_size, to_float(cfg.get("max_size_usd", 180.0), 180.0))
        min_order_size_usd = max(0.01, to_float(cfg.get("min_order_size_usd", 2.0), 2.0))
        sizing_policy = str(cfg.get("sizing_policy", "kelly") or "kelly")
        kelly_fractional_scale = to_float(cfg.get("kelly_fractional_scale", 0.35), 0.35)

        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        probability = selected_probability(signal, payload, direction)
        sizing = compute_position_size(
            base_size_usd=base_size,
            max_size_usd=max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=selected_price,
            kelly_fractional_scale=kelly_fractional_scale,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=to_float(cfg.get("liquidity_cap_fraction", 0.07), 0.07),
            min_size_usd=min_order_size_usd,
        )

        size_usd = max(min_order_size_usd, float(sizing.get("size_usd") or min_order_size_usd))

        return StrategyDecision(
            decision="selected",
            reason="Standalone sniper signal selected",
            score=score,
            size_usd=size_usd,
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "seconds_left": seconds_left,
                "oracle_age_seconds": oracle_age_seconds,
                "oracle_diff_pct": oracle_diff_pct,
                "adaptive_min_oracle_move_pct": adaptive_min_oracle_move_pct,
                "selected_price": selected_price,
                "spread_pct": spread_pct,
                "liquidity": liquidity,
                "tail_alignment": tail_alignment,
                "spread_quality": spread_quality,
                "sizing": sizing,
                "min_order_size_usd": min_order_size_usd,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        config = getattr(position, "config", None) or {}
        strategy_config = dict(getattr(self, "config", None) or {})

        config.setdefault("take_profit_pct", float(strategy_config.get("take_profit_pct", 7.0)))
        config.setdefault("stop_loss_pct", float(strategy_config.get("stop_loss_pct", 3.8)))
        config.setdefault("max_hold_minutes", float(strategy_config.get("max_hold_minutes", 6.0)))

        position.config = config
        return self.default_exit_check(position, market_state)

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s blocked: %s market=%s", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s size capped $%.0f -> $%.0f (%s)", self.name, original_size, capped_size, reason)
